"""
src/backtest/pipeline.py — V4.0 shared dataset-construction library.

Single source of truth for HOW the V3/V4 backtest dataset is built, so the two
decoupled entry-points agree byte-for-byte:

    train_models.py   (heavy lifter)  — ingest → features → labels → align →
                                        split → iron-fist select → HMM → train
                                        4-seed ensemble → checkpoint.
    run_backtest.py   (fast evaluator) — load checkpoint → re-materialize the
                                        SAME dataset → threshold sweep / DSR /
                                        PBO → persist the live-bot payload.

Why a shared module (and not two self-contained scripts)?  `run_backtest.py`
must reconstruct the exact feature matrix the ensemble was trained on.  If the
feature engineering were duplicated, any drift between the two copies would
silently mis-align columns and feed the model garbage.  Centralising the
pipeline here makes train/serve parity a *structural* guarantee rather than a
discipline.  The two scripts never import each other — they are decoupled and
communicate only through `models/saved/v3_training_checkpoint.joblib`.

Nothing in this module trains a model or runs a backtest; it only turns raw
sources into a clean, split, leak-free `(X, y, w, dates, t1, tickers)` dataset.
"""
from __future__ import annotations

import io as _io
import logging
import sys as _sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ── Phase 1 / 1.5 / 9 features (cross-sectional alpha factors + candidates) ───
from src.data.tensor_builder import (
    FracDiffConfig,
    apply_frac_diff,
    add_cross_sectional_features,
    add_overextension_features,
    add_alpha_factors,
    add_advanced_statistical_features,
)
# ── Phase 2 labels ────────────────────────────────────────────────────────────
from src.labels.triple_barrier import TripleBarrierConfig, triple_barrier_pipeline
# ── Phase 6.5 corporate actions ───────────────────────────────────────────────
from src.execution.vn_cost_model import InventoryTracker
# ── 8-regime structural context (rule-based Polars classifier) ─────────────────
from src.features.market_regime import build_regime_features
from src.utils.schema_hash import compute_feature_schema_hash

LOGGER = logging.getLogger("quant.pipeline")
TRADING_DAYS = 252

# Authoritative schema for the feature pool in `build_features`.  Column order
# is load-bearing (continuous pool first, then categoricals).  Dtype strings are
# Polars canonical names (Float32, Int8, etc.).
FEATURE_SCHEMA: list[tuple[str, str]] = [
    ("close_fd_xsz",           "Float32"),
    ("volume_fd_xsz",          "Float32"),
    ("mom20_xsz",              "Float32"),
    ("overext_5_xsz",          "Float32"),
    ("overext_20_xsz",         "Float32"),
    ("rs_10_xsz",              "Float32"),
    ("rs_20_xsz",              "Float32"),
    ("smart_money_20_xsz",     "Float32"),
    ("vol_squeeze_xsz",        "Float32"),
    ("amihud_liquidity_xsz",   "Float32"),
    ("realized_skewness_20d_xsz", "Float32"),
    ("vol_of_vol_20d_xsz",     "Float32"),
    ("hl_range_ratio_xsz",     "Float32"),
    ("gap_risk_xsz",           "Float32"),
    ("market_regime",          "Int8"),
]

# FEATURE_RECIPE_VERSION is computed after RunConfig is defined (see below).

# Features the GBMs treat as CATEGORICAL (native split), NOT continuous.  They
# bypass the iron-fist corr/MI selection (always survive) and are declared
# categorical to LightGBM/CatBoost at fit time (see TabularEnsemble).
CATEGORICAL_FEATURES: list[str] = ["market_regime"]


# ─────────────────────────────────────────────────────────────────────────────
# Console / logging setup (shared by both entry-points)
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    """UTF-8-safe stdout/stderr + a single INFO logging config.

    Windows cp1252 otherwise crashes on the teardown's decorative chars (and
    emits ``\\uXXXX`` escape literals in redirected file logs).  We wrap a
    stream ONLY when its codec is not already utf-* so Linux/macOS terminals
    are untouched.  ``errors="replace"`` is the final safety net for any stray
    non-encodable byte.
    """
    for _name in ("stdout", "stderr"):
        _s = getattr(_sys, _name)
        _enc = (getattr(_s, "encoding", "") or "").lower()
        if "utf" not in _enc and hasattr(_s, "buffer"):
            setattr(_sys, _name,
                    _io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


@contextmanager
def phase(name: str):
    LOGGER.info("▶ %s …", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        LOGGER.info("✔ %s  (%.1fs)", name, time.perf_counter() - t0)


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration — V4.0 (T+20 swing, VN50 gate, PT=3.0σ / SL=2.0σ, 4 seeds)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    # Data sources (tried in order for OHLCV)
    bitemporal_duckdb: Path = Path("data/bitemporal_store.duckdb")
    core_duckdb: Path = Path("data/quant_v6_core.duckdb")
    parquet_glob: str = "data/ohlcv_*.parquet"

    # Universe / span limits (for tractability on local hardware)
    ticker_limit: int | None = None        # None = full universe
    start_date: str | None = None          # e.g. "2018-01-01"
    min_history: int = 120                 # drop tickers with < this many bars

    # Split
    train_frac: float = 0.70

    # Feature / label hyper-params  (V4.0)
    frac_diff_d: float = 0.4
    tb_horizon: int = 20     # V4.0 SWING — T+20 horizon to amortise VN T+2.5 costs over a longer hold
    tb_pt: float = 3.0       # V4.0 profit-target barrier = 3.0σ
    tb_sl: float = 2.0       # V4.0 stop-loss barrier     = 2.0σ  (1.5:1 reward:risk)

    # Seed pool — PBO needs ≥2 configs; V4.0 trains 4 ensembles.
    n_configs: int = 4

    # Walk-forward / portfolio (EVAL knobs — run_backtest.py REFRESHES these from
    # the current defaults each run, so editing them here takes effect WITHOUT
    # retraining).  max_positions=5 × max_weight=0.20 = 100% NAV worst-case gross
    # (long-only, unlevered) — MIRRORS the live bot advisory (src/bot/sizing.py:
    # DEFAULT_TOP_N=5, DEFAULT_NAV_CAP=0.20) so the backtest's concentration risk
    # profile matches what the bot actually advises (no train/serve skew).
    initial_capital: float = 10_000_000_000.0
    max_positions: int = 5
    rebalance_frequency: int = 5
    signal_threshold: float = 0.35     # engine gate (the sweep drives this per-threshold)
    max_weight: float = 0.20
    target_vol: float = 0.15
    kelly_fraction: float = 0.5
    risk_aversion: float = 2.0
    liquid_top_n: int = 50         # VN50 gate — top-N by trailing-20d ADV

    # CSCV
    cscv_S: int = 12

    seed: int = 42

    # Macro Risk Oracle (HMM soft regime scaling)
    use_macro_hmm: bool = True
    hmm_n_states: int = 2

    def __post_init__(self) -> None:
        # Coerce path-like strings to Path so callers can pass either.
        self.bitemporal_duckdb = Path(self.bitemporal_duckdb)
        self.core_duckdb = Path(self.core_duckdb)


# Structural feature-recipe version.  Computed automatically from FEATURE_SCHEMA
# and the frac_diff_d hyperparameter — any column add/remove/reorder, dtype
# change, or frac-diff tuning produces a new hash, forcing a retrain before
# the serve path accepts the artifact (main._load_v3_bot tripwire).
FEATURE_RECIPE_VERSION: str = compute_feature_schema_hash(
    FEATURE_SCHEMA, RunConfig().frac_diff_d
)


# ─────────────────────────────────────────────────────────────────────────────
# Aligned dataset containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignedData:
    X: np.ndarray            # (M, F)  ← V3/V4: 2-D tabular, no sequence dim
    y: np.ndarray            # (M,) bins {0,1,2}
    w: np.ndarray            # (M,) sample weights
    dates: np.ndarray        # (M,) decision-bar dates (t0 — start_times for PurgedKFold)
    t1: np.ndarray           # (M,) vertical-barrier date (end_times for PurgedKFold)
    tickers: np.ndarray      # (M,)


@dataclass
class Dataset:
    """Everything both entry-points need from a single materialization pass.

    `aligned.X` carries the FULL feature pool (pre-selection); column order ==
    `all_features`.  train_models prunes it via `select_features`; run_backtest
    prunes it via `subset_features` using the checkpoint's saved feature list.
    """
    panel: pl.DataFrame
    aligned: AlignedData
    all_features: list[str]        # full pool from build_features (pre-selection)
    original_features: list[str]   # the always-survive baseline alphas
    candidate_features: list[str]  # iron-fist candidate pool
    categorical_features: list[str]  # forced-survive categorical(s) (e.g. market_regime)
    cutoff: date                   # chronological train/OOS split boundary
    train_mask: np.ndarray         # aligned.dates < cutoff


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION  (graceful fallback chain)
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(cfg: RunConfig) -> pl.DataFrame:
    """RAW OHLCV — **PARQUET-FIRST** so train/backtest use the SAME fresh vintage
    the live bot serves on.

    The live serve path (`Alpha360Generator.load_live_ohlcv_window`) reads
    `data/ohlcv_*.parquet`, which `crawl_hose` keeps current.  The legacy core
    DuckDB `stock_ohlcv` table has been RETIRED (it was never updated by the
    crawler and drifted ~18 days stale; reading it silently trained the model on
    stale data — see the DB audit).  Source priority is therefore:

      1. `data/ohlcv_*.parquet`        ← AUTHORITATIVE (fresh; train/serve parity)
      2. bitemporal store (if present) ← survivorship-free fallback, only if no parquet

    If neither source yields rows we RAISE rather than silently fall back to a
    stale table.  `_post_ohlcv` logs the resulting date range, so the loaded
    vintage is always visible in the run log.
    """
    import duckdb

    # (1) PRIMARY — fresh parquet shards (the live bot's exact source).
    files = sorted(Path().glob(cfg.parquet_glob))
    if files:
        df = (pl.scan_parquet([str(f) for f in files])
              .select(["ticker", "date", "open", "high", "low", "close", "volume"])
              .collect())
        if df.height > 0:
            LOGGER.info("OHLCV ← %d parquet shards (%d rows)  [PRIMARY — fresh]",
                        len(files), df.height)
            return _post_ohlcv(df, cfg)
        LOGGER.warning("Parquet glob %s matched %d files but yielded 0 rows; falling back.",
                       cfg.parquet_glob, len(files))

    # (2) FALLBACK — bitemporal store (survivorship-free), only when no parquet.
    if cfg.bitemporal_duckdb.exists():
        try:
            con = duckdb.connect(str(cfg.bitemporal_duckdb))
            tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()}
            if "ohlcv_bitemporal" in tables:
                df = con.execute("""
                    WITH latest AS (
                        SELECT ticker, event_date, MAX(knowledge_date) AS kd
                        FROM ohlcv_bitemporal GROUP BY ticker, event_date)
                    SELECT b.ticker, b.event_date AS date,
                           b.open_raw AS open, b.high_raw AS high,
                           b.low_raw AS low, b.close_raw AS close, b.volume_raw AS volume
                    FROM ohlcv_bitemporal b
                    JOIN latest l ON b.ticker=l.ticker AND b.event_date=l.event_date
                                 AND b.knowledge_date=l.kd
                    ORDER BY b.ticker, b.event_date
                """).pl()
                con.close()
                if df.height > 0:
                    LOGGER.warning("OHLCV ← bitemporal_store (%d rows)  [FALLBACK — no parquet found]",
                                   df.height)
                    return _post_ohlcv(df, cfg)
            con.close()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("bitemporal store read failed (%s); falling back", exc)

    # NOTE: the legacy core-DuckDB `stock_ohlcv` last-resort branch was REMOVED.
    # That table is retired (stale, crawler-orphaned); falling back to it silently
    # reintroduced the train/serve freshness skew this parquet-first design fixes.
    raise FileNotFoundError(
        "No OHLCV source found (parquet glob / bitemporal store). The legacy "
        "stock_ohlcv table has been retired and is no longer consulted."
    )


def _post_ohlcv(df: pl.DataFrame, cfg: RunConfig) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
    ]).sort(["ticker", "date"])

    # ── DATA-QUALITY FILTER (CRITICAL) ───────────────────────────────────────
    # Real VN feeds carry dirty rows: suspended penny stocks and data artifacts
    # with zero / null / negative OHLC. A zero close becomes p0 == 0 in the
    # triple-barrier labeller → divide-by-zero → inf/NaN poisons the vol, return
    # and sample-weight accumulators and hangs the run. Excise them HERE, at the
    # single ingestion choke-point, before any feature or label math executes.
    price_cols = ["open", "high", "low", "close"]
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in price_cols])
    valid = pl.lit(True)
    for c in price_cols:
        valid = valid & pl.col(c).is_finite() & (pl.col(c) > 0)
    valid = valid & pl.col("volume").is_not_null() & (pl.col("volume") >= 0)
    n_before = df.height
    df = df.filter(valid)
    dropped = n_before - df.height
    if dropped:
        LOGGER.warning("Dropped %d dirty OHLCV rows (non-positive / non-finite price).",
                       dropped)

    if cfg.start_date:
        df = df.filter(pl.col("date") >= pl.lit(cfg.start_date).str.to_date())
    # Drop short-history tickers.
    counts = df.group_by("ticker").len()
    # Polars 1.x: `is_in` against a SERIES of the same dtype is deprecated as
    # ambiguous (column-vs-collection conflated). Convert to a Python list so
    # the call is unambiguous element-wise membership.
    keep = counts.filter(pl.col("len") >= cfg.min_history)["ticker"].to_list()
    df = df.filter(pl.col("ticker").is_in(keep))
    if cfg.ticker_limit:
        chosen = sorted(df["ticker"].unique().to_list())[:cfg.ticker_limit]
        df = df.filter(pl.col("ticker").is_in(chosen))
    # Add an exchange column if missing (microstructure rules default to HOSE).
    if "exchange" not in df.columns:
        df = df.with_columns(pl.lit("HOSE").alias("exchange"))
    LOGGER.info("OHLCV prepared | tickers=%d  rows=%d  range=%s..%s",
                df["ticker"].n_unique(), df.height,
                df["date"].min(), df["date"].max())
    return df


def load_corporate_actions(cfg: RunConfig) -> list:
    """Corporate actions from the bitemporal store (empty list if absent)."""
    import duckdb

    if not cfg.bitemporal_duckdb.exists():
        return []
    try:
        con = duckdb.connect(str(cfg.bitemporal_duckdb))
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        if "corporate_actions" not in tables:
            con.close()
            return []
        ca = con.execute("""
            WITH latest AS (
                SELECT ticker, event_date, action_type, MAX(knowledge_date) AS kd
                FROM corporate_actions GROUP BY ticker, event_date, action_type)
            SELECT c.ticker, c.event_date, c.action_type, c.factor, c.cash_amount
            FROM corporate_actions c
            JOIN latest l ON c.ticker=l.ticker AND c.event_date=l.event_date
                         AND c.action_type=l.action_type AND c.knowledge_date=l.kd
        """).pl()
        con.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("corporate_actions read failed (%s); proceeding with none", exc)
        return []
    if ca.height == 0:
        return []
    events = InventoryTracker.parse_corporate_actions(ca)
    LOGGER.info("Corporate actions loaded: %d events", len(events))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE PIPELINE  (Phase 1 + 1.5 + 9) — all leak-free (≤ t only)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    ohlcv: pl.DataFrame, cfg: RunConfig,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Returns (panel, all_features, candidate_features).

    `all_features` is the FULL pre-selection pool (originals + candidates) in a
    deterministic, hardcoded order.  The iron-fist selection (train) and the
    `subset_features` re-projection (eval) both rely on that order being stable.
    """
    df = ohlcv

    # Phase 1: FracDiff of close & volume (stationary, memory-preserving).
    df = apply_frac_diff(df, ["close", "volume"],
                         cfg=FracDiffConfig(d=cfg.frac_diff_d))

    # A momentum feature for the cross-section.
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(20).over("ticker") - 1.0).alias("mom20")
    )

    # Phase 1: cross-sectional Gaussian-rank Z on the per-name features.
    xs_inputs = ["close_fd", "volume_fd", "mom20"]
    df = add_cross_sectional_features(df, xs_inputs, suffix="_xsz")

    # Phase 1.5: anti-FOMO over-extension (raw + cross-sectional).
    df = add_overextension_features(df, ma_windows=(5, 20), cross_sectional=True)

    # GROUP A alpha factors (RS / smart-money / vol-squeeze) — cross-sectionally Z.
    df = add_alpha_factors(df, rs_windows=(10, 20), money_flow_window=20,
                           vol_short=5, vol_long=20, cross_sectional=True)

    # GROUP C ADVANCED STATISTICAL FEATURES (V3.1 candidate pool — selected by
    # the iron-fist filter+MI loop against the train split, capped at +3
    # survivors so the final pool stays ≤ 12 features and PBO stays clean).
    df = add_advanced_statistical_features(df, cross_sectional=True)

    # V3/V4: GROUP B seasonality + macro are BOTH OFF for the GBM stack.
    #   - day_of_year_{sin,cos} were GLOBAL → calendar memorisation overfit.
    #   - Macro/regime is handled entirely by the HMM Oracle, which derives its
    #     P(Bull) from a PRICE-based market proxy (build_market_proxy_returns) —
    #     no macro DataFrame is ingested anywhere in the V4 pipeline.  The GBM
    #     stack sees ONLY cross-sectional alphas.

    alpha_xsz = ["rs_10_xsz", "rs_20_xsz", "smart_money_20_xsz", "vol_squeeze_xsz"]
    original_features = [
        "close_fd_xsz", "volume_fd_xsz", "mom20_xsz",
        "overext_5_xsz", "overext_20_xsz",
        *alpha_xsz,                                  # 9 baseline alpha factors
    ]
    candidate_features = [
        "amihud_liquidity_xsz", "realized_skewness_20d_xsz",
        "vol_of_vol_20d_xsz", "hl_range_ratio_xsz", "gap_risk_xsz",
    ]

    # 8-REGIME structural context (rule-based, from RAW OHLCV).  Appended as a
    # CATEGORICAL feature the GBMs split on natively: it bypasses the iron-fist
    # corr/MI selection (always survives) and is declared categorical to
    # LightGBM/CatBoost in TabularEnsemble.  `market_regime` is non-null by
    # construction (warm-up rows default to CHOPPY), so it is deliberately NOT in
    # the continuous-feature dropna below.
    df = build_regime_features(df.lazy()).collect()

    # Continuous pool first, THEN the categorical(s) — column order is load-bearing.
    all_features = original_features + candidate_features + CATEGORICAL_FEATURES

    assert [name for name, _ in FEATURE_SCHEMA] == all_features, (
        f"FEATURE_SCHEMA names do not match all_features order. "
        f"Schema: {[n for n,_ in FEATURE_SCHEMA]} | built: {all_features}. "
        f"Update FEATURE_SCHEMA to match the hardcoded pool."
    )

    # Drop rows with any NaN in the CONTINUOUS features (FracDiff / MA / factor
    # warm-up).  market_regime is excluded — it is never null.
    df = df.drop_nulls(subset=original_features + candidate_features)
    LOGGER.info(
        "Features built (V3.1 pool + regime) | total=%d (originals=%d + candidates=%d "
        "+ categorical=%d)  rows=%d",
        len(all_features), len(original_features), len(candidate_features),
        len(CATEGORICAL_FEATURES), df.height)
    return df, all_features, candidate_features


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABELING  (Phase 2 — triple-barrier T+H, PT/SL σ-barriers, AFML §4.10 w)
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(panel: pl.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    labels = triple_barrier_pipeline(
        panel,
        cfg=TripleBarrierConfig(pt_mult=cfg.tb_pt, sl_mult=cfg.tb_sl,
                                vol_span=20, horizon=cfg.tb_horizon, label_scheme="012"),
    ).to_pandas()
    labels["t0_date"] = pd.to_datetime(labels["t0"]).dt.date
    LOGGER.info("Labels | events=%d  bin_dist=%s  mean_w=%.3f",
                len(labels), labels["bin"].value_counts().to_dict(),
                float(labels["w"].mean()))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. ALIGN features ↔ labels
# ─────────────────────────────────────────────────────────────────────────────

def align(panel: pl.DataFrame, labels: pd.DataFrame,
          tabular_features: list[str]) -> AlignedData:
    """V3/V4 tabular alignment — decision-bar features inner-joined with labels.

    No sequence windowing, no 3-D tensor build — just a (ticker, date) join.
    `tabular_features` defines the column order of the returned `X`.
    """
    pdf = (panel
           .select(["ticker", "date"] + tabular_features)
           .to_pandas()
           .dropna(subset=tabular_features))
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.date

    lab = labels[["ticker", "t0_date", "t1", "bin", "w"]].rename(columns={"t0_date": "date"})
    lab = lab.dropna(subset=["bin"])

    merged = pdf.merge(lab, on=["ticker", "date"], how="inner", sort=False)
    # CRITICAL: PurgedKFold uses POSITIONAL `np.array_split` to define fold
    # boundaries, then purges training rows whose [start_time, end_time] window
    # overlaps the test window.  If rows are sorted by TICKER first (the merge's
    # default ordering), each fold spans many tickers across the WHOLE date
    # range → the test window covers ~all dates → the purge eats the entire
    # train set → empty folds → `Mean of empty slice` crash downstream.
    # Strict (date, ticker) sort makes each positional fold a contiguous date
    # block so purge/embargo behave correctly.
    merged = merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    LOGGER.info("Aligned (V3/V4 tabular) | panel_rows=%d  label_rows=%d  matched=%d  features=%d",
                len(pdf), len(lab), len(merged), len(tabular_features))
    return AlignedData(
        X=merged[tabular_features].to_numpy(dtype=np.float32),
        y=merged["bin"].to_numpy().astype(np.int64),
        w=merged["w"].to_numpy().astype(np.float32),
        dates=merged["date"].to_numpy(),
        t1=pd.to_datetime(merged["t1"]).to_numpy(),         # PurgedKFold end_times
        tickers=merged["ticker"].to_numpy(),
    )


def chronological_split(aligned: AlignedData, train_frac: float) -> tuple[date, np.ndarray]:
    """Single chronological cut WITH a boundary purge (AFML §7 — leak-free).

    `cutoff` = the date at the `train_frac` quantile of unique decision-bar dates.

    A naive ``train_mask = dates < cutoff`` LEAKS look-ahead: a training sample
    with t0 just before the cutoff has a triple-barrier vertical barrier
    t1 ≈ t0 + horizon that lands INSIDE the OOS window, so its label was computed
    from post-cutoff prices the model must not have seen.  We therefore PURGE any
    training sample whose label window spills past the cutoff (``t1 >= cutoff``).
    No extra embargo is required because train is entirely *before* test — the
    only overlap is this forward spillover (an embargo guards the symmetric case
    of train rows that come *after* a test fold, which cannot occur here).
    """
    all_dates = np.array(sorted(set(aligned.dates.tolist())))
    cutoff = all_dates[int(len(all_dates) * train_frac)]
    cutoff_dt = np.datetime64(cutoff)
    before_cut = aligned.dates < cutoff
    no_spill = aligned.t1 < cutoff_dt          # PURGE: drop train labels peeking into OOS
    train_mask = before_cut & no_spill
    n_purged = int(before_cut.sum()) - int(train_mask.sum())
    LOGGER.info("Boundary purge | dropped %d train rows whose t1 ≥ cutoff (%s) — "
                "leak-free train/OOS split", n_purged, cutoff)
    return cutoff, train_mask


# ─────────────────────────────────────────────────────────────────────────────
# 5. IRON-FIST FEATURE SELECTION (V3.1) — Steps A/B/C, TRAIN-SPLIT ONLY
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    aligned: AlignedData,
    train_mask: np.ndarray,
    *,
    all_features: list[str],
    original_features: list[str],
    candidate_features: list[str],
    categorical_features: list[str] | tuple[str, ...] = (),
    corr_threshold: float = 0.65,
    top_k: int = 3,
    mi_subsample: int = 50_000,
    mi_seed: int = 42,
) -> tuple[AlignedData, list[str]]:
    """
    Run the V3.1 selection sequence ON THE TRAIN SPLIT, then prune aligned.X
    + the feature list in place.  Originals always survive; candidates pass
    through Steps A (collinearity) → B (mutual information) → C (top-K cap).

    The MI step is subsampled to `mi_subsample` rows (default 50 k) because
    `sklearn.feature_selection.mutual_info_classif` is O(n·log n·k) — on real
    half-million-row train splits a full pass takes minutes for a quantity
    that converges by a few tens of thousands of samples.

    Returns (mutated aligned, new feature list).  Column order of `aligned.X`
    after this call is `original_features + selected_candidates`.
    """
    from sklearn.feature_selection import mutual_info_classif

    LOGGER.info(
        "Feature selection | originals=%d  candidates=%d  "
        "corr_threshold=%.2f  top_k=%d",
        len(original_features), len(candidate_features), corr_threshold, top_k)

    name_to_col = {n: i for i, n in enumerate(all_features)}
    orig_idx = [name_to_col[n] for n in original_features]
    cand_idx = [name_to_col[n] for n in candidate_features]

    X_tr = aligned.X[train_mask]
    y_tr = aligned.y[train_mask]

    # ── Step A: Collinearity filter ────────────────────────────────────────
    # |Pearson r| between every candidate and every original, on standardized
    # columns.  Drop any candidate whose max correlation exceeds the threshold.
    X_orig = X_tr[:, orig_idx].astype(np.float64)
    X_cand = X_tr[:, cand_idx].astype(np.float64)
    Az = (X_orig - X_orig.mean(0)) / (X_orig.std(0, ddof=0) + 1e-12)
    Bz = (X_cand - X_cand.mean(0)) / (X_cand.std(0, ddof=0) + 1e-12)
    corr_cand_x_orig = (Bz.T @ Az) / max(1, X_orig.shape[0])          # (n_cand, n_orig)
    max_abs = np.nanmax(np.abs(corr_cand_x_orig), axis=1)

    surv_mask = max_abs <= corr_threshold
    surv_names = [candidate_features[i] for i in range(len(candidate_features)) if surv_mask[i]]
    surv_local_idx = [i for i in range(len(candidate_features)) if surv_mask[i]]
    surv_global_idx = [cand_idx[i] for i in surv_local_idx]

    LOGGER.info("  Step A | collinearity max |r| vs originals:")
    for i, name in enumerate(candidate_features):
        argmax_orig = original_features[int(np.argmax(np.abs(corr_cand_x_orig[i])))]
        flag = "KEEP" if surv_mask[i] else "DROP"
        LOGGER.info("    %-30s  max|r|=%+.3f (vs %s)  -> %s",
                    name, float(max_abs[i]), argmax_orig, flag)

    if not surv_names:
        LOGGER.warning(
            "  No candidates survived collinearity filter — falling back to originals (+ categoricals).")
        cat_keep = [n for n in categorical_features if n in name_to_col]
        cat_idx = [name_to_col[n] for n in cat_keep]
        aligned.X = aligned.X[:, orig_idx + cat_idx]
        return aligned, list(original_features) + cat_keep

    # ── Step B: Mutual Information ranking ────────────────────────────────
    X_surv = X_tr[:, surv_global_idx].astype(np.float64)
    rng = np.random.default_rng(mi_seed)
    if X_surv.shape[0] > mi_subsample:
        sub = rng.choice(X_surv.shape[0], size=mi_subsample, replace=False)
        X_mi, y_mi = X_surv[sub], y_tr[sub]
        LOGGER.info("  Step B | MI subsample = %d / %d rows", mi_subsample, X_surv.shape[0])
    else:
        X_mi, y_mi = X_surv, y_tr
    mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=mi_seed)

    ranked = sorted(zip(surv_names, mi_scores.tolist(), surv_global_idx),
                    key=lambda x: -x[1])
    LOGGER.info("  Step B | mutual information vs y_train (survivors only):")
    for name, mi, _ in ranked:
        LOGGER.info("    %-30s  MI=%.5f", name, mi)

    # ── Step C: Top-K cap ──────────────────────────────────────────────────
    k = min(top_k, len(ranked))
    top = ranked[:k]
    top_names = [n for n, _, _ in top]
    top_global = [i for _, _, i in top]
    LOGGER.info("  Step C | top-%d selected: %s", k, top_names)

    cat_keep = [n for n in categorical_features if n in name_to_col]
    cat_idx = [name_to_col[n] for n in cat_keep]
    keep_cols = orig_idx + top_global + cat_idx
    new_features = list(original_features) + top_names + cat_keep
    aligned.X = aligned.X[:, keep_cols]

    LOGGER.info("Feature selection DONE | final pool=%d features (incl %d categorical):",
                len(new_features), len(cat_keep))
    for n in new_features:
        if n in cat_keep:
            flag = "(categorical-forced)"
        elif n in original_features:
            flag = "(original)"
        else:
            flag = "(candidate-selected)"
        LOGGER.info("    %-30s  %s", n, flag)
    return aligned, new_features


def subset_features(aligned: AlignedData, all_features: list[str],
                    selected_features: list[str]) -> AlignedData:
    """Re-project a FULL-pool `aligned.X` down to `selected_features`, in that
    exact order.  This is the eval-side mirror of `select_features`: instead of
    re-running the (stochastic) iron-fist selection, run_backtest replays the
    train-time decision recorded in the checkpoint, guaranteeing the eval matrix
    matches the columns the ensemble was trained on.
    """
    missing = [n for n in selected_features if n not in all_features]
    if missing:
        raise ValueError(
            f"subset_features: {missing} not in materialized pool {all_features}. "
            "Checkpoint feature list is incompatible with the current pipeline.")
    idx = [all_features.index(n) for n in selected_features]
    aligned.X = aligned.X[:, idx]
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# 6. MATERIALIZE — the one call both entry-points use to build the dataset
# ─────────────────────────────────────────────────────────────────────────────

def materialize_dataset(cfg: RunConfig) -> Dataset:
    """ingest → features → labels → align → chronological split.

    Returns the FULL pre-selection dataset.  Feature *selection* is deliberately
    left to the caller: train_models runs `select_features` (and records the
    result); run_backtest runs `subset_features` (replaying that record).
    """
    with phase("Phase 5 — data ingestion"):
        ohlcv = load_ohlcv(cfg)

    with phase("Phase 1/1.5/9 — feature pipeline"):
        panel, all_features, candidate_features = build_features(ohlcv, cfg)
        # market_regime (and any future categoricals) are forced-survive + declared
        # categorical downstream — keep them OUT of the continuous "originals".
        original_features = [
            f for f in all_features
            if f not in candidate_features and f not in CATEGORICAL_FEATURES
        ]

    with phase("Phase 2 — triple-barrier labels + AFML §4.10 weights"):
        labels = build_labels(panel, cfg)

    with phase("Align features ↔ labels"):
        aligned = align(panel, labels, all_features)

    cutoff, train_mask = chronological_split(aligned, cfg.train_frac)
    LOGGER.info("Chronological split | cutoff=%s  train=%d  oos=%d",
                cutoff, int(train_mask.sum()), int((~train_mask).sum()))

    return Dataset(
        panel=panel, aligned=aligned, all_features=all_features,
        original_features=original_features, candidate_features=candidate_features,
        categorical_features=list(CATEGORICAL_FEATURES),
        cutoff=cutoff, train_mask=train_mask,
    )


__all__ = [
    "TRADING_DAYS",
    "FEATURE_RECIPE_VERSION",
    "FEATURE_SCHEMA",
    "CATEGORICAL_FEATURES",
    "configure_logging",
    "phase",
    "RunConfig",
    "AlignedData",
    "Dataset",
    "load_ohlcv",
    "load_corporate_actions",
    "build_features",
    "build_labels",
    "align",
    "chronological_split",
    "select_features",
    "subset_features",
    "materialize_dataset",
]
