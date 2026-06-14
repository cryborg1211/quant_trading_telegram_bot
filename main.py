import argparse
import html
import json
import logging
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import polars as pl                                          # V3 feature pipeline is polars-native

from config.settings import CONFIG
from src.features.alpha360_generator import Alpha360Generator
from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features
from src.features.market_regime import REGIME_LABELS_VI, regime_label_vi
from src.data import price_lookup  # fresh-parquet price lookups (stock_ohlcv retired)
# ─── V3 V1-faithful Tabular Ensemble (locked-in GOLDEN, the only inference route) ─
from src.bot.bot_inference import V3BotInference
from src.bot.sizing import suggested_weight
# Train/serve parity: live inference builds features through the SAME pipeline
# used by train_models.py — single source of truth, no duplicated feature math.
from src.backtest.pipeline import (
    RunConfig as V3FeatureConfig,
    build_features as build_v3_feature_panel,
    FEATURE_RECIPE_VERSION,
)
from src.models.quant_agent_arbitrator import (
    evaluate_trades_batch,
    get_rebalance_advice,
    map_tickers_to_news,
    scrape_centralized_news,
)
from src.trading.portfolio_manager import PortfolioManager
from src.trading import signal_ledger
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_FACTOR,
)
from src.utils.telegram_alerter import TelegramBot, format_source_links
from src.reports.builders import (
    FEATURE_HUMAN_NAMES,
    SHORT_HORIZON_DAYS,
    _MACRO_INNER_NAMES,
    _NUMERIC_SUFFIX_RE,
    _MACRO_PREFIX_RE,
    _REPORT_SEPARATOR,
    _SELL_DECISION,
    _MR_SELL_VETO,
    _VERIFY_5D_PRED_LABELS,
    _VERIFY_20D_PRED_LABELS,
    _VERIFY_VERDICT_LABELS,
    _REBALANCE_PRED_LABELS,
    _humanize_feature,
    _build_feature_explanation,
    _format_sentiment_status,
    _build_combined_report,
    _smart_truncate,
    _build_fallback_observability_report_vi,
    _build_sell_hold_report,
    _mr_state_line,
    _build_verify_report,
    _build_rebalance_report,
)

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
# OHLCV crawl guard: 15:00 ICT.
# Vietnamese market officially closes at 15:00. The daily candle is NOT
# considered finalized until then — fetching mid-day:
#   • returns an incomplete (still-moving) close price
#   • adds DB write pressure while reads are happening from the bot path
#   • poisons training data if the new partial candle gets persisted
# This must be 15:00 sharp (NOT 14:45 ATC) to ensure the close is final.
MARKET_CLOSE = dt_time(15, 0)



_VN_PRICE_SCALE_THRESHOLD = 1_000.0  # VN stocks quoted in thousands; raw < 1000 → multiply by 1000


def _get_live_exec_prices(latest_df: Any, tickers: list[str]) -> dict[str, float]:
    """
    Extract unscaled latest market prices from live feature frame.

    VN market convention: prices are stored in thousands (e.g. 10.5 = 10,500 VND).
    If extracted price < 1,000 we multiply by 1,000 to restore full VND value.
    """
    price_col = next((c for c in ("raw_close", "close", "price") if c in latest_df.columns), None)
    if price_col is None:
        LOGGER.warning("No raw price column found in latest_df. Portfolio/alerts will skip missing prices.")
        return {}

    out: dict[str, float] = {}
    for ticker in tickers:
        rows = latest_df[latest_df["ticker"].astype(str) == ticker]
        if rows.empty:
            continue
        price = float(rows.iloc[-1][price_col])
        if not np.isfinite(price) or price <= 0:
            continue
        # VN market stores prices in thousands (e.g. 10.5 → 10,500 VND)
        if price < _VN_PRICE_SCALE_THRESHOLD:
            price = price * 1_000.0
        out[ticker] = price
    return out



def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=False,
    )
    return logging.getLogger(__name__)


LOGGER = setup_logging()


@contextmanager
def timed_step(message: str):
    start = time.perf_counter()
    LOGGER.info("%s started...", message)
    try:
        yield
    finally:
        LOGGER.info("%s finished in %.2fs.", message, time.perf_counter() - start)


def is_crawl_allowed(force_crawl: bool = False) -> bool:
    """Return True iff the OHLCV crawl may run.

    Skips with a clear log if current VN local time is before `MARKET_CLOSE`
    (15:00 ICT). The caller is expected to continue straight into inference
    using whatever data is already in the DuckDB / parquet store rather than
    pulling fresh (and incomplete) candles.

    `force_crawl=True` bypasses the guard — for operator-initiated rebuilds.
    """
    now = datetime.now(VN_TZ)
    if force_crawl:
        LOGGER.warning("force_crawl=True. Market-hour crawl guard bypassed.")
        return True
    if now.time() < MARKET_CLOSE:
        LOGGER.warning(
            "[Crawler] Skipped OHLCV fetch. Current time is before %s. "
            "Using existing DB data.",
            MARKET_CLOSE.strftime("%H:%M"),
        )
        LOGGER.info(
            "[Crawler] Current VN local time=%s (threshold=%s ICT).",
            now.strftime("%H:%M:%S %Z"),
            MARKET_CLOSE.strftime("%H:%M"),
        )
        return False
    return True


def crawl_hose(
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    data_dir: str = "data",
    force_crawl: bool = False,
    days_back: int | None = None,
) -> None:
    """Overnight-safe HOSE OHLCV crawl.

    `days_back` (INCREMENTAL mode): when set, only the last `days_back` calendar
    days are fetched — `start_date` is overridden to (today − days_back) in VN
    time.  Use `--days-back 1` for a previous-day-only daily refresh; a small
    window (3–5) adds overlap against late data corrections.  When None, a FULL
    crawl from `start_date` runs.
    """
    from src.data.crawlers import StockCrawler

    if not is_crawl_allowed(force_crawl=force_crawl):
        return

    if days_back is not None:
        start_date = (datetime.now(VN_TZ) - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
        LOGGER.info("Incremental crawl | last %d day(s) → start_date=%s", int(days_back), start_date)

    LOGGER.info("Starting Quant V6 HOSE overnight crawler...")
    LOGGER.info("Start date: %s", start_date)
    LOGGER.info("End date: %s", end_date or datetime.now().strftime("%Y-%m-%d"))
    LOGGER.info("Data dir: %s", data_dir)
    LOGGER.info("Errors: logs/crawler_errors.txt")

    crawler = StockCrawler()
    with timed_step("HOSE overnight crawl"):
        summary = crawler.crawl_hose_overnight(
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
        )

    LOGGER.info("crawl_hose completed. Summary=%s", summary)


# (RETIRED: build_alpha360() + the Alpha360 feature factory were removed in the
#  parquet-first migration. The V4 pipeline recomputes features from raw OHLCV
#  via src/backtest/pipeline.build_features and never reads an Alpha360 matrix.)


# ─── Mean-Reversion (knife-catch) sub-model — shared live scorer ──────────
# Loaded once and cached (module-level) so /verify, /suggest_sell and
# /suggest_buy all reuse one in-memory model. Returns, per ticker:
#   {"prob": float, "fired": bool, "tau": float}
# `fired` == panic alert (prob >= the strict τ* from training).
_MR_MODEL: Any = None
_MR_TAU: float | None = None
_MR_ART = Path("models/mr")


def _load_mr() -> tuple[Any, float]:
    global _MR_MODEL, _MR_TAU
    if _MR_MODEL is None:
        _MR_MODEL = joblib.load(_MR_ART / "mr_lgbm.joblib")
        with (_MR_ART / "mr_threshold.json").open("r", encoding="utf-8") as fh:
            _MR_TAU = float(json.load(fh)["tau"])
        LOGGER.info("[MR] sub-model loaded | strict τ*=%.2f", _MR_TAU)
    return _MR_MODEL, float(_MR_TAU)


def mr_score_tickers(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Live MR panic score per ticker (leak-free oversold features).

    Loads an 80-bar OHLCV tail (≥ SMA50 history), runs build_mr_features,
    takes the LATEST row per ticker, and scores it with mr_lgbm. Any
    failure degrades to {} so the caller's primary flow is never broken.
    """
    tickers = sorted({t.upper().strip() for t in tickers if t})
    if not tickers:
        return {}
    try:
        model, tau = _load_mr()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[MR] model unavailable (%s) — MR features skipped.", exc)
        return {}
    try:
        win = Alpha360Generator()._load_live_stock_window(
            tickers=tickers, window_rows=80
        )
        pdf = win.to_pandas() if hasattr(win, "to_pandas") else win
        feat = build_mr_features(pdf)
        latest = (
            feat.sort_values(["ticker", "date"])
            .groupby("ticker", sort=False)
            .tail(1)
        )
        x = (
            latest[MR_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        proba = model.predict_proba(x)[:, 1]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[MR] scoring failed (%s) — MR features skipped.", exc)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for tk, p in zip(latest["ticker"].astype(str), proba, strict=False):
        out[tk.upper()] = {
            "prob": float(p),
            "fired": bool(p >= tau),
            "tau": float(tau),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# V3 TABULAR ENSEMBLE — drop-in replacement for the legacy V6 stacker
# ─────────────────────────────────────────────────────────────────────────────
# Loads the GOLDEN-config joblib bundle written by run_backtest.py and
# exposes the SAME 5-tuple contract as the legacy `predict_stacking_horizon`,
# so the four call sites in this file (daily_inference, inference_for_holdings,
# verify, rebalance_portfolio) need only swap the function name.

# DUAL-HORIZON ARTIFACT REGISTRY ─────────────────────────────────────────────
# Each horizon has its own GOLDEN bundle written by run_backtest.py
# (the artifact filename embeds the horizon to prevent cross-horizon overwrite).
# Loader is cached per horizon so the bot keeps a hot in-memory copy of each.
_V3_BUNDLE_PATHS: dict[int, Path] = {
    3:  Path("models/saved/v3_ensemble_3d.joblib"),
    5:  Path("models/saved/v3_ensemble_5d.joblib"),   # retired short horizon (kept loadable)
    20: Path("models/saved/v3_ensemble_20d.joblib"),
}
# Short horizon = /verify-only cross-check model (T+3). The daily broadcast
# dispatches the T+20 tranche book; the short model never drives sizing.
SHORT_HORIZON: int = SHORT_HORIZON_DAYS
# Legacy fallback for the un-suffixed bundle from V3.0 runs.
_V3_BUNDLE_LEGACY_FALLBACK = Path("models/saved/v3_ensemble.joblib")
_V3_BOT_CACHE: dict[int, V3BotInference] = {}


def _load_v3_bot(horizon: int = 5) -> V3BotInference:
    """Lazy-load + cache the V3 GOLDEN bundle for `horizon` ∈ {5, 20}.

    Looks for `models/saved/v3_ensemble_{horizon}d.joblib`.  If absent AND
    the horizon-agnostic legacy `v3_ensemble.joblib` exists, falls back to it
    (so V3.0 single-horizon installs keep working).  Raises a clear error if
    neither file is present — the operator must run
    `python train_models.py --tb-horizon <h>` then `python run_backtest.py`
    to produce the missing bundle.
    """
    if horizon in _V3_BOT_CACHE:
        return _V3_BOT_CACHE[horizon]
    path = _V3_BUNDLE_PATHS.get(int(horizon))
    if path is None or not path.exists():
        if _V3_BUNDLE_LEGACY_FALLBACK.exists():
            LOGGER.warning(
                "V3 horizon=%d bundle not found at %s — falling back to "
                "horizon-agnostic legacy %s.  Run `python train_models.py "
                "--tb-horizon %d` then `python run_backtest.py` to produce the proper artifact.",
                horizon, path, _V3_BUNDLE_LEGACY_FALLBACK, horizon,
            )
            path = _V3_BUNDLE_LEGACY_FALLBACK
        else:
            raise FileNotFoundError(
                f"V3 horizon={horizon} artifact not found at {path}. "
                f"Run `python train_models.py --tb-horizon {horizon}` then "
                f"`python run_backtest.py` first."
            )
    bot = V3BotInference.from_artifact(path)

    # ── FEATURE-RECIPE TRIPWIRE ──────────────────────────────────────────────
    # The artifact's structural feature recipe MUST match the live pipeline's.
    # A mismatch means `build_features` changed shape (a new/removed/retuned
    # feature, window, or ordering) since this model was trained → its live
    # inputs no longer mean what the model learned (silent train/serve skew).
    # Refuse to serve a drifted model; demand a retrain.
    artifact_recipe = bot.metadata.get("feature_recipe_version")
    if artifact_recipe is None:
        # Backward-compat: artifacts trained before the stamp existed.  Assume
        # compatible — the operator is expected to retrain to stamp the version.
        LOGGER.warning(
            "V3 horizon=%d artifact has no feature_recipe_version (pre-tripwire "
            "build) — assuming compatible with pipeline %s.  Retrain to stamp it.",
            horizon, FEATURE_RECIPE_VERSION,
        )
    elif artifact_recipe != FEATURE_RECIPE_VERSION:
        raise RuntimeError(
            f"FEATURE-RECIPE MISMATCH (horizon={horizon}): the artifact was trained "
            f"with feature_recipe_version={artifact_recipe!r}, but the live pipeline "
            f"is {FEATURE_RECIPE_VERSION!r}.  build_features has changed shape since "
            f"this model was trained, so its live features no longer match what the "
            f"model expects (train/serve skew).  RETRAIN: `python train_models.py "
            f"--tb-horizon {horizon}` then `python run_backtest.py` to regenerate {path}."
        )

    _V3_BOT_CACHE[int(horizon)] = bot
    LOGGER.info("V3 ensemble loaded (horizon=%d, recipe=%s): %s",
                horizon, artifact_recipe or "unstamped", bot.card())
    return bot


# ── V4.0 sizing note: Kelly math lives in `src/bot/sizing.py` (suggested_weight).
#    run_trade_execution computes the half-Kelly size for EACH dispatched ticker
#    from its own P(UP) and shows it ON that ticker's card.  There is NO summary
#    table (the old format_kelly_section_html block was purged).


# Latest market_regime (0–7) per ticker, refreshed on every _compute_v3_features
# pass.  Read by run_trade_execution for regime-aware sizing + the Telegram card.
# (Decoupled from the prediction return-tuple so the 4 call sites stay untouched.)
_LATEST_REGIME_BY_TICKER: dict[str, int] = {}


def _compute_v3_features(latest_df: pd.DataFrame, feature_list: list[str],
                         frac_diff_d: float) -> pd.DataFrame:
    """Build today's live feature row via the SHARED training pipeline.

    Train/serve parity is structural: this calls
    `src.backtest.pipeline.build_features` — the EXACT recipe `train_models.py`
    uses — so FracDiff, the cross-sectional Gaussian-rank Z-scores, anti-FOMO
    over-extension, the alpha factors AND the 5 advanced statistical features are
    all computed identically.  We then project to `feature_list` (the iron-fist
    selection persisted in the artifact's `tabular_features`), replaying the
    train-time selection rather than hardcoding a feature set.

    Input:   pandas DataFrame with (ticker, date, open, high, low, close,
             volume) + per-ticker history for the rolling windows (the Alpha360
             live path provides 120 bars).
    Output:  pandas DataFrame indexed by ticker, columns = `feature_list` in the
             model's trained order, one row per ticker (latest decision-bar).
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(latest_df.columns)
    if missing:
        raise ValueError(f"_compute_v3_features: missing OHLCV columns: {missing}")

    ohlcv = pl.from_pandas(latest_df[list(required)].copy()).sort(["ticker", "date"])
    # SAME recipe as training, with the EXACT frac_diff_d the artifact was trained
    # with (threaded down from the bundle metadata — never a library default).
    # The iron-fist SELECTION is replayed via feature_list.
    panel, all_features, _ = build_v3_feature_panel(
        ohlcv, V3FeatureConfig(frac_diff_d=frac_diff_d))

    unknown = [f for f in feature_list if f not in all_features]
    if unknown:
        raise ValueError(
            f"_compute_v3_features: the artifact requires features the live "
            f"pipeline does not produce: {unknown}. Train/serve recipe drift — "
            f"retrain against the current src/backtest/pipeline.py.")

    # Stash today's market_regime per ticker (the panel ALWAYS carries it, even if
    # a pre-regime artifact's feature_list does not) for regime-aware sizing + the
    # Telegram card.  One cheap tail(1) group-by on the already-built panel.
    if "market_regime" in panel.columns:
        _reg = (panel.sort(["ticker", "date"])
                     .group_by("ticker", maintain_order=True)
                     .tail(1)
                     .select(["ticker", "market_regime"]))
        _LATEST_REGIME_BY_TICKER.update({
            str(t): int(v)
            for t, v in zip(_reg["ticker"].to_list(), _reg["market_regime"].to_list())
            if v is not None
        })

    return (
        panel.sort(["ticker", "date"])
             .group_by("ticker", maintain_order=True)
             .tail(1)                                  # today's decision-bar per ticker
             .select(["ticker"] + list(feature_list))  # artifact's trained order
             .to_pandas()
             .set_index("ticker")
             .dropna()
    )


def predict_v3_horizon(latest_df: pd.DataFrame, horizon: int = 5) -> tuple[
    dict[str, list[float]], dict[str, Any], Any, list[str], dict[str, bool]
]:
    """DROP-IN replacement for `predict_stacking_horizon(latest_df, horizon)`.

    Returns the SAME 5-tuple shape — the four call sites in main.py and the
    downstream `_build_feature_explanation`, arbitrator, and meta-gate logic
    all keep working with no change:

        ( predictions_dict,   # {ticker: [p_down, p_flat, p_up]}
          thresholds,         # {'pnl_threshold_tau': ..., 'meta_labeler_enabled': False, ...}
          model_obj,          # V3BotInference (exposes .feature_importances_ + .feature_names_in_)
          selected_features,  # the artifact's iron-fist `tabular_features` (train/serve parity)
          meta_gate )         # {ticker: True if P(UP) >= up_threshold else False}

    The `horizon` parameter is accepted for API parity.  V3 was trained at the
    horizon stored in the bundle (T+5 for the current GOLDEN); other horizons
    return the SAME T+5 signal with a one-time warning so the arbitrator's
    5d-vs-20d compare degrades gracefully instead of crashing.
    """
    # Route to the horizon-specific bundle (5d ↔ v3_ensemble_5d.joblib, 20d ↔
    # v3_ensemble_20d.joblib).  _load_v3_bot is cached per horizon so dual-
    # horizon bot endpoints share their loaded models across calls.
    bot = _load_v3_bot(horizon)
    trained_h = int(bot.metadata.get("tb_horizon", horizon))
    if horizon != trained_h:
        LOGGER.warning(
            "predict_v3_horizon: bundle reports tb_horizon=%d but caller asked "
            "for %d.  Using the bundle as-is.", trained_h, horizon,
        )

    feature_list = list(bot.tabular_features)          # iron-fist selection from the artifact
    # RECIPE LOCK: conform to the artifact's training feature hyper-params, not
    # library defaults.  frac_diff_d is the only config-driven knob in
    # build_features; older artifacts without it fall back to the pipeline default.
    frac_diff_d = float(bot.metadata.get("frac_diff_d", V3FeatureConfig().frac_diff_d))
    feats = _compute_v3_features(latest_df, feature_list, frac_diff_d)
    if feats.empty:
        LOGGER.warning("predict_v3_horizon: V3 feature panel is empty after dropna.")
        return ({}, {"pnl_threshold_tau": bot.up_threshold}, bot, feature_list, {})

    proba_3 = bot.predict_proba_3class(feats)              # DataFrame indexed by ticker
    predictions_dict: dict[str, list[float]] = {
        str(t): [float(r["p_down"]), float(r["p_flat"]), float(r["p_up"])]
        for t, r in proba_3.iterrows()
    }
    thresholds: dict[str, Any] = {
        "pnl_threshold_tau": float(bot.up_threshold),
        "meta_labeler_enabled": False,
        "round_trip_cost": 0.008,                          # canonical VN cost (matches V6)
        "method": "v3_tabular_ensemble",
        "horizon_days": trained_h,
    }
    # V3 has no separate meta-labeler: the gate IS the up_threshold.
    meta_gate: dict[str, bool] = {
        str(t): bool(proba_3.loc[t, "p_up"] >= bot.up_threshold)
        for t in proba_3.index
    }
    return predictions_dict, thresholds, bot, feature_list, meta_gate


# ─────────────────────────────────────────────────────────────────────────────
# Legacy V6 stacker chain (load_stacking_artifacts / aligned_proba /
# predict_stacking_horizon) + _LATEST_5D_BREAKDOWN cache were PURGED in the
# V3.2 refactor.  All four call sites in this file (daily_inference,
# inference_for_holdings, verify_single_ticker, rebalance_portfolio) route
# through `predict_v3_horizon` above, which loads the per-horizon V3
# TabularEnsemble bundle (models/saved/v3_ensemble_{5,20}d.joblib).
# ─────────────────────────────────────────────────────────────────────────────



# ---------------------------------------------------------------------------
# TD-05: RL prediction logging + T+5 outcome backfill
# ---------------------------------------------------------------------------
# Before this fix, the `rl_mistake_logs` table was being filled with rows
# whose `actual_t5_outcome` was a hardcoded `-0.05` — making the entire
# table useless for Phase-3 RL training. The corrected flow is two-phase:
#
#   T0:  INSERT row with actual_t5_outcome = NULL for every high-confidence
#        UP prediction. We don't know the outcome yet.
#   T+5: UPDATE every NULL row whose predicted_date is ≥5 days in the past,
#        looking up the actual close prices from stock_ohlcv and computing
#        (t5_close - t0_close) / t0_close.
#
# Both phases run inside `run_trade_execution` on every daily_inference
# call, so backfill happens automatically on the next session after a
# 5-day window elapses.

_RL_UP_CONFIDENCE_THRESHOLD: float = 0.6  # only log strong UP predictions
_RL_HORIZON_DAYS: int = 20                # match the PRIMARY dispatch model (T+20)


def _log_rl_predictions(
    db: Any,
    predictions_5d: dict[str, list[float]],
    top_pos_features_text: str,
) -> int:
    """Phase 1 (T0): record every high-confidence UP prediction with NULL outcome.

    The actual outcome is filled in by `_backfill_rl_outcomes` once ≥5
    trading days have elapsed. Returns the number of rows inserted today.

    TD-50: each INSERT runs under `db._audit_lock` so concurrent writers
    (the bot's audit-log thread + this RL thread) serialize cleanly
    instead of relying on DuckDB's internal mutex alone.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    logged = 0
    for ticker, probs in predictions_5d.items():
        if not probs or len(probs) < 3:
            continue
        if probs[2] <= _RL_UP_CONFIDENCE_THRESHOLD:
            continue
        features_json = json.dumps(
            {
                "model": f"stacking_gbdt_{_RL_HORIZON_DAYS}d",
                "p_up": round(float(probs[2]), 4),
                "top_drivers": top_pos_features_text,
            },
            ensure_ascii=False,
        )
        with db._audit_lock:
            db.conn.execute(
                """
                INSERT INTO rl_mistake_logs
                (ticker, predicted_date, predicted_action, actual_t5_outcome, features_snapshot)
                VALUES (?, ?, 'BUY', NULL, ?)
                """,
                [ticker, today_str, features_json],
            )
        logged += 1
    return logged


def _backfill_rl_outcomes(db: Any) -> int:
    """Phase 2 (T+5+): fill in `actual_t5_outcome` for rows where predicted_date
    is ≥5 days in the past and outcome is still NULL.

    For each pending row, looks up:
        • t0_close = stock_ohlcv close on `predicted_date`
                     (or the most recent trading day on/before it — handles
                      weekends/holidays defensively)
        • t5_close = first available close from stock_ohlcv whose date is
                     ≥ predicted_date + 5 days

    Computes `(t5_close - t0_close) / t0_close` as the actual % return
    and UPDATEs the row. Rows with missing OHLCV (delisted, illiquid)
    stay NULL and will be retried on the next pipeline run.

    Returns the number of rows backfilled in this run.

    TD-50: the UPDATE write runs under `db._audit_lock` so concurrent
    writers serialize. The two SELECT lookups are read-only and don't
    need the lock — DuckDB's internal mutex handles read safety.
    """
    pending = db.conn.execute(
        f"""
        SELECT ticker, predicted_date, predicted_action
        FROM rl_mistake_logs
        WHERE actual_t5_outcome IS NULL
          AND predicted_date <= CURRENT_DATE - INTERVAL {_RL_HORIZON_DAYS} DAY
        """
    ).fetchall()

    backfilled = 0
    for ticker, predicted_date, action in pending:
        # Price lookups now hit the FRESH parquet vintage (the same source the
        # model trains/serves on) via price_lookup — the stale stock_ohlcv table
        # was retired. Semantics unchanged:
        #   t0 = close on/before predicted_date (walks back over weekends);
        #   t5 = first close on/after predicted_date + horizon.
        t0_close = price_lookup.close_on_or_before(ticker, predicted_date, conn=db.conn)
        t5_close = price_lookup.close_on_or_after(
            ticker, predicted_date + timedelta(days=_RL_HORIZON_DAYS), conn=db.conn
        )
        if t0_close is None or t5_close is None or t0_close <= 0:
            continue

        actual = (t5_close - t0_close) / t0_close
        # TD-50: UPDATE is the WRITE — lock it.
        with db._audit_lock:
            db.conn.execute(
                """
                UPDATE rl_mistake_logs
                SET actual_t5_outcome = ?
                WHERE ticker = ? AND predicted_date = ? AND predicted_action = ?
                  AND actual_t5_outcome IS NULL
                """,
                [actual, ticker, predicted_date, action],
            )
        backfilled += 1
    return backfilled


# VN30 constituents — STRICT live universe gate (avoid noisy mid-caps → false
# signals in weak markets). Update on the quarterly VN30 review.
_VN30_UNIVERSE: frozenset[str] = frozenset({
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
})


def _select_candidates(
    predictions: dict[str, list[float]],
    meta_gate: dict[str, bool],
    vn30_universe: frozenset[str],
    max_candidates: int,
) -> tuple[list[str], set[str], bool, dict[str, str]]:
    """VN30 gate + meta-gate filter + top-N sort + fallback mode.

    Returns (candidate_tickers, universe_tickers, fallback_mode, fallback_reasons).
    """
    _ARBITRATOR_POOL = 6

    liquid_tickers: set[str] = {
        str(t) for t in predictions if str(t).upper() in vn30_universe
    }
    LOGGER.info("[VN30Gate] %s / %s predicted tickers are in VN30.",
                len(liquid_tickers), len(predictions))
    if not liquid_tickers:
        LOGGER.warning("[VN30Gate] no predicted ticker in VN30 — using all predictions as fallback.")
        liquid_tickers = set(predictions.keys())

    universe_tickers: set[str] = set(liquid_tickers)

    _gated_out = [
        t for t, _p in sorted(predictions.items(),
                               key=lambda i: i[1][2], reverse=True)
        if t in universe_tickers and not meta_gate.get(t, True)
    ]
    candidate_tickers = [
        ticker
        for ticker, _probs in sorted(
            predictions.items(),
            key=lambda item: item[1][2],
            reverse=True,
        )
        if ticker in universe_tickers and meta_gate.get(ticker, True)
    ][: min(max_candidates, _ARBITRATOR_POOL)]
    LOGGER.info(
        "[Brain] Meta-labeler gate: %s liquid tickers rejected (e.g. %s). "
        "Top-%s survivors → arbitrator pool: %s",
        len(_gated_out), _gated_out[:5], len(candidate_tickers), candidate_tickers,
    )

    fallback_mode = False
    fallback_reasons: dict[str, str] = {}
    if not candidate_tickers:
        fallback_mode = True
        _tau5 = 0.45
        _ranked = [
            t for t, _p in sorted(
                predictions.items(),
                key=lambda kv: kv[1][2], reverse=True,
            )
            if t in universe_tickers
        ]
        candidate_tickers = _ranked[:3]
        _floor_pct = _tau5 * 100.0
        for t in candidate_tickers:
            _pu = predictions[t][2] * 100.0
            if _pu < _floor_pct:
                fallback_reasons[t] = (
                    f"Cửa tăng chỉ {_pu:.0f}% (dưới ngưỡng an toàn "
                    f"{_floor_pct:.0f}%) — không bõ công đánh đổi với rủi ro "
                    f"thị trường chung đang yếu."
                )
            elif not meta_gate.get(t, True):
                fallback_reasons[t] = (
                    "Cửa tăng tạm ổn nhưng kỳ vọng lợi nhuận không đủ bù "
                    "chi phí và rủi ro — bộ lọc an toàn loại bỏ."
                )
            else:
                fallback_reasons[t] = (
                    f"Chỉ để theo dõi: thị trường yếu, cửa tăng {_pu:.0f}%."
                )
        LOGGER.warning(
            "[FallbackObservability] No gated candidates — MONITORING-ONLY "
            "Top-3 by P(UP): %s (tau*=%.2f). These will NOT be traded.",
            candidate_tickers, _tau5,
        )

    return candidate_tickers, universe_tickers, fallback_mode, fallback_reasons


def _rescue_loop(
    fallback_mode: bool,
    stacking_predictions_5d: dict[str, list[float]],
    universe_tickers: set[str],
    top_buy_signals: list[str],
    all_sentiments: dict[str, Any],
    horizon_predictions: dict[str, dict],
) -> tuple[list[str], dict[str, dict]]:
    """Rescue bull-bypass + bear veto event layer.

    Side effect: may extend `all_sentiments` in-place with fetched rescue
    candidate data.
    """
    if fallback_mode:
        return top_buy_signals, {}

    _rescue_pool = [
        t for t, _p in stacking_predictions_5d.items()
        if t in universe_tickers and t not in top_buy_signals
        and EVENT_MIN_P_UP <= float(_p[2]) < SAFE_BUY_THRESHOLD
    ]
    _missing = [t for t in _rescue_pool if t not in all_sentiments]
    if _missing:
        try:
            _, _resc_sent = evaluate_trades_batch(horizon_predictions, _missing)
            all_sentiments.update(_resc_sent)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[Rescue] sentiment fetch failed for %s: %s", _missing, exc)
    event_overrides, _rescued = build_event_overrides(
        stacking_predictions_5d, all_sentiments, universe_tickers, top_buy_signals)
    if _rescued:
        extended = list(top_buy_signals) + _rescued
    else:
        extended = top_buy_signals
    if event_overrides:
        LOGGER.warning("[EventLayer] overrides=%s (rescued=%s)",
                       {t: o["status"] for t, o in event_overrides.items()}, _rescued)
    return extended, event_overrides


def daily_inference(
    window_rows: int = 120,
    max_candidates: int = 6,
    broadcast: bool = True,
    horizon: int = 20,
) -> str:
    """Daily trading path. No crawling. No full Alpha360 parquet load.

    Args:
        window_rows: per-ticker rows to load for live Alpha360 features.
        max_candidates: Top-N pool size sent to the arbitrator (default 6).
        broadcast: If True (default, cron path), push per-ticker alerts to
            every chat ID in `TELEGRAM_CHAT_ID` env. If False (bot on-demand
            path) suppress those pushes — caller is responsible for routing
            the returned report to its own chat. This prevents duplicate
            alerts when `/suggest_buy` triggers `daily_inference()` from the
            interactive bot.

    Returns:
        Combined HTML report of the dispatched Top-3 BUY signals (one block
        per ticker, separated by a horizontal rule). Suitable for posting to
        Telegram with `parse_mode=HTML`. Returns "" when no signals were
        produced (e.g. liquidity filter cleared the pool, or no live prices).
    """
    LOGGER.info("Starting dual-horizon daily inference. No crawling will run in this task.")
    total_start = time.perf_counter()

    with timed_step(f"Building live Alpha360 features from recent OHLCV/macro windows ({window_rows} rows/ticker)"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    LOGGER.info("Live feature frame loaded: %s rows x %s cols.", len(latest_df), len(latest_df.columns))
    if latest_df.empty:
        raise ValueError("Live feature frame is empty.")

    # DUAL-HORIZON: the user-selected `horizon` drives the primary signal +
    # the Kelly sizing block, while the OTHER horizon is fetched for the
    # arbitrator cross-check.  Variable names `_5d` / `_20d` are kept for
    # back-compat with the downstream arbitrator / sentiment code, but they
    # now mean "primary" / "secondary".
    stacking_predictions_5d, thr_5d, xgb_model_5d, selected_features_5d, meta_gate_5d = predict_v3_horizon(latest_df, int(horizon))
    # SECONDARY horizon = arbitrator cross-check ONLY; it must NEVER abort the
    # PRIMARY command.  If the OTHER brain's artifact is missing or version-
    # mismatched, degrade to an empty dict (evaluate_trades_batch falls back to
    # the primary probs per-ticker).  Short horizon = T+3 (verify-only model);
    # it serves here purely as the arbitrator's second opinion.
    _secondary_h = 20 if int(horizon) == SHORT_HORIZON else SHORT_HORIZON
    try:
        stacking_predictions_20d, _, _, _, _ = predict_v3_horizon(latest_df, _secondary_h)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning(
            "Secondary horizon T+%d unavailable (%s) — running PRIMARY T+%d only; "
            "arbitrator loses its dual-horizon cross-check.",
            _secondary_h, exc.__class__.__name__, int(horizon),
        )
        stacking_predictions_20d = {}

    # ── V4.0 KELLY SIZING ─────────────────────────────────────────────────
    # Per-ticker half-Kelly size is computed from EACH dispatched ticker's OWN
    # P(UP) inside run_trade_execution (src/bot/sizing.suggested_weight) and
    # shown on its card.  No pre-filtered lookup map: the old map gated at
    # P>=0.50 + top-5, so dispatched names below that threshold rendered "N/A".

    sorted_preds = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=True)
    top_10_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in sorted_preds[:10]])
    LOGGER.info("[StackingGBDT] TOP 10 UP PROBS: %s", top_10_str)

    bottom_3_sorted = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=False)[:3]
    bottom_3_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in bottom_3_sorted])
    LOGGER.info("[StackingGBDT] BOTTOM 3 RISK: %s", bottom_3_str)

    candidate_tickers, universe_tickers, fallback_mode, fallback_reasons = _select_candidates(
        stacking_predictions_5d, meta_gate_5d, _VN30_UNIVERSE, max_candidates
    )

    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}
    with timed_step("Evaluating candidates with dual-horizon arbitrator/sentiment layer"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, candidate_tickers)

    # Fallback returns the flagged Vietnamese observability report and
    # BYPASSES run_trade_execution — guaranteeing no trades are executed.
    if fallback_mode:
        if not candidate_tickers:
            return (
                "<b>[⚠️ THỊ TRƯỜNG YẾU]</b>\n<i>Không có mã nào trong vũ trụ "
                "giao dịch sau bộ lọc thanh khoản/Universe hôm nay.</i>"
            )
        # MR knife-catch scores for the monitored names → [🔪 BẮT ĐÁY] tag.
        mr_scores = mr_score_tickers(candidate_tickers)
        fb_prices = _get_live_exec_prices(latest_df, candidate_tickers)
        report_html = _build_fallback_observability_report_vi(
            candidate_tickers,
            stacking_predictions_5d,
            all_sentiments,
            fallback_reasons,
            mr_scores,
            live_prices=fb_prices,
        )
        LOGGER.warning(
            "[FallbackObservability] Returning monitoring-only report for %s "
            "— run_trade_execution SKIPPED (no trades).", candidate_tickers,
        )
        LOGGER.info(
            "Dual-horizon daily inference completed in %.2fs.",
            time.perf_counter() - total_start,
        )
        return report_html

    # --- Top-6 → Top-3 Sentiment Filter ---
    # Scope strictly to candidate_tickers (the 6 evaluated by arbitrator+LLM).
    # Sort primarily by sentiment_score DESC, secondarily by 5d quant probability DESC.
    # Only candidates that passed arbitration (decision == 2) are eligible for dispatch.
    _BUY_DECISION = 2
    evaluated_buys = [
        t for t in candidate_tickers
        if final_decisions.get(t) == _BUY_DECISION
    ]
    top_buy_signals = sorted(
        evaluated_buys,
        key=lambda t: (
            float(all_sentiments.get(t, {}).get("sentiment_score", 0.0)),  # primary  ↓ desc
            float(stacking_predictions_5d.get(t, [0, 0, 0])[2]),          # secondary ↓ desc
        ),
        reverse=True,
    )[:3]

    # Log full ranking for auditability
    _rank_str = " | ".join(
        f"{t} [sent={all_sentiments.get(t,{}).get('sentiment_score',0.0):+.2f}"
        f" quant={stacking_predictions_5d.get(t,[0,0,0])[2]*100:.1f}%]"
        for t in sorted(
            candidate_tickers,
            key=lambda t: (
                float(all_sentiments.get(t, {}).get("sentiment_score", 0.0)),
                float(stacking_predictions_5d.get(t, [0, 0, 0])[2]),
            ),
            reverse=True,
        )
    )
    LOGGER.info("[Brain] Sentiment-ranked pool (Top6): %s", _rank_str)
    LOGGER.info("[Brain] Top-3 Buy Signals after sentiment filter: %s", top_buy_signals)

    top_buy_signals, event_overrides = _rescue_loop(
        fallback_mode,
        stacking_predictions_5d,
        universe_tickers,
        top_buy_signals,
        all_sentiments,
        horizon_predictions,
    )

    report_html = run_trade_execution(
        top_buy_signals=top_buy_signals,
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        stacking_predictions=horizon_predictions,
        latest_df=latest_df,
        xgb_model_5d=xgb_model_5d,
        selected_features_5d=selected_features_5d,
        horizon=int(horizon),
        broadcast=broadcast,
        event_overrides=event_overrides,
    )
    LOGGER.info("Dual-horizon daily inference completed in %.2fs.", time.perf_counter() - total_start)
    return report_html


# ── Event-driven thresholds — used by the rescue + bear-veto loops in
#    daily_inference (additive event_overrides; core technical gate untouched). ──
SAFE_BUY_THRESHOLD = 0.45      # standard P(UP) technical gate
EVENT_MIN_P_UP = 0.42         # rescue floor: 0.42 ≤ P(UP) < 0.45
EVENT_BULL_SENTIMENT = 0.60   # very bullish Gemini sentiment → rescue
EVENT_BEAR_SENTIMENT = -0.50  # very bearish sentiment → bear veto (hard block)
_EVENT_CAP = 0.05             # 5% NAV HARD cap for rescued (news-probe) entries

# (_arbitrate_signal removed — superseded by the additive event_overrides rescue +
#  Bear-VETO loops in daily_inference. The thresholds above are still used there.)


def build_event_overrides(
    stacking_predictions_5d: dict, all_sentiments: dict,
    universe_tickers, top_buy_signals,
) -> tuple[dict[str, dict], list[str]]:
    """PURE (no I/O): event_overrides from calibrated P(UP) + Gemini sentiment.

      • RESCUE (bull bypass): name the 0.45 gate rejected but
        EVENT_MIN_P_UP ≤ P(UP) < SAFE_BUY_THRESHOLD AND sentiment ≥
        EVENT_BULL_SENTIMENT → forced EVENT-DRIVEN @ _EVENT_CAP (5%).
      • BEAR VETO: approved technical BUY with sentiment ≤
        EVENT_BEAR_SENTIMENT → hard-blocked (0% NAV).

    Returns (overrides, rescued_tickers). Tested: tests/test_event_overrides.py.
    """
    universe = set(universe_tickers)
    held = set(top_buy_signals)
    overrides: dict[str, dict] = {}
    rescued: list[str] = []

    for t, p in stacking_predictions_5d.items():     # RESCUE
        if t in held or t not in universe:
            continue
        pu = float(p[2])
        if not (EVENT_MIN_P_UP <= pu < SAFE_BUY_THRESHOLD):
            continue
        sd = all_sentiments.get(t, {}) or {}
        ss = float(sd.get("sentiment_score", 0.0) or 0.0)
        if ss < EVENT_BULL_SENTIMENT:
            continue
        reason = str(sd.get("reasoning_vi") or sd.get("reason_vi")
                     or sd.get("explanation") or "tin tức tích cực mạnh").strip()
        overrides[t] = {
            "status": "EVENT-DRIVEN (BẮT TIN)",
            "weight": _EVENT_CAP,
            "ly_do": (
                f"P(Tăng)={pu * 100:.1f}% (dưới ngưỡng an toàn "
                f"{SAFE_BUY_THRESHOLD * 100:.0f}% nhưng ≥ {EVENT_MIN_P_UP * 100:.0f}%) "
                f"+ sentiment={ss:+.2f} ≥ {EVENT_BULL_SENTIMENT:.2f}, GIỚI HẠN "
                f"{_EVENT_CAP * 100:.0f}% NAV → Bắt tin: {_smart_truncate(reason, 160)}"
            ),
        }
        rescued.append(t)

    for t in top_buy_signals:                        # BEAR VETO
        if t in overrides:
            continue
        ss = float(all_sentiments.get(t, {}).get("sentiment_score", 0.0) or 0.0)
        if ss <= EVENT_BEAR_SENTIMENT:
            overrides[t] = {
                "status": "HỦY BỎ (TIN XẤU)",
                "weight": 0.0,
                "ly_do": (
                    f"Kỹ thuật ủng hộ MUA nhưng tin tức RẤT xấu (sentiment={ss:+.2f} ≤ "
                    f"{EVENT_BEAR_SENTIMENT:+.2f}) → PHỦ QUYẾT, chặn lệnh (0% NAV)."
                ),
            }
    return overrides, rescued


def _tranche_signal_fields(strategy: dict | None, n_picks: int) -> dict:
    """Tranche-mode dispatch enrichment from the artifact's `strategy` dict.

    Returns {} for legacy artifacts (no strategy / non-tranche mode) so the
    half-Kelly path is untouched.  Otherwise: the per-name NAV weight the
    backtest was validated under (NAV/hold_days split across today's picks),
    the trading-day exit date, and the optional PT/SL barrier rule.
    """
    if not strategy or strategy.get("mode") != "tranche":
        return {}
    hold_days = int(strategy.get("hold_days", 30))
    weight = 1.0 / (hold_days * max(1, n_picks))
    exit_date = pd.bdate_range(datetime.now().date(), periods=hold_days + 1)[-1]
    fields = {
        "suggested_weight": weight,
        "hold_label": f"{hold_days} phiên (đến ~{exit_date.strftime('%d/%m/%Y')})",
    }
    pt, sl = strategy.get("pt_sigma"), strategy.get("sl_sigma")
    if pt is not None or sl is not None:
        parts = []
        if pt is not None:
            parts.append(f"chốt lời sớm tại +{float(pt):.1f}σ")
        if sl is not None:
            parts.append(f"cắt lỗ tại −{float(sl):.1f}σ")
        fields["exit_rule"] = " / ".join(parts)
    return fields


def _dispatch_signals(
    top_buy_signals: list[str],
    all_sentiments: dict[str, Any],
    stacking_predictions: dict[str, dict],
    live_exec_prices: dict[str, float],
    event_overrides: dict[str, dict] | None,
    top_pos_features: str,
    top_neg_features: str,
    horizon: int,
    broadcast: bool,
    bot: TelegramBot,
    strategy: dict | None = None,
) -> list[dict]:
    """Per-ticker signal build + Telegram dispatch loop.

    Reads module-level global `_LATEST_REGIME_BY_TICKER` set by
    `_compute_v3_features`. This coupling is intentional and documented.

    `strategy` — the artifact's portfolio-construction dict.  Tranche mode
    overrides the half-Kelly size with the validated cohort weight and adds
    hold-horizon / barrier guidance to the card; None keeps legacy behavior.
    """
    tranche_fields = _tranche_signal_fields(strategy, len(top_buy_signals))
    dispatched_signals: list[dict] = []
    for ticker in top_buy_signals:
        exec_price = live_exec_prices.get(ticker)
        if exec_price is None:
            LOGGER.warning("Skipping Telegram alert for %s: no live market price.", ticker)
            continue

        sentiment_data = all_sentiments.get(ticker, {})
        _reason_vi = str(sentiment_data.get("reasoning_vi", "Chưa có dữ liệu tin tức đáng kể."))
        _reason_vi = _reason_vi.replace("None", "").strip() or "Chưa có dữ liệu tin tức đáng kể."
        _reason_vi = _smart_truncate(_reason_vi, 800)
        source_urls: list[str] = (sentiment_data.get("source_urls", []) or [])[:6]
        _p5 = stacking_predictions.get("5d", {}).get(ticker, [0, 0, 0])
        confidence_5d = round(_p5[2] * 100, 2)
        _regime = _LATEST_REGIME_BY_TICKER.get(ticker)
        LOGGER.info("[Alert] %s source_urls=%s regime=%s", ticker, source_urls, _regime)

        _ov = (event_overrides or {}).get(ticker)
        if _ov:
            _w = float(_ov["weight"])
            _status = _ov["status"]
            _ly_do = _ov["ly_do"]
        else:
            # ── Regime-conditional sizing (serve/backtest parity) ──────────────
            # else-branch ONLY → event overrides above keep precedence. Mirrors
            # walk_forward._tranche_day: NO_TRADE {0,7} skip the name (its cohort
            # weight stays cash — the n_picks denominator in tranche_fields was
            # frozen before the loop, so survivors are NOT inflated); PENALTY {1,6}
            # → 0.5x notional in tranche mode only (the legacy half-Kelly path
            # already penalises inside suggested_weight via REGIME_PENALTY_CAP).
            if (CONFIG.trading.regime_sizing_enabled
                    and _regime is not None
                    and _regime in NO_TRADE_REGIMES):
                LOGGER.info("[Regime] %s skipped — NO_TRADE regime %s", ticker, _regime)
                continue
            # Tranche artifacts size at the validated cohort weight
            # (NAV/hold_days across picks); legacy artifacts keep half-Kelly.
            _w = tranche_fields.get(
                "suggested_weight",
                suggested_weight(float(_p5[2]), market_regime=_regime))
            if (CONFIG.trading.regime_sizing_enabled
                    and tranche_fields
                    and _regime is not None
                    and _regime in PENALTY_REGIMES):
                _w = _w * REGIME_PENALTY_FACTOR
                LOGGER.info("[Regime] %s PENALTY regime %s -> weight x0.5", ticker, _regime)
            _status = "MUA"
            _ly_do = ""

        signal_data = {
            "action": "MUA",
            "ticker": ticker,
            "price": f"{exec_price:,.0f} VND",
            "horizon_label": f"T+{int(horizon)}",
            "suggested_weight": _w,
            "status": _status,
            "ly_do": _ly_do,
            "market_regime": _regime,
            "regime_label": regime_label_vi(_regime),
            "prob_up": round(_p5[2] * 100, 1),
            "prob_side": round(_p5[1] * 100, 1),
            "prob_down": round(_p5[0] * 100, 1),
            "conclusion": _reason_vi,
            "sentiment_score": sentiment_data.get("sentiment_score", 0.0),
            "sentiment_status": _format_sentiment_status(sentiment_data),
            "gemini_summary": _reason_vi,
            "article_urls": source_urls,
            "confidence": confidence_5d,
            "top_pos_features": top_pos_features,
            "top_neg_features": top_neg_features,
        }
        if "hold_label" in tranche_fields:
            signal_data["hold_label"] = tranche_fields["hold_label"]
        if "exit_rule" in tranche_fields:
            signal_data["exit_rule"] = tranche_fields["exit_rule"]

        if broadcast:
            bot.send_signal_alert(signal_data)
        dispatched_signals.append(signal_data)
    return dispatched_signals


def run_trade_execution(
    top_buy_signals: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    latest_df: Any,
    xgb_model_5d: Any,
    selected_features_5d: list[str],
    horizon: int = 20,
    broadcast: bool = True,
    event_overrides: dict | None = None,
) -> str:
    """Execute portfolio updates, RL outcome logging, and dispatch Telegram alerts.

    Args:
        stacking_predictions: dual-horizon dict {"5d": {...}, "20d": {...}} produced by the
            Stacking GBDT (XGBoost+LightGBM+CatBoost → logistic meta) model.
        horizon: primary horizon (5 or 20) — rendered as the "T+{h} Model"
            label at the top of each card. The half-Kelly position size is
            computed per-ticker from its own P(UP) (suggested_weight) below.
        broadcast: When False, the per-ticker push alert via
            `TelegramBot.send_signal_alert()` is skipped. The combined HTML
            report is still built and returned. Used by the interactive bot
            path (`/suggest_buy`) to avoid duplicating alerts.

    Returns:
        Combined HTML report of every signal dispatched (for chat-reply UX),
        or "" if no signals were dispatched.
    """
    LOGGER.info("Starting Trade Execution (Portfolio Manager)...")
    dispatched_signals: list[dict] = []
    try:
        manager = PortfolioManager()
        live_exec_prices = _get_live_exec_prices(latest_df, top_buy_signals)

        if not live_exec_prices:
            LOGGER.warning("No executable live prices available. Skipping portfolio price updates and alerts.")
            return ""

        with timed_step("Portfolio update/process_daily_trades"):
            manager.update_live_performance(live_exec_prices)
            manager.process_daily_trades(
                top_buy_signals=top_buy_signals,
                next_day_open_prices=live_exec_prices,
                predictions=final_decisions,
            )

        # TD-05: replaced the hardcoded `actual_t5_outcome = -0.05` stub with
        # a proper two-phase logger:
        #   (1) at T0 — INSERT a row per high-confidence UP prediction with
        #               actual_t5_outcome = NULL (we don't know the outcome yet).
        #   (2) at T+5+ — UPDATE any old NULL rows by looking up real close
        #                 prices from stock_ohlcv and computing the actual return.
        # Both phases run every daily_inference call, so backfill happens on the
        # next session after a 5-day window elapses for any given prediction.
        with timed_step("RL prediction logging (T0 INSERT + T+5 backfill UPDATE)"):
            db = manager.db
            predictions_5d = stacking_predictions.get("5d", stacking_predictions)
            top_pos_text, _ = _build_feature_explanation(
                xgb_model_5d, selected_features_5d, top_k=3
            )
            logged = _log_rl_predictions(db, predictions_5d, top_pos_text)
            backfilled = _backfill_rl_outcomes(db)
            LOGGER.info(
                "RL T0 logged=%s (UP prob > %.2f); T+5 backfilled=%s",
                logged, _RL_UP_CONFIDENCE_THRESHOLD, backfilled,
            )

        LOGGER.info("Dispatching Telegram Alerts...")
        bot = TelegramBot()
        top_pos_features, top_neg_features = _build_feature_explanation(
            xgb_model_5d,
            selected_features_5d,
            top_k=3,
        )

        # Market-summary HEADER as a SEPARATE first message. Each stock card then
        # sends as its own message below; `_dispatch` already sleeps 0.5s between
        # every send → Telegram Flood-Control safe, and no single message nears 4096.
        if broadcast and top_buy_signals:
            _hdr = (
                f"📊 <b>BÁO CÁO TÍN HIỆU NGÀY {datetime.now().strftime('%d/%m/%Y')}</b>\n"
                f"Số mã phân tích: <b>{len(top_buy_signals)}</b> (mô hình T+{int(horizon)}).\n"
                f"<i>Chi tiết từng mã ở các tin nhắn bên dưới.</i>"
            )
            bot.send_text_alert(_hdr, label="header")

        # Portfolio-construction contract from the loaded artifact (cached per
        # horizon).  Missing artifact / pre-tranche bundle → None → legacy
        # half-Kelly dispatch.
        try:
            _strategy = _load_v3_bot(int(horizon)).strategy or None
        except Exception:  # noqa: BLE001 — dispatch must not die on artifact issues
            _strategy = None

        dispatched_signals = _dispatch_signals(
            top_buy_signals=top_buy_signals,
            all_sentiments=all_sentiments,
            stacking_predictions=stacking_predictions,
            live_exec_prices=live_exec_prices,
            event_overrides=event_overrides,
            top_pos_features=top_pos_features,
            top_neg_features=top_neg_features,
            horizon=horizon,
            broadcast=broadcast,
            bot=bot,
            strategy=_strategy,
        )
        sent = len(dispatched_signals)
        LOGGER.info("Telegram alerts dispatched: %s (broadcast=%s)", sent, broadcast)

        # Tranche exit ledger: book the dispatched cohort so full_pipeline can
        # alert when its hold horizon elapses.  Broadcast-only — interactive
        # /suggest_buy previews are not committed positions.
        if broadcast and dispatched_signals:
            signal_ledger.record_dispatch(dispatched_signals, _strategy, int(horizon))

    except Exception:
        LOGGER.exception("Error during trade execution")
        return _build_combined_report(dispatched_signals)

    return _build_combined_report(dispatched_signals)


# ---------------------------------------------------------------------------
# /suggest_sell — on-demand inference for an arbitrary ticker list
# ---------------------------------------------------------------------------

def inference_for_holdings(
    holding_tickers: list[str],
    window_rows: int = 120,
) -> str:
    """Run dual-horizon Stacking GBDT + arbitrator on the user's holdings only.

    Skips the liquidity gate and the Top-6/Top-3 funnel — every holding gets
    a recommendation. Used by the /suggest_sell bot command.

    Args:
        holding_tickers: tickers from the `portfolio` DuckDB table.
        window_rows: per-ticker rows for live Alpha360 features.

    Returns:
        HTML report (HTML-escaped) suitable for `parse_mode=HTML`. Empty
        string if no live features or no overlap with the live universe.
    """
    if not holding_tickers:
        return ""

    holding_tickers = sorted({t.upper().strip() for t in holding_tickers if t})
    LOGGER.info("Holdings inference for %s tickers: %s", len(holding_tickers), holding_tickers)
    total_start = time.perf_counter()

    with timed_step(f"Building live Alpha360 features ({window_rows} rows/ticker) for /suggest_sell"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    if latest_df.empty:
        LOGGER.warning("[/suggest_sell] live feature frame is empty.")
        return ""

    universe = set(latest_df["ticker"].astype(str).tolist())
    present = [t for t in holding_tickers if t in universe]
    missing = [t for t in holding_tickers if t not in universe]
    if missing:
        LOGGER.warning("[/suggest_sell] holdings absent from live universe: %s", missing)

    if not present:
        # Build an "empty" report that still warns about missing tickers.
        return _build_sell_hold_report(
            holding_tickers=[],
            final_decisions={},
            all_sentiments={},
            stacking_predictions={},
            live_exec_prices={},
            missing_tickers=missing,
        )

    # CROSS-SECTIONAL PARITY: predict over the FULL universe (do NOT slice to the
    # held set first, or the `_xsz` ranks degenerate); the predictions dict is
    # keyed by ticker, so the arbitrator / exec-price / MR steps below read off
    # `present`.  SECONDARY horizon (20d) is arbitrator cross-check only → non-fatal.
    stacking_predictions_5d, _, _, _, _ = predict_v3_horizon(latest_df, SHORT_HORIZON)
    try:
        stacking_predictions_20d, _, _, _, _ = predict_v3_horizon(latest_df, 20)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("[/suggest_sell] T+20 cross-check unavailable (%s) — using T+%d only.",
                       exc.__class__.__name__, SHORT_HORIZON)
        stacking_predictions_20d = {}
    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}

    with timed_step("Holdings arbitrator + sentiment scoring"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, present)

    live_exec_prices = _get_live_exec_prices(latest_df, present)

    # MR knife-catch scores → drives the "don't sell into the bottom" veto.
    with timed_step("Holdings MR (knife-catch) scoring"):
        mr_scores = mr_score_tickers(present)

    LOGGER.info("[/suggest_sell] completed in %.2fs.", time.perf_counter() - total_start)
    return _build_sell_hold_report(
        holding_tickers=present,
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        stacking_predictions=horizon_predictions,
        live_exec_prices=live_exec_prices,
        missing_tickers=missing,
        mr_scores=mr_scores,
    )


# ---------------------------------------------------------------------------
# /verify — single-ticker ad-hoc analysis (rumor / news verification)
# ---------------------------------------------------------------------------

def verify_single_ticker(ticker: str, window_rows: int = 120) -> str:
    """Run ad-hoc 5d + 20d quant + LLM-sentiment verification for one ticker.

    Used by the /verify Telegram command for rumor / news fact-checks before
    a manual trade decision (e.g., "HPG announced dividends — should I buy?").

    Args:
        ticker: VN equity symbol (case-insensitive; coerced to upper).
        window_rows: per-ticker rows to load for live Alpha360 features.

    Returns:
        HTML report (HTML-escaped) suitable for `parse_mode=HTML`. Empty
        string only if the ticker name is itself empty. Liquidity / data-
        availability failures return a Vietnamese warning HTML message.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return ""

    LOGGER.info("[/verify] Single-ticker analysis: %s", ticker)
    total_start = time.perf_counter()

    # --- Step 1: Build features over the FULL active universe ---
    # CROSS-SECTIONAL PARITY: the alpha features are cross-sectional Gaussian-
    # rank Z-scores (`_xsz`).  Ranking a single ticker against itself collapses
    # every `_xsz` to 0 → garbage prediction.  So we build the WHOLE universe's
    # feature panel (exactly like daily_inference), let `_xsz` materialize over
    # the full cross-section, and slice to `ticker` only from the RESULT below.
    try:
        with timed_step(f"Building live universe OHLCV window for /verify {ticker}"):
            generator = Alpha360Generator()
            # RAW multi-row OHLCV window (full OHLCV + history) for the V3/V4
            # tabular pipeline — not the Alpha360 tail-1 lag row.
            live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
            latest_df = live_pl.to_pandas()
    except FileNotFoundError:
        LOGGER.warning("[/verify] No OHLCV parquet for %s.", ticker)
        return (
            f"⚠️ Mã <b>{html.escape(ticker)}</b> không đủ thanh khoản hoặc "
            f"không có dữ liệu để phân tích.\n"
            f"<i>(Không tìm thấy <code>data/ohlcv_{html.escape(ticker)}.parquet</code> — "
            f"có thể chưa được crawl hoặc đã hủy niêm yết.)</i>"
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] Live feature build failed for %s.", ticker)
        return (
            f"⚠️ Lỗi khi xây feature cho <b>{html.escape(ticker)}</b>: "
            f"<code>{html.escape(str(exc))}</code>"
        )

    if latest_df.empty or ticker not in set(latest_df["ticker"].astype(str)):
        return (
            f"⚠️ Mã <b>{html.escape(ticker)}</b> không đủ thanh khoản hoặc "
            f"không có dữ liệu để phân tích."
        )

    # Do NOT slice latest_df to `ticker` here — inference must see the full
    # universe so the cross-sectional `_xsz` ranks are non-degenerate.  The
    # predictions dict below is keyed by ticker, so we slice the RESULT instead.

    # --- Step 2: Stacking GBDT inference (T+3 short primary + 20d cross-check) ---
    try:
        stacking_5d, _, _, _, _ = predict_v3_horizon(latest_df, SHORT_HORIZON)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] T+%d inference failed for %s.", SHORT_HORIZON, ticker)
        return (
            f"⚠️ Lỗi mô hình Stacking GBDT cho <b>{html.escape(ticker)}</b>: "
            f"<code>{html.escape(str(exc))}</code>"
        )
    # 20d is the secondary cross-check — optional; degrade gracefully if its
    # artifact is missing/mismatched so /verify still shows the 5d view.
    try:
        stacking_20d, _, _, _, _ = predict_v3_horizon(latest_df, 20)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("[/verify] T+20 cross-check unavailable for %s (%s) — T+%d only.",
                       ticker, exc.__class__.__name__, SHORT_HORIZON)
        stacking_20d = {}

    if ticker not in stacking_5d:
        return (
            f"⚠️ Mô hình không thể dự đoán cho <b>{html.escape(ticker)}</b> "
            f"(thiếu feature hoặc lịch sử quá ngắn)."
        )

    horizon_predictions = {"5d": stacking_5d, "20d": stacking_20d}

    # --- Step 3: Arbitrator (news scrape + Gemini sentiment + decision) ---
    # `evaluate_trades_batch` accepts a candidate ticker list — passing a
    # singleton scopes the whole pipeline (GNews queries, LLM batch, decision)
    # to this one symbol. ~3-5s of news scrape + 1 Gemini call.
    try:
        with timed_step(f"Arbitrator + sentiment for /verify {ticker}"):
            final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, [ticker])
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] Arbitrator failed for %s.", ticker)
        # Fall through with empty sentiment so the user still sees the quant view.
        final_decisions, all_sentiments = {}, {}

    # --- Step 4: Live execution price (VN scaling already applied) ---
    live_exec_prices = _get_live_exec_prices(latest_df, [ticker])

    # --- Step 5: Mean-Reversion (knife-catch) state — parallel sub-model ---
    mr_state = mr_score_tickers([ticker]).get(ticker)

    LOGGER.info("[/verify] %s completed in %.2fs.", ticker, time.perf_counter() - total_start)
    return _build_verify_report(
        ticker=ticker,
        decision=final_decisions.get(ticker),
        sentiment=all_sentiments.get(ticker, {}),
        stacking_5d=list(stacking_5d.get(ticker, [0.33, 0.34, 0.33])),
        stacking_20d=list(stacking_20d.get(ticker, [0.33, 0.34, 0.33])),
        live_exec_price=live_exec_prices.get(ticker),
        mr_state=mr_state,
    )


# ---------------------------------------------------------------------------
# /rebalance — AI portfolio rebalancing advisor
# ---------------------------------------------------------------------------

def rebalance_portfolio(user_id: str, window_rows: int = 120) -> str:
    """Fetch live positions, run 5d model, scrape news, call Gemini for rebalance advice.

    Reads from the `portfolio` DuckDB table (multi-user, keyed by user_id).
    Returns an HTML report suitable for parse_mode=HTML, or "" for empty portfolio.
    """
    LOGGER.info("[/rebalance] Starting for user_id=%s", user_id)

    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        rows = db.conn.execute(
            "SELECT ticker, price FROM portfolio WHERE user_id = ?",
            [user_id],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/rebalance] DB read failed for user_id=%s", user_id)
        raise

    if not rows:
        return ""

    # Deduplicate: if the same ticker was added multiple times, use the most recent price.
    entry_prices: dict[str, float] = {}
    for ticker_raw, price_raw in rows:
        t = str(ticker_raw).upper().strip()
        entry_prices[t] = float(price_raw or 0)
    held_tickers = sorted(entry_prices)

    LOGGER.info("[/rebalance] Holdings: %s", held_tickers)

    # CROSS-SECTIONAL PARITY: build the FULL active universe so `_xsz`
    # Gaussian-rank Z-scores span the whole cross-section, not just the
    # (possibly tiny) held set.  Predictions are sliced to held names afterwards.
    with timed_step("[/rebalance] Building live universe features"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    if latest_df.empty:
        LOGGER.warning("[/rebalance] live feature frame empty for %s", held_tickers)
        return ""

    present = [t for t in held_tickers if t in set(latest_df["ticker"].astype(str))]
    if not present:
        return ""

    # Predict over the FULL universe (correct cross-section), then read off the
    # held names — do NOT filter latest_df to `present` before inference.
    stacking_predictions_5d, _, _, _, _ = predict_v3_horizon(latest_df, SHORT_HORIZON)
    live_prices = _get_live_exec_prices(latest_df, present)

    holdings_context: list[dict] = []
    for ticker in present:
        entry = entry_prices.get(ticker, 0.0)
        current = live_prices.get(ticker, entry)
        pnl_pct = ((current - entry) / entry * 100.0) if entry > 0 else 0.0
        probs = stacking_predictions_5d.get(ticker, [0.33, 0.34, 0.33])
        pred_idx = int(np.argmax(probs))
        holdings_context.append({
            "ticker": ticker,
            "pnl_pct": pnl_pct,
            "pred_label": _REBALANCE_PRED_LABELS.get(pred_idx, "N/A"),
            "p_up": float(probs[2]),
        })

    raw_news = scrape_centralized_news(target_tickers=present)
    ticker_news_dict, _ = map_tickers_to_news(raw_news, present)
    advice = get_rebalance_advice(holdings_context, ticker_news_dict)

    return _build_rebalance_report(holdings_context, advice)


def full_pipeline(force_crawl: bool = False, days_back: int | None = None) -> None:
    """End-of-day pipeline: OHLCV crawl → LLM sentiment → inference → exit alerts.

    A single `--task full_pipeline` runs the full EOD sequence so the bot is
    primed for tomorrow's session:
      1. ingest the day's HOSE OHLCV (market-hour guarded; `force_crawl` bypasses;
         `days_back` limits to the last N days — e.g. 1 for previous-day-only);
      2. refresh daily LLM news sentiment;
      3. run the daily inference (T+20 tranche Top-3 broadcast);
      4. alert tranche cohorts whose hold horizon has elapsed (signal ledger).

    Step 1 is the ONLY OHLCV ingestion entrypoint in the bot/CLI; the V4 backtest
    engine (train_models.py / run_backtest.py) reads the populated store and never
    crawls.
    """
    from src.crawlers.sentiment_crawler import update_daily_sentiment

    LOGGER.warning("Running full_pipeline (EOD): OHLCV crawl + sentiment refresh + inference.")

    # 1. EOD OHLCV ingestion (guarded by 15:00 ICT close unless --force-crawl).
    #    days_back=1 (previous-day) keeps the daily refresh incremental.
    crawl_hose(force_crawl=force_crawl, days_back=days_back)

    # 2. Daily LLM news sentiment.
    with timed_step("Fetching daily LLM sentiment"):
        sentiment_df = update_daily_sentiment(db_path="data/quant_v6_core.duckdb")
        LOGGER.info("Fetched %s sentiment records.", len(sentiment_df))

    # 3. Run the daily inference (T+20 tranche Top-3 broadcast). The V4 serve
    #    path recomputes features from raw OHLCV via load_live_ohlcv_window;
    #    there is no Alpha360 feature-matrix build (that path was retired).
    daily_inference()

    # 4. Tranche exit alerts — cohorts dispatched hold_days trading sessions
    #    ago are due for ATC liquidation per the strategy contract.
    with timed_step("Tranche exit-due check (signal ledger)"):
        notify_tranche_exits()


def notify_tranche_exits() -> int:
    """Alert (then close) every ledgered signal whose hold horizon elapsed.

    Mirrors the backtest exit rule: a cohort dispatched on day D with
    hold_days=H liquidates at the close of trading session D+H. Returns the
    number of signals alerted. Never raises — the EOD pipeline must not die
    on an alerting hiccup.
    """
    try:
        due = signal_ledger.check_exits_due()
        if not due:
            LOGGER.info("[SignalLedger] No tranche exits due today.")
            return 0

        lines = [
            f"⏰ <b>ĐẾN HẠN THOÁT VỊ THẾ (tranche)</b>\n"
            f"{datetime.now().strftime('%d/%m/%Y')} — các mã dưới đây đã đủ số phiên nắm giữ.\n"
            f"Quy tắc chiến lược: <b>thoát ATC phiên hôm nay</b>.\n"
        ]
        for d in due:
            disp = d["dispatch_date"]
            disp_str = disp.strftime("%d/%m/%Y") if hasattr(disp, "strftime") else str(disp)
            lines.append(
                f"• <b>{html.escape(str(d['ticker']))}</b> — vào {disp_str}, "
                f"đủ {int(d['hold_days'])} phiên (đã qua {int(d['sessions_elapsed'])})"
            )
        TelegramBot().send_text_alert("\n".join(lines), label="tranche_exit")
        signal_ledger.mark_closed(due)
        LOGGER.info("[SignalLedger] Exit alerts sent + closed: %s", len(due))
        return len(due)
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] notify_tranche_exits failed.")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant V6 orchestration CLI")
    parser.add_argument(
        "--task",
        default="daily_inference",
        choices=["daily_inference", "crawl_hose", "full_pipeline"],
        help="Task to run. daily_inference is the no-crawl live path; "
             "crawl_hose / full_pipeline are the EOD ingestion paths.",
    )
    parser.add_argument("--window-rows", type=int, default=120, help="Rows per ticker for live Alpha360 inference.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=6,
        help=(
            "Pool size sent to the arbitrator (LLM sentiment evaluation). "
            "The final dispatched count is always the Top-3 of this pool, "
            "ranked by sentiment_score DESC then quant probability DESC. "
            "Default 6 matches the Top-6→Top-3 design."
        ),
    )
    parser.add_argument("--force-crawl", action="store_true",
                        help="Bypass the 15:00 ICT market-hour crawl guard (operator rebuild).")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Incremental crawl: fetch only the last N calendar days "
                             "(1 = previous day). Applies to --task crawl_hose / full_pipeline. "
                             "Omit for a full crawl from the 2016 start date.")
    return parser.parse_args()


def _send_crash_alert(task_name: str, exc: BaseException, tb_text: str) -> None:
    """Best-effort Telegram crash notification. Never raises.

    Used by the top-level `main()` wrapper so a single unhandled exception
    surfaces as a Telegram message before the process exits, instead of
    silently dying in a cron log.

    Args:
        task_name: the --task value (or whatever the operator passed).
        exc: the caught exception instance.
        tb_text: a pre-formatted traceback string (already captured from
            the except block — re-running `format_exc()` after another
            exception would corrupt it).
    """
    try:
        # Telegram caps at 4096 chars per message; reserve ~500 chars for
        # the header, headline, and HTML overhead. Keep the LAST 1500 chars
        # of the traceback because the deepest frame is usually the cause.
        tb_snippet = tb_text[-1500:] if tb_text else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:300]

        # TD-31: include the running code version so we can correlate a
        # crash to a specific deploy without timestamp archaeology.
        from src.utils.version import get_version  # noqa: PLC0415
        version = get_version()

        msg = (
            "🚨 <b>[SYSTEM CRASH] Pipeline Failed!</b>\n"
            f"📅 <b>Time:</b> {html.escape(ts)}\n"
            f"🔖 <b>Version:</b> <code>{html.escape(version)}</code>\n"
            f"⚙️ <b>Task:</b> <code>{html.escape(task_name)}</code>\n"
            f"❌ <b>Exception:</b> <code>{html.escape(exc_type)}</code>\n"
            f"📝 <b>Message:</b> <code>{html.escape(exc_msg)}</code>\n\n"
            f"<b>Traceback (last frames):</b>\n"
            f"<pre>{html.escape(tb_snippet)}</pre>"
        )

        bot = TelegramBot()
        bot.send_text_alert(msg, label=f"crash:{task_name}")
        LOGGER.info("Crash alert dispatched to Telegram chat IDs.")
    except Exception as alert_exc:  # noqa: BLE001
        # The alerter itself failed (token missing, network down, etc.).
        # Log but do NOT re-raise — we want the original exception to be
        # the one that propagates.
        LOGGER.exception("Crash alert dispatch itself failed: %s", alert_exc)


def main() -> None:
    """Entry point with global crash alerting (TD-09).

    Any unhandled exception inside a task body is caught here, formatted
    into a Telegram alert, then re-raised so the cron / shell sees a
    non-zero exit code. KeyboardInterrupt and SystemExit pass through
    unalerted (those are explicit operator actions / nested sys.exit calls).
    """
    # TD-26: install rotating file handler BEFORE anything else logs.
    # Idempotent — safe even if another import already configured logging.
    from src.utils.logging_utils import setup_rotating_logging  # noqa: PLC0415
    setup_rotating_logging()

    # TD-31: capture the running code version (git SHA / VERSION file fallback)
    # for log correlation. Memoized — first call resolves, the rest are O(1).
    from src.utils.version import get_version  # noqa: PLC0415
    version = get_version()

    args = parse_args()
    task_name = args.task or "unknown"

    LOGGER.info("=" * 70)
    LOGGER.info("Quant V6 starting | task=%s pid=%s version=%s", task_name, os.getpid(), version)
    LOGGER.info("=" * 70)

    try:
        if task_name == "crawl_hose":
            crawl_hose(force_crawl=args.force_crawl, days_back=args.days_back)
        elif task_name == "full_pipeline":
            full_pipeline(force_crawl=args.force_crawl, days_back=args.days_back)
        else:
            daily_inference(window_rows=args.window_rows, max_candidates=args.max_candidates)
    except (KeyboardInterrupt, SystemExit):
        # Operator-initiated stop or a nested sys.exit(): no crash alert.
        LOGGER.warning("Pipeline interrupted (KeyboardInterrupt/SystemExit). No crash alert sent.")
        raise
    except Exception as exc:  # noqa: BLE001
        # Catch ALL pipeline exceptions (API timeout, DuckDB lock, Gemini
        # failure, file-not-found, etc.) — alert THEN re-raise so cron sees
        # exit code != 0.
        tb_text = traceback.format_exc()
        LOGGER.error("=" * 70)
        LOGGER.error("Pipeline crashed during task=%s. Dispatching alert before exit.", task_name)
        LOGGER.error(tb_text)
        LOGGER.error("=" * 70)
        _send_crash_alert(task_name, exc, tb_text)
        raise

    LOGGER.info("Quant V6 task=%s completed cleanly.", task_name)


if __name__ == "__main__":
    main()