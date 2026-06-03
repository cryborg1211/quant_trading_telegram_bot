"""
src/data/tensor_builder.py — Quant Engine V2.0, Phase 1

Institutional feature-engineering pipeline for the V3 tabular ensemble.

╔══════════════════════════════════════════════════════════════════════════════╗
║  V2.0 ROADMAP NOTE  (§13.4 audit)                                           ║
║  This module delivers the data-engineering layer.  Per the Principal-Quant  ║
║  audit, LSTM deployment is *gated* on §13.1 (CPCV / DSR / PBO) and §13.3  ║
║  (portfolio construction) being in place first. `tensor_builder` is the      ║
║  correct Phase 1 foundation; do NOT route live capital through it until      ║
║  the statistical-rigor and risk layers are green.                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Recommended pipeline order
──────────────────────────
    1. `apply_frac_diff(df, price_cols)`
          Per-ticker stationary transformation that preserves long memory.

    2. `add_cross_sectional_features(df, all_feature_cols)`
          Daily Gaussian rank Z-scores across the full ticker universe.
          Run AFTER FracDiff so cross-section operates on stationary inputs.

    3. `add_alpha_factors(df)` + `add_advanced_statistical_features(df)`
          Append the V3 tabular feature pool (alpha + statistical factors).

References
──────────
    FracDiff:        de Prado, AFML (2018), Chapter 5
    Cross-sectional: Grinold & Kahn, Active Portfolio Management (1999)
    Gaussian rank:   Blom (1958); standard in equity factor preprocessing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
from scipy.stats import norm as scipy_norm

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FracDiffConfig",
    "apply_frac_diff",
    "add_cross_sectional_features",
    "add_overextension_features",
    "add_alpha_factors",
    "add_advanced_statistical_features",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FracDiffConfig:
    """Hyper-parameters for Fixed-Width Window Fractional Differentiation."""
    d: float = 0.4
    """
    Differencing order in (0, 1).
    Tune per-feature to the minimum d at which an ADF test rejects the
    unit-root null.  Typical range for VN equity prices:
    0.30–0.45.  Higher d → more stationary, less memory preserved.
    """
    tau: float = 1e-4
    """
    Weight truncation threshold.  Kernel weights with |π_k| < tau are dropped.
    Controls the trade-off between approximation accuracy and kernel width W:
    smaller tau → wider kernel → more history used → higher compute cost.
    """
    max_window: int = 200
    """Hard cap on kernel width W to avoid O(T·W) blowing up for near-unit-root series."""




# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _ffd_weights(d: float, tau: float, max_window: int) -> np.ndarray:
    """
    Compute Fixed-Width Window FracDiff kernel weights in oldest-to-newest order.

    The fractional binomial expansion of (1 − B)^d yields weights:

        π₀ = 1
        πₖ = −πₖ₋₁ · (d − k + 1) / k,    k = 1, 2, …

    For d ∈ (0, 1) the weights are alternating in sign and decay toward zero,
    preserving a geometrically-fading influence of distant observations.

    Returns:
        1-D float64 array of length W (oldest weight first), where W is
        determined by the truncation criterion |πₖ| < tau or k ≥ max_window.
        The dot product `weights @ series_window` = FracDiff value at time t.
    """
    w: list[float] = [1.0]
    k = 1
    while True:
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < tau or k >= max_window:
            break
        w.append(next_w)
        k += 1
    # Reverse: index 0 = oldest weight π_{W-1}, index -1 = π₀ = 1.0
    return np.array(w[::-1], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FRACTIONAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_frac_diff(
    df: pl.DataFrame,
    cols: Sequence[str],
    cfg: FracDiffConfig | None = None,
    ticker_col: str = "ticker",
    date_col: str = "date",
    suffix: str = "_fd",
) -> pl.DataFrame:
    """
    Apply Fixed-Width Window Fractional Differentiation (AFML §5) to panel data.

    Why FracDiff instead of rolling-Z or log-returns?
    ───────────────────────────────────────────────────
    Rolling-Z (current Alpha360 normalisation) removes scale bias but preserves
    the *level* process.  A price of 100 vs. 50 between training and inference
    creates covariate shift that tree splits absorb implicitly but that will
    fool an LSTM's hidden state.

    Integer differencing (log-returns, d = 1) achieves stationarity but
    collapses ALL long-range autocorrelation to zero.  The LSTM would
    effectively see only yesterday's return — the core advantage of recurrent
    memory is wasted.

    Fractional differencing at d ∈ (0, 1) solves both problems simultaneously:

        (1 − B)^d Pₜ = Σ_{k=0}^{W} πₖ Pₜ₋ₖ

    Weights πₖ decay slowly enough that the distant past still contributes
    (preserving autocorrelation and trend structure), while applying just enough
    differencing to remove the unit root (stationarity condition).

    Implementation: FFD (Fixed-Width Window FracDiff)
    ─────────────────────────────────────────────────
    The infinite sum is truncated at weight threshold `tau`, giving a kernel of
    width W.  Cost is O(T · W · F) where F = number of columns.  Vectorised
    over all F columns simultaneously per ticker via:

        FD[t] = windows[t] @ weights      (broadcasting over the feature axis)

    where `windows` is a stride-tricks view of shape (T − W + 1, W, F).

    Leading (W − 1) rows per ticker that cannot be computed are set to NaN.
    Downstream tabular alignment drops these rows before model fitting.

    Args:
        df:         Panel DataFrame — will be sorted (ticker, date) internally.
        cols:       Columns to fractionally difference (e.g. log-price, log-volume).
        cfg:        FracDiffConfig; defaults used if None.
        ticker_col: Partition key (one independent kernel per ticker).
        date_col:   Chronological sort key within each partition.
        suffix:     New column names are `{col}{suffix}`.

    Returns:
        Input DataFrame with `len(cols)` new float32 columns appended.

    Raises:
        ValueError: If any column in `cols` is not found in `df`.
    """
    cfg = cfg or FracDiffConfig()
    weights = _ffd_weights(cfg.d, cfg.tau, cfg.max_window)
    W = len(weights)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"apply_frac_diff: columns not in DataFrame: {missing}")

    LOGGER.info(
        "FracDiff | d=%.3f  kernel_width=%d  tau=%.1e  cols=%s",
        cfg.d, W, cfg.tau, list(cols),
    )

    df = df.sort([ticker_col, date_col])
    n_cols = len(cols)
    n_rows = len(df)

    # Extract all target columns as a contiguous (n_rows, n_cols) float64 matrix.
    feat_matrix: np.ndarray = (
        df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    )
    ticker_arr: np.ndarray = df[ticker_col].to_numpy()

    result = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for ticker in sorted(set(ticker_arr.tolist())):
        positions: np.ndarray = np.where(ticker_arr == ticker)[0]
        T = len(positions)
        n_valid = T - W + 1
        if n_valid <= 0:
            continue

        # Contiguous copy required by sliding_window_view.
        S: np.ndarray = np.ascontiguousarray(feat_matrix[positions])  # (T, n_cols)

        # Vectorised FracDiff over all n_cols simultaneously.
        #
        # sliding_window_view on (T, n_cols) with shape (W, n_cols):
        #   output → (T-W+1, 1, W, n_cols)  [axis-1 collapses because window
        #             spans the full feature width]
        # squeeze(1) → (T-W+1, W, n_cols)
        #
        # FD[i, c] = Σ_k weights[k] * S[i+k, c]
        #          = (windows[i, :, c] · weights)
        # Vectorised: (n_valid, W, n_cols) * (1, W, 1) → sum over axis-1 →
        #             (n_valid, n_cols)
        windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(
            S, window_shape=(W, n_cols)
        ).squeeze(1)  # (n_valid, W, n_cols)

        fd_block: np.ndarray = (windows * weights[None, :, None]).sum(axis=1)
        result[positions[W - 1 :]] = fd_block  # first W-1 rows stay NaN

    new_col_names = [f"{c}{suffix}" for c in cols]
    df = df.with_columns([
        pl.Series(name, result[:, i].astype(np.float32))
        for i, name in enumerate(new_col_names)
    ])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROSS-SECTIONAL RANK Z-SCORING
# ─────────────────────────────────────────────────────────────────────────────

def add_cross_sectional_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_tickers: int = 10,
    clip_z: float = 4.0,
    suffix: str = "_xsz",
) -> pl.DataFrame:
    """
    Add cross-sectional Gaussian rank Z-scores for each feature column.

    The Gaussian rank transform (Blom, 1958)
    ──────────────────────────────────────────
    For each (date t, feature f), across all N_t tickers active on date t:

        rank_{i,t}    = rank of ticker i among all tickers  (average-ties, 1-indexed)
        uniform_{i,t} = (rank_{i,t} − 0.5) / N_t           ∈ (0, 1)  [van der Waerden]
        z_{i,t}       = Φ⁻¹(uniform_{i,t})                 ∈ ℝ        [probit / inverse-normal]

    where Φ⁻¹ is the probit function (scipy.stats.norm.ppf).

    Why Gaussian rank over simple (x − mean) / std?
    ─────────────────────────────────────────────────
    • Outlier robustness: one extreme ticker cannot shift all other z-scores.
      Ranking is not affected by the magnitude of outliers, only their order.
    • Exact Normal(0,1) marginals by construction regardless of the raw
      distribution — LSTM inputs benefit from well-conditioned activations.
    • Isolates *relative* cross-sectional standing; absolute levels (which
      differ across tickers and over time) are discarded.  This is the
      standard step in institutional equity factor preprocessing (Barra, APT).
    • Combined with FracDiff (which handles the time-series dimension), these
      two transforms address both axes of the non-stationarity problem.

    Dates with fewer than `min_tickers` valid observations receive 0.0 (the
    distribution median) rather than potentially unreliable z-scores from thin
    cross-sections.  The probit output is hard-clipped to ±clip_z to guard
    against numerical divergence at tied ranks.

    Implementation note:
        Rank and count are computed entirely inside Polars (.rank().over(date),
        .count().over(date)) — fully vectorised C++ operations on the full panel.
        The probit is then applied in a single numpy call per feature column,
        avoiding any Python-level row iteration.

    Args:
        df:           Panel DataFrame with (date, ticker, feature…) columns.
        feature_cols: Columns to transform; originals are preserved unchanged.
        date_col:     Cross-sectional grouping key.
        ticker_col:   Row identifier within each cross-section.
        min_tickers:  Minimum non-null ticker count to produce a z-score; else 0.0.
        clip_z:       Symmetric hard clip applied after probit.
        suffix:       New column name = `{col}{suffix}`.

    Returns:
        DataFrame with `len(feature_cols)` new float32 columns appended.

    Raises:
        ValueError: If any element of `feature_cols` is absent from `df`.
    """
    if not feature_cols:
        return df

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"add_cross_sectional_features: columns not found: {missing}")

    LOGGER.info(
        "Cross-sectional Z-score | %d features | %d dates | %d tickers",
        len(feature_cols),
        df[date_col].n_unique(),
        df[ticker_col].n_unique(),
    )

    # ── Step 1: Uniform quantile via Polars rank (pure C++, no Python loop) ──
    # For each feature col:
    #   - rank across tickers within each date group (average-ties, ascending)
    #   - n = non-null count on that date
    #   - uniform = (rank - 0.5) / n  ∈ (0, 1)
    #   - if n < min_tickers: None  (→ imputed to 0.0 in step 2)
    uniform_exprs: list[pl.Expr] = []
    for col in feature_cols:
        rank_expr = (
            pl.col(col)
            .rank(method="average", descending=False)
            .over(date_col)
        )
        n_expr = (
            pl.col(col)
            .count()           # count() excludes nulls — correct for rank denominator
            .over(date_col)
            .cast(pl.Float64)
        )
        uniform_exprs.append(
            pl.when(n_expr < min_tickers)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((rank_expr - 0.5) / n_expr)
            .alias(f"__u_{col}")
        )

    df = df.with_columns(uniform_exprs)

    # ── Step 2: Probit (numpy bulk operation — one scipy call per feature) ──
    probit_series: list[pl.Series] = []
    for col in feature_cols:
        u_col = f"__u_{col}"
        uniform_arr: np.ndarray = df[u_col].to_numpy(allow_copy=True).astype(np.float64)

        null_mask: np.ndarray = np.isnan(uniform_arr)

        # Clip interior of (0, 1) so ppf never diverges at 0 or 1.
        uniform_arr = np.clip(uniform_arr, 1e-7, 1.0 - 1e-7)
        z_arr: np.ndarray = scipy_norm.ppf(uniform_arr).astype(np.float32)
        z_arr = np.clip(z_arr, -clip_z, clip_z)
        z_arr[null_mask] = 0.0   # thin cross-section dates → cross-sectional median

        probit_series.append(pl.Series(f"{col}{suffix}", z_arr, dtype=pl.Float32))

    df = df.with_columns(probit_series)
    df = df.drop([f"__u_{col}" for col in feature_cols])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2b. OVER-EXTENSION (ANTI-FOMO) FEATURES — Phase 1.5
# ─────────────────────────────────────────────────────────────────────────────

def add_overextension_features(
    df: pl.DataFrame,
    ma_windows: Sequence[int] = (5, 20),
    *,
    close_col: str = "close",
    ticker_col: str = "ticker",
    date_col: str = "date",
    cross_sectional: bool = True,
    min_tickers: int = 10,
    clip_z: float = 4.0,
) -> pl.DataFrame:
    """
    Anti-FOMO over-extension features: signed distance of close from its own
    short-term moving averages, optionally cross-sectionally Gaussian-ranked.

    The structural flaw this fixes
    ───────────────────────────────
    Cross-sectional momentum z-scores reward whatever moved up most today.  An
    LSTM trained only on those will happily BUY a stock printing a +7% HOSE
    ceiling (Trần) — exactly the over-extended blow-off that mean-reverts
    tomorrow.  The model has no notion of "statistically exhausted".

    The feature
    ───────────
    For each window w ∈ `ma_windows`:

        ma_w       = rolling_mean(close, w)        per ticker, causal
        overext_w  = close / ma_w − 1              signed % distance from the MA

    Reading:
        overext_5 ≫ 0   → price far above its 5-day mean → short-term blow-off
                          (the ceiling pump the LSTM must learn to FADE, not chase)
        overext_5 ≪ 0   → washed-out / capitulation (the knife-catch setup)

    When `cross_sectional=True`, each `overext_w` is additionally Gaussian-rank
    Z-scored across the universe per date (reusing `add_cross_sectional_features`)
    to produce `overext_w_xsz`.  The cross-sectional rank is what lets the
    attention head learn "this name is the MOST exhausted in the market today",
    a relative judgment, instead of brittle absolute thresholds.

    Leak-safety
    ───────────
    `ma_w` spans close[t−w+1 .. t] INCLUSIVE of the decision bar t.  This is a
    state feature observed at end-of-day t, not a forward label — no look-ahead.
    (Confirm: the over-extension is computed only from data up to bar t, so it
    is part of the end-of-day observation t, never a forward label.)

    Args:
        df:              Panel DataFrame with (ticker, date, close).
        ma_windows:      MA lookbacks in bars; default (5, 20).
        cross_sectional: also emit Gaussian-rank `_xsz` columns.
        min_tickers,
        clip_z:          forwarded to `add_cross_sectional_features`.

    Returns:
        df + raw `overext_{w}` columns (+ `overext_{w}_xsz` if cross_sectional).
    """
    if close_col not in df.columns:
        raise ValueError(f"add_overextension_features: '{close_col}' not in DataFrame")
    if not ma_windows:
        return df

    df = df.sort([ticker_col, date_col])

    raw_cols: list[str] = []
    exprs: list[pl.Expr] = []
    for w in ma_windows:
        if w < 1:
            raise ValueError(f"ma_window must be >= 1, got {w}")
        ma = pl.col(close_col).rolling_mean(window_size=w).over(ticker_col)
        name = f"overext_{w}"
        # (close / ma) − 1 ; guard the warm-up region where ma is null → null.
        exprs.append((pl.col(close_col) / ma - 1.0).alias(name))
        raw_cols.append(name)
    df = df.with_columns(exprs)

    LOGGER.info(
        "Over-extension features | windows=%s | cross_sectional=%s",
        list(ma_windows), cross_sectional,
    )

    if cross_sectional:
        df = add_cross_sectional_features(
            df, raw_cols,
            date_col=date_col, ticker_col=ticker_col,
            min_tickers=min_tickers, clip_z=clip_z, suffix="_xsz",
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2d. ALPHA FACTORS (GROUP A — cross-sectionally Gaussian-rank Z-scored)
# ─────────────────────────────────────────────────────────────────────────────

def add_alpha_factors(
    df: pl.DataFrame,
    *,
    close_col: str = "close",
    volume_col: str = "volume",
    ticker_col: str = "ticker",
    date_col: str = "date",
    rs_windows: Sequence[int] = (10, 20),
    money_flow_window: int = 20,
    vol_short: int = 5,
    vol_long: int = 20,
    cross_sectional: bool = True,
    min_tickers: int = 10,
    clip_z: float = 4.0,
) -> pl.DataFrame:
    """
    GROUP A alpha factors — each cross-sectionally Gaussian-rank Z-scored (`_xsz`).

    1. Relative Strength (rs_{w}): the w-bar return `close/close.shift(w) − 1`.
       The cross-sectional Gaussian-rank Z-score IS the "relative to market"
       transform — it is rank-based, hence shift-invariant, so explicitly
       subtracting the daily cross-sectional mean first would be redundant
       (rank(x − c) = rank(x)). Output: `rs_{w}_xsz`.

    2. Smart Money Flow (smart_money_{W}): rolling-W sum of (return × volume)
       normalised by the rolling-W sum of volume — an accumulation/distribution
       proxy (up-days on heavy volume push it positive). Output: `smart_money_{W}_xsz`.

    3. Volatility Squeeze (vol_squeeze): short-horizon return std ÷ long-horizon
       return std. < 1 ⇒ compression (a squeeze); > 1 ⇒ expansion. Output:
       `vol_squeeze_xsz`.

    All raw factors are per-ticker time-series; the `_xsz` step ranks them across
    the universe each day. Leak-safe: every input uses only data ≤ t.
    """
    for c in (close_col, volume_col):
        if c not in df.columns:
            raise ValueError(f"add_alpha_factors: '{c}' not in DataFrame")

    df = df.sort([ticker_col, date_col])

    # Materialise the 1-bar return ONCE (avoids nested `.over` inside rolling ops).
    df = df.with_columns(
        (pl.col(close_col) / pl.col(close_col).shift(1).over(ticker_col) - 1.0).alias("_af_ret")
    )

    raw_cols: list[str] = []
    exprs: list[pl.Expr] = []

    # 1. Relative Strength — w-bar returns.
    for w in rs_windows:
        name = f"rs_{w}"
        exprs.append(
            (pl.col(close_col) / pl.col(close_col).shift(w).over(ticker_col) - 1.0).alias(name)
        )
        raw_cols.append(name)

    # 2. Smart Money Flow — Σ(ret·vol) / Σ(vol) over the window.
    sm_name = f"smart_money_{money_flow_window}"
    exprs.append(
        ((pl.col("_af_ret") * pl.col(volume_col)).rolling_sum(money_flow_window).over(ticker_col)
         / (pl.col(volume_col).rolling_sum(money_flow_window).over(ticker_col) + 1e-12)
         ).alias(sm_name)
    )
    raw_cols.append(sm_name)

    # 3. Volatility Squeeze — short std ÷ long std of returns.
    exprs.append(
        (pl.col("_af_ret").rolling_std(vol_short).over(ticker_col)
         / (pl.col("_af_ret").rolling_std(vol_long).over(ticker_col) + 1e-12)
         ).alias("vol_squeeze")
    )
    raw_cols.append("vol_squeeze")

    df = df.with_columns(exprs).drop("_af_ret")

    LOGGER.info("Alpha factors | raw=%s  cross_sectional=%s", raw_cols, cross_sectional)
    if cross_sectional:
        df = add_cross_sectional_features(
            df, raw_cols, date_col=date_col, ticker_col=ticker_col,
            min_tickers=min_tickers, clip_z=clip_z, suffix="_xsz",
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Advanced statistical features (V3.1 candidate pool for iron-fist selection)
# ─────────────────────────────────────────────────────────────────────────────
def add_advanced_statistical_features(
    df: pl.DataFrame,
    *,
    window_amihud: int = 20,
    window_skew: int = 20,
    window_vov_short: int = 5,
    window_vov_long: int = 20,
    cross_sectional: bool = True,
    clip_z: float = 4.0,
) -> pl.DataFrame:
    """Five candidate statistical features intended to give the GBM stack new,
    uncorrelated perspectives on price-action microstructure.  All operations
    are STRICTLY per-ticker (`.over("ticker")`) and leak-safe (no peek beyond
    the decision bar).

    Raw outputs (before cross-sectional ranking)
    ────────────────────────────────────────────
        amihud_liquidity         rolling-`window_amihud`-day mean of
                                 |daily return| / (close × volume).
                                 Amihud (2002) illiquidity proxy.  Higher =
                                 more price impact per unit dollar volume.
        realized_skewness_20d    rolling-`window_skew`-day skewness of daily
                                 LOG returns.  Captures left-tail crash risk.
        vol_of_vol_20d           rolling-`window_vov_long`-day std of the
                                 rolling-`window_vov_short`-day daily-return
                                 vol.  Picks up regime-shifting names.
        hl_range_ratio           (high − low) / close.  Daily intraday range
                                 normalised by price level (no window).
        gap_risk                 log(open_t / close_{t-1}).  Overnight gap
                                 risk indicator (no window).

    When `cross_sectional=True` each raw feature is also Gaussian-rank
    Z-scored cross-sectionally per date via `add_cross_sectional_features`,
    producing the suffixed `_xsz` columns that the V3 GBM stack consumes.

    Numerical safety
    ────────────────
    Division denominators (close × volume, close) are floored at 1e-9 to
    avoid /0 on halted/zero-volume bars.  The downstream `drop_nulls` /
    rank operations handle leading-NaN warm-ups (window_size bars).
    """
    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_advanced_statistical_features: missing columns: {missing}")

    df = df.sort(["ticker", "date"])

    # ── Materialise once: daily return, log return, dollar volume ──────────
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_adv_ret"),
        (pl.col("close").log() - pl.col("close").shift(1).over("ticker").log())
            .alias("_adv_logret"),
        (pl.col("close") * pl.col("volume")).clip(lower_bound=1e-9).alias("_adv_dvol"),
    ])

    # 1) Amihud illiquidity (rolling mean of |r| / dollar_volume)
    df = df.with_columns([
        (pl.col("_adv_ret").abs() / pl.col("_adv_dvol"))
            .rolling_mean(window_size=window_amihud, min_samples=window_amihud)
            .over("ticker")
            .alias("amihud_liquidity"),
    ])

    # 2) Realized skewness 20d of daily log returns
    df = df.with_columns([
        pl.col("_adv_logret")
            .rolling_skew(window_size=window_skew, min_samples=window_skew)
            .over("ticker")
            .alias("realized_skewness_20d"),
    ])

    # 3) Vol-of-vol: rolling-long std of rolling-short return vol
    df = df.with_columns([
        pl.col("_adv_ret")
            .rolling_std(window_size=window_vov_short, min_samples=window_vov_short)
            .over("ticker")
            .alias("_adv_vol5"),
    ])
    df = df.with_columns([
        pl.col("_adv_vol5")
            .rolling_std(window_size=window_vov_long, min_samples=window_vov_long)
            .over("ticker")
            .alias("vol_of_vol_20d"),
    ])

    # 4) (High − Low) / Close — daily intraday range (no window)
    df = df.with_columns([
        ((pl.col("high") - pl.col("low")) / pl.col("close").clip(lower_bound=1e-9))
            .alias("hl_range_ratio"),
    ])

    # 5) Gap risk — log(open_t / close_{t-1})
    df = df.with_columns([
        (pl.col("open").log() - pl.col("close").shift(1).over("ticker").log())
            .alias("gap_risk"),
    ])

    # Drop temp columns — keep namespace clean for the XS rank step
    df = df.drop(["_adv_ret", "_adv_logret", "_adv_dvol", "_adv_vol5"])

    raw_cols = [
        "amihud_liquidity",
        "realized_skewness_20d",
        "vol_of_vol_20d",
        "hl_range_ratio",
        "gap_risk",
    ]
    LOGGER.info("Advanced statistical features | raw=%s  cross_sectional=%s",
                raw_cols, cross_sectional)

    if cross_sectional:
        df = add_cross_sectional_features(df, raw_cols, suffix="_xsz", clip_z=clip_z)

    return df
