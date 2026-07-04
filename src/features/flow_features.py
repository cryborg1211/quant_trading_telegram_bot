"""Foreign/Proprietary flow feature engineering — pure Polars, PROTOTYPE.

Companion to `src/data/foreign_flow_crawler.py`. NOT wired into
`src/backtest/pipeline.build_features` yet — this is a standalone module for
walk-forward backtesting before any recipe-version bump / retrain decision
(see FEATURE_RECIPE_VERSION in pipeline.py for the gate this must pass
through once validated).

LOOK-AHEAD SAFETY (mirrors mr_features.py's audited pattern)
──────────────────────────────────────────────────────────────
A feature at bar t may use any flow/price data dated <= t. Every rolling
window here ends at t (`.rolling_sum`/`.rolling_mean` with no forward
shift), and all window ops are `.over("ticker")` so they never bleed across
symbols. `foreign_net_val` for date t is the flow REPORTED for t's session
(exchange EOD data) — it becomes knowable only after t's close, same timing
as the OHLCV close it is joined against. Do not consume this as a same-day
feature for a decision made intraday on day t; it is a T's-close feature,
consumed starting T+1 same as every other bar-t feature in this pipeline.

INPUT CONTRACT
───────────────
`build_flow_features(df)` expects a polars DataFrame with columns:
    ticker, date, close, volume,                # from the OHLCV join
    foreign_buy_val, foreign_sell_val, foreign_net_val,
    prop_buy_val, prop_sell_val, prop_net_val    # may be all-null if the
                                                  # active crawler adapter
                                                  # doesn't source tu doanh
Returns a COPY with `FLOW_FEATURE_COLUMNS` appended. Missing prop_* columns
degrade their derived features to null, not an error — a caller with
foreign-room-only data should still get the foreign-side features.
"""
from __future__ import annotations

import polars as pl

from src.utils.schema_hash import compute_feature_schema_hash

FLOW_FEATURE_COLUMNS: list[str] = [
    "flow_net_scaled_adv20",
    "flow_cum_net_3d",
    "flow_cum_net_5d",
    "flow_momentum_3d",
    "flow_momentum_5d",
    "flow_prop_net_scaled_adv20",
    "flow_knife_catch_divergence",
]

_FLOW_SCHEMA: list[tuple[str, str]] = [(col, "float64") for col in FLOW_FEATURE_COLUMNS]
FLOW_SCHEMA_HASH: str = compute_feature_schema_hash(_FLOW_SCHEMA, None)

_REQUIRED = ("ticker", "date", "close", "volume", "foreign_net_val")

# Divergence signal thresholds — starting points, MUST be tuned against the
# Phase-2 lead-lag correlation study before trusting them; these are not
# calibrated to this market yet.
_DIVERGENCE_PRICE_DROP_PCT = -0.01     # price down >= 1% on the day
_DIVERGENCE_FLOW_ZSCORE = 1.5          # net flow >= 1.5 rolling-sigma spike


def build_flow_features(df: pl.DataFrame) -> pl.DataFrame:
    """Pure-expression Polars feature build. No Python-level loops.

    Sort contract: caller need not pre-sort — this sorts by (ticker, date)
    internally before any `.over("ticker")` window op, since Polars window
    functions over an unsorted frame silently produce wrong rolling values.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"build_flow_features: missing required columns {missing}")

    out = df.sort(["ticker", "date"])

    has_prop = "prop_net_val" in out.columns

    out = out.with_columns(
        # 20-day average dollar volume, per ticker. Using close*volume as the
        # dollar-value proxy keeps this comparable across price levels
        # without needing a separate turnover feed.
        (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=20, min_samples=5)
            .over("ticker")
            .alias("_adv20_val"),
    )

    out = out.with_columns(
        # Net flow scaled by ADV20 — dimensionless, comparable across
        # large-cap vs small-cap tickers (a 10B VND net-buy means very
        # different things for VCB vs a thin small-cap).
        pl.when(pl.col("_adv20_val") > 0)
          .then(pl.col("foreign_net_val") / pl.col("_adv20_val"))
          .otherwise(None)
          .alias("flow_net_scaled_adv20"),

        # Cumulative flow momentum — rolling SUM (not mean) so the magnitude
        # reflects sustained multi-day accumulation, not just an average day.
        pl.col("foreign_net_val")
            .rolling_sum(window_size=3, min_samples=2)
            .over("ticker")
            .alias("flow_cum_net_3d"),
        pl.col("foreign_net_val")
            .rolling_sum(window_size=5, min_samples=3)
            .over("ticker")
            .alias("flow_cum_net_5d"),
    )

    # `has_prop` is a static, call-time fact about the input schema (not a
    # per-row condition) — branch on it in Python, not inside a `pl.when`
    # predicate. Mixing a Python bool into `pl.when(bool and Expr)` would
    # short-circuit to a bare `False` whenever `has_prop` is False instead
    # of producing a valid all-null expression.
    if has_prop:
        prop_expr = (
            pl.when(pl.col("_adv20_val") > 0)
              .then(pl.col("prop_net_val") / pl.col("_adv20_val"))
              .otherwise(None)
              .alias("flow_prop_net_scaled_adv20")
        )
    else:
        prop_expr = pl.lit(None, dtype=pl.Float64).alias("flow_prop_net_scaled_adv20")

    out = out.with_columns(
        # Momentum = today's flow vs. its own recent rolling mean (self-
        # relative, per ticker) — not a level, a "flow accelerating" signal.
        (pl.col("foreign_net_val") - pl.col("foreign_net_val").rolling_mean(3, min_samples=2).over("ticker"))
            .alias("flow_momentum_3d"),
        (pl.col("foreign_net_val") - pl.col("foreign_net_val").rolling_mean(5, min_samples=3).over("ticker"))
            .alias("flow_momentum_5d"),
        prop_expr,
    )

    out = out.with_columns(
        pl.col("close").pct_change().over("ticker").alias("_ret_1d"),
        pl.col("foreign_net_val").rolling_std(20, min_samples=5).over("ticker").alias("_flow_std20"),
        pl.col("foreign_net_val").rolling_mean(20, min_samples=5).over("ticker").alias("_flow_mean20"),
    )

    out = out.with_columns(
        pl.when(pl.col("_flow_std20") > 0)
          .then((pl.col("foreign_net_val") - pl.col("_flow_mean20")) / pl.col("_flow_std20"))
          .otherwise(None)
          .alias("_flow_zscore"),
    )

    out = out.with_columns(
        # Knife-catch divergence: price fell hard on the day but smart money
        # (foreign flow) net-bought at an unusual (z-scored) magnitude —
        # binary flag, 1.0/0.0/null (null when insufficient history for the
        # z-score, e.g. first 5 rows per ticker).
        pl.when(pl.col("_flow_zscore").is_null())
          .then(None)
          .when(
              (pl.col("_ret_1d") <= _DIVERGENCE_PRICE_DROP_PCT)
              & (pl.col("_flow_zscore") >= _DIVERGENCE_FLOW_ZSCORE)
          )
          .then(1.0)
          .otherwise(0.0)
          .alias("flow_knife_catch_divergence"),
    )

    return out.drop(["_adv20_val", "_ret_1d", "_flow_std20", "_flow_mean20", "_flow_zscore"])


__all__ = ["build_flow_features", "FLOW_FEATURE_COLUMNS", "FLOW_SCHEMA_HASH"]
