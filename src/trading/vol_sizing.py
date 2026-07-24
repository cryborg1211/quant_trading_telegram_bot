"""Volatility-scaled burst-budget divisor (22-07-26 burst-sizing follow-up).

Burst sizing (`WalkForwardConfig.tranche_budget_days`) proved a FIXED nav/10
divisor recovers ~3x the concentration A/B's PnL for a flat DD cost (see
`scripts/analyze_burst_sizing_ab.py`). This module makes the divisor
DYNAMIC: bigger clips (smaller divisor) when trailing realized market vol is
calm, smaller clips (bigger divisor) when it's elevated — the same "spend
headroom when it's cheap, pull back when it's not" shape every other
exposure leg in `garch_brake.py` already uses, applied to SIZE instead of
participation.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

_ANNUALIZATION = 252.0 ** 0.5


def market_vol_time_series(
    panel: pl.DataFrame,
    window: int = 20,
    min_tickers: int = 30,
) -> pd.Series:
    """Market-wide trailing realized vol for EVERY date in `panel`.

    Cross-sectional MEDIAN of each ticker's trailing `window`-session
    annualized return-vol. `panel` needs columns [ticker, date, close] (the
    standard OHLCV panel shape). Vectorized single polars pass, mirrors
    `src.trading.breadth.breadth_time_series`'s exact shape/reliability-floor
    convention. Returns a date-indexed pandas Series — the same contract as
    `breadth_series`/`p_bull_series`, for a drop-in
    `WalkForwardEngine.run(budget_days_series=...)` argument.
    """
    if panel is None or panel.is_empty() or window <= 0:
        return pd.Series(dtype=float)
    lf = panel.lazy().sort(["ticker", "date"])
    lf = lf.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_ret1")
    )
    lf = lf.with_columns(
        (
            pl.col("_ret1").rolling_std(window_size=window, min_periods=window).over("ticker")
            * _ANNUALIZATION
        ).alias("_vol_ann")
    )
    per_date = (
        lf.group_by("date")
        .agg([
            pl.col("_vol_ann").median().alias("_vol_med"),
            pl.col("_vol_ann").is_not_null().sum().alias("_valid"),
        ])
        .sort("date")
        .collect()
    )
    per_date = per_date.filter(pl.col("_valid") >= min_tickers)
    result = per_date.select(["date", pl.col("_vol_med").alias("vol")]).to_pandas()
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")["vol"]


def vol_scaled_budget_days(
    trailing_vol: float,
    *,
    base_divisor: int = 10,
    low_vol_mult: float = 0.6,
    high_vol_mult: float = 2.0,
    low_vol_trigger: float = 0.15,
    high_vol_trigger: float = 0.35,
) -> int:
    """Map trailing annualized market vol -> an integer budget divisor.

    ``trailing_vol <= low_vol_trigger`` -> ``base_divisor * low_vol_mult``
    (bigger clips, calm market). ``>= high_vol_trigger`` ->
    ``base_divisor * high_vol_mult`` (smaller clips, stressed market).
    Linear ramp in between. Non-finite vol or degenerate knob ordering fails
    open to `base_divisor` (unscaled — matches every other leg's fail-open
    discipline in this codebase).
    """
    if not pd.notna(trailing_vol) or high_vol_trigger <= low_vol_trigger:
        return base_divisor
    if trailing_vol <= low_vol_trigger:
        mult = low_vol_mult
    elif trailing_vol >= high_vol_trigger:
        mult = high_vol_mult
    else:
        frac = (trailing_vol - low_vol_trigger) / (high_vol_trigger - low_vol_trigger)
        mult = low_vol_mult + frac * (high_vol_mult - low_vol_mult)
    return max(1, round(base_divisor * mult))
