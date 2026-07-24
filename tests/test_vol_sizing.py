"""Vol-scaled burst-budget divisor (22-07-26 burst-sizing follow-up).

market_vol_time_series: cross-sectional median trailing realized vol per date.
vol_scaled_budget_days: piecewise-linear vol -> divisor mapping, same shape
as breadth_scalar / garch_brake.drift_scalar_from_returns.
"""
from __future__ import annotations

import polars as pl

from src.trading.vol_sizing import market_vol_time_series, vol_scaled_budget_days


def _panel(tickers_closes: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for ticker, closes in tickers_closes.items():
        for i, c in enumerate(closes):
            rows.append({"ticker": ticker, "date": f"2026-01-{i + 1:02d}", "close": c})
    return pl.DataFrame(rows)


def _alternating(n: int, start: float, amp: float) -> list[float]:
    px = [start]
    for i in range(n - 1):
        px.append(px[-1] * (1.0 + (amp if i % 2 == 0 else -amp)))
    return px


# ---------------------------------------------------------------------------
# market_vol_time_series
# ---------------------------------------------------------------------------


def test_empty_panel_returns_empty_series() -> None:
    assert market_vol_time_series(pl.DataFrame(), window=5, min_tickers=1).empty


def test_below_min_tickers_dropped() -> None:
    data = {f"T{i}": _alternating(15, 100.0, 0.02) for i in range(5)}
    s = market_vol_time_series(_panel(data), window=5, min_tickers=30)
    assert s.empty


def test_higher_amplitude_gives_higher_median_vol() -> None:
    calm = {f"C{i}": _alternating(30, 100.0, 0.002) for i in range(10)}
    wild = {f"W{i}": _alternating(30, 100.0, 0.05) for i in range(10)}
    calm_s = market_vol_time_series(_panel(calm), window=10, min_tickers=10)
    wild_s = market_vol_time_series(_panel(wild), window=10, min_tickers=10)
    assert not calm_s.empty and not wild_s.empty
    assert wild_s.iloc[-1] > calm_s.iloc[-1] * 3


# ---------------------------------------------------------------------------
# vol_scaled_budget_days
# ---------------------------------------------------------------------------

_KW = dict(base_divisor=10, low_vol_trigger=0.15, high_vol_trigger=0.35,
           low_vol_mult=0.6, high_vol_mult=2.0)


def test_calm_market_smaller_divisor_bigger_clips() -> None:
    assert vol_scaled_budget_days(0.05, **_KW) == 6


def test_stressed_market_bigger_divisor_smaller_clips() -> None:
    assert vol_scaled_budget_days(0.50, **_KW) == 20


def test_midpoint_linear_ramp() -> None:
    assert vol_scaled_budget_days(0.25, **_KW) == 13  # 10*(0.6+0.5*1.4)=13.0


def test_nan_fails_open_to_base_divisor() -> None:
    assert vol_scaled_budget_days(float("nan"), base_divisor=10) == 10


def test_degenerate_knobs_fail_open() -> None:
    assert vol_scaled_budget_days(0.5, base_divisor=10,
                                  low_vol_trigger=0.5, high_vol_trigger=0.3) == 10


def test_divisor_never_below_one() -> None:
    assert vol_scaled_budget_days(0.05, base_divisor=1, low_vol_mult=0.01) >= 1
