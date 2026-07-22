"""Market breadth pure functions (meta-controller leg, 20-07-26).

breadth_from_panel: fraction of tickers with positive trailing-window return.
breadth_scalar: piecewise-linear exposure mapping, same shape as
garch_brake.drift_scalar_from_returns.
"""
from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

import pandas as pd

from src.trading.breadth import (
    breadth_delta_from_panel,
    breadth_from_panel,
    breadth_scalar,
    breadth_time_series,
    live_breadth_inflection,
)


def _panel(tickers_closes: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for ticker, closes in tickers_closes.items():
        for i, c in enumerate(closes):
            rows.append({"ticker": ticker, "date": f"2026-01-{i+1:02d}", "close": c})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# breadth_from_panel
# ---------------------------------------------------------------------------


def _flat_series(n: int, start: float = 100.0) -> list[float]:
    return [start] * n


def test_all_positive_breadth_is_one():
    # 30 tickers, each up 5% over the 20-session window.
    data = {f"T{i}": [100.0] * 20 + [105.0] for i in range(30)}
    assert breadth_from_panel(_panel(data), window=20, min_tickers=30) == pytest.approx(1.0)


def test_all_negative_breadth_is_zero():
    data = {f"T{i}": [100.0] * 20 + [95.0] for i in range(30)}
    assert breadth_from_panel(_panel(data), window=20, min_tickers=30) == pytest.approx(0.0)


def test_mixed_breadth_fraction():
    data = {}
    for i in range(20):
        data[f"UP{i}"] = [100.0] * 20 + [110.0]
    for i in range(10):
        data[f"DN{i}"] = [100.0] * 20 + [90.0]
    assert breadth_from_panel(_panel(data), window=20, min_tickers=30) == pytest.approx(20 / 30)


def test_insufficient_history_excluded():
    # Only 5 bars of history — below window+1 — must be skipped, not error.
    data = {f"T{i}": [100.0] * 20 + [105.0] for i in range(29)}
    data["SHORT"] = [100.0] * 5
    assert breadth_from_panel(_panel(data), window=20, min_tickers=29) == pytest.approx(1.0)


def test_below_min_tickers_returns_none():
    data = {f"T{i}": [100.0] * 20 + [105.0] for i in range(10)}
    assert breadth_from_panel(_panel(data), window=20, min_tickers=30) is None


def test_empty_panel_returns_none():
    assert breadth_from_panel(pl.DataFrame({"ticker": [], "date": [], "close": []})) is None


def test_zero_or_none_base_price_excluded_not_crashed():
    data = {f"T{i}": [100.0] * 20 + [105.0] for i in range(29)}
    data["ZERO"] = [0.0] * 20 + [10.0]
    # ZERO excluded (non-positive base) — still resolves against the 29 good names.
    assert breadth_from_panel(_panel(data), window=20, min_tickers=29) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# breadth_scalar
# ---------------------------------------------------------------------------


def test_breadth_above_trigger_full_exposure():
    assert breadth_scalar(0.50, trigger=0.40, floor_level=0.25, floor=0.5) == 1.0


def test_breadth_at_or_below_floor_level_hits_floor():
    assert breadth_scalar(0.25, trigger=0.40, floor_level=0.25, floor=0.5) == 0.5
    assert breadth_scalar(0.10, trigger=0.40, floor_level=0.25, floor=0.5) == 0.5


def test_breadth_ramp_midpoint():
    # Exactly halfway between floor_level (0.25) and trigger (0.40) → 0.325.
    s = breadth_scalar(0.325, trigger=0.40, floor_level=0.25, floor=0.5)
    assert s == pytest.approx(0.75)


def test_july_shape_lands_mid_ramp():
    # The actual 20-07 check_drift.py reading (0.295, different computation
    # basis but same rough scale) must land strictly inside the ramp.
    s = breadth_scalar(0.295, trigger=0.40, floor_level=0.25, floor=0.5)
    assert 0.5 < s < 1.0


def test_degenerate_knobs_fail_open():
    assert breadth_scalar(0.10, trigger=0.25, floor_level=0.40, floor=0.5) == 1.0  # inverted
    assert breadth_scalar(0.10, trigger=0.40, floor_level=0.25, floor=0.0) == 1.0  # bad floor


# ---------------------------------------------------------------------------
# breadth_delta_from_panel (22-07-26, knife-catch inflection annotation)
# ---------------------------------------------------------------------------


def _rising_panel(n_tickers: int = 30, n_days: int = 30, jump_day: int = 26) -> pl.DataFrame:
    """Breadth genuinely rises from 'earlier' to 'now' (window=delta_window=5).

    All tickers flat until `jump_day` (26), then +5%. With the "now" window
    ending at day29 comparing back to day24, the jump falls INSIDE that
    window (24 < 26 <= 29) -> every ticker shows a positive trailing return
    -> now_breadth=1.0. The "earlier" window ends at day24 comparing back to
    day19, entirely BEFORE the jump (19 < 24 < 26) -> earlier_breadth=0.0.
    """
    rows = []
    for i in range(n_tickers):
        for d in range(n_days):
            close = 100.0 if d < jump_day else 105.0
            rows.append({"ticker": f"T{i}", "date": f"2026-01-{d+1:02d}", "close": close})
    return pl.DataFrame(rows)


def test_breadth_delta_detects_rising():
    panel = _rising_panel()
    result = breadth_delta_from_panel(panel, window=5, delta_window=5, min_tickers=20)
    assert result is not None
    now, delta = result
    assert delta > 0  # breadth higher now than 5 sessions ago


def test_breadth_delta_flat_series_is_zero():
    data = {f"T{i}": [100.0] * 30 for i in range(30)}
    rows = [
        {"ticker": t, "date": f"2026-01-{d+1:02d}", "close": c}
        for t, closes in data.items()
        for d, c in enumerate(closes)
    ]
    panel = pl.DataFrame(rows)
    result = breadth_delta_from_panel(panel, window=5, delta_window=5, min_tickers=20)
    assert result is not None
    now, delta = result
    assert delta == pytest.approx(0.0)


def test_breadth_delta_too_short_history_returns_none():
    data = {f"T{i}": [100.0] * 3 for i in range(30)}
    rows = [
        {"ticker": t, "date": f"2026-01-{d+1:02d}", "close": c}
        for t, closes in data.items()
        for d, c in enumerate(closes)
    ]
    panel = pl.DataFrame(rows)
    assert breadth_delta_from_panel(panel, window=5, delta_window=5) is None


def test_breadth_delta_empty_panel_returns_none():
    assert breadth_delta_from_panel(pl.DataFrame({"ticker": [], "date": [], "close": []})) is None


def test_breadth_delta_below_min_tickers_returns_none():
    data = {f"T{i}": [100.0] * 30 for i in range(5)}
    rows = [
        {"ticker": t, "date": f"2026-01-{d+1:02d}", "close": c}
        for t, closes in data.items()
        for d, c in enumerate(closes)
    ]
    panel = pl.DataFrame(rows)
    assert breadth_delta_from_panel(panel, window=5, delta_window=5, min_tickers=30) is None


# ---------------------------------------------------------------------------
# live_breadth_inflection — fail-open contract
# ---------------------------------------------------------------------------


def test_live_breadth_inflection_favorable_low_and_rising():
    with patch("src.backtest.pipeline.load_ohlcv", return_value="fake_panel"), \
         patch("src.trading.breadth.breadth_delta_from_panel",
               return_value=(0.20, 0.05)):
        result = live_breadth_inflection(low_cut=0.41, rising_threshold=0.03)
    assert result == {"breadth": 0.20, "breadth_delta": 0.05, "favorable": True}


def test_live_breadth_inflection_unfavorable_low_but_falling():
    with patch("src.backtest.pipeline.load_ohlcv", return_value="fake_panel"), \
         patch("src.trading.breadth.breadth_delta_from_panel",
               return_value=(0.20, -0.05)):
        result = live_breadth_inflection(low_cut=0.41, rising_threshold=0.03)
    assert result == {"breadth": 0.20, "breadth_delta": -0.05, "favorable": False}


def test_live_breadth_inflection_unfavorable_high_level_even_if_rising():
    with patch("src.backtest.pipeline.load_ohlcv", return_value="fake_panel"), \
         patch("src.trading.breadth.breadth_delta_from_panel",
               return_value=(0.80, 0.05)):
        result = live_breadth_inflection(low_cut=0.41, rising_threshold=0.03)
    assert result["favorable"] is False


def test_live_breadth_inflection_none_on_insufficient_data():
    with patch("src.backtest.pipeline.load_ohlcv", return_value="fake_panel"), \
         patch("src.trading.breadth.breadth_delta_from_panel", return_value=None):
        assert live_breadth_inflection() is None


def test_live_breadth_inflection_fails_open_on_exception():
    with patch("src.backtest.pipeline.load_ohlcv", side_effect=RuntimeError("no parquet")):
        assert live_breadth_inflection() is None  # must not raise


# ---------------------------------------------------------------------------
# breadth_time_series (22-07-26, feeds rank_breadth admission in walk_forward)
# ---------------------------------------------------------------------------


def test_breadth_time_series_empty_panel_returns_empty_series():
    result = breadth_time_series(pl.DataFrame({"ticker": [], "date": [], "close": []}))
    assert isinstance(result, pd.Series)
    assert result.empty


def test_breadth_time_series_index_is_datetime64():
    panel = _rising_panel()
    result = breadth_time_series(panel, window=5, min_tickers=20)
    assert not result.empty
    assert pd.api.types.is_datetime64_any_dtype(result.index)


def test_breadth_time_series_last_value_matches_breadth_from_panel():
    # Cross-check: the two functions compute the SAME math (breadth_from_panel
    # for a single snapshot, breadth_time_series for the whole history) — they
    # must agree at the panel's own latest date.
    panel = _rising_panel()
    series = breadth_time_series(panel, window=5, min_tickers=20)
    snapshot = breadth_from_panel(panel, window=5, min_tickers=20)
    assert series.iloc[-1] == pytest.approx(snapshot)


def test_breadth_time_series_rises_across_the_jump():
    # _rising_panel: flat until day26, then +5% for every ticker. A date
    # BEFORE the jump has breadth=0.0 (trailing window entirely flat); a date
    # comfortably AFTER has breadth=1.0 (every ticker's window captured it).
    panel = _rising_panel(n_tickers=30, n_days=30, jump_day=26)
    series = breadth_time_series(panel, window=5, min_tickers=20)
    before = pd.Timestamp("2026-01-24")   # day23 (0-idx) — window ends before jump
    after = series.index.max()            # day29 (0-idx) — window captures the jump
    assert series.loc[before] == pytest.approx(0.0)
    assert series.loc[after] == pytest.approx(1.0)


def test_breadth_time_series_drops_dates_below_min_tickers():
    data = {f"T{i}": [100.0] * 30 for i in range(10)}  # only 10 tickers
    rows = [
        {"ticker": t, "date": f"2026-01-{d+1:02d}", "close": c}
        for t, closes in data.items()
        for d, c in enumerate(closes)
    ]
    panel = pl.DataFrame(rows)
    result = breadth_time_series(panel, window=5, min_tickers=30)
    assert result.empty  # every date has only 10 valid readings, below the floor


def test_breadth_time_series_zero_window_returns_empty():
    panel = _rising_panel()
    result = breadth_time_series(panel, window=0)
    assert result.empty
