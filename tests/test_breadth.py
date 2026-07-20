"""Market breadth pure functions (meta-controller leg, 20-07-26).

breadth_from_panel: fraction of tickers with positive trailing-window return.
breadth_scalar: piecewise-linear exposure mapping, same shape as
garch_brake.drift_scalar_from_returns.
"""
from __future__ import annotations

import polars as pl
import pytest

from src.trading.breadth import breadth_from_panel, breadth_scalar


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
