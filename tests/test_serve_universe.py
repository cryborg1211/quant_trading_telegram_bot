"""Unit tests for src/trading/serve_universe.liquid_universe() — pure top-N ADV gate.

These prove the builder is correct in isolation BEFORE it is wired into main.py.
Note the DELIBERATE no-`shift(1)` design (test_no_shift_uses_latest_session):
the live snapshot uses the most recent session's own volume, unlike the
backtest's leak-safe shift — see the module docstring.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.trading.serve_universe import liquid_universe


# ── helpers ──────────────────────────────────────────────────────────────────


def _panel(rows: dict[str, list[tuple[float, float]]]) -> pl.DataFrame:
    """Build an OHLCV panel from {ticker: [(close, volume), ...]} chronological.

    Dates are assigned sequentially (one trading session per row), oldest first,
    so the LAST tuple per ticker is its most recent completed session.
    """
    records: list[dict] = []
    base = date(2026, 1, 1)
    for ticker, series in rows.items():
        for i, (close, volume) in enumerate(series):
            d = base + timedelta(days=i)
            records.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": volume,
                }
            )
    return pl.DataFrame(records)


# ── tests ────────────────────────────────────────────────────────────────────


def test_top_n_correctness():
    """Hand-computed ADV ranks over a 10-ticker panel; assert exact top-N set.

    Each ticker has 3 rows, adv_window=3 → ADV = mean(close*volume) over all 3.
    close is fixed at 10.0 for all; volume increases with ticker index so the
    dollar-volume ADV strictly orders T0 < T1 < ... < T9. top_n=4 must keep the
    four HIGHEST-ADV names: T6, T7, T8, T9.
    """
    rows = {
        f"T{i}": [(10.0, float((i + 1) * 100))] * 3 for i in range(10)
    }
    panel = _panel(rows)
    result = liquid_universe(panel, top_n=4, adv_window=3, min_valid_names=5)
    assert result == frozenset({"T6", "T7", "T8", "T9"})


def test_adv_window_respects_trailing_window():
    """A volume spike OUTSIDE the trailing window must not affect ranked ADV.

    adv_window=3 means only the LAST 3 rows count. AAA has a huge spike in its
    FIRST row (outside the trailing 3-row window given 5 total rows) but modest
    recent volume; BBB has steadily higher recent volume. BBB must outrank AAA
    despite AAA's early spike.
    """
    rows = {
        # AAA: massive early spike (row 0) then low; last-3 window = rows 2,3,4.
        "AAA": [(10.0, 1_000_000.0), (10.0, 10.0), (10.0, 10.0),
                (10.0, 10.0), (10.0, 10.0)],
        # BBB: steady, higher recent volume in the trailing window.
        "BBB": [(10.0, 10.0), (10.0, 10.0), (10.0, 100.0),
                (10.0, 100.0), (10.0, 100.0)],
        # Filler names so min_valid_names(5) is satisfied.
        "CCC": [(10.0, 50.0)] * 5,
        "DDD": [(10.0, 50.0)] * 5,
        "EEE": [(10.0, 50.0)] * 5,
    }
    panel = _panel(rows)
    result = liquid_universe(panel, top_n=1, adv_window=3, min_valid_names=5)
    # If the early spike leaked in, AAA would win; correct trailing-window math
    # keeps only rows 2-4 for AAA (all volume 10) → BBB (volume 100) wins.
    assert result == frozenset({"BBB"})


def test_no_shift_uses_latest_session():
    """Mutating the LAST row's volume changes that ticker's ADV / rank.

    Proves the deliberate no-`shift(1)` design: the most recent completed
    session DOES enter its own trailing ADV. A `.shift(1)` implementation would
    ignore the last row and this assertion would fail.
    """
    rows_low = {
        "AAA": [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
        "BBB": [(10.0, 50.0), (10.0, 50.0), (10.0, 50.0)],
        "CCC": [(10.0, 50.0), (10.0, 50.0), (10.0, 50.0)],
        "DDD": [(10.0, 50.0), (10.0, 50.0), (10.0, 50.0)],
        "EEE": [(10.0, 50.0), (10.0, 50.0), (10.0, 50.0)],
    }
    # AAA with a LOW last row is nowhere near the top.
    low = liquid_universe(_panel(rows_low), top_n=1, adv_window=3,
                          min_valid_names=5)
    assert "AAA" not in low

    rows_high = dict(rows_low)
    # Mutate ONLY AAA's LAST session volume to a huge value.
    rows_high["AAA"] = [(10.0, 10.0), (10.0, 10.0), (10.0, 1_000_000.0)]
    high = liquid_universe(_panel(rows_high), top_n=1, adv_window=3,
                          min_valid_names=5)
    # The latest session's volume DID enter AAA's ADV → AAA now wins.
    assert high == frozenset({"AAA"})


def test_insufficient_history_returns_empty():
    """Fewer than min_valid_names tickers with valid ADV → frozenset(), no raise.

    adv_window=5 but every ticker has only 3 rows → all ADVs are null (min_periods
    unmet). Zero valid names < min_valid_names(5) → empty set, NOT a partial set
    and NOT an exception.
    """
    rows = {f"T{i}": [(10.0, 100.0)] * 3 for i in range(10)}
    panel = _panel(rows)
    result = liquid_universe(panel, top_n=5, adv_window=5, min_valid_names=5)
    assert result == frozenset()


def test_missing_columns_raises():
    """Malformed input (missing `volume`) raises ValueError — loud schema break."""
    panel = pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "close": [10.0, 20.0],
            # no 'volume' column
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        liquid_universe(panel, top_n=1, adv_window=1, min_valid_names=1)


def test_ticker_isolation():
    """One ticker's ADV must be unaffected by another ticker's volume/price.

    Two identical-history tickers plus fillers. If the rolling window leaked
    across tickers (mis-grouped `over`), interleaving would corrupt AAA's ADV.
    Here AAA and BBB have IDENTICAL series → identical ADV, and a third ticker
    ZZZ with a wildly different volume must not change AAA's or BBB's ranking.
    """
    rows = {
        "AAA": [(10.0, 100.0), (10.0, 100.0), (10.0, 100.0)],
        "BBB": [(10.0, 100.0), (10.0, 100.0), (10.0, 100.0)],
        "ZZZ": [(10.0, 999_999.0), (10.0, 999_999.0), (10.0, 999_999.0)],
        "CCC": [(10.0, 1.0), (10.0, 1.0), (10.0, 1.0)],
        "DDD": [(10.0, 1.0), (10.0, 1.0), (10.0, 1.0)],
    }
    panel = _panel(rows)
    # top_n=3 → ZZZ (huge), then AAA & BBB (tie at 1000 dvol, alpha tie-break),
    # excluding CCC/DDD (dvol 10). Isolation: AAA==BBB exactly despite ZZZ.
    result = liquid_universe(panel, top_n=3, adv_window=3, min_valid_names=5)
    assert result == frozenset({"ZZZ", "AAA", "BBB"})
