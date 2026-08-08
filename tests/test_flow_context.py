"""Live per-ticker foreign-flow divergence context (08-08-26).

live_flow_divergence fetches FastConnect history + local OHLCV for ONE
ticker, joins, runs build_flow_features (real, not mocked -- already
validated elsewhere), and reads the latest divergence flag. External I/O
(fetch_fastconnect_history_for_symbol, ohlc_history) is mocked at its
SOURCE module -- both are imported locally inside the function, so
monkeypatching the source module's attribute is what actually takes
effect (a module-level `from X import Y` alias would not).
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import polars as pl
import pytest

from src.trading.flow_context import live_flow_divergence

N = 40


def _flow_rows(ticker: str, n: int, start: dt.date) -> pl.DataFrame:
    days = [start + dt.timedelta(days=i) for i in range(n)]
    # Alternating net flow -- some variance is required, or the rolling std
    # is exactly 0 and build_flow_features correctly nulls the z-score (and
    # therefore the divergence flag) for every row, by design.
    rows = [{
        "date": d, "ticker": ticker,
        "foreign_buy_val": 1000.0, "foreign_sell_val": 1000.0,
        "foreign_net_val": 500.0 if i % 2 == 0 else -300.0,
        "prop_buy_val": None, "prop_sell_val": None, "prop_net_val": None,
    } for i, d in enumerate(days)]
    return pl.DataFrame(rows)


def _price_bars(n: int, start: dt.date, spike_last: bool = False) -> list[tuple]:
    days = [start + dt.timedelta(days=i) for i in range(n)]
    px = [100.0 - 0.2 * i for i in range(n)]
    if spike_last:
        px[-1] = px[-2] * 0.97  # -3% on the last bar, for a price-drop divergence test
    return [(days[i], px[i] * 1.001, px[i] * 1.01, px[i] * 0.99, px[i], 1_000_000.0)
            for i in range(n)]


def test_empty_flow_returns_none():
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               return_value=pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8})):
        assert live_flow_divergence("AAA") is None


def test_empty_price_history_returns_none():
    start = dt.date(2026, 6, 1)
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               return_value=_flow_rows("AAA", N, start)), \
         patch("src.data.price_lookup.ohlc_history", return_value=[]):
        assert live_flow_divergence("AAA") is None


def test_no_overlapping_dates_returns_none():
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               return_value=_flow_rows("AAA", N, dt.date(2020, 1, 1))), \
         patch("src.data.price_lookup.ohlc_history",
               return_value=_price_bars(N, dt.date(2026, 6, 1))):
        assert live_flow_divergence("AAA") is None


def test_happy_path_returns_divergence_dict():
    start = dt.date(2026, 6, 1)
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               return_value=_flow_rows("AAA", N, start)), \
         patch("src.data.price_lookup.ohlc_history",
               return_value=_price_bars(N, start)):
        result = live_flow_divergence("AAA")
    assert result is not None
    assert "divergence" in result
    assert isinstance(result["divergence"], bool)
    assert "flow_net_scaled_adv20" in result


def test_fetch_exception_fails_open_to_none():
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               side_effect=RuntimeError("network boom")):
        assert live_flow_divergence("AAA") is None


def test_ticker_is_uppercased():
    start = dt.date(2026, 6, 1)
    with patch("src.data.foreign_flow_crawler.fetch_fastconnect_history_for_symbol",
               return_value=_flow_rows("AAA", N, start)) as m_fetch, \
         patch("src.data.price_lookup.ohlc_history",
               return_value=_price_bars(N, start)):
        live_flow_divergence("aaa")
    assert m_fetch.call_args.args[0] == "AAA"
