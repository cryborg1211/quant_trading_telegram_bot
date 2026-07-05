"""Tests for src/data/foreign_flow_crawler.py.

Focus: the 5xx-retry contract fix (deep-dive M1) and the degrade-to-empty
contract. Retry-predicate logic is unit-tested directly (fast, no real
tenacity sleeps); the crawl_today degrade paths are tested by monkeypatching
the single network call.
"""
from __future__ import annotations

import polars as pl
import requests

from src.data import foreign_flow_crawler as ffc


def _http_error(status: int) -> requests.exceptions.HTTPError:
    """Build an HTTPError carrying a response with the given status code."""
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(f"{status} error", response=resp)


# ── _is_retryable_5xx: only 5xx HTTPError retries ────────────────────────────

def test_is_retryable_5xx_true_for_server_errors() -> None:
    for status in (500, 502, 503, 504):
        assert ffc._is_retryable_5xx(_http_error(status)) is True


def test_is_retryable_5xx_false_for_client_errors() -> None:
    for status in (400, 401, 404, 429):
        assert ffc._is_retryable_5xx(_http_error(status)) is False


def test_is_retryable_5xx_false_for_non_http_exception() -> None:
    # Connection/timeout are covered by _TRANSIENT_EXC, not this predicate.
    assert ffc._is_retryable_5xx(ConnectionError("boom")) is False
    assert ffc._is_retryable_5xx(ValueError("nope")) is False


def test_is_retryable_5xx_false_when_no_response_attached() -> None:
    # A raw HTTPError with no response object must not blow up or retry.
    assert ffc._is_retryable_5xx(requests.exceptions.HTTPError("bare")) is False


# ── crawl_today: degrade-to-empty contract ───────────────────────────────────

def test_crawl_today_degrades_to_empty_on_fetch_failure(monkeypatch) -> None:
    def _boom(exchange: str = "HOSE"):
        raise requests.exceptions.HTTPError("503", response=None)

    monkeypatch.setattr(ffc, "_fetch_ssi_hose_snapshot", _boom)
    out = ffc.crawl_today()
    assert out.is_empty()
    assert out.columns == list(ffc._SCHEMA)  # correct empty frame, not a crash


def test_crawl_today_empty_snapshot_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(ffc, "_fetch_ssi_hose_snapshot", lambda exchange="HOSE": [])
    out = ffc.crawl_today()
    assert out.is_empty()
    assert out.columns == list(ffc._SCHEMA)


def test_crawl_today_parses_and_scales_to_thousands(monkeypatch) -> None:
    snapshot = [{
        "stockSymbol": "SSI",
        "tradingDate": "20260705",
        "buyForeignValue": 2_000_000.0,   # absolute VND
        "sellForeignValue": 1_000_000.0,
        "buyForeignQtty": 100.0,
        "sellForeignQtty": 50.0,
        "remainForeignQtty": 999.0,
    }]
    monkeypatch.setattr(ffc, "_fetch_ssi_hose_snapshot", lambda exchange="HOSE": snapshot)
    out = ffc.crawl_today()
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["ticker"] == "SSI"
    # absolute VND scaled to thousands (on-disk convention)
    assert row["foreign_buy_val"] == 2000.0
    assert row["foreign_sell_val"] == 1000.0
    assert row["foreign_net_val"] == 1000.0


def test_crawl_today_ticker_filter(monkeypatch) -> None:
    snapshot = [
        {"stockSymbol": "SSI", "tradingDate": "20260705",
         "buyForeignValue": 1000.0, "sellForeignValue": 0.0},
        {"stockSymbol": "VJC", "tradingDate": "20260705",
         "buyForeignValue": 2000.0, "sellForeignValue": 0.0},
    ]
    monkeypatch.setattr(ffc, "_fetch_ssi_hose_snapshot", lambda exchange="HOSE": snapshot)
    out = ffc.crawl_today(tickers=["SSI"])
    assert out.height == 1
    assert out.row(0, named=True)["ticker"] == "SSI"
