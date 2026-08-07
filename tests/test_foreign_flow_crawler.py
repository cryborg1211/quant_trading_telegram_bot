"""Tests for src/data/foreign_flow_crawler.py.

Focus: the 5xx-retry contract fix (deep-dive M1) and the degrade-to-empty
contract. Retry-predicate logic is unit-tested directly (fast, no real
tenacity sleeps); the crawl_today degrade paths are tested by monkeypatching
the single network call. FastConnect (SOURCE 2, 26-07-26) tests never hit
the real network -- every HTTP call is monkeypatched.
"""
from __future__ import annotations

from datetime import date, datetime

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


# ── _fastconnect_access_token ────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, json_body: dict, status: int = 200) -> None:
        self._json = json_body
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code), response=self)

    def json(self) -> dict:
        return self._json


def test_access_token_missing_env_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("Consumer_Key", raising=False)
    monkeypatch.delenv("ConsumerSecret_Key", raising=False)
    assert ffc._fastconnect_access_token() is None


def test_access_token_success_extracts_token(monkeypatch) -> None:
    monkeypatch.setenv("Consumer_Key", "fake_id")
    monkeypatch.setenv("ConsumerSecret_Key", "fake_secret")
    monkeypatch.setattr(
        ffc.requests, "post",
        lambda *a, **kw: _FakeResponse({"data": {"accessToken": "eyJfake"}, "status": 200}),
    )
    assert ffc._fastconnect_access_token() == "eyJfake"


def test_access_token_malformed_response_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("Consumer_Key", "fake_id")
    monkeypatch.setenv("ConsumerSecret_Key", "fake_secret")
    monkeypatch.setattr(
        ffc.requests, "post",
        lambda *a, **kw: _FakeResponse({"data": {}, "message": "bad creds", "status": 400}),
    )
    assert ffc._fastconnect_access_token() is None


def test_access_token_network_failure_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("Consumer_Key", "fake_id")
    monkeypatch.setenv("ConsumerSecret_Key", "fake_secret")
    def _boom(*a, **kw):
        raise requests.exceptions.ConnectionError("no route")
    monkeypatch.setattr(ffc.requests, "post", _boom)
    assert ffc._fastconnect_access_token() is None


# ── _parse_fastconnect_row ────────────────────────────────────────────────────

_FC_RAW_ROW = {
    "TradingDate": "07/08/2026",
    "Symbol": "SSI",
    "ForeignBuyValTotal": "25932114700",
    "ForeignSellValTotal": "12770530500",
    "NetBuySellVal": "13161584200",
    "ForeignBuyVolTotal": "1059455",
    "ForeignSellVolTotal": "521650",
    "ForeignCurrentRoom": "1753039675",
}


def test_parse_fastconnect_row_maps_and_scales_to_thousands() -> None:
    fetched_at = datetime(2026, 8, 7, 15, 0)
    row = ffc._parse_fastconnect_row(_FC_RAW_ROW, fetched_at)
    assert row is not None
    assert row["date"] == date(2026, 8, 7)
    assert row["ticker"] == "SSI"
    assert row["foreign_buy_val"] == 25_932_114.7
    assert row["foreign_sell_val"] == 12_770_530.5
    assert row["foreign_net_val"] == 13_161_584.2
    assert row["foreign_buy_vol"] == 1_059_455.0
    assert row["foreign_remain_room_vol"] == 1_753_039_675.0
    assert row["prop_buy_val"] is None
    assert row["source"] == "ssi_fastconnect_history"


def test_parse_fastconnect_row_malformed_date_returns_none() -> None:
    bad = dict(_FC_RAW_ROW, TradingDate="not-a-date")
    assert ffc._parse_fastconnect_row(bad, datetime.now()) is None


def test_parse_fastconnect_row_missing_symbol_returns_none() -> None:
    bad = dict(_FC_RAW_ROW, Symbol=None)
    assert ffc._parse_fastconnect_row(bad, datetime.now()) is None


def test_parse_fastconnect_row_empty_string_field_is_none() -> None:
    bad = dict(_FC_RAW_ROW, ForeignBuyValTotal="")
    row = ffc._parse_fastconnect_row(bad, datetime.now())
    assert row is not None
    assert row["foreign_buy_val"] is None


# ── _chunk_date_range: SSI's undocumented 30-day-per-call cap ────────────────

def test_chunk_date_range_short_span_is_one_window() -> None:
    windows = ffc._chunk_date_range(date(2026, 8, 1), date(2026, 8, 7), 30)
    assert windows == [(date(2026, 8, 1), date(2026, 8, 7))]


def test_chunk_date_range_splits_at_max_days() -> None:
    windows = ffc._chunk_date_range(date(2026, 1, 1), date(2026, 3, 1), 30)
    # 60 calendar days -> 2 windows of 30, no gaps, no overlaps, covers the full span.
    assert len(windows) >= 2
    assert windows[0][0] == date(2026, 1, 1)
    assert windows[-1][1] == date(2026, 3, 1)
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert e1 < s2  # no overlap
        assert (s2 - e1).days == 1  # no gap


def test_chunk_date_range_inverted_range_is_empty() -> None:
    assert ffc._chunk_date_range(date(2026, 8, 7), date(2026, 8, 1), 30) == []


def test_chunk_date_range_single_day() -> None:
    assert ffc._chunk_date_range(date(2026, 8, 1), date(2026, 8, 1), 30) == [
        (date(2026, 8, 1), date(2026, 8, 1))
    ]


# ── fetch_fastconnect_history_for_symbol: pagination ─────────────────────────

def test_history_for_symbol_paginates_until_exhausted(monkeypatch) -> None:
    page1 = dict(_FC_RAW_ROW, TradingDate="01/08/2026")
    page2 = dict(_FC_RAW_ROW, TradingDate="02/08/2026")
    calls: list[int] = []

    def _fake_fetch(*, symbol, from_date, to_date, market="HOSE",
                     page_index=1, page_size=1000, token=None):
        calls.append(page_index)
        if page_index == 1:
            return {"data": [page1], "totalRecord": 2}
        if page_index == 2:
            return {"data": [page2], "totalRecord": 2}
        return {"data": []}

    monkeypatch.setattr(ffc, "fetch_fastconnect_daily_stock_price", _fake_fetch)
    out = ffc.fetch_fastconnect_history_for_symbol(
        "SSI", date(2026, 8, 1), date(2026, 8, 2), token="fake-token", page_size=1)
    assert calls == [1, 2]
    assert out.height == 2


def test_history_for_symbol_no_token_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(ffc, "_fastconnect_access_token", lambda: None)
    out = ffc.fetch_fastconnect_history_for_symbol("SSI", date(2026, 8, 1), date(2026, 8, 2))
    assert out.is_empty()
    assert out.columns == list(ffc._SCHEMA)


def test_history_for_symbol_empty_data_stops_loop(monkeypatch) -> None:
    monkeypatch.setattr(
        ffc, "fetch_fastconnect_daily_stock_price",
        lambda **kw: {"data": []},
    )
    out = ffc.fetch_fastconnect_history_for_symbol(
        "SSI", date(2026, 8, 1), date(2026, 8, 2), token="fake-token")
    assert out.is_empty()


# ── backfill_foreign_flow_history ─────────────────────────────────────────────

def test_backfill_merges_multiple_tickers(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ffc, "_fastconnect_access_token", lambda: "fake-token")

    def _fake_history(symbol, from_date, to_date, *, market="HOSE",
                       token=None, page_size=1000):
        row = dict(_FC_RAW_ROW, Symbol=symbol)
        parsed = ffc._parse_fastconnect_row(row, datetime.now())
        return pl.DataFrame([parsed], schema=ffc._SCHEMA)

    monkeypatch.setattr(ffc, "fetch_fastconnect_history_for_symbol", _fake_history)
    path = tmp_path / "foreign_flow_daily.parquet"
    total = ffc.backfill_foreign_flow_history(
        ["SSI", "VJC"], date(2026, 8, 1), date(2026, 8, 7), parquet_path=path)
    assert total == 2
    out = pl.read_parquet(path)
    assert set(out["ticker"].to_list()) == {"SSI", "VJC"}
    assert set(out["source"].to_list()) == {"ssi_fastconnect_history"}


def test_backfill_one_bad_ticker_does_not_abort_others(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ffc, "_fastconnect_access_token", lambda: "fake-token")

    def _fake_history(symbol, from_date, to_date, *, market="HOSE",
                       token=None, page_size=1000):
        if symbol == "BAD":
            raise ValueError("boom")
        row = dict(_FC_RAW_ROW, Symbol=symbol)
        parsed = ffc._parse_fastconnect_row(row, datetime.now())
        return pl.DataFrame([parsed], schema=ffc._SCHEMA)

    monkeypatch.setattr(ffc, "fetch_fastconnect_history_for_symbol", _fake_history)
    path = tmp_path / "foreign_flow_daily.parquet"
    total = ffc.backfill_foreign_flow_history(
        ["SSI", "BAD", "VJC"], date(2026, 8, 1), date(2026, 8, 7), parquet_path=path)
    assert total == 2
    out = pl.read_parquet(path)
    assert set(out["ticker"].to_list()) == {"SSI", "VJC"}


def test_backfill_no_token_leaves_parquet_unchanged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ffc, "_fastconnect_access_token", lambda: None)
    path = tmp_path / "foreign_flow_daily.parquet"
    total = ffc.backfill_foreign_flow_history(["SSI"], date(2026, 8, 1), date(2026, 8, 7), parquet_path=path)
    assert total == 0
    assert not path.exists()


def test_backfill_idempotent_merge_with_existing_parquet(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ffc, "_fastconnect_access_token", lambda: "fake-token")
    path = tmp_path / "foreign_flow_daily.parquet"

    # Pre-seed the parquet with an OLD row for the same (date, ticker) via
    # SOURCE 1's own live-crawl shape, to prove fresh (FastConnect) wins.
    old_row = {
        "date": date(2026, 8, 7), "ticker": "SSI",
        "foreign_buy_val": 1.0, "foreign_sell_val": 1.0, "foreign_net_val": 0.0,
        "foreign_buy_vol": 1.0, "foreign_sell_vol": 1.0, "foreign_remain_room_vol": 1.0,
        "prop_buy_val": None, "prop_sell_val": None, "prop_net_val": None,
        "source": "ssi_iboard_live_snapshot", "fetched_at": datetime(2026, 8, 7, 9, 0),
    }
    pl.DataFrame([old_row], schema=ffc._SCHEMA).write_parquet(path)

    def _fake_history(symbol, from_date, to_date, *, market="HOSE",
                       token=None, page_size=1000):
        row = dict(_FC_RAW_ROW, Symbol=symbol)
        parsed = ffc._parse_fastconnect_row(row, datetime(2026, 8, 7, 15, 0))
        return pl.DataFrame([parsed], schema=ffc._SCHEMA)

    monkeypatch.setattr(ffc, "fetch_fastconnect_history_for_symbol", _fake_history)
    total = ffc.backfill_foreign_flow_history(["SSI"], date(2026, 8, 7), date(2026, 8, 7), parquet_path=path)
    assert total == 1  # merged onto the same (date, ticker), not appended
    out = pl.read_parquet(path)
    assert out.height == 1
    assert out.row(0, named=True)["source"] == "ssi_fastconnect_history"  # fresher fetched_at wins
