"""FastConnect OHLCV fast path (10-08-26).

THE TWO REGRESSIONS THESE GUARD
───────────────────────────────
1. A shared rate limiter. If each worker paces itself instead, N workers run
   N times over the 60 req/min budget and the account gets throttled or
   banned mid-crawl. `test_rate_limiter_*` pin the window to being shared.
2. Prefetch-window sufficiency. Serving a ticker from a 30-day prefetch when
   its local history is 60 days behind would silently leave a permanent hole
   — the parquet's max date advances, so the next run's incremental never
   looks back at the gap again. `test_*_covers_from_*` pin the guard.
"""
from __future__ import annotations

import threading
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data import fastconnect_ohlcv as fc
from src.data.crawlers import StockCrawler


def _fc_row(d: str, close: str = "50000") -> dict:
    return {"Symbol": "AAA", "Market": "HOSE", "TradingDate": d, "Time": None,
            "Open": "49000", "High": "51000", "Low": "48000", "Close": close,
            "Volume": "1000000", "Value": "50000000000"}


# ── _parse_rows ─────────────────────────────────────────────────────────────

def test_parse_scales_absolute_vnd_to_thousands():
    # The whole codebase's price-scale convention: parquet holds thousands of
    # VND. Getting this wrong by 1000x breaks every cost and tick calc.
    df = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    assert df.loc[0, "close"] == pytest.approx(50.0)
    assert df.loc[0, "open"] == pytest.approx(49.0)
    assert df.loc[0, "volume"] == pytest.approx(1_000_000.0)


def test_parse_sets_adj_close_to_close():
    df = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    assert df.loc[0, "adj_close"] == df.loc[0, "close"]


def test_parse_yields_exact_expected_columns():
    df = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    assert list(df.columns) == ["ticker", "date", "open", "high", "low",
                                "close", "volume", "adj_close"]


def test_parse_converts_ddmmyyyy_not_mmddyyyy():
    # 10/08/2026 is 10 August in SSI's format. Reading it as 8 October would
    # place bars ~2 months into the future and corrupt every label.
    df = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    assert df.loc[0, "date"] == date(2026, 8, 10)


def test_parse_drops_zero_price_rows():
    rows = [_fc_row("10/08/2026", close="0"), _fc_row("11/08/2026")]
    df = fc._parse_rows("AAA", rows)
    assert len(df) == 1 and df.loc[0, "date"] == date(2026, 8, 11)


def test_parse_skips_malformed_rows_without_raising():
    rows = [{"TradingDate": "garbage"}, {}, _fc_row("10/08/2026")]
    df = fc._parse_rows("AAA", rows)
    assert len(df) == 1


def test_parse_empty_returns_typed_empty_frame():
    df = fc._parse_rows("AAA", [])
    assert df.empty and "close" in df.columns


def test_parse_sorts_ascending_by_date():
    df = fc._parse_rows("AAA", [_fc_row("11/08/2026"), _fc_row("10/08/2026")])
    assert list(df["date"]) == [date(2026, 8, 10), date(2026, 8, 11)]


# ── _RateLimiter ────────────────────────────────────────────────────────────

def test_rate_limiter_allows_burst_up_to_budget():
    limiter = fc._RateLimiter(max_calls=5, per_seconds=60.0)
    t0 = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - t0 < 0.5     # no sleeping inside the budget


def test_rate_limiter_blocks_past_budget():
    limiter = fc._RateLimiter(max_calls=2, per_seconds=0.5)
    limiter.acquire()
    limiter.acquire()
    t0 = time.monotonic()
    limiter.acquire()                      # must wait for the window to roll
    assert time.monotonic() - t0 >= 0.4


def test_rate_limiter_budget_is_shared_across_threads():
    # The regression: a per-thread limiter would let 4 threads each take the
    # full budget. With a shared window, 4 threads x 2 calls against a
    # 4-call budget must still be forced to wait.
    limiter = fc._RateLimiter(max_calls=4, per_seconds=0.6)
    done: list[float] = []
    lock = threading.Lock()

    def worker():
        for _ in range(2):
            limiter.acquire()
            with lock:
                done.append(time.monotonic())

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(done) == 8
    # 8 calls against a 4-per-0.6s budget cannot finish inside one window.
    assert max(done) - t0 >= 0.5


# ── fetch_daily_ohlc ────────────────────────────────────────────────────────

def test_fetch_clamps_a_span_over_the_30_day_cap():
    # Past 30 days FastConnect returns 0 rows with status="Success" — silent,
    # so the clamp must happen on our side or a long window returns nothing.
    session = MagicMock()
    session.get.return_value = MagicMock(
        json=lambda: {"data": [_fc_row("10/08/2026")]}, raise_for_status=lambda: None)
    fc.fetch_daily_ohlc("AAA", date(2026, 1, 1), date(2026, 8, 10), "tok", session=session)
    sent_from = session.get.call_args.kwargs["params"]["FromDate"]
    assert sent_from == "12/07/2026"       # 30 calendar days back from 10/08


def test_fetch_returns_empty_on_network_error():
    session = MagicMock()
    session.get.side_effect = RuntimeError("connection reset")
    df = fc.fetch_daily_ohlc("AAA", date(2026, 8, 1), date(2026, 8, 10), "tok", session=session)
    assert df.empty and list(df.columns) == fc._OHLCV_COLUMNS


def test_fetch_uses_ddmmyyyy_params():
    session = MagicMock()
    session.get.return_value = MagicMock(
        json=lambda: {"data": []}, raise_for_status=lambda: None)
    fc.fetch_daily_ohlc("AAA", date(2026, 8, 3), date(2026, 8, 10), "tok", session=session)
    params = session.get.call_args.kwargs["params"]
    assert params["FromDate"] == "03/08/2026" and params["ToDate"] == "10/08/2026"


# ── bulk_prefetch ───────────────────────────────────────────────────────────

def test_bulk_prefetch_without_credentials_returns_empty_not_raises():
    with patch.object(fc, "_access_token", return_value=None):
        got, start = fc.bulk_prefetch(["AAA", "BBB"], date(2026, 8, 10))
    assert got == {} and start == date(2026, 7, 12)


def test_bulk_prefetch_omits_empty_frames():
    # A missing key is the caller's signal to use the vnstock fallback; an
    # empty frame in the dict would be read as "covered, nothing new".
    def fake(symbol, *_a, **_k):
        return (fc._parse_rows(symbol, [_fc_row("10/08/2026")]) if symbol == "AAA"
                else pd.DataFrame(columns=fc._OHLCV_COLUMNS))

    with patch.object(fc, "_access_token", return_value="tok"), \
         patch.object(fc, "fetch_daily_ohlc", side_effect=fake):
        got, _ = fc.bulk_prefetch(["AAA", "BBB"], date(2026, 8, 10))
    assert set(got) == {"AAA"}


def test_bulk_prefetch_empty_ticker_list_short_circuits():
    with patch.object(fc, "_access_token") as tok:
        got, _ = fc.bulk_prefetch([], date(2026, 8, 10))
    assert got == {} and tok.call_count == 0


# ── crawlers.fetch_ohlcv prefetch integration ───────────────────────────────

@pytest.fixture()
def crawler():
    with patch("src.data.crawlers.Listing"):
        return StockCrawler()


def _existing_parquet(tmp_path, last_date: str):
    path = tmp_path / "ohlcv_AAA.parquet"
    pd.DataFrame({
        "ticker": ["AAA"], "date": [pd.Timestamp(last_date).date()],
        "open": [40.0], "high": [41.0], "low": [39.0], "close": [40.5],
        "volume": [1000.0], "adj_close": [40.5],
    }).to_parquet(path, index=False)
    return str(path)


def test_prefetch_is_used_and_skips_the_throttle(crawler, tmp_path):
    path = _existing_parquet(tmp_path, "2026-08-07")
    pre = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    with patch.object(crawler, "_throttle_request") as throttle, \
         patch("src.data.crawlers.Quote") as quote:
        out = crawler.fetch_ohlcv(
            ticker="AAA", end_date="2026-08-10", file_path=path,
            sleep_before_request=True,
            prefetched=pre, prefetch_covers_from="2026-07-12")
    # The 4.25s sleep and the vnstock call are exactly what we are removing.
    assert throttle.call_count == 0
    assert quote.call_count == 0
    assert date(2026, 8, 10) in set(out["date"])
    assert len(out) == 2                    # old bar kept, new bar appended


def test_prefetch_ignored_when_window_starts_after_the_gap(crawler, tmp_path):
    # Local history ends 2026-05-01 but the prefetch only reaches 2026-07-12.
    # Using it would advance the parquet to 10/08 and orphan May-July forever.
    path = _existing_parquet(tmp_path, "2026-05-01")
    pre = fc._parse_rows("AAA", [_fc_row("10/08/2026")])
    with patch.object(crawler, "_throttle_request") as throttle, \
         patch("src.data.crawlers.Quote") as quote:
        quote.return_value.history.return_value = pd.DataFrame()
        crawler.fetch_ohlcv(
            ticker="AAA", end_date="2026-08-10", file_path=path,
            sleep_before_request=True,
            prefetched=pre, prefetch_covers_from="2026-07-12")
    assert throttle.call_count == 1         # fell through to the slow path
    assert quote.call_count == 1


def test_no_prefetch_behaves_exactly_as_before(crawler, tmp_path):
    path = _existing_parquet(tmp_path, "2026-08-07")
    with patch.object(crawler, "_throttle_request") as throttle, \
         patch("src.data.crawlers.Quote") as quote:
        quote.return_value.history.return_value = pd.DataFrame()
        crawler.fetch_ohlcv(ticker="AAA", end_date="2026-08-10", file_path=path,
                            sleep_before_request=True)
    assert throttle.call_count == 1 and quote.call_count == 1


def test_prefetch_filters_bars_at_or_before_the_local_max(crawler, tmp_path):
    # The prefetch window deliberately overlaps existing history; only bars
    # strictly newer than the local max may be appended.
    path = _existing_parquet(tmp_path, "2026-08-07")
    pre = fc._parse_rows("AAA", [_fc_row("06/08/2026"), _fc_row("07/08/2026"),
                                 _fc_row("10/08/2026")])
    with patch("src.data.crawlers.Quote"):
        out = crawler.fetch_ohlcv(
            ticker="AAA", end_date="2026-08-10", file_path=path,
            prefetched=pre, prefetch_covers_from="2026-07-12")
    assert sorted(str(d) for d in out["date"]) == ["2026-08-07", "2026-08-10"]
    # The pre-existing 07/08 row must not be duplicated by the overlap.
    assert len(out) == 2


def test_merge_canonicalizes_volume_dtype_across_sources(crawler, tmp_path):
    """THE 10-08-26 PRODUCTION CRASH.

    vnstock writes int64 volume, FastConnect float64. That split the 359
    OHLCV shards into two parquet schemas, and the EOD pipeline's
    `pl.concat(how="diagonal")` died with
    `SchemaError: type Int64 is incompatible with expected type Float64`
    (`failed to vstack column 'volume'`) — after a clean crawl, so the run
    produced no signals at all.
    """
    path = tmp_path / "ohlcv_AAA.parquet"
    old = pd.DataFrame({
        "ticker": ["AAA"], "date": [date(2026, 8, 7)],
        "open": [40.0], "high": [41.0], "low": [39.0], "close": [40.5],
        "volume": pd.Series([1000], dtype="int64"),      # vnstock's dtype
        "adj_close": [40.5],
    })
    old.to_parquet(path, index=False)
    pre = fc._parse_rows("AAA", [_fc_row("10/08/2026")])  # FastConnect: float64

    with patch("src.data.crawlers.Quote"):
        crawler.fetch_ohlcv(ticker="AAA", end_date="2026-08-10", file_path=str(path),
                            prefetched=pre, prefetch_covers_from="2026-07-12")

    written = pd.read_parquet(path)
    for col in ("open", "high", "low", "close", "volume", "adj_close"):
        assert written[col].dtype == "float64", f"{col} is {written[col].dtype}"


def test_merge_canonicalizes_dtype_on_the_vnstock_path_too(crawler, tmp_path):
    # The fix must not be prefetch-only, or a fallback ticker would keep
    # writing int64 and re-split the schemas.
    path = tmp_path / "ohlcv_BBB.parquet"
    old = pd.DataFrame({
        "ticker": ["BBB"], "date": [date(2026, 8, 7)],
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
        "volume": pd.Series([500], dtype="int64"), "adj_close": [10.5],
    })
    old.to_parquet(path, index=False)
    vn = pd.DataFrame({
        "time": [pd.Timestamp("2026-08-10")], "open": [10.6], "high": [11.2],
        "low": [10.1], "close": [11.0], "volume": pd.Series([700], dtype="int64"),
    })
    with patch("src.data.crawlers.Quote") as quote:
        quote.return_value.history.return_value = vn
        crawler.fetch_ohlcv(ticker="BBB", end_date="2026-08-10", file_path=str(path))

    written = pd.read_parquet(path)
    assert written["volume"].dtype == "float64"
    assert len(written) == 2


def test_prefetch_with_nothing_new_leaves_history_intact(crawler, tmp_path):
    path = _existing_parquet(tmp_path, "2026-08-10")
    pre = fc._parse_rows("AAA", [_fc_row("07/08/2026")])
    with patch("src.data.crawlers.Quote") as quote:
        out = crawler.fetch_ohlcv(
            ticker="AAA", end_date="2026-08-10", file_path=path,
            prefetched=pre, prefetch_covers_from="2026-07-12")
    assert quote.call_count == 0
    assert len(out) == 1 and str(out.iloc[0]["date"]) == "2026-08-10"
