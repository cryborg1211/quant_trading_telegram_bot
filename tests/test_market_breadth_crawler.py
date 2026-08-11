"""Official exchange breadth feed (FastConnect DailyIndex).

Pins the traps found while building it against the live API on 2026-08-11:
  * an UNSETTLED session returns IndexValue=0 with counts already partly
    filled — persisting that stores a mid-session snapshot as settled EOD
  * TradingDate is DD/MM/YYYY; reading it as MM/DD lands rows months away
  * every number arrives as a STRING
  * values stay in EXCHANGE units — this is not a price series, so the
    thousands-of-VND OHLCV convention must NOT be applied
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl

from src.data import market_breadth_crawler as mbc


def _row(**over) -> dict:
    base = {
        "IndexId": "VNINDEX", "TradingDate": "10/08/2026", "IndexValue": "1776.77",
        "Change": "0.087", "RatioChange": "0.49",
        "Advances": "232", "NoChanges": "47", "Declines": "86",
        "Ceilings": "13", "Floors": "2",
        "TotalMatchVol": "622931255", "TotalMatchVal": "15554436065900",
        "TotalDealVol": "107879827", "TotalDealVal": "2171320069800",
        "TotalVol": "730811082", "TotalVal": "17725756135700",
        "TradingSession": "C",
    }
    base.update(over)
    return base


# ── parse ───────────────────────────────────────────────────────────────────

def test_parses_ddmmyyyy_not_mmddyyyy():
    df = mbc.parse_daily_index_rows([_row(TradingDate="10/08/2026")])
    assert df["date"][0] == date(2026, 8, 10)


def test_coerces_string_numbers():
    df = mbc.parse_daily_index_rows([_row()])
    assert df["index_value"][0] == 1776.77
    assert df["advances"][0] == 232
    assert df["floors"][0] == 2
    assert df["total_deal_val"][0] == 2171320069800.0


def test_values_stay_in_exchange_units():
    """No /1000 rescaling. The OHLCV shards use thousands-of-VND because they
    feed VNCostModel's tick grid; index counts and turnover do not, and
    rescaling would silently make every downstream comparison wrong."""
    df = mbc.parse_daily_index_rows([_row(TotalDealVal="2171320069800")])
    assert df["total_deal_val"][0] == 2171320069800.0


def test_unsettled_session_row_is_dropped():
    # THE TRAP: a 14:18 crawl (before the 14:45 ATC close) returns IndexValue=0
    # with Advances/Declines already partly populated.
    rows = [_row(TradingDate="11/08/2026", IndexValue="0"),
            _row(TradingDate="10/08/2026")]
    df = mbc.parse_daily_index_rows(rows)
    assert df.height == 1
    assert df["date"][0] == date(2026, 8, 10)


def test_missing_index_value_is_dropped():
    df = mbc.parse_daily_index_rows([_row(IndexValue=None)])
    assert df.is_empty()


def test_negative_index_value_is_dropped():
    df = mbc.parse_daily_index_rows([_row(IndexValue="-1")])
    assert df.is_empty()


def test_unparseable_date_skipped_without_raising():
    df = mbc.parse_daily_index_rows([_row(TradingDate="garbage"), _row()])
    assert df.height == 1


def test_blank_numeric_becomes_null_not_zero():
    # Zero and "not reported" are different facts; conflating them would make a
    # missing floors count read as "no limit-down names".
    df = mbc.parse_daily_index_rows([_row(Floors="")])
    assert df["floors"][0] is None


def test_empty_input_returns_typed_frame():
    df = mbc.parse_daily_index_rows([])
    assert df.is_empty() and "floors" in df.columns


def test_rows_sorted_by_index_then_date():
    df = mbc.parse_daily_index_rows([
        _row(TradingDate="10/08/2026"), _row(TradingDate="03/08/2026"),
    ])
    assert df["date"].to_list() == [date(2026, 8, 3), date(2026, 8, 10)]


# ── chunking ────────────────────────────────────────────────────────────────

def test_chunks_respect_the_thirty_day_cap():
    # Past the cap FastConnect returns 0 rows with status="Success" — silently.
    wins = mbc._chunk_date_range(date(2026, 1, 1), date(2026, 3, 31))
    assert wins
    assert all((b - a).days + 1 <= mbc._FC_MAX_RANGE_DAYS for a, b in wins)
    assert wins[0][0] == date(2026, 1, 1) and wins[-1][1] == date(2026, 3, 31)


def test_chunks_are_contiguous_with_no_gap_or_overlap():
    wins = mbc._chunk_date_range(date(2026, 1, 1), date(2026, 4, 15))
    for (_, prev_end), (next_start, _) in zip(wins, wins[1:]):
        assert (next_start - prev_end).days == 1


def test_reversed_range_yields_no_windows():
    assert mbc._chunk_date_range(date(2026, 3, 1), date(2026, 1, 1)) == []


# ── fetch / persist ─────────────────────────────────────────────────────────

def test_fetch_without_token_returns_empty_not_raises():
    with patch.object(mbc, "_access_token", return_value=None):
        df = mbc.fetch_daily_index("VNINDEX", date(2026, 8, 1), date(2026, 8, 10))
    assert df.is_empty()


def test_fetch_sends_ddmmyyyy_params():
    session = MagicMock()
    session.get.return_value = MagicMock(
        json=lambda: {"data": [], "totalRecord": 0}, raise_for_status=lambda: None)
    mbc.fetch_daily_index("VNINDEX", date(2026, 8, 3), date(2026, 8, 10),
                          token="tok", session=session)
    params = session.get.call_args.kwargs["params"]
    assert params["FromDate"] == "03/08/2026" and params["ToDate"] == "10/08/2026"
    assert params["IndexId"] == "VNINDEX"


def test_fetch_survives_a_failing_window():
    session = MagicMock()
    session.get.side_effect = RuntimeError("network")
    df = mbc.fetch_daily_index("VNINDEX", date(2026, 8, 1), date(2026, 8, 10),
                               token="tok", session=session)
    assert df.is_empty()


def test_merge_is_idempotent_on_index_and_date(tmp_path):
    path = tmp_path / "breadth.parquet"
    first = mbc.parse_daily_index_rows([_row()])
    assert mbc._merge_and_persist(first, path) == 1
    # Same (index_id, date) re-written with a corrected count — fresh must win.
    again = mbc.parse_daily_index_rows([_row(Floors="9")])
    assert mbc._merge_and_persist(again, path) == 1
    out = pl.read_parquet(path)
    assert out.height == 1 and out["floors"][0] == 9


def test_merge_keeps_distinct_indices_for_the_same_date(tmp_path):
    path = tmp_path / "breadth.parquet"
    df = mbc.parse_daily_index_rows([_row(IndexId="VNINDEX"), _row(IndexId="VN30")])
    assert mbc._merge_and_persist(df, path) == 2


def test_daily_update_never_raises(tmp_path):
    path = tmp_path / "breadth.parquet"
    with patch.object(mbc, "backfill_market_breadth",
                      side_effect=RuntimeError("feed down")):
        assert mbc.update_market_breadth_daily(parquet_path=path) == 0


def test_backfill_without_token_leaves_parquet_untouched(tmp_path):
    path = tmp_path / "breadth.parquet"
    mbc._merge_and_persist(mbc.parse_daily_index_rows([_row()]), path)
    with patch.object(mbc, "_access_token", return_value=None):
        total = mbc.backfill_market_breadth(date(2026, 8, 1), date(2026, 8, 10),
                                           parquet_path=path)
    assert total == 1
    assert pl.read_parquet(path).height == 1
