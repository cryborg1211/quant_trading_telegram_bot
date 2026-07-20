"""Candidate admission hysteresis (meta-controller optimization #4, 20-07-26).

BSR bounced in/out of the arbitrator pool 4 consecutive July days — require
N straight qualifying runs before admission. Module tests use a tmp_path
DuckDB file (never the real repo DB — same isolation discipline as the
paperlog/promote-gate tests earlier this session).
"""
from __future__ import annotations

from datetime import date, timedelta

from src.trading import candidate_hysteresis as ch


# ---------------------------------------------------------------------------
# read_streaks / update_streaks — isolated tmp DuckDB file
# ---------------------------------------------------------------------------


def test_never_seen_ticker_reads_zero(tmp_path):
    db = str(tmp_path / "h.duckdb")
    assert ch.read_streaks(["BSR"], db_path=db) == {}


def test_first_qualify_sets_streak_one(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    assert ch.read_streaks(["BSR"], db_path=db) == {"BSR": 1}


def test_consecutive_day_increments_streak(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 10), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 13), db_path=db)  # Mon after Fri — 3-day gap, within tolerance
    assert ch.read_streaks(["BSR"], db_path=db) == {"BSR": 3}


def test_gap_beyond_tolerance_resets_streak(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 20), db_path=db)  # 11-day gap
    assert ch.read_streaks(["BSR"], db_path=db) == {"BSR": 1}


def test_same_day_rerun_is_noop(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 9), db_path=db)
    assert ch.read_streaks(["BSR"], db_path=db) == {"BSR": 1}


def test_multiple_tickers_independent_streaks(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks(["BSR", "VCB"], date(2026, 7, 9), db_path=db)
    ch.update_streaks(["BSR"], date(2026, 7, 10), db_path=db)  # VCB skipped a day
    assert ch.read_streaks(["BSR", "VCB"], db_path=db) == {"BSR": 2, "VCB": 1}


def test_empty_ticker_list_is_noop(tmp_path):
    db = str(tmp_path / "h.duckdb")
    ch.update_streaks([], date(2026, 7, 9), db_path=db)  # must not raise
    assert ch.read_streaks([], db_path=db) == {}


def test_read_streaks_never_raises_on_bad_path(tmp_path):
    bad = str(tmp_path / "missing_parent_dir" / "sub" / "x.duckdb")
    assert ch.read_streaks(["BSR"], db_path=bad) == {}


def test_update_streaks_never_raises_on_bad_path(tmp_path):
    bad = str(tmp_path / "missing_parent_dir" / "sub" / "x.duckdb")
    ch.update_streaks(["BSR"], date.today(), db_path=bad)  # must not raise
