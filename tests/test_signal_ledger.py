"""Dispatched-signal ledger — record / exit-due / close lifecycle.

The ledger mirrors the tranche book's exit rule: a cohort dispatched on day D
with hold_days=H is due for liquidation once H TRADING sessions (per the fresh
parquet calendar) have elapsed. Uses a tmp DuckDB file + a monkeypatched
trading calendar so no parquet shards are touched.
"""
from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest

from src.trading import signal_ledger

_STRATEGY = {"mode": "tranche", "hold_days": 3, "signal_threshold": 0.43}
_SIGNALS = [
    {"ticker": "HPG", "suggested_weight": 0.0111},
    {"ticker": "FPT", "suggested_weight": 0.0111},
]


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "ledger.duckdb")


def _calendar(d0: date, n: int) -> list[date]:
    """n consecutive weekday 'sessions' strictly after d0."""
    out, d = [], d0
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


class TestRecordDispatch:
    def test_records_open_rows(self, db_path) -> None:
        n = signal_ledger.record_dispatch(
            _SIGNALS, _STRATEGY, horizon=20, db_path=db_path, today=date(2026, 6, 1))
        assert n == 2
        with duckdb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, hold_days, status FROM dispatched_signals ORDER BY ticker"
            ).fetchall()
        assert rows == [("FPT", 3, "OPEN"), ("HPG", 3, "OPEN")]

    def test_idempotent_same_day(self, db_path) -> None:
        d = date(2026, 6, 1)
        assert signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d) == 2
        assert signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d) == 0

    def test_non_tranche_strategy_is_noop(self, db_path) -> None:
        assert signal_ledger.record_dispatch(_SIGNALS, None, 20, db_path) == 0
        assert signal_ledger.record_dispatch(_SIGNALS, {"mode": "grid"}, 20, db_path) == 0
        assert signal_ledger.record_dispatch(_SIGNALS, {"mode": "tranche"}, 20, db_path) == 0


class TestExitsDue:
    def test_due_after_hold_sessions(self, db_path, monkeypatch) -> None:
        d0 = date(2026, 6, 1)  # Monday
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d0)
        sessions = _calendar(d0, 5)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)

        # 2 sessions elapsed < hold 3 → not due
        assert signal_ledger.check_exits_due(db_path, today=sessions[1]) == []
        # 3rd session → due
        due = signal_ledger.check_exits_due(db_path, today=sessions[2])
        assert sorted(d["ticker"] for d in due) == ["FPT", "HPG"]
        assert due[0]["sessions_elapsed"] == 3

    def test_future_sessions_not_counted(self, db_path, monkeypatch) -> None:
        # Calendar knows MORE dates than 'today' (e.g. stale today arg):
        # only sessions <= today may count.
        d0 = date(2026, 6, 1)
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d0)
        sessions = _calendar(d0, 10)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)
        assert signal_ledger.check_exits_due(db_path, today=sessions[0]) == []

    def test_mark_closed_removes_from_due(self, db_path, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d0)
        sessions = _calendar(d0, 5)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)
        due = signal_ledger.check_exits_due(db_path, today=sessions[4])
        assert len(due) == 2
        assert signal_ledger.mark_closed(due, db_path, today=sessions[4]) == 2
        assert signal_ledger.check_exits_due(db_path, today=sessions[4]) == []
        with duckdb.connect(db_path) as conn:
            statuses = {r[0] for r in conn.execute(
                "SELECT status FROM dispatched_signals").fetchall()}
        assert statuses == {"CLOSED"}

    def test_empty_ledger(self, db_path) -> None:
        assert signal_ledger.check_exits_due(db_path) == []
        assert signal_ledger.mark_closed([], db_path) == 0
