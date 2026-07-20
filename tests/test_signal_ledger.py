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


class TestListOpen:
    def test_remaining_sessions(self, db_path, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d0)
        sessions = _calendar(d0, 5)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)
        rows = signal_ledger.list_open(db_path, today=sessions[0])
        assert len(rows) == 2
        assert all(r["sessions_elapsed"] == 1 and r["sessions_remaining"] == 2 for r in rows)

    def test_exits_report_formatting(self, db_path, monkeypatch) -> None:
        from src.utils.telegram_bot import _build_exits_report
        d0 = date(2026, 6, 1)
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=d0)
        sessions = _calendar(d0, 5)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)

        msg = _build_exits_report(signal_ledger.list_open(db_path, today=sessions[0]))
        assert "HPG" in msg and "FPT" in msg
        assert "1/3 phiên" in msg and "còn <b>2</b> phiên" in msg
        assert "[T20]" in msg              # horizon tag on the non-due variant

        due_msg = _build_exits_report(signal_ledger.list_open(db_path, today=sessions[4]))
        assert "ĐẾN HẠN" in due_msg
        assert "[T20]" in due_msg          # horizon tag on the due variant

        assert "Không có vị thế" in _build_exits_report([])


_T20_STRATEGY = {"mode": "tranche", "hold_days": 30}
_T5_STRATEGY = {"mode": "tranche", "hold_days": 5}
_T5_SIGNALS = [{"ticker": "HPG"}, {"ticker": "FPT"}]  # no suggested_weight → 0.0


class TestDualHorizonDispatch:
    def test_t5_and_t20_rows_same_ticker_day_both_persist(self, db_path) -> None:
        d = date(2026, 6, 1)
        assert signal_ledger.record_dispatch(
            _SIGNALS, _T20_STRATEGY, horizon=20, db_path=db_path, today=d) == 2
        assert signal_ledger.record_dispatch(
            _T5_SIGNALS, _T5_STRATEGY, horizon=5, db_path=db_path, today=d) == 2
        with duckdb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, horizon, hold_days, weight "
                "FROM dispatched_signals ORDER BY ticker, horizon"
            ).fetchall()
        assert rows == [
            ("FPT", 5, 5, 0.0), ("FPT", 20, 30, 0.0111),
            ("HPG", 5, 5, 0.0), ("HPG", 20, 30, 0.0111),
        ]

    def test_second_call_same_horizon_still_idempotent(self, db_path) -> None:
        d = date(2026, 6, 1)
        assert signal_ledger.record_dispatch(_SIGNALS, _T20_STRATEGY, 20, db_path, today=d) == 2
        assert signal_ledger.record_dispatch(_SIGNALS, _T20_STRATEGY, 20, db_path, today=d) == 0

    def test_mark_closed_only_closes_matching_horizon(self, db_path, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        signal_ledger.record_dispatch(_SIGNALS, _T20_STRATEGY, 20, db_path, today=d0)
        signal_ledger.record_dispatch(_T5_SIGNALS, _T5_STRATEGY, 5, db_path, today=d0)
        sessions = _calendar(d0, 10)
        monkeypatch.setattr(
            signal_ledger.price_lookup, "trading_dates_after", lambda ref, conn=None: sessions)
        # 5 sessions elapsed → T5 (hold=5) due, T20 (hold=30) not due.
        due = signal_ledger.check_exits_due(db_path, today=sessions[4])
        assert {d["ticker"] for d in due} == {"HPG", "FPT"}
        assert all(d["hold_days"] == 5 and d["horizon"] == 5 for d in due)
        assert signal_ledger.mark_closed(due, db_path, today=sessions[4]) == 2
        with duckdb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT horizon, status FROM dispatched_signals ORDER BY horizon, status"
            ).fetchall()
        # Only the two T5 rows closed; both T20 rows remain OPEN.
        assert rows == [(5, "CLOSED"), (5, "CLOSED"), (20, "OPEN"), (20, "OPEN")]


class TestEvaluateSignalPnl:
    # Every test monkeypatches `closes_between` (mandatory — the CA-gap retrofit
    # calls it on the success path; without a stub these tests would silently do
    # real parquet I/O against data/ohlcv_HPG.parquet, which exists on disk).
    def test_open_provisional(self, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 2)  # only 2 sessions elapsed < hold 5
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: 100.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 110.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 105.0, 110.0])
        out = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=5, today=sessions[1])
        assert out["matured"] is False
        assert out["t_exit"] == 110.0
        assert out["pct"] == pytest.approx(9.70)  # (110-100)/100*100 - 0.30
        assert out["gap_flag"] is False
        assert out["adjustment_factor"] is None

    def test_matured_exit_price(self, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]  # hold_days=3 → 3rd session
        prices = {d0: 100.0, exit_date: 130.0}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)  # must NOT be used
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 109.0, 119.0, 130.0])
        out = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=3, today=sessions[4])
        assert out["matured"] is True
        assert out["t_exit"] == 130.0
        assert out["pct"] == pytest.approx(29.70)  # (130-100)/100*100 - 0.30
        assert out["gap_flag"] is False
        assert out["adjustment_factor"] is None

    def test_missing_price_returns_error(self, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: None)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: None)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [])
        out = signal_ledger.evaluate_signal_pnl("HPG", date(2026, 6, 1), hold_days=5)
        # Error path returns BEFORE the gap check — no gap_flag key by design.
        assert "error" in out and "pct" not in out and "gap_flag" not in out
        assert "adjustment_factor" not in out

    def test_t0_zero_or_negative_returns_error(self, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: 0.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 50.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [])
        out = signal_ledger.evaluate_signal_pnl("HPG", date(2026, 6, 1), hold_days=5)
        assert "error" in out and "gap_flag" not in out
        assert "adjustment_factor" not in out

    def test_gap_flag_true_reproduces_pvd_case(self, monkeypatch) -> None:
        # PVD 2026-07-09→07-10: 33.3 → 19.47 VND overnight (66.9% stock-dividend
        # ex-rights). Pre-retrofit this booked a fake −41% "loss"; the retrofit
        # now AUTO-ADJUSTS it (T0 rebased by the observed gap ratio) to a trusted,
        # near-zero number.
        d0 = date(2026, 7, 9)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]  # hold_days=3 → matured
        prices = {d0: 33.3, exit_date: 19.47}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)  # must NOT be used
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [33.3, 19.47])
        out = signal_ledger.evaluate_signal_pnl("PVD", d0, hold_days=3, today=sessions[4])
        assert out["gap_flag"] is True
        assert out["matured"] is True
        assert out["adjustment_factor"] == pytest.approx(19.47 / 33.3)
        # Corrected — near zero (the round-trip cost only), NOT the pre-fix fake
        # -41.3%/-41.8% "loss". t0 is rebased to 33.3 * (19.47/33.3) == 19.47, so
        # the underlying move nets to flat once adjusted.
        assert out["pct"] == pytest.approx(-0.30)

    def test_gap_flag_false_normal_move(self, monkeypatch) -> None:
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 2)  # provisional (< hold 5)
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: 100.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 105.0)  # +5% mild move
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 102.0, 105.0])
        out = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=5, today=sessions[1])
        assert out["gap_flag"] is False
        assert out.get("adjustment_factor") is None

    def test_multi_gap_composition(self, monkeypatch) -> None:
        # Two CA resets (0.6 each) with a +5% organic move between them compose
        # to a cumulative factor 0.36. t0_eff = 100*0.36 = 36.0; gross (37.8-36)/
        # 36*100 = 5.0%; NET 5.0 - 0.30 = 4.70% (plan Decision 2 worked example).
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]  # hold_days=3 → matured
        prices = {d0: 100.0, exit_date: 37.8}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)  # must NOT be used
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 60.0, 63.0, 37.8])
        out = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=3, today=sessions[4])
        assert out["gap_flag"] is True
        assert out["adjustment_factor"] == pytest.approx(0.36)
        assert out["pct"] == pytest.approx(4.70)

    def test_window_bounds_scoped_to_matured_exit_date(self, monkeypatch) -> None:
        # closes_between must be called with (entry_date, exit_date) — NOT
        # (entry_date, today) — so a gap dated after the true exit but before
        # today is structurally excluded from the matured window.
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]  # hold_days=3 → matured
        prices = {d0: 100.0, exit_date: 103.0}
        calls: list[tuple] = []

        def _cb(t, s, e, conn=None):
            calls.append((s, e))
            return [100.0, 101.0, 102.0, 103.0]  # flat, no gap

        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between", _cb)
        out = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=3, today=sessions[4])
        assert calls == [(d0, exit_date)]  # NOT (d0, today)
        assert out["gap_flag"] is False
        assert out["adjustment_factor"] is None

    def test_window_bounds_scoped_to_today_when_open(self, monkeypatch) -> None:
        # Provisional/open path: closes_between is scoped to (entry_date, today).
        d0 = date(2026, 6, 1)
        sessions = _calendar(d0, 5)
        calls: list[tuple] = []

        def _cb(t, s, e, conn=None):
            calls.append((s, e))
            return [100.0, 101.0]  # flat, no gap

        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: 100.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 101.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between", _cb)
        # today = sessions[1] < hold_days=5 → open/provisional.
        signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=5, today=sessions[1])
        assert calls == [(d0, sessions[1])]


class TestListClosedSince:
    def _seed_closed(self, db_path, ticker, d0, closed_date, horizon=20, hold=30) -> None:
        signal_ledger.record_dispatch(
            [{"ticker": ticker}], {"mode": "tranche", "hold_days": hold},
            horizon, db_path, today=d0)
        with duckdb.connect(db_path) as conn:
            conn.execute(
                "UPDATE dispatched_signals SET status='CLOSED', closed_date=? "
                "WHERE ticker=? AND dispatch_date=?", [closed_date, ticker, d0])

    def test_returns_todays_closure(self, db_path) -> None:
        today = date(2026, 6, 10)
        self._seed_closed(db_path, "HPG", date(2026, 6, 1), today)
        out = signal_ledger.list_closed_since(7, today, db_path)
        assert [r["ticker"] for r in out] == ["HPG"]
        assert out[0]["horizon"] == 20 and out[0]["hold_days"] == 30

    def test_includes_closure_inside_window(self, db_path) -> None:
        today = date(2026, 6, 10)
        # closed 3 days ago → inside a 7-day window → included
        self._seed_closed(db_path, "HPG", date(2026, 6, 1), date(2026, 6, 7))
        out = signal_ledger.list_closed_since(7, today, db_path)
        assert [r["ticker"] for r in out] == ["HPG"]

    def test_excludes_closure_outside_window(self, db_path) -> None:
        today = date(2026, 6, 10)
        # closed 10 days ago → outside a 7-day window → excluded
        self._seed_closed(db_path, "HPG", date(2026, 5, 20), date(2026, 5, 31))
        assert signal_ledger.list_closed_since(7, today, db_path) == []

    def test_excludes_open_rows(self, db_path) -> None:
        signal_ledger.record_dispatch(_SIGNALS, _STRATEGY, 20, db_path, today=date(2026, 6, 1))
        assert signal_ledger.list_closed_since(7, date(2026, 6, 10), db_path) == []

    def test_excludes_future_closed_date(self, db_path) -> None:
        today = date(2026, 6, 10)
        # closed_date after today → excluded by the inclusive upper bound
        self._seed_closed(db_path, "HPG", date(2026, 6, 1), date(2026, 6, 11))
        assert signal_ledger.list_closed_since(7, today, db_path) == []


class TestRssDedupe:
    def test_title_dedupe_across_sources(self) -> None:
        from src.crawlers.sentiment_crawler import NewsItem, SentimentCrawler
        items = [
            NewsItem(date=date(2026, 6, 1), title="HPG tăng trần phiên sáng",
                     url="https://vietstock.vn/abc", text="t"),
            NewsItem(date=date(2026, 6, 1), title="HPG tăng trần phiên sáng - Vietstock",
                     url="https://news.google.com/xyz", text="t"),
            NewsItem(date=date(2026, 6, 1), title="Tin khác hoàn toàn",
                     url="https://cafef.vn/def", text="t"),
        ]
        out = SentimentCrawler._dedupe(items)
        assert len(out) == 2
        assert {i.url for i in out} == {"https://vietstock.vn/abc", "https://cafef.vn/def"}


_CANCELLED = [
    {"ticker": "HPG", "p_down": 0.5, "p_side": 0.3, "p_up": 0.2, "reason": "Cửa tăng thấp"},
    {"ticker": "FPT", "p_down": 0.4, "p_side": 0.4, "p_up": 0.2, "reason": "Trọng tài từ chối"},
]


class TestCancelledSignalsLedger:
    def test_record_cancelled_inserts_rows(self, db_path) -> None:
        n = signal_ledger.record_cancelled(
            _CANCELLED, horizon=20, hold_days=30, db_path=db_path, today=date(2026, 7, 9))
        assert n == 2
        with duckdb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, horizon, hold_days, p_up, reason "
                "FROM cancelled_signals ORDER BY ticker"
            ).fetchall()
        assert rows == [
            ("FPT", 20, 30, 0.2, "Trọng tài từ chối"),
            ("HPG", 20, 30, 0.2, "Cửa tăng thấp"),
        ]

    def test_record_cancelled_dedup_same_day_ticker_horizon(self, db_path) -> None:
        d = date(2026, 7, 9)
        assert signal_ledger.record_cancelled(_CANCELLED, 20, 30, db_path, today=d) == 2
        assert signal_ledger.record_cancelled(_CANCELLED, 20, 30, db_path, today=d) == 0

    def test_record_cancelled_different_horizons_same_day_both_persist(self, db_path) -> None:
        # A ticker rejected by BOTH the T5 and T20 screen on the same day yields
        # 2 independent rows (mirrors TestDualHorizonDispatch).
        d = date(2026, 7, 9)
        assert signal_ledger.record_cancelled(_CANCELLED, 5, 5, db_path, today=d) == 2
        assert signal_ledger.record_cancelled(_CANCELLED, 20, 30, db_path, today=d) == 2
        with duckdb.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, horizon, hold_days FROM cancelled_signals "
                "ORDER BY ticker, horizon"
            ).fetchall()
        assert rows == [
            ("FPT", 5, 5), ("FPT", 20, 30),
            ("HPG", 5, 5), ("HPG", 20, 30),
        ]

    def test_record_cancelled_never_raises(self, db_path, monkeypatch) -> None:
        def _boom(*a, **k):
            raise RuntimeError("db down")
        monkeypatch.setattr(signal_ledger, "_connect", _boom)
        assert signal_ledger.record_cancelled(
            _CANCELLED, 20, 30, db_path, today=date(2026, 7, 9)) == 0


class TestListCancelledSince:
    def _seed(self, db_path, ticker, screen_date, horizon=20, hold=30) -> None:
        signal_ledger.record_cancelled(
            [{"ticker": ticker, "p_down": 0.5, "p_side": 0.3, "p_up": 0.2,
              "reason": "x"}],
            horizon, hold, db_path, today=screen_date)

    def test_returns_todays_screen(self, db_path, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        today = date(2026, 7, 10)
        self._seed(db_path, "HPG", today)
        out = signal_ledger.list_cancelled_since(7, today, db_path)
        assert [r["ticker"] for r in out] == ["HPG"]
        assert out[0]["horizon"] == 20 and out[0]["hold_days"] == 30

    def test_includes_inside_window(self, db_path, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        today = date(2026, 7, 10)
        self._seed(db_path, "HPG", date(2026, 7, 7))  # 3 days ago → inside 7-day window
        out = signal_ledger.list_cancelled_since(7, today, db_path)
        assert [r["ticker"] for r in out] == ["HPG"]

    def test_excludes_outside_window(self, db_path, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        today = date(2026, 7, 10)
        self._seed(db_path, "HPG", date(2026, 6, 30))  # 10 days ago → outside window
        assert signal_ledger.list_cancelled_since(7, today, db_path) == []

    def test_excludes_future_screen_date(self, db_path, monkeypatch) -> None:
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: [])
        today = date(2026, 7, 10)
        self._seed(db_path, "HPG", date(2026, 7, 11))  # future → excluded by upper bound
        assert signal_ledger.list_cancelled_since(7, today, db_path) == []


class TestEvaluateRegretPnl:
    def test_open_provisional(self, monkeypatch) -> None:
        d0 = date(2026, 7, 9)
        sessions = _calendar(d0, 2)  # provisional (< hold 5)
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: 100.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 110.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 105.0, 110.0])
        out = signal_ledger.evaluate_regret_pnl("HPG", d0, hold_days=5, today=sessions[1])
        assert out["matured"] is False
        assert out["t_exit"] == 110.0
        assert out["pct"] == pytest.approx(10.0)   # GROSS: (110-100)/100*100, NO cost
        assert out["gap_flag"] is False
        assert out["adjustment_factor"] is None

    def test_matured_exit_price(self, monkeypatch) -> None:
        d0 = date(2026, 7, 9)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]  # hold_days=3
        prices = {d0: 100.0, exit_date: 130.0}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)  # must NOT be used
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 109.0, 119.0, 130.0])
        out = signal_ledger.evaluate_regret_pnl("HPG", d0, hold_days=3, today=sessions[4])
        assert out["matured"] is True
        assert out["pct"] == pytest.approx(30.0)   # GROSS: (130-100)/100*100, NO cost
        assert out["gap_flag"] is False
        assert out["adjustment_factor"] is None

    def test_gross_no_cost_deduction(self, monkeypatch) -> None:
        # Identical price path fed to BOTH evaluators — the regret (GROSS) pct
        # must be EXACTLY _VN_ROUND_TRIP_COST_PCT pp higher than the signal (NET).
        d0 = date(2026, 7, 9)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]
        prices = {d0: 100.0, exit_date: 130.0}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [100.0, 109.0, 119.0, 130.0])
        net = signal_ledger.evaluate_signal_pnl("HPG", d0, hold_days=3, today=sessions[4])
        gross = signal_ledger.evaluate_regret_pnl("HPG", d0, hold_days=3, today=sessions[4])
        assert gross["pct"] - net["pct"] == pytest.approx(signal_ledger._VN_ROUND_TRIP_COST_PCT)

    def test_gap_flag_propagates(self, monkeypatch) -> None:
        # Same PVD-style fixture as Phase 1 — the shared _evaluate_pnl gap guard
        # must fire through the regret path too.
        d0 = date(2026, 7, 9)
        sessions = _calendar(d0, 5)
        exit_date = sessions[2]
        prices = {d0: 33.3, exit_date: 19.47}
        monkeypatch.setattr(signal_ledger.price_lookup, "close_on_or_before",
                            lambda t, d, conn=None: prices[d])
        monkeypatch.setattr(signal_ledger.price_lookup, "trading_dates_after",
                            lambda ref, conn=None: sessions)
        monkeypatch.setattr(signal_ledger.price_lookup, "latest_close",
                            lambda t, conn=None: 999.0)
        monkeypatch.setattr(signal_ledger.price_lookup, "closes_between",
                            lambda t, s, e, conn=None: [33.3, 19.47])
        out = signal_ledger.evaluate_regret_pnl("PVD", d0, hold_days=3, today=sessions[4])
        assert out["gap_flag"] is True
        assert out["adjustment_factor"] == pytest.approx(19.47 / 33.3)
        # GROSS — no cost deduction, so the corrected pct lands exactly at 0.0
        # (NET side lands at -0.30 due to the round-trip cost).
        assert out["pct"] == pytest.approx(0.0)
