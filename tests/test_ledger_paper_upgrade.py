"""A paper row must not block the real dispatch of the same name that day.

LIVE FAILURE (27-08-26). A /suggest tap at 08:47 wrote a paper row for GVR. That
evening's cron dispatched GVR for real - cards went to both chats - but the plain
(ticker, horizon) dedup saw the paper row and SKIPPED the insert:

    [SignalLedger] Recorded 2 OPEN signals   <- three cards had just been sent

GVR then sat in the book as paper: excluded from the real position report and from
tranche exit alerts, so the operator held a name the system would never tell them
to close. Introduced the same morning by the is_paper fix, which stopped previews
vetoing the cron but left them able to swallow it.
"""
from __future__ import annotations

from datetime import date

import duckdb
import pytest

from src.trading import signal_ledger

_STRAT = {"mode": "tranche", "hold_days": 30}
_DAY = date(2026, 8, 27)


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "ledger.duckdb")


def _rows(db_path):
    con = duckdb.connect(db_path)
    out = con.execute(
        "SELECT ticker, horizon, weight, hold_days, COALESCE(is_paper, FALSE) "
        "FROM dispatched_signals ORDER BY ticker, horizon"
    ).fetchall()
    con.close()
    return out


def test_real_dispatch_upgrades_an_existing_paper_row(db):
    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.014690}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=True)
    n = signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.004665}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=False)

    rows = _rows(db)
    assert len(rows) == 1, "must upgrade in place, not duplicate the cohort"
    ticker, horizon, weight, _hold, is_paper = rows[0]
    assert (ticker, horizon) == ("GVR", 20)
    assert is_paper is False, "the cron dispatch makes it a real position"
    assert weight == pytest.approx(0.004665), "weight must become the cron's, not the preview's"
    assert n == 1, "an upgrade counts as recorded — the log must not say 0"


def test_upgraded_row_reaches_open_tickers(db):
    """The point of the upgrade: it must now veto re-buying and be trackable."""
    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.01}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=True)
    assert "GVR" not in signal_ledger.open_tickers(db_path=db)

    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.004665}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=False)
    assert "GVR" in signal_ledger.open_tickers(db_path=db)


def test_a_preview_never_demotes_a_real_cohort(db):
    """Reverse direction must NOT happen — a /suggest tap after the cron ran."""
    signal_ledger.record_dispatch(
        [{"ticker": "PVT", "suggested_weight": 0.009330}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=False)
    signal_ledger.record_dispatch(
        [{"ticker": "PVT", "suggested_weight": 0.99}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=True)

    rows = _rows(db)
    assert len(rows) == 1
    assert rows[0][4] is False, "a preview must not turn a held position into paper"
    assert rows[0][2] == pytest.approx(0.009330), "nor rewrite its weight"


def test_repeated_real_dispatch_is_still_idempotent(db):
    """The original guarantee: a same-day re-run cannot double-book."""
    for _ in range(3):
        signal_ledger.record_dispatch(
            [{"ticker": "VIX", "suggested_weight": 0.004665}], _STRAT, 20,
            db_path=db, today=_DAY, is_paper=False)
    assert len(_rows(db)) == 1


def test_repeated_preview_is_still_idempotent(db):
    for _ in range(3):
        signal_ledger.record_dispatch(
            [{"ticker": "VIX", "suggested_weight": 0.01}], _STRAT, 20,
            db_path=db, today=_DAY, is_paper=True)
    rows = _rows(db)
    assert len(rows) == 1 and rows[0][4] is True


def test_horizons_stay_independent_across_the_upgrade(db):
    """T+5 tracking row and the T+20 cohort are separate keys."""
    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.01}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=True)
    signal_ledger.record_dispatch(
        [{"ticker": "GVR"}], {"mode": "tranche", "hold_days": 5}, 5,
        db_path=db, today=_DAY, is_paper=False)
    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.004665}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=False)

    rows = _rows(db)
    assert len(rows) == 2
    by_h = {r[1]: r for r in rows}
    assert by_h[20][4] is False and by_h[20][2] == pytest.approx(0.004665)
    assert by_h[5][4] is False


def test_mixed_batch_inserts_and_upgrades_together(db):
    """The real shape of the 27-08 run: one held-as-paper name, two new."""
    signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.0147}], _STRAT, 20,
        db_path=db, today=_DAY, is_paper=True)
    n = signal_ledger.record_dispatch(
        [{"ticker": "GVR", "suggested_weight": 0.004665},
         {"ticker": "PVT", "suggested_weight": 0.009330},
         {"ticker": "VIX", "suggested_weight": 0.004665}],
        _STRAT, 20, db_path=db, today=_DAY, is_paper=False)

    assert n == 3, "all three cards sent => all three recorded"
    assert signal_ledger.open_tickers(db_path=db) == {"GVR", "PVT", "VIX"}
