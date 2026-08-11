"""Best-effort dispatched_signals backfill from the sentiment paperlog.

Covers the four load-bearing behaviors of the reconstruction core:
  1. Horizon mapping: stored PRIMARY argmax -> T20; derived SECONDARY argmax -> T5.
  2. Matured (CLOSED + historical closed_date) vs still-open (OPEN) branching,
     driven by the REAL today, not the paperlog's log_date.
  3. Dedup skip against existing ledger keys (idempotent re-run).
  4. source='verify' rows excluded (only engine 'daily' dispatches count).

Pure-function tests: ``build_backfill_plan`` takes an injected price-lookup fake,
so no DuckDB / parquet / monkeypatch is needed.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from scripts.backfill_dispatched_signals_from_paperlog import (
    PaperlogRow,
    build_backfill_plan,
    decision_from_probs,
)

_TODAY = date(2026, 7, 16)


def _weekday_sessions(d0: date, n: int) -> list[date]:
    """``n`` consecutive weekday 'sessions' strictly after ``d0``."""
    out, d = [], d0
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


class _FakePriceLookup:
    """Injectable stand-in for ``src.data.price_lookup``.

    ``sessions`` are the full trading calendar (strictly-after filtering mirrors
    the real ``trading_dates_after`` contract). ``closes`` maps ticker â†’ close;
    a missing ticker yields ``None`` (drives the no-price skip path).
    """

    def __init__(self, closes: dict[str, float], sessions: list[date]) -> None:
        self._closes = closes
        self._sessions = sessions

    def close_on_or_before(self, ticker: str, ref_date: Any) -> float | None:
        return self._closes.get(str(ticker).upper())

    def trading_dates_after(self, ref_date: Any) -> list[Any]:
        return [s for s in self._sessions if s > ref_date]


def _row(
    *,
    ticker: str = "HPG",
    log_date: date = date(2026, 6, 1),
    source: str = "daily",
    decision_primary: int | None = 1,
    p_secondary: tuple[float | None, float | None, float | None] = (0.6, 0.3, 0.1),
    entry_close: float | None = 100.0,
) -> PaperlogRow:
    """One paperlog row.

    HORIZON MAPPING (corrected 10-08-26): on `source='daily'` rows the PRIMARY
    horizon is T+20 and the SECONDARY is T+5. So `decision_primary` (the stored
    argmax) drives the **T20** reconstruction and `p_secondary` (whose argmax is
    derived) drives the **T5** one. The backfill had these swapped, which is why
    these tests previously asserted the opposite.
    """
    return PaperlogRow(
        log_date=log_date,
        ticker=ticker,
        source=source,
        decision_primary=decision_primary,
        p_down_secondary=p_secondary[0],
        p_side_secondary=p_secondary[1],
        p_up_secondary=p_secondary[2],
        entry_close=entry_close,
    )


# --- 1. T20 argmax derivation -------------------------------------------------

class TestDecisionFromProbs:
    def test_sell_hold_buy_argmax(self) -> None:
        assert decision_from_probs(0.5, 0.3, 0.2) == 0  # DOWN â†’ SELL
        assert decision_from_probs(0.2, 0.5, 0.3) == 1  # SIDE â†’ HOLD
        assert decision_from_probs(0.2, 0.3, 0.5) == 2  # UP   â†’ BUY

    def test_null_any_returns_none(self) -> None:
        assert decision_from_probs(None, 0.3, 0.5) is None
        assert decision_from_probs(0.2, None, 0.5) is None
        assert decision_from_probs(0.2, 0.3, None) is None

    def test_tie_resolves_to_lowest_index(self) -> None:
        # DOWN and UP tied â†’ lowest index (0 = SELL) wins; never spuriously BUY.
        assert decision_from_probs(0.4, 0.2, 0.4) == 0

    def test_t5_row_planned_via_derived_secondary_argmax(self) -> None:
        # PRIMARY is HOLD (no T20 row), but the SECONDARY (T+5) probs argmax BUY.
        row = _row(decision_primary=1, p_secondary=(0.1, 0.3, 0.6))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(date(2026, 6, 1), 40))
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert stats.t20_found == 0
        assert stats.t5_found == 1 and stats.t5_planned == 1
        assert [p.horizon for p in planned] == [5]
        assert planned[0].hold_days == 5

    def test_t20_row_planned_from_stored_primary_argmax(self) -> None:
        # Mirror image: PRIMARY (T+20) says BUY, SECONDARY says SELL.
        row = _row(decision_primary=2, p_secondary=(0.6, 0.3, 0.1))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(date(2026, 6, 1), 40))
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert stats.t5_found == 0
        assert stats.t20_found == 1 and stats.t20_planned == 1
        assert [p.horizon for p in planned] == [20]
        assert planned[0].hold_days == 30

    def test_t5_null_prob_row_skipped_and_counted(self) -> None:
        row = _row(decision_primary=1, p_secondary=(0.1, None, 0.6))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(date(2026, 6, 1), 40))
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert planned == []
        assert stats.t5_null_prob == 1 and stats.t5_found == 0


# --- 2. Matured (CLOSED) vs still-open (OPEN) branching ------------------------

class TestStatusBranching:
    def test_t5_matured_closed_t20_still_open_same_row(self) -> None:
        # One row that is BOTH a T5 BUY and a T20 BUY. With only 10 sessions
        # elapsed (<= today): T5 (hold 5) matured â†’ CLOSED @ sessions[4];
        # T20 (hold 30) not matured â†’ OPEN.
        d0 = date(2026, 6, 1)
        sessions = _weekday_sessions(d0, 10)
        row = _row(log_date=d0, decision_primary=2, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"HPG": 100.0}, sessions)
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)

        by_h = {p.horizon: p for p in planned}
        assert by_h[5].status == "CLOSED" and by_h[5].closed_date == sessions[4]
        assert by_h[20].status == "OPEN" and by_h[20].closed_date is None
        assert stats.closed_count == 1 and stats.open_count == 1

    def test_t20_matured_closed_date_is_thirtieth_session(self) -> None:
        d0 = date(2026, 6, 1)
        sessions = _weekday_sessions(d0, 40)  # all <= today â†’ T20 (hold 30) matured
        # PRIMARY BUY drives the T20 row; SECONDARY SELL keeps T5 out of it.
        row = _row(log_date=d0, decision_primary=2, p_secondary=(0.7, 0.2, 0.1))
        pl = _FakePriceLookup({"HPG": 100.0}, sessions)
        planned, _ = build_backfill_plan([row], set(), _TODAY, pl)
        assert [p.horizon for p in planned] == [20]
        assert planned[0].status == "CLOSED"
        assert planned[0].closed_date == sessions[29]  # hold_days=30 â†’ 30th session

    def test_still_open_when_fewer_sessions_than_hold(self) -> None:
        # Only 3 sessions elapsed â†’ even T5 (hold 5) is still OPEN.
        d0 = date(2026, 6, 1)
        sessions = _weekday_sessions(d0, 3)
        # SECONDARY BUY drives the T5 row; PRIMARY HOLD keeps T20 out.
        row = _row(log_date=d0, decision_primary=1, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"HPG": 100.0}, sessions)
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert [p.horizon for p in planned] == [5]
        assert planned[0].status == "OPEN" and planned[0].closed_date is None
        assert stats.open_count == 1 and stats.closed_count == 0

    def test_no_price_row_skipped_and_counted(self) -> None:
        # entry_close absent AND no parquet close â†’ no usable t0 â†’ skip.
        row = _row(decision_primary=1, p_secondary=(0.1, 0.2, 0.7), entry_close=None)
        pl = _FakePriceLookup({}, _weekday_sessions(date(2026, 6, 1), 40))
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert planned == []
        assert stats.t5_found == 1 and stats.t5_no_price == 1 and stats.t5_planned == 0

    def test_entry_close_used_as_t0_when_shard_missing(self) -> None:
        # entry_close present (positive) â†’ row still reconstructs even though the
        # parquet close lookup would return None.
        row = _row(decision_primary=1, p_secondary=(0.1, 0.2, 0.7), entry_close=27.5)
        pl = _FakePriceLookup({}, _weekday_sessions(date(2026, 6, 1), 40))
        planned, stats = build_backfill_plan([row], set(), _TODAY, pl)
        assert stats.t5_planned == 1 and planned[0].status == "CLOSED"


# --- 3. Dedup skip ------------------------------------------------------------

class TestDedup:
    def test_existing_key_skips_that_horizon_only(self) -> None:
        d0 = date(2026, 6, 1)
        # Row is BOTH T5 BUY and T20 BUY. Pre-seed only the T5 key as existing.
        row = _row(log_date=d0, decision_primary=2, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(d0, 40))
        existing = {("HPG", d0, 5)}
        planned, stats = build_backfill_plan([row], existing, _TODAY, pl)

        assert stats.t5_dup == 1 and stats.t5_planned == 0
        assert stats.t20_planned == 1
        assert [p.horizon for p in planned] == [20]  # T5 deduped, T20 inserted

    def test_same_key_not_double_planned_within_run(self) -> None:
        # Two identical daily rows (e.g. re-run artifact) â†’ the second is deduped
        # by the in-run key set, not re-planned.
        d0 = date(2026, 6, 1)
        # SECONDARY BUY only â†’ a single T5 row per input row.
        row = _row(log_date=d0, decision_primary=1, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(d0, 40))
        planned, stats = build_backfill_plan([row, row], set(), _TODAY, pl)
        assert stats.t5_planned == 1 and stats.t5_dup == 1
        assert len(planned) == 1


# --- 4. source='verify' exclusion ---------------------------------------------

class TestSourceExclusion:
    def test_verify_rows_excluded(self) -> None:
        d0 = date(2026, 6, 1)
        # A verify row that WOULD qualify as both T5 and T20 BUY â€” must be skipped.
        verify_row = _row(log_date=d0, source="verify", decision_primary=2, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"HPG": 100.0}, _weekday_sessions(d0, 40))
        planned, stats = build_backfill_plan([verify_row], set(), _TODAY, pl)

        assert planned == []
        assert stats.non_daily_skipped == 1
        assert stats.t5_found == 0 and stats.t20_found == 0

    def test_daily_kept_verify_dropped_in_mixed_batch(self) -> None:
        d0 = date(2026, 6, 1)
        # SECONDARY BUY only → the daily row yields a single T5 reconstruction.
        daily = _row(ticker="FPT", log_date=d0, source="daily",
                     decision_primary=1, p_secondary=(0.1, 0.2, 0.7))
        verify = _row(ticker="HPG", log_date=d0, source="verify",
                      decision_primary=2, p_secondary=(0.1, 0.2, 0.7))
        pl = _FakePriceLookup({"FPT": 100.0, "HPG": 100.0}, _weekday_sessions(d0, 40))
        planned, stats = build_backfill_plan([daily, verify], set(), _TODAY, pl)

        assert [p.ticker for p in planned] == ["FPT"]
        assert stats.scanned == 2 and stats.non_daily_skipped == 1
        assert stats.t5_planned == 1
