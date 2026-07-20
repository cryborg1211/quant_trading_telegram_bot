"""Best-effort reconstruction of historical ``dispatched_signals`` ledger rows.

The EOD dual-horizon position report (``eod-position-report_PLAN_16-07-26``) reads
the ``dispatched_signals`` ledger (``src/trading/signal_ledger.py``), but that
table currently has ZERO rows ever written — the live ``record_dispatch`` path
only starts accruing rows from the report's ship date forward. This script seeds
that table with a PROXY history so the report has something to show on day one
instead of starting empty.

⚠️ THIS IS A PROXY, NOT A VERIFIED DISPATCH RECORD.
``sentiment_entry_paperlog`` logs EVERY scored candidate in the universe each day
(``source='daily'``), not just the Top-N names that were actually dispatched to
the user. Filtering to ``source='daily' AND decision == BUY`` is the closest
available approximation of "the engine suggested this ticker" — but it is not a
ground-truth record of what was broadcast. Only ``source='daily'`` rows count;
``source='verify'`` rows are user-pulled ``/verify`` replies (not engine
dispatches) and are excluded.

Reconstruction rule (per ``source='daily'`` paperlog row):
  • T5 candidate  — ``decision_5d == 2`` (BUY). horizon=5, hold_days=5 (mirrors
    the live T5-tracking synthetic ``{"mode":"tranche","hold_days":5}`` in
    main.py — NOT tied to any bot strategy artifact).
  • T20 candidate — the paperlog stores NO ``decision_20d`` column, so it is
    derived here as ``argmax(p_down_20d, p_side_20d, p_up_20d)`` (NULL-safe:
    any NULL prob → skip). Only rows where that argmax == 2 (BUY). horizon=20,
    hold_days=30 (the REAL live production convention — the T+20 GOLDEN artifact
    is backtested with the tranche book's ``--hold-days 30``; see main.py's T20
    dispatch call site and Decision 3 of the EOD-report plan).

Decision encoding (shared with the arbitrator/paperlog): argmax over
``[DOWN, SIDE, UP]`` → ``0=SELL, 1=HOLD, 2=BUY`` (the argmax index IS the code).

Each candidate is reconstructed with a historically-correct status/closed_date so
today's EOD report does NOT wrongly show weeks of backlog under "ĐÃ ĐÓNG HÔM NAY":
a cohort whose ``hold_days`` trading sessions have already elapsed (as of the REAL
today, not the paperlog's log_date) is written ``status='CLOSED'`` with
``closed_date`` = the actual historical maturity session; otherwise ``status='OPEN'``.
All backfilled rows carry ``weight = 0.0`` (paper-tracking only — these were never
real capital allocations, matching the live T5-tracking convention). Dedup key
``(ticker, dispatch_date, horizon)`` makes re-runs idempotent, matching
``record_dispatch``'s own convention.

Usage (dry-run by default — prints what WOULD be inserted, writes nothing):
    python scripts/backfill_dispatched_signals_from_paperlog.py
    python scripts/backfill_dispatched_signals_from_paperlog.py --commit   # INSERT
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import namedtuple
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import CONFIG
from src.data import price_lookup as _price_lookup
from src.trading import signal_ledger

LOGGER = logging.getLogger(__name__)

_PAPERLOG_TABLE = "sentiment_entry_paperlog"
_DAILY_SOURCE = "daily"
_BUY = 2  # decision code: argmax over [DOWN, SIDE, UP] == index 2 (UP) → BUY

# Horizon → hold_days conventions (see module docstring).
_T5_HORIZON, _T5_HOLD_DAYS = 5, 5
_T20_HORIZON, _T20_HOLD_DAYS = 20, 30

# Columns pulled from the paperlog, in fetch order.
PaperlogRow = namedtuple(
    "PaperlogRow",
    "log_date ticker source decision_5d p_down_20d p_side_20d p_up_20d entry_close",
)


@dataclass(frozen=True)
class PlannedRow:
    """One reconstructed ``dispatched_signals`` row ready to INSERT."""

    ticker: str
    dispatch_date: date
    horizon: int
    hold_days: int
    status: str
    closed_date: date | None
    weight: float = 0.0


@dataclass
class BackfillStats:
    """Dry-run / commit summary counters."""

    scanned: int = 0
    non_daily_skipped: int = 0
    # Per-horizon: candidates found (BUY) / planned to insert / skipped-duplicate /
    # skipped-no-price. T20 additionally: rows dropped because a 20d prob was NULL.
    t5_found: int = 0
    t5_planned: int = 0
    t5_dup: int = 0
    t5_no_price: int = 0
    t20_found: int = 0
    t20_planned: int = 0
    t20_dup: int = 0
    t20_no_price: int = 0
    t20_null_prob: int = 0
    open_count: int = 0
    closed_count: int = 0


class _PriceLookup(Protocol):
    def close_on_or_before(self, ticker: str, ref_date: Any) -> float | None: ...
    def trading_dates_after(self, ref_date: Any) -> list[Any]: ...


def decision_from_probs(
    p_down: float | None, p_side: float | None, p_up: float | None
) -> int | None:
    """Argmax over ``[DOWN, SIDE, UP]`` → decision code (0/1/2), or None if any
    probability is NULL. The argmax index equals the decision code by construction
    (``0=SELL/DOWN, 1=HOLD/SIDE, 2=BUY/UP``). Ties resolve to the lowest index."""
    if p_down is None or p_side is None or p_up is None:
        return None
    probs = (float(p_down), float(p_side), float(p_up))
    return int(max(range(3), key=probs.__getitem__))


def _resolve_status(
    ticker: str,
    log_date: date,
    entry_close: float | None,
    hold_days: int,
    today: date,
    pl: _PriceLookup,
    sessions_cache: dict[date, list[Any]],
) -> tuple[str, date | None] | None:
    """Historically-correct ``(status, closed_date)`` for one candidate, or None
    when the row must be skipped (missing/non-positive t0 → no usable anchor).

    t0 = ``entry_close`` if present in the paperlog row, else the parquet
    close-on-or-before ``log_date``. Sessions are counted on the fresh parquet
    trading calendar, strictly after ``log_date`` and up to the REAL ``today``.
    Never raises — a price-lookup miss degrades to a skip (None), never a crash.
    """
    try:
        t0 = entry_close if entry_close is not None else pl.close_on_or_before(ticker, log_date)
        if t0 is None or float(t0) <= 0.0:
            return None

        if log_date not in sessions_cache:
            sessions_cache[log_date] = [
                s for s in pl.trading_dates_after(log_date) if s <= today
            ]
        sessions = sessions_cache[log_date]

        if len(sessions) >= hold_days:
            return "CLOSED", sessions[hold_days - 1]
        return "OPEN", None
    except Exception as exc:  # noqa: BLE001 — best-effort backfill, never crash on one row
        LOGGER.warning("[backfill-dispatch] status resolve failed for %s @ %s: %s",
                       ticker, log_date, exc)
        return None


def _plan_one_horizon(
    ticker: str,
    log_date: date,
    entry_close: float | None,
    horizon: int,
    hold_days: int,
    existing_keys: set[tuple[str, date, int]],
    today: date,
    pl: _PriceLookup,
    sessions_cache: dict[date, list[Any]],
) -> tuple[PlannedRow | None, str]:
    """Build a ``PlannedRow`` for one (already-confirmed-BUY) horizon candidate.

    Returns ``(row_or_None, outcome)`` where outcome is one of
    ``"planned" | "dup" | "no_price"``. Dedup runs before the price probe so an
    already-present row costs no parquet I/O (matches ``record_dispatch`` order).
    """
    key = (ticker, log_date, horizon)
    if key in existing_keys:
        return None, "dup"

    resolved = _resolve_status(ticker, log_date, entry_close, hold_days, today, pl, sessions_cache)
    if resolved is None:
        return None, "no_price"

    status, closed_date = resolved
    existing_keys.add(key)  # in-run dedup: don't double-plan the same key
    return PlannedRow(
        ticker=ticker,
        dispatch_date=log_date,
        horizon=horizon,
        hold_days=hold_days,
        status=status,
        closed_date=closed_date,
    ), "planned"


def build_backfill_plan(
    rows: Iterable[PaperlogRow],
    existing_keys: set[tuple[str, date, int]],
    today: date,
    pl: _PriceLookup,
) -> tuple[list[PlannedRow], BackfillStats]:
    """Pure planning core: paperlog rows → reconstructed ledger rows + counts.

    ``existing_keys`` is the set of ``(ticker, dispatch_date, horizon)`` tuples
    already present in ``dispatched_signals`` — candidates matching an existing
    key are skipped (idempotent re-runs). ``today`` is the REAL current date (NOT
    a paperlog log_date), used to decide which historical cohorts have matured.

    Non-``daily`` source rows (e.g. ``source='verify'``) are excluded — those are
    user-pulled ``/verify`` replies, never engine dispatches.
    """
    planned: list[PlannedRow] = []
    stats = BackfillStats()
    sessions_cache: dict[date, list[Any]] = {}

    for row in rows:
        stats.scanned += 1
        if row.source != _DAILY_SOURCE:
            stats.non_daily_skipped += 1
            continue

        ticker = str(row.ticker).upper().strip()
        log_date = row.log_date

        # --- T5 candidate: stored decision_5d == BUY ---------------------------
        if row.decision_5d == _BUY:
            stats.t5_found += 1
            planned_row, outcome = _plan_one_horizon(
                ticker, log_date, row.entry_close, _T5_HORIZON, _T5_HOLD_DAYS,
                existing_keys, today, pl, sessions_cache)
            if outcome == "planned" and planned_row is not None:
                stats.t5_planned += 1
                planned.append(planned_row)
            elif outcome == "dup":
                stats.t5_dup += 1
            else:  # no_price
                stats.t5_no_price += 1

        # --- T20 candidate: derived argmax over the 20d probs == BUY -----------
        decision_20d = decision_from_probs(row.p_down_20d, row.p_side_20d, row.p_up_20d)
        if decision_20d is None:
            stats.t20_null_prob += 1
        elif decision_20d == _BUY:
            stats.t20_found += 1
            planned_row, outcome = _plan_one_horizon(
                ticker, log_date, row.entry_close, _T20_HORIZON, _T20_HOLD_DAYS,
                existing_keys, today, pl, sessions_cache)
            if outcome == "planned" and planned_row is not None:
                stats.t20_planned += 1
                planned.append(planned_row)
            elif outcome == "dup":
                stats.t20_dup += 1
            else:  # no_price
                stats.t20_no_price += 1

    stats.open_count = sum(1 for r in planned if r.status == "OPEN")
    stats.closed_count = sum(1 for r in planned if r.status == "CLOSED")
    return planned, stats


def fetch_paperlog_rows(conn: Any) -> list[PaperlogRow]:
    """All paperlog rows (any source) needed for reconstruction, ordered
    oldest-first. Source filtering happens in ``build_backfill_plan`` so the
    non-daily-excluded count is reported honestly."""
    raw = conn.execute(
        f"SELECT log_date, ticker, source, decision_5d, "
        f"p_down_20d, p_side_20d, p_up_20d, entry_close "
        f"FROM {_PAPERLOG_TABLE} ORDER BY log_date, ticker"
    ).fetchall()
    return [PaperlogRow(*r) for r in raw]


def fetch_existing_keys(conn: Any) -> set[tuple[str, date, int]]:
    """Existing ``(ticker, dispatch_date, horizon)`` keys already in the ledger,
    normalized to match reconstruction (uppercased ticker). Enables idempotent
    re-runs."""
    signal_ledger.ensure_table(conn)
    rows = conn.execute(
        f"SELECT ticker, dispatch_date, horizon FROM {signal_ledger.TABLE}"
    ).fetchall()
    return {
        (str(r[0]).upper().strip(), r[1], int(r[2]) if r[2] is not None else None)
        for r in rows
    }


def insert_planned_rows(conn: Any, planned: list[PlannedRow]) -> int:
    """INSERT the planned rows into ``dispatched_signals``. Returns rows written."""
    if not planned:
        return 0
    signal_ledger.ensure_table(conn)
    conn.executemany(
        f"INSERT INTO {signal_ledger.TABLE} "
        "(ticker, dispatch_date, horizon, hold_days, weight, status, closed_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (p.ticker, p.dispatch_date, p.horizon, p.hold_days,
             p.weight, p.status, p.closed_date)
            for p in planned
        ],
    )
    return len(planned)


def _print_summary(stats: BackfillStats, planned: list[PlannedRow], committed: bool) -> None:
    mode = "COMMITTED" if committed else "DRY-RUN (no writes — pass --commit)"
    LOGGER.info("[backfill-dispatch] %s", mode)
    LOGGER.info("  paperlog rows scanned: %d  (non-daily excluded: %d)",
                stats.scanned, stats.non_daily_skipped)
    LOGGER.info("  T5  (decision_5d==BUY):  found=%d inserted=%d dup=%d no_price=%d",
                stats.t5_found, stats.t5_planned, stats.t5_dup, stats.t5_no_price)
    LOGGER.info("  T20 (argmax_20d==BUY):   found=%d inserted=%d dup=%d no_price=%d "
                "(null_prob_skipped=%d)",
                stats.t20_found, stats.t20_planned, stats.t20_dup, stats.t20_no_price,
                stats.t20_null_prob)
    LOGGER.info("  planned rows: total=%d  OPEN=%d  CLOSED=%d",
                len(planned), stats.open_count, stats.closed_count)

    if planned:
        LOGGER.info("  sample (first 10 planned rows):")
        for p in planned[:10]:
            closed = p.closed_date.isoformat() if p.closed_date is not None else "-"
            LOGGER.info("    %-8s %s  T%-2d hold=%-2d %-6s closed=%s",
                        p.ticker, p.dispatch_date.isoformat(), p.horizon,
                        p.hold_days, p.status, closed)


def run(db_path: str, *, commit: bool, today: date | None = None) -> BackfillStats:
    """Orchestration: fetch → plan → summarize → (optionally) insert."""
    today = today or datetime.now().date()
    with duckdb.connect(db_path) as conn:
        rows = fetch_paperlog_rows(conn)
        existing = fetch_existing_keys(conn)
        planned, stats = build_backfill_plan(rows, existing, today, _price_lookup)
        _print_summary(stats, planned, committed=commit)
        if commit and planned:
            written = insert_planned_rows(conn, planned)
            LOGGER.info("[backfill-dispatch] INSERTED %d rows into %s.",
                        written, signal_ledger.TABLE)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(CONFIG.paths.duckdb_path), help="DuckDB path.")
    parser.add_argument("--commit", action="store_true",
                        help="Write INSERTs (else dry-run — the default).")
    args = parser.parse_args()
    run(args.db, commit=args.commit)


if __name__ == "__main__":
    main()
