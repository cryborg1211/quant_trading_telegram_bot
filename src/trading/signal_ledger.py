"""Dispatched-signal ledger — closes the tranche entry/exit loop on the serve path.

The backtest's tranche book holds every cohort exactly ``hold_days`` TRADING
sessions, then liquidates at ATC. The bot dispatches the entries but (before
this module) never alerted the exits, so live behavior silently diverged from
the simulated strategy after the first horizon elapsed.

Flow:
    1. ``record_dispatch``  — called by ``run_trade_execution`` after a broadcast
       dispatch; one OPEN row per (ticker, dispatch_date).
    2. ``check_exits_due``  — called daily by ``full_pipeline``; an OPEN row is
       due once ``hold_days`` trading sessions (per the fresh parquet calendar,
       NOT calendar days) have elapsed since dispatch.
    3. ``mark_closed``      — flips rows to CLOSED after the exit alert is sent.

Storage: ``dispatched_signals`` table in the core DuckDB file. All writes go
through short-lived connections (same convention as the sentiment crawler — no
mixed-config handles to the same file).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import duckdb  # type: ignore[import-untyped]

from config.settings import CONFIG
from src.data import price_lookup

LOGGER = logging.getLogger(__name__)

TABLE = "dispatched_signals"


def _connect(db_path: str | None) -> Any:
    return duckdb.connect(db_path or str(CONFIG.paths.duckdb_path))


def ensure_table(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            ticker        VARCHAR NOT NULL,
            dispatch_date DATE    NOT NULL,
            horizon       INTEGER,
            hold_days     INTEGER NOT NULL,
            weight        DOUBLE,
            status        VARCHAR DEFAULT 'OPEN',
            closed_date   DATE,
            dispatched_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def record_dispatch(
    signals: list[dict],
    strategy: dict | None,
    horizon: int,
    db_path: str | None = None,
    today: date | None = None,
) -> int:
    """Record OPEN ledger rows for a broadcast tranche dispatch.

    No-op for legacy half-Kelly artifacts (no tranche strategy → no fixed
    exit horizon to track). Idempotent per (ticker, dispatch_date) so a
    pipeline re-run on the same day cannot double-book a cohort.

    Returns the number of rows actually inserted.
    """
    if not strategy or strategy.get("mode") != "tranche":
        return 0
    hold_days = int(strategy.get("hold_days") or 0)
    if hold_days <= 0 or not signals:
        return 0

    today = today or datetime.now().date()
    rows = [
        (str(s["ticker"]).upper(), today, int(horizon), hold_days,
         float(s.get("suggested_weight") or 0.0))
        for s in signals
        if s.get("ticker")
    ]
    if not rows:
        return 0

    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            existing = {
                str(r[0]).upper()
                for r in conn.execute(
                    f"SELECT ticker FROM {TABLE} WHERE dispatch_date = ?", [today]
                ).fetchall()
            }
            rows = [r for r in rows if r[0] not in existing]
            if rows:
                conn.executemany(
                    f"INSERT INTO {TABLE} (ticker, dispatch_date, horizon, hold_days, weight) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
    except Exception:  # noqa: BLE001 — ledger must never kill the dispatch path
        LOGGER.exception("[SignalLedger] record_dispatch failed.")
        return 0

    LOGGER.info("[SignalLedger] Recorded %s OPEN signals (hold=%s sessions).",
                len(rows), hold_days)
    return len(rows)


def check_exits_due(db_path: str | None = None, today: date | None = None) -> list[dict]:
    """OPEN signals whose hold horizon has elapsed in TRADING sessions.

    A signal dispatched on D with hold_days=H is due once the fresh parquet
    calendar contains >= H trading dates strictly after D (mirrors the
    backtest engine, which exits the cohort at the close of session D+H).
    """
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            open_rows = conn.execute(
                f"SELECT ticker, dispatch_date, horizon, hold_days, weight "
                f"FROM {TABLE} WHERE status = 'OPEN'"
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] check_exits_due read failed.")
        return []

    if not open_rows:
        return []

    min_dispatch = min(r[1] for r in open_rows)
    sessions = price_lookup.trading_dates_after(min_dispatch)
    if not sessions:
        return []

    today = today or datetime.now().date()
    due: list[dict] = []
    for ticker, d0, horizon, hold_days, weight in open_rows:
        elapsed = sum(1 for s in sessions if d0 < s <= today)
        if elapsed >= int(hold_days):
            due.append({
                "ticker": str(ticker),
                "dispatch_date": d0,
                "horizon": int(horizon) if horizon is not None else None,
                "hold_days": int(hold_days),
                "weight": float(weight or 0.0),
                "sessions_elapsed": elapsed,
            })
    return due


def mark_closed(
    due: list[dict],
    db_path: str | None = None,
    today: date | None = None,
) -> int:
    """Flip the given (ticker, dispatch_date) rows to CLOSED."""
    if not due:
        return 0
    today = today or datetime.now().date()
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            for d in due:
                conn.execute(
                    f"UPDATE {TABLE} SET status = 'CLOSED', closed_date = ? "
                    "WHERE ticker = ? AND dispatch_date = ? AND status = 'OPEN'",
                    [today, d["ticker"], d["dispatch_date"]],
                )
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] mark_closed failed.")
        return 0
    return len(due)
