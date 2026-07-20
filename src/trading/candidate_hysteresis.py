"""Candidate admission hysteresis (2026-07-20 meta-controller optimization #4).

BSR bounced in and out of the arbitrator pool across 4 consecutive July
dispatch days (07-09/13/14/15) — a name that barely crosses the meta-gate one
day and falls back out the next still gets a fresh look every single run.
The open-cohort dedup (`main._select_candidates`, shipped 20-07-26) guards a
DIFFERENT failure mode — re-buying into an already-OPEN cohort; this guards
against ADMITTING a name in the first place without two straight days of
genuine conviction.

State: `candidate_qualify_streak` DuckDB table (ticker PK), self-contained
connection lifecycle — mirrors `signal_ledger.py`'s `_connect(db_path)`
pattern exactly, no dependency on an external engine object.

Read/write are split so the caller can persist-gate correctly: `read_streaks`
is always safe (informational, no mutation — fine even under a dashboard
preview's `persist=False`); `update_streaks` mutates and MUST be called only
when the caller's own persist flag is True (mirrors the sentiment-entry
paperlog's persist=False contract — see main.py's `_select_candidates`).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import duckdb

from config.settings import CONFIG

LOGGER = logging.getLogger(__name__)

TABLE = "candidate_qualify_streak"


def _connect(db_path: str | None) -> Any:
    return duckdb.connect(db_path or str(CONFIG.paths.duckdb_path))


def ensure_table(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            ticker VARCHAR PRIMARY KEY,
            last_qualify_date DATE,
            streak INTEGER
        )
        """
    )


def read_streaks(tickers: list[str], db_path: str | None = None) -> dict[str, int]:
    """Current streak per ticker (absent → 0). Read-only. Never raises."""
    if not tickers:
        return {}
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            placeholders = ",".join("?" * len(tickers))
            rows = conn.execute(
                f"SELECT ticker, streak FROM {TABLE} WHERE ticker IN ({placeholders})",
                list(tickers),
            ).fetchall()
        return {str(t): int(s) for t, s in rows}
    except Exception:  # noqa: BLE001
        LOGGER.exception("[hysteresis] read_streaks failed — treating all as streak=0.")
        return {}


def update_streaks(
    tickers: list[str],
    today: date,
    max_gap_days: int = 4,
    db_path: str | None = None,
) -> None:
    """Record today's qualification for each ticker (upsert). Never raises.

    Consecutive-day streak logic: a gap of 1–`max_gap_days` calendar days
    since the ticker's last qualifying run extends the streak; anything
    longer (or a first-ever appearance) resets it to 1. A second call on the
    SAME date (e.g. a manual re-run) is a no-op for tickers already recorded
    today — it does not double-increment.
    """
    if not tickers:
        return
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            placeholders = ",".join("?" * len(tickers))
            existing = {
                str(t): (d, int(s))
                for t, d, s in conn.execute(
                    f"SELECT ticker, last_qualify_date, streak FROM {TABLE} "
                    f"WHERE ticker IN ({placeholders})",
                    list(tickers),
                ).fetchall()
            }
            for ticker in tickers:
                prev = existing.get(ticker)
                if prev is None:
                    streak = 1
                else:
                    last_date, prev_streak = prev
                    if last_date == today:
                        continue
                    gap = (today - last_date).days
                    streak = prev_streak + 1 if 1 <= gap <= max_gap_days else 1
                conn.execute(
                    f"INSERT INTO {TABLE} (ticker, last_qualify_date, streak) "
                    f"VALUES (?, ?, ?) ON CONFLICT (ticker) DO UPDATE SET "
                    f"last_qualify_date = excluded.last_qualify_date, "
                    f"streak = excluded.streak",
                    [ticker, today, streak],
                )
    except Exception:  # noqa: BLE001
        LOGGER.exception("[hysteresis] update_streaks failed — non-fatal.")
