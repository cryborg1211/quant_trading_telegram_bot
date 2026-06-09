"""Point price / volume lookups backed by the FRESH parquet vintage.

Single source of truth for ad-hoc price queries now that the core DuckDB
``stock_ohlcv`` table is retired. That table drifted ~18 days stale because
``crawl_hose`` writes the per-ticker parquet shards (``data/ohlcv_*.parquet``),
not the table — see the DB audit.

Per-ticker lookups (``close_on_or_before`` / ``close_on_or_after`` /
``latest_close``) read the SINGLE shard ``data/ohlcv_<TICKER>.parquet`` — the
file IS the partition, so there is no glob footer-scan and no ``WHERE ticker``
needed. The ticker is validated against ``[A-Z0-9]{1,12}`` before it is
interpolated into the ``read_parquet`` path, so it can neither traverse the
filesystem nor inject SQL. Cross-ticker scans (``top_tickers_by_volume``) still
glob.

Predicate pushdown: the ``date`` column is left BARE and the *parameter* is cast
(``date <= CAST(? AS DATE)``) so DuckDB can prune parquet row-groups by the
date min/max statistics. (Wrapping the column in ``CAST(date AS DATE)`` defeats
that.) The shards are written with a ``date32`` (DATE) column by
``pipeline.load_ohlcv``, so the bare-column comparison is exact.

Callers that already hold a DuckDB connection (e.g. the ``DuckDBEngine``
singleton's ``db.conn``) may pass it to avoid per-call connection churn;
otherwise an ephemeral in-memory connection is used (``read_parquet`` needs no
attached database). Every function is defensive: a missing shard / read failure
logs a warning and returns ``None`` / ``[]`` so the live RL-backfill / audit /
sentiment paths degrade gracefully.
"""
from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import duckdb

LOGGER = logging.getLogger(__name__)

# Repo-root-anchored, forward-slash paths so lookups resolve regardless of the
# caller's cwd and parse cleanly inside a DuckDB string literal on Windows
# (backslashes in a single-quoted SQL literal are error-prone).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_PARQUET_GLOB = (_DATA_DIR / "ohlcv_*.parquet").as_posix()

# Tickers are uppercase alphanumerics (HOSE: 3 chars; allow 1-12 for indices/
# warrants). Validated BEFORE path interpolation — blocks traversal + SQL
# injection through the read_parquet() path literal.
_TICKER_RE = re.compile(r"[A-Z0-9]{1,12}")


def _glob_source() -> str:
    return f"read_parquet('{_PARQUET_GLOB}')"


def _shard_source(ticker: str) -> str | None:
    """``read_parquet('<abs>/data/ohlcv_<TICKER>.parquet')`` for ONE ticker, or
    ``None`` if the ticker is malformed or its shard is absent (delisted /
    not yet crawled). Sanitising the ticker first makes the path-literal safe."""
    t = str(ticker).upper().strip()
    if not _TICKER_RE.fullmatch(t):
        return None
    shard = _DATA_DIR / f"ohlcv_{t}.parquet"
    if not shard.exists():
        return None
    return f"read_parquet('{shard.as_posix()}')"


def _conn_ctx(conn: Any | None):
    """A context manager yielding a usable connection.

    If the caller supplied one, reuse it via ``nullcontext`` (we do NOT close
    a connection we don't own). Otherwise open an ephemeral in-memory DuckDB
    connection that closes itself on context exit.
    """
    return nullcontext(conn) if conn is not None else duckdb.connect()


def close_on_or_before(ticker: str, ref_date: Any, conn: Any | None = None) -> float | None:
    """Most recent close at-or-before ``ref_date`` (defensively handles
    weekends/holidays by walking back to the prior trading day)."""
    src = _shard_source(ticker)
    if src is None:
        return None
    try:
        with _conn_ctx(conn) as c:
            row = c.execute(
                f"SELECT close FROM {src} "
                "WHERE date <= CAST(? AS DATE) "
                "ORDER BY date DESC LIMIT 1",
                [ref_date],
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("price_lookup.close_on_or_before(%s, %s) failed: %s", ticker, ref_date, exc)
        return None


def close_on_or_after(ticker: str, ref_date: Any, conn: Any | None = None) -> float | None:
    """First available close at-or-after ``ref_date``."""
    src = _shard_source(ticker)
    if src is None:
        return None
    try:
        with _conn_ctx(conn) as c:
            row = c.execute(
                f"SELECT close FROM {src} "
                "WHERE date >= CAST(? AS DATE) "
                "ORDER BY date ASC LIMIT 1",
                [ref_date],
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("price_lookup.close_on_or_after(%s, %s) failed: %s", ticker, ref_date, exc)
        return None


def latest_close(ticker: str, conn: Any | None = None) -> float | None:
    """Latest available close for ``ticker``."""
    src = _shard_source(ticker)
    if src is None:
        return None
    try:
        with _conn_ctx(conn) as c:
            row = c.execute(
                f"SELECT close FROM {src} ORDER BY date DESC LIMIT 1",
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("price_lookup.latest_close(%s) failed: %s", ticker, exc)
        return None


def top_tickers_by_volume(limit: int, lookback_days: int = 10, conn: Any | None = None) -> list[str]:
    """Top-``limit`` tickers by summed volume over the last ``lookback_days``
    days of available data — most-liquid names first. Cross-ticker → globs all
    shards. Date column left bare for row-group pushdown."""
    try:
        with _conn_ctx(conn) as c:
            rows = c.execute(
                f"SELECT ticker FROM {_glob_source()} "
                f"WHERE date >= (SELECT MAX(date) - INTERVAL {int(lookback_days)} DAY "
                f"FROM {_glob_source()}) "
                "GROUP BY ticker ORDER BY SUM(volume) DESC NULLS LAST LIMIT ?",
                [int(limit)],
            ).fetchall()
        return [str(r[0]).upper() for r in rows if r and r[0]]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("price_lookup.top_tickers_by_volume(%s) failed: %s", limit, exc)
        return []
