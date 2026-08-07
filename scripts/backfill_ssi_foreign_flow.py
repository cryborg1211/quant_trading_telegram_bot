"""Overnight full-universe SSI FastConnect foreign-flow backfill (26-07-26).

Backfills data/foreign_flow_daily.parquet with real historical foreign-flow
(+ block-deal + aggressor-side trade) data for every ticker with local
OHLCV history, via SOURCE 2 (src/data/foreign_flow_crawler.py). Idempotent
-- safe to re-run, merge is keyed on (date, ticker), fresher fetched_at wins
(see _merge_and_persist).

BACKFILL_FROM is deliberately 2020-01-01, not further back: live-verified
this session that the SSI ticker's real history starts 2020-02-26 despite
requesting from 2015 -- depth is ticker-dependent, but requesting further
back than any ticker plausibly has just wastes empty-result API calls.

RESUME (26-07-26): the underlying FastConnect fetch calls have no retry on
a dropped connection -- a network outage mid-run silently truncates the
CURRENT ticker's history (windows fetch in ascending date order, see
_chunk_date_range, so a failure leaves early dates and drops the tail) and
then races through remaining tickers near-instantly, same failure. A plain
re-run would safely re-merge but cost the full ~5h again. _tickers_needing_
backfill skips any ticker whose most recent FastConnect row is already
within RESUME_BUFFER_DAYS of today -- a restart only pays for what's
actually missing.

Run: python scripts/backfill_ssi_foreign_flow.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import polars as pl  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from src.data import foreign_flow_crawler as ffc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OHLCV_GLOB = "data/ohlcv_*.parquet"
BACKFILL_FROM = date(2020, 1, 1)
RESUME_BUFFER_DAYS = 7


def _tickers_needing_backfill(
    tickers: list[str],
    to_date: date,
    parquet_path: Path,
    recent_buffer_days: int = RESUME_BUFFER_DAYS,
) -> list[str]:
    """Drop tickers whose FastConnect backfill already reached near `to_date`.

    Not perfectly rigorous (a mid-range gap could theoretically survive
    alongside a recent date), but matches the actual failure mode: windows
    fetch oldest-first, so a truncated ticker is missing its TAIL, and a
    ticker with a recent row is very likely complete. Absent parquet or
    read failure -> nothing is skipped (full backfill, matches a cold start).
    """
    if not parquet_path.exists():
        return list(tickers)
    try:
        existing = pl.read_parquet(parquet_path)
    except Exception:  # noqa: BLE001 -- corrupt/unreadable -> don't skip anything
        return list(tickers)

    fc = existing.filter(pl.col("source") == "ssi_fastconnect_history")
    if fc.is_empty():
        return list(tickers)

    cutoff = to_date - timedelta(days=recent_buffer_days)
    latest_by_ticker = fc.group_by("ticker").agg(pl.col("date").max().alias("latest"))
    covered = set(latest_by_ticker.filter(pl.col("latest") >= cutoff)["ticker"].to_list())

    remaining = [t for t in tickers if t not in covered]
    skipped = len(tickers) - len(remaining)
    if skipped:
        print(f"Resume: skipping {skipped}/{len(tickers)} tickers already backfilled "
              f"within {recent_buffer_days}d of {to_date}.")
    return remaining


def main() -> None:
    files = sorted(REPO.glob(OHLCV_GLOB))
    all_tickers = sorted({f.stem.replace("ohlcv_", "").upper() for f in files})
    if not all_tickers:
        print(f"No files matched {OHLCV_GLOB} -- nothing to backfill.")
        return

    to_date = date.today()
    tickers = _tickers_needing_backfill(all_tickers, to_date, ffc._DEFAULT_PARQUET)
    if not tickers:
        print(f"All {len(all_tickers)} tickers already backfilled within "
              f"{RESUME_BUFFER_DAYS}d of {to_date} -- nothing to do.")
        return

    print(f"Backfilling {len(tickers)}/{len(all_tickers)} tickers, {BACKFILL_FROM} -> {to_date} ...")
    print("Idempotent -- safe to interrupt and re-run, already-fetched (date, ticker) rows just get overwritten.\n")

    t0 = time.perf_counter()
    total = ffc.backfill_foreign_flow_history(tickers, BACKFILL_FROM, to_date)
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 72}")
    print(f"DONE in {elapsed / 60:.1f} min ({elapsed / max(len(tickers), 1):.1f}s/ticker avg)")
    print(f"Parquet now has {total} total rows -> {ffc._DEFAULT_PARQUET}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
