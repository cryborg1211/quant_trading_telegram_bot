"""Overnight full-universe SSI FastConnect foreign-flow backfill (26-07-26).

Backfills data/foreign_flow_daily.parquet with real historical foreign-flow
(+ block-deal + aggressor-side trade) data for every ticker with local
OHLCV history, via SOURCE 2 (src/data/foreign_flow_crawler.py). Idempotent
-- safe to re-run/resume, merge is keyed on (date, ticker), fresher
fetched_at wins (see _merge_and_persist).

BACKFILL_FROM is deliberately 2020-01-01, not further back: live-verified
this session that the SSI ticker's real history starts 2020-02-26 despite
requesting from 2015 -- depth is ticker-dependent, but requesting further
back than any ticker plausibly has just wastes empty-result API calls.

Run: python scripts/backfill_ssi_foreign_flow.py
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from src.data import foreign_flow_crawler as ffc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OHLCV_GLOB = "data/ohlcv_*.parquet"
BACKFILL_FROM = date(2020, 1, 1)


def main() -> None:
    files = sorted(REPO.glob(OHLCV_GLOB))
    tickers = sorted({f.stem.replace("ohlcv_", "").upper() for f in files})
    if not tickers:
        print(f"No files matched {OHLCV_GLOB} -- nothing to backfill.")
        return

    to_date = date.today()
    print(f"Backfilling {len(tickers)} tickers, {BACKFILL_FROM} -> {to_date} ...")
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
