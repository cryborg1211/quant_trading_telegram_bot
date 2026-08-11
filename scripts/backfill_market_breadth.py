"""Backfill official exchange breadth (DailyIndex) from SSI FastConnect.

Captures Advances / Declines / NoChanges plus CEILINGS / FLOORS — the limit-up
and limit-down counts, which have no equivalent anywhere in this system and are
a direct capitulation measure (what the MR knife-catch signal is trying to time,
and what the unconfirmed breadth-inflection work infers indirectly from rolling
returns). Also carries the put-through block-deal totals.

Dry-run by default; --commit writes. Idempotent on (index_id, date), so a re-run
over an overlapping range is safe.

    python scripts/backfill_market_breadth.py                       # dry-run, 30d
    python scripts/backfill_market_breadth.py --from 2016-01-01 --commit
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import polars as pl  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# FastConnect credentials live in .env; without this the token call silently
# returns None and the whole backfill reports "fetched nothing".
load_dotenv(REPO / ".env")

from src.data.market_breadth_crawler import (  # noqa: E402
    DEFAULT_INDICES,
    _DEFAULT_PARQUET,
    backfill_market_breadth,
    fetch_daily_index,
)

LOGGER = logging.getLogger("backfill_breadth")


def _parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", type=_parse_day,
                    default=date.today() - timedelta(days=30))
    ap.add_argument("--to", dest="to_date", type=_parse_day, default=date.today())
    ap.add_argument("--indices", default=",".join(DEFAULT_INDICES),
                    help="comma-separated index ids")
    ap.add_argument("--commit", action="store_true",
                    help="write the parquet (else dry-run: fetch and report only)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    indices = tuple(i.strip().upper() for i in args.indices.split(",") if i.strip())
    LOGGER.info("range %s .. %s   indices=%s   commit=%s",
                args.from_date, args.to_date, indices, args.commit)

    if args.commit:
        total = backfill_market_breadth(args.from_date, args.to_date, indices)
        LOGGER.info("COMMITTED — %d total rows on disk at %s", total, _DEFAULT_PARQUET)
        df = pl.read_parquet(_DEFAULT_PARQUET)
    else:
        frames = [fetch_daily_index(i, args.from_date, args.to_date) for i in indices]
        frames = [f for f in frames if not f.is_empty()]
        if not frames:
            LOGGER.warning("DRY-RUN fetched nothing.")
            return
        df = pl.concat(frames, how="diagonal_relaxed")
        LOGGER.info("DRY-RUN — fetched %d rows, wrote NOTHING (pass --commit)", df.height)

    print()
    for idx in sorted(df["index_id"].unique().to_list()):
        sub = df.filter(pl.col("index_id") == idx).sort("date")
        print(f"{idx}: {sub.height} rows  {sub['date'].min()} .. {sub['date'].max()}")
    print("\nnull counts (a permanently-null column means the field is not served):")
    nulls = df.null_count().to_dicts()[0]
    for k, v in nulls.items():
        flag = "  <-- ALL NULL" if v == df.height else ""
        print(f"    {k:18s} {v:6d}/{df.height}{flag}")

    print("\nmost recent 6 sessions:")
    cols = ["date", "index_id", "index_value", "advances", "declines",
            "ceilings", "floors", "total_deal_val"]
    print(df.sort("date", descending=True).select(cols).head(6))

    # Floors is the reason this feed exists — show whether it actually varies.
    fl = df.filter(pl.col("floors").is_not_null())["floors"]
    if fl.len():
        print(f"\nfloors  min={fl.min()} median={fl.median()} max={fl.max()} "
              f"nonzero={int((fl > 0).sum())}/{fl.len()}")
    ce = df.filter(pl.col("ceilings").is_not_null())["ceilings"]
    if ce.len():
        print(f"ceilings min={ce.min()} median={ce.median()} max={ce.max()} "
              f"nonzero={int((ce > 0).sum())}/{ce.len()}")


if __name__ == "__main__":
    main()
