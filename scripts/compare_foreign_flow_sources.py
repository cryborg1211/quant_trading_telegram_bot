"""Compare data quality: SOURCE 1 (iBoard live snapshot, unofficial) vs
SOURCE 2 (FastConnect history, official) on their overlapping (date,
ticker) rows (26-07-26 backlog plan item 3, user-requested 07-08-26).

Read-only diagnostic. Answers: do the two sources agree on foreign flow
for the same (date, ticker), or diverge -- and if they diverge, which one
looks more trustworthy? Informs whether SOURCE 1's daily production cron
stays live going forward or gets replaced by a daily FastConnect call.

Run: python scripts/compare_foreign_flow_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.data.foreign_flow_crawler import _DEFAULT_PARQUET  # noqa: E402


def main() -> None:
    df = pl.read_parquet(_DEFAULT_PARQUET)
    print(f"Total rows: {df.height}")
    print(df.group_by("source").agg(pl.len().alias("rows")).sort("source"))

    s1 = df.filter(pl.col("source") == "ssi_iboard_live_snapshot").select(
        "date", "ticker", "foreign_buy_val", "foreign_sell_val", "foreign_net_val")
    s2 = df.filter(pl.col("source") == "ssi_fastconnect_history").select(
        "date", "ticker",
        pl.col("foreign_buy_val").alias("fc_buy"),
        pl.col("foreign_sell_val").alias("fc_sell"),
        pl.col("foreign_net_val").alias("fc_net"),
    )

    overlap = s1.join(s2, on=["date", "ticker"], how="inner")
    print(f"\nOverlapping (date, ticker) rows: {overlap.height}")
    if overlap.is_empty():
        print("No overlap -- SOURCE 1's daily cron hasn't run on any date SOURCE 2 also covers. Nothing to compare yet.")
        return

    for col_a, col_b, label in [
        ("foreign_buy_val", "fc_buy", "buy_val"),
        ("foreign_sell_val", "fc_sell", "sell_val"),
        ("foreign_net_val", "fc_net", "net_val"),
    ]:
        a = overlap[col_a].to_numpy()
        b = overlap[col_b].to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) < 2:
            print(f"\n{label}: not enough valid pairs to compare (n={len(a)})")
            continue
        corr = np.corrcoef(a, b)[0, 1]
        diff = a - b
        denom = np.where(np.abs(b) > 1e-9, np.abs(b), np.nan)
        pct_diff = np.nanmedian(np.abs(diff) / denom) * 100
        exact_match = np.mean(np.isclose(a, b, rtol=1e-6, atol=1e-6)) * 100
        print(f"\n{label} (n={len(a)}):")
        print(f"  correlation:        {corr:.4f}")
        print(f"  exact match:        {exact_match:.1f}%")
        print(f"  median |%diff|:     {pct_diff:.2f}%  (where SOURCE 2 != 0)")
        print(f"  SOURCE1 mean/std:   {a.mean():+,.1f} / {a.std():,.1f}")
        print(f"  SOURCE2 mean/std:   {b.mean():+,.1f} / {b.std():,.1f}")

    print("\nVerdict lens: >99% exact match / corr~1.0 = sources agree, trust either. "
          "High correlation but non-trivial %diff = same signal, different rounding/timing. "
          "Low correlation = real disagreement, investigate which one is right before trusting either for research.")


if __name__ == "__main__":
    main()
