"""Autopsy: join engine fills with the per-row profiler frame.

The engine's realized trips at T+20 cadence average -0.12% gross while
per-row forward returns for thr>=0.40 liquid signals average +1.55%.
Joining each engine BUY to its (ticker, date) profiler row answers:
  1. What p_up did the engine ACTUALLY buy?  (scrambled mapping => ~0.40 noise)
  2. What did close-to-close T+20 say those rows should earn?
  3. Realized trip return vs row-implied return => exit-timing damage.

Run:
    python scripts/engine_trade_autopsy.py
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)
sys.path.insert(0, str(_REPO))

import pandas as pd

FILLS = Path("data/engine_fills_thr0.40.parquet")
TRADES = Path("data/edge_profiler_trades.parquet")


def _fifo_trips(fills: pd.DataFrame) -> pd.DataFrame:
    trips = []
    for ticker, grp in fills.sort_values("date").groupby("ticker"):
        lots: deque = deque()
        for f in grp.itertuples(index=False):
            side = str(f.side).upper()
            if side == "BUY":
                lots.append([f.date, f.qty, f.price])
            elif side == "SELL":
                remaining = f.qty
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot[1])
                    trips.append({
                        "ticker": ticker,
                        "entry_date": lot[0], "exit_date": f.date,
                        "qty": take, "entry_price": lot[2], "exit_price": f.price,
                        "trip_ret_pct": (f.price / lot[2] - 1.0) * 100,
                        "hold_days": (pd.Timestamp(f.date) - pd.Timestamp(lot[0])).days,
                    })
                    lot[1] -= take
                    remaining -= take
                    if lot[1] <= 0:
                        lots.popleft()
    return pd.DataFrame(trips)


def main() -> None:
    fills = pd.read_parquet(FILLS)
    trades = pd.read_parquet(TRADES)
    trades["date_key"] = trades["date"].astype(str)

    trips = _fifo_trips(fills)
    trips["date_key"] = trips["entry_date"].astype(str)
    print(f"trips: {len(trips):,}")

    cols = ["ticker", "date_key", "p_up", "score_decile", "regime",
            "fwd_ret_t20", "fwd_ret_t5"]
    m = trips.merge(trades[cols], on=["ticker", "date_key"], how="left")
    matched = m[m["p_up"].notna()]
    print(f"matched to profiler rows: {len(matched):,} / {len(m):,}")

    bar = "=" * 76
    print(f"\n{bar}\n ENGINE TRADE AUTOPSY\n{bar}")

    # 1. What conviction did the engine actually buy?
    print("\n  p_up OF ENGINE ENTRIES (should skew HIGH if mapping is correct):")
    print(matched["p_up"].describe(percentiles=[0.1, 0.5, 0.9]).round(4).to_string())
    print("\n  score_decile distribution of entries:")
    print(matched["score_decile"].value_counts().sort_index().to_string())

    # 2. Row-implied vs realized
    print(f"\n  ROW-IMPLIED close-to-close T+20 of entry rows : "
          f"{matched['fwd_ret_t20'].mean() * 100:+.3f}%  (what the signal earned)")
    print(f"  REALIZED engine trip return (gross)           : "
          f"{matched['trip_ret_pct'].mean():+.3f}%  (what the engine captured)")
    print(f"  gap = exit-timing + entry-px-vs-close damage  : "
          f"{matched['trip_ret_pct'].mean() - matched['fwd_ret_t20'].mean() * 100:+.3f}pp")

    # 3. Damage by hold length (premature/forced exits?)
    print("\n  BY HOLD LENGTH:")
    bins = pd.cut(matched["hold_days"], [0, 10, 21, 35, 60, 10_000],
                  labels=["<=10d", "11-21d", "22-35d", "36-60d", ">60d"])
    g = matched.groupby(bins, observed=True).agg(
        n=("trip_ret_pct", "size"),
        realized=("trip_ret_pct", "mean"),
        row_implied_t20=("fwd_ret_t20", lambda s: s.mean() * 100),
    ).round(3)
    print(g.to_string())

    # 4. Damage by exit date clustering (regime liquidations?)
    exit_counts = matched.groupby("exit_date").agg(
        n=("trip_ret_pct", "size"), mean_ret=("trip_ret_pct", "mean")).round(3)
    mass_exits = exit_counts[exit_counts["n"] >= 5].sort_values("n", ascending=False)
    print(f"\n  MASS-EXIT DATES (>=5 positions closed same day): {len(mass_exits)}")
    print(mass_exits.head(12).to_string())

    # 5. Worst trips
    print("\n  10 WORST TRIPS:")
    worst = matched.nsmallest(10, "trip_ret_pct")[
        ["ticker", "entry_date", "exit_date", "hold_days",
         "trip_ret_pct", "fwd_ret_t20", "p_up"]]
    worst = worst.assign(fwd_ret_t20=(worst["fwd_ret_t20"] * 100).round(2))
    print(worst.to_string(index=False))
    print(bar)


if __name__ == "__main__":
    main()
