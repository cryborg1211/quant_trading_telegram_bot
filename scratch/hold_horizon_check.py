"""Does the edge keep accruing past T+20?  Per-row check for hold extension."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)
sys.path.insert(0, str(_REPO))

import duckdb
import pandas as pd

from src.execution.vn_cost_model import FeeSchedule

RT = FeeSchedule().round_trip_pct() * 100

trades = pd.read_parquet("data/edge_profiler_trades.parquet")
trades = trades.sort_values(["ticker", "date"]).reset_index(drop=True)
trades["p_up_lag"] = trades.groupby("ticker", sort=False)["p_up"].shift(1)

for h in [30, 40, 60]:
    trades[f"fwd_ret_t{h}"] = trades.groupby("ticker", sort=False)["close"].transform(
        lambda s, h=h: s.shift(-h) / s - 1.0)

sql = f"""
WITH adv AS (
    SELECT ticker, date,
        AVG(close * volume) OVER (PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS adv20,
        COUNT(close * volume) OVER (PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS n_obs
    FROM read_parquet('{(_REPO / "data" / "ohlcv_*.parquet").as_posix()}')
    WHERE close > 0 AND volume IS NOT NULL
)
SELECT ticker, date,
    RANK() OVER (PARTITION BY date ORDER BY adv20 DESC NULLS LAST) AS adv_rank
FROM adv WHERE n_obs >= 20 AND date >= CAST('2022-09-01' AS DATE)
"""
with duckdb.connect() as conn:
    ranks = conn.execute(sql).df()
ranks["date"] = pd.to_datetime(ranks["date"]).dt.date
trades = trades.merge(ranks, on=["ticker", "date"], how="left")

cohort = trades[(trades["adv_rank"] <= 50) & (trades["p_up_lag"] >= 0.43)].copy()
cohort["day_rank"] = cohort.groupby("date")["p_up_lag"].rank(
    method="first", ascending=False)
top5 = cohort[cohort["day_rank"] <= 5]
print(f"liquid lag>=0.43 within-day top-5: n={len(top5):,}\n")
print(f"{'H':>4} {'gross%':>8} {'net%':>8} {'net/day%':>9} {'ann_net%':>9}")
for h in [10, 20, 30, 40, 60]:
    col = f"fwd_ret_t{h}"
    v = top5[col].dropna()
    gross = v.mean() * 100
    net = gross - RT
    per_day = net / h
    ann = per_day * 252
    print(f"{h:>4} {gross:>8.3f} {net:>8.3f} {per_day:>9.4f} {ann:>9.1f}")
