"""Within-day score rank vs forward return — is the cross-section inverted?

The engine takes the top-5 names by p_up each rebalance and they underperform
the full >=0.40 cohort by 1.7pp.  Hypothesis: p_up's edge is mostly TEMPORAL
(when many names clear the bar, the market is about to rise) and its
WITHIN-DAY ranking adds nothing — or is inverted at the extreme top.

Implementable basis: lag-1 scores (signal D-1, entry close D), liquid top-50.

Run:
    python scripts/within_day_rank_check.py
"""
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

TRADES = Path("data/edge_profiler_trades.parquet")
_OHLCV_GLOB = (_REPO / "data" / "ohlcv_*.parquet").as_posix()
ROUND_TRIP = FeeSchedule().round_trip_pct() * 100
SIG_THR = 0.40
TOP_N = 50


def _adv_ranks() -> pd.DataFrame:
    sql = f"""
    WITH adv AS (
        SELECT ticker, date,
            AVG(close * volume) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS adv20,
            COUNT(close * volume) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS n_obs
        FROM read_parquet('{_OHLCV_GLOB}')
        WHERE close > 0 AND volume IS NOT NULL
    )
    SELECT ticker, date,
        RANK() OVER (PARTITION BY date ORDER BY adv20 DESC NULLS LAST) AS adv_rank
    FROM adv
    WHERE n_obs >= 20 AND date >= CAST('2022-09-01' AS DATE)
    """
    with duckdb.connect() as conn:
        df = conn.execute(sql).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def main() -> None:
    trades = pd.read_parquet(TRADES)
    trades = trades.sort_values(["ticker", "date"]).reset_index(drop=True)
    trades["p_up_lag"] = trades.groupby("ticker", sort=False)["p_up"].shift(1)
    trades = trades.merge(_adv_ranks(), on=["ticker", "date"], how="left")

    cohort = trades[
        (trades["adv_rank"] <= TOP_N) & (trades["p_up_lag"] >= SIG_THR)
    ].copy()
    cohort["day_rank"] = (
        cohort.groupby("date")["p_up_lag"].rank(method="first", ascending=False)
    )
    cohort["rank_bucket"] = pd.cut(
        cohort["day_rank"], [0, 5, 10, 25, 50, 10_000],
        labels=["top1-5", "6-10", "11-25", "26-50", ">50"],
    )

    bar = "=" * 74
    print(f"{bar}\n WITHIN-DAY SCORE RANK vs T+20 RETURN (liquid, lag-score >= {SIG_THR})\n{bar}")
    g = cohort.groupby("rank_bucket", observed=True).agg(
        n=("fwd_ret_t20", "size"),
        gross_t20=("fwd_ret_t20", lambda s: s.mean() * 100),
        wr=("fwd_ret_t20", lambda s: (s.dropna() > 0).mean() * 100),
        p_up_mean=("p_up_lag", "mean"),
    ).round(3)
    g["net_t20"] = (g["gross_t20"] - ROUND_TRIP).round(3)
    print(g.to_string())

    # Same cut but only on days with a deep cohort (>=15 candidates) to avoid
    # thin days where top-5 IS the whole cohort.
    deep_days = cohort.groupby("date")["ticker"].transform("size") >= 15
    deep = cohort[deep_days]
    print(f"\n  DEEP DAYS ONLY (>=15 candidates; n_days="
          f"{deep['date'].nunique()}):")
    g2 = deep.groupby("rank_bucket", observed=True).agg(
        n=("fwd_ret_t20", "size"),
        gross_t20=("fwd_ret_t20", lambda s: s.mean() * 100),
        wr=("fwd_ret_t20", lambda s: (s.dropna() > 0).mean() * 100),
    ).round(3)
    g2["net_t20"] = (g2["gross_t20"] - ROUND_TRIP).round(3)
    print(g2.to_string())

    # Equal-weight ALL candidates vs top-5: the portfolio the engine SHOULD run?
    by_day = cohort.groupby("date").agg(
        all_mean=("fwd_ret_t20", "mean"),
        n=("ticker", "size"),
    )
    top5 = cohort[cohort["day_rank"] <= 5].groupby("date")["fwd_ret_t20"].mean()
    cmp = by_day.join(top5.rename("top5_mean"), how="inner").dropna()
    print(f"\n  PER-DAY PORTFOLIO COMPARISON ({len(cmp)} days):")
    print(f"  equal-weight ALL candidates : {cmp['all_mean'].mean() * 100:+.3f}% gross T+20")
    print(f"  top-5 by score              : {cmp['top5_mean'].mean() * 100:+.3f}% gross T+20")
    print(f"  top-5 wins on               : {(cmp['top5_mean'] > cmp['all_mean']).mean() * 100:.1f}% of days")
    print(bar)


if __name__ == "__main__":
    main()
