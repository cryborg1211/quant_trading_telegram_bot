"""Grid-date luck vs allocator anti-selection.

The engine's 214 trips have row-implied T+20 of -0.84% while the candidate
cohort averages +1.5%.  On the ~45 rebalance grid dates, compare:
  1. MARKET    — mean fwd T+20 over all tickers that date
  2. COHORT    — liquid top-50, lag-score >= sig_thr (what the engine chose from)
  3. PICKS     — what the engine actually bought
If COHORT ~ PICKS << +1.5%, the grid dates are the problem (timing luck /
crash clustering).  If COHORT >> PICKS, the Kelly/MV allocator anti-selects.

Run:
    python scripts/grid_date_check.py
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

FILLS = Path("data/engine_fills_thr0.40.parquet")
TRADES = Path("data/edge_profiler_trades.parquet")
_OHLCV_GLOB = (_REPO / "data" / "ohlcv_*.parquet").as_posix()
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
    fills = pd.read_parquet(FILLS)
    buys = fills[fills["side"].str.upper() == "BUY"].copy()
    buys["date"] = pd.to_datetime(buys["date"]).dt.date
    grid_dates = sorted(buys["date"].unique())
    print(f"grid entry dates: {len(grid_dates)}")

    trades = pd.read_parquet(TRADES)
    trades = trades.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Lag score: the engine's decision on date D uses the previous row's p_up.
    trades["p_up_lag"] = trades.groupby("ticker", sort=False)["p_up"].shift(1)
    trades = trades.merge(_adv_ranks(), on=["ticker", "date"], how="left")
    trades["is_liquid"] = trades["adv_rank"] <= TOP_N

    rows = []
    for d in grid_dates:
        day = trades[trades["date"] == d]
        if len(day) == 0:
            continue
        market = day["fwd_ret_t20"].dropna()
        cohort = day[(day["is_liquid"]) & (day["p_up_lag"] >= SIG_THR)]["fwd_ret_t20"].dropna()
        pick_names = set(buys[buys["date"] == d]["ticker"])
        picks = day[day["ticker"].isin(pick_names)]["fwd_ret_t20"].dropna()
        rows.append({
            "date": d,
            "market_t20": market.mean() * 100,
            "cohort_n": len(cohort),
            "cohort_t20": cohort.mean() * 100 if len(cohort) else None,
            "picks_n": len(picks),
            "picks_t20": picks.mean() * 100 if len(picks) else None,
        })
    g = pd.DataFrame(rows)

    bar = "=" * 78
    print(f"\n{bar}\n GRID-DATE CHECK — T+20 forward returns on engine rebalance dates\n{bar}")
    print(f"  ALL-DAYS baselines: market={trades['fwd_ret_t20'].mean() * 100:+.3f}%   "
          f"liquid cohort={trades[(trades['is_liquid']) & (trades['p_up_lag'] >= SIG_THR)]['fwd_ret_t20'].mean() * 100:+.3f}%")
    print(f"\n  ON GRID DATES (n={len(g)}):")
    print(f"  market mean : {g['market_t20'].mean():+.3f}%")
    print(f"  cohort mean : {g['cohort_t20'].mean():+.3f}%   (mean cohort size {g['cohort_n'].mean():.0f})")
    print(f"  picks  mean : {g['picks_t20'].mean():+.3f}%   (mean picks {g['picks_n'].mean():.1f})")
    print(f"\n  PER-DATE DETAIL (worst 12 by picks):")
    print(g.sort_values("picks_t20").head(12).round(2).to_string(index=False))
    print(bar)


if __name__ == "__main__":
    main()
