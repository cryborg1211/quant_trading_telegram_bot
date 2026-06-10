"""Reconcile edge_profiler vs walk-forward engine: liquidity-stratified edge.

The profiler found +2.0% net T+20 at p_up >= 0.45 over ALL 351 tickers.
The engine only trades the top-50 ADV universe and lost money.
Hypothesis: the per-row edge lives in illiquid small-caps the engine
(correctly) refuses to trade.

Replicates the engine's exact gate: adv20 = trailing 20d mean(close*volume),
shifted 1 day (leak-safe), ranked within date, top-50 kept.

Run after edge_profiler.py has written data/edge_profiler_trades.parquet:
    python scripts/edge_liquidity_check.py
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

TRADES_CACHE = Path("data/edge_profiler_trades.parquet")
_OHLCV_GLOB = (_REPO / "data" / "ohlcv_*.parquet").as_posix()
ROUND_TRIP = FeeSchedule().round_trip_pct() * 100  # ~0.43%
TOP_N = 50


def _load_adv_ranks() -> pd.DataFrame:
    """Per (ticker, date) within-date ADV rank, mirroring WalkForwardEngine."""
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


def _edge_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}  (n={len(df):,})")
    rows = []
    for thr in [0.40, 0.43, 0.45, 0.47]:
        sub = df[df["p_up"] >= thr]
        if len(sub) == 0:
            rows.append({"threshold": thr, "n": 0})
            continue
        row = {"threshold": thr, "n": len(sub),
               "n_tickers": sub["ticker"].nunique()}
        for h in [5, 20]:
            valid = sub[f"fwd_ret_t{h}"].dropna()
            if len(valid):
                gross = valid.mean() * 100
                row[f"gross_t{h}"] = round(gross, 3)
                row[f"net_t{h}"] = round(gross - ROUND_TRIP, 3)
                row[f"wr_t{h}"] = round((valid > 0).mean() * 100, 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    if not TRADES_CACHE.exists():
        print(f"[error] {TRADES_CACHE} missing — run edge_profiler.py first.")
        sys.exit(1)

    trades = pd.read_parquet(TRADES_CACHE)
    print(f"Trade frame: {len(trades):,} rows")

    ranks = _load_adv_ranks()
    print(f"ADV rank table: {len(ranks):,} rows")

    trades = trades.merge(ranks, on=["ticker", "date"], how="left")
    trades["is_liquid"] = trades["adv_rank"] <= TOP_N
    n_liq = int(trades["is_liquid"].sum())
    print(f"Liquid (top-{TOP_N} ADV): {n_liq:,} rows "
          f"({n_liq / len(trades) * 100:.1f}%)  |  unmatched adv_rank: "
          f"{int(trades['adv_rank'].isna().sum()):,}")

    bar = "=" * 80
    print(f"\n{bar}")
    print(f" EDGE BY LIQUIDITY UNIVERSE (engine gate: top-{TOP_N} trailing-20d ADV)")
    print(bar)
    _edge_table(trades[trades["is_liquid"]], f"LIQUID — top-{TOP_N} ADV (the engine's tradable universe)")
    _edge_table(trades[~trades["is_liquid"]], "ILLIQUID — rest of market (engine never trades these)")

    # Where do the 0.45+ signals actually live?
    hot = trades[trades["p_up"] >= 0.45]
    if len(hot):
        liq_share = hot["is_liquid"].mean() * 100
        print(f"\n  p_up >= 0.45 cohort: {len(hot):,} signals, "
              f"only {liq_share:.1f}% inside the liquid universe")
        liq_hot = hot[hot["is_liquid"]]
        if len(liq_hot):
            valid = liq_hot["fwd_ret_t20"].dropna()
            print(f"  liquid 0.45+ signals: n={len(liq_hot):,}  "
                  f"net_t20={valid.mean() * 100 - ROUND_TRIP:+.3f}%  "
                  f"({liq_hot['ticker'].nunique()} tickers)")
    print(bar)


if __name__ == "__main__":
    main()
