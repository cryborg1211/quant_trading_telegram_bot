"""Implementable edge: score at D, enter at close(D+1) — the live-bot reality.

The model is trained (and the profiler measures) returns from close(D) where D
is the feature bar.  But the live bot computes signals at 15:30 after close(D)
and can only enter on D+1.  This script shifts entry one bar forward and
re-runs the threshold/decile study: whatever survives is harvestable alpha;
the rest is a training-label artifact.

Run (needs data/edge_profiler_trades.parquet from edge_profiler.py):
    python scripts/implementable_edge_check.py
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


def _table(df: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}  (n={len(df):,})")
    rows = []
    for thr in [0.40, 0.43, 0.45, 0.47]:
        sub = df[df["p_up"] >= thr]
        if len(sub) == 0:
            rows.append({"threshold": thr, "n": 0})
            continue
        row = {"threshold": thr, "n": len(sub)}
        for col, tag in [("fwd_ret_t20", "sameday"), ("fwd_ret_t20_next", "nextbar")]:
            valid = sub[col].dropna()
            if len(valid):
                gross = valid.mean() * 100
                row[f"{tag}_gross"] = round(gross, 3)
                row[f"{tag}_net"] = round(gross - ROUND_TRIP, 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    trades = pd.read_parquet(TRADES)
    trades = trades.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Next-bar entry: this row's signal, NEXT row's close-to-close T+20 return.
    for h in [5, 20]:
        trades[f"fwd_ret_t{h}_next"] = (
            trades.groupby("ticker", sort=False)[f"fwd_ret_t{h}"].shift(-1)
        )

    ranks = _adv_ranks()
    trades = trades.merge(ranks, on=["ticker", "date"], how="left")
    trades["is_liquid"] = trades["adv_rank"] <= TOP_N

    bar = "=" * 80
    print(f"{bar}\n IMPLEMENTABLE EDGE — signal at D, entry close(D) vs close(D+1)\n{bar}")
    print(f" round-trip cost: {ROUND_TRIP:.3f}%")

    _table(trades[trades["is_liquid"]], f"LIQUID top-{TOP_N} (engine/bot tradable universe)")
    _table(trades, "FULL MARKET (reference)")

    # Decile view, liquid only, next-bar entry
    liq = trades[trades["is_liquid"]]
    print(f"\n  LIQUID DECILE VIEW (next-bar entry, T+20):")
    rows = []
    for d in sorted(liq["score_decile"].dropna().unique()):
        sub = liq[liq["score_decile"] == d]
        valid = sub["fwd_ret_t20_next"].dropna()
        same = sub["fwd_ret_t20"].dropna()
        rows.append({
            "decile": int(d), "n": len(sub),
            "sameday_gross": round(same.mean() * 100, 3),
            "nextbar_gross": round(valid.mean() * 100, 3),
            "nextbar_net": round(valid.mean() * 100 - ROUND_TRIP, 3),
            "decay_pp": round((same.mean() - valid.mean()) * 100, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(bar)


if __name__ == "__main__":
    main()
