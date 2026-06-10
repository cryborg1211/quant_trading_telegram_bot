"""Train/serve score parity: engine `_inference` p_up vs profiler p_up.

Both paths score the SAME (ticker, date) rows with the SAME seed-42 ensemble:
  - profiler: aligned.X rows from materialize_dataset (train-path features)
  - engine:   per-day tensors built from the panel inside WalkForwardEngine

If the two disagree, the live serve path trades on different scores than any
offline analysis predicts — silent alpha decay.

Run after engine_pnl_attribution.py has dumped engine_inference_scores.parquet:
    python scripts/score_parity_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

ENGINE_SCORES = Path("data/engine_inference_scores.parquet")
PROFILER_TRADES = Path("data/edge_profiler_trades.parquet")


def main() -> None:
    eng = pd.read_parquet(ENGINE_SCORES)
    prof = pd.read_parquet(PROFILER_TRADES)
    prof = prof[["ticker", "date", "p_up"]].rename(columns={"p_up": "p_up_profiler"})

    # The engine deliberately scores features through D-1 for decision day D
    # (walk_forward._inference uses `date < D`).  The apples-to-apples join is
    # therefore engine(D) vs profiler(previous trading row per ticker).
    prof = prof.sort_values(["ticker", "date"])
    prof["next_date"] = prof.groupby("ticker", sort=False)["date"].shift(-1)
    prof_lag = prof.dropna(subset=["next_date"]).copy()
    prof_lag["date"] = prof_lag["next_date"].astype(str)
    prof_lag = prof_lag[["ticker", "date", "p_up_profiler"]]

    m = eng.merge(prof_lag, on=["ticker", "date"], how="inner")
    print(f"engine rows: {len(eng):,}   joined (1-bar-lag basis): {len(m):,}")
    if len(m) == 0:
        print("[error] empty join — date formats?")
        sys.exit(1)

    diff = m["p_up_engine"] - m["p_up_profiler"]
    corr = m["p_up_engine"].corr(m["p_up_profiler"])

    bar = "=" * 70
    print(f"\n{bar}\n TRAIN/SERVE SCORE PARITY\n{bar}")
    print(f"  pearson corr            : {corr:.6f}")
    print(f"  mean |diff|             : {diff.abs().mean():.6f}")
    print(f"  p95 |diff|              : {diff.abs().quantile(0.95):.6f}")
    print(f"  max |diff|              : {diff.abs().max():.6f}")
    exact = (diff.abs() < 1e-6).mean() * 100
    print(f"  rows identical (<1e-6)  : {exact:.1f}%")

    # Disagreement on the dispatch decision itself
    thr = 0.40
    eng_yes = m["p_up_engine"] >= thr
    prof_yes = m["p_up_profiler"] >= thr
    flip = (eng_yes != prof_yes).mean() * 100
    print(f"  dispatch flips @ {thr:.2f}  : {flip:.2f}% of rows")

    bad = m[(m["p_up_engine"] >= thr) & (m["p_up_profiler"] < 0.36)]
    print(f"  engine>= {thr:.2f} but profiler < 0.36 : {len(bad):,} rows")
    if len(bad):
        print("\n  WORST MISMATCHES (engine says trade, profiler says decile-1):")
        worst = bad.assign(diff=(bad["p_up_engine"] - bad["p_up_profiler"]).abs())
        print(worst.nlargest(10, "diff").to_string(index=False))
    print(bar)


if __name__ == "__main__":
    main()
