"""Regime-conditioned MR-LGBM precision analysis (2026-07-20, READ-ONLY).

Question: does gating the knife-catch signal on `market_regime` improve
precision? MR-LGBM currently has ZERO regime awareness (grep-confirmed —
nothing in mr_features.py or train_mr_lgbm.py touches regime) yet its own
OOF precision (0.578, retrained 17-07) misses its 0.60 design target.
Hypothesis: mixing "oversold in a genuine reversal setup" (regime 5
Mean-Reversion, RSI extreme by definition) with "oversold and still
falling" (regime 6 Choppy / regime 0 Freeze) dilutes precision.

Reuses the EXACT same purged_oof machinery, fold structure, features, and
chronological split as train_mr_lgbm.py — this script only ADDS a
regime-conditioned precision breakdown on top of the same OOF probabilities
production would compute. Zero writes, zero model/artifact changes.

Run: python scripts/analyze_mr_regime_conditioning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.features.market_regime import REGIME_LABELS_EN, build_regime_features  # noqa: E402
from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features  # noqa: E402
from src.models.train_mr_lgbm import (  # noqa: E402
    EMBARGO_BARS,
    N_SPLITS,
    chrono_split,
    label_3d_bounce,
    load_ohlcv,
    purged_oof,
)

SHIPPED_TAU = 0.96  # current production tau* (models/mr/mr_threshold.json)
MIN_RELIABLE_FIRES = 10  # below this, precision is too noisy to read


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building MR features + label ...")
    feat = build_mr_features(ohlcv)
    feat = label_3d_bounce(feat)

    print("Building regime labels (same panel, joined by ticker+date) ...")
    regime_pd = (
        build_regime_features(pl.from_pandas(ohlcv).lazy())
        .select(["ticker", "date", "market_regime"])
        .collect()
        .to_pandas()
    )
    feat = feat.merge(regime_pd, on=["ticker", "date"], how="left")
    n_missing = int(feat["market_regime"].isna().sum())
    if n_missing:
        print(f"  WARNING: {n_missing} rows missing a regime join — dropped")
        feat = feat.dropna(subset=["market_regime"])
    feat["market_regime"] = feat["market_regime"].astype(int)

    train, _test = chrono_split(feat)
    cols = list(MR_FEATURE_COLUMNS)  # unchanged — regime is NOT a training feature

    def X(d: pd.DataFrame) -> np.ndarray:
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    train = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_tr, y_tr = X(train), train["y"].to_numpy(np.int64)
    start = pd.to_datetime(train["date"]).to_numpy()
    end = pd.to_datetime(train["t1"]).to_numpy()

    print(f"Running purged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS}) "
          f"— identical to production ...")
    oof = purged_oof(x_tr, y_tr, start, end)

    mask = np.isfinite(oof)
    regimes = train["market_regime"].to_numpy()[mask]
    p, y = oof[mask], y_tr[mask]

    print(f"\n{'=' * 76}\nOVERALL @ shipped tau*={SHIPPED_TAU:.2f}\n{'=' * 76}")
    fire = p >= SHIPPED_TAU
    n = int(fire.sum())
    tp = int((fire & (y == 1)).sum())
    overall_prec = tp / n if n else float("nan")
    print(f"  n={len(p)}  fires={n}  precision={overall_prec:.3f}  "
          f"recall={tp / max(int(y.sum()), 1):.3f}  base_rate={y.mean():.4f}")

    print(f"\n{'=' * 76}\nPER-REGIME @ SAME shipped tau*={SHIPPED_TAU:.2f}\n{'=' * 76}")
    print(f"{'regime':<18}{'n_rows':>8}{'base_rate':>11}{'fires':>8}"
          f"{'precision':>11}{'recall':>9}  note")
    for r in sorted(set(regimes.tolist())):
        rm = regimes == r
        pr, yr = p[rm], y[rm]
        fr = pr >= SHIPPED_TAU
        nfr = int(fr.sum())
        tpr = int((fr & (yr == 1)).sum())
        prec = tpr / nfr if nfr else float("nan")
        rec = tpr / max(int(yr.sum()), 1)
        label = REGIME_LABELS_EN.get(r, str(r))
        note = "" if nfr >= MIN_RELIABLE_FIRES else f"<{MIN_RELIABLE_FIRES} fires, noisy"
        print(f"{label:<18}{len(pr):>8}{yr.mean():>11.4f}{nfr:>8}"
              f"{prec:>11.3f}{rec:>9.3f}  {note}")

    print(f"\n{'=' * 76}\nPER-REGIME BEST-tau on its OWN grid (min {MIN_RELIABLE_FIRES} fires)\n"
          f"{'=' * 76}")
    print(f"{'regime':<18}{'best_tau':>9}{'fires':>7}{'precision':>11}{'recall':>9}")
    for r in sorted(set(regimes.tolist())):
        rm = regimes == r
        pr, yr = p[rm], y[rm]
        best = None
        for tau in np.arange(0.50, 1.00, 0.01):
            fr = pr >= tau
            nfr = int(fr.sum())
            if nfr < MIN_RELIABLE_FIRES:
                continue
            tpr = int((fr & (yr == 1)).sum())
            prec = tpr / nfr
            if best is None or prec > best[1]:
                best = (tau, prec, tpr / max(int(yr.sum()), 1), nfr)
        label = REGIME_LABELS_EN.get(r, str(r))
        if best:
            print(f"{label:<18}{best[0]:>9.2f}{best[3]:>7}{best[1]:>11.3f}{best[2]:>9.3f}")
        else:
            print(f"{label:<18}{'never clears ' + str(MIN_RELIABLE_FIRES) + ' fires':>36}")

    print(f"\nDone. Overall OOF precision (informational, matches production "
          f"selection logic) recap: {overall_prec:.3f} vs 0.60 target.")


if __name__ == "__main__":
    main()
