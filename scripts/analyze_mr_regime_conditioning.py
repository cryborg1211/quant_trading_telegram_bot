"""Regime-conditioned MR-LGBM precision analysis (2026-07-20/22, READ-ONLY).

Question: does gating the knife-catch signal on `market_regime` improve
precision? MR-LGBM currently has ZERO regime awareness (grep-confirmed —
nothing in mr_features.py or train_mr_lgbm.py touches regime) yet its own
OOF precision (0.578, retrained 17-07) misses its 0.60 design target.

ROUND 1 (22-07) result: REJECTED. Gating on regime==5 (Mean-Reversion) at
the raw label underperformed (precision 0.563) both the overall rate
(0.576) and Choppy/regime-6 (0.601, also 59% of all rows). Root cause
found: regime 5 is BIDIRECTIONAL by construction — market_regime.py's
`rsi_extreme = (_rsi > 70) | (_rsi < 30)` fires the SAME label for
overbought (>70) and oversold (<30) tape, so half the bucket is irrelevant
to catching a bottom, diluting its precision reading.

ROUND 2 (this run) — direction-aware split: `_rsi` is a SCRATCH column
market_regime.py drops before returning (never leaked to callers), so this
script mirrors the EXACT SAME 14-period Wilder RSI formula (same rolling
gain/loss windows, same `_EPS`) independently, purely for this READ-ONLY
analysis — market_regime.py itself is untouched. Splits regime-5 rows into
oversold (<30) vs overbought (>70) — mutually exclusive by construction —
AND, as the more direct test of "when to catch the knife," splits ALL rows
(any regime) the same way, since raw RSI-direction may be a cleaner signal
than the regime label it feeds into.

Reuses the EXACT same purged_oof machinery, fold structure, features, and
chronological split as train_mr_lgbm.py. Zero writes, zero model/artifact
changes.

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
_RSI_EPS = 1e-12  # matches market_regime.py's _EPS exactly


def _mirror_rsi14(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Independent replica of market_regime.py's internal 14-period RSI.

    That module computes `_rsi` as a SCRATCH column (SIMPLE 14-bar rolling
    mean of gain/loss — NOT Wilder's exponential smoothing, despite the
    common name) and drops it before returning — never leaked to callers.
    Mirrors the exact same formula (same windows, same `_EPS`) purely for
    this read-only analysis; market_regime.py itself is untouched.
    """
    df = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)
    diff = df.groupby("ticker")["close"].diff()
    gain = diff.where(diff > 0, 0.0)
    loss = (-diff).where(diff < 0, 0.0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda s: s.rolling(14).mean())
    df["rsi14"] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + _RSI_EPS))
    return df[["ticker", "date", "rsi14"]]


def _print_bucket_table(title: str, buckets: dict, p: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(f"{'bucket':<22}{'n_rows':>8}{'base_rate':>11}{'fires':>8}"
          f"{'precision':>11}{'recall':>9}  note")
    for label, mask in buckets.items():
        pr, yr = p[mask], y[mask]
        if len(pr) == 0:
            print(f"{label:<22}{'(empty)':>8}")
            continue
        fr = pr >= SHIPPED_TAU
        nfr = int(fr.sum())
        tpr = int((fr & (yr == 1)).sum())
        prec = tpr / nfr if nfr else float("nan")
        rec = tpr / max(int(yr.sum()), 1)
        note = "" if nfr >= MIN_RELIABLE_FIRES else f"<{MIN_RELIABLE_FIRES} fires, noisy"
        print(f"{label:<22}{len(pr):>8}{yr.mean():>11.4f}{nfr:>8}"
              f"{prec:>11.3f}{rec:>9.3f}  {note}")


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

    print("Mirroring the internal RSI14 (regime-5's own oversold/overbought split) ...")
    rsi_pd = _mirror_rsi14(ohlcv)
    feat = feat.merge(rsi_pd, on=["ticker", "date"], how="left")

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
    rsi14 = train["rsi14"].to_numpy()[mask]
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

    # ── ROUND 2: direction-aware split ────────────────────────────────────
    rsi_ok = np.isfinite(rsi14)
    oversold = rsi_ok & (rsi14 < 30.0)
    overbought = rsi_ok & (rsi14 > 70.0)
    neither = rsi_ok & ~oversold & ~overbought

    _print_bucket_table(
        "REGIME-5 ONLY, split by direction (mutually exclusive by construction)",
        {
            "MeanRev-Oversold(<30)": oversold & (regimes == 5),
            "MeanRev-Overbought(>70)": overbought & (regimes == 5),
        },
        p, y,
    )

    _print_bucket_table(
        "ALL ROWS (any regime), split by RSI14 direction — the more direct test",
        {
            "Oversold RSI14<30": oversold,
            "Overbought RSI14>70": overbought,
            "Neither (30-70)": neither,
        },
        p, y,
    )

    print(f"\n{'=' * 76}\nOVERSOLD-ONLY (any regime) BEST-tau on its own grid "
          f"(min {MIN_RELIABLE_FIRES} fires)\n{'=' * 76}")
    pr, yr = p[oversold], y[oversold]
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
    if best:
        print(f"  best_tau={best[0]:.2f}  fires={best[3]}  "
              f"precision={best[1]:.3f}  recall={best[2]:.3f}")
    else:
        print(f"  never clears {MIN_RELIABLE_FIRES} fires at any tau")

    print(f"\nDone. Overall OOF precision (informational, matches production "
          f"selection logic) recap: {overall_prec:.3f} vs 0.60 target.")


if __name__ == "__main__":
    main()
