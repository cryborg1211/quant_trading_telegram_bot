"""Breadth-inflection MR-LGBM precision analysis (2026-07-22, READ-ONLY).

Third round of the knife-catch-timing investigation (regime-gate and
RSI-direction both REJECTED — see analyze_mr_regime_conditioning.py).
Hypothesis this time: market-wide breadth TURNING UP (a rising trend in the
fraction of the universe with positive trailing returns) predicts a more
reliable MR-LGBM fire than breadth's raw LEVEL — a capitulation bounce is
more likely to hold once the broad market has found a floor, even if the
absolute level is still low.

`src/trading/breadth.py` (the live meta-controller module) only computes
breadth for the LATEST date in a panel — this script needs a full breadth
TIME SERIES across ~10 years of history, so it builds one locally with a
single vectorized polars pass (per-ticker trailing-window return -> per-date
fraction positive), never touching the production module. Both breadth
LEVEL and breadth DELTA (its own trailing change - the "inflection") are
tested, market-wide values broadcast onto every ticker on that date (a pure
backward-looking join, same leak discipline as market_regime/rsi14).

Reuses the EXACT same purged_oof machinery, fold structure, features, and
chronological split as train_mr_lgbm.py. Zero writes, zero model changes.

Run: python scripts/analyze_mr_breadth_inflection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features  # noqa: E402
from src.models.train_mr_lgbm import (  # noqa: E402
    EMBARGO_BARS,
    N_SPLITS,
    _spw,
    chrono_split,
    label_3d_bounce,
    load_ohlcv,
    make_lgbm,
    purged_oof,
)

SHIPPED_TAU = 0.96  # current production tau* (models/mr/mr_threshold.json)
MIN_RELIABLE_FIRES = 10  # below this, precision is too noisy to read
BREADTH_WINDOW = 20  # matches src/trading/breadth.py's production default
DELTA_WINDOW = 5  # "is breadth higher now than 5 sessions ago" — the inflection
MIN_TICKERS_FOR_BREADTH = 30  # per-date reliability floor, matches breadth.py
EXTENDED_HOLDOUT_DAYS = 1095  # ~3 years — the 365-day production holdout was
# too thin (6 vs 8 fires) to confirm/deny the hypothesis independently. This
# is STILL all historical parquet data already on disk — no new data needed,
# just a wider out-of-training confirmation window than production's own
# 1-year convention.


def _breadth_time_series(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Market-wide breadth LEVEL + DELTA for every date in the panel.

    Single vectorized polars pass: trailing BREADTH_WINDOW-session return
    per (ticker, date) -> per-date fraction positive -> DELTA_WINDOW-session
    change of that fraction. Both are strictly backward-looking (no forward
    return anywhere), so joining them onto MR-LGBM rows by date is leak-free
    by construction — same discipline as market_regime/rsi14.
    """
    lf = pl.from_pandas(ohlcv).lazy().sort(["ticker", "date"])
    lf = lf.with_columns(
        (pl.col("close") / pl.col("close").shift(BREADTH_WINDOW).over("ticker") - 1.0)
        .alias("_ret_w")
    )
    per_date = (
        lf.group_by("date")
        .agg([
            (pl.col("_ret_w") > 0).sum().alias("_pos"),
            pl.col("_ret_w").is_not_null().sum().alias("_valid"),
        ])
        .sort("date")
        .collect()
    )
    per_date = per_date.filter(pl.col("_valid") >= MIN_TICKERS_FOR_BREADTH)
    per_date = per_date.with_columns(
        (pl.col("_pos") / pl.col("_valid")).alias("breadth")
    )
    per_date = per_date.with_columns(
        (pl.col("breadth") - pl.col("breadth").shift(DELTA_WINDOW)).alias("breadth_delta")
    )
    return per_date.select(["date", "breadth", "breadth_delta"]).to_pandas()


def _print_bucket_table(title: str, buckets: dict, p: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(f"{'bucket':<26}{'n_rows':>8}{'base_rate':>11}{'fires':>8}"
          f"{'precision':>11}{'recall':>9}  note")
    for label, mask in buckets.items():
        pr, yr = p[mask], y[mask]
        if len(pr) == 0:
            print(f"{label:<26}{'(empty)':>8}")
            continue
        fr = pr >= SHIPPED_TAU
        nfr = int(fr.sum())
        tpr = int((fr & (yr == 1)).sum())
        prec = tpr / nfr if nfr else float("nan")
        rec = tpr / max(int(yr.sum()), 1)
        note = "" if nfr >= MIN_RELIABLE_FIRES else f"<{MIN_RELIABLE_FIRES} fires, noisy"
        print(f"{label:<26}{len(pr):>8}{yr.mean():>11.4f}{nfr:>8}"
              f"{prec:>11.3f}{rec:>9.3f}  {note}")


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building MR features + label ...")
    feat = build_mr_features(ohlcv)
    feat = label_3d_bounce(feat)

    print(f"Building market-wide breadth time series (window={BREADTH_WINDOW}, "
          f"delta_window={DELTA_WINDOW}) ...")
    breadth_pd = _breadth_time_series(ohlcv)
    feat = feat.merge(breadth_pd, on="date", how="left")
    n_missing = int(feat["breadth"].isna().sum())
    if n_missing:
        print(f"  {n_missing} rows before enough breadth history — dropped")
        feat = feat.dropna(subset=["breadth", "breadth_delta"])

    train, test = chrono_split(feat)
    cols = list(MR_FEATURE_COLUMNS)  # unchanged — breadth is NOT a training feature

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
    breadth = train["breadth"].to_numpy()[mask]
    breadth_delta = train["breadth_delta"].to_numpy()[mask]
    p, y = oof[mask], y_tr[mask]

    print(f"\n{'=' * 76}\nOVERALL @ shipped tau*={SHIPPED_TAU:.2f}\n{'=' * 76}")
    fire = p >= SHIPPED_TAU
    n = int(fire.sum())
    tp = int((fire & (y == 1)).sum())
    overall_prec = tp / n if n else float("nan")
    print(f"  n={len(p)}  fires={n}  precision={overall_prec:.3f}  "
          f"recall={tp / max(int(y.sum()), 1):.3f}  base_rate={y.mean():.4f}")

    # ── LEVEL: terciles of breadth at fire time ────────────────────────────
    lo_cut, hi_cut = np.nanpercentile(breadth, [33.3, 66.7])
    _print_bucket_table(
        f"BREADTH LEVEL terciles (cuts at {lo_cut:.3f} / {hi_cut:.3f})",
        {
            f"Low breadth(<{lo_cut:.2f})": breadth < lo_cut,
            f"Mid breadth": (breadth >= lo_cut) & (breadth <= hi_cut),
            f"High breadth(>{hi_cut:.2f})": breadth > hi_cut,
        },
        p, y,
    )

    # ── INFLECTION: is breadth rising, falling, or flat vs DELTA_WINDOW ago ─
    delta_std = np.nanstd(breadth_delta)
    rising = breadth_delta > (0.25 * delta_std)
    falling = breadth_delta < (-0.25 * delta_std)
    flat = ~rising & ~falling
    _print_bucket_table(
        f"BREADTH INFLECTION (delta over {DELTA_WINDOW} sessions, "
        f"+/-0.25 sigma band, sigma={delta_std:.4f})",
        {
            "Rising (turning up)": rising,
            "Flat": flat,
            "Falling (still declining)": falling,
        },
        p, y,
    )

    # ── The specific hypothesis: LOW level + RISING delta (the knife-catch
    #    entry the theory predicts is safest) vs LOW level + FALLING delta
    #    (the one the theory predicts is dangerous — same low level, wrong
    #    direction) ─────────────────────────────────────────────────────────
    low = breadth < lo_cut
    _print_bucket_table(
        "THE HYPOTHESIS: low breadth split by direction",
        {
            "Low + Rising (predicted SAFE)": low & rising,
            "Low + Flat": low & flat,
            "Low + Falling (predicted RISKY)": low & falling,
        },
        p, y,
    )

    # ── HONEST OUT-OF-SAMPLE CHECK: strict chronological hold-out ──────────
    # OOF is cross-validated but still selected/read on the SAME data the
    # model trained on. Fit final model on ALL of train (mirrors
    # train_mr_lgbm.main exactly) and score the untouched 1-year hold-out —
    # cut points (lo_cut, delta_std) are TRAIN-derived and reused unchanged,
    # never recomputed on hold-out (same leak discipline as tau* selection).
    print(f"\n{'=' * 76}\nFitting FINAL model on full train, scoring the STRICT "
          f"1-year HOLD-OUT ...\n{'=' * 76}")
    final = make_lgbm(_spw(y_tr))
    final.fit(x_tr, y_tr)
    x_te = X(test)
    y_te = test["y"].to_numpy(np.int64)
    p_te = final.predict_proba(x_te)[:, 1]
    breadth_te = test["breadth"].to_numpy()
    delta_te = test["breadth_delta"].to_numpy()
    low_te = breadth_te < lo_cut
    rising_te = delta_te > (0.25 * delta_std)
    falling_te = delta_te < (-0.25 * delta_std)
    flat_te = ~rising_te & ~falling_te

    fire_te = p_te >= SHIPPED_TAU
    n_te = int(fire_te.sum())
    tp_te = int((fire_te & (y_te == 1)).sum())
    print(f"  hold-out overall: n={len(p_te)}  fires={n_te}  "
          f"precision={(tp_te / n_te if n_te else float('nan')):.3f}  "
          f"recall={tp_te / max(int(y_te.sum()), 1):.3f}")

    _print_bucket_table(
        "HOLD-OUT: THE HYPOTHESIS (train-derived cuts, never touched hold-out)",
        {
            "Low + Rising (predicted SAFE)": low_te & rising_te,
            "Low + Flat": low_te & flat_te,
            "Low + Falling (predicted RISKY)": low_te & falling_te,
        },
        p_te, y_te,
    )

    # ── EXTENDED confirmatory check: ~3-year out-of-training window ────────
    # Same historical parquets, no new data — just a wider carve-out than
    # production's own 365-day holdout convention, specifically to get
    # enough fires to actually read the hypothesis on unseen data. Cut
    # points (lo_cut, delta_std) are the SAME train-derived values from
    # above — reused unchanged, never peeked at this extended test slice.
    max_d = pd.to_datetime(feat["date"]).max()
    ext_cutoff = (max_d - pd.Timedelta(days=EXTENDED_HOLDOUT_DAYS)).date()
    d_all = pd.to_datetime(feat["date"]).dt.date
    ext_train = feat[d_all < ext_cutoff].reset_index(drop=True)
    ext_test = feat[d_all >= ext_cutoff].reset_index(drop=True)
    print(f"\n{'=' * 76}\nEXTENDED {EXTENDED_HOLDOUT_DAYS}-day (~3yr) out-of-training check "
          f"| cutoff={ext_cutoff}\n{'=' * 76}")
    print(f"  ext_train={len(ext_train)} rows (pos={100*ext_train['y'].mean():.3f}%)  "
          f"ext_test={len(ext_test)} rows (pos={100*ext_test['y'].mean():.3f}%)")

    ext_train = ext_train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_ext_tr = X(ext_train)
    y_ext_tr = ext_train["y"].to_numpy(np.int64)
    ext_final = make_lgbm(_spw(y_ext_tr))
    ext_final.fit(x_ext_tr, y_ext_tr)

    x_ext_te = X(ext_test)
    y_ext_te = ext_test["y"].to_numpy(np.int64)
    p_ext_te = ext_final.predict_proba(x_ext_te)[:, 1]
    breadth_ext_te = ext_test["breadth"].to_numpy()
    delta_ext_te = ext_test["breadth_delta"].to_numpy()
    low_ext_te = breadth_ext_te < lo_cut
    rising_ext_te = delta_ext_te > (0.25 * delta_std)
    falling_ext_te = delta_ext_te < (-0.25 * delta_std)
    flat_ext_te = ~rising_ext_te & ~falling_ext_te

    fire_ext_te = p_ext_te >= SHIPPED_TAU
    n_ext_te = int(fire_ext_te.sum())
    tp_ext_te = int((fire_ext_te & (y_ext_te == 1)).sum())
    print(f"  extended-holdout overall: n={len(p_ext_te)}  fires={n_ext_te}  "
          f"precision={(tp_ext_te / n_ext_te if n_ext_te else float('nan')):.3f}  "
          f"recall={tp_ext_te / max(int(y_ext_te.sum()), 1):.3f}")

    _print_bucket_table(
        f"EXTENDED {EXTENDED_HOLDOUT_DAYS}-day HOLD-OUT: THE HYPOTHESIS "
        f"(same train-derived cuts, never touched this slice)",
        {
            "Low + Rising (predicted SAFE)": low_ext_te & rising_ext_te,
            "Low + Flat": low_ext_te & flat_ext_te,
            "Low + Falling (predicted RISKY)": low_ext_te & falling_ext_te,
        },
        p_ext_te, y_ext_te,
    )

    print(f"\nDone. Overall OOF precision (informational, matches production "
          f"selection logic) recap: {overall_prec:.3f} vs 0.60 target.")


if __name__ == "__main__":
    main()
