"""Mean-Reversion sub-model trainer — single LightGBM (capitulation catcher).

WHY A SEPARATE, SINGLE-MODEL PIPELINE
─────────────────────────────────────
The reversal audit proved the Alpha360 stack is ~81% momentum and blind
to V-bottoms. This trains a DEDICATED knife-catch model on the oversold
feature set from ``src/features/mr_features.py``. Per the design brief:
  • single LightGBM (NOT the 3-model stack) — light inference, far less
    overfit surface on rare capitulation labels.
  • aggressive imbalance handling (scale_pos_weight) + shallow, heavily
    regularized trees.
  • leak-free validation: PurgedKFold OOF on TRAIN for the threshold,
    then a STRICT chronological 1-year hold-out for the honest report.
  • an EXTREMELY strict τ* — fire only on absolute panic
    (high precision, deliberately low recall).

LABEL
─────
``target_return_3d = close[t+3]/close[t] - 1`` (per ticker).
``y = 1 if target_return_3d > +3% else 0``  → explosive 3-day bounce.
The forward window is the prediction TARGET (not a feature leak); the
mr_* features are all backward-looking (audited in mr_features.py).
PurgedKFold purges/embargoes the 3-bar label horizon at fold edges.

ARTIFACTS  (models/mr/)
───────────────────────
  mr_lgbm.joblib          fitted LightGBM
  mr_threshold.json       τ*, label def, feature list, split, metrics
  mr_report.json          full OOF + hold-out precision/recall/confusion
"""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features
from src.models.stacking_model.purged_kfold import PurgedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("train_mr_lgbm")

SEED = 42
OHLCV_GLOB = "data/ohlcv_*.parquet"
ART_DIR = Path("models/mr")
HORIZON = 3                       # forward bars for the bounce label
BOUNCE_THRESHOLD = 0.03          # +3% in 3 days = explosive bounce
HOLDOUT_DAYS = 365               # strict chronological 1-year hold-out
N_SPLITS = 5
EMBARGO_BARS = HORIZON           # >= label horizon (de Prado)
# Threshold selection: only fire on absolute panic.
MIN_FIRES = 40                   # reject degenerate 1-sample "100% precision"
TARGET_PRECISION = 0.60          # aim; report honestly if unreachable
TAU_GRID = np.round(np.arange(0.50, 0.991, 0.01), 4)


@contextmanager
def timed(msg: str):
    t0 = time.perf_counter()
    LOGGER.info("%s ...", msg)
    try:
        yield
    finally:
        LOGGER.info("%s done in %.1fs", msg, time.perf_counter() - t0)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_ohlcv() -> pd.DataFrame:
    """Load the full HOSE OHLCV history from data/ohlcv_*.parquet."""
    import polars as pl

    files = sorted(Path().glob(OHLCV_GLOB))
    if not files:
        raise FileNotFoundError(f"No {OHLCV_GLOB} files found. Run the crawler.")
    LOGGER.info("Loading %s OHLCV parquet files ...", len(files))
    df = (
        pl.scan_parquet([str(p) for p in files])
        .select(["ticker", "date", "open", "high", "low", "close", "volume"])
        .with_columns(
            [
                pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
                pl.col("date").cast(pl.Date),
            ]
        )
        .sort(["ticker", "date"])
        .collect()
        .to_pandas()
    )
    LOGGER.info("OHLCV rows=%s tickers=%s", len(df), df["ticker"].nunique())
    return df


def label_3d_bounce(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_return_3d, y (1 if >+3% in 3 bars), and t1 (event-end
    date for PurgedKFold). Rows without a full +3-bar window are dropped.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False, group_keys=False)
    fwd_close = g["close"].shift(-HORIZON)          # close[t+3]  (TARGET, not a feature)
    df["target_return_3d"] = fwd_close / df["close"] - 1.0
    df["t1"] = g["date"].shift(-HORIZON)            # date[t+3] — label-decided date
    df = df.dropna(subset=["target_return_3d", "t1"]).reset_index(drop=True)
    df["y"] = (df["target_return_3d"] > BOUNCE_THRESHOLD).astype(np.int8)
    return df


def chrono_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split: last HOLDOUT_DAYS = untouched hold-out."""
    max_d = pd.to_datetime(df["date"]).max()
    cutoff = (max_d - timedelta(days=HOLDOUT_DAYS)).date()
    d = pd.to_datetime(df["date"]).dt.date
    train = df[d < cutoff].reset_index(drop=True)
    test = df[d >= cutoff].reset_index(drop=True)
    LOGGER.info(
        "Chrono split @ %s | train=%s (pos=%.3f%%) | holdout=%s (pos=%.3f%%)",
        cutoff, len(train), 100 * train["y"].mean(),
        len(test), 100 * test["y"].mean() if len(test) else 0.0,
    )
    if train.empty or test.empty:
        raise ValueError("Empty train/holdout after chronological split.")
    return train, test


def make_lgbm(scale_pos_weight: float) -> LGBMClassifier:
    """Shallow, heavily-regularized LGBM for a rare-positive target.

    scale_pos_weight (= n_neg / n_pos) handles the ~99% imbalance without
    is_unbalance (more controllable). Depth/leaves kept tiny and
    min_child_samples high so it cannot memorize the few capitulation
    positives.
    """
    return LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=7,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def _spw(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


def purged_oof(
    x: np.ndarray, y: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """Leak-free out-of-fold P(bounce) via PurgedKFold (embargo>=horizon).
    scale_pos_weight is recomputed PER FOLD on that fold's train labels."""
    cv = PurgedKFold(
        n_splits=N_SPLITS, start_times=start, end_times=end,
        embargo_bars=EMBARGO_BARS,
    )
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for k, (tr, va) in enumerate(cv.split(x), 1):
        m = make_lgbm(_spw(y[tr]))
        m.fit(x[tr], y[tr])
        oof[va] = m.predict_proba(x[va])[:, 1]
        LOGGER.info(
            "  OOF fold %s/%s: train=%s val=%s val_pos=%s",
            k, N_SPLITS, len(tr), len(va), int(y[va].sum()),
        )
    # Any rows never in a validation block (purged-degenerate) stay NaN and
    # are excluded from threshold selection.
    return oof


def select_strict_tau(p: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Pick the LOWEST τ whose OOF precision >= TARGET_PRECISION with
    >= MIN_FIRES (keeps a little recall). If the target precision is
    unreachable, fall back to the τ with MAX precision among grid points
    clearing MIN_FIRES — and report that honestly.
    """
    mask = np.isfinite(p)
    p, y = p[mask], y[mask]
    rows, best_fallback = [], None
    for tau in TAU_GRID:
        fire = p >= tau
        n = int(fire.sum())
        if n < MIN_FIRES:
            continue
        tp = int((fire & (y == 1)).sum())
        prec = tp / n
        rec = tp / max(int(y.sum()), 1)
        rows.append((float(tau), prec, rec, n))
        if best_fallback is None or prec > best_fallback[1]:
            best_fallback = (float(tau), prec, rec, n)
    if not rows:
        raise RuntimeError(
            "No τ on the grid cleared MIN_FIRES — model never fires; "
            "loosen MIN_FIRES or revisit features/label."
        )
    hit = [r for r in rows if r[1] >= TARGET_PRECISION]
    chosen = min(hit, key=lambda r: r[0]) if hit else best_fallback
    tau, prec, rec, n = chosen
    info = {
        "tau": tau, "oof_precision": round(prec, 4),
        "oof_recall": round(rec, 4), "oof_fires": n,
        "target_precision": TARGET_PRECISION,
        "target_precision_met": bool(prec >= TARGET_PRECISION),
        "min_fires": MIN_FIRES,
    }
    LOGGER.info(
        "Selected τ*=%.2f | OOF precision=%.3f recall=%.3f fires=%s | "
        "target_met=%s", tau, prec, rec, n, info["target_precision_met"],
    )
    return tau, info


def main() -> None:
    t0 = time.perf_counter()
    seed_everything()
    ART_DIR.mkdir(parents=True, exist_ok=True)

    with timed("Load OHLCV"):
        ohlcv = load_ohlcv()
    with timed("Build MR (oversold) features"):
        feat = build_mr_features(ohlcv)
    with timed("Label 3-day explosive bounce"):
        feat = label_3d_bounce(feat)

    train, test = chrono_split(feat)
    cols = list(MR_FEATURE_COLUMNS)

    def X(d: pd.DataFrame) -> np.ndarray:
        # LightGBM handles NaN natively — no imputation (cleaner, no bias).
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    # Sort train by date for a valid time-ordered PurgedKFold.
    train = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_tr, y_tr = X(train), train["y"].to_numpy(np.int64)
    start = pd.to_datetime(train["date"]).to_numpy()
    end = pd.to_datetime(train["t1"]).to_numpy()

    with timed(f"Purged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS})"):
        oof = purged_oof(x_tr, y_tr, start, end)
    tau, tau_info = select_strict_tau(oof, y_tr)

    with timed("Fit FINAL LGBM on full train"):
        final = make_lgbm(_spw(y_tr))
        final.fit(x_tr, y_tr)

    # ── Honest hold-out evaluation (last 1 year, never seen) ───────────
    x_te, y_te = X(test), test["y"].to_numpy(np.int64)
    p_te = final.predict_proba(x_te)[:, 1]
    fire_te = p_te >= tau
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_te, fire_te.astype(int), average="binary", zero_division=0
    )
    cm = confusion_matrix(y_te, fire_te.astype(int), labels=[0, 1])
    ap = average_precision_score(y_te, p_te) if y_te.sum() else 0.0
    holdout = {
        "n": int(len(y_te)), "pos": int(y_te.sum()),
        "fires": int(fire_te.sum()),
        "precision": round(float(pr), 4), "recall": round(float(rc), 4),
        "f1": round(float(f1), 4), "avg_precision": round(float(ap), 4),
        "confusion_matrix": cm.tolist(),
    }
    LOGGER.info(
        "HOLD-OUT @ τ*=%.2f | fires=%s precision=%.3f recall=%.3f "
        "(pos_base_rate=%.3f%%)",
        tau, holdout["fires"], pr, rc, 100 * y_te.mean() if len(y_te) else 0,
    )

    # ── Persist artifacts ──────────────────────────────────────────────
    joblib.dump(final, ART_DIR / "mr_lgbm.joblib")
    threshold_doc = {
        "model": "mr_lgbm_single",
        "tau": tau,
        "label": {
            "name": "target_return_3d",
            "horizon_bars": HORIZON,
            "rule": f"y=1 if 3-bar fwd return > {BOUNCE_THRESHOLD:+.0%}",
        },
        "features": cols,
        "split": {"type": "chronological", "holdout_days": HOLDOUT_DAYS},
        "selection": tau_info,
        "holdout": holdout,
    }
    with (ART_DIR / "mr_threshold.json").open("w", encoding="utf-8") as fh:
        json.dump(threshold_doc, fh, indent=2)
    with (ART_DIR / "mr_report.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "train_rows": int(len(train)),
                "train_pos_rate": float(y_tr.mean()),
                "oof_selection": tau_info,
                "holdout": holdout,
                "lgbm_params": final.get_params(),
            },
            fh, indent=2, default=str,
        )
    LOGGER.info(
        "Artifacts saved to %s | total %.1fs", ART_DIR,
        time.perf_counter() - t0,
    )
    if not tau_info["target_precision_met"]:
        LOGGER.warning(
            "⚠️ τ* did NOT reach target precision %.2f (best OOF %.3f). "
            "Capitulation bounces are hard — deploy only with eyes open.",
            TARGET_PRECISION, tau_info["oof_precision"],
        )


if __name__ == "__main__":
    main()
