#!/usr/bin/env python3
"""Strict OOS backtest of the CERTIFIED 5d model — last ~2 years.

Window  : 2024-05-01 -> 2026-05-15 (as requested)
Model   : models/stacking/5d ONLY  (xgb+lgb+cat -> meta_model ->
          meta_labeler), with the cost-aware tau* from the artifact.
          The 20d model is ignored entirely.
Costs   : 0.8% round-trip (round_trip_cost(): 2*fee + 2*slip), the EXACT
          same economic_report() the trainer/gate use — zero skew.
Gate    : DUAL — long iff  P(UP) >= tau*  AND  meta-labeler P(profit) >= 0.5
          (identical to main.py predict_stacking_horizon).
Truth   : gross trade return = target_return_5d (triple-barrier close->t1
          realized return already in the feature parquet).

HONESTY GUARDRAIL
─────────────────
CONFIG.training.split_date is the train/test boundary. Any backtest row
with date < split_date was IN the training set (the model saw its label).
This script prints THREE blocks:
  (A) FULL requested window           2024-05-01 .. 2026-05-15
  (B) IN-SAMPLE contamination slice   2024-05-01 .. split_date-1
  (C) STRICT OOS (the number to trust) split_date .. 2026-05-15
Trust (C). (A) is shown only because it was explicitly requested.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from config.settings import CONFIG
from src.models.stacking_model.economic_metrics import (
    N_CLOSE_LAGS_FOR_META,
    economic_report,
    meta_label_feature_matrix,
    round_trip_cost,
)

FEATURES_PARQUET = Path("data/alpha360_features.parquet")
ART = Path("models/stacking/5d")
WIN_START = date(2024, 5, 1)
WIN_END = date(2026, 5, 15)
HORIZON = 5
RET_COL = "target_return_5d"
DATE_COL, TICKER_COL = "date", "ticker"
CLASSES = np.array([0, 1, 2], dtype=np.int64)


def aligned_proba(model, x: np.ndarray) -> np.ndarray:
    """VERBATIM from main.py:aligned_proba — guarantees [P(DOWN),P(SIDE),P(UP)]
    column order regardless of each estimator's internal class ordering."""
    probs = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = getattr(model, "classes_", CLASSES)
    out = np.zeros((x.shape[0], 3), dtype=np.float32)
    for idx, cls in enumerate(classes):
        c = int(cls)
        if c in (0, 1, 2):
            out[:, c] = probs[:, idx]
    denom = out.sum(axis=1, keepdims=True)
    return out / np.where(denom == 0.0, 1.0, denom)


def _summary(tag: str, decisions: np.ndarray, realized: np.ndarray) -> dict:
    rep = economic_report(decisions, realized, HORIZON)  # 0.8% RT inside
    print(f"\n===== {tag} =====")
    print(f"  Total trades   : {rep['n_trades']}")
    print(f"  Win rate       : {rep['hit_rate'] * 100:.2f}%")
    print(f"  Net P&L (sum r) : {rep['net_pnl']:+.4f}")
    print(f"  Net Sharpe      : {rep['net_sharpe']:+.3f}")
    print(f"  Avg net/trade   : {rep['avg_net_trade'] * 100:+.3f}%  "
          f"(gross {rep['avg_gross_trade'] * 100:+.3f}%, "
          f"cost {rep['cost_per_trade'] * 100:.2f}%/RT)")
    return rep


def main() -> None:
    # ── split_date (the in-sample / OOS boundary) ──────────────────────
    split_dt = datetime.strptime(CONFIG.training.split_date, "%Y-%m-%d").date()
    print(f"CONFIG.training.split_date = {split_dt}  "
          f"(rows before this date were IN the training set)")
    rt = round_trip_cost()
    print(f"Round-trip friction       = {rt * 100:.2f}%  "
          f"(economic_report, horizon={HORIZON})")

    # ── load CERTIFIED 5d artifacts ONLY ───────────────────────────────
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PARQUET}; run build_alpha360.")
    feats = json.load((ART / "selected_features.json").open())
    thr = json.load((ART / "quantile_thresholds.json").open())
    tau = float(thr["pnl_threshold_tau"])
    xgb = joblib.load(ART / "xgboost_model.joblib")
    lgbm = joblib.load(ART / "lightgbm_model.joblib")
    cat = CatBoostClassifier()
    cat.load_model(str(ART / "catboost_model.cbm"))
    meta = joblib.load(ART / "meta_model.joblib")
    ml_path = ART / "meta_labeler.joblib"
    meta_labeler = joblib.load(ml_path) if ml_path.exists() else None
    print(f"Loaded 5d artifacts | tau*={tau:.2f} | "
          f"meta_labeler={'ON' if meta_labeler else 'OFF'} | "
          f"selected_features={len(feats)}")

    # ── load + window the feature matrix (Alpha360+Macro+Sentiment) ────
    lag_cols = [f"close_{i}" for i in range(N_CLOSE_LAGS_FOR_META)]
    need = list(dict.fromkeys(
        [DATE_COL, TICKER_COL, RET_COL, *feats, *lag_cols]
    ))
    df = (
        pl.scan_parquet(FEATURES_PARQUET)
        .select(need)
        .with_columns(pl.col(DATE_COL).cast(pl.Date))
        .filter(
            (pl.col(DATE_COL) >= WIN_START) & (pl.col(DATE_COL) <= WIN_END)
        )
        .drop_nulls([RET_COL])  # need a realized triple-barrier outcome
        .collect()
        .sort([DATE_COL, TICKER_COL])
    )
    if df.height == 0:
        raise ValueError("No rows in the 2024-05-01..2026-05-15 window.")
    print(f"\nWindow rows (with realized 5d outcome): {df.height:,} | "
          f"{df[DATE_COL].min()} .. {df[DATE_COL].max()} | "
          f"tickers={df[TICKER_COL].n_unique()}")

    # ── inference: EXACT main.py predict_stacking_horizon path ─────────
    x_raw = df.select(feats).to_pandas()
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    x_input = (
        x_raw.fillna(x_raw.median(numeric_only=True)).fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    base_meta = np.hstack([
        aligned_proba(xgb, x_input),
        aligned_proba(lgbm, x_input),
        aligned_proba(cat, x_input),
    ]).astype(np.float32)
    meta_probs = aligned_proba(meta, base_meta)
    p_up = meta_probs[:, 2]

    # dual gate
    primary = p_up >= tau
    if meta_labeler is not None:
        close_lags = (
            df.select(lag_cols)
            .to_pandas()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        x_meta = meta_label_feature_matrix(meta_probs, tau, close_lags)
        p_profit = np.asarray(
            meta_labeler.predict_proba(x_meta), dtype=np.float64
        )[:, 1]
        decisions = primary & (p_profit >= 0.5)
    else:
        decisions = primary

    realized = df[RET_COL].to_numpy().astype(np.float64)
    dates = df[DATE_COL].to_numpy()
    oos_mask = dates >= np.datetime64(split_dt)
    ins_mask = ~oos_mask

    print(f"\nGate firing (dual: P(UP)>={tau:.2f} AND P(profit)>=0.5): "
          f"{int(decisions.sum())} / {df.height:,} rows")

    # ── THREE blocks ───────────────────────────────────────────────────
    _summary("(A) FULL REQUESTED WINDOW 2024-05-01..2026-05-15  "
             "[CONTAINS IN-SAMPLE — do not trust blindly]",
             decisions, realized)
    _summary("(B) IN-SAMPLE CONTAMINATION  2024-05-01..%s  "
             "[model trained on these labels]" % (split_dt - timedelta(days=1)),
             decisions & ins_mask, realized)
    rep_oos = _summary(
        "(C) STRICT OOS  %s..2026-05-15  [** THE NUMBER TO TRUST **]"
        % split_dt, decisions & oos_mask, realized)

    print("\n" + "=" * 64)
    verdict = (
        "SURVIVES the 2024-2026 chop — net Sharpe positive OOS"
        if rep_oos["net_sharpe"] > 0 and not rep_oos["no_trades"]
        else "DOES NOT survive OOS — net Sharpe <= 0 or no trades"
    )
    print(f"VERDICT (strict OOS, block C): 5d model {verdict}.")
    print("=" * 64)


if __name__ == "__main__":
    main()
