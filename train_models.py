"""
train_models.py — V4.0 "Heavy Lifter".

The expensive, infrequent half of the pipeline.  Run this ONLY when the data,
features, labels, or model architecture change — NOT when you are merely
tuning walk-forward / threshold-sweep parameters (that's `run_backtest.py`,
which is fast because it never retrains the GBMs).

Pipeline
────────
    1. Materialize the dataset            (ingest → features → labels → align)
    2. Chronological train / OOS split    (RunConfig.train_frac)
    3. Iron-fist feature selection        (collinearity |r|>0.65 → MI → top-3)
                                          ON THE TRAIN SPLIT ONLY
    4. Train the Macro HMM Oracle         (in-sample market proxy < cutoff)
    5. Train the TabularEnsemble          (LGB+XGB+CAT → LogReg) × N seeds
    6. Dump a pure training checkpoint    → models/saved/v3_training_checkpoint.joblib

The checkpoint is the ONLY hand-off to `run_backtest.py`.  It carries:
    • ensembles            — list[(seed, TabularEnsemble)]
    • macro_hmm            — the trained HMM regime Oracle (or None)
    • tabular_features     — the FINAL selected feature list (post iron-fist)
    • cutoff               — the train/OOS split date
    • train_cfg            — the RunConfig used (so the evaluator rebuilds an
                             IDENTICAL dataset before scoring)

V4.0 defaults (see src/backtest/pipeline.RunConfig): T+20 horizon, PT=3.0σ,
SL=2.0σ, 4 seeds, VN50 (liquid_top_n=50) gate.

Run
───
    python train_models.py
    python train_models.py --start-date 2018-01-01 --n-configs 4 --tb-horizon 20
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.backtest.pipeline import (
    FEATURE_RECIPE_VERSION,
    effective_recipe_version,
    RunConfig,
    configure_logging,
    phase,
    materialize_dataset,
    select_features,
)
from src.models.tabular_ensemble import TabularEnsemble
from src.models.macro_risk_hmm import (
    build_market_proxy_returns,
    build_regime_observation,
    train_macro_risk_hmm,
)

LOGGER = logging.getLogger("quant.train")

CHECKPOINT_PATH = Path("models/saved/v3_training_checkpoint.joblib")
CHECKPOINT_SCHEMA = "v4-train-ckpt-1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble training (the expensive step — ~3-4 min per seed on the full panel)
# ─────────────────────────────────────────────────────────────────────────────

def train_tabular_ensemble(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                           start_times: np.ndarray, end_times: np.ndarray,
                           tabular_features: list[str], cfg: RunConfig,
                           seed: int,
                           categorical_features: list[str] | None = None) -> TabularEnsemble:
    """V4: pure tabular stacking ensemble (LightGBM + XGBoost + CatBoost → LogReg).

    Uses purged time-K-fold (embargo = `cfg.tb_horizon` bars) to generate
    out-of-fold base-learner P(UP) predictions, fits the LogisticRegression
    meta-stacker on the OOF matrix, then refits each base learner on the FULL
    train set.  Architecture: src/models/tabular_ensemble.py.

    `categorical_features` (e.g. ["market_regime"]) are declared to LightGBM /
    CatBoost for NATIVE categorical splits; the value is pickled with the
    ensemble so the live bot needs no special handling at serve time.
    """
    ensemble = TabularEnsemble(
        feature_names=list(tabular_features),
        categorical_features=list(categorical_features or []),
        n_folds=5,
        embargo_bars=cfg.tb_horizon,                # purge = label horizon (AFML §7.4)
        seed=seed,
    )
    ensemble.fit(X, y, start_times=start_times, end_times=end_times, sample_weight=w)
    return ensemble


def main(cfg: RunConfig, out_path: Path = CHECKPOINT_PATH) -> None:
    configure_logging()
    np.random.seed(cfg.seed)
    t_start = time.perf_counter()

    LOGGER.info("=" * 78)
    LOGGER.info(" TRAIN_MODELS (Heavy Lifter) | horizon=T+%d  PT=%.1fσ  SL=%.1fσ  seeds=%d",
                cfg.tb_horizon, cfg.tb_pt, cfg.tb_sl, max(1, cfg.n_configs))
    LOGGER.info("=" * 78)

    # ── 1-2. Materialize + chronological split ───────────────────────────────
    ds = materialize_dataset(cfg)

    # ── 3. Iron-fist feature selection (TRAIN SPLIT ONLY) ────────────────────
    # Mutates ds.aligned.X → the selected columns; returns the final feature list.
    with phase("Phase 1.7 — iron-fist feature selection (collinearity + MI)"):
        ds.aligned, tabular_features = select_features(
            ds.aligned, ds.train_mask,
            all_features=ds.all_features,
            original_features=ds.original_features,
            candidate_features=ds.candidate_features,
            categorical_features=ds.categorical_features,   # market_regime — forced-survive
            corr_threshold=0.65,
            top_k=3,
        )

    # ── 4. Macro Risk Oracle: 2-state HMM on the IN-SAMPLE market proxy ──────
    macro_hmm = None
    if cfg.use_macro_hmm:
        with phase("Macro Risk Oracle — train HMM (in-sample)"):
            obs = build_regime_observation(
                ds.panel, use_macro=cfg.use_macro_in_hmm, macro_parquet=cfg.macro_parquet)
            cutoff_ts = pd.Timestamp(ds.cutoff)
            train_ret = obs[obs.index < cutoff_ts]
            try:
                macro_hmm = train_macro_risk_hmm(
                    train_ret, n_states=cfg.hmm_n_states, seed=cfg.seed)
                # Leak-free filtered P(Bull) — logged here for sanity; the
                # evaluator recomputes it from this same HMM at backtest time.
                p_bull = macro_hmm.p_bull_series(obs, filtered=True)
                oos_pb = p_bull[p_bull.index >= cutoff_ts]
                LOGGER.info("HMM P(Bull) | bull_state=%d  OOS mean=%.3f  OOS min=%.3f",
                            macro_hmm.bull_state, float(oos_pb.mean()), float(oos_pb.min()))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Macro HMM unavailable (%s) — bot will run full exposure.", exc)
                macro_hmm = None

    # ── 5. Train ensembles ONCE per seed (the expensive step) ────────────────
    seeds = [cfg.seed + k for k in range(max(1, cfg.n_configs))]
    X_tr = ds.aligned.X[ds.train_mask]
    y_tr = ds.aligned.y[ds.train_mask]
    w_tr = ds.aligned.w[ds.train_mask]
    st_tr = ds.aligned.dates[ds.train_mask]
    et_tr = ds.aligned.t1[ds.train_mask]
    LOGGER.info("Training pool | train_rows=%d  features=%d  seeds=%s",
                len(X_tr), len(tabular_features), seeds)

    ensembles: list[tuple[int, TabularEnsemble]] = []
    for ci, seed in enumerate(seeds):
        with phase(f"Train {ci + 1}/{len(seeds)} (seed={seed}) — ensemble fit"):
            ens = train_tabular_ensemble(
                X_tr, y_tr, w_tr, st_tr, et_tr, tabular_features, cfg, seed,
                categorical_features=ds.categorical_features)
            ensembles.append((seed, ens))

    # ── 6. Dump the pure training checkpoint ─────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "ensembles": ensembles,                 # list[(seed, TabularEnsemble)]
        "macro_hmm": macro_hmm,                  # trained HMM regime Oracle (or None)
        "tabular_features": list(tabular_features),   # FINAL selected pool (order is load-bearing)
        "categorical_features": list(ds.categorical_features),  # GBM-native categorical subset
        "cutoff": ds.cutoff,                     # train/OOS split boundary
        "train_cfg": cfg,                        # RunConfig — evaluator rebuilds the SAME dataset
        "metadata": {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "feature_schema_hash": effective_recipe_version(cfg.use_macro_features),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "tb_horizon": int(cfg.tb_horizon),
            "tb_pt": float(cfg.tb_pt),
            "tb_sl": float(cfg.tb_sl),
            "train_frac": float(cfg.train_frac),
            "n_features": len(tabular_features),
            "n_train_rows": int(ds.train_mask.sum()),
            "n_oos_rows": int((~ds.train_mask).sum()),
            "has_macro_hmm": macro_hmm is not None,
        },
    }
    joblib.dump(checkpoint, out_path, compress=3)
    size_kb = out_path.stat().st_size / 1024

    LOGGER.info("=" * 78)
    LOGGER.info(" ✔ TRAINING CHECKPOINT PERSISTED → %s  (%.1f KB)", out_path, size_kb)
    LOGGER.info("   seeds=%s  features=%d  cutoff=%s  HMM=%s",
                seeds, len(tabular_features), ds.cutoff, macro_hmm is not None)
    LOGGER.info("   Next:  python run_backtest.py   (fast sweep + persist the live-bot payload)")
    LOGGER.info(" Wall-clock: %.1fs", time.perf_counter() - t_start)
    LOGGER.info("=" * 78)


def _cli() -> "tuple[RunConfig, Path]":
    p = argparse.ArgumentParser(
        description="V4.0 Heavy Lifter — train the ensemble + HMM and checkpoint them.")
    # Data sources / universe
    p.add_argument("--ohlcv-duckdb", type=Path, default=None,
                   help="Override bitemporal store path")
    p.add_argument("--core-duckdb", type=Path, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--ticker-limit", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)
    # Training / label knobs
    p.add_argument("--n-configs", type=int, default=None, help="number of seeds (ensembles)")
    p.add_argument("--tb-horizon", type=int, default=None, help="triple-barrier horizon (T+H)")
    p.add_argument("--tb-pt", type=float, default=None, help="profit-target σ multiple")
    p.add_argument("--tb-sl", type=float, default=None, help="stop-loss σ multiple")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-hmm", action="store_true", help="disable the Macro Risk HMM Oracle")
    p.add_argument("--use-macro-features", action="store_true",
                   help="P3/P4 A/B: add macro returns as GBM features (bumps the recipe hash)")
    p.add_argument("--out", type=Path, default=CHECKPOINT_PATH,
                   help="checkpoint output path (default: the standard checkpoint)")
    a = p.parse_args()

    cfg = RunConfig()
    if a.ohlcv_duckdb: cfg.bitemporal_duckdb = a.ohlcv_duckdb
    if a.core_duckdb: cfg.core_duckdb = a.core_duckdb
    if a.start_date: cfg.start_date = a.start_date
    if a.ticker_limit is not None: cfg.ticker_limit = a.ticker_limit
    if a.train_frac is not None: cfg.train_frac = a.train_frac
    if a.n_configs is not None: cfg.n_configs = a.n_configs
    if a.tb_horizon is not None: cfg.tb_horizon = a.tb_horizon
    if a.tb_pt is not None: cfg.tb_pt = a.tb_pt
    if a.tb_sl is not None: cfg.tb_sl = a.tb_sl
    if a.seed is not None: cfg.seed = a.seed
    if a.no_hmm: cfg.use_macro_hmm = False
    if a.use_macro_features: cfg.use_macro_features = True
    return cfg, a.out


if __name__ == "__main__":
    main(*_cli())
