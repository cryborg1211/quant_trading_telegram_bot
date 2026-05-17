import json
import logging
import random
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config.settings import CONFIG
from src.models.stacking_model.economic_metrics import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_PER_SIDE,
    economic_report,
    round_trip_cost,
    select_pnl_threshold,
)
from src.models.stacking_model.purged_kfold import PurgedKFold

warnings.filterwarnings("ignore")


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=False,
    )
    return logging.getLogger(__name__)


LOGGER = setup_logging()


@contextmanager
def timed_step(message: str):
    start = time.perf_counter()
    LOGGER.info("%s started...", message)
    try:
        yield
    finally:
        LOGGER.info("%s finished in %.2fs.", message, time.perf_counter() - start)


SEED = 42
DATA_PATH = Path("data/alpha360_features.parquet")
ARTIFACT_ROOT = Path("models/stacking")
DATE_COL = "date"
TICKER_COL = "ticker"
# Flaw 8 fix: increased from 3 → 5 for more stable OOF meta-feature estimates.
N_SPLITS = 5
HORIZONS = [5, 20]
RETURN_COLS = {5: "target_return_5d", 20: "target_return_20d"}
TARGET_COLS = {5: "target_class_5d", 20: "target_class_20d"}
LEAKAGE_COLS = {
    "raw_close",
    "close",
    "target_return_5d",
    "target_return_20d",
    "target_class_5d",
    "target_class_20d",
    # Triple-barrier additions (de Prado). target_bin_* are Int8 → numeric,
    # so they MUST be excluded or they leak the label. t1_* are Date dtype
    # (auto-excluded by infer_feature_columns' numeric filter) but listed
    # here defensively.
    "target_bin_5d",
    "target_bin_20d",
    "t1_5d",
    "t1_20d",
}
TOP_K_FEATURES = 70
FEATURE_SELECTION_MAX_ROWS = 150_000
# macro-F1 kept ONLY as a secondary diagnostic. The model-selection GATE is
# now economic (Task 2): a strategy must clear a positive cost-adjusted,
# risk-adjusted return or it is not deployable, no matter its F1.
BASELINE_MACRO_F1 = 0.20
BASELINE_NET_SHARPE = 0.0  # after VN fees + aggressive slippage
CLASSES = np.array([0, 1, 2], dtype=np.int64)


def artifact_paths(horizon: int) -> dict[str, Path]:
    artifact_dir = ARTIFACT_ROOT / f"{horizon}d"
    return {
        "dir": artifact_dir,
        "selected_features": artifact_dir / "selected_features.json",
        "xgboost": artifact_dir / "xgboost_model.joblib",
        "lightgbm": artifact_dir / "lightgbm_model.joblib",
        "catboost": artifact_dir / "catboost_model.cbm",
        "meta_model": artifact_dir / "meta_model.joblib",
        "report": artifact_dir / "classification_report.json",
        "confusion_matrix": artifact_dir / "confusion_matrix.json",
        "thresholds": artifact_dir / "quantile_thresholds.json",
    }


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_raw_returns() -> pl.DataFrame:
    LOGGER.info("Loading Alpha360 parquet: %s", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    schema = pl.scan_parquet(DATA_PATH).collect_schema()
    missing = [c for c in RETURN_COLS.values() if c not in schema]
    if missing:
        raise ValueError(f"Missing return columns {missing}. Regenerate Alpha360.")

    with timed_step("Collecting Alpha360 training dataset from Parquet; this might take a minute"):
        df = (
            pl.scan_parquet(DATA_PATH)
            .sort([TICKER_COL, DATE_COL])
            .drop_nulls([DATE_COL, TICKER_COL, *RETURN_COLS.values()])
            .collect()
            .sort([DATE_COL, TICKER_COL])
        )
    LOGGER.info("Loaded %s rows x %s cols from Parquet.", df.height, df.width)
    return df


def chronological_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split train/test on `CONFIG.training.split_date` (single source of truth)."""
    split_str = CONFIG.training.split_date  # "YYYY-MM-DD"
    split_dt = datetime.strptime(split_str, "%Y-%m-%d").date()
    split_lit = pl.date(split_dt.year, split_dt.month, split_dt.day)
    LOGGER.info("Chronological split date (from CONFIG.training.split_date): %s", split_str)

    train_df = df.filter(pl.col(DATE_COL) < split_lit).sort([DATE_COL, TICKER_COL])
    test_df = df.filter(pl.col(DATE_COL) >= split_lit).sort([DATE_COL, TICKER_COL])
    if train_df.height == 0 or test_df.height == 0:
        raise ValueError(f"Chronological split on {split_str} produced empty train/test set")
    return train_df, test_df


def infer_feature_columns(df: pl.DataFrame) -> list[str]:
    excluded = {DATE_COL, TICKER_COL, *LEAKAGE_COLS}
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    cols = [c for c, dtype in df.schema.items() if c not in excluded and dtype in numeric_types]
    leaked = sorted(set(cols) & LEAKAGE_COLS)
    if leaked:
        raise ValueError(f"Leakage columns detected in feature set: {leaked}")
    if len(cols) < TOP_K_FEATURES:
        raise ValueError(f"Need >= {TOP_K_FEATURES} numeric features, found {len(cols)}")
    return cols


def to_clean_numpy(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    x = df.select(columns).to_numpy().astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    r, c = np.where(np.isnan(x))
    x[r, c] = med[c]
    return x


def purged_feature_selection(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    t1_col: str,
    horizon: int,
) -> list[str]:
    """Select top-K features using purged cross-validation to avoid optimistic selection.

    Flaw 2 fix: instead of fitting a feature selector on the entire training set
    (which leaks future labels into the selection), we run feature selection
    independently on each purged fold's training split and average importances
    across folds. No fold's feature selector ever sees that fold's validation labels.

    Also fixes Flaw 2's second issue: this function is called per-horizon so 5d
    and 20d get independently selected feature sets.
    """
    x = to_clean_numpy(train_df, feature_cols)
    y = train_df[target_col].to_numpy().astype(np.int64)

    purged_cv = PurgedKFold(
        n_splits=N_SPLITS,
        start_times=train_df[DATE_COL].to_numpy(),
        end_times=train_df[t1_col].to_numpy(),
        embargo_bars=horizon,
    )

    importance_sum = np.zeros(len(feature_cols), dtype=np.float64)
    n_valid_folds = 0

    for fold_idx, (tr, _va) in enumerate(purged_cv.split(x), start=1):
        x_fold = x[tr]
        y_fold = y[tr]

        if x_fold.shape[0] > FEATURE_SELECTION_MAX_ROWS:
            rng = np.random.default_rng(SEED + fold_idx)
            sample_idx = rng.choice(x_fold.shape[0], size=FEATURE_SELECTION_MAX_ROWS, replace=False)
            x_fold = x_fold[sample_idx]
            y_fold = y_fold[sample_idx]

        selector = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            tree_method="hist",
            device="cuda",
            objective="multi:softprob",
            num_class=3,
            n_jobs=-1,
            random_state=SEED + fold_idx,
        )
        with timed_step(f"  Feature selection fold {fold_idx}/{N_SPLITS} (horizon={horizon}d)"):
            selector.fit(x_fold, y_fold)
        importance_sum += selector.feature_importances_
        n_valid_folds += 1

    if n_valid_folds == 0:
        raise RuntimeError("All feature selection folds were empty — check PurgedKFold config.")

    avg_importances = importance_sum / n_valid_folds
    selected = [feature_cols[i] for i in np.argsort(avg_importances)[::-1][:TOP_K_FEATURES]]
    LOGGER.info(
        "Purged feature selection (%sd): %s folds averaged → top %s features selected.",
        horizon, n_valid_folds, len(selected),
    )
    return selected


def build_base_models() -> dict[str, Any]:
    return {
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            n_estimators=180,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            objective="multiclass",
            num_class=3,
            device_type="gpu",
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            task_type="GPU",
            iterations=220,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def aligned_predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = getattr(model, "classes_", CLASSES)
    out = np.zeros((x.shape[0], 3), dtype=np.float32)
    for i, cls in enumerate(classes):
        cls_int = int(cls)
        if cls_int in (0, 1, 2):
            out[:, cls_int] = p[:, i]
    denom = out.sum(axis=1, keepdims=True)
    return out / np.where(denom == 0.0, 1.0, denom)


def fit_with_weight(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Any:
    model.fit(x, y, sample_weight=sample_weight)
    return model


def manual_oof(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cv: PurgedKFold,
) -> np.ndarray:
    """Out-of-fold probabilities under an injected, purged CV splitter."""
    oof = np.zeros((x.shape[0], 3), dtype=np.float32)
    covered = np.zeros(x.shape[0], dtype=bool)
    for fold_idx, (tr, va) in enumerate(cv.split(x), start=1):
        LOGGER.info("OOF fold %s/%s train=%s valid=%s started...", fold_idx, N_SPLITS, len(tr), len(va))
        m = clone(model)
        fit_with_weight(m, x[tr], y[tr], sample_weight[tr])
        oof[va] = aligned_predict_proba(m, x[va])
        covered[va] = True
    if not covered.all():
        m = clone(model)
        idx = np.where(covered)[0]
        fit_with_weight(m, x[idx], y[idx], sample_weight[idx])
        missing = np.where(~covered)[0]
        oof[missing] = aligned_predict_proba(m, x[missing])
    return oof


def safe_cross_val_predict(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cv: PurgedKFold,
) -> np.ndarray:
    """OOF probabilities via sklearn cross_val_predict with PurgedKFold."""
    try:
        p = cross_val_predict(
            model,
            x,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
            fit_params={"sample_weight": sample_weight},
        )
        p = np.asarray(p, dtype=np.float32)
        if p.shape == (x.shape[0], 3):
            return p
    except Exception as exc:
        LOGGER.warning("cross_val_predict failed (%s). Falling back to manual OOF.", exc)
    return manual_oof(model, x, y, sample_weight, cv)


def build_oof_meta_features(
    models: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    cv: PurgedKFold,
) -> tuple[np.ndarray, list[str]]:
    chunks = []
    names = []
    for name, model in models.items():
        with timed_step(f"Generating OOF probabilities: {name}"):
            chunks.append(safe_cross_val_predict(model, x, y, sample_weight, cv))
        names.extend([f"{name}_p0", f"{name}_p1", f"{name}_p2"])
    return np.hstack(chunks).astype(np.float32), names


def fit_final_base_models(models: dict[str, Any], x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    fitted = {}
    for name, model in models.items():
        with timed_step(f"Training final base model: {name}"):
            m = clone(model)
            fit_with_weight(m, x, y, sample_weight)
            fitted[name] = m
    return fitted


def make_meta_features(models: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return np.hstack([aligned_predict_proba(m, x) for m in models.values()]).astype(np.float32)


def print_contribution_importance(meta_model: LGBMClassifier, model_names: list[str]) -> dict[str, float]:
    """Log per-base-model contribution from the LightGBM meta-learner's feature importances."""
    importances = meta_model.feature_importances_  # shape: (9,) — 3 probs × 3 base models
    raw = {name: float(importances[i * 3 : (i + 1) * 3].sum()) for i, name in enumerate(model_names)}
    total = sum(raw.values()) or 1.0
    out = {k: v / total for k, v in raw.items()}
    LOGGER.info("Contribution Importance (meta LightGBM): %s", out)
    return out


def save_artifacts(
    horizon: int,
    selected_features: list[str],
    fitted_models: dict[str, Any],
    meta_model: LGBMClassifier,
    report: dict,
    cm: np.ndarray,
    thresholds: dict[str, Any],
) -> None:
    # Flaw 3 fix: StandardScaler removed — Alpha360 features are already rolling
    # Z-scores; double-scaling added imputation bias and was a near-identity op.
    paths = artifact_paths(horizon)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models["xgboost"], paths["xgboost"])
    joblib.dump(fitted_models["lightgbm"], paths["lightgbm"])
    fitted_models["catboost"].save_model(str(paths["catboost"]))
    joblib.dump(meta_model, paths["meta_model"])
    with paths["selected_features"].open("w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)
    with paths["report"].open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with paths["confusion_matrix"].open("w", encoding="utf-8") as f:
        json.dump({"labels": CLASSES.tolist(), "matrix": cm.tolist()}, f, indent=2)
    with paths["thresholds"].open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def train_horizon(
    horizon: int,
    train_raw: pl.DataFrame,
    test_raw: pl.DataFrame,
    feature_cols: list[str],
) -> None:
    return_col = RETURN_COLS[horizon]
    target_col = TARGET_COLS[horizon]
    t1_col = f"t1_{horizon}d"
    LOGGER.info("========== Training %sd horizon ==========", horizon)

    train_df = train_raw.drop_nulls([target_col, t1_col])
    test_df = test_raw.drop_nulls([target_col, t1_col])

    # Pure triple-barrier metadata. The legacy q33/q66 return terciles were
    # removed: labels are now decided by volatility-scaled barriers, NOT by
    # global return quantiles, so persisting tercile cut-points was misleading.
    thresholds = {
        "horizon_days": horizon,
        "return_col": return_col,
        "method": "triple_barrier",
        "pt_mult": 2.0,
        "sl_mult": 2.0,
        "vol_span": 20,
        "use_intrabar_extremes": True,
    }

    LOGGER.info("Rows | train=%s test=%s", train_df.height, test_df.height)
    LOGGER.info("Label Dist | train=%s test=%s", dict(Counter(train_df[target_col].to_list())), dict(Counter(test_df[target_col].to_list())))

    # Flaw 2 fix: feature selection runs per-horizon inside purged CV folds,
    # so the selector never sees the validation labels it will be evaluated on.
    with timed_step(f"Purged feature selection for {horizon}d horizon"):
        selected_features = purged_feature_selection(train_df, feature_cols, target_col, t1_col, horizon)
    LOGGER.info("Selected %s features for %sd: %s", len(selected_features), horizon, selected_features[:10])

    # Flaw 3 fix: StandardScaler removed. Alpha360 features are rolling Z-scores
    # (already ~N(0,1)); a second StandardScaler was a near-identity transform
    # that introduced imputation-bias into the scaler's mean/std estimates.
    x_train = to_clean_numpy(train_df, selected_features)
    x_test = to_clean_numpy(test_df, selected_features)
    y_train = train_df[target_col].to_numpy().astype(np.int64)
    y_test = test_df[target_col].to_numpy().astype(np.int64)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)
    base_models = build_base_models()

    # Embargo is ENFORCED: embargo_bars == horizon means the `horizon`
    # positional samples immediately after every test block are dropped from
    # training (de Prado Snippet 7.4), on top of the [start,t1] purge. With an
    # H-day label this guarantees no train sample's outcome window can touch
    # the validation window — zero autocorrelation leakage.
    assert horizon > 0, "embargo_bars (=horizon) must be > 0 to enforce embargo"
    purged_cv = PurgedKFold(
        n_splits=N_SPLITS,
        start_times=train_df[DATE_COL].to_numpy(),
        end_times=train_df[t1_col].to_numpy(),
        embargo_bars=horizon,
    )
    LOGGER.info(
        "PurgedKFold ACTIVE | n_splits=%s purge=[start,t1]-overlap embargo_bars=%s",
        N_SPLITS, horizon,
    )
    oof_meta, meta_feature_names = build_oof_meta_features(
        base_models, x_train, y_train, sample_weight, purged_cv
    )

    # Flaw 7 fix: replace LogisticRegression (linear, cannot capture nonlinear
    # base-model disagreements) with a shallow LightGBM meta-learner.
    # max_depth=3, num_leaves=4 keeps the meta-model simple enough to avoid
    # overfitting the 9 OOF probability columns while capturing interactions
    # like "when XGB and LGB disagree, prefer CatBoost".
    meta_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        max_depth=3,
        num_leaves=4,
        learning_rate=0.05,
        n_estimators=100,
        reg_alpha=0.1,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    with timed_step(f"Training {horizon}d meta LightGBM"):
        meta_model.fit(oof_meta, y_train)

    # ── COST-AWARE SELECTION (Task 2) ───────────────────────────────────
    # Pick the long-entry P(UP) threshold τ* that maximises NET SHARPE on
    # LEAK-FREE meta out-of-fold predictions (the meta-model re-run through
    # the SAME purged+embargoed CV). The selection objective is economic,
    # not statistical — argmax/F1 never enter the decision.
    train_ret = train_df[return_col].to_numpy().astype(np.float64)
    test_ret = test_df[return_col].to_numpy().astype(np.float64)
    with timed_step(f"Meta OOF (purged) for {horizon}d cost-aware threshold"):
        meta_oof_proba = safe_cross_val_predict(
            meta_model, oof_meta, y_train, sample_weight, purged_cv
        )
    p_up_oof = meta_oof_proba[:, 2]
    tau_star, oof_econ = select_pnl_threshold(p_up_oof, train_ret, horizon)
    LOGGER.info(
        "%sd cost-aware tau*=%.2f | OOF net_sharpe=%.3f net_pnl=%.4f trades=%s",
        horizon, tau_star, oof_econ["net_sharpe"], oof_econ["net_pnl"],
        oof_econ["n_trades"],
    )

    fitted_models = fit_final_base_models(base_models, x_train, y_train, sample_weight)
    test_meta = make_meta_features(fitted_models, x_test)
    p_up_test = meta_model.predict_proba(test_meta)[:, 2]
    y_pred = meta_model.predict(test_meta)  # argmax — DIAGNOSTIC ONLY

    # Economic truth: long iff P(UP) >= τ*, exit at t1, pay round-trip
    # fee + aggressive VN slippage. THIS drives beats_baseline.
    long_decisions = p_up_test >= tau_star
    test_econ = economic_report(long_decisions, test_ret, horizon)
    net_sharpe = float(test_econ["net_sharpe"])

    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    report = cast(
        dict[str, Any],
        classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            target_names=["DOWN", "SIDEWAYS", "UP"],
            output_dict=True,
            zero_division=0,
        ),
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    contribution = print_contribution_importance(meta_model, list(fitted_models.keys()))

    report["horizon_days"] = horizon
    # PRIMARY economic metrics (Task 2) — these gate deployment.
    report["net_sharpe"] = net_sharpe
    report["baseline_net_sharpe"] = float(BASELINE_NET_SHARPE)
    report["beats_baseline"] = bool(net_sharpe > BASELINE_NET_SHARPE)
    report["selection_metric"] = "net_sharpe"
    report["pnl_threshold_tau"] = float(tau_star)
    report["cost_model"] = {
        "fee_rate_per_side": float(DEFAULT_FEE_RATE),
        "slippage_per_side": float(DEFAULT_SLIPPAGE_PER_SIDE),
        "round_trip_cost": float(round_trip_cost()),
    }
    report["economics_test"] = test_econ
    report["economics_oof"] = oof_econ
    # macro-F1 retained as a SECONDARY diagnostic only.
    report["macro_f1"] = float(macro_f1)
    report["baseline_macro_f1"] = float(BASELINE_MACRO_F1)
    report["macro_f1_beats_baseline"] = bool(macro_f1 > BASELINE_MACRO_F1)
    report["selected_features"] = selected_features
    report["meta_features"] = meta_feature_names
    report["contribution_importance"] = contribution
    report["triple_barrier_info"] = thresholds
    report["train_label_distribution"] = {str(k): int(v) for k, v in Counter(y_train.tolist()).items()}
    report["test_label_distribution"] = {str(k): int(v) for k, v in Counter(y_test.tolist()).items()}

    LOGGER.info(
        "%sd ECONOMIC (gate) | net_sharpe=%.3f net_pnl=%.4f trades=%s hit=%.1f%% "
        "| macro_f1=%.4f (diagnostic)",
        horizon, net_sharpe, test_econ["net_pnl"], test_econ["n_trades"],
        100.0 * test_econ["hit_rate"], macro_f1,
    )
    LOGGER.info("%sd Confusion Matrix\n%s", horizon, cm)
    save_artifacts(horizon, selected_features, fitted_models, meta_model, report, cm, thresholds)
    LOGGER.info("%sd artifacts saved: %s", horizon, artifact_paths(horizon)["dir"])


def main() -> None:
    total_start = time.perf_counter()
    LOGGER.info("Starting dual-horizon stacking ensemble training...")
    seed_everything()

    raw_df = load_raw_returns()
    train_raw, test_raw = chronological_split(raw_df)

    # Infer candidate feature columns from the 5d-labeled training subset.
    # Feature SELECTION (top-K) now happens per-horizon inside train_horizon
    # via purged CV folds — not here on the full training set.
    train_5d = train_raw.drop_nulls([TARGET_COLS[5], "t1_5d"])
    feature_cols = infer_feature_columns(train_5d)
    LOGGER.info("Leakage guard active. Excluded columns: %s", sorted(LEAKAGE_COLS))
    LOGGER.info("Candidate feature pool: %s columns. Per-horizon selection runs inside train_horizon.", len(feature_cols))

    for horizon in HORIZONS:
        train_horizon(horizon, train_raw, test_raw, feature_cols)

    LOGGER.info("Dual-horizon stacking training completed in %.2fs.", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
