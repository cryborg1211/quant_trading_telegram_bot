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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config.settings import CONFIG

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
}
TOP_K_FEATURES = 70
FEATURE_SELECTION_MAX_ROWS = 150_000
N_SPLITS = 3
BASELINE_MACRO_F1 = 0.20
CLASSES = np.array([0, 1, 2], dtype=np.int64)


def artifact_paths(horizon: int) -> dict[str, Path]:
    artifact_dir = ARTIFACT_ROOT / f"{horizon}d"
    return {
        "dir": artifact_dir,
        "selected_features": artifact_dir / "selected_features.json",
        "scaler": artifact_dir / "scaler.joblib",
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


def compute_quantile_thresholds(train_df: pl.DataFrame, return_col: str, horizon: int) -> dict[str, Any]:
    q33 = float(train_df.select(pl.col(return_col).quantile(0.3333333333, interpolation="nearest")).item())
    q66 = float(train_df.select(pl.col(return_col).quantile(0.6666666667, interpolation="nearest")).item())
    if not np.isfinite(q33) or not np.isfinite(q66) or q33 >= q66:
        raise ValueError(f"Invalid {horizon}d quantile thresholds: q33={q33}, q66={q66}")
    limit = 0.08 if horizon == 5 else 0.25
    if q33 < -limit or q66 > limit:
        raise ValueError(f"Unrealistic {horizon}d thresholds: q33={q33}, q66={q66}")
    return {
        "horizon_days": horizon,
        "return_col": return_col,
        "q33_return": q33,
        "q66_return": q66,
        "q33_percent": q33 * 100.0,
        "q66_percent": q66 * 100.0,
    }


def apply_quantile_labels(df: pl.DataFrame, thresholds: dict[str, Any], target_col: str) -> pl.DataFrame:
    q33 = thresholds["q33_return"]
    q66 = thresholds["q66_return"]
    return df.with_columns(
        pl.when(pl.col(thresholds["return_col"]) <= q33)
        .then(0)
        .when(pl.col(thresholds["return_col"]) <= q66)
        .then(1)
        .otherwise(2)
        .cast(pl.Int64)
        .alias(target_col)
    )


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


def select_features(train_df: pl.DataFrame, feature_cols: list[str], target_col: str) -> list[str]:
    LOGGER.info("Feature selection input rows=%s candidate_features=%s target=%s", train_df.height, len(feature_cols), target_col)
    x = to_clean_numpy(train_df, feature_cols)
    y = train_df[target_col].to_numpy().astype(np.int64)

    if x.shape[0] > FEATURE_SELECTION_MAX_ROWS:
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(x.shape[0], size=FEATURE_SELECTION_MAX_ROWS, replace=False)
        x_selector = x[sample_idx]
        y_selector = y[sample_idx]
        LOGGER.info("Feature selection downsampled rows from %s to %s.", x.shape[0], FEATURE_SELECTION_MAX_ROWS)
    else:
        x_selector = x
        y_selector = y

    selector_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        tree_method="hist",
        device="cuda",
        objective="multi:softprob",
        num_class=3,
        n_jobs=-1,
        random_state=42,
    )
    with timed_step("GPU XGBoost feature selection"):
        selector_model.fit(x_selector, y_selector)
    importances = selector_model.feature_importances_
    return [feature_cols[i] for i in np.argsort(importances)[::-1][:TOP_K_FEATURES]]


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


def manual_oof(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
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


def safe_cross_val_predict(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
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
    return manual_oof(model, x, y, sample_weight)


def build_oof_meta_features(models: dict[str, Any], x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> tuple[np.ndarray, list[str]]:
    chunks = []
    names = []
    for name, model in models.items():
        with timed_step(f"Generating OOF probabilities: {name}"):
            chunks.append(safe_cross_val_predict(model, x, y, sample_weight))
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


def print_contribution_importance(meta_model: LogisticRegression, model_names: list[str]) -> dict[str, float]:
    coef = np.abs(meta_model.coef_)
    raw = {name: float(coef[:, i * 3 : (i + 1) * 3].sum()) for i, name in enumerate(model_names)}
    total = sum(raw.values()) or 1.0
    out = {k: v / total for k, v in raw.items()}
    LOGGER.info("Contribution Importance: %s", out)
    return out


def save_artifacts(
    horizon: int,
    selected_features: list[str],
    fitted_models: dict[str, Any],
    meta_model: LogisticRegression,
    scaler: StandardScaler,
    report: dict,
    cm: np.ndarray,
    thresholds: dict[str, Any],
) -> None:
    paths = artifact_paths(horizon)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models["xgboost"], paths["xgboost"])
    joblib.dump(fitted_models["lightgbm"], paths["lightgbm"])
    fitted_models["catboost"].save_model(str(paths["catboost"]))
    joblib.dump(meta_model, paths["meta_model"])
    joblib.dump(scaler, paths["scaler"])
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
    selected_features: list[str],
) -> None:
    return_col = RETURN_COLS[horizon]
    target_col = TARGET_COLS[horizon]
    LOGGER.info("========== Training %sd horizon ==========", horizon)

    thresholds = compute_quantile_thresholds(train_raw, return_col, horizon)
    train_df = apply_quantile_labels(train_raw, thresholds, target_col)
    test_df = apply_quantile_labels(test_raw, thresholds, target_col)

    LOGGER.info("Rows | train=%s test=%s", train_df.height, test_df.height)
    LOGGER.info("Label Dist | train=%s test=%s", dict(Counter(train_df[target_col].to_list())), dict(Counter(test_df[target_col].to_list())))

    x_train_raw = to_clean_numpy(train_df, selected_features)
    x_test_raw = to_clean_numpy(test_df, selected_features)
    y_train = train_df[target_col].to_numpy().astype(np.int64)
    y_test = test_df[target_col].to_numpy().astype(np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)
    base_models = build_base_models()
    oof_meta, meta_feature_names = build_oof_meta_features(base_models, x_train, y_train, sample_weight)

    meta_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    with timed_step(f"Training {horizon}d meta LogisticRegression"):
        meta_model.fit(oof_meta, y_train)

    fitted_models = fit_final_base_models(base_models, x_train, y_train, sample_weight)
    y_pred = meta_model.predict(make_meta_features(fitted_models, x_test))

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
    report["macro_f1"] = float(macro_f1)
    report["baseline_macro_f1"] = float(BASELINE_MACRO_F1)
    report["beats_baseline"] = bool(macro_f1 > BASELINE_MACRO_F1)
    report["selected_features"] = selected_features
    report["meta_features"] = meta_feature_names
    report["contribution_importance"] = contribution
    report["quantile_threshold_info"] = thresholds
    report["train_label_distribution"] = {str(k): int(v) for k, v in Counter(y_train.tolist()).items()}
    report["test_label_distribution"] = {str(k): int(v) for k, v in Counter(y_test.tolist()).items()}

    LOGGER.info("%sd Macro F1: %.6f", horizon, macro_f1)
    LOGGER.info("%sd Confusion Matrix\n%s", horizon, cm)
    save_artifacts(horizon, selected_features, fitted_models, meta_model, scaler, report, cm, thresholds)
    LOGGER.info("%sd artifacts saved: %s", horizon, artifact_paths(horizon)["dir"])


def main() -> None:
    total_start = time.perf_counter()
    LOGGER.info("Starting dual-horizon stacking ensemble training...")
    seed_everything()

    raw_df = load_raw_returns()
    train_raw, test_raw = chronological_split(raw_df)

    train_5d = apply_quantile_labels(
        train_raw,
        compute_quantile_thresholds(train_raw, RETURN_COLS[5], 5),
        TARGET_COLS[5],
    )
    feature_cols = infer_feature_columns(train_5d)
    LOGGER.info("Leakage guard active. Excluded columns: %s", sorted(LEAKAGE_COLS))
    selected_features = select_features(train_5d, feature_cols, TARGET_COLS[5])
    LOGGER.info("Selected %s shared features: %s", len(selected_features), selected_features)

    for horizon in HORIZONS:
        train_horizon(horizon, train_raw, test_raw, selected_features)

    LOGGER.info("Dual-horizon stacking training completed in %.2fs.", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()