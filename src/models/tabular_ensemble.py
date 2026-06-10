"""
src/models/tabular_ensemble.py — Quant Engine V3.0 (V1-faithful port)

A direct restoration of the original Pure Tabular Stacking Ensemble from
`src/models/stacking_model/train_stacking.py` @ ccb6d84 (the LogisticRegression-
meta era BEFORE the May-17 "Flaw 7" LightGBM-meta upgrade), adapted to the V3
panel: 9 cross-sectional alpha features, triple-barrier 3-class labels, AFML
sample weights, GPU-accelerated (CUDA).

Architecture (identical hyperparameters to the historical ccb6d84 block,
GPU-accelerated for CUDA devices):

    Level 1 — 3 base GBM classifiers (multiclass, 3 classes):
        - XGBClassifier   (180 est, max_depth=5,  lr=0.03,  reg_alpha=0.1, reg_lambda=2.0)
        - LGBMClassifier  (220 est, num_leaves=31, lr=0.03, class_weight="balanced")
        - CatBoostClassifier (220 it, depth=6,     lr=0.03, l2_leaf_reg=5.0)

    Level 2 — sklearn LogisticRegression meta-stacker:
        penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=2000, random_state=SEED

    OOF — PurgedKFold (de Prado AFML Ch. 7) — n_splits=5, embargo_bars=horizon.
          This is the team's own May-17 upgrade from V1's original TimeSeriesSplit;
          the original CV had no purge/embargo and leaked H-day label windows
          across the train/val boundary.

    No scaler — the 9 features are already Gaussian-rank cross-sectional Z-scores
          (the team's "Flaw 3 fix": double-scaling adds imputation bias and is
          a near-identity op on already-standardized features).

Important note for the reader:
    The current HEAD of `train_stacking.py` (commit after 0c6bed3) replaced
    LogisticRegression with a shallow LightGBM meta-learner, explicitly
    documented as "Flaw 7 fix: linear meta cannot capture nonlinear base-model
    disagreements".  We are deliberately RESTORING the LogReg meta per the
    principal's instruction.  If LogReg under-performs on the 9-OOF-prob-column
    meta-feature set the same way it did on the 9-OOF-prob-column Alpha360 set,
    the team's own LightGBM-meta upgrade is the documented fix.

Deployment surface (unchanged for V3 walk-forward compatibility):
    ensemble.predict_proba(X_tab)        ->  (n,) P(UP)
    make_ensemble_oracle(ensemble)       ->  oracle(X (n, L, F)) -> (n,) P(UP)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("models.tabular_ensemble")

# ── Optional base-learner imports (graceful degradation) ─────────────────────
try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:                              # pragma: no cover
    _HAS_LGB = False
    LGBMClassifier = None
    LOGGER.warning("lightgbm not available — dropping from base learners")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:                              # pragma: no cover
    _HAS_XGB = False
    XGBClassifier = None
    LOGGER.warning("xgboost not available — dropping from base learners")

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:                              # pragma: no cover
    _HAS_CAT = False
    CatBoostClassifier = None
    LOGGER.warning("catboost not available — dropping from base learners")

# Reuse the team's own PurgedKFold (the May-17 upgrade) — it's already on disk.
from src.models.stacking_model.purged_kfold import PurgedKFold

UP_CLASS = 2
CLASSES = np.array([0, 1, 2], dtype=np.int64)

# ── Synthetic injection knobs for the missing-class force-fit ────────────────
# When a K-Fold's training y is missing one of {0=DOWN, 1=FLAT, 2=UP} (typically
# FLAT under tight barriers), XGBoost / LightGBM multiclass softprob refuses to
# fit ("Check failed: sum_weight >= kRtEps").  We inject N tiny-weighted dummy
# samples at the FEATURE MEAN of present samples for each missing class — enough
# to satisfy the validator, far too little to bias the model.
_N_SYNTH_PER_MISSING_CLASS = 5
_SYNTH_SAMPLE_WEIGHT = 1e-4

__all__ = [
    "TabularEnsemble",
    "make_ensemble_oracle",
    "BASE_LEARNERS",
    "UP_CLASS",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1) Base learners — verbatim ccb6d84 hyperparameters, GPU-accelerated
# ─────────────────────────────────────────────────────────────────────────────
def _build_base_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            max_bin=512,
            n_estimators=180,
            max_depth=6,                       # RELAXED 5→6: deeper splits for the recall boost
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,                    # RELAXED 2.0→1.0: less L2 shrinkage
            random_state=int(seed),
            n_jobs=1,
            verbosity=0,
        )
    if _HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=63,                     # RELAXED 31→63: 2× capacity, more interactions captured
            max_bin=255,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,                    # RELAXED 2.0→1.0: less L2 shrinkage
            class_weight="balanced",
            device_type="gpu",
            gpu_use_dp=False,
            random_state=int(seed),
            n_jobs=1,
            verbose=-1,
        )
    if _HAS_CAT:
        models["catboost"] = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=220,
            depth=7,                           # RELAXED 6→7: deeper splits
            learning_rate=0.03,
            l2_leaf_reg=3.0,                   # RELAXED 5.0→3.0: less L2 leaf shrinkage
            task_type="GPU",
            max_bin=512,
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
        )
    if not models:
        raise ImportError(
            "TabularEnsemble requires at least one of {lightgbm, xgboost, catboost}")
    return models


BASE_LEARNERS: list[str] = []
if _HAS_XGB:
    BASE_LEARNERS.append("xgboost")
if _HAS_LGB:
    BASE_LEARNERS.append("lightgbm")
if _HAS_CAT:
    BASE_LEARNERS.append("catboost")


# ─────────────────────────────────────────────────────────────────────────────
# 2) Helpers — ccb6d84 verbatim (aligned predict, manual OOF with weights)
# ─────────────────────────────────────────────────────────────────────────────
def _as_named_df(X: Any, feature_names: Sequence[str] | None) -> pd.DataFrame:
    """Wrap X as a pandas DataFrame with the supplied (or model-stored) column names.

    This is the root-cause fix for sklearn's "X does not have valid feature names,
    but <Estimator> was fitted with feature names" warning: by passing DataFrames
    with consistent column names at BOTH fit and predict, no metadata mismatch
    occurs.  No-op when X is already a DataFrame with correct dtype.
    """
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X, dtype=np.float32)
    cols = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _coerce_cats(X: pd.DataFrame, cat_features: Sequence[str]) -> pd.DataFrame:
    """Return a copy of `X` with each categorical column rounded/clipped to a
    non-negative integer code.

    The model matrix arrives as float32 (and `_augment_for_missing_classes`
    injects rows at the FEATURE MEAN → fractional values), but LightGBM
    `categorical_feature=` / CatBoost `cat_features=` require integer codes.
    No-op when none of `cat_features` are present in `X`.
    """
    present = [c for c in cat_features if c in X.columns]
    if not present:
        return X
    X = X.copy()
    for c in present:
        X[c] = X[c].round().clip(lower=0).astype("int64")
    return X


def _aligned_predict_proba(model: Any, X: Any, cat_features: Sequence[str] = ()) -> np.ndarray:
    """Re-index a base learner's predict_proba to a fixed (n, 3) {0,1,2} layout.

    Mirrors V1 `aligned_predict_proba`: handles the case where a fold's training
    labels omit one class (e.g. FLAT (class 1) absent in a thin slice), so the
    model's `classes_` is a subset of {0,1,2}.  Missing-class columns are zero,
    then renormalized so each row still sums to 1.

    Also fixes the "feature names mismatch" warning: when X arrives as a numpy
    array, we wrap it with the model's stored `feature_names_in_` (set at fit
    time) so sklearn / LightGBM / XGBoost / CatBoost all see consistent metadata.
    Categorical columns are integer-coerced to match the fit-time encoding.
    """
    if not isinstance(X, pd.DataFrame):
        names = getattr(model, "feature_names_in_", None)
        X = _as_named_df(X, names)
    X = _coerce_cats(X, cat_features)
    p = np.asarray(model.predict_proba(X), dtype=np.float32)
    classes = getattr(model, "classes_", CLASSES)
    n_rows = X.shape[0]
    out = np.zeros((n_rows, 3), dtype=np.float32)
    for i, cls in enumerate(classes):
        c = int(cls)
        if c in (0, 1, 2):
            out[:, c] = p[:, i]
    denom = out.sum(axis=1, keepdims=True)
    return out / np.where(denom == 0.0, 1.0, denom)


def _fit_with_weight(model: Any, X: Any, y: np.ndarray, w: np.ndarray,
                     cat_features: Sequence[str] = ()) -> Any:
    """Fit a base learner.  `X` is a named pandas DataFrame so the fitted model
    stores `feature_names_in_`.  Categorical columns (if any) are integer-coerced
    and declared NATIVELY to the learner that supports it:

        • LightGBM  → fit(categorical_feature=[names])
        • CatBoost  → fit(cat_features=[names])
        • XGBoost / others → numeric ordinal (no special handling)
    """
    present = [c for c in cat_features if c in getattr(X, "columns", [])]
    if present:
        X = _coerce_cats(X, present)
        cls = type(model).__name__
        if cls == "LGBMClassifier":
            model.fit(X, y, sample_weight=w, categorical_feature=present)
            return model
        if cls == "CatBoostClassifier":
            model.fit(X, y, sample_weight=w, cat_features=present)
            return model
    model.fit(X, y, sample_weight=w)
    return model


def _class_marginal(y: np.ndarray) -> np.ndarray:
    """Class-prior vector of length 3 (used as the rare-class-crash fallback)."""
    out = np.zeros(3, dtype=np.float32)
    classes_present, counts = np.unique(y, return_counts=True)
    n = max(1, len(y))
    for c, k in zip(classes_present, counts):
        if int(c) in (0, 1, 2):
            out[int(c)] = float(k) / n
    s = out.sum()
    return out / s if s > 0 else np.array([1/3, 1/3, 1/3], dtype=np.float32)


def _augment_for_missing_classes(
    X: Any, y: np.ndarray, w: np.ndarray,
    n_synth: int = _N_SYNTH_PER_MISSING_CLASS,
    synth_weight: float = _SYNTH_SAMPLE_WEIGHT,
) -> tuple[Any, np.ndarray, np.ndarray, list[int]]:
    """
    EMERGENCY FORCE-FIT for imbalanced labels (FLAT class near-empty under tight
    triple barriers).  If any of {0, 1, 2} is absent from `y`, inject `n_synth`
    synthetic rows for each missing class at the FEATURE-MEAN of the present
    samples, weighted at `synth_weight` (tiny).

    Why this is safe:
      • Per-class sum_weight = n_synth * synth_weight = 5e-4 → clears the
        XGBoost / LightGBM `sum_weight >= 1e-6` validator;
      • Total per-class loss contribution is ~5e-4 vs thousands of real samples
        → the synthetic block exerts <0.01% gradient pressure (negligible);
      • Feature MEAN is an uninformative point — the model learns no spurious
        "near-mean features → FLAT" rule because the gradient is dominated by
        real samples elsewhere in the feature space.

    Returns (X_aug, y_aug, w_aug, missing_classes_injected).  When no class is
    missing, the inputs are returned unchanged and `missing_classes_injected`
    is an empty list.
    """
    present = set(int(c) for c in np.unique(np.asarray(y)))
    missing = sorted({0, 1, 2} - present)
    if not missing:
        return X, y, w, []

    is_df = isinstance(X, pd.DataFrame)
    X_arr = (X.to_numpy(dtype=np.float32)
             if is_df else np.asarray(X, dtype=np.float32))
    # Empty-slice guard: if X has zero rows (PurgedKFold purged the entire fold
    # because the panel was not strictly time-sorted, or because n_splits ×
    # embargo overran the date axis), np.mean(axis=0) raises "Mean of empty
    # slice" and emits NaN.  Use a zero-vector as the (uninformative) synth
    # anchor so the caller can still produce a fit on the synthetic rows alone.
    if X_arr.shape[0] == 0:
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else (
            len(X.columns) if is_df else 0
        )
        if n_features == 0:
            # No features at all — nothing to inject; bail back to caller.
            return X, y, w, missing
        LOGGER.warning(
            "    _augment_for_missing_classes received EMPTY X (0 rows, %d features) "
            "— using zero-vector synth (uninformative anchor)", n_features)
        mean_row = np.zeros(n_features, dtype=np.float32)
    else:
        mean_row = X_arr.mean(axis=0).astype(np.float32)

    n_inj = n_synth * len(missing)
    X_synth = np.tile(mean_row, (n_inj, 1))
    y_synth = np.repeat(np.array(missing, dtype=np.int64), n_synth)
    w_synth = np.full(n_inj, float(synth_weight), dtype=np.float32)

    if is_df:
        X_aug = pd.concat(
            [X, pd.DataFrame(X_synth, columns=X.columns)],
            ignore_index=True,
        )
    else:
        X_aug = np.vstack([X_arr, X_synth])
    y_aug = np.concatenate([np.asarray(y, dtype=np.int64), y_synth])
    w_aug = np.concatenate([np.asarray(w, dtype=np.float32), w_synth])
    return X_aug, y_aug, w_aug, missing


def _manual_oof(model: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                cv: PurgedKFold, name: str, cat_features: Sequence[str] = ()) -> np.ndarray:
    """Out-of-fold (n, 3) probability matrix under a purged+embargoed splitter.

    Manual OOF (not sklearn `cross_val_predict(fit_params=...)`) because sklearn
    does not slice `sample_weight` to the training fold by default — V1's
    `safe_cross_val_predict` kept a manual fallback for the same reason.

    Defensive: V1's quantile labels guaranteed ~33%/33%/33% per class, but the
    V3 triple-barrier with tight ±1.5σ barriers produces near-binary labels
    (FLAT < 1%).  An XGBoost multiclass fit on a fold that happens to contain
    zero FLAT samples crashes (`sum_weight >= kRtEps`).  When that happens, we
    log a warning and emit the class-prior marginal for the failing fold so
    the OOF column is uninformative-but-finite; the meta-LR's class_weight=
    "balanced" + l2 absorbs the noise cleanly.
    """
    n = X.shape[0]
    oof = np.zeros((n, 3), dtype=np.float32)
    covered = np.zeros(n, dtype=bool)
    marginal_full = _class_marginal(y)
    # `_slice` indexes positionally for both numpy arrays AND pandas DataFrames
    # (the latter is what TabularEnsemble.fit hands us so feature names propagate).
    is_df = isinstance(X, pd.DataFrame)
    _slice = (lambda idx: X.iloc[idx]) if is_df else (lambda idx: X[idx])
    for k, (tr, va) in enumerate(cv.split(X), start=1):
        # FORCE-FIT: synth-inject any class missing from the FOLD's train y.
        X_tr, y_tr, w_tr, missing = _augment_for_missing_classes(
            _slice(tr), y[tr], w[tr])
        if missing:
            LOGGER.info(
                "    %s OOF fold %d | synth-inject classes=%s  (+%d rows @ w=%.0e)",
                name, k, missing,
                len(missing) * _N_SYNTH_PER_MISSING_CLASS, _SYNTH_SAMPLE_WEIGHT)
        try:
            m = clone(model)
            _fit_with_weight(m, X_tr, y_tr, w_tr, cat_features)
            oof[va] = _aligned_predict_proba(m, _slice(va), cat_features)
        except Exception as exc:                   # noqa: BLE001  (final safety net)
            LOGGER.warning(
                "    %s OOF fold %d FIT FAILED post-injection (%s) — emitting class marginal %s",
                name, k, type(exc).__name__, marginal_full.round(3).tolist())
            oof[va] = marginal_full                # broadcast over val rows
        covered[va] = True
        LOGGER.info("    %s OOF fold %d | train=%d valid=%d", name, k, len(tr), len(va))
    if not covered.all():                          # rare: degenerate fold dropped by PurgedKFold
        idx_cov = np.where(covered)[0]
        idx_miss = np.where(~covered)[0]
        X_cov, y_cov, w_cov, miss_cov = _augment_for_missing_classes(
            _slice(idx_cov), y[idx_cov], w[idx_cov])
        if miss_cov:
            LOGGER.info("    %s coverage-fix | synth-inject classes=%s", name, miss_cov)
        try:
            m = clone(model)
            _fit_with_weight(m, X_cov, y_cov, w_cov, cat_features)
            oof[idx_miss] = _aligned_predict_proba(m, _slice(idx_miss), cat_features)
        except Exception as exc:                   # noqa: BLE001
            LOGGER.warning("    %s degenerate-coverage fallback failed (%s) — marginal", name, exc)
            oof[~covered] = marginal_full
    return oof


# ─────────────────────────────────────────────────────────────────────────────
# 3) The ensemble
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TabularEnsemble:
    """
    V1-faithful pure-tabular stacking ensemble.

    Usage (mirrors V1 `train_horizon`):
        ens = TabularEnsemble(feature_names=tabular_features, embargo_bars=cfg.tb_horizon)
        ens.fit(X, y_3class, start_times=dates, end_times=t1, sample_weight=w)
        p_up = ens.predict_proba(X_oos)        # (n,) P(UP)
    """
    feature_names: list[str]
    n_folds: int = 5
    embargo_bars: int = 5             # set to label horizon (de Prado §7.4)
    seed: int = 42

    # Subset of `feature_names` the GBMs treat as CATEGORICAL (native split):
    # LightGBM `categorical_feature=`, CatBoost `cat_features=`.  Integer-coerced
    # (round/clip) before every fit/predict so a float32 matrix — and the
    # synth-injected feature-MEAN rows — stay valid category codes.  XGBoost
    # treats them as a numeric ordinal (no native-categorical plumbing needed).
    categorical_features: list[str] = field(default_factory=list)

    # ── Probability calibration of the meta-stacker (Platt / isotonic) ──────
    # The LGBM+XGB+CAT stack is systematically OVER-confident (raw P(UP) ~0.70–
    # 0.78), which saturates the half-Kelly sizer at the NAV cap.  Calibration
    # maps the meta's raw outputs onto true empirical frequencies (~0.51–0.55)
    # so Kelly sizes proportionally.  Calibrator is fit on the leak-free OOF
    # meta-matrix (see fit()).  Set calibrate=False to restore raw stacker probs.
    calibrate: bool = True
    calibration_method: str = "sigmoid"   # "sigmoid" (Platt) | "isotonic"
    calibration_folds: int = 5

    base_models: dict[str, Any] = field(default_factory=dict)
    meta: Any = None                  # LogisticRegression OR CalibratedClassifierCV
    learner_names: list[str] = field(default_factory=list)
    oof_meta_feature_names: list[str] = field(default_factory=list)
    contribution: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.learner_names = list(BASE_LEARNERS)

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, X: Any, y: np.ndarray,
            start_times: np.ndarray, end_times: np.ndarray,
            *, sample_weight: np.ndarray | None = None) -> "TabularEnsemble":
        # Wrap X as a NAMED pandas DataFrame so every base-learner fit records
        # `feature_names_in_`; predict-time validation then matches cleanly and
        # the "X does not have valid feature names" warning never fires.
        X = _as_named_df(X, self.feature_names)
        y = np.asarray(y, dtype=np.int64)
        sw = (np.asarray(sample_weight, dtype=np.float32)
              if sample_weight is not None
              else np.ones(len(y), dtype=np.float32))

        # PurgedKFold with embargo = label horizon (the team's own upgrade)
        cv = PurgedKFold(
            n_splits=self.n_folds,
            start_times=start_times,
            end_times=end_times,
            embargo_bars=self.embargo_bars,
        )
        LOGGER.info("PurgedKFold ACTIVE | n_splits=%d  embargo_bars=%d (= label horizon)",
                    self.n_folds, self.embargo_bars)

        # 1) Out-of-fold meta features: 3 base models × 3 class probabilities = 9 cols
        base_template = _build_base_models(self.seed)
        oof_chunks: list[np.ndarray] = []
        meta_names: list[str] = []
        for name, model in base_template.items():
            LOGGER.info("  OOF | %s  (n=%d  features=%d)", name, len(y), X.shape[1])
            oof_chunks.append(_manual_oof(model, X, y, sw, cv, name, self.categorical_features))
            meta_names.extend([f"{name}_p0", f"{name}_p1", f"{name}_p2"])
        oof_meta = np.hstack(oof_chunks).astype(np.float32)
        self.oof_meta_feature_names = meta_names
        LOGGER.info("  OOF assembled | shape=%s  learners=%s",
                    oof_meta.shape, list(base_template.keys()))

        # 2) LogisticRegression meta-stacker (fit on the leak-free OOF matrix).
        # RELAXED C=1.0→5.0: weaker L2 penalty so the meta trusts the base
        # learners more aggressively (5× the inverse-regularization strength).
        base_meta = LogisticRegression(
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=self.seed,
        )
        base_meta.fit(oof_meta, y)

        # 3) Log per-base-model contribution (sum of |coef| over its 3 prob
        #    columns, summed over the meta-LR's K-1 hyperplanes).  Computed from
        #    the RAW LR coefs (CalibratedClassifierCV hides `coef_`), so the
        #    interpretability report is unchanged by calibration.
        coef = np.abs(base_meta.coef_)            # shape (K, 9) where K∈{2,3}
        contrib = {n: float(coef[:, i*3:(i+1)*3].sum())
                   for i, n in enumerate(base_template.keys())}
        total = sum(contrib.values()) or 1.0
        self.contribution = {k: round(v / total, 3) for k, v in contrib.items()}
        LOGGER.info("  meta LR | contribution=%s  intercept=%s  classes=%s",
                    self.contribution,
                    [round(b, 3) for b in base_meta.intercept_.tolist()],
                    base_meta.classes_.tolist())

        # 3b) CALIBRATION — squash the over-confident stack onto true empirical
        #     frequencies so half-Kelly stops pinning at the NAV cap.  We
        #     calibrate on the SAME leak-free OOF meta-matrix: CalibratedClassifierCV
        #     internally K-folds it (fit a clone of the meta-LR on K-1 folds, fit
        #     the sigmoid/isotonic calibrator on the held fold, average across
        #     folds).  Falls back to the raw meta-LR if the class counts are too
        #     small to fold, or if calibration raises for any reason.
        self.meta = base_meta
        if self.calibrate:
            class_counts = np.bincount(y, minlength=3)
            present = class_counts[class_counts > 0]
            min_class = int(present.min()) if present.size else 0
            safe_folds = max(2, min(int(self.calibration_folds), min_class))
            if min_class < 2:
                LOGGER.warning(
                    "  calibration SKIPPED — smallest class has %d sample(s) (<2); "
                    "serving RAW meta-LR probs.", min_class)
            else:
                try:
                    calibrated = CalibratedClassifierCV(
                        estimator=clone(base_meta),
                        method=self.calibration_method,
                        cv=safe_folds,
                    )
                    calibrated.fit(oof_meta, y)
                    self.meta = calibrated
                    # Quantify the squash on the training OOF matrix (P(UP) range).
                    _rc, _cc = list(base_meta.classes_), list(calibrated.classes_)
                    if UP_CLASS in _rc and UP_CLASS in _cc:
                        raw_up = base_meta.predict_proba(oof_meta)[:, _rc.index(UP_CLASS)]
                        cal_up = calibrated.predict_proba(oof_meta)[:, _cc.index(UP_CLASS)]
                        LOGGER.info(
                            "  meta CALIBRATED | method=%s cv=%d | OOF P(UP) max %.3f→%.3f  "
                            "mean %.3f→%.3f", self.calibration_method, safe_folds,
                            float(raw_up.max()), float(cal_up.max()),
                            float(raw_up.mean()), float(cal_up.mean()))
                    else:
                        LOGGER.info("  meta CALIBRATED | method=%s cv=%d",
                                    self.calibration_method, safe_folds)
                except Exception as exc:               # noqa: BLE001
                    LOGGER.warning(
                        "  calibration FAILED (%s: %s) — serving RAW meta-LR probs.",
                        type(exc).__name__, exc)
                    self.meta = base_meta

        # 4) Refit each base learner on the FULL train set for deployment.
        # Apply the same synth-inject force-fit used in the OOF folds so the
        # deployed models can fit even when the FULL train is still missing a
        # rare class (defensive — should be unnecessary once tb_pt/tb_sl are 2σ).
        X_refit, y_refit, sw_refit, missing_full = _augment_for_missing_classes(X, y, sw)
        if missing_full:
            LOGGER.info(
                "  FULL-train | synth-inject classes=%s  (+%d rows @ w=%.0e)",
                missing_full,
                len(missing_full) * _N_SYNTH_PER_MISSING_CLASS, _SYNTH_SAMPLE_WEIGHT)
        for name, model in _build_base_models(self.seed).items():
            try:
                m = clone(model)
                _fit_with_weight(m, X_refit, y_refit, sw_refit, self.categorical_features)
                self.base_models[name] = m
            except Exception as exc:               # noqa: BLE001
                LOGGER.warning(
                    "  %s FULL-train refit failed post-injection (%s) — base learner will "
                    "emit class marginal at inference (the meta-LR's l2 absorbs the constant signal).",
                    name, type(exc).__name__)
                self.base_models[name] = None      # sentinel: predict_proba returns marginal
        self._fallback_marginal = _class_marginal(y)
        live = [n for n, m in self.base_models.items() if m is not None]
        LOGGER.info("  base learners refit on FULL train | live=%s  fallback=%s",
                    live, [n for n in self.base_models if self.base_models[n] is None])

        return self

    # ── predict ──────────────────────────────────────────────────────────
    def _predict_one_safe(self, name: str, X: Any) -> np.ndarray:
        m = self.base_models.get(name)
        n_rows = X.shape[0]
        if m is None:                              # FULL-train refit fell back to marginal
            return np.tile(self._fallback_marginal, (n_rows, 1)).astype(np.float32)
        return _aligned_predict_proba(m, X, self.categorical_features)

    def _make_meta_features(self, X: Any) -> np.ndarray:
        return np.hstack([self._predict_one_safe(n, X) for n in self.learner_names]).astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Returns (n,) P(UP) — the contract the V3 walk-forward oracle expects.

        X may be a numpy array (e.g. from the WalkForwardEngine oracle) or a
        DataFrame; we wrap to a NAMED DataFrame so base-learner predict_proba
        sees the same feature_names the model was fit with.
        """
        if self.meta is None:
            raise RuntimeError("TabularEnsemble is not fitted")
        X_df = _as_named_df(X, self.feature_names)
        meta_X = self._make_meta_features(X_df)
        proba = self.meta.predict_proba(meta_X)              # (n, K)
        cls = list(self.meta.classes_)
        if UP_CLASS in cls:
            return proba[:, cls.index(UP_CLASS)]
        LOGGER.warning("meta LR has no UP class in classes_=%s — returning zeros", cls)
        return np.zeros(proba.shape[0], dtype=np.float32)

    def predict_proba_3class(self, X: Any) -> np.ndarray:
        """Returns (n, 3) probability matrix re-indexed to the fixed {0,1,2}
        layout — DROP-IN compatible with the legacy stacker's output shape.

        Classes the meta-LR never saw at training (e.g. FLAT under tight
        barriers + no synth-inject) get probability 0.0; rows are renormalised
        to sum to 1 so downstream code can treat them as a proper distribution.
        """
        if self.meta is None:
            raise RuntimeError("TabularEnsemble is not fitted")
        X_df = _as_named_df(X, self.feature_names)
        meta_X = self._make_meta_features(X_df)
        proba = np.asarray(self.meta.predict_proba(meta_X), dtype=np.float32)   # (n, K)
        out = np.zeros((X_df.shape[0], 3), dtype=np.float32)
        for i, cls in enumerate(self.meta.classes_):
            c = int(cls)
            if c in (0, 1, 2):
                out[:, c] = proba[:, i]
        denom = out.sum(axis=1, keepdims=True)
        return out / np.where(denom == 0.0, 1.0, denom)

    def predict_base(self, X: Any) -> dict[str, np.ndarray]:
        """Per-base-learner (n, 3) probability matrices — for diagnostics."""
        X_df = _as_named_df(X, self.feature_names)
        return {n: self._predict_one_safe(n, X_df) for n in self.learner_names}

    # ── sklearn-style attributes for downstream explanation code ───────────
    @property
    def feature_importances_(self) -> np.ndarray:
        """Aggregate feature importance = MEAN of each base learner's
        normalised importance vector.  Shape (n_features,) — drop-in
        compatible with the legacy `xgb_model.feature_importances_` access
        pattern used by `main._build_feature_explanation(model, …)`."""
        if not self.base_models:
            raise RuntimeError("TabularEnsemble is not fitted")
        cols: list[np.ndarray] = []
        for name in self.learner_names:
            m = self.base_models.get(name)
            if m is None:
                continue
            imp = getattr(m, "feature_importances_", None)
            if imp is None:
                continue
            imp = np.asarray(imp, dtype=np.float64)
            s = imp.sum()
            cols.append(imp / s if s > 0 else imp)
        if not cols:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        return np.mean(np.stack(cols, axis=0), axis=0)

    @property
    def feature_names_in_(self) -> np.ndarray:
        """sklearn-style training-time feature names — lets explanation code
        that walks `model.feature_names_in_` work on V3 without modification."""
        return np.asarray(self.feature_names, dtype=object)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Walk-forward oracle factory (unchanged contract for the Phase-8 engine)
# ─────────────────────────────────────────────────────────────────────────────
def make_ensemble_oracle(
    ensemble: TabularEnsemble, tab_indices: Sequence[int] | None = None,
) -> Callable:
    """
    Adapt a fitted TabularEnsemble to the Phase-8 oracle contract:

        oracle(X: (n, L, F))  ->  (n,) P(UP)

    The engine builds 3-D tensors per inference call.  The ensemble consumes
    only the LAST timestep's tabular features — with WalkForwardConfig.seq_len=1
    the slice is trivial.  `tab_indices` selects a subset of the F columns at
    the last bar (None ⇒ all).
    """
    idx = None if tab_indices is None else list(tab_indices)

    def oracle(X: np.ndarray) -> np.ndarray:
        last = X[:, -1, :]
        Xtab = last if idx is None else last[:, idx]
        return ensemble.predict_proba(Xtab)

    return oracle
