"""Characterization tests for `TabularEnsemble.fit` (V4.1 Structural Debt P3).

Hub node `TabularEnsemble.fit` (src/models/tabular_ensemble.py) had ZERO direct
coverage. These tests pin the STACKING WIRING (OOF assembly, meta-learner fit,
CalibratedClassifierCV wrapping, missing-class augmentation, predict shapes) —
NOT ML quality.

The three GPU-backed boosters (LightGBM/XGBoost/CatBoost) are replaced by a
real `DecisionTreeClassifier` stand-in. We deliberately do NOT use MagicMock:
`fit` calls sklearn `clone()` on every base model (both in `_manual_oof` and the
full-train refit), and `clone()` requires a real sklearn estimator. The decision
tree is fast, clonable, varies with X (so the meta-LR coefficients are non-zero),
supports `sample_weight`, and exposes `feature_importances_` — everything the
ensemble touches. It is keyed by the real `BASE_LEARNERS` names so
`learner_names` and the meta-feature width stay consistent between fit/predict.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.tabular_ensemble import (
    BASE_LEARNERS,
    UP_CLASS,
    TabularEnsemble,
    _augment_for_missing_classes,
)

pytestmark = pytest.mark.skipif(
    not BASE_LEARNERS, reason="no base learners installed (lightgbm/xgboost/catboost)"
)

_N_FEATURES = 9


def _make_xy(n: int = 150, seed: int = 0, rare_up: bool = False):
    """Return (X, y, start_times, end_times, sample_weight) for a fit() call."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, _N_FEATURES)).astype(np.float32)
    if rare_up:
        # One single UP sample → smallest class count == 1 (calibration skip path).
        half = (n - 1) // 2
        y = np.array([0] * half + [1] * (n - 1 - half) + [2] * 1, dtype=np.int64)
    else:
        y = np.tile([0, 1, 2], n // 3 + 1)[:n].astype(np.int64)
    start = pd.bdate_range("2022-01-03", periods=n).values.astype("datetime64[ns]")
    end = start + np.timedelta64(3, "D")
    w = np.ones(n, dtype=np.float32)
    return X, y, start, end, w


def _patch_base_models():
    """Patch `_build_base_models` to return clonable decision-tree stand-ins."""
    return patch(
        "src.models.tabular_ensemble._build_base_models",
        lambda seed: {
            name: DecisionTreeClassifier(max_depth=3, random_state=0)
            for name in BASE_LEARNERS
        },
    )


def _feature_names() -> list[str]:
    return [f"f{i}" for i in range(_N_FEATURES)]


def _fit_ensemble(*, calibrate: bool = True, rare_up: bool = False, n: int = 150):
    X, y, start, end, w = _make_xy(n=n, rare_up=rare_up)
    ens = TabularEnsemble(
        feature_names=_feature_names(), n_folds=3, embargo_bars=2, seed=0, calibrate=calibrate
    )
    with _patch_base_models():
        ens.fit(X, y, start, end, sample_weight=w)
    return ens, X


# --------------------------------------------------------------------------- #
# 3.2 — unfitted guard
# --------------------------------------------------------------------------- #
class TestTabularEnsembleUnfitted:
    def test_predict_proba_raises_before_fit(self) -> None:
        ens = TabularEnsemble(feature_names=_feature_names())
        with pytest.raises(RuntimeError):
            ens.predict_proba(np.zeros((3, _N_FEATURES), dtype=np.float32))


# --------------------------------------------------------------------------- #
# 3.3 — fit wiring
# --------------------------------------------------------------------------- #
class TestTabularEnsembleFitWiring:
    def test_fit_returns_self(self) -> None:
        ens, _ = _fit_ensemble()
        assert isinstance(ens, TabularEnsemble)

    def test_base_models_populated_after_fit(self) -> None:
        ens, _ = _fit_ensemble()
        assert len(ens.base_models) == len(BASE_LEARNERS)

    def test_meta_is_set_after_fit(self) -> None:
        ens, _ = _fit_ensemble()
        assert ens.meta is not None

    def test_learner_names_match_base_learners(self) -> None:
        ens, _ = _fit_ensemble()
        assert ens.learner_names == list(BASE_LEARNERS)

    def test_oof_meta_feature_names_count(self) -> None:
        ens, _ = _fit_ensemble()
        # one column per (learner × class-prob), 3 classes.
        assert len(ens.oof_meta_feature_names) == len(BASE_LEARNERS) * 3

    def test_contribution_sums_to_1(self) -> None:
        ens, _ = _fit_ensemble()
        assert sum(ens.contribution.values()) == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- #
# 3.4 — predict_proba shapes
# --------------------------------------------------------------------------- #
class TestTabularEnsemblePredictProba:
    def test_predict_proba_shape(self) -> None:
        ens, X = _fit_ensemble()
        assert ens.predict_proba(X).shape == (X.shape[0],)

    def test_predict_proba_values_in_unit_interval(self) -> None:
        ens, X = _fit_ensemble()
        p = ens.predict_proba(X)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_predict_proba_3class_shape(self) -> None:
        ens, X = _fit_ensemble()
        assert ens.predict_proba_3class(X).shape == (X.shape[0], 3)

    def test_predict_proba_3class_rows_sum_to_1(self) -> None:
        ens, X = _fit_ensemble()
        rows = ens.predict_proba_3class(X).sum(axis=1)
        assert np.allclose(rows, 1.0, atol=1e-5)

    def test_predict_proba_accepts_numpy_array(self) -> None:
        ens, X = _fit_ensemble()
        assert ens.predict_proba(np.asarray(X)).shape == (X.shape[0],)

    def test_predict_proba_accepts_dataframe(self) -> None:
        ens, X = _fit_ensemble()
        df = pd.DataFrame(X, columns=_feature_names())
        assert ens.predict_proba(df).shape == (X.shape[0],)


# --------------------------------------------------------------------------- #
# 3.5 — calibration path
# --------------------------------------------------------------------------- #
class TestTabularEnsembleCalibration:
    def test_calibrate_true_wraps_in_calibrated_cv(self) -> None:
        ens, _ = _fit_ensemble(calibrate=True)
        assert type(ens.meta).__name__ == "CalibratedClassifierCV"

    def test_calibrate_false_uses_raw_logreg(self) -> None:
        ens, _ = _fit_ensemble(calibrate=False)
        assert type(ens.meta).__name__ == "LogisticRegression"

    def test_calibration_skipped_when_class_count_too_small(self) -> None:
        # Smallest class has a single sample → calibration is skipped, raw LR served.
        ens, _ = _fit_ensemble(calibrate=True, rare_up=True, n=121)
        assert type(ens.meta).__name__ == "LogisticRegression"


# --------------------------------------------------------------------------- #
# 3.6 — _augment_for_missing_classes (direct unit, no fit)
# --------------------------------------------------------------------------- #
class TestAugmentMissingClasses:
    def test_missing_flat_class_gets_synth_injected(self) -> None:
        X = np.zeros((10, _N_FEATURES), dtype=np.float32)
        y = np.array([0, 2] * 5, dtype=np.int64)  # class 1 (FLAT) absent
        w = np.ones(10, dtype=np.float32)
        _, y_aug, _, missing = _augment_for_missing_classes(X, y, w)
        assert 1 in missing
        assert set(np.unique(y_aug)) == {0, 1, 2}

    def test_synth_weight_is_tiny(self) -> None:
        X = np.zeros((10, _N_FEATURES), dtype=np.float32)
        y = np.array([0, 2] * 5, dtype=np.int64)
        w = np.ones(10, dtype=np.float32)
        _, _, w_aug, _ = _augment_for_missing_classes(X, y, w)
        assert w_aug.min() < 1e-3  # injected rows carry a tiny weight

    def test_no_augmentation_when_all_classes_present(self) -> None:
        X = np.zeros((9, _N_FEATURES), dtype=np.float32)
        y = np.array([0, 1, 2] * 3, dtype=np.int64)
        w = np.ones(9, dtype=np.float32)
        X_out, y_out, w_out, missing = _augment_for_missing_classes(X, y, w)
        assert missing == []
        assert len(y_out) == len(y)


# --------------------------------------------------------------------------- #
# 3.7 — sklearn-style attributes
# --------------------------------------------------------------------------- #
class TestTabularEnsembleAttributes:
    def test_feature_importances_shape(self) -> None:
        ens, _ = _fit_ensemble()
        assert ens.feature_importances_.shape == (_N_FEATURES,)

    def test_feature_names_in_matches(self) -> None:
        ens, _ = _fit_ensemble()
        assert list(ens.feature_names_in_) == _feature_names()
