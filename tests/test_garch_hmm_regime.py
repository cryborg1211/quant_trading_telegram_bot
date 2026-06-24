"""Tests for src/models/garch_hmm_regime.py — GARCH+HMM regime overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.garch_hmm_regime import (
    GarchHmmRegime,
    _reconstruct_garch_vol,
    train_garch_hmm,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_obs(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic 4-column macro DataFrame with two regime-like segments."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Bull regime (first half): positive drift, low vol
    n1 = n // 2
    bull_ret = rng.normal(0.001, 0.008, n1)
    # Bear regime (second half): negative drift, high vol
    n2 = n - n1
    bear_ret = rng.normal(-0.002, 0.020, n2)
    market_ret = np.concatenate([bull_ret, bear_ret])

    sp500_ret = market_ret * 0.6 + rng.normal(0, 0.005, n)
    dxy_ret = -market_ret * 0.3 + rng.normal(0, 0.003, n)
    usdvnd_ret = rng.normal(0.0001, 0.002, n)

    return pd.DataFrame(
        {
            "market_ret": market_ret,
            "sp500_ret": sp500_ret,
            "dxy_ret": dxy_ret,
            "usdvnd_ret": usdvnd_ret,
        },
        index=dates[:n],
    )


# ── GARCH vol reconstruction ────────────────────────────────────────────

class TestGarchVol:
    def test_length_matches_input(self) -> None:
        params = {"omega": 1e-6, "alpha": 0.05, "beta": 0.90, "mu": 0.0}
        ret = np.random.randn(100) * 0.01
        vol = _reconstruct_garch_vol(ret, params)
        assert len(vol) == 100

    def test_all_positive(self) -> None:
        params = {"omega": 1e-6, "alpha": 0.05, "beta": 0.90, "mu": 0.0}
        ret = np.random.randn(200) * 0.01
        vol = _reconstruct_garch_vol(ret, params)
        assert np.all(vol > 0)

    def test_high_persistence_smooth(self) -> None:
        """High β → vol changes slowly (low autocorrelation of Δσ)."""
        params = {"omega": 1e-7, "alpha": 0.02, "beta": 0.97, "mu": 0.0}
        ret = np.random.RandomState(1).randn(300) * 0.01
        vol = _reconstruct_garch_vol(ret, params)
        diffs = np.abs(np.diff(vol))
        assert np.mean(diffs) < np.std(vol)

    def test_unit_persistence_fallback(self) -> None:
        """α + β >= 1 → uses sample variance for init (no blow-up)."""
        params = {"omega": 1e-6, "alpha": 0.5, "beta": 0.6, "mu": 0.0}
        ret = np.random.randn(100) * 0.01
        vol = _reconstruct_garch_vol(ret, params)
        assert np.all(np.isfinite(vol))


# ── Training ─────────────────────────────────────────────────────────────

class TestTrainGarchHmm:
    def test_basic_fit(self) -> None:
        obs = _make_obs(400)
        model = train_garch_hmm(obs, n_states=3, n_restarts=4, seed=0)
        assert isinstance(model, GarchHmmRegime)
        assert model.n_states == 3
        assert 0 <= model.bull_state < 3
        assert model.train_end is not None

    def test_garch_params_stored(self) -> None:
        obs = _make_obs(400)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4)
        for key in ("omega", "alpha", "beta", "mu"):
            assert key in model.garch_params
            assert np.isfinite(model.garch_params[key])

    def test_persistence_finite(self) -> None:
        """α + β should be finite and ≤ 1 (IGARCH boundary is valid)."""
        obs = _make_obs(400)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4)
        p = model.garch_params
        persistence = p["alpha"] + p["beta"]
        assert np.isfinite(persistence)
        assert persistence <= 1.0 + 1e-6, f"GARCH persistence {persistence} > 1"

    def test_too_short_raises(self) -> None:
        obs = _make_obs(20)
        with pytest.raises(ValueError, match="too short"):
            train_garch_hmm(obs, n_states=3)

    def test_missing_columns_raises(self) -> None:
        obs = _make_obs(200).drop(columns=["dxy_ret"])
        with pytest.raises(ValueError, match="missing columns"):
            train_garch_hmm(obs, n_states=2)

    def test_four_states(self) -> None:
        obs = _make_obs(500)
        model = train_garch_hmm(obs, n_states=4, n_restarts=6, seed=7)
        assert model.n_states == 4
        assert 0 <= model.bull_state < 4

    def test_emission_zscore_stored(self) -> None:
        """Z-score normalization params frozen from train split."""
        obs = _make_obs(400)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4)
        assert model.emission_mean.shape == (5,)
        assert model.emission_std.shape == (5,)
        assert np.all(model.emission_std > 0)

    def test_persistence_capped(self) -> None:
        """α + β must respect the IGARCH guard cap."""
        obs = _make_obs(500)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4, max_persistence=0.96)
        p = model.garch_params
        persistence = p["alpha"] + p["beta"]
        assert persistence <= 0.96 + 1e-6, f"persistence {persistence} exceeds cap"

    def test_persistence_cap_preserves_unconditional_var(self) -> None:
        """Capping rescales α/β but keeps σ²_unc finite + positive."""
        obs = _make_obs(500)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4, max_persistence=0.90)
        p = model.garch_params
        persistence = p["alpha"] + p["beta"]
        assert persistence <= 0.90 + 1e-6
        sigma2_unc = p["omega"] / (1.0 - persistence)
        assert np.isfinite(sigma2_unc) and sigma2_unc > 0


# ── Linear exposure scaler ───────────────────────────────────────────────

class TestExposureScaler:
    @pytest.fixture()
    def fitted(self) -> tuple[GarchHmmRegime, pd.DataFrame]:
        obs = _make_obs(400, seed=99)
        model = train_garch_hmm(obs, n_states=3, n_restarts=6, seed=0)
        return model, obs

    def test_bounded_by_floor_and_cap(self, fitted: tuple) -> None:
        model, obs = fitted
        s = model.exposure_scaler(obs, min_exposure=0.2, max_exposure=1.0)
        assert s.min() >= 0.2 - 1e-9
        assert s.max() <= 1.0 + 1e-9
        assert s.name == "exposure"

    def test_floor_respected_even_when_pbull_low(self, fitted: tuple) -> None:
        model, obs = fitted
        s = model.exposure_scaler(obs, min_exposure=0.3, max_exposure=1.0)
        assert s.min() >= 0.3 - 1e-9

    def test_continuous_not_binary(self, fitted: tuple) -> None:
        """Linear scaler emits values strictly between floor and cap."""
        model, obs = fitted
        s = model.exposure_scaler(obs, min_exposure=0.2, max_exposure=1.0)
        interior = s[(s > 0.21) & (s < 0.99)]
        assert len(interior) > 0, "scaler is effectively binary, not continuous"


# ── Inference ────────────────────────────────────────────────────────────

class TestInference:
    @pytest.fixture()
    def fitted(self) -> tuple[GarchHmmRegime, pd.DataFrame]:
        obs = _make_obs(400, seed=99)
        model = train_garch_hmm(obs, n_states=3, n_restarts=6, seed=0)
        return model, obs

    def test_p_bull_series_shape(self, fitted: tuple) -> None:
        model, obs = fitted
        p = model.p_bull_series(obs)
        assert len(p) == len(obs.dropna())
        assert p.name == "p_bull"

    def test_p_bull_bounded(self, fitted: tuple) -> None:
        model, obs = fitted
        p = model.p_bull_series(obs)
        assert p.min() >= 0.0
        assert p.max() <= 1.0

    def test_p_bull_latest_scalar(self, fitted: tuple) -> None:
        model, obs = fitted
        val = model.p_bull_latest(obs)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_smoothed_differs_from_filtered(self, fitted: tuple) -> None:
        model, obs = fitted
        filt = model.p_bull_series(obs, filtered=True)
        smooth = model.p_bull_series(obs, filtered=False)
        assert not np.allclose(filt.values, smooth.values, atol=1e-6)

    def test_regime_labels_valid(self, fitted: tuple) -> None:
        model, obs = fitted
        labels = model.regime_labels(obs)
        assert set(labels.unique()).issubset(set(range(model.n_states)))

    def test_empty_obs_returns_default(self, fitted: tuple) -> None:
        model, _ = fitted
        empty = pd.DataFrame(columns=["market_ret", "sp500_ret", "dxy_ret", "usdvnd_ret"])
        assert model.p_bull_latest(empty) == 0.5
        assert model.p_bull_series(empty).empty

    def test_missing_column_raises(self, fitted: tuple) -> None:
        model, obs = fitted
        bad = obs.drop(columns=["sp500_ret"])
        with pytest.raises(ValueError, match="missing columns"):
            model.p_bull_series(bad)


# ── Exposure brake ───────────────────────────────────────────────────────

class TestExposureBrake:
    def test_binary_output(self) -> None:
        obs = _make_obs(400)
        model = train_garch_hmm(obs, n_states=3, n_restarts=4, seed=0)
        brake = model.exposure_brake(obs, threshold=0.5)
        assert set(brake.dropna().unique()).issubset({0.0, 1.0})
        assert brake.name == "exposure"

    def test_threshold_zero_all_on(self) -> None:
        obs = _make_obs(300)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4, seed=0)
        brake = model.exposure_brake(obs, threshold=0.0)
        assert (brake.dropna() == 1.0).all()

    def test_threshold_one_mostly_off(self) -> None:
        obs = _make_obs(300)
        model = train_garch_hmm(obs, n_states=2, n_restarts=4, seed=0)
        brake = model.exposure_brake(obs, threshold=1.0)
        # P(Bull) can equal exactly 1.0 in rare cases, so "mostly" off
        assert (brake.dropna() == 0.0).mean() >= 0.5
