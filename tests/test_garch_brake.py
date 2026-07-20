"""Fail-open + clipping tests for src/bot/garch_brake.live_exposure_scalar.

The brake runs on the daily live cron. These tests pin the FAIL-OPEN contract:
any failure (disabled, missing model, missing data, exception) must return 1.0
(full exposure) so the live pipeline never breaks — and that a healthy path
clips P(Bull) into [floor, 1.0].
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import CONFIG
from src.bot import garch_brake


@pytest.fixture(autouse=True)
def _reset_model_cache(monkeypatch):
    """Clear the module-level model cache + isolate the drift leg per test.

    The drift leg (19-07-26) loads real OHLCV parquets when enabled — these
    GARCH-leg tests disable it so they stay hermetic; TestDriftBrake covers
    the drift leg with the pure core + a stubbed loader.
    """
    monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", False, raising=False)
    garch_brake._MODEL = None
    garch_brake._MODEL_TRIED = False
    yield
    garch_brake._MODEL = None
    garch_brake._MODEL_TRIED = False


@pytest.fixture
def _enabled(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "garch_brake_enabled", True, raising=False)
    monkeypatch.setattr(CONFIG.trading, "garch_brake_floor", 0.2, raising=False)


def _fake_obs() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=40)
    return pd.DataFrame(
        {c: [0.0] * 40 for c in ("market_ret", "sp500_ret", "dxy_ret", "usdvnd_ret")},
        index=idx,
    )


class TestFailOpen:
    def test_disabled_returns_one(self, monkeypatch):
        monkeypatch.setattr(CONFIG.trading, "garch_brake_enabled", False, raising=False)
        assert garch_brake.live_exposure_scalar() == 1.0

    def test_model_none_returns_one(self, _enabled):
        with patch.object(garch_brake, "_load_model", return_value=None):
            assert garch_brake.live_exposure_scalar() == 1.0

    def test_obs_none_returns_one(self, _enabled):
        with patch.object(garch_brake, "_load_model", return_value=object()), \
             patch.object(garch_brake, "_build_live_obs", return_value=None):
            assert garch_brake.live_exposure_scalar() == 1.0

    def test_exception_returns_one(self, _enabled):
        with patch.object(garch_brake, "_load_model", side_effect=RuntimeError("boom")):
            assert garch_brake.live_exposure_scalar() == 1.0


class TestClipping:
    def _model(self, p_bull: float):
        m = types.SimpleNamespace()
        m.p_bull_latest = lambda obs: p_bull
        return m

    def test_mid_pbull_passthrough(self, _enabled):
        with patch.object(garch_brake, "_load_model", return_value=self._model(0.6)), \
             patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
            assert garch_brake.live_exposure_scalar() == pytest.approx(0.6)

    def test_low_pbull_floored(self, _enabled):
        with patch.object(garch_brake, "_load_model", return_value=self._model(0.02)), \
             patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
            assert garch_brake.live_exposure_scalar() == pytest.approx(0.2)

    def test_high_pbull_capped(self, _enabled):
        with patch.object(garch_brake, "_load_model", return_value=self._model(1.5)), \
             patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
            assert garch_brake.live_exposure_scalar() == pytest.approx(1.0)

    def test_model_raises_is_fail_open(self, _enabled):
        bad = types.SimpleNamespace()
        def _raise(obs):
            raise ValueError("inference blew up")
        bad.p_bull_latest = _raise
        with patch.object(garch_brake, "_load_model", return_value=bad), \
             patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
            assert garch_brake.live_exposure_scalar() == 1.0


class TestDriftBrake:
    """Drift leg (19-07-26): slow-bleed brake the vol-triggered GARCH leg misses."""

    # -- pure core -----------------------------------------------------------

    def test_flat_market_full_exposure(self):
        assert garch_brake.drift_scalar_from_returns([0.0] * 10, 10, -0.03, -0.06, 0.5) == 1.0

    def test_short_history_fails_open(self):
        assert garch_brake.drift_scalar_from_returns([-0.02] * 5, 10, -0.03, -0.06, 0.5) == 1.0

    def test_deep_drawdown_hits_floor(self):
        # 10 sessions x -1% ~= -9.6% cum — beyond `full` (-6%) → floor.
        assert garch_brake.drift_scalar_from_returns([-0.01] * 10, 10, -0.03, -0.06, 0.5) == 0.5

    def test_july_grind_down_shape_trips_ramp(self):
        # ~-0.45%/session x 10 ~= -4.4% cum — the July-2026 slow-bleed shape the
        # GARCH brake ignored. Must land strictly inside the ramp (floor, 1.0).
        s = garch_brake.drift_scalar_from_returns([-0.0045] * 10, 10, -0.03, -0.06, 0.5)
        assert 0.5 < s < 1.0

    def test_ramp_is_linear_midpoint(self):
        # cum exactly halfway between trigger and full → scalar halfway to floor.
        import math
        r = math.exp(math.log(1.0 - 0.045) / 10.0) - 1.0  # 10 sessions → -4.5% cum
        s = garch_brake.drift_scalar_from_returns([r] * 10, 10, -0.03, -0.06, 0.5)
        assert s == pytest.approx(0.75, abs=1e-9)

    def test_degenerate_knobs_fail_open(self):
        assert garch_brake.drift_scalar_from_returns([-0.01] * 10, 10, -0.06, -0.03, 0.5) == 1.0
        assert garch_brake.drift_scalar_from_returns([-0.01] * 10, 0, -0.03, -0.06, 0.5) == 1.0
        assert garch_brake.drift_scalar_from_returns([-0.01] * 10, 10, -0.03, -0.06, 0.0) == 1.0

    # -- combined scalar -----------------------------------------------------

    def test_min_of_garch_and_drift(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", True, raising=False)
        fake_ret = pd.Series([-0.01] * 10)  # drift floor 0.5
        fake_pipeline = types.SimpleNamespace(
            RunConfig=lambda: None, load_ohlcv=lambda cfg: None)
        fake_hmm = types.SimpleNamespace(build_market_proxy_returns=lambda panel: fake_ret)

        class _M:
            def p_bull_latest(self, obs):
                return 0.9

        with patch.dict(sys.modules, {
            "src.backtest.pipeline": fake_pipeline,
            "src.models.macro_risk_hmm": fake_hmm,
        }):
            # GARCH leg healthy at 0.9 → combined must take the drift 0.5.
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.5)

    def test_drift_failure_fails_open_to_garch(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", True, raising=False)

        def _boom(cfg):
            raise RuntimeError("parquet gone")

        fake_pipeline = types.SimpleNamespace(RunConfig=lambda: None, load_ohlcv=_boom)
        fake_hmm = types.SimpleNamespace(build_market_proxy_returns=lambda panel: None)

        class _M:
            def p_bull_latest(self, obs):
                return 0.6

        with patch.dict(sys.modules, {
            "src.backtest.pipeline": fake_pipeline,
            "src.models.macro_risk_hmm": fake_hmm,
        }):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.6)
