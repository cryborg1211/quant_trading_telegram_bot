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
    """Clear the module-level model cache + isolate the drift/breadth legs.

    The drift (19-07-26) and breadth (20-07-26) legs load real OHLCV
    parquets when enabled — these GARCH-leg tests disable both so they stay
    hermetic; TestDriftBrake / TestBreadthBrake cover their own legs with
    the pure core + a stubbed loader.
    """
    monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", False, raising=False)
    monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", False, raising=False)
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


class TestBreadthBrake:
    """Breadth leg (20-07-26): the direct July-collapse signal the other two
    legs (price micro-regime, macro vol, index drift) never read."""

    def _fake_pipeline(self):
        return types.SimpleNamespace(RunConfig=lambda: None, load_ohlcv=lambda cfg: object())

    def test_all_legs_disabled_no_panel_load(self, monkeypatch):
        # drift/breadth default False (autouse fixture); GARCH also off here
        # → none of the three legs need the shared panel — must never even
        # attempt to import/load it.
        monkeypatch.setattr(CONFIG.trading, "garch_brake_enabled", False, raising=False)
        with patch("src.backtest.pipeline.load_ohlcv", side_effect=AssertionError("must not load")):
            assert garch_brake.live_exposure_scalar() == 1.0

    def test_breadth_alone_triggers_shared_panel_load(self, monkeypatch):
        # GARCH off, only breadth on — the shared-load gate must still fire
        # for breadth alone (not just when GARCH is enabled).
        monkeypatch.setattr(CONFIG.trading, "garch_brake_enabled", False, raising=False)
        monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", True, raising=False)
        with patch.dict(sys.modules, {"src.backtest.pipeline": self._fake_pipeline()}):
            with patch.object(garch_brake, "_breadth_leg_scalar", return_value=0.4) as mock_leg:
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.4)
                mock_leg.assert_called_once()

    def test_combined_takes_breadth_when_binding(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", True, raising=False)

        class _M:
            def p_bull_latest(self, obs):
                return 0.9  # healthy GARCH leg → must not bind

        with patch.dict(sys.modules, {"src.backtest.pipeline": self._fake_pipeline()}):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()), \
                 patch.object(garch_brake, "_breadth_leg_scalar", return_value=0.5):
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.5)

    def test_breadth_failure_fails_open_to_garch(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", True, raising=False)

        class _M:
            def p_bull_latest(self, obs):
                return 0.6

        with patch.dict(sys.modules, {"src.backtest.pipeline": self._fake_pipeline()}):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()), \
                 patch.object(garch_brake, "_breadth_leg_scalar",
                               side_effect=RuntimeError("boom")):
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.6)

    def test_raw_readings_distinguish_ran_from_failed_open(self, monkeypatch, _enabled):
        """A leg that failed open and a leg with nothing to brake BOTH read 1.0.

        Only the raw reading tells them apart, and two of three legs sat dead
        for hours precisely because nothing did. `drift_raw` / `breadth_raw` are
        None when a leg did not run and a number when it did.
        """
        monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", True, raising=False)
        monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", True, raising=False)
        # A rising market: the drift leg RUNS and correctly brakes nothing.
        fake_hmm = types.SimpleNamespace(
            build_market_proxy_returns=lambda panel: [0.004] * 12)

        class _M:
            def p_bull_latest(self, obs):
                return 0.9

        with patch.dict(sys.modules, {
            "src.backtest.pipeline": self._fake_pipeline(),
            "src.models.macro_risk_hmm": fake_hmm,
        }):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()), \
                 patch.object(garch_brake, "_breadth_leg_scalar", return_value=1.0), \
                 patch.object(garch_brake, "_breadth_raw", return_value=0.62):
                legs = garch_brake.live_exposure_legs()

        assert legs["drift"] == pytest.approx(1.0)
        # RAN: +0.4% x 10 sessions compounded, so ~+4%, comfortably above the
        # -3% trigger. Present == proof the leg executed.
        assert legs["drift_raw"] is not None
        assert legs["drift_raw"] > 0.0
        assert legs["breadth_raw"] == pytest.approx(0.62)

    def test_drift_raw_is_none_when_the_leg_fails(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", True, raising=False)
        fake_hmm = types.SimpleNamespace(
            build_market_proxy_returns=lambda panel: (_ for _ in ()).throw(
                RuntimeError("panel gone")))

        class _M:
            def p_bull_latest(self, obs):
                return 0.9

        with patch.dict(sys.modules, {
            "src.backtest.pipeline": self._fake_pipeline(),
            "src.models.macro_risk_hmm": fake_hmm,
        }):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()):
                legs = garch_brake.live_exposure_legs()

        # Failed open to 1.0 — same scalar a healthy leg would report...
        assert legs["drift"] == pytest.approx(1.0)
        # ...but the missing raw reading is what exposes it.
        assert legs["drift_raw"] is None

    def test_all_three_legs_stack_takes_the_min(self, monkeypatch, _enabled):
        monkeypatch.setattr(CONFIG.trading, "drift_brake_enabled", True, raising=False)
        monkeypatch.setattr(CONFIG.trading, "breadth_brake_enabled", True, raising=False)
        fake_hmm = types.SimpleNamespace(build_market_proxy_returns=lambda panel: None)

        class _M:
            def p_bull_latest(self, obs):
                return 0.9

        with patch.dict(sys.modules, {
            "src.backtest.pipeline": self._fake_pipeline(),
            "src.models.macro_risk_hmm": fake_hmm,
        }):
            with patch.object(garch_brake, "_load_model", return_value=_M()), \
                 patch.object(garch_brake, "_build_live_obs", return_value=_fake_obs()), \
                 patch.object(garch_brake, "_breadth_leg_scalar", return_value=0.3):
                # garch=0.9, drift=1.0 (market_ret None → no-op), breadth=0.3 → min=0.3.
                assert garch_brake.live_exposure_scalar() == pytest.approx(0.3)
