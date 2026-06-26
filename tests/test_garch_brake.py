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
def _reset_model_cache():
    """Clear the module-level model cache around each test."""
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
