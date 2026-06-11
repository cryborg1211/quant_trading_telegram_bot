"""Config-propagation tests for the tranche integration in run_backtest.py.

Pure-function tests on `_build_wf_config` — no engine run, no checkpoint, no
dataset materialization.
"""
from __future__ import annotations

from datetime import date

from run_backtest import _build_wf_config
from src.backtest.pipeline import RunConfig

CUTOFF = date(2022, 10, 13)
FEATURES = ["close_fd_xsz", "mom20_xsz", "market_regime"]


def _cfg(**overrides) -> RunConfig:
    cfg = RunConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestBuildWfConfig:
    def test_tranche_default_propagates(self) -> None:
        wf = _build_wf_config(FEATURES, CUTOFF, _cfg(signal_threshold=0.43))
        assert wf.rebalance_mode == "tranche"
        assert wf.tranche_hold_days == 30
        assert wf.signal_threshold == 0.43
        assert wf.start_trading_date == CUTOFF
        assert wf.feature_cols == FEATURES

    def test_explicit_hold_days(self) -> None:
        wf = _build_wf_config(FEATURES, CUTOFF, _cfg(), mode="tranche", hold_days=20)
        assert wf.tranche_hold_days == 20

    def test_grid_mode_is_legacy_config(self) -> None:
        cfg = _cfg(rebalance_frequency=5, signal_threshold=0.40)
        wf = _build_wf_config(FEATURES, CUTOFF, cfg, mode="grid", hold_days=30)
        assert wf.rebalance_mode == "grid"
        assert wf.rebalance_frequency == 5
        # Grid path must keep every legacy knob exactly as before.
        assert wf.seq_len == 1
        assert wf.cov_lookback == 60
        assert wf.liquid_top_n == cfg.liquid_top_n
        assert wf.constraints.max_weight == cfg.max_weight
        assert wf.constraints.target_leverage == 0.95
        assert wf.constraints.long_only is True

    def test_barrier_sigmas_propagate(self) -> None:
        wf = _build_wf_config(FEATURES, CUTOFF, _cfg(), mode="tranche",
                              hold_days=30, pt_sigma=3.0, sl_sigma=2.0)
        assert wf.tranche_pt_sigma == 3.0
        assert wf.tranche_sl_sigma == 2.0

    def test_barriers_default_off(self) -> None:
        wf = _build_wf_config(FEATURES, CUTOFF, _cfg())
        assert wf.tranche_pt_sigma is None
        assert wf.tranche_sl_sigma is None

    def test_price_unit_default_untouched(self) -> None:
        # The thousand-VND scaling must stay active regardless of mode.
        for mode in ("tranche", "grid"):
            wf = _build_wf_config(FEATURES, CUTOFF, _cfg(), mode=mode)
            assert wf.price_unit_vnd == 1000.0
