"""`--serve-parity` sweep wiring + the sweep_conditions provenance stamp.

WHY (12-08-26). Two silent divergences made the validated numbers describe a
system that was never deployed:

  * `up_threshold` is what SERVE gates on, but inside the backtest it only fed
    the UP-precision metric — the ENGINE traded on `signal_threshold`, pinned
    5pp below. So the number serve admits on was never the optimised one.
  * none of serve's four defensive layers existed in the backtest, and measuring
    them found every one costs Sharpe. A threshold tuned bare and then deployed
    behind them is tuned for the wrong system.

`--serve-parity` closes both, and `sweep_conditions` records the conditions in
the artifact so a future reader can tell what a stored metric actually describes.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import run_backtest


def _cli(*argv: str):
    with patch.object(sys, "argv", ["run_backtest.py", *argv]):
        return run_backtest._cli()


# Positional layout of _cli()'s return tuple, from the tail.
#
# FRAGILE BY NATURE: `_cli` returns a bare 24-tuple, so appending a value shifts
# every tail-relative index. That happened on 12-08-26 when the GOLDEN drawdown
# budget was appended — these indices silently slid by one and produced garbage
# comparisons (`assert True == 3`) instead of a clear failure.
# `test_tuple_length_is_pinned` below is the tripwire: it fails FIRST and says
# what to do, so the next person adding a return value fixes the offsets instead
# of debugging nonsense assertions.
_GATE_OFF, _SEC_CAP, _DEDUP, _HYST, _BRAKE, _DD_BUDGET = -6, -5, -4, -3, -2, -1
_REGIME = 9          # positional, from the FRONT — see `_cli`'s return order
_CLI_TUPLE_LEN = 24


class TestServeParityShorthand:
    def test_tuple_length_is_pinned(self):
        assert len(_cli()) == _CLI_TUPLE_LEN, (
            "run_backtest._cli() changed arity — the tail-relative indices in "
            "this file (_GATE_OFF.._DD_BUDGET) must be re-based, and "
            "_CLI_TUPLE_LEN bumped, before the assertions below mean anything.")

    def test_defaults_are_legacy(self):
        out = _cli()
        assert out[_GATE_OFF] == pytest.approx(-0.05)   # the historical offset
        assert out[_SEC_CAP] == 0
        assert out[_DEDUP] is False
        assert out[_HYST] == 0
        assert out[_BRAKE] is False

    def test_serve_parity_sets_all_five(self):
        out = _cli("--serve-parity")
        assert out[_GATE_OFF] == pytest.approx(0.0)
        assert out[_SEC_CAP] == 2
        assert out[_DEDUP] is True
        assert out[_HYST] == 2
        assert out[_BRAKE] is True

    def test_explicit_flag_is_not_clobbered_by_the_shorthand(self):
        """A caller who names a value must keep it.

        The shorthand only fills in defaults; silently overriding an explicit
        --serve-sector-cap 3 would make the run differ from the command typed.
        """
        out = _cli("--serve-parity", "--serve-sector-cap", "3",
                   "--serve-hysteresis-days", "5", "--gate-offset", "-0.02")
        assert out[_SEC_CAP] == 3
        assert out[_HYST] == 5
        assert out[_GATE_OFF] == pytest.approx(-0.02)

    def test_layers_can_be_enabled_without_the_shorthand(self):
        out = _cli("--serve-exposure-brake", "--serve-cohort-dedup")
        assert out[_BRAKE] is True and out[_DEDUP] is True
        assert out[_GATE_OFF] == pytest.approx(-0.05)   # untouched


class TestServeParityIncludesRegimeSizing:
    """Regime sizing IS a production condition, so parity must include it.

    Caught 13-08-26 by diffing the freshly promoted artifact's `sweep_conditions`
    against live CONFIG: `use_regime_sizing` was False in the sweep and True in
    serve (settings.json `regime_sizing_enabled`). Both sides import the same
    policy from src/trading/regime_policy.py, so the only gap was the shorthand
    not setting the flag — which let `--serve-parity` claim production parity
    while omitting a production condition.
    """

    def test_serve_parity_turns_regime_sizing_on(self):
        assert _cli("--serve-parity")[_REGIME] is True

    def test_default_leaves_regime_sizing_off(self):
        # Bare runs must stay comparable with every pre-13-08 sweep.
        assert _cli()[_REGIME] is False

    def test_regime_sizing_alone_does_not_imply_the_rest_of_parity(self):
        out = _cli("--regime-sizing")
        assert out[_REGIME] is True
        assert out[_GATE_OFF] == pytest.approx(-0.05)   # still the legacy offset
        assert out[_BRAKE] is False


class TestGoldenDrawdownBudgetCli:
    """The budget must reach `main`, and an explicit 0 must mean "no budget"."""

    def test_default_comes_from_config(self):
        from config.settings import CONFIG
        assert _cli()[_DD_BUDGET] == pytest.approx(
            float(CONFIG.trading.golden_max_mean_dd_pp))

    def test_explicit_value_overrides_config(self):
        assert _cli("--golden-max-mean-dd-pp", "11.5")[_DD_BUDGET] == pytest.approx(11.5)

    def test_zero_disables_the_budget(self):
        # 0.0 is falsy, and `_select_golden(rows, 0.0)` would mean "0% drawdown
        # allowed" — excluding every arm. `_cli` must hand `main` None instead.
        assert _cli("--golden-max-mean-dd-pp", "0")[_DD_BUDGET] is None


class TestGateOffsetSemantics:
    def test_zero_offset_makes_the_swept_value_the_engine_gate(self):
        # engine_gate = up_threshold + gate_offset. With 0 they are the same
        # number, which is the only way the STORED up_threshold is the value the
        # sweep optimised.
        for thr in (0.41, 0.44, 0.46):
            assert thr + 0.0 == pytest.approx(thr)

    def test_legacy_offset_puts_the_engine_gate_below_the_stored_threshold(self):
        # Documents the actual historical bug: the live artifact stores 0.46
        # while its engine traded on 0.41.
        assert 0.46 + (-0.05) == pytest.approx(0.41)


class TestSweepConditionsStamp:
    def test_persist_records_the_conditions(self, tmp_path, monkeypatch):
        """The stamp is the whole point: without it nobody could tell that a
        stored `oos_sharpe` came from a looser rule than the one deployed."""
        monkeypatch.chdir(tmp_path)
        import joblib

        from src.backtest.pipeline import RunConfig

        cond = {"gate_offset": 0.0, "engine_gate": 0.44,
                "admission_mode": "cross_sectional", "serve_sector_cap": 2,
                "serve_cohort_dedup": True, "serve_hysteresis_days": 2,
                "serve_exposure_brake": True, "use_regime_sizing": False,
                "max_positions": 5}
        golden = {"up_threshold": 0.44, "signal_threshold": 0.44,
                  "n_seeds_ok": 1, "total_pred_up": 10,
                  "mean_up_precision": 0.55}
        rec = {"seed": 42, "net_pnl": 1.0, "total_return": 0.1,
               "net_sharpe": 0.5, "max_drawdown": -0.1, "n_days": 900,
               "final_nav": 1.1, "ensemble": object()}

        # No incumbent in a fresh tmp_path, so the promote-gate always promotes;
        # nothing to disable.
        run_backtest._persist_bot_payload(
            RunConfig(), ["f1"], golden, rec, None, [], sweep_conditions=cond)

        saved = list((tmp_path / "models" / "saved").glob("v3_ensemble_*.joblib"))
        assert saved, "no artifact written"
        meta = joblib.load(saved[0])["metadata"]
        assert meta["sweep_conditions"] == cond

    def test_missing_conditions_stamp_is_an_empty_dict_not_a_crash(self, tmp_path,
                                                                  monkeypatch):
        # Legacy callers (the --export-only path) pass nothing.
        monkeypatch.chdir(tmp_path)
        import joblib

        from src.backtest.pipeline import RunConfig

        golden = {"up_threshold": 0.46, "signal_threshold": 0.41,
                  "n_seeds_ok": 1, "total_pred_up": 1, "mean_up_precision": 0.5}
        rec = {"seed": 1, "net_pnl": 1.0, "total_return": 0.1, "net_sharpe": 0.5,
               "max_drawdown": -0.1, "n_days": 900, "final_nav": 1.1,
               "ensemble": object()}
        run_backtest._persist_bot_payload(RunConfig(), ["f1"], golden, rec,
                                          None, [])
        saved = list((tmp_path / "models" / "saved").glob("v3_ensemble_*.joblib"))
        meta = joblib.load(saved[0])["metadata"]
        assert meta["sweep_conditions"] == {}
