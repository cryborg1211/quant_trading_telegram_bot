"""Promote-gate for the weekly auto-retrain (20-07-26).

retrain moved from manual/occasional to EVERY Saturday (quant_weekly_retrain
schtask) — run_backtest._persist_bot_payload must compare the new candidate
against the incumbent artifact's embedded metrics before overwriting it, and
must never touch the real repo's models/saved/ (mirrors the paperlog
test-pollution lesson from earlier this session — everything here runs
inside a chdir'd tmp_path).
"""
from __future__ import annotations

import joblib
import pytest

from run_backtest import _persist_bot_payload, _promote_decision
from src.backtest.pipeline import RunConfig


# ---------------------------------------------------------------------------
# _promote_decision — pure
# ---------------------------------------------------------------------------


def test_no_incumbent_always_promotes():
    promote, reason = _promote_decision({"oos_sharpe": 0.1}, None, -0.10, 3.0, 0.35)
    assert promote
    assert "no incumbent" in reason


def test_sharpe_regression_rejected():
    new = {"oos_sharpe": 0.30, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.40}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.40}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "Sharpe regression" in reason


def test_sharpe_within_tolerance_promotes():
    # -0.05 delta is within the -0.10 allowed tolerance.
    new = {"oos_sharpe": 0.55, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.40}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.40}
    promote, _ = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert promote


def test_dd_is_not_compared_against_the_incumbent():
    """MaxDD is an extremum, so it is NOT comparable across OOS windows.

    THE DEADLOCK THIS FIXES (10-08-26). Every walk-forward runs from a fixed
    start to the newest bar, so a candidate's OOS window strictly CONTAINS the
    incumbent's — same start, ~21 more days. A max over the longer window can
    only rise once any new drawdown lands, so a relative DD check rejects good
    candidates by construction. It rejected four consecutive retrains and froze
    the live T+20 artifact at 2026-07-17.

    This case is the real 08-08 T+20 candidate: Sharpe essentially unchanged
    (0.645 -> 0.646) while DD jumped 12.60% -> 19.26%. That signature means a
    longer window holding a bigger drawdown, not a worse model.
    """
    new = {"oos_sharpe": 0.646, "oos_max_dd": -0.1926,
           "golden_mean_up_precision": 0.548,
           "trained_at": "2026-08-08T03:03:33Z"}
    old = {"oos_sharpe": 0.645, "oos_max_dd": -0.1260,
           "golden_mean_up_precision": 0.5698,
           "trained_at": "2026-07-17T08:47:10Z"}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert promote, reason
    assert "NOT comparable" in reason


def test_dd_still_blocked_by_the_absolute_ceiling():
    # Dropping the relative check must not remove the DD guard entirely — a
    # runaway drawdown is still a reject, judged on its own merits.
    new = {"oos_sharpe": 0.80, "oos_max_dd": -0.40, "golden_mean_up_precision": 0.50}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.13, "golden_mean_up_precision": 0.45}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "absolute MaxDD ceiling" in reason


def test_genuine_sharpe_regression_still_rejected_after_the_fix():
    # The real 08-08 T+5 candidate. Sharpe fell 0.590 -> 0.489 on a near-
    # identical window: a real degradation, and the one rejection of the four
    # that SHOULD stand. Guards against the DD fix loosening the gate wholesale.
    new = {"oos_sharpe": 0.489, "oos_max_dd": -0.1943,
           "golden_mean_up_precision": 0.550,
           "trained_at": "2026-08-08T04:06:24Z"}
    old = {"oos_sharpe": 0.590, "oos_max_dd": -0.1344,
           "golden_mean_up_precision": 0.4585,
           "trained_at": "2026-07-19T16:59:19Z"}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "Sharpe regression" in reason


def test_absolute_sharpe_floor_blocks_a_broken_run():
    # Needs an incumbent: with none, the gate promotes unconditionally rather
    # than leave the bot with no artifact at all.
    new = {"oos_sharpe": 0.05, "oos_max_dd": -0.05, "golden_mean_up_precision": 0.50}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.13, "golden_mean_up_precision": 0.45}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "absolute Sharpe floor" in reason


def test_no_incumbent_promotes_even_a_weak_candidate():
    # Explicit: the absolute floors must NOT wedge the very first deploy.
    new = {"oos_sharpe": 0.05, "oos_max_dd": -0.40, "golden_mean_up_precision": 0.10}
    promote, reason = _promote_decision(new, None, -0.10, 3.0, 0.35)
    assert promote
    assert "no incumbent" in reason


def test_precision_floor_rejected_even_if_better_than_incumbent():
    # Incumbent itself was already below floor — the floor is absolute, not relative.
    new = {"oos_sharpe": 0.60, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.30}
    old = {"oos_sharpe": 0.50, "oos_max_dd": -0.10, "golden_mean_up_precision": 0.20}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "precision" in reason.lower()


def test_all_gates_pass_promotes():
    new = {"oos_sharpe": 0.65, "oos_max_dd": -0.12, "golden_mean_up_precision": 0.46}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.13, "golden_mean_up_precision": 0.45}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert promote
    assert "passed" in reason


def test_malformed_incumbent_does_not_let_a_weak_candidate_through():
    # Malformed incumbent metadata (missing oos_sharpe) must not crash, and must
    # not auto-pass: old_sharpe defaults to -inf, which makes the RELATIVE check
    # vacuous, so the absolute floor is the backstop that has to catch this.
    new = {"oos_sharpe": 0.10, "oos_max_dd": 0.0, "golden_mean_up_precision": 0.40}
    promote, reason = _promote_decision(new, {}, -0.10, 3.0, 0.35)
    assert not promote
    assert "absolute Sharpe floor" in reason


def test_malformed_incumbent_still_promotes_a_healthy_candidate():
    new = {"oos_sharpe": 0.62, "oos_max_dd": -0.14, "golden_mean_up_precision": 0.46}
    promote, reason = _promote_decision(new, {}, -0.10, 3.0, 0.35)
    assert promote, reason


# ---------------------------------------------------------------------------
# Sharpe needs a SHARED SWEEP BASIS to be comparable (12-08-26)
#
# The MaxDD fix above cured one apples-to-oranges comparison and left another.
# Sharpe compares across two OOS *windows* but not across two *strategies*, and
# `--serve-parity` changed the strategy the sweep measures (gate_offset 0 instead
# of -0.05, plus the four defensive layers, together worth ~0.2 Sharpe — double
# the -0.10 delta the gate allows).
# ---------------------------------------------------------------------------

_PARITY_BASIS = {
    "gate_offset": 0.0, "engine_gate": 0.43, "admission_mode": "cross_sectional",
    "serve_sector_cap": 2, "serve_cohort_dedup": True,
    "serve_hysteresis_days": 2, "serve_exposure_brake": True,
    "use_regime_sizing": False, "max_positions": 5,
}


def test_saturdays_real_candidate_would_have_been_rejected_before_the_fix():
    """Replay of the live numbers, and the reason the fix was needed.

    Incumbent T+20 on disk: oos_sharpe 0.645 (best seed), earned BARE. The
    12-08 parity smoke test's best seed was ~0.53. 0.645 - 0.10 = 0.545, so the
    candidate misses by 0.015 — a reject caused entirely by the measurement basis
    changing, not by the model getting worse.
    """
    new = {"oos_sharpe": 0.53, "oos_max_dd": -0.1516,
           "golden_mean_up_precision": 0.4545,
           "trained_at": "2026-08-15T09:00:00Z",
           "sweep_conditions": dict(_PARITY_BASIS)}
    old = {"oos_sharpe": 0.645, "oos_max_dd": -0.1260,
           "golden_mean_up_precision": 0.5698,
           "trained_at": "2026-07-17T08:47:10Z"}          # no stamp — swept bare

    # What the pre-fix gate did: a bare relative comparison.
    assert new["oos_sharpe"] < old["oos_sharpe"] - 0.10

    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert promote, reason
    assert "RELATIVE SHARPE CHECK SKIPPED" in reason
    assert "incumbent has no sweep_conditions stamp" in reason


def test_both_unstamped_are_the_same_basis_so_regression_still_rejects():
    """Two pre-stamp artifacts were both swept bare — comparable to each other.

    This is what keeps the genuine 08-08 T+5 regression a reject. If unstamped
    were treated as "unknown", the fix would have silently disabled the relative
    check for every legacy pair.
    """
    new = {"oos_sharpe": 0.489, "oos_max_dd": -0.1943,
           "golden_mean_up_precision": 0.550}
    old = {"oos_sharpe": 0.590, "oos_max_dd": -0.1344,
           "golden_mean_up_precision": 0.4585}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "Sharpe regression" in reason
    assert "same sweep basis" in reason


def test_matching_stamps_restore_the_relative_check():
    """The widening is ONE-TIME: parity-vs-parity compares normally again."""
    new = {"oos_sharpe": 0.30, "oos_max_dd": -0.12,
           "golden_mean_up_precision": 0.45,
           "sweep_conditions": dict(_PARITY_BASIS)}
    old = {"oos_sharpe": 0.53, "oos_max_dd": -0.15,
           "golden_mean_up_precision": 0.45,
           "sweep_conditions": dict(_PARITY_BASIS)}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "Sharpe regression" in reason


def test_engine_gate_differing_does_not_break_comparability():
    """`engine_gate` is the OPTIMISED OUTPUT, not a measurement condition.

    Two parity runs that picked different thresholds must stay comparable —
    otherwise every run is incomparable to every other and the relative check is
    permanently off.
    """
    old_sc = dict(_PARITY_BASIS, engine_gate=0.43)
    new_sc = dict(_PARITY_BASIS, engine_gate=0.41)
    new = {"oos_sharpe": 0.30, "oos_max_dd": -0.12,
           "golden_mean_up_precision": 0.45, "sweep_conditions": new_sc}
    old = {"oos_sharpe": 0.53, "oos_max_dd": -0.15,
           "golden_mean_up_precision": 0.45, "sweep_conditions": old_sc}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote, "engine_gate must not be treated as a basis difference"
    assert "Sharpe regression" in reason


def test_admission_mode_differing_does_not_break_comparability():
    """`admission_mode` was measured VACUOUS — byte-identical at every gate level.

    absolute_gate and cross_sectional cannot change a single trade, because
    admission_pool_cap (6) >= max_positions (5) so the pool cap never binds.
    Treating it as material would wedge the gate on a non-difference.
    """
    old_sc = dict(_PARITY_BASIS, admission_mode="cross_sectional")
    new_sc = dict(_PARITY_BASIS, admission_mode="absolute_gate")
    new = {"oos_sharpe": 0.30, "oos_max_dd": -0.12,
           "golden_mean_up_precision": 0.45, "sweep_conditions": new_sc}
    old = {"oos_sharpe": 0.53, "oos_max_dd": -0.15,
           "golden_mean_up_precision": 0.45, "sweep_conditions": old_sc}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "Sharpe regression" in reason


def test_a_real_layer_difference_does_skip_the_check():
    old_sc = dict(_PARITY_BASIS, serve_exposure_brake=False, gate_offset=-0.05)
    new = {"oos_sharpe": 0.30, "oos_max_dd": -0.12,
           "golden_mean_up_precision": 0.45,
           "sweep_conditions": dict(_PARITY_BASIS)}
    old = {"oos_sharpe": 0.60, "oos_max_dd": -0.15,
           "golden_mean_up_precision": 0.45, "sweep_conditions": old_sc}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert promote, reason
    assert "basis differs" in reason
    assert "gate_offset" in reason and "serve_exposure_brake" in reason


def test_skipping_the_relative_check_does_not_disable_the_absolute_floors():
    """The one-time widening must not become an open door."""
    base = {"golden_mean_up_precision": 0.45,
            "sweep_conditions": dict(_PARITY_BASIS)}
    old = {"oos_sharpe": 0.645, "oos_max_dd": -0.1260,
           "golden_mean_up_precision": 0.5698}                 # unstamped
    for meta, frag in (
        ({**base, "oos_sharpe": 0.05, "oos_max_dd": -0.12}, "absolute Sharpe floor"),
        ({**base, "oos_sharpe": 0.80, "oos_max_dd": -0.40}, "absolute MaxDD ceiling"),
        ({**base, "oos_sharpe": 0.80, "oos_max_dd": -0.12,
          "golden_mean_up_precision": 0.20}, "precision"),
    ):
        promote, reason = _promote_decision(meta, old, -0.10, 3.0, 0.35)
        assert not promote, reason
        assert frag in reason


def test_an_empty_stamp_reads_as_unstamped():
    """`_persist_bot_payload` writes `dict(sweep_conditions or {})`.

    A caller that omits the stamp therefore stores `{}`, not a missing key — that
    must mean the same thing as absent, or legacy pairs would look mismatched.
    """
    new = {"oos_sharpe": 0.489, "oos_max_dd": -0.19,
           "golden_mean_up_precision": 0.55, "sweep_conditions": {}}
    old = {"oos_sharpe": 0.590, "oos_max_dd": -0.13,
           "golden_mean_up_precision": 0.46}
    promote, reason = _promote_decision(new, old, -0.10, 3.0, 0.35)
    assert not promote
    assert "same sweep basis" in reason


# ---------------------------------------------------------------------------
# _persist_bot_payload integration — isolated filesystem (tmp_path, chdir)
# ---------------------------------------------------------------------------


def _args(sharpe=0.60, dd=-0.13, prec=0.46, horizon=20):
    cfg = RunConfig()
    cfg.tb_horizon = horizon
    golden = {"up_threshold": 0.45, "signal_threshold": 0.40,
              "total_pred_up": 1000, "mean_up_precision": prec, "n_seeds_ok": 4}
    best_seed_record = {
        "ensemble": object(), "seed": 42, "net_pnl": 1_000_000.0,
        "total_return": 0.10, "net_sharpe": sharpe, "max_drawdown": dd,
        "n_days": 900, "final_nav": 11_000_000.0,
    }
    return cfg, golden, best_seed_record


def _persist(tmp_path, monkeypatch, **kw):
    monkeypatch.chdir(tmp_path)
    cfg, golden, best_seed_record = _args(**kw)
    _persist_bot_payload(cfg, ["f1", "f2"], golden, best_seed_record,
                         macro_hmm=None, sweep_results=[], mode="tranche", hold_days=30)
    return tmp_path / "models" / "saved" / f"v3_ensemble_{cfg.tb_horizon}d.joblib"


def test_first_run_promotes_no_incumbent(tmp_path, monkeypatch):
    artifact = _persist(tmp_path, monkeypatch, sharpe=0.50)
    assert artifact.exists()
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.50)


def test_regression_rejected_incumbent_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.CONFIG.trading.promote_gate_enabled", True, raising=False)
    artifact = _persist(tmp_path, monkeypatch, sharpe=0.60)
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.60)

    # Regressed candidate — must NOT overwrite the incumbent.
    artifact2 = _persist(tmp_path, monkeypatch, sharpe=0.20)
    assert artifact2 == artifact
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.60)  # unchanged

    rejected_dir = tmp_path / "models" / "saved" / "rejected"
    assert rejected_dir.exists()
    assert len(list(rejected_dir.glob("*_REJECTED.joblib"))) == 1


def test_improvement_promotes_and_backs_up_incumbent(tmp_path, monkeypatch):
    _persist(tmp_path, monkeypatch, sharpe=0.60)
    artifact = _persist(tmp_path, monkeypatch, sharpe=0.70)
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.70)

    backup_dir = tmp_path / "models" / "saved" / "backups"
    assert backup_dir.exists()
    assert len(list(backup_dir.glob("*.joblib"))) == 1


def test_gate_disabled_always_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.CONFIG.trading.promote_gate_enabled", False, raising=False)
    _persist(tmp_path, monkeypatch, sharpe=0.60)
    artifact = _persist(tmp_path, monkeypatch, sharpe=0.10)  # would fail the gate if enabled
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.10)


def test_corrupt_incumbent_does_not_block_promotion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad_dir = tmp_path / "models" / "saved"
    bad_dir.mkdir(parents=True)
    (bad_dir / "v3_ensemble_20d.joblib").write_bytes(b"not a joblib file")

    cfg, golden, best_seed_record = _args(sharpe=0.55)
    _persist_bot_payload(cfg, ["f1"], golden, best_seed_record,
                         macro_hmm=None, sweep_results=[], mode="tranche", hold_days=30)
    artifact = bad_dir / "v3_ensemble_20d.joblib"
    assert joblib.load(artifact)["metadata"]["oos_sharpe"] == pytest.approx(0.55)
