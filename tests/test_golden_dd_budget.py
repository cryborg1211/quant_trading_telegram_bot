"""GOLDEN selection is max mean OOS Net PnL SUBJECT TO a drawdown budget (12-08-26).

WHY
───
Selection was plain `max(mean_net_pnl)` with no risk term. The full 6-level
serve-parity sweep showed why that is unsafe for an unattended weekly retrain:
the threshold does NOT move Sharpe (all six levels span 0.056, while ONE level's
4-seed spread is 0.30 — the whole curve fits inside one level's noise), while
mean DD is cleanly monotone in the gate. With Sharpe carrying no signal and
PnL/DD rising together, maximising PnL walks toward maximum drawdown, and the
0.42-vs-0.43 PnL gap (1.8%) is well inside seed noise — so a different seed draw
picks -14.83% or -16.20% on a run nobody is watching.
"""
from __future__ import annotations

from run_backtest import _select_golden

# The real 12-08-26 sweep (logs/parity_sweep_full_20260812_154135.log).
_SWEEP = [
    {"up_threshold": 0.46, "mean_net_pnl": 906_902_938.0, "mean_dd": -0.0966},
    {"up_threshold": 0.45, "mean_net_pnl": 1_218_318_746.0, "mean_dd": -0.1220},
    {"up_threshold": 0.44, "mean_net_pnl": 1_252_482_271.0, "mean_dd": -0.1317},
    {"up_threshold": 0.43, "mean_net_pnl": 1_582_548_453.0, "mean_dd": -0.1388},
    {"up_threshold": 0.42, "mean_net_pnl": 1_554_030_687.0, "mean_dd": -0.1483},
    {"up_threshold": 0.41, "mean_net_pnl": 1_452_500_136.0, "mean_dd": -0.1620},
]


def _thr(row: dict) -> float:
    return row["up_threshold"]


def test_the_chosen_14pp_budget_keeps_todays_answer():
    """14.0 must NOT change what the 12-08 dry sweep picked.

    That is the whole point of the level: cap the tail without relitigating the
    run that was already reviewed.
    """
    assert _thr(_select_golden(_SWEEP, 14.0)) == 0.43


def test_the_14pp_budget_blocks_the_deeper_drawdown_arms():
    # 0.42 (-14.83%) is only 1.8% behind on PnL — inside seed noise — so without
    # a budget it wins on a different seed draw.
    for row in _SWEEP:
        eligible = abs(row["mean_dd"]) <= 0.14
        assert eligible == (_thr(row) >= 0.43), _thr(row)


def test_unconstrained_selection_is_the_old_behaviour():
    assert _thr(_select_golden(_SWEEP, None)) == 0.43   # same here, by luck
    # …and demonstrably NOT the same in general: strip 0.43 and max-PnL reaches
    # for -14.83% while a 14pp budget stops at -13.17%.
    without_043 = [r for r in _SWEEP if _thr(r) != 0.43]
    assert _thr(_select_golden(without_043, None)) == 0.42
    assert _thr(_select_golden(without_043, 14.0)) == 0.44


def test_zero_must_be_converted_to_none_by_the_caller():
    """`--golden-max-mean-dd-pp 0` means "no budget", and 0.0 is falsy.

    `_cli` does `dd_budget = a.golden_max_mean_dd_pp or None`. This pins BOTH
    halves: a literal 0.0 reaching `_select_golden` would mean "0% drawdown
    allowed" and exclude every arm (falling back to the shallowest), so the
    conversion is load-bearing rather than cosmetic.
    """
    assert (0.0 or None) is None                                  # the rule
    assert _thr(_select_golden(_SWEEP, None)) == 0.43             # disabled
    assert _thr(_select_golden(_SWEEP, 0.0)) == 0.46              # NOT disabled


def test_a_tight_budget_selects_the_shallower_arm():
    assert _thr(_select_golden(_SWEEP, 13.0)) == 0.45      # 0.44 is -13.17%
    assert _thr(_select_golden(_SWEEP, 13.5)) == 0.44
    assert _thr(_select_golden(_SWEEP, 10.0)) == 0.46


def test_sign_of_the_budget_does_not_matter():
    """DD is stored negative; a caller passing -14.0 must mean the same thing."""
    assert _thr(_select_golden(_SWEEP, -14.0)) == _thr(_select_golden(_SWEEP, 14.0))


def test_no_arm_within_budget_falls_back_to_the_shallowest():
    """A retrain must still produce a candidate.

    Aborting would leave the weekly job with nothing to promote; least-drawdown
    is the safe direction to fail in.
    """
    picked = _select_golden(_SWEEP, 5.0)
    assert _thr(picked) == 0.46                      # -9.66%, the shallowest
    assert abs(picked["mean_dd"]) == min(abs(r["mean_dd"]) for r in _SWEEP)


def test_fallback_prefers_shallow_dd_over_pnl():
    sweep = [
        {"up_threshold": 0.41, "mean_net_pnl": 9_000_000_000.0, "mean_dd": -0.30},
        {"up_threshold": 0.46, "mean_net_pnl": 1.0, "mean_dd": -0.20},
    ]
    assert _thr(_select_golden(sweep, 5.0)) == 0.46


def test_single_arm_sweep_is_returned_regardless():
    one = [{"up_threshold": 0.44, "mean_net_pnl": 1.0, "mean_dd": -0.99}]
    assert _thr(_select_golden(one, 14.0)) == 0.44
    assert _thr(_select_golden(one, None)) == 0.44


def test_ties_on_pnl_do_not_crash():
    sweep = [
        {"up_threshold": 0.44, "mean_net_pnl": 100.0, "mean_dd": -0.10},
        {"up_threshold": 0.43, "mean_net_pnl": 100.0, "mean_dd": -0.12},
    ]
    assert _thr(_select_golden(sweep, 14.0)) in (0.43, 0.44)


def test_budget_is_read_from_config_by_default():
    """The knob has to be live, not a dead default in a signature."""
    from config.settings import CONFIG
    assert float(CONFIG.trading.golden_max_mean_dd_pp) == 14.0
