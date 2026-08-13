"""Serve opens up to 5 new names/day, matching the engine's max_positions (13-08-26).

WHY
───
Serve sliced its dispatch list to a bare `[:3]` literal while the validated engine
runs `max_positions=5`. That was measured VACUOUS earlier — under τ=0.46 the
admitted set never reached 3 names anyway — but the finding carried the condition
"this changes if the gate is ever loosened", and the 13-08 retrain loosened it
(τ 0.46 → 0.43 took scores clearing the gate from 1.1% to 5.5%).

THE PROPERTY THAT MAKES THIS SAFE, and the one worth pinning: it is NOT a leverage
change. The tranche cohort weight is `min(1/(hold_days × n_picks), 0.20)`, so the
daily budget NAV/hold_days is SPLIT across today's picks — 3.33% NAV/day deployed
whether that is 3 names or 5. Raising the count lowers single-name concentration
and leaves total exposure alone. If a future edit breaks that, this file fails.
"""
from __future__ import annotations

import pytest

import main
from main import _MAX_NEW_POSITIONS_PER_DAY, _tranche_signal_fields

_STRATEGY = {"mode": "tranche", "hold_days": 30, "signal_threshold": 0.43}


def test_serve_slice_matches_the_engines_max_positions():
    assert _MAX_NEW_POSITIONS_PER_DAY == 5


def test_arbitrator_pool_can_fill_a_full_dispatch():
    """Pool < slice would silently starve the dispatch.

    Mirrors the engine's `admission_pool_cap` (6) >= `max_positions` (5). This is
    also why `admission_mode` measured vacuous: the cap never binds.
    """
    assert main._ARBITRATOR_POOL_SIZE >= _MAX_NEW_POSITIONS_PER_DAY


def test_total_daily_deployment_is_unchanged_by_the_pick_count():
    """THE invariant. Same NAV/day at 1..5 picks — a split, not extra leverage."""
    totals = {}
    for n in range(1, _MAX_NEW_POSITIONS_PER_DAY + 1):
        w = _tranche_signal_fields(_STRATEGY, n_picks=n, horizon=20)["suggested_weight"]
        totals[n] = w * n
    # 1/30 = 3.333% NAV/day regardless of how many names it is split across.
    for n, total in totals.items():
        assert total == pytest.approx(1.0 / 30.0), f"{n} picks deployed {total:.4%}"


def test_per_name_weight_shrinks_as_picks_grow():
    w3 = _tranche_signal_fields(_STRATEGY, n_picks=3, horizon=20)["suggested_weight"]
    w5 = _tranche_signal_fields(_STRATEGY, n_picks=5, horizon=20)["suggested_weight"]
    assert w5 < w3
    assert w3 == pytest.approx(1.0 / 90.0)    # 1.111%
    assert w5 == pytest.approx(1.0 / 150.0)   # 0.667%


def test_five_picks_stay_far_below_the_per_name_nav_cap():
    """The 20% cap exists for tiny-horizon single picks; 5 picks must not near it."""
    w = _tranche_signal_fields(_STRATEGY, n_picks=5, horizon=20)["suggested_weight"]
    assert w < 0.20
    assert w < 0.01     # 0.667% — two orders below the cap


def test_short_horizon_cards_also_split_across_five():
    """T+5 cards size on their own horizon, so verify the split holds there too."""
    w = _tranche_signal_fields(_STRATEGY, n_picks=5, horizon=5)["suggested_weight"]
    assert w * 5 == pytest.approx(1.0 / 5.0)


def test_zero_picks_does_not_divide_by_zero():
    w = _tranche_signal_fields(_STRATEGY, n_picks=0, horizon=20)["suggested_weight"]
    assert w == pytest.approx(1.0 / 30.0)     # max(1, n_picks) guard
