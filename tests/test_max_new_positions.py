"""How many new names one dispatch may open — and the invariant that must hold.

HISTORY
───────
13-08-26: raised 3 → 5 to match the engine's validated `max_positions=5` (serve
had a bare `[:3]` literal). 26-08-26: back to 3 on an operator breadth decision.

The count itself is a preference and is expected to move. What must NOT move is
the invariant below.

THE INVARIANT: changing the count is NOT a leverage change. The tranche cohort
weight is `min(1/(hold_days × n_picks), 0.20)`, so the daily budget NAV/hold_days
is SPLIT across today's picks — 3.33% NAV/day deployed whether that is 3 names or
5. FEWER names therefore means MORE concentration per name, not less money at
risk. If a future edit turns the split into extra exposure, this file fails.

Why the count started mattering at all: top-3-vs-top-5 was measured VACUOUS under
τ=0.46 because the admitted set never reached 3 names. That carried the condition
"this changes if the gate is ever loosened" — and 0.46 → 0.43 → 0.42 loosened it
to a measured ~5 liquid names clearing per session.
"""
from __future__ import annotations

import pytest

import main
from main import _MAX_NEW_POSITIONS_PER_DAY, _tranche_signal_fields

_STRATEGY = {"mode": "tranche", "hold_days": 30, "signal_threshold": 0.43}


def test_slice_is_the_operator_configured_count():
    assert _MAX_NEW_POSITIONS_PER_DAY == 3


def test_arbitrator_pool_can_fill_a_full_dispatch():
    """Pool < slice would silently starve the dispatch.

    The engine's own shape is `admission_pool_cap` (6) >= `max_positions` (5);
    serve keeps the same relation whatever the slice is set to. This is also why
    `admission_mode` measured vacuous: the cap never binds.
    """
    assert main._ARBITRATOR_POOL_SIZE >= _MAX_NEW_POSITIONS_PER_DAY


def test_total_daily_deployment_is_unchanged_by_the_pick_count():
    """THE invariant. Deliberately swept 1..6 — a FIXED range, not
    `range(1, _MAX_NEW_POSITIONS_PER_DAY + 1)`.

    Tying the sweep to the configured count would let the test silently weaken
    every time that count is lowered: at 3 it would stop checking 4 and 5 at all.
    The invariant is a property of the weight formula, not of today's preference.
    """
    for n in range(1, 7):
        w = _tranche_signal_fields(_STRATEGY, n_picks=n, horizon=20)["suggested_weight"]
        # 1/30 = 3.333% NAV/day regardless of how many names it is split across.
        assert w * n == pytest.approx(1.0 / 30.0), f"{n} picks deployed {w * n:.4%}"


def test_per_name_weight_shrinks_as_picks_grow():
    """FEWER names = MORE per name. The direction operators most often get wrong."""
    w1 = _tranche_signal_fields(_STRATEGY, n_picks=1, horizon=20)["suggested_weight"]
    w3 = _tranche_signal_fields(_STRATEGY, n_picks=3, horizon=20)["suggested_weight"]
    w5 = _tranche_signal_fields(_STRATEGY, n_picks=5, horizon=20)["suggested_weight"]
    assert w5 < w3 < w1
    assert w3 == pytest.approx(1.0 / 90.0)    # 1.111%
    assert w5 == pytest.approx(1.0 / 150.0)   # 0.667%


def test_a_full_dispatch_stays_far_below_the_per_name_nav_cap():
    """The 20% cap exists for tiny-horizon single picks.

    Checked at the CONFIGURED count, since that is the concentration actually
    shipped — the whole point of lowering the count is a bigger per-name weight.
    """
    w = _tranche_signal_fields(
        _STRATEGY, n_picks=_MAX_NEW_POSITIONS_PER_DAY, horizon=20)["suggested_weight"]
    assert w < 0.20
    assert w < 0.02     # 1.111% at 3 picks — still an order below the cap


def test_short_horizon_cards_split_the_same_way():
    """T+5 cards size on their own horizon, so verify the split holds there too."""
    for n in (3, 5):
        w = _tranche_signal_fields(_STRATEGY, n_picks=n, horizon=5)["suggested_weight"]
        assert w * n == pytest.approx(1.0 / 5.0)


def test_zero_picks_does_not_divide_by_zero():
    w = _tranche_signal_fields(_STRATEGY, n_picks=0, horizon=20)["suggested_weight"]
    assert w == pytest.approx(1.0 / 30.0)     # max(1, n_picks) guard
