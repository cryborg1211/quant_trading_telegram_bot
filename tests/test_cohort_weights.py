"""Prob-scaled cohort weights (plan prob-scaled-tranche-weights_PLAN_20-07-26).

Pure-formula tests plus the serve/backtest parity pin: both paths import THE
SAME `prob_scaled_weights` — these tests are the single spec for it.
"""
from __future__ import annotations

import pytest

from src.trading.cohort_weights import prob_scaled_weights


def test_empty_and_single():
    assert prob_scaled_weights([], 0.4) == []
    assert prob_scaled_weights([0.6], 0.4) == [1.0]


def test_zero_edge_falls_back_to_equal():
    assert prob_scaled_weights([0.30, 0.35, 0.40], 0.40) == [pytest.approx(1 / 3)] * 3


def test_proportional_to_edge():
    # edges 0.02 / 0.04 → 1:2 split within cap.
    w = prob_scaled_weights([0.46, 0.48], 0.44, cap_mult=2.0)
    assert w[0] == pytest.approx(1 / 3)
    assert w[1] == pytest.approx(2 / 3)
    assert sum(w) == pytest.approx(1.0)


def test_cap_binds_and_waterfills():
    # edges 0.10 / 0.01 / 0.01 → raw 0.833 exceeds cap 2/3; capped name freezes,
    # remainder splits by edge (equal here) among the rest.
    w = prob_scaled_weights([0.54, 0.45, 0.45], 0.44, cap_mult=2.0)
    assert w[0] == pytest.approx(2 / 3)
    assert w[1] == pytest.approx(1 / 6)
    assert w[2] == pytest.approx(1 / 6)
    assert sum(w) == pytest.approx(1.0)


def test_never_exceeds_cap():
    for n in (2, 3, 5, 10):
        p = [0.9] + [0.45] * (n - 1)
        w = prob_scaled_weights(p, 0.44, cap_mult=2.0)
        assert sum(w) == pytest.approx(1.0)
        assert max(w) <= 2.0 / n + 1e-9


def test_infeasible_cap_falls_back_to_equal():
    assert prob_scaled_weights([0.5, 0.6], 0.4, cap_mult=0.5) == [0.5, 0.5]


def test_deterministic():
    a = prob_scaled_weights([0.5, 0.47, 0.46], 0.44)
    b = prob_scaled_weights([0.5, 0.47, 0.46], 0.44)
    assert a == b


def test_backtest_and_serve_share_the_formula():
    # Parity pin: both consumers must resolve to THIS function object.
    import main
    from src.backtest import walk_forward

    assert main.prob_scaled_weights is prob_scaled_weights
    assert walk_forward.prob_scaled_weights is prob_scaled_weights
