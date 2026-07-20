"""Probability-scaled within-cohort weight fractions (shared serve/backtest).

Single source of truth for the prob-weighted tranche formula (plan
`prob-scaled-tranche-weights_PLAN_20-07-26.md`) — imported by BOTH
`src/backtest/walk_forward.py` and `main._dispatch_signals`, mirroring the
`regime_policy.py` no-duplication precedent.

Formula: edge_i = max(0, p_up_i − threshold); fraction_i = edge_i / Σ edge,
water-filled so no name exceeds `cap_mult ×` its equal-weight share (the
July-2026 lesson: never let a scaling rule re-concentrate the book). Zero
total edge (or any degenerate input) falls back to equal weight — the
pre-existing validated behavior.
"""
from __future__ import annotations


def prob_scaled_weights(
    p_ups: list[float],
    threshold: float,
    cap_mult: float = 2.0,
) -> list[float]:
    """Within-cohort weight fractions (sum to 1.0) from calibrated P(UP)s.

    Pure + deterministic. Falls back to equal weight when there is no positive
    edge or the cap is infeasible (`cap_mult < 1` cannot hold n fractions
    summing to 1). Water-fills iteratively: capped names freeze at the cap,
    the remainder re-splits by edge among the uncapped.
    """
    n = len(p_ups)
    if n == 0:
        return []
    equal = 1.0 / n
    if n == 1:
        return [1.0]
    cap = cap_mult * equal
    if cap < equal or cap_mult < 1.0:
        return [equal] * n

    edges = [max(0.0, float(p) - float(threshold)) for p in p_ups]
    if sum(edges) <= 0.0:
        return [equal] * n

    fracs = [0.0] * n
    capped: set[int] = set()
    remaining = 1.0
    # ≤ n rounds: each round either converges or caps ≥1 new name.
    for _ in range(n):
        free = [i for i in range(n) if i not in capped]
        edge_sum = sum(edges[i] for i in free)
        if edge_sum <= 0.0:
            # No edge left among the uncapped — split the remainder equally.
            for i in free:
                fracs[i] = remaining / len(free)
            break
        overflow = False
        for i in free:
            fracs[i] = remaining * edges[i] / edge_sum
        for i in free:
            if fracs[i] > cap + 1e-12:
                fracs[i] = cap
                capped.add(i)
                overflow = True
        if not overflow:
            break
        remaining = 1.0 - cap * len(capped)
        if remaining <= 0.0:
            # Everyone at cap (only possible when cap·n ≈ 1) — equal fallback.
            return [equal] * n
    return fracs
