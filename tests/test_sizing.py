"""Kelly position-sizing contract — pure, fast, no heavy deps.

Locks the V4.0 sizing config (R=2.0, half-Kelly, 20% cap, top-5) and the weight
curve.  Would have caught the 'N/A sizing' regression at its source: the
suggested_weight math.
"""
from src.bot.sizing import (
    suggested_weight,
    rank_buy_signals,
    DEFAULT_NAV_CAP,
    DEFAULT_REWARD_TO_RISK,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_TOP_N,
    BUY_THRESHOLD,
)


def test_config_locked():
    assert DEFAULT_REWARD_TO_RISK == 2.0
    assert DEFAULT_KELLY_FRACTION == 0.5
    assert DEFAULT_NAV_CAP == 0.20          # the user's decisive 20% cap
    assert DEFAULT_TOP_N == 5               # 5 x 20% = 100% gross, unlevered
    assert BUY_THRESHOLD == 0.50


def test_weight_curve():
    # R=2.0, half-Kelly, 20% cap → w = min(max(0, 0.75p - 0.25), 0.20)
    assert suggested_weight(0.30) == 0.0                  # below break-even (p<1/3)
    assert abs(suggested_weight(0.42) - 0.065) < 1e-9     # the old 'N/A' band now sizes
    assert abs(suggested_weight(0.50) - 0.125) < 1e-9
    assert abs(suggested_weight(0.55) - 0.1625) < 1e-9
    assert abs(suggested_weight(0.60) - 0.20) < 1e-9      # cap-onset exactly at 0.60
    assert suggested_weight(0.66) == 0.20                 # pinned at cap beyond 0.60


def test_weight_monotonic_nondecreasing():
    ws = [suggested_weight(p / 100) for p in range(33, 70)]
    assert all(b >= a - 1e-12 for a, b in zip(ws, ws[1:]))


def test_rank_buy_signals_gate_and_cap():
    # 8 strong names → only top DEFAULT_TOP_N returned, all pinned at the cap.
    out = rank_buy_signals([(f"T{i}", 0.70) for i in range(8)])
    assert len(out) == DEFAULT_TOP_N
    assert all(abs(s.suggested_weight - 0.20) < 1e-9 for s in out)
    assert sum(s.suggested_weight for s in out) == 1.00   # 100% NAV worst-case gross


def test_rank_buy_signals_filters_below_threshold():
    assert rank_buy_signals([("X", 0.40), ("Y", 0.49)]) == []
    assert [s.ticker for s in rank_buy_signals([("Z", 0.51)])] == ["Z"]
