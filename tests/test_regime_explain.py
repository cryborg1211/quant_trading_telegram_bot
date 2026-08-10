"""Regime explanation lines for the signal card (10-08-26).

The wording must be DERIVED from regime_policy, never restated as literals,
so a policy change cannot leave the card explaining an action the system no
longer takes.
"""
from __future__ import annotations

from src.features.market_regime import REGIME_LABELS_VI
from src.reports.regime_explain import regime_action_label, regime_explain_lines
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_FACTOR,
    STRONG_TREND_REGIME,
)


def test_none_and_unknown_regime_yield_no_lines():
    assert regime_explain_lines(None) == []
    assert regime_explain_lines(99) == []


def test_every_known_regime_produces_two_lines():
    for r in REGIME_LABELS_VI:
        lines = regime_explain_lines(r)
        assert len(lines) == 2, f"regime {r} produced {len(lines)} line(s)"
        assert REGIME_LABELS_VI[r] in lines[0]
        assert f"Regime {r}" in lines[0]


def test_no_trade_regimes_say_do_not_trade():
    for r in NO_TRADE_REGIMES:
        assert "KHÔNG giao dịch" in regime_action_label(r)
        assert "KHÔNG giao dịch" in regime_explain_lines(r)[0]


def test_penalty_regimes_quote_the_actual_policy_factor():
    for r in PENALTY_REGIMES:
        label = regime_action_label(r)
        assert "giảm tỷ trọng" in label
        # Must echo the live constant, not a hardcoded 0.5.
        assert f"{REGIME_PENALTY_FACTOR:.2f}" in label


def test_strong_trend_allows_max_weight():
    assert "tối đa" in regime_action_label(STRONG_TREND_REGIME)


def test_neutral_regimes_are_normal():
    neutral = (set(REGIME_LABELS_VI) - set(NO_TRADE_REGIMES)
               - set(PENALTY_REGIMES) - {STRONG_TREND_REGIME})
    assert neutral, "expected at least one neutral regime"
    for r in neutral:
        assert "bình thường" in regime_action_label(r)


def test_action_label_unknown_is_safe():
    assert regime_action_label(None) == "không xác định"
