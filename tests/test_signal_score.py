"""Checklist score + disqualifier warnings for the signal card (10-08-26)."""
from __future__ import annotations

from src.reports.signal_score import (
    TOTAL_CHECKS,
    build_facts,
    score_line,
    score_signal,
    warning_lines,
)
from src.trading.regime_policy import NO_TRADE_REGIMES, PENALTY_REGIMES


def test_all_none_scores_zero_not_full():
    # A missing input must never silently count as a pass.
    facts = build_facts()
    assert score_signal(facts) == (0, TOTAL_CHECKS)


def test_perfect_signal_scores_full():
    facts = build_facts(
        p_up_20d=0.60, tau_20d=0.46, p_up_5d=0.60, tau_5d=0.44,
        room_exhausted=False, market_regime=3, garch_scalar=1.0,
        breadth=0.55, sentiment_score=0.4, mr_fired=True,
    )
    assert score_signal(facts) == (TOTAL_CHECKS, TOTAL_CHECKS)
    assert score_line(facts) == f"ĐIỂM TỔNG: {TOTAL_CHECKS}/{TOTAL_CHECKS}"


def test_gate_uses_the_supplied_threshold():
    below = build_facts(p_up_20d=0.45, tau_20d=0.46)
    at = build_facts(p_up_20d=0.46, tau_20d=0.46)
    assert below["gate_20d"] is False
    assert at["gate_20d"] is True          # inclusive, mirrors the serve meta-gate


def test_penalty_regime_is_not_regime_ok():
    for r in PENALTY_REGIMES:
        assert build_facts(market_regime=r)["regime_ok"] is False
    for r in NO_TRADE_REGIMES:
        assert build_facts(market_regime=r)["regime_ok"] is False
    assert build_facts(market_regime=3)["regime_ok"] is True


def test_brake_engaged_fails_no_brake():
    assert build_facts(garch_scalar=0.82)["no_brake"] is False
    assert build_facts(garch_scalar=1.0)["no_brake"] is True


def test_breadth_threshold_respected():
    assert build_facts(breadth=0.19)["breadth_ok"] is False
    assert build_facts(breadth=0.55)["breadth_ok"] is True


# ── warnings ────────────────────────────────────────────────────────────────

def test_no_warnings_on_a_clean_signal():
    facts = build_facts(
        p_up_20d=0.60, tau_20d=0.46, p_up_5d=0.60, tau_5d=0.44,
        room_exhausted=False, market_regime=3, garch_scalar=1.0,
        breadth=0.55, sentiment_score=0.4, mr_fired=True,
    )
    assert warning_lines(facts, market_regime=3) == []


def test_room_exhausted_warns():
    facts = build_facts(room_exhausted=True)
    assert any("room ngoại" in w for w in warning_lines(facts))


def test_t5_only_warns():
    facts = build_facts(p_up_20d=0.40, tau_20d=0.46, p_up_5d=0.60, tau_5d=0.44)
    warns = warning_lines(facts)
    assert any("T+5" in w for w in warns)


def test_t5_only_warning_absent_when_t20_also_passes():
    facts = build_facts(p_up_20d=0.60, tau_20d=0.46, p_up_5d=0.60, tau_5d=0.44)
    assert not any("T+5" in w for w in warning_lines(facts))


def test_no_trade_regime_warns():
    for r in NO_TRADE_REGIMES:
        warns = warning_lines(build_facts(market_regime=r), market_regime=r)
        assert any("KHÔNG giao dịch" in w for w in warns)


def test_penalty_regime_does_not_warn():
    # Penalty regimes are context, not a disqualifier — a warning list that
    # fires constantly stops being read.
    for r in PENALTY_REGIMES:
        warns = warning_lines(build_facts(market_regime=r), market_regime=r)
        assert not any("KHÔNG giao dịch" in w for w in warns)
