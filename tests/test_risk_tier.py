"""Discrete risk-tier contract tests (src/trading/risk_tier.py).

Guards the guardrails:
  1. nav_cap_pct is EXACTLY one of {20, 60, 80} and always an int
     (the "no float multiplier pollution" contract).
  2. Classification boundaries are pinned (0.40 / 0.75) so a silent
     threshold edit cannot slip through a refactor.
  3. apply_nav_tier_cap clamp math — blocks new deployment above the cap,
     never goes negative, never force-liquidates (clamp only shrinks budget).
"""
from src.trading.regime_policy import NO_TRADE_REGIMES, PENALTY_REGIMES
from src.trading.risk_tier import (
    RiskTier,
    TIER_HIGH_PBULL_BELOW,
    TIER_LOW_PBULL_AT_LEAST,
    apply_nav_tier_cap,
    classify_risk_tier,
)

# ── 1. Discrete-int contract ─────────────────────────────────────────────────


def test_tier_caps_are_exact_discrete_ints():
    assert RiskTier.RISK_HIGH.nav_cap_pct == 20
    assert RiskTier.RISK_MID.nav_cap_pct == 60
    assert RiskTier.RISK_LOW.nav_cap_pct == 80
    for tier in RiskTier:
        assert type(tier.nav_cap_pct) is int
        assert tier.nav_cap_pct in {20, 60, 80}


def test_nav_cap_fraction_matches_pct():
    for tier in RiskTier:
        assert tier.nav_cap == tier.nav_cap_pct / 100.0


def test_every_tier_has_vi_label():
    for tier in RiskTier:
        assert tier.label_vi  # non-empty, no KeyError


# ── 2. Classification boundaries ─────────────────────────────────────────────


def test_thresholds_pinned():
    # If these change, the backtest A/B evidence is stale — retest before merge.
    assert TIER_HIGH_PBULL_BELOW == 0.40
    assert TIER_LOW_PBULL_AT_LEAST == 0.75


def test_low_pbull_is_high_risk():
    assert classify_risk_tier(0.0) is RiskTier.RISK_HIGH
    assert classify_risk_tier(0.39) is RiskTier.RISK_HIGH


def test_boundary_040_is_mid():
    # < 0.40 is HIGH; exactly 0.40 falls through to MID.
    assert classify_risk_tier(0.40) is RiskTier.RISK_MID


def test_transitional_band_is_mid():
    assert classify_risk_tier(0.50) is RiskTier.RISK_MID
    assert classify_risk_tier(0.749) is RiskTier.RISK_MID


def test_boundary_075_is_low():
    assert classify_risk_tier(0.75) is RiskTier.RISK_LOW
    assert classify_risk_tier(1.0) is RiskTier.RISK_LOW


def test_missing_inputs_fail_neutral_to_mid():
    assert classify_risk_tier(None) is RiskTier.RISK_MID
    assert classify_risk_tier(None, market_regime=None) is RiskTier.RISK_MID


def test_no_trade_regime_forces_high_regardless_of_pbull():
    for r in NO_TRADE_REGIMES:
        assert classify_risk_tier(0.99, market_regime=r) is RiskTier.RISK_HIGH
        assert classify_risk_tier(None, market_regime=r) is RiskTier.RISK_HIGH


def test_penalty_regime_blocks_low_but_not_mid():
    for r in PENALTY_REGIMES:
        # Bullish macro + penalty price regime → demoted from LOW to MID.
        assert classify_risk_tier(0.90, market_regime=r) is RiskTier.RISK_MID
        # Bearish macro still HIGH.
        assert classify_risk_tier(0.10, market_regime=r) is RiskTier.RISK_HIGH


def test_benign_regime_does_not_demote_low():
    # Regime 3 (strong trend) is neither NO_TRADE nor PENALTY.
    assert classify_risk_tier(0.80, market_regime=3) is RiskTier.RISK_LOW


def test_serve_scalar_proxy_preserves_boundaries():
    # Serve passes clip(p_bull, 0.2, 1.0); both thresholds sit above the 0.2
    # floor so the clip can never flip a tier decision.
    floor = 0.2
    for raw, clipped in [(0.05, max(0.05, floor)), (0.39, 0.39), (0.75, 0.75)]:
        assert classify_risk_tier(raw) is classify_risk_tier(max(clipped, floor))


# ── 3. Budget clamp math ─────────────────────────────────────────────────────

_NAV = 10_000_000_000.0


def test_clamp_noop_when_headroom_ample():
    # 10% deployed, MID cap 60% → 50% headroom ≫ budget.
    out = apply_nav_tier_cap(100.0, _NAV, deployed=0.10 * _NAV, tier=RiskTier.RISK_MID)
    assert out == 100.0


def test_clamp_shrinks_to_headroom():
    # HIGH cap 20%, 15% already deployed → only 5% NAV of new budget allowed.
    budget = 0.10 * _NAV
    out = apply_nav_tier_cap(budget, _NAV, deployed=0.15 * _NAV, tier=RiskTier.RISK_HIGH)
    assert abs(out - 0.05 * _NAV) < 1.0


def test_clamp_blocks_fully_when_over_cap():
    # Already above the cap (tier downgraded mid-flight) → zero new deployment,
    # never negative (no forced liquidation signal).
    out = apply_nav_tier_cap(100.0, _NAV, deployed=0.30 * _NAV, tier=RiskTier.RISK_HIGH)
    assert out == 0.0


def test_clamp_exact_at_cap_boundary():
    out = apply_nav_tier_cap(100.0, _NAV, deployed=0.20 * _NAV, tier=RiskTier.RISK_HIGH)
    assert out == 0.0
