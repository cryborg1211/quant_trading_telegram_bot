"""
src/trading/risk_tier.py — discrete portfolio-level risk-tier NAV cap.

Three strict tiers mapping market health to a TOTAL-portfolio deployment
ceiling (per-name sizing stays owned by regime_policy.py / sizing.py — this
layer caps the SUM, not the parts):

    RISK_HIGH → 20% NAV   (bear / GARCH-brake territory / no-trade regime)
    RISK_MID  → 60% NAV   (volatile / transitional — and the fail-neutral default)
    RISK_LOW  → 80% NAV   (quiet bull)

DISCRETE-INT CONTRACT: `nav_cap_pct` returns exactly one of the ints
{20, 60, 80} — never a continuous multiplier. Enforced by tests.

EVIDENCE STATUS — READ BEFORE WIRING INTO SERVE
────────────────────────────────────────────────
UNVALIDATED. The classification thresholds below are starting points, not
calibrated constants. This module is wired into the BACKTEST tranche engine
only (``WalkForwardConfig.use_nav_tier_cap``, default OFF) so it can be
A/B-tested against the incumbent regime-sizing policy — the only risk
overlay with a validated A/B result (MaxDD −23.3%→−16.9%). Do NOT enforce
tiers on the serve path until that A/B says they beat the incumbent.

INPUT NOTE (serve proxy): the serve path only has the GARCH brake's
``exposure_scalar = clip(P(Bull), 0.2, 1.0)``, not raw P(Bull). Both tier
thresholds (0.40, 0.75) sit strictly above the 0.2 floor, so passing the
scalar as ``p_bull`` preserves every tier boundary exactly.
"""
from __future__ import annotations

from enum import Enum

from src.trading.regime_policy import NO_TRADE_REGIMES, PENALTY_REGIMES

# ── Classification thresholds (UNVALIDATED — swept by the backtest A/B) ──────
# P(Bull) below this → RISK_HIGH. Chosen to sit inside GARCH-brake territory
# (the brake floor is 0.2; a sub-0.40 posterior is already braking hard).
TIER_HIGH_PBULL_BELOW: float = 0.40
# P(Bull) at/above this (and no penalty regime) → RISK_LOW.
TIER_LOW_PBULL_AT_LEAST: float = 0.75


class RiskTier(Enum):
    """Portfolio deployment tier. Value IS the exact discrete NAV-cap percent."""

    RISK_HIGH = 20
    RISK_MID = 60
    RISK_LOW = 80

    @property
    def nav_cap_pct(self) -> int:
        """Exact discrete cap percent — one of {20, 60, 80}, always int."""
        return int(self.value)

    @property
    def nav_cap(self) -> float:
        """Cap as a NAV fraction for sizing math (0.20 / 0.60 / 0.80)."""
        return self.value / 100.0

    @property
    def label_vi(self) -> str:
        """Professional-VN display label for the Telegram card."""
        return _LABELS_VI[self]


_LABELS_VI: dict[RiskTier, str] = {
    RiskTier.RISK_HIGH: "CAO — phòng thủ",
    RiskTier.RISK_MID: "TRUNG BÌNH — thận trọng",
    RiskTier.RISK_LOW: "THẤP — thuận lợi",
}


def classify_risk_tier(
    p_bull: float | None,
    market_regime: int | None = None,
) -> RiskTier:
    """Map market health to a discrete tier.

    Args:
        p_bull: Macro bull-state posterior in [0, 1] (GARCH-HMM ``p_bull_latest``,
            the backtest's ``p_bull_series`` value, or the serve exposure scalar —
            see module INPUT NOTE). ``None`` = macro state unavailable.
        market_regime: Optional 0–7 price regime (``build_regime_features``).
            A market-wide reading, not one name's — callers with only
            per-ticker regimes should pass the day's modal regime or ``None``.

    Rules (first match wins):
        1. regime ∈ NO_TRADE_REGIMES {0,7}      → RISK_HIGH
        2. p_bull < 0.40                        → RISK_HIGH
        3. p_bull ≥ 0.75 and regime ∉ PENALTY   → RISK_LOW
        4. everything else (incl. both-None)    → RISK_MID

    Fail-neutral: missing inputs land in RISK_MID, never crash — same
    degrade philosophy as the GARCH brake's fail-open, but the middle course
    is the sane default for a *portfolio cap* (fail-open to 80% would let a
    data outage silently max out deployment).
    """
    if market_regime is not None and int(market_regime) in NO_TRADE_REGIMES:
        return RiskTier.RISK_HIGH
    if p_bull is not None:
        p = float(p_bull)
        if p < TIER_HIGH_PBULL_BELOW:
            return RiskTier.RISK_HIGH
        if p >= TIER_LOW_PBULL_AT_LEAST and (
            market_regime is None or int(market_regime) not in PENALTY_REGIMES
        ):
            return RiskTier.RISK_LOW
    return RiskTier.RISK_MID


def apply_nav_tier_cap(
    budget: float,
    nav: float,
    deployed: float,
    tier: RiskTier,
) -> float:
    """Clamp a new-deployment budget so total deployment never exceeds the tier cap.

    ``deployed`` is the mark-to-market value already invested (nav − cash for a
    long-only unlevered book). The cap BLOCKS/shrinks NEW deployment only — it
    never force-liquidates existing positions (a tier downgrade bleeds exposure
    down naturally as tranches expire).

    Pure function so the clamp math is unit-testable without the engine.
    """
    headroom = tier.nav_cap * nav - deployed
    return min(budget, max(headroom, 0.0))


__all__ = [
    "RiskTier",
    "classify_risk_tier",
    "apply_nav_tier_cap",
    "TIER_HIGH_PBULL_BELOW",
    "TIER_LOW_PBULL_AT_LEAST",
]
