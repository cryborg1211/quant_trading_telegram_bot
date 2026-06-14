"""
src/trading/regime_policy.py — canonical market-regime sizing policy constants.

Single source of truth for the rule-based regime overrides applied BEFORE any
Kelly / notional sizing. `market_regime` is the integer 0-7 produced by
`src/features/market_regime.py::build_regime_features`.

Two consumers MUST stay in lock-step (an anti-drift test enforces this):

  • SERVE path — `src/bot/sizing.py::suggested_weight` uses NO_TRADE_REGIMES,
    PENALTY_REGIMES, REGIME_PENALTY_CAP, STRONG_TREND_REGIME to veto / shrink
    the per-name NAV ceiling.
  • BACKTEST path — `src/backtest/walk_forward.py::_tranche_day` uses
    NO_TRADE_REGIMES, PENALTY_REGIMES, and REGIME_PENALTY_FACTOR to exclude
    no-trade names from the cohort and to halve the per-name notional in a
    penalty regime.

REGIME_PENALTY_FACTOR is the SCALE-INVARIANT mirror of the serve-path cap
shrink. In serve, a PENALTY regime collapses the per-name NAV ceiling from
DEFAULT_NAV_CAP (0.20) to REGIME_PENALTY_CAP (0.10) — a 0.5x reduction of the
effective size. The tranche engine's per-name notional (~nav/(H*picks) ≈ 0.7%
NAV) is far BELOW the 0.10 absolute cap, so applying the absolute cap there is
inert (a silent no-op). To mirror the serve INTENT regardless of sizing base,
the tranche engine multiplies per_name by this ratio instead:

    REGIME_PENALTY_FACTOR == REGIME_PENALTY_CAP / sizing.DEFAULT_NAV_CAP
                          == 0.10 / 0.20 == 0.5

`DEFAULT_NAV_CAP` deliberately stays in `sizing.py` (it is a Kelly default, not
a regime constant); `tests/test_regime_tranche_sizing.py` cross-checks the ratio
so the two can never silently diverge.
"""
from __future__ import annotations

# ── Market-regime structural overrides (rule-based; applied BEFORE sizing) ────
NO_TRADE_REGIMES: frozenset[int] = frozenset({0, 7})   # 0 Freeze, 7 Liquidity Sweep → stand aside
PENALTY_REGIMES:  frozenset[int] = frozenset({1, 6})   # 1 Squeeze, 6 Choppy → harsh ceiling
REGIME_PENALTY_CAP: float = 0.10                        # ≤10% NAV per name in the penalty regimes (serve cap)
STRONG_TREND_REGIME: int = 3                            # 3 Strong Trend → full Kelly up to `cap`

# Scale-invariant penalty multiplier for the tranche engine (see module docstring).
# Hardcoded 0.5 == REGIME_PENALTY_CAP / sizing.DEFAULT_NAV_CAP (0.10 / 0.20),
# enforced by an anti-drift test so a future NAV-cap change cannot silently desync.
REGIME_PENALTY_FACTOR: float = 0.5

__all__ = [
    "NO_TRADE_REGIMES",
    "PENALTY_REGIMES",
    "REGIME_PENALTY_CAP",
    "STRONG_TREND_REGIME",
    "REGIME_PENALTY_FACTOR",
]
