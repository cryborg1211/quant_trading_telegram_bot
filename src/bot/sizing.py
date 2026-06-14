"""
src/bot/sizing.py — V3.2 position-sizing primitives.

Pure functions.  No I/O, no globals, no logging.  All math is closed-form so
it's instantly unit-testable in isolation from the bot/inference plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Market-regime override constants live in the shared single-source module so the
# serve path (here) and the backtest engine (walk_forward._tranche_day) can never
# drift. Re-exported below for backward-compatible `from src.bot.sizing import ...`.
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_CAP,
    STRONG_TREND_REGIME,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — single source of truth for V3.2 Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REWARD_TO_RISK = 2.0    # R in the Kelly formula (2:1 reward:risk assumption) — LOCKED
DEFAULT_KELLY_FRACTION = 0.5    # half-Kelly (defensive)
# 20% NAV ceiling per name.  With R=2.0 + half-Kelly the cap only binds at
#   p >= (2·cap·R + 1)/(R + 1) = 0.60,
# so the calibrated sweet-spot p∈[0.50,0.55] sizes SMOOTHLY at 12.5%–16.25% NAV
# instead of flatlining at the cap (the prior 10% cap bound from p≈0.467).
DEFAULT_NAV_CAP        = 0.20
BUY_THRESHOLD          = 0.50   # P(UP) >= this triggers a BUY signal
# Max names the bot advises.  5 × 20% cap = 100% NAV worst-case gross
# (long-only, unlevered) — preserves the 100% ceiling the old 10 × 10% gave, so
# raising the per-name cap does NOT silently introduce portfolio leverage.
DEFAULT_TOP_N          = 5

# ── Market-regime structural overrides (rule-based; applied BEFORE Kelly) ─────
# `market_regime` is the integer 0–7 from src/features/market_regime.py.
# NO_TRADE_REGIMES, PENALTY_REGIMES, REGIME_PENALTY_CAP, STRONG_TREND_REGIME are
# imported above from src.trading.regime_policy (single source of truth).


@dataclass(frozen=True)
class KellySizing:
    """One per-ticker sizing decision."""
    ticker: str
    p_up: float
    raw_kelly: float          # the un-clipped Kelly fraction
    suggested_weight: float   # post-half + post-cap; what the bot reports


def kelly_fraction(p_up: float, reward_to_risk: float = DEFAULT_REWARD_TO_RISK) -> float:
    """Raw Kelly fraction.  May be negative — caller decides what to do with that."""
    p = float(p_up)
    return p - (1.0 - p) / float(reward_to_risk)


def suggested_weight(
    p_up: float,
    *,
    market_regime: int | None = None,
    reward_to_risk: float = DEFAULT_REWARD_TO_RISK,
    kelly_fraction_used: float = DEFAULT_KELLY_FRACTION,
    cap: float = DEFAULT_NAV_CAP,
) -> float:
    """
    Half-Kelly NAV weight with a hard cap, modulated by the market regime.

        kelly      = p_up - (1 - p_up) / R
        half_kelly = max(0, kelly * kelly_fraction_used)
        weight     = min(half_kelly, cap)

    Returns 0.0 for any p_up below the break-even point (negative Kelly).  At
    R=2.0 + half-Kelly + the 20% cap the weight scales smoothly from break-even
    (p ≈ 0.333) and only saturates at the cap for p ≥ 0.60 — so the calibrated
    band p∈[0.50,0.55] maps to 12.5%–16.25% NAV.

    Regime overrides (rule-based, applied BEFORE the Kelly math; the structural
    state of the tape vetoes/penalises sizing regardless of P(UP)):
      • 0 Freeze / 7 Liquidity Sweep → 0.0  (do NOT trade — dead or trap tape)
      • 1 Squeeze  / 6 Choppy        → cap shrunk to REGIME_PENALTY_CAP (≤10% NAV)
      • 3 Strong Trend               → full half-Kelly up to the 20% `cap`
      • 2/4/5 and `market_regime=None` (default) → unmodified half-Kelly to `cap`
        (so existing call sites and unit tests are byte-for-byte unchanged).
    """
    if market_regime is not None:
        r = int(market_regime)
        if r in NO_TRADE_REGIMES:
            return 0.0                              # Freeze / Liquidity Sweep — stand aside
        if r in PENALTY_REGIMES:
            cap = min(cap, REGIME_PENALTY_CAP)      # Squeeze / Choppy — shrink the ceiling
        # Strong Trend (3) + the rest size by full half-Kelly up to `cap`.
    k = kelly_fraction(p_up, reward_to_risk)
    return float(min(max(0.0, k * kelly_fraction_used), cap))


def rank_buy_signals(
    predictions: Iterable[tuple[str, float]],
    *,
    threshold: float = BUY_THRESHOLD,
    reward_to_risk: float = DEFAULT_REWARD_TO_RISK,
    kelly_fraction_used: float = DEFAULT_KELLY_FRACTION,
    cap: float = DEFAULT_NAV_CAP,
    top_n: int = DEFAULT_TOP_N,
) -> list[KellySizing]:
    """
    Given an iterable of (ticker, P(UP)), return KellySizing records for every
    name above `threshold`, sorted by descending P(UP), capped at `top_n`.

    This is the ONE function the bot needs to call — it does the gating, the
    sizing, and the ordering in a single pure step.
    """
    rows: list[KellySizing] = []
    for ticker, p in predictions:
        try:
            p_f = float(p)
        except (TypeError, ValueError):
            continue
        if p_f < threshold:
            continue
        rk = kelly_fraction(p_f, reward_to_risk)
        w = float(min(max(0.0, rk * kelly_fraction_used), cap))
        rows.append(KellySizing(ticker=str(ticker), p_up=p_f, raw_kelly=rk, suggested_weight=w))
    rows.sort(key=lambda r: -r.p_up)
    return rows[:top_n]


__all__ = [
    "DEFAULT_REWARD_TO_RISK",
    "DEFAULT_KELLY_FRACTION",
    "DEFAULT_NAV_CAP",
    "DEFAULT_TOP_N",
    "BUY_THRESHOLD",
    "NO_TRADE_REGIMES",
    "PENALTY_REGIMES",
    "REGIME_PENALTY_CAP",
    "STRONG_TREND_REGIME",
    "KellySizing",
    "kelly_fraction",
    "suggested_weight",
    "rank_buy_signals",
]
