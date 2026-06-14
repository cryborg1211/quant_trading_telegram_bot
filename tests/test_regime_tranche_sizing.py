"""Regime-conditional tranche sizing.

The backtest engine mirrors the serve-path src/bot/sizing.py regime policy via the
shared src/trading/regime_policy constants:

  • NO_TRADE_REGIMES {0,7} → the name is skipped for that day's cohort; its
    per-name share stays as CASH (it is NOT redistributed to survivors — that
    smaller deployment on bad-regime days is the DD-reducing behaviour).
  • PENALTY_REGIMES  {1,6} → the name's per-name notional is multiplied by
    REGIME_PENALTY_FACTOR (0.5×); the freed half stays cash. A 0.5× multiplier
    (not the serve-path absolute 0.10×NAV cap) is used because the tranche
    per-name notional (~nav/(H·picks) ≈ 0.7% NAV) is far below 10% NAV, so the
    absolute cap would be an inert no-op.

`use_regime_sizing` defaults to False → byte-for-byte the original behaviour.

Synthetic 4-ticker constant-price panel; the per-ticker `feat` value doubles as
the oracle score, so AAA=0.90 and BBB=0.80 are the top-2 picks (max_positions=2),
DDD=0.45 is third, CCC=0.10 never qualifies. NAV=1B, HOLD=5 ⇒ day-1 budget=NAV/5
=200M, per_name=budget/2=100M (price 20k VND ⇒ exact lot fills, no rounding loss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from src.bot import sizing as serve_sizing
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_CAP,
    REGIME_PENALTY_FACTOR,
    STRONG_TREND_REGIME,
)

N_DAYS = 20
HOLD = 5
PRICE = 20.0          # thousand-VND scale → 20,000 VND after _prepare
SCORES = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.10, "DDD": 0.45}
NEUTRAL = 2           # a non-special regime


def _panel(regimes: dict[str, int] | None = None) -> pd.DataFrame:
    """4-ticker constant-price panel. `regimes` maps ticker→constant
    market_regime; omit a ticker to default it to NEUTRAL. Pass None to build a
    panel with NO market_regime column at all (flag-OFF / legacy shape)."""
    days = pd.bdate_range("2024-01-02", periods=N_DAYS).date
    frames = []
    for tk, score in SCORES.items():
        cols = {
            "ticker": tk, "date": days,
            "open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE,
            "volume": 10_000_000, "feat": score,
        }
        if regimes is not None:
            cols["market_regime"] = int(regimes.get(tk, NEUTRAL))
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def _oracle(X: np.ndarray) -> np.ndarray:
    return X[:, -1, 0].astype(np.float64)   # p_up = the feature value


def _engine(use_regime_sizing: bool = False, **overrides) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=2, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        use_regime_sizing=use_regime_sizing, **overrides,
    )
    return WalkForwardEngine(cfg, _oracle)


def _first_buy_notionals(res) -> dict[str, float]:
    """Per-ticker bought notional on the first day any buy fills."""
    fills = pd.DataFrame(res.fills)
    buys = fills[fills["side"] == "buy"]
    first_day = buys["date"].min()
    day1 = buys[buys["date"] == first_day]
    out: dict[str, float] = {}
    for r in day1.itertuples():
        out[r.ticker] = out.get(r.ticker, 0.0) + r.qty * r.price
    return out


# ── NO_TRADE ──────────────────────────────────────────────────────────────────

def test_no_trade_regime_ticker_not_bought() -> None:
    res = _engine(use_regime_sizing=True).run(_panel({"AAA": 0}))   # 0 = Freeze
    bought = set(pd.DataFrame(res.fills).query("side == 'buy'")["ticker"])
    assert "AAA" not in bought
    assert "BBB" in bought


def test_no_trade_regime_does_not_affect_others() -> None:
    res = _engine(use_regime_sizing=True).run(_panel({"AAA": 7}))   # 7 = Liquidity Sweep
    notion = _first_buy_notionals(res)
    assert "AAA" not in notion
    # BBB still gets its NORMAL per_name = budget/2 ≈ 100M (denominator unchanged).
    assert notion["BBB"] == pytest.approx(100_000_000, rel=0.05)


def test_no_trade_freed_capital_stays_cash() -> None:
    # AAA NO_TRADE: BBB must get only its own per_name (budget/2 ≈ 100M), NOT the
    # redistributed full budget (≈ 200M). This is the Design-2 (cash-preserving)
    # behaviour the plan's rationale requires.
    res = _engine(use_regime_sizing=True).run(_panel({"AAA": 0}))
    notion = _first_buy_notionals(res)
    assert notion["BBB"] == pytest.approx(100_000_000, rel=0.05)
    assert notion["BBB"] < 150_000_000


# ── PENALTY ─────────────────────────────────────────────────────────────────

def test_penalty_regime_halves_notional() -> None:
    res = _engine(use_regime_sizing=True).run(_panel({"AAA": 1}))   # 1 = Squeeze
    notion = _first_buy_notionals(res)
    # AAA (penalty) ≈ 0.5 × BBB (neutral) — a real, scale-invariant cut, NOT the
    # inert 0.10×NAV cap which would leave AAA == BBB.
    assert notion["AAA"] == pytest.approx(0.5 * notion["BBB"], rel=0.02)
    assert notion["AAA"] < 0.6 * notion["BBB"]


def test_penalty_factor_matches_serve_ratio() -> None:
    # Anti-drift: the tranche multiplier is exactly the serve-path cap shrink ratio.
    assert REGIME_PENALTY_FACTOR == REGIME_PENALTY_CAP / serve_sizing.DEFAULT_NAV_CAP
    assert REGIME_PENALTY_FACTOR == 0.5


# ── NEUTRAL / STRONG_TREND / flag-OFF (no-ops) ────────────────────────────────

def test_strong_trend_regime_unaffected() -> None:
    on = _engine(use_regime_sizing=True).run(_panel({"AAA": STRONG_TREND_REGIME}))
    off = _engine(use_regime_sizing=False).run(_panel({"AAA": STRONG_TREND_REGIME}))
    assert _first_buy_notionals(on)["AAA"] == pytest.approx(
        _first_buy_notionals(off)["AAA"], rel=1e-9)


def test_flag_off_ignores_no_trade_regime() -> None:
    res = _engine(use_regime_sizing=False).run(_panel({"AAA": 0}))   # Freeze, flag OFF
    bought = set(pd.DataFrame(res.fills).query("side == 'buy'")["ticker"])
    assert "AAA" in bought


def test_flag_off_panel_without_regime_column() -> None:
    # Legacy panel shape (no market_regime column) must run unaffected.
    res = _engine(use_regime_sizing=False).run(_panel(regimes=None))
    assert len(res.equity_curve) == N_DAYS


# ── Parity ──────────────────────────────────────────────────────────────────

def test_parity_constants_identical() -> None:
    # The serve path and the shared module expose the SAME objects (anti-drift).
    assert serve_sizing.NO_TRADE_REGIMES is NO_TRADE_REGIMES
    assert serve_sizing.PENALTY_REGIMES is PENALTY_REGIMES
    assert serve_sizing.REGIME_PENALTY_CAP == REGIME_PENALTY_CAP
    assert serve_sizing.STRONG_TREND_REGIME == STRONG_TREND_REGIME
