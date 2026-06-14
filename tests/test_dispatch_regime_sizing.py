"""Serve-path regime-conditional sizing in main._dispatch_signals.

Mirrors the backtest (walk_forward._tranche_day) on the LIVE dispatch path:
  • NO_TRADE regimes {0,7} → name skipped (not dispatched); its cohort weight
    stays cash (the n_picks denominator is frozen before the loop, so survivors
    are NOT inflated).
  • PENALTY regimes {1,6} → tranche weight × REGIME_PENALTY_FACTOR (0.5).
  • Gated by CONFIG.trading.regime_sizing_enabled (default True; settings.json
    kill-switch). Event overrides keep precedence over regime.

_dispatch_signals returns the dispatched list when broadcast=False (no bot
call), so we can assert on it directly.
"""
from __future__ import annotations

import pytest

import main
from src.trading.regime_policy import REGIME_PENALTY_FACTOR

TRANCHE_STRATEGY = {"mode": "tranche", "hold_days": 30}


def _predictions(tickers: list[str]) -> dict[str, dict]:
    # P(UP)=0.70 for every ticker; only the regime path matters here.
    return {"5d": {tk: [0.1, 0.2, 0.7] for tk in tickers}}


def _prices(tickers: list[str]) -> dict[str, float]:
    return {tk: 25_000.0 for tk in tickers}


def _dispatch(tickers: list[str], *, strategy=TRANCHE_STRATEGY, overrides=None) -> list[dict]:
    return main._dispatch_signals(
        top_buy_signals=tickers,
        all_sentiments={},
        stacking_predictions=_predictions(tickers),
        live_exec_prices=_prices(tickers),
        event_overrides=overrides,
        top_pos_features="",
        top_neg_features="",
        horizon=20,
        broadcast=False,            # → returns list, never touches the bot
        bot=None,
        strategy=strategy,
    )


@pytest.fixture
def regime(monkeypatch):
    """Set per-ticker regimes + the flag, auto-restored after each test."""
    def _set(mapping: dict[str, int], *, enabled: bool = True):
        monkeypatch.setattr(main, "_LATEST_REGIME_BY_TICKER", dict(mapping))
        monkeypatch.setattr(main.CONFIG.trading, "regime_sizing_enabled", enabled)
    return _set


def test_no_trade_regime_not_dispatched(regime) -> None:
    regime({"AAA": 0, "BBB": 2})                       # AAA = Freeze
    out = _dispatch(["AAA", "BBB"])
    tickers = [s["ticker"] for s in out]
    assert "AAA" not in tickers
    assert tickers == ["BBB"]
    # Cash-preserving: BBB keeps its frozen-denominator weight 1/(30*2), NOT 1/(30*1).
    assert out[0]["suggested_weight"] == pytest.approx(1.0 / (30 * 2), rel=1e-6)


def test_penalty_regime_halves_weight(regime) -> None:
    regime({"AAA": 1, "BBB": 2})                       # AAA = Squeeze
    out = _dispatch(["AAA", "BBB"])
    by_tk = {s["ticker"]: s["suggested_weight"] for s in out}
    full = 1.0 / (30 * 2)
    assert by_tk["AAA"] == pytest.approx(full * REGIME_PENALTY_FACTOR, rel=1e-6)
    assert by_tk["BBB"] == pytest.approx(full, rel=1e-6)


def test_flag_off_dispatches_no_trade_name(regime) -> None:
    regime({"AAA": 0, "BBB": 2}, enabled=False)        # kill-switch off
    out = _dispatch(["AAA", "BBB"])
    tickers = {s["ticker"] for s in out}
    assert tickers == {"AAA", "BBB"}                   # byte-for-byte legacy behaviour


def test_event_override_wins_over_no_trade(regime) -> None:
    regime({"AAA": 0})                                 # NO_TRADE, but overridden
    out = _dispatch(
        ["AAA"],
        overrides={"AAA": {"weight": 0.05, "status": "OVERRIDE", "ly_do": "test"}},
    )
    assert len(out) == 1                               # override fires before regime skip
    assert out[0]["suggested_weight"] == pytest.approx(0.05)


def test_none_regime_is_noop(regime) -> None:
    regime({})                                         # AAA absent → regime None
    out = _dispatch(["AAA"])
    assert len(out) == 1
    assert out[0]["suggested_weight"] == pytest.approx(1.0 / (30 * 1), rel=1e-6)
