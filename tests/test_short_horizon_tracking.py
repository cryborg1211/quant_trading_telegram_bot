"""T+5 tracking dispatch wiring in main.run_trade_execution.

Every broadcast T+20 dispatch now books a paired, independently-tracked T+5
paper position for the same tickers (EOD dual-horizon position-report plan,
Phase 2). This asserts:
  • exactly 2 `signal_ledger.record_dispatch` calls on broadcast=True — one
    T+20 with the real dispatched dicts (weight present), one T+5 with
    ticker-only dicts (no weight key, defaults to 0.0);
  • 0 ledger calls on broadcast=False (interactive /suggest_buy previews are
    never committed positions).

Mirrors tests/test_dispatch_regime_sizing.py's monkeypatch style — the heavy
serve stack (_get_live_exec_prices, _dispatch_signals, TelegramBot,
_load_v3_bot, report builders) is stubbed so only the ledger wiring is exercised.
"""
from __future__ import annotations

import pytest

import main


class _FakeBot:
    def send_text_alert(self, *a, **k) -> None:  # noqa: D401
        pass

    def send_signal_alert(self, *a, **k) -> None:
        pass


class _FakeArtifact:
    strategy = {"mode": "tranche", "hold_days": 30}


_DISPATCHED = [
    {"ticker": "AAA", "suggested_weight": 0.01},
    {"ticker": "BBB", "suggested_weight": 0.01},
]


@pytest.fixture
def wired(monkeypatch):
    """Stub the serve stack; return the list capturing record_dispatch calls."""
    calls: list[dict] = []

    def _capture(signals, strategy, horizon=None, db_path=None, today=None):
        calls.append({"signals": signals, "strategy": strategy, "horizon": horizon})
        return len(signals)

    monkeypatch.setattr(main.signal_ledger, "record_dispatch", _capture)
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)
    monkeypatch.setattr(main, "_get_live_exec_prices",
                        lambda df, tickers: {"AAA": 25_000.0, "BBB": 25_000.0})
    monkeypatch.setattr(main, "_build_feature_explanation", lambda *a, **k: ("", ""))
    monkeypatch.setattr(main, "_load_v3_bot", lambda h=5: _FakeArtifact())
    monkeypatch.setattr(main, "_dispatch_signals", lambda **k: list(_DISPATCHED))
    monkeypatch.setattr(main, "build_regime_pulse", lambda *a, **k: "")
    monkeypatch.setattr(main, "_build_combined_report", lambda *a, **k: "")
    return calls


def _run(broadcast: bool):
    return main.run_trade_execution(
        top_buy_signals=["AAA", "BBB"],
        final_decisions={},
        all_sentiments={},
        stacking_predictions={},
        latest_df=None,
        xgb_model_5d=None,
        selected_features_5d=[],
        horizon=20,
        broadcast=broadcast,
        persist=False,
    )


def test_broadcast_books_both_horizons(wired) -> None:
    _run(broadcast=True)
    assert len(wired) == 2

    t20, t5 = wired[0], wired[1]
    # First call = the real T+20 tranche book (weight-carrying dicts).
    assert t20["horizon"] == 20
    assert all("suggested_weight" in s for s in t20["signals"])
    assert [s["ticker"] for s in t20["signals"]] == ["AAA", "BBB"]

    # Second call = the T+5 tracking rows (ticker-only, no weight, hold_days=5).
    assert t5["horizon"] == main.SHORT_HORIZON
    assert t5["strategy"] == {"mode": "tranche", "hold_days": main.SHORT_HORIZON}
    assert [s["ticker"] for s in t5["signals"]] == ["AAA", "BBB"]
    assert all("suggested_weight" not in s for s in t5["signals"])


def test_no_broadcast_books_nothing(wired) -> None:
    _run(broadcast=False)
    assert wired == []
