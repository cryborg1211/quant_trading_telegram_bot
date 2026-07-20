"""Orchestration tests for main.notify_position_report (EOD dual-horizon report).

Mirrors tests/test_portfolio_guard.py's "Part E" pattern: monkeypatch the
signal-ledger read/eval surface + TelegramBot so only the never-raise wrapper
logic is exercised (config short-circuit, nothing-to-report gate, send path,
exception safety).
"""
from __future__ import annotations

from datetime import date

import main


def test_notify_disabled_no_db_read(monkeypatch) -> None:
    reads: list = []
    monkeypatch.setattr(main.CONFIG.trading, "eod_position_report_enabled", False)
    monkeypatch.setattr(main.signal_ledger, "list_open",
                        lambda *a, **k: reads.append(1) or [])
    monkeypatch.setattr(main.signal_ledger, "list_closed_since",
                        lambda *a, **k: reads.append(1) or [])
    assert main.notify_position_report() == 0
    assert reads == []                      # short-circuit before any DB read


def test_notify_nothing_to_report_returns_zero(monkeypatch) -> None:
    constructed: list = []

    class _FakeBot:
        def __init__(self) -> None:
            constructed.append(1)

        def send_text_alert(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(main.CONFIG.trading, "eod_position_report_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_open", lambda *a, **k: [])
    monkeypatch.setattr(main.signal_ledger, "list_closed_since", lambda *a, **k: [])
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)
    assert main.notify_position_report() == 0
    assert constructed == []                 # never constructed → never sent


def test_notify_sends_combined_report(monkeypatch) -> None:
    sent: list = []

    class _FakeBot:
        def send_text_alert(self, msg, label="alert") -> None:
            sent.append((msg, label))

    open_rows = [{"ticker": "HPG", "dispatch_date": date(2026, 6, 1),
                  "horizon": 20, "hold_days": 30, "weight": 0.01,
                  "sessions_elapsed": 5, "sessions_remaining": 25}]
    closed_rows = [{"ticker": "FPT", "dispatch_date": date(2026, 6, 1),
                    "horizon": 5, "hold_days": 5, "weight": 0.0}]

    monkeypatch.setattr(main.CONFIG.trading, "eod_position_report_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_open", lambda *a, **k: open_rows)
    monkeypatch.setattr(main.signal_ledger, "list_closed_since", lambda *a, **k: closed_rows)
    monkeypatch.setattr(main.signal_ledger, "evaluate_signal_pnl",
                        lambda t, d, h, today=None: {"ticker": t, "pct": 3.0, "matured": h <= 5})
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)

    assert main.notify_position_report() == 1
    assert len(sent) == 1
    msg, label = sent[0]
    assert label == "position_report"
    assert "HPG" in msg and "FPT" in msg


def test_notify_never_raises_on_ledger_exception(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(main.CONFIG.trading, "eod_position_report_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_open", _boom)
    assert main.notify_position_report() == 0     # exception swallowed → 0
