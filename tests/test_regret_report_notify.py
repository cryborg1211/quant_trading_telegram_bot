"""Orchestration tests for main.notify_regret_report (EOD cancelled-signal report).

Mirrors tests/test_position_report_notify.py: monkeypatch the signal-ledger
read/eval surface + TelegramBot so only the never-raise wrapper logic is
exercised (config short-circuit, nothing-to-report gate, send path, exception
safety).
"""
from __future__ import annotations

from datetime import date

import main


def test_notify_disabled_no_db_read(monkeypatch) -> None:
    reads: list = []
    monkeypatch.setattr(main.CONFIG.trading, "cancelled_signal_tracking_enabled", False)
    monkeypatch.setattr(main.signal_ledger, "list_cancelled_since",
                        lambda *a, **k: reads.append(1) or [])
    assert main.notify_regret_report() == 0
    assert reads == []                      # short-circuit before any DB read


def test_notify_nothing_to_report_returns_zero(monkeypatch) -> None:
    constructed: list = []

    class _FakeBot:
        def __init__(self) -> None:
            constructed.append(1)

        def send_text_alert(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(main.CONFIG.trading, "cancelled_signal_tracking_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_cancelled_since", lambda *a, **k: [])
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)
    assert main.notify_regret_report() == 0
    assert constructed == []                 # never constructed → never sent


def test_notify_sends_combined_report(monkeypatch) -> None:
    sent: list = []

    class _FakeBot:
        def send_text_alert(self, msg, label="alert") -> None:
            sent.append((msg, label))

    cancelled_rows = [
        {"ticker": "HPG", "screen_date": date(2026, 7, 3), "horizon": 20,
         "hold_days": 30, "p_up": 0.30, "reason": "Cửa tăng thấp",
         "sessions_elapsed": 4, "sessions_remaining": 26},
        {"ticker": "FPT", "screen_date": date(2026, 7, 1), "horizon": 5,
         "hold_days": 5, "p_up": 0.25, "reason": "Trọng tài từ chối",
         "sessions_elapsed": 5, "sessions_remaining": 0},
    ]
    monkeypatch.setattr(main.CONFIG.trading, "cancelled_signal_tracking_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_cancelled_since",
                        lambda *a, **k: cancelled_rows)
    # matured keyed off hold_days (30 → still tracking, 5 → matured).
    monkeypatch.setattr(main.signal_ledger, "evaluate_regret_pnl",
                        lambda t, sd, h, today=None: {"ticker": t, "pct": 3.0,
                                                      "matured": h <= 5,
                                                      "gap_flag": False})
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)

    assert main.notify_regret_report() == 1
    assert len(sent) == 1
    msg, label = sent[0]
    assert label == "regret_report"
    assert "HPG" in msg and "FPT" in msg
    assert "KHÔNG phải khuyến nghị" in msg     # non-recommendation disclaimer present


def test_notify_never_raises_on_ledger_exception(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(main.CONFIG.trading, "cancelled_signal_tracking_enabled", True)
    monkeypatch.setattr(main.signal_ledger, "list_cancelled_since", _boom)
    assert main.notify_regret_report() == 0     # exception swallowed → 0
