"""Tests for /suggest_buy5 + /suggest_buy20 signal-ledger paper-tracking.

`_suggest_buy_dispatch` calls `daily_inference(broadcast=False, persist=False)`.
Both gates are correct (persist=False protects the shared automated
portfolio/paperlog book), but they also mean the cron-path
`signal_ledger.record_dispatch` (gated on `broadcast`) never fires — so these
manual picks were previously untracked entirely. The dispatch now paper-tracks
its delivered picks into `dispatched_signals`, gated on a config kill-switch,
never-raise so a ledger failure can never block card delivery.

These tests drive the async coroutine with `asyncio.run` (no pytest-asyncio
plugin is pinned) and monkeypatch the heavy dependencies (rate-limit, audit log,
daily_inference, per-ticker delivery, record_dispatch) so only the tracking
wiring is exercised. Convention mirrors tests/test_per_ticker_dispatch.py and
tests/test_user2_mirror.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import main
import src.utils.telegram_bot as bot_mod
from config.settings import CONFIG
from src.trading import signal_ledger
from src.utils.telegram_bot import _suggest_buy_dispatch


def _signals() -> list[dict]:
    return [
        {"ticker": "VCB", "suggested_weight": 0.10, "action": "MUA"},
        {"ticker": "BID", "suggested_weight": 0.08, "action": "MUA"},
    ]


def _make_update():
    """update with an async reply_text that returns a wait_msg mock."""
    update = MagicMock()
    update.message = MagicMock()
    wait_msg = MagicMock()
    wait_msg.edit_text = AsyncMock()
    update.message.reply_text = AsyncMock(return_value=wait_msg)
    return update, wait_msg


def _install_common_stubs(monkeypatch, signals):
    """Stub the shared machinery; return the record_dispatch call capture list."""
    monkeypatch.setattr(bot_mod, "_check_and_record_rate_limit", lambda user_id, command: None)
    monkeypatch.setattr(bot_mod, "_audit_log_async", AsyncMock())
    monkeypatch.setattr(bot_mod, "_send_per_ticker_reports", AsyncMock())

    def _fake_daily_inference(*, broadcast, horizon, persist):
        # Mirror the real signature used at the call site (kwargs).
        return ("<b>report</b>", signals)

    monkeypatch.setattr(main, "daily_inference", _fake_daily_inference)

    calls: list[dict] = []

    def _capture(sigs, strategy, horizon, *args, **kwargs):
        calls.append({"signals": sigs, "strategy": strategy, "horizon": horizon})
        return len(sigs)

    monkeypatch.setattr(signal_ledger, "record_dispatch", _capture)
    return calls


def test_suggest_buy20_tracks_with_hold_days_30(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "suggest_buy_ledger_tracking_enabled", True)
    signals = _signals()
    calls = _install_common_stubs(monkeypatch, signals)
    update, _ = _make_update()

    asyncio.run(_suggest_buy_dispatch(update, horizon=20))

    assert len(calls) == 1
    call = calls[0]
    assert call["horizon"] == 20
    assert call["strategy"] == {"mode": "tranche", "hold_days": 30}
    # The delivered picks (not zeroed placeholders) are what gets tracked.
    assert call["signals"] is signals
    # Delivery still happened.
    bot_mod._send_per_ticker_reports.assert_awaited_once()


def test_suggest_buy5_tracks_with_hold_days_5(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "suggest_buy_ledger_tracking_enabled", True)
    signals = _signals()
    calls = _install_common_stubs(monkeypatch, signals)
    update, _ = _make_update()

    asyncio.run(_suggest_buy_dispatch(update, horizon=5))

    assert len(calls) == 1
    call = calls[0]
    assert call["horizon"] == 5
    # T5 uses a synthetic 5-session window, NOT the 30-session tranche window.
    assert call["strategy"] == {"mode": "tranche", "hold_days": 5}
    bot_mod._send_per_ticker_reports.assert_awaited_once()


def test_disabled_config_skips_tracking(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "suggest_buy_ledger_tracking_enabled", False)
    signals = _signals()
    calls = _install_common_stubs(monkeypatch, signals)
    update, _ = _make_update()

    asyncio.run(_suggest_buy_dispatch(update, horizon=20))

    # Kill-switch off → no ledger write, but cards still delivered.
    assert calls == []
    bot_mod._send_per_ticker_reports.assert_awaited_once()


def test_record_dispatch_exception_does_not_break_delivery(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "suggest_buy_ledger_tracking_enabled", True)
    signals = _signals()
    _install_common_stubs(monkeypatch, signals)
    update, _ = _make_update()

    def _boom(*args, **kwargs):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(signal_ledger, "record_dispatch", _boom)

    # Must not raise, and delivery must still happen.
    asyncio.run(_suggest_buy_dispatch(update, horizon=20))

    bot_mod._send_per_ticker_reports.assert_awaited_once()


def test_no_tracking_when_no_signals(monkeypatch):
    """Empty signal list → early return before tracking (no ledger write)."""
    monkeypatch.setattr(CONFIG.trading, "suggest_buy_ledger_tracking_enabled", True)
    calls = _install_common_stubs(monkeypatch, [])
    # daily_inference returns no report_html + empty signals → generic no-signal reply.
    monkeypatch.setattr(main, "daily_inference", lambda **kw: ("", []))
    update, _ = _make_update()

    asyncio.run(_suggest_buy_dispatch(update, horizon=5))

    assert calls == []
    bot_mod._send_per_ticker_reports.assert_not_awaited()
