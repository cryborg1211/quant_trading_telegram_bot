"""Tests for the ADMIN -> User2 auto-forward mirror (2026-07-14, user-requested).

`_send_or_reply_chunks` / `_send_per_ticker_reports` gained an opt-in
`mirror_to_user2` flag: when True AND the caller is ADMIN (role=="admin") AND
USER_CHAT_ID is configured, the same content sent to Admin is also broadcast
to User2. Wired into suggest_buy5/20, suggest_sell, audit_weekly/monthly,
audit_accuracy only — NOT /guard, /verify, /rebalance (never asked for).

Mirrors the existing user->admin oversight test conventions: drive the
coroutine with asyncio.run, mock update.get_bot()/message, monkeypatch the
module-level role/chat-id state directly rather than faking Telegram auth.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import src.utils.telegram_bot as bot_mod
from src.utils.telegram_bot import _send_or_reply_chunks, _send_per_ticker_reports


def _signal(ticker: str) -> dict:
    return {
        "action": "MUA",
        "ticker": ticker,
        "price": "25,000 VND",
        "horizon_label": "T+20",
        "suggested_weight": 0.10,
        "prob_up": 66.0,
        "prob_side": 9.0,
        "prob_down": 25.0,
        "conclusion": "Dòng tin tích cực.",
        "article_urls": [],
    }


def _make_update_and_wait():
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    bot = MagicMock()
    bot.send_message = AsyncMock()
    update.get_bot = MagicMock(return_value=bot)
    wait_msg = MagicMock()
    wait_msg.edit_text = AsyncMock()
    return update, wait_msg, bot


def test_chunks_mirrors_to_user2_when_admin_and_flag_set(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(
        _send_or_reply_chunks(update, wait_msg, ["report body"], mirror_to_user2=True)
    )

    mirror_calls = [c for c in bot.send_message.call_args_list if c.kwargs.get("chat_id") == "222"]
    # 1 header + 1 body chunk forwarded to User2.
    assert len(mirror_calls) == 2
    assert "report body" in mirror_calls[1].kwargs["text"]


def test_chunks_no_mirror_when_flag_false(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(_send_or_reply_chunks(update, wait_msg, ["report body"]))

    assert bot.send_message.call_count == 0


def test_chunks_no_mirror_when_caller_is_user2_itself(monkeypatch):
    """No self-forward: User2 calling their own audit doesn't re-send to User2."""
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "user")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(
        _send_or_reply_chunks(update, wait_msg, ["report body"], mirror_to_user2=True)
    )

    mirror_calls = [c for c in bot.send_message.call_args_list if c.kwargs.get("chat_id") == "222"]
    assert mirror_calls == []


def test_chunks_no_mirror_when_user_chat_id_unset(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(
        _send_or_reply_chunks(update, wait_msg, ["report body"], mirror_to_user2=True)
    )

    assert bot.send_message.call_count == 0


def test_chunks_mirror_failure_does_not_break_admin_reply(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()
    bot.send_message = AsyncMock(side_effect=RuntimeError("network down"))

    asyncio.run(
        _send_or_reply_chunks(update, wait_msg, ["report body"], mirror_to_user2=True)
    )

    # Primary delivery still happened (single-chunk -> wait_msg.edit_text).
    wait_msg.edit_text.assert_called_once()


def test_per_ticker_mirrors_to_user2_when_admin_and_flag_set(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(
        _send_per_ticker_reports(
            update, wait_msg, [_signal("VCB"), _signal("BID")], mirror_to_user2=True
        )
    )

    mirror_calls = [c for c in bot.send_message.call_args_list if c.kwargs.get("chat_id") == "222"]
    # 1 header + 2 per-ticker cards forwarded to User2.
    assert len(mirror_calls) == 3


def test_per_ticker_no_mirror_by_default(monkeypatch):
    monkeypatch.setattr(bot_mod, "USER_CHAT_ID", "222")
    monkeypatch.setattr(bot_mod, "_role_for", lambda update: "admin")
    update, wait_msg, bot = _make_update_and_wait()

    asyncio.run(_send_per_ticker_reports(update, wait_msg, [_signal("VCB")]))

    assert bot.send_message.call_count == 0
