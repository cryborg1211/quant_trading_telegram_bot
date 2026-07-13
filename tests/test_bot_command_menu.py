"""Command-menu registration: global vs Admin-scoped lists.

No real Telegram — `_set_bot_commands` is driven with a mocked Application.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils import telegram_bot as tb


def test_msg_user2_is_admin_scoped_not_global() -> None:
    global_names = [c.command for c in tb._BOT_COMMANDS]
    admin_names = [c.command for c in tb._ADMIN_COMMANDS]
    assert "msg_user2" in admin_names
    assert "msg_user2" not in global_names  # User2's menu must not advertise it


def test_set_bot_commands_pushes_admin_scope(monkeypatch) -> None:
    monkeypatch.setattr(tb, "ADMIN_CHAT_ID", "111")
    app = MagicMock()
    app.bot.set_my_commands = AsyncMock()

    asyncio.run(tb._set_bot_commands(app))

    assert app.bot.set_my_commands.await_count == 2
    scoped = app.bot.set_my_commands.await_args_list[1]
    names = [c.command for c in scoped.args[0]]
    assert "msg_user2" in names
    assert scoped.kwargs["scope"].chat_id == 111


def test_set_bot_commands_global_only_without_admin_id(monkeypatch) -> None:
    monkeypatch.setattr(tb, "ADMIN_CHAT_ID", "")
    app = MagicMock()
    app.bot.set_my_commands = AsyncMock()

    asyncio.run(tb._set_bot_commands(app))

    assert app.bot.set_my_commands.await_count == 1


def test_scoped_push_failure_degrades(monkeypatch) -> None:
    monkeypatch.setattr(tb, "ADMIN_CHAT_ID", "111")
    app = MagicMock()
    calls = {"n": 0}

    async def _set(*args, **kwargs):
        calls["n"] += 1
        if "scope" in kwargs:
            raise RuntimeError("telegram api down")

    app.bot.set_my_commands = _set
    asyncio.run(tb._set_bot_commands(app))  # must not raise
    assert calls["n"] == 2
