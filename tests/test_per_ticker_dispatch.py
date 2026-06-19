"""Unit tests for `_send_per_ticker_reports` in src/utils/telegram_bot.py.

The per-ticker sender replaces the combined-string + splitter reply path for
/suggest_buy: it sends ONE message per ticker (the institutional BUY card via
`AlerterBot._build_message`), editing the loading message into a 1-line header
first. On a Telegram `BadRequest` parse error each send degrades to
`html.escape`d text rather than leaking raw HTML tags.

These tests drive the coroutine with `asyncio.run` so they need no
`pytest-asyncio` plugin (none is pinned in requirements.txt). The
oversight-mirror branch is avoided by leaving ADMIN_CHAT_ID unset (the default
in a bare test env), so `update.get_bot()` is never invoked.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from telegram.error import BadRequest

from src.utils.telegram_bot import _send_per_ticker_reports


def _signal(ticker: str) -> dict:
    """Minimal signal_data dict accepted by AlerterBot._build_message."""
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
        "article_urls": ["https://vnexpress.net/a"],
    }


def _make_update_and_wait():
    """Return (update, wait_msg) with async reply_text / edit_text mocks."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    wait_msg = MagicMock()
    wait_msg.edit_text = AsyncMock()
    return update, wait_msg


def test_send_per_ticker_reports_calls_reply_n_times():
    update, wait_msg = _make_update_and_wait()
    signals = [_signal("VCB"), _signal("BID"), _signal("VHM")]

    asyncio.run(_send_per_ticker_reports(update, wait_msg, signals))

    # 1 header edit + N per-ticker replies.
    wait_msg.edit_text.assert_called_once()
    assert update.message.reply_text.call_count == len(signals)
    # Each reply is one ticker's card (HTML).
    for call in update.message.reply_text.call_args_list:
        sent = call.args[0]
        assert "KHUYẾN NGHỊ MUA" in sent


def test_send_per_ticker_reports_empty_list():
    update, wait_msg = _make_update_and_wait()

    asyncio.run(_send_per_ticker_reports(update, wait_msg, []))

    # Empty → edit the loading message once, never post per-ticker replies.
    wait_msg.edit_text.assert_called_once()
    update.message.reply_text.assert_not_called()


def test_send_per_ticker_reports_badrequest_fallback():
    update, wait_msg = _make_update_and_wait()
    # First reply raises BadRequest (parse error); the retry must be escaped.
    update.message.reply_text = AsyncMock(side_effect=[BadRequest("can't parse"), None])

    asyncio.run(_send_per_ticker_reports(update, wait_msg, [_signal("VCB")]))

    # Two calls: the failed HTML send + the escaped retry.
    assert update.message.reply_text.call_count == 2
    retry_text = update.message.reply_text.call_args_list[1].args[0]
    # The escaped retry must NOT contain raw opening tags.
    assert "<b>" not in retry_text
    assert "&lt;b&gt;" in retry_text


def test_send_per_ticker_reports_skips_malformed_signal():
    update, wait_msg = _make_update_and_wait()
    # A signal whose card build blows up is skipped, not fatal to the batch.
    bad = MagicMock()
    bad.get.side_effect = RuntimeError("boom")
    good = _signal("BID")

    asyncio.run(_send_per_ticker_reports(update, wait_msg, [bad, good]))

    # Only the good ticker is sent.
    assert update.message.reply_text.call_count == 1
    assert "KHUYẾN NGHỊ MUA" in update.message.reply_text.call_args_list[0].args[0]
