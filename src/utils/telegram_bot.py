"""
Quant V6 — Interactive Telegram Bot (long-running service).

This is the 2-way command bot. It is intentionally separate from
`src/utils/telegram_alerter.py`, which remains a one-shot push alerter
called from the daily inference cron.

Run as a long-lived process (server / supervisor entry point):
    python run_bot.py

Or for local development:
    python -m src.utils.telegram_bot

Required env vars (load from `.env`, see `.env.example`):
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather

Phases:
    Phase 1: framework + /help                   (DONE)
    Phase 2: /suggest_buy                         (DONE — non-blocking via to_thread)
    Phase 3: /add, /suggest_sell, /news           (DONE)
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import subprocess
import sys
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from telegram import BotCommand, BotCommandScopeChat, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ApplicationHandlerStop,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

# Per-ticker BUY card builder. Aliased to avoid any confusion with PTB's own
# internals; `telegram_alerter` does NOT import from this module, so there is
# no circular-import risk.
from src.utils.telegram_alerter import TelegramBot as AlerterBot

load_dotenv()

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Telegram per-message hard limit is 4096 chars. We leave a little head-room
# for HTML tag overhead inside chunks.
TELEGRAM_MAX_MESSAGE_CHARS = 4000

# Permissive ticker regex — VN tickers are 2-5 uppercase letters; we allow
# A-Z and digits (some funds use digits) up to length 6.
_TICKER_RE = re.compile(r"^[A-Z0-9]{2,6}$")

# ─── TD-30: per-(user, command) rate limiter ──────────────────────────────
# Applied only to the two Gemini-touching commands: /suggest_buy and /verify.
# Lightweight in-memory dict keyed by (user_id, command); cleared on bot
# restart (acceptable — restart effectively expires the cooldown anyway).
# `_time.monotonic()` is used so wall-clock changes / NTP slew don't bypass.
_RATE_LIMIT_WINDOW_SECONDS: float = 30.0
_rate_limit_tracker: dict[tuple[str, str], float] = {}
_rate_limit_lock = threading.Lock()


def _check_and_record_rate_limit(user_id: str | None, command: str) -> float | None:
    """Atomic check-and-set against the in-memory cooldown table.

    Returns:
        None  → call is allowed (and `now` has been recorded for next time).
        float → seconds remaining before this (user, command) pair can call again.

    Anonymous callers (user_id is None) bypass — anonymous-rejection happens
    upstream via `_extract_user_id`.
    """
    if not user_id:
        return None
    key = (user_id, command)
    now = _time.monotonic()
    with _rate_limit_lock:
        last = _rate_limit_tracker.get(key, 0.0)
        elapsed = now - last
        if elapsed < _RATE_LIMIT_WINDOW_SECONDS:
            return _RATE_LIMIT_WINDOW_SECONDS - elapsed
        _rate_limit_tracker[key] = now
        return None

WAIT_MESSAGE_BUY = (
    "⏳ Đang chạy mô hình Stacking Quant &amp; LLM Sentiment... "
    "Vui lòng đợi khoảng 1-2 phút!"
)

EMPTY_BUY_RESULT_MESSAGE = (
    "📭 <b>Không có tín hiệu MUA nào sau bộ lọc Sentiment hôm nay.</b>\n"
    "<i>Mô hình đã chạy xong nhưng không có ticker nào vượt qua điều kiện "
    "(thanh khoản + Quant Top 6 + Sentiment).</i>"
)

EMPTY_PORTFOLIO_MESSAGE = (
    "<i>Danh mục đang trống. Hãy dùng lệnh /add để thêm cổ phiếu.</i>"
)

# HTML body. Telegram supports a strict subset: <b>, <i>, <u>, <s>, <a>,
# <code>, <pre>, <blockquote>, <tg-spoiler>. The ampersand in "Quant & Sentiment"
# is escaped as &amp;. Square brackets do NOT need escaping (only <, >, & do).
HELP_TEXT = (
    "🤖 <b>TRỢ LÝ ĐẦU TƯ — DANH SÁCH LỆNH</b>\n"
    "\n"
    "📊 <b>/suggest</b> — Danh sách <b>ứng viên</b> MUA (T+20). Bạn tự quyết vào lệnh.\n"
    "🔍 <b>/verify</b> <i>[Mã]</i> — Soi nhanh 1 cổ phiếu, mô hình T+5 &amp; T+20 "
    "(VD: <code>/verify HPG</code>).\n"
    "🎯 <b>/audit_accuracy</b> — <b>Bảng điểm</b>: mô hình đúng/sai bao nhiêu (nên xem trước khi tin).\n"
    "🛡️ <b>/guard</b> — Kiểm tra cảnh báo bảo vệ danh mục (cắt lỗ / trailing / pha thị trường).\n"
    "🔴 <b>/suggest_sell</b> — Đánh giá NÊN BÁN hay GIỮ danh mục của bạn.\n"
    "📒 <b>/exits</b> — Vị thế tranche đang mở + số phiên còn lại.\n"
    "⚖️ <b>/rebalance</b> — Gợi ý cơ cấu lại danh mục hiện tại.\n"
    "➕ <b>/add</b> <i>[Mã] [Số lượng] [Giá]</i> — Thêm cổ phiếu vào danh mục "
    "(VD: <code>/add VNE 1000 32.5</code>).\n"
    "➖ <b>/remove</b> <i>[Mã]</i> — Xóa cổ phiếu khỏi danh mục "
    "(VD: <code>/remove VNE</code>).\n"
    "📅 <b>/audit_weekly</b> — Xem lại hiệu quả các quyết định 7 ngày qua.\n"
    "🗓️ <b>/audit_monthly</b> — Xem lại hiệu quả các quyết định 30 ngày qua.\n"
    "📰 <b>/news</b> — Tổng hợp tin tức thị trường mới nhất.\n"
    "💬 <b>/feedback</b> <i>[Nội dung]</i> — Gửi góp ý trực tiếp đến Admin.\n"
    "📨 <b>/msg_user2</b> <i>[Nội dung]</i> — Gửi thông báo đến User (chỉ Admin).\n"
    "ℹ️ <b>/help</b> — Hiển thị menu này.\n"
    "\n"
    "💡 Lưu ý: Để hỗ trợ và cải thiện hệ thống, các truy vấn của bạn "
    "có thể được Admin ghi nhận."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_split_block(block: str, max_len: int) -> list[str]:
    """Split a single oversized HTML block without cutting inside a tag.

    Strategy for each cut (applied left-to-right over the remainder):
    1. Walk backward from index `max_len` to find the best boundary that is
       NOT inside an unclosed ``<...>`` tag — prefer a newline, else a space.
    2. If no whitespace boundary exists at/under `max_len`, fall back to the
       raw `max_len` cut, but if that index lands inside an unclosed ``<``,
       back the cut up to just before that ``<`` so a tag is never severed.
    3. If even index 0 opens a tag that does not close within `max_len` (a
       single tag LONGER than `max_len`), emit that whole tag intact as one
       oversized chunk rather than slicing through it — invalid HTML is worse
       on every send path than one over-length message (the caller's
       ``BadRequest`` fallback escapes an over-length card if Telegram rejects
       it). Only when the tag never closes at all do we hard-cut at `max_len`
       to guarantee termination (matches the legacy hard-slice for tag-free
       input).

    Returns a list of chunks. Each chunk is ``<= max_len`` EXCEPT a lone tag
    longer than `max_len`, which is kept intact (see rule 3).
    """
    if len(block) <= max_len:
        return [block] if block else []

    chunks: list[str] = []
    rest = block
    while len(rest) > max_len:
        window = rest[:max_len]
        cut = _last_safe_boundary(window)
        if cut <= 0:
            # No safe whitespace boundary at/under max_len. Retreat out of any
            # tag the raw cut would sever.
            cut = _retreat_out_of_tag(window, max_len)
            if cut <= 0:
                # Index 0 opens a tag that does not close within max_len: emit
                # the whole tag intact (cut just AFTER its real '>' in `rest`),
                # or hard-cut at max_len if it never closes at all.
                close_gt = rest.find(">")
                cut = close_gt + 1 if close_gt != -1 else max_len
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n ")
    if rest:
        chunks.append(rest)
    return chunks


def _last_safe_boundary(window: str) -> int:
    """Index to cut `window` at: last newline (else space) outside any tag.

    Returns 0 when no whitespace boundary exists outside a tag (caller then
    falls back to a tag-retreated hard cut).
    """
    best_space = 0
    depth_open = False  # True while inside an unclosed '<...>'
    last_newline = 0
    for i, ch in enumerate(window):
        if ch == "<":
            depth_open = True
        elif ch == ">":
            depth_open = False
        elif not depth_open and ch in ("\n", " "):
            best_space = i
            if ch == "\n":
                last_newline = i
    # Prefer the last newline boundary; otherwise the last space boundary.
    # Cut AFTER the boundary char so the whitespace stays with the left chunk
    # (then stripped from the right remainder by the caller).
    boundary = last_newline if last_newline else best_space
    return boundary + 1 if boundary else 0


def _retreat_out_of_tag(window: str, idx: int) -> int:
    """If cutting `window` at `idx` lands inside an unclosed '<', back up.

    Returns the largest index <= `idx` that is not inside an open tag, or 0
    if even index 0 is inside the (single, oversized) tag — which cannot
    happen here since index 0 is the block start.
    """
    open_lt = window.rfind("<", 0, idx)
    if open_lt == -1:
        return idx
    close_gt = window.find(">", open_lt)
    if close_gt == -1 or close_gt >= idx:
        # The '<' opened before idx is still unclosed at idx → retreat to
        # just before that '<' so the partial tag moves to the next chunk.
        return open_lt
    return idx


def _split_html_report(report: str, max_len: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """Split a long combined HTML report into Telegram-safe chunks.

    Splits on the visual `══════` separator built by `_build_combined_report`
    in main.py so each chunk is one self-contained per-ticker block. Falls
    back to a TAG-SAFE splitter (`_safe_split_block`) only if a single block
    somehow exceeds the limit, so a chunk never ends inside a `<...>` tag.
    """
    if len(report) <= max_len:
        return [report] if report.strip() else []

    blocks = [b.strip() for b in report.split("══════════════════════════════") if b.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(block) > max_len:
            chunks.extend(_safe_split_block(block, max_len))
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks


def _chunk_lines(lines: list[str], max_len: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """Pack lines into chunks each below `max_len` chars (used for /news output)."""
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = (current + "\n" + line) if current else line
        if len(candidate) > max_len and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def _send_per_ticker_reports(
    update: Update,
    wait_msg: Any,
    signal_data_list: list[dict],
    *,
    disable_preview: bool = True,
    mirror_to_user2: bool = False,
) -> None:
    """Send ONE Telegram message per ticker (no combined-string bundling).

    Each card is built by `AlerterBot._build_message(sd)` — the exact same
    institutional BUY card the cron push path uses — so it is pure HTML and
    well under the 4096-char limit, sidestepping the splitter entirely.

    Mirrors the 1-way oversight behaviour of `_send_or_reply_chunks`: every
    card sent to a monitored "user" is also forwarded to ADMIN_CHAT_ID
    (best-effort). Each send degrades to `html.escape`d text on a `BadRequest`
    parse error rather than leaking raw tags.

    `mirror_to_user2` (2026-07-14): opt-in reverse mirror, ADMIN → User2, for
    the small set of commands the user asked to auto-broadcast (suggest_buy5,
    suggest_buy20). Only fires when the caller is ADMIN — never when User2 is
    the caller (no self-forward) and never for commands that didn't pass it.
    """
    if not signal_data_list:
        if wait_msg is not None:
            try:
                await wait_msg.edit_text(EMPTY_BUY_RESULT_MESSAGE, parse_mode=ParseMode.HTML)
            except BadRequest as exc:
                LOGGER.warning("empty-result edit_text failed (%s).", exc)
        return

    # Build cards up-front so a malformed signal_data dict is skipped (logged)
    # rather than aborting the whole batch.
    cards: list[str] = []
    for sd in signal_data_list:
        try:
            cards.append(AlerterBot._build_message(sd))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[PerTicker] _build_message failed for %s: %s",
                (sd.get("ticker", "?") if isinstance(sd, dict) else "?"), exc,
            )
    if not cards:
        if wait_msg is not None:
            try:
                await wait_msg.edit_text(EMPTY_BUY_RESULT_MESSAGE, parse_mode=ParseMode.HTML)
            except BadRequest as exc:
                LOGGER.warning("empty-card edit_text failed (%s).", exc)
        return

    # Header: edit the loading message in place into a 1-line summary.
    header = f"📊 {len(cards)} tín hiệu — chi tiết từng mã bên dưới."
    if wait_msg is not None:
        try:
            await wait_msg.edit_text(header, parse_mode=ParseMode.HTML)
        except BadRequest as exc:
            LOGGER.warning("header edit_text failed (%s).", exc)

    # 1-way oversight: mirror the header + every card to the Admin when the
    # requester is the monitored User. Best-effort — never affects the reply.
    if ADMIN_CHAT_ID and _role_for(update) == "user":
        try:
            _bot = update.get_bot()
            await _bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="👁️ <b>[GIÁM SÁT] Bản sao phản hồi gửi cho người dùng:</b>",
                parse_mode=ParseMode.HTML,
            )
            for _card in cards:
                await _bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=_card,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=disable_preview,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[Oversight] per-ticker mirror failed: %s", exc)

    # Reverse mirror: ADMIN's suggest_buy5/20 results auto-broadcast to User2.
    if mirror_to_user2 and USER_CHAT_ID and _role_for(update) == "admin":
        try:
            _bot = update.get_bot()
            await _bot.send_message(
                chat_id=USER_CHAT_ID,
                text="📤 <b>[ADMIN] Báo cáo tự động:</b>",
                parse_mode=ParseMode.HTML,
            )
            for _card in cards:
                await _bot.send_message(
                    chat_id=USER_CHAT_ID,
                    text=_card,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=disable_preview,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[User2 mirror] per-ticker forward failed: %s", exc)

    if update.message is None:
        return
    for card in cards:
        try:
            await update.message.reply_text(
                card,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )
        except BadRequest as exc:
            LOGGER.warning(
                "per-ticker reply_text failed (%s); retrying with HTML-escaped text.",
                exc,
            )
            await update.message.reply_text(html.escape(card))


async def _send_or_reply_chunks(
    update: Update,
    wait_msg: Any,
    chunks: list[str],
    *,
    disable_preview: bool = False,
    mirror_to_user2: bool = False,
) -> None:
    """Edit `wait_msg` in place if 1 chunk, else delete it and post fresh replies.

    `mirror_to_user2` (2026-07-14): opt-in reverse mirror, ADMIN → User2. Set
    by the specific command handlers the user asked to auto-broadcast
    (suggest_sell, audit_weekly, audit_monthly, audit_accuracy) — NOT a
    blanket default, since some callers of this shared sender (/guard,
    /verify, /rebalance) were never asked to forward and may carry data the
    user did not approve broadcasting.
    """
    if not chunks:
        return

    # 1-way oversight: mirror the FULL response to the Admin (ID1) whenever
    # the requester is the monitored User (ID2). Runs before the user-facing
    # delivery branches so it fires for both the single-edit and multi-chunk
    # paths. Best-effort — a mirror failure never affects the user reply.
    if ADMIN_CHAT_ID and _role_for(update) == "user":
        try:
            _bot = update.get_bot()
            await _bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="👁️ <b>[GIÁM SÁT] Bản sao phản hồi gửi cho người dùng:</b>",
                parse_mode=ParseMode.HTML,
            )
            for _c in chunks:
                await _bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=_c,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[Oversight] response mirror failed: %s", exc)

    # Reverse mirror: opted-in ADMIN commands auto-broadcast to User2.
    if mirror_to_user2 and USER_CHAT_ID and _role_for(update) == "admin":
        try:
            _bot = update.get_bot()
            await _bot.send_message(
                chat_id=USER_CHAT_ID,
                text="📤 <b>[ADMIN] Báo cáo tự động:</b>",
                parse_mode=ParseMode.HTML,
            )
            for _c in chunks:
                await _bot.send_message(
                    chat_id=USER_CHAT_ID,
                    text=_c,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[User2 mirror] response forward failed: %s", exc)

    if len(chunks) == 1 and wait_msg is not None:
        try:
            await wait_msg.edit_text(
                chunks[0],
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )
            return
        except BadRequest as exc:
            LOGGER.warning("edit_text failed (%s); falling back to fresh message.", exc)

    if wait_msg is not None:
        try:
            await wait_msg.delete()
        except Exception:  # noqa: BLE001
            pass

    if update.message is None:
        return
    for chunk in chunks:
        try:
            await update.message.reply_text(
                chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )
        except BadRequest as exc:
            LOGGER.warning(
                "reply_text failed (%s); retrying with HTML-escaped text.", exc,
            )
            await update.message.reply_text(html.escape(chunk))


def _log_request(name: str, update: Update) -> None:
    user = update.effective_user
    chat = update.effective_chat
    LOGGER.info(
        "%s requested by user_id=%s username=%s chat_id=%s",
        name,
        user.id if user else None,
        user.username if user else None,
        chat.id if chat else None,
    )


async def _audit_log_async(
    user_id: str | None,
    command: str,
    ticker: str | None = None,
    details: str | None = None,
) -> None:
    """Fire-and-forget audit-trail write. Never blocks, never raises.

    DuckDB writes typically take <1ms but we still dispatch via
    `asyncio.to_thread` so the event loop is fully isolated from any
    transient lock contention or disk I/O hiccup.

    `user_id` may be None for anonymous channel posts — in that case we
    skip the write (an audit-log row with NULL user_id has no value).
    """
    if not user_id:
        return
    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        await asyncio.to_thread(db.log_user_action, user_id, command, ticker, details)
    except Exception as exc:  # noqa: BLE001
        # Audit failure must NEVER break the user command. Log and swallow.
        LOGGER.warning("Audit log write failed (user=%s cmd=%s): %s", user_id, command, exc)


def _extract_user_id(update: Update) -> str | None:
    """Return the Telegram user_id of the requester as a string, or None.

    We store user_id as VARCHAR in DuckDB (consistent with `live_positions`),
    so the str() cast happens here. Returns None if the update somehow lacks
    a user (anonymous channel post, etc.) — handlers must reject those.
    """
    # Prefer message.from_user.id (per the spec); fall back to effective_user.id
    # in case the message object is unusual (e.g. an edited callback query).
    if update.message is not None and update.message.from_user is not None:
        return str(update.message.from_user.id)
    if update.effective_user is not None:
        return str(update.effective_user.id)
    return None


# ---------------------------------------------------------------------------
# Split-ID access control + 1-way admin oversight
# ---------------------------------------------------------------------------
# .env defines TELEGRAM_CHAT_ID_1 (Admin) and TELEGRAM_CHAT_ID_2 (User).
# In a Telegram private chat the requester user_id == chat_id, so we match
# on the requester id. ID1 = full access. ID2 = analytical/read-only, BLOCKED
# from portfolio-mutating commands. Every ID2 command (request + response)
# is shadow-copied to ID1 for real-time oversight.
ADMIN_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID_1") or "").strip()
USER_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID_2") or "").strip()
_EDIT_ONLY_COMMANDS = {"add", "remove"}  # portfolio mutation → ADMIN only
_PERMISSION_DENIED_VI = (
    "🚫 Bạn không có quyền thực hiện tính năng chỉnh sửa danh mục này."
)
_ACCESS_DENIED_VI = (
    "🚫 Tài khoản của bạn chưa được cấp quyền sử dụng bot này."
)


def _role_for(update: Update) -> str:
    """'admin' (ID1), 'user' (ID2), or 'unknown'."""
    uid = _extract_user_id(update)
    if uid and ADMIN_CHAT_ID and uid == ADMIN_CHAT_ID:
        return "admin"
    if uid and USER_CHAT_ID and uid == USER_CHAT_ID:
        return "user"
    return "unknown"


def _command_of(update: Update) -> str:
    """Bare command name from the message text ('/add@Bot x' -> 'add')."""
    txt = (update.message.text if update.message and update.message.text else "") or ""
    if not txt.startswith("/") or len(txt) < 2:
        return ""
    return txt[1:].split("@")[0].split()[0].lower()


async def _notify_admin(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Fire-and-forget oversight message to ID1. Never raises/blocks."""
    if not ADMIN_CHAT_ID:
        return
    try:
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[Oversight] admin notify failed: %s", exc)


async def _oversight_gate(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """group=-1 pre-dispatch gate, runs BEFORE every command handler.

    • unknown id      → polite VN denial, stop pipeline.
    • ID2 (user)      → shadow the REQUEST to ID1; if the command mutates
                         the portfolio (/add,/remove) deny + stop.
    • ID1 (admin)     → full access, no shadowing.

    Raising ApplicationHandlerStop prevents the real handler from running
    when access is denied. The RESPONSE copy for ID2 is mirrored centrally
    in `_send_or_reply_chunks` (covers the analytical command outputs).
    """
    if update.message is None or not (update.message.text or "").startswith("/"):
        return

    role = _role_for(update)
    cmd = _command_of(update)

    if role == "unknown":
        try:
            await update.message.reply_text(
                _ACCESS_DENIED_VI, parse_mode=ParseMode.HTML
            )
        finally:
            raise ApplicationHandlerStop

    if role == "user":
        uid = _extract_user_id(update) or "?"
        raw = (update.message.text or "").strip()
        await _notify_admin(
            context,
            f"👁️ <b>[GIÁM SÁT]</b> Người dùng "
            f"<code>{html.escape(uid)}</code> vừa gọi: "
            f"<code>{html.escape(raw)}</code>",
        )
        if cmd in _EDIT_ONLY_COMMANDS:
            await update.message.reply_text(
                _PERMISSION_DENIED_VI, parse_mode=ParseMode.HTML
            )
            await _notify_admin(
                context,
                f"⛔ <b>[GIÁM SÁT]</b> Đã CHẶN lệnh "
                f"<code>/{html.escape(cmd)}</code> của người dùng "
                f"(không có quyền chỉnh sửa danh mục).",
            )
            raise ApplicationHandlerStop
    # role == "admin": full access, no shadow, fall through.


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Reply with the full command menu."""
    if update.message is None:
        return
    _log_request("/help", update)
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def _suggest_buy_dispatch(update: Update, horizon: int) -> None:
    """Backend for /suggest_buy20 and /suggest_buy5 (T+5 short horizon restored).

    Posts ONE minimal loading message (so the bot doesn't look dead during the
    1–2 min inference), then replaces/deletes it with the final cards.  NO
    banners, NO summary table.  Calls `daily_inference(broadcast=False,
    horizon=horizon)` (per-chat push alerts suppressed so only this reply
    delivers the report).  Rate-limit key includes the horizon so the two
    commands have independent cooldowns.
    """
    if update.message is None:
        return
    cmd = f"suggest_buy{horizon}"
    _log_request(f"/{cmd}", update)

    user_id = _extract_user_id(update)
    cooldown_left = _check_and_record_rate_limit(user_id, cmd)
    if cooldown_left is not None:
        await update.message.reply_text(
            f"<b>Quá nhanh.</b> Vui lòng đợi {cooldown_left:.0f}s trước khi gọi lại /{cmd}.",
            parse_mode=ParseMode.HTML,
        )
        return

    await _audit_log_async(user_id=user_id, command=cmd)

    # Minimal loading indicator — replaced (single card) or deleted (multi-card)
    # by the final delivery below, so the bot never looks dead during the 1–2 min
    # inference.  `&amp;` is the HTML entity that renders the "&" character.
    wait_msg = await update.message.reply_text(
        f"⏳ Đang tổng hợp dữ liệu &amp; chạy mô hình T+{horizon}...",
        parse_mode=ParseMode.HTML,
    )

    try:
        from main import daily_inference  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/%s: failed to import daily_inference", cmd)
        await wait_msg.edit_text(
            f"<b>Lỗi import pipeline:</b>\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        # persist=False: this is an interactive preview call (user tapped
        # /suggest_buy5 or /suggest_buy20), NOT the daily cron run. Without
        # this, every manual invocation phantom-executes into the shared
        # automated `portfolio`/`trade_history` ledger (user_id="cron") via
        # PortfolioManager.process_daily_trades — same class of bug the
        # dashboard's persist gate (test_dashboard_persist_gate.py) already
        # fixes for the local-dashboard preview path.
        report_html, signal_data_list = await asyncio.to_thread(
            daily_inference, broadcast=False, horizon=int(horizon), persist=False,
        )
    except FileNotFoundError as exc:
        LOGGER.exception("/%s: V3 horizon=%d bundle missing", cmd, horizon)
        await wait_msg.edit_text(
            f"<b>Thiếu model T+{horizon}.</b>\n"
            f"<code>{html.escape(str(exc))}</code>\n"
            f"<i>Run: python train_models.py --tb-horizon {horizon} → python run_backtest.py</i>",
            parse_mode=ParseMode.HTML,
        )
        return
    except RuntimeError as exc:
        # Feature-recipe tripwire (train/serve skew) — actionable: retrain.
        LOGGER.exception("/%s: V3 horizon=%d feature-recipe mismatch", cmd, horizon)
        await wait_msg.edit_text(
            f"<b>Model T+{horizon} lệch phiên bản feature (cần train lại).</b>\n"
            f"<code>{html.escape(str(exc))}</code>\n"
            f"<i>Run: python train_models.py --tb-horizon {horizon} → python run_backtest.py</i>",
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/%s: daily_inference crashed", cmd)
        await wait_msg.edit_text(
            f"<b>Lỗi khi chạy mô hình:</b>\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not signal_data_list:
        # Fallback observability reports carry no dispatched signals but DO carry
        # report_html — surface it (single tag-safe message) rather than the
        # generic "no signal" line so the weak-market diagnostics still reach
        # the user.
        if report_html and report_html.strip():
            await _send_or_reply_chunks(
                update, wait_msg, _split_html_report(report_html), mirror_to_user2=True
            )
            return
        await wait_msg.edit_text(
            f"<b>Không có tín hiệu MUA T+{horizon} hôm nay.</b>\n"
            "<i>Không có mã nào vượt qua Kelly threshold + Liquidity gate + Sentiment filter.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Paper-track these delivered picks in the dispatched-signals ledger,
    # independent of the persist/broadcast gates above. daily_inference was
    # called with broadcast=False (so the cron-path signal_ledger.record_dispatch
    # never fires) AND persist=False (so the PortfolioManager/paperlog writes are
    # correctly skipped) — without this, every /suggest_buy5|20 pick the user
    # actually received went completely untracked. This does NOT loosen the
    # persist gate: record_dispatch only touches the dispatched_signals ledger,
    # never the shared automated portfolio/trade_history book. Never-raise: a
    # tracking failure must never block delivery of the cards below.
    from config.settings import CONFIG  # noqa: PLC0415

    if CONFIG.trading.suggest_buy_ledger_tracking_enabled:
        try:
            from src.trading import signal_ledger  # noqa: PLC0415

            # T5 tracks a synthetic 5-session window; T20 uses the real tranche
            # portfolio-window convention (hold_days=30, NOT 20) — mirrors the
            # dual-dispatch convention in main.py / the EOD position-report plan.
            _hold_days = 5 if int(horizon) == 5 else 30
            signal_ledger.record_dispatch(
                signal_data_list,
                {"mode": "tranche", "hold_days": _hold_days},
                int(horizon),
            )
        except Exception:  # noqa: BLE001 — ledger tracking must never break delivery
            LOGGER.exception(
                "/%s: signal_ledger.record_dispatch tracking failed", cmd
            )

    # Deliver ONE message per ticker (each is the institutional BUY card, well
    # below the 4096-char limit — no combined-string splitting needed).
    # mirror_to_user2: /suggest_buy5 and /suggest_buy20 are global market
    # signals (not portfolio-scoped) — Admin's results auto-broadcast to User2.
    await _send_per_ticker_reports(update, wait_msg, signal_data_list, mirror_to_user2=True)


async def suggest_buy20_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """T+20 BUY recommendations with tranche cohort sizing."""
    await _suggest_buy_dispatch(update, horizon=20)


_SUGGEST_BUY5_RETIRED = (
    "⚠️ <b>/suggest_buy5 đã ngừng hoạt động</b>\n\n"
    "Nghiên cứu 22-07-26 (<code>scripts/analyze_confluence_backtest.py</code>) cho thấy "
    "cổng T+20 làm toàn bộ việc lọc chất lượng, còn tín hiệu chỉ qua cổng T+5 "
    "gần như là <b>nhiễu</b> — và <b>13/13 lệnh lỗ tháng 7/2026</b> đều vào qua đúng cửa này.\n\n"
    "Dùng <b>/suggest</b> (T+20) để lấy danh sách ứng viên.\n"
    "Vẫn muốn xem góc nhìn T+5 cho một mã cụ thể thì dùng <b>/verify &lt;MÃ&gt;</b> — "
    "ở đó T+5 đóng vai trò xác nhận chéo, không phải nguồn tín hiệu độc lập."
)


async def suggest_buy5_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Retired (10-08-26). Answers with the reason instead of vanishing silently.

    Kept as a handler on purpose: a removed command that simply does nothing
    invites the operator to wonder whether the bot is broken, and the reason it
    was removed is exactly the thing worth restating at the point of use.
    """
    if update.message is None:
        return
    _log_request("/suggest_buy5", update)
    await update.message.reply_text(_SUGGEST_BUY5_RETIRED, parse_mode=ParseMode.HTML)


async def add_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Insert one row into the DuckDB `portfolio` table for the requesting user.

    Usage: /add <ticker> <volume:int> <price:float>
    Example: /add VNE 1000 32.5
    """
    if update.message is None:
        return
    _log_request("/add", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    args = list(context.args or [])
    if len(args) != 3:
        await update.message.reply_text(
            "❌ <b>Sai cú pháp.</b>\n"
            "Đúng: <code>/add [Mã] [Khối lượng] [Giá]</code>\n"
            "VD: <code>/add VNE 1000 32.5</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    raw_ticker, raw_volume, raw_price = args
    ticker = raw_ticker.strip().upper()

    if not _TICKER_RE.fullmatch(ticker):
        await update.message.reply_text(
            f"❌ <b>Mã không hợp lệ:</b> <code>{html.escape(ticker)}</code>\n"
            "<i>Chỉ chấp nhận 2-6 ký tự A-Z hoặc 0-9.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        volume = int(raw_volume)
        price = float(raw_price)
    except ValueError:
        await update.message.reply_text(
            "❌ <b>Khối lượng phải là số nguyên, giá phải là số thực.</b>\n"
            f"<i>Bạn nhập: volume={html.escape(raw_volume)}, "
            f"price={html.escape(raw_price)}</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    if volume <= 0 or price <= 0:
        await update.message.reply_text(
            "❌ <b>Khối lượng và giá phải lớn hơn 0.</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        db.conn.execute(
            "INSERT INTO portfolio (user_id, ticker, volume, price, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [user_id, ticker, volume, price, datetime.now()],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/add DB insert failed for user_id=%s ticker=%s", user_id, ticker)
        await update.message.reply_text(
            f"❌ <b>Lỗi DB:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit trail (after successful insert so we log committed state only).
    await _audit_log_async(
        user_id=user_id,
        command="add",
        ticker=ticker,
        details=f"vol:{volume}, price:{price}",
    )

    # Pretty-print the price: drop the trailing .0 for whole values, else 1 dp.
    price_str = f"{price:.1f}" if price != int(price) else str(int(price))
    await update.message.reply_text(
        f"✅ <b>Đã thêm vào danh mục:</b> {html.escape(ticker)} | "
        f"SL: {volume} | Giá: {price_str}",
        parse_mode=ParseMode.HTML,
    )


async def remove_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete every row matching (user_id, ticker) from the `portfolio` table.

    Usage: /remove <ticker>
    Example: /remove VNE
    """
    if update.message is None:
        return
    _log_request("/remove", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    args = list(context.args or [])
    if len(args) != 1:
        await update.message.reply_text(
            "❌ <b>Sai cú pháp.</b>\n"
            "Đúng: <code>/remove [Mã]</code>\n"
            "VD: <code>/remove VNE</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ticker = args[0].strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        await update.message.reply_text(
            f"❌ <b>Mã không hợp lệ:</b> <code>{html.escape(ticker)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        # Count first so we can give the user a meaningful response.
        row = db.conn.execute(
            "SELECT COUNT(*) FROM portfolio WHERE user_id = ? AND ticker = ?",
            [user_id, ticker],
        ).fetchone()
        existing_count = int(row[0]) if row else 0

        if existing_count == 0:
            await update.message.reply_text(
                f"⚠️ <b>Không tìm thấy {html.escape(ticker)} trong danh mục của bạn.</b>",
                parse_mode=ParseMode.HTML,
            )
            return

        db.conn.execute(
            "DELETE FROM portfolio WHERE user_id = ? AND ticker = ?",
            [user_id, ticker],
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/remove DB delete failed for user_id=%s ticker=%s", user_id, ticker)
        await update.message.reply_text(
            f"❌ <b>Lỗi DB:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit trail.
    await _audit_log_async(
        user_id=user_id,
        command="remove",
        ticker=ticker,
        details=f"rows_deleted:{existing_count}",
    )

    await update.message.reply_text(
        f"✅ <b>Đã xóa {html.escape(ticker)} khỏi danh mục</b> "
        f"({existing_count} dòng).",
        parse_mode=ParseMode.HTML,
    )


async def suggest_sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Recommend BÁN/GIỮ for every ticker in THIS user's portfolio.

    Strictly filters by `user_id` so each Telegram user only sees their own
    holdings — multi-user safe.
    """
    if update.message is None:
        return
    _log_request("/suggest_sell", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    # 1. Read holdings — scoped strictly to this user_id.
    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        rows = db.conn.execute(
            "SELECT DISTINCT ticker FROM portfolio WHERE user_id = ?",
            [user_id],
        ).fetchall()
        holding_tickers = sorted({str(r[0]).upper().strip() for r in rows if r and r[0]})
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/suggest_sell: DB read failed for user_id=%s", user_id)
        await update.message.reply_text(
            f"❌ <b>Lỗi DB:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit trail: log EVERY /suggest_sell, including empty-portfolio cases,
    # so we can compare "tried to sell" vs "actually had holdings" rates.
    await _audit_log_async(
        user_id=user_id,
        command="suggest_sell",
        details=f"portfolio_size:{len(holding_tickers)}",
    )

    if not holding_tickers:
        await update.message.reply_text(EMPTY_PORTFOLIO_MESSAGE, parse_mode=ParseMode.HTML)
        return

    # 2. Acknowledge.
    wait_msg = await update.message.reply_text(
        f"⏳ Đang phân tích <b>{len(holding_tickers)}</b> cổ phiếu trong danh mục...\n"
        f"<i>Tickers:</i> <code>{html.escape(', '.join(holding_tickers))}</code>",
        parse_mode=ParseMode.HTML,
    )

    # 3. Lazy-import the heavy ML pipeline.
    try:
        from main import inference_for_holdings  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/suggest_sell: import failed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # 4. Run inference off the event loop.
    try:
        report_html: str = await asyncio.to_thread(inference_for_holdings, holding_tickers)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/suggest_sell: inference crashed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi mô hình:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        await wait_msg.edit_text(
            "📭 <b>Không thể phân tích danh mục.</b>\n"
            "<i>Có thể các mã đã bị hủy niêm yết hoặc chưa có dữ liệu live.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # mirror_to_user2 (2026-07-14, user-requested): when ADMIN runs
    # /suggest_sell, the recommendation for ADMIN's OWN /add holdings is also
    # broadcast to User2 — User2 will see what Admin holds and Admin's BÁN/GIỮ
    # recommendation for it. Approved explicitly by the user despite the
    # privacy tradeoff being flagged.
    await _send_or_reply_chunks(
        update, wait_msg, _split_html_report(report_html), mirror_to_user2=True
    )


async def guard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """On-demand portfolio guard — protective trigger check for THIS user's holdings.

    Mirrors /suggest_sell: strictly `user_id`-scoped, READ-ONLY, never mutates the
    portfolio and never auto-sells. Reuses `main._run_guard_for_users` (the exact
    evaluation core the EOD sweep uses), so the five deterministic triggers +
    optional arbitrator enrichment are identical to the automatic alert.
    """
    if update.message is None:
        return
    _log_request("/guard", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    # 1. Read THIS user's non-cron holdings (read-only, cron-excluded by loader).
    try:
        from src.trading.portfolio_guard import (  # noqa: PLC0415
            GUARD_NO_TRIGGERS_MESSAGE,
            load_guard_positions,
        )
        positions = load_guard_positions(user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/guard: position load failed for user_id=%s", user_id)
        await update.message.reply_text(
            f"❌ <b>Lỗi DB:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await _audit_log_async(user_id=user_id, command="guard",
                           details=f"portfolio_size:{len(positions)}")

    if not positions:
        await update.message.reply_text(EMPTY_PORTFOLIO_MESSAGE, parse_mode=ParseMode.HTML)
        return

    # 2. Acknowledge.
    wait_msg = await update.message.reply_text(
        f"🛡️ Đang kiểm tra <b>{len(positions)}</b> vị thế trong danh mục...",
        parse_mode=ParseMode.HTML,
    )

    # 3. Lazy-import the evaluation core (heavy ML pipeline).
    try:
        from main import _run_guard_for_users  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/guard: import failed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # 4. Run off the event loop.
    try:
        cards: dict[str, str] = await asyncio.to_thread(
            _run_guard_for_users, {user_id: positions}
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/guard: evaluation crashed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi mô hình:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # 5. Reply with the card, or the "all stable" message when nothing fired.
    card_html = cards.get(user_id) if cards else None
    if card_html:
        await _send_or_reply_chunks(update, wait_msg, _split_html_report(card_html))
    else:
        await wait_msg.edit_text(GUARD_NO_TRIGGERS_MESSAGE, parse_mode=ParseMode.HTML)


def _build_exits_report(rows: list[dict]) -> str:
    """HTML report for /exits — open tranche cohorts + sessions remaining.

    Pure formatter (unit-testable). `rows` is `signal_ledger.list_open()`
    output: ticker, dispatch_date, hold_days, weight, sessions_elapsed,
    sessions_remaining.
    """
    header = (
        f"📒 <b>VỊ THẾ TRANCHE ĐANG MỞ</b>\n"
        f"📅 {datetime.now().strftime('%d/%m/%Y')}\n"
        f"══════════════════════════════"
    )
    if not rows:
        return (f"{header}\n\n<i>Không có vị thế tranche nào đang mở. "
                f"Tín hiệu mới sẽ được ghi sổ khi mô hình T+20 phát lệnh.</i>")

    # Horizon tag ([T5]/[T20]) so mixed dual-horizon ledger rows are unambiguous
    # (a T5 paper-tracking row must not read as a real weighted tranche position).
    from src.reports.builders import _HORIZON_LABEL  # noqa: PLC0415
    lines = []
    for r in rows:
        disp = r.get("dispatch_date")
        disp_str = disp.strftime("%d/%m/%Y") if hasattr(disp, "strftime") else str(disp)
        weight_pct = float(r.get("weight") or 0.0) * 100
        remaining = int(r.get("sessions_remaining", 0))
        hz_tag = _HORIZON_LABEL.get(r.get("horizon"), "?")
        status = ("⏰ <b>ĐẾN HẠN — thoát ATC hôm nay</b>" if remaining <= 0
                  else f"còn <b>{remaining}</b> phiên")
        lines.append(
            f"• <b>{html.escape(str(r['ticker']))}</b> [{hz_tag}] — vào {disp_str} "
            f"({weight_pct:.1f}% NAV), đã {int(r['sessions_elapsed'])}/{int(r['hold_days'])} phiên, "
            f"{status}"
        )
    return f"{header}\n\n" + "\n".join(lines)


async def exits_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """List open tranche cohorts from the dispatched-signal ledger."""
    if update.message is None:
        return
    _log_request("/exits", update)
    await _audit_log_async(user_id=_extract_user_id(update), command="exits")

    try:
        from src.trading import signal_ledger  # noqa: PLC0415
        rows = await asyncio.to_thread(signal_ledger.list_open)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/exits: ledger read failed")
        await update.message.reply_text(
            f"❌ <b>Lỗi đọc sổ lệnh:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.message.reply_text(_build_exits_report(rows), parse_mode=ParseMode.HTML)


async def verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run ad-hoc short-horizon (T+5) quant + LLM-sentiment analysis on a single ticker.

    Use case: a user hears a rumor / sees news and wants instant verification
    before manually trading. Output combines the 5d Stacking GBDT prediction
    with the latest LLM-derived sentiment + sources, so the user can sanity-
    check whether the model agrees or disagrees with the rumor.
    """
    if update.message is None:
        return
    _log_request("/verify", update)

    args = list(context.args or [])
    if len(args) != 1:
        await update.message.reply_text(
            "❌ <b>Sai cú pháp.</b>\n"
            "Đúng: <code>/verify [Mã]</code>\n"
            "VD: <code>/verify HPG</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ticker = args[0].strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        await update.message.reply_text(
            f"❌ <b>Mã không hợp lệ:</b> <code>{html.escape(ticker)}</code>\n"
            "<i>Chỉ chấp nhận 2-6 ký tự A-Z hoặc 0-9.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # TD-30: per-user 30 s cooldown — same reasoning as /suggest_buy.
    user_id = _extract_user_id(update)
    cooldown_left = _check_and_record_rate_limit(user_id, "verify")
    if cooldown_left is not None:
        await update.message.reply_text(
            f"⏳ <b>Quá nhanh!</b> Vui lòng đợi <b>{cooldown_left:.0f}s</b> "
            f"trước khi gọi lại /verify.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit trail (after validation so we only log well-formed requests).
    await _audit_log_async(
        user_id=user_id,
        command="verify",
        ticker=ticker,
    )

    # Initial acknowledgement (per spec). The wait message is edited /
    # replaced by the final report so the chat stays clean.
    wait_msg = await update.message.reply_text(
        f"⏳ Đang kiểm định mã <b>{html.escape(ticker)}</b> "
        f"qua hệ thống Quant &amp; Sentiment...",
        parse_mode=ParseMode.HTML,
    )

    # Lazy-import the ML pipeline (keeps bot startup fast).
    try:
        from main import verify_single_ticker  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/verify: import failed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Run the heavy work off the event loop so /help and other commands
    # stay responsive while news scraping + Gemini call are in flight.
    try:
        report_html: str = await asyncio.to_thread(verify_single_ticker, ticker)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/verify: helper crashed for %s", ticker)
        await wait_msg.edit_text(
            f"❌ <b>Lỗi mô hình:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        await wait_msg.edit_text(
            f"📭 <b>Không thể phân tích {html.escape(ticker)}.</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Reply with the 20 most recent items from the configured RSS feeds."""
    if update.message is None:
        return
    _log_request("/news", update)

    try:
        from src.crawlers.sentiment_crawler import fetch_latest_market_news  # noqa: PLC0415
        # feedparser blocks on network → run in worker thread.
        news_items = await asyncio.to_thread(fetch_latest_market_news, 20)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/news fetch failed")
        await update.message.reply_text(
            f"❌ <b>Lỗi lấy tin tức:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not news_items:
        await update.message.reply_text(
            "📭 <b>Không có tin tức nào.</b>\n"
            "<i>Các RSS feed có thể đang offline.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    lines: list[str] = [f"📰 <b>Top {len(news_items)} tin tức tài chính mới nhất</b>", ""]
    for item in news_items:
        title = html.escape(str(item.get("title", "")))
        # html.escape with quote=True covers `&` and `"` for use inside href="...".
        url = html.escape(str(item.get("url", "")), quote=True)
        source = html.escape(str(item.get("source", "")))
        if not title or not url:
            continue
        lines.append(f"• [<i>{source}</i>] <a href=\"{url}\">{title}</a>")

    body = "\n".join(lines)
    chunks = _chunk_lines(lines)

    # `disable_web_page_preview=True` prevents Telegram from rendering 20 fat
    # link cards which would make the chat unusable.
    if len(chunks) == 1:
        try:
            await update.message.reply_text(
                body,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return
        except BadRequest as exc:
            LOGGER.warning("news single-chunk reply failed: %s", exc)

    for chunk in chunks:
        try:
            await update.message.reply_text(
                chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except BadRequest as exc:
            LOGGER.warning("news chunk reply failed (%s); retrying without parse_mode.", exc)
            await update.message.reply_text(chunk)


async def _admin_retrain_task(update: Update) -> None:
    """Non-blocking rolling-retrain for the ADMIN audit flow.

    Runs ``src/scripts/audit_and_retrain.py`` in a worker thread (via
    ``asyncio.to_thread`` — keeps the event loop fully responsive and is
    Windows-safe vs. event-loop subprocess transports), then pushes a
    plain-VN completion summary to ID1.
    """
    bot = update.get_bot()

    async def _to_admin(text: str) -> None:
        if not ADMIN_CHAT_ID:
            return
        try:
            await bot.send_message(
                chat_id=ADMIN_CHAT_ID, text=text,
                parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[AdminRetrain] notify failed: %s", exc)

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "src/scripts/audit_and_retrain.py"],
            capture_output=True, text=True, timeout=900,
        )
        rc = proc.returncode
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[AdminRetrain] subprocess crashed")
        await _to_admin(
            f"❌ <b>[ADMIN] Luồng học lại lỗi:</b> "
            f"<code>{html.escape(str(exc))}</code>"
        )
        return

    summary = "Hoàn tất (không đọc được tóm tắt)."
    try:
        sp = Path("models/mr/last_retrain_summary.txt")
        if sp.exists():
            summary = sp.read_text(encoding="utf-8").strip() or summary
    except Exception:  # noqa: BLE001
        pass
    head = "🤖 <b>[ADMIN] Luồng học lại cuốn chiếu đã xong</b>"
    if rc != 0:
        head = "🤖 <b>[ADMIN] Luồng học lại KẾT THÚC (có lỗi)</b>"
    await _to_admin(f"{head}\n{html.escape(summary)}")


async def _run_audit_command(
    update: Update,
    command_label: str,
    days: int,
) -> None:
    """Shared implementation for /audit_weekly and /audit_monthly.

    Pulls the user's auditable commands from audit_log, prices them against
    historical OHLCV, asks Gemini for catalysts, and replies with an HTML
    digest. Runs off the event loop via `asyncio.to_thread` so /help and
    other handlers stay responsive during the ~5–15s evaluation.
    """
    if update.message is None:
        return
    _log_request(f"/{command_label}", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> "
            "Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit-trail this audit invocation itself — useful to see how often
    # users review their own performance.
    await _audit_log_async(user_id=user_id, command=command_label)

    wait_msg = await update.message.reply_text(
        f"⏳ Đang chạy hậu kiểm <b>{days} ngày qua</b>...\n"
        f"<i>(Tải giá lịch sử + phân tích tin tức bằng AI — mất ~10s.)</i>",
        parse_mode=ParseMode.HTML,
    )

    # Lazy import — keeps bot startup time low and avoids importing the
    # heavy ML stack until a user actually requests an audit.
    try:
        from src.utils.audit_evaluator import run_post_mortem  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/%s: import failed", command_label)
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        report_html: str = await asyncio.to_thread(run_post_mortem, user_id, days)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/%s: evaluator crashed for user_id=%s", command_label, user_id)
        await wait_msg.edit_text(
            f"❌ <b>Lỗi hậu kiểm:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        await wait_msg.edit_text(
            "📭 <b>Không có dữ liệu để hậu kiểm.</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    # mirror_to_user2 (2026-07-14, user-requested): covers /audit_weekly and
    # /audit_monthly (this shared handler). Same privacy tradeoff as
    # /suggest_sell — the caller's own /verify+/add history broadcasts to
    # User2 when Admin is the caller. Approved explicitly by the user.
    await _send_or_reply_chunks(
        update, wait_msg, _split_html_report(report_html), mirror_to_user2=True
    )

    # ── DUAL-AUDIT ROUTING ───────────────────────────────────────────
    # Both roles get the IDENTICAL post-mortem above. ADMIN (ID1) ALSO
    # kicks off the rolling retrain in the background (non-blocking) and
    # is notified on completion. USER (ID2) gets the report ONLY — no
    # retrain. (Unknown ids never reach here — the group=-1 oversight
    # gate already denied them.)
    if _role_for(update) == "admin":
        await update.message.reply_text(
            "⏳ [ADMIN] Báo cáo hiệu suất đã gửi. Hệ thống đang kích hoạt "
            "luồng học lại cuốn chiếu (Retrain) dưới nền, dự kiến hoàn "
            "thành sau 1-2 phút...",
        )
        # Fire-and-forget: returns immediately, event loop stays free;
        # the subprocess runs in a worker thread inside the task.
        asyncio.create_task(_admin_retrain_task(update))


async def audit_weekly_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """7-day post-mortem of /verify and /add commands."""
    await _run_audit_command(update, command_label="audit_weekly", days=7)


async def audit_monthly_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """30-day post-mortem of /verify and /add commands."""
    await _run_audit_command(update, command_label="audit_monthly", days=30)


async def audit_accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Confusion-matrix accuracy audit of the model's OWN settled predictions.

    System-wide (paperlog is global, not user-scoped): grades every settled
    forward prediction in `sentiment_entry_paperlog` into TP/FP/TN/FN and reports
    BUY precision + defensive recall + the last 15 settled positions. Read-only.
    """
    if update.message is None:
        return
    _log_request("/audit_accuracy", update)

    wait_msg = await update.message.reply_text(
        "⏳ Đang chấm điểm độ chính xác mô hình "
        "<i>(ma trận nhầm lẫn trên các dự báo đã chốt T+20)</i>...",
        parse_mode=ParseMode.HTML,
    )

    try:
        from src.utils.accuracy_audit import build_accuracy_report  # noqa: PLC0415

        report_html: str = await asyncio.to_thread(build_accuracy_report)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/audit_accuracy crashed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi hậu kiểm:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        await wait_msg.edit_text(
            "📭 <b>Chưa có dự báo nào đã chốt (T+20) để hậu kiểm.</b>\n"
            "<i>Cần ít nhất một dự báo đủ ~21 ngày để chốt kết quả.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    # /audit_accuracy is system-wide (paperlog is global) — safe broadcast.
    await _send_or_reply_chunks(
        update, wait_msg, _split_html_report(report_html), mirror_to_user2=True
    )


async def rebalance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Fetch live holdings and reply with an AI portfolio rebalance recommendation."""
    if update.message is None:
        return
    _log_request("/rebalance", update)

    user_id = _extract_user_id(update)
    if user_id is None:
        await update.message.reply_text(
            "❌ <b>Không xác định được user_id.</b> Lệnh này yêu cầu chat 1-1 với bot.",
            parse_mode=ParseMode.HTML,
        )
        return

    cooldown_left = _check_and_record_rate_limit(user_id, "rebalance")
    if cooldown_left is not None:
        await update.message.reply_text(
            f"⏳ <b>Quá nhanh!</b> Vui lòng đợi <b>{cooldown_left:.0f}s</b> "
            f"trước khi gọi lại /rebalance.",
            parse_mode=ParseMode.HTML,
        )
        return

    await _audit_log_async(user_id=user_id, command="rebalance")

    wait_msg = await update.message.reply_text(
        "⏳ Đang phân tích danh mục &amp; tư vấn cơ cấu lại... "
        "Vui lòng đợi khoảng 1 phút!",
        parse_mode=ParseMode.HTML,
    )

    try:
        from main import rebalance_portfolio  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/rebalance: import failed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        report_html: str = await asyncio.to_thread(rebalance_portfolio, user_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/rebalance: rebalance_portfolio crashed for user_id=%s", user_id)
        await wait_msg.edit_text(
            f"❌ <b>Lỗi khi phân tích danh mục:</b> <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        await wait_msg.edit_text(EMPTY_PORTFOLIO_MESSAGE, parse_mode=ParseMode.HTML)
        return

    await _send_or_reply_chunks(update, wait_msg, [report_html])


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — log the exception with the offending update for debugging."""
    LOGGER.exception("Unhandled exception while processing update: %r", update, exc_info=context.error)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

async def feedback_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Direct User → Admin feedback channel.

    `/feedback <message>` forwards the verbatim message to ADMIN_CHAT_ID.
    Bare `/feedback` returns usage help. The oversight gate (group=-1)
    already denies unknown ids, so only ID1/ID2 reach this handler.
    """
    if update.message is None:
        return
    _log_request("/feedback", update)

    msg = " ".join(context.args).strip() if context.args else ""
    if not msg:
        await update.message.reply_text(
            "Vui lòng nhập nội dung. Ví dụ: /feedback Bot dạo này lag quá!"
        )
        return

    uid = _extract_user_id(update) or "?"
    LOGGER.info("[Feedback] from user_id=%s: %s", uid, msg)
    await _notify_admin(
        context,
        f"📩 <b>[FEEDBACK TỪ USER]</b> "
        f"(<code>{html.escape(uid)}</code>): {html.escape(msg)}",
    )
    await update.message.reply_text(
        "✅ Cảm ơn bạn! Feedback đã được gửi trực tiếp đến Admin."
    )


async def msg_id2_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Admin (ID1) → User (ID2) one-way announcement.

    STRICT AUTH: only ID1 may use this. Any other sender — including ID2 —
    is SILENTLY ignored (no reply, no error), per spec. Deliberately NOT
    added to the public command menu / /help so the monitored user never
    sees it advertised.
    """
    if update.message is None:
        return
    if _role_for(update) != "admin":          # silent ignore for non-admin
        return
    _log_request("/msg_id2", update)

    msg = " ".join(context.args).strip() if context.args else ""
    if not msg:
        await update.message.reply_text(
            "Vui lòng nhập nội dung. Ví dụ: /msg_id2 Hệ thống vừa cập nhật "
            "tính năng mới nha!"
        )
        return

    if not USER_CHAT_ID:
        await update.message.reply_text(
            "⚠️ Chưa cấu hình TELEGRAM_CHAT_ID_2 — không thể gửi thông báo."
        )
        return

    try:
        await context.bot.send_message(
            chat_id=USER_CHAT_ID,
            text=f"📢 <b>[THÔNG BÁO TỪ ADMIN]:</b> {html.escape(msg)}",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:  # noqa: BLE001
        # Never falsely confirm delivery — report the real failure to Admin.
        LOGGER.warning("[msg_id2] delivery to ID2 failed: %s", exc)
        await update.message.reply_text(
            f"❌ Gửi thất bại: <code>{html.escape(str(exc)[:200])}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    LOGGER.info("[msg_id2] Admin announcement delivered to ID2 (%s chars).", len(msg))
    await update.message.reply_text(
        "✅ Đã gửi thông báo cho ID 2 thành công!"
    )


# The canonical command list pushed to Telegram via set_my_commands on startup.
# This populates the "/" autocomplete menu in every chat.
# Menu order and wording (10-08-26 review), for an operator who decides every
# trade themselves:
#   * ONE candidate command. /suggest_buy5 is retired — T+5-only signals are
#     ~noise and produced every losing July-2026 dispatch.
#   * "ứng viên" not "khuyến nghị": the system presents candidates, the operator
#     decides. Live paperlog has BUY calls averaging -0.69 pct over 43 settled
#     rows, which does not support recommendation language.
#   * /audit_accuracy promoted near the top — it is the scoreboard that tells
#     the operator how much to trust everything above it.
_BOT_COMMANDS: list[BotCommand] = [
    BotCommand("suggest", "Danh sách ứng viên MUA (T+20) — bạn tự quyết"),
    BotCommand("verify", "Kiểm định nhanh 1 cổ phiếu — T+5 & T+20 (VD: /verify HPG)"),
    BotCommand("audit_accuracy", "Bảng điểm mô hình — TP/FP/TN/FN, Precision/Recall"),
    BotCommand("guard", "Kiểm tra cảnh báo bảo vệ danh mục (cắt lỗ/trailing/pha)"),
    BotCommand("suggest_sell", "Đánh giá BÁN/GIỮ cho danh mục cá nhân"),
    BotCommand("exits", "Vị thế tranche đang mở + số phiên còn lại"),
    BotCommand("add", "Thêm cổ phiếu vào danh mục (VD: /add VNE 1000 32.5)"),
    BotCommand("remove", "Xóa cổ phiếu khỏi danh mục (VD: /remove VNE)"),
    BotCommand("audit_weekly", "Hậu kiểm /verify & /add trong 7 ngày qua"),
    BotCommand("audit_monthly", "Hậu kiểm /verify & /add trong 30 ngày qua"),
    BotCommand("rebalance", "AI gợi ý cơ cấu danh mục hiện tại"),
    BotCommand("news", "Tổng hợp tin tức từ 20 nguồn gần nhất"),
    BotCommand("feedback", "Gửi góp ý trực tiếp đến Admin (VD: /feedback ...)"),
    BotCommand("help", "Hiển thị danh sách lệnh"),
]

# Admin-only commands: appended to the menu ONLY in the Admin chat via a
# chat-scoped set_my_commands — User2's "/" menu must not advertise commands
# whose auth gate would just reject them.
_ADMIN_COMMANDS: list[BotCommand] = [
    BotCommand("msg_user2", "Gửi thông báo đến User (ID 2) — chỉ Admin"),
]


async def _set_bot_commands(app: Application) -> None:
    """Post-init hook: push command list to Telegram so the slash menu is current.

    Default scope gets the shared list; the Admin chat additionally gets the
    admin-only commands (chat-scoped menus override the default for that chat).
    """
    await app.bot.set_my_commands(_BOT_COMMANDS)
    LOGGER.info(
        "Telegram bot commands registered: %s",
        [c.command for c in _BOT_COMMANDS],
    )
    if ADMIN_CHAT_ID:
        try:
            await app.bot.set_my_commands(
                _BOT_COMMANDS + _ADMIN_COMMANDS,
                scope=BotCommandScopeChat(chat_id=int(ADMIN_CHAT_ID)),
            )
            LOGGER.info(
                "Admin-scoped commands registered: %s",
                [c.command for c in _ADMIN_COMMANDS],
            )
        except Exception as exc:  # noqa: BLE001 — menu cosmetics must not kill startup
            LOGGER.warning("Admin-scoped command menu failed (%s) — global menu only.", exc)


async def msg_user2_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin → User relay. Only TELEGRAM_CHAT_ID_1 (Admin) may send free-text to
    TELEGRAM_CHAT_ID_2 (User). Usage: /msg_user2 <nội dung>."""
    if update.message is None:
        return
    _log_request("/msg_user2", update)

    # Auth gate — strictly the Admin id (TELEGRAM_CHAT_ID_1).
    if not ADMIN_CHAT_ID or _extract_user_id(update) != ADMIN_CHAT_ID:
        await update.message.reply_text(
            "⛔ Quyền truy cập bị từ chối.", parse_mode=ParseMode.HTML,
        )
        return

    # Message body = everything after the command.
    body = " ".join(context.args).strip() if context.args else ""
    if not body:
        await update.message.reply_text(
            "Cú pháp: <code>/msg_user2 &lt;nội dung&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    if not USER_CHAT_ID:
        await update.message.reply_text(
            "❌ <b>Chưa cấu hình TELEGRAM_CHAT_ID_2.</b>", parse_mode=ParseMode.HTML,
        )
        return

    # Deliver to User (HTML, escape the body so user text can't break parsing).
    try:
        await context.bot.send_message(
            chat_id=USER_CHAT_ID,
            text=f"📩 <b>Tin nhắn từ Admin:</b>\n{html.escape(body)}",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[msg_user2] delivery to ID2 failed: %s", exc)
        await update.message.reply_text(
            f"❌ Gửi thất bại: <code>{html.escape(str(exc)[:200])}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.message.reply_text("✅ Đã gửi tin nhắn cho User (ID2) thành công!")


# INTRADAY ATTACK SCANNER — UNWIRED 13-08-26 (operator decision)
# ───────────────────────────────────────────────────────────────
# `_intraday_scan_job` (the PTB repeating-job adapter) and
# `_register_intraday_scanner` (its config-gated registration) were removed here.
# The scanner alerted on price/probability MOVEMENT during the session.
#
# It WAS live, contrary to the "ships disabled" note it carried: the dataclass
# default was False but `config/settings.json` overrode it to `true` (interval 60,
# clamped to 30), so every bot run registered the job and broadcast movement cards
# to BOTH chats during HOSE hours. Removing the settings.json keys is what
# actually stops it — the dataclass default never had the final say.
#
# `src/trading/intraday_scanner.py` and its 39 tests are KEPT ON PURPOSE. The
# module is pure (no telegram, no asyncio) and its provisional-bar machinery
# — `snapshot_row_to_provisional_bar`, `splice_provisional_bar`, `splice_all` —
# is exactly what an intraday FOREIGN-FLOW experiment needs: the 13-08 re-test
# found foreign flow correlates +0.26 (Spearman) with the SAME-DAY return and
# ~0 at t+1, so the only place that correlation could ever be tradeable is
# mid-session. Deleting the splicing code and rebuilding it later would be waste.
#
# Consequence: the bot registers NO repeating jobs, so `app.job_queue` is now
# unused. The `python-telegram-bot[job-queue]==22.7` pin in requirements.txt is
# left alone deliberately — it is harmless, and dropping an installed extra is a
# way to break an environment for no gain.


def build_application() -> Application:
    """Build the python-telegram-bot Application with every command handler wired up."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "YOUR_BOT_TOKEN":
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN env var is missing or placeholder. "
            "Set it in `.env` (see `.env.example`)."
        )

    app = ApplicationBuilder().token(token).post_init(_set_bot_commands).build()

    # --- Phase 0: split-ID access control + 1-way oversight (runs FIRST) ---
    # group=-1 → evaluated before any CommandHandler. Denies unknown ids,
    # blocks ID2 from /add,/remove, and shadow-copies ID2 activity to ID1.
    app.add_handler(TypeHandler(Update, _oversight_gate), group=-1)

    # --- Phase 1 ---
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("start", help_command))

    # --- Phase 2 ---
    # NOTE: Telegram's bot_command entity parser only matches `/[a-zA-Z0-9_]+`.
    # Hyphens are NOT valid in slash commands. Canonical commands use the
    # underscore form; the hyphen MessageHandler regexes are fallbacks for
    # users typing the hyphenated forms manually.
    # Single canonical candidate command (10-08-26): /suggest == T+20, the only
    # horizon whose gate demonstrably filters for quality. /suggest_buy20 stays
    # as a working alias so existing muscle memory keeps functioning.
    app.add_handler(CommandHandler("suggest", suggest_buy20_command))
    app.add_handler(CommandHandler("suggest_buy20", suggest_buy20_command))
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-buy20(@\w+)?(\s|$)"),
            suggest_buy20_command,
        )
    )
    # /suggest_buy5 RETIRED — replies with the reason (T+5-only signals are
    # ~noise and produced every losing July-2026 dispatch). Handler kept so the
    # command explains itself rather than silently doing nothing.
    app.add_handler(CommandHandler("suggest_buy5", suggest_buy5_command))
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-buy5(@\w+)?(\s|$)"),
            suggest_buy5_command,
        )
    )

    # --- Phase 3 ---
    app.add_handler(CommandHandler("add", add_portfolio_command))
    app.add_handler(CommandHandler("remove", remove_portfolio_command))
    app.add_handler(CommandHandler("suggest_sell", suggest_sell_command))
    app.add_handler(CommandHandler("guard", guard_command))
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-sell(@\w+)?(\s|$)"),
            suggest_sell_command,
        )
    )
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("exits", exits_command))
    app.add_handler(CommandHandler("verify", verify_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    # Admin-only announcement channel — intentionally NOT in _BOT_COMMANDS
    # / /help so it stays hidden from the monitored user.
    app.add_handler(CommandHandler("msg_id2", msg_id2_command))
    app.add_handler(CommandHandler("msg_user2", msg_user2_command))

    # Post-mortem audit — weekly/monthly review of /verify + /add accuracy.
    app.add_handler(CommandHandler("audit_weekly", audit_weekly_command))
    app.add_handler(CommandHandler("audit_monthly", audit_monthly_command))
    app.add_handler(CommandHandler("audit_accuracy", audit_accuracy_command))

    # --- Phase 4 ---
    app.add_handler(CommandHandler("rebalance", rebalance_command))

    app.add_error_handler(_on_error)

    # NOTE: the intraday attack scanner's repeating job was UNWIRED on 13-08-26
    # (see the block above `_INTRADAY_STATE_KEY`). The bot now registers no
    # repeating jobs at all.

    # --- Release the DuckDB file lock between commands (24-08-26) ---
    # group=99 → runs AFTER the command handlers in group 0, on every update.
    # PTB walks every group in ascending order (an exception in one group goes to
    # the error handler and processing continues), so this fires even when a
    # command failed — which is exactly when a lock must not be left held.
    #
    # Safe to release per-update because this Application is built WITHOUT
    # `concurrent_updates`, i.e. PTB processes updates sequentially, so no other
    # handler can be mid-query when this runs.
    app.add_handler(TypeHandler(Update, _release_db_lock), group=99)

    return app


async def _release_db_lock(_update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Drop the process's DuckDB connection once a command has finished.

    DuckDB allows ONE writing process, and DuckDBEngine holds the file
    read-write for the process lifetime. A running bot therefore locked out the
    15:30 EOD cron (it crashed AFTER paying for the Gemini sentiment stage) and
    even refused read-only diagnostics. Releasing here shrinks the bot's hold
    from "always" to "only while a command is actually executing".

    This is a complement to `scripts/run_with_bot_paused.ps1`, not a replacement:
    that wrapper remains the DETERMINISTIC serialisation for scheduled jobs. This
    only removes the idle-bot blockade.

    Never raises — a failure to release must not surface to the user as a command
    error, and the next `DuckDBEngine()` re-opens lazily either way.
    """
    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        DuckDBEngine.dispose()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        LOGGER.debug("[DuckDB] post-update release failed.", exc_info=True)


def main() -> None:
    """Long-running entrypoint — starts polling and blocks until SIGINT/SIGTERM."""
    # TD-26: install rotating file handler BEFORE anything else logs.
    # `basicConfig` below becomes a no-op once the root logger has handlers.
    from src.utils.logging_utils import setup_rotating_logging  # noqa: PLC0415
    setup_rotating_logging()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)

    try:
        app = build_application()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)

    # TD-31: stamp the version into the startup banner for log correlation.
    from src.utils.version import get_version  # noqa: PLC0415
    LOGGER.info(
        "Quant V6 Telegram bot starting (polling mode) | version=%s",
        get_version(),
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
