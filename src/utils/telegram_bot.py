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
import sys
import threading
import time as _time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from telegram import BotCommand, Update
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
    "🟢 <b>/suggest_buy</b> — Gợi ý cổ phiếu nên MUA hôm nay.\n"
    "🔴 <b>/suggest_sell</b> — Đánh giá NÊN BÁN hay GIỮ danh mục của bạn.\n"
    "⚖️ <b>/rebalance</b> — Tư vấn cơ cấu lại danh mục hiện tại.\n"
    "🔍 <b>/verify</b> <i>[Mã]</i> — Soi nhanh 1 cổ phiếu "
    "(VD: <code>/verify HPG</code>).\n"
    "➕ <b>/add</b> <i>[Mã] [Số lượng] [Giá]</i> — Thêm cổ phiếu vào danh mục "
    "(VD: <code>/add VNE 1000 32.5</code>).\n"
    "➖ <b>/remove</b> <i>[Mã]</i> — Xóa cổ phiếu khỏi danh mục "
    "(VD: <code>/remove VNE</code>).\n"
    "📅 <b>/audit_weekly</b> — Xem lại hiệu quả các quyết định 7 ngày qua.\n"
    "🗓️ <b>/audit_monthly</b> — Xem lại hiệu quả các quyết định 30 ngày qua.\n"
    "📰 <b>/news</b> — Tổng hợp tin tức thị trường mới nhất.\n"
    "ℹ️ <b>/help</b> — Hiển thị menu này."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_html_report(report: str, max_len: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """Split a long combined HTML report into Telegram-safe chunks.

    Splits on the visual `══════` separator built by `_build_combined_report`
    in main.py so each chunk is one self-contained per-ticker block. Falls
    back to hard char-slicing only if a single block somehow exceeds the limit.
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
            for i in range(0, len(block), max_len):
                chunks.append(block[i : i + max_len])
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


async def _send_or_reply_chunks(
    update: Update,
    wait_msg: Any,
    chunks: list[str],
    *,
    disable_preview: bool = False,
) -> None:
    """Edit `wait_msg` in place if 1 chunk, else delete it and post fresh replies."""
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
            LOGGER.warning("reply_text failed (%s); retrying without parse_mode.", exc)
            await update.message.reply_text(chunk)


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


async def suggest_buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Run the full daily inference pipeline on demand and reply with Top-3 BUY signals.

    Calls `daily_inference(broadcast=False)` so the per-chat-ID push alerts
    in `TELEGRAM_CHAT_ID` are suppressed — only the bot reply is delivered,
    preventing the duplicate-spam bug.
    """
    if update.message is None:
        return
    _log_request("/suggest_buy", update)

    # TD-30: per-user 30 s cooldown. /suggest_buy is expensive (Gemini call
    # + 30 GNews fan-out) so we gate it BEFORE the wait message and audit
    # write — repeat spam costs us nothing past one cheap reply.
    user_id = _extract_user_id(update)
    cooldown_left = _check_and_record_rate_limit(user_id, "suggest_buy")
    if cooldown_left is not None:
        await update.message.reply_text(
            f"⏳ <b>Quá nhanh!</b> Vui lòng đợi <b>{cooldown_left:.0f}s</b> "
            f"trước khi gọi lại /suggest_buy.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Audit trail at start so we record every invocation, including ones that
    # subsequently fail (model crash, network blip) — useful for diagnosing
    # "why is the bot slow at 9am" review questions.
    await _audit_log_async(
        user_id=user_id,
        command="suggest_buy",
    )

    wait_msg = await update.message.reply_text(
        WAIT_MESSAGE_BUY,
        parse_mode=ParseMode.HTML,
    )

    try:
        from main import daily_inference  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to import daily_inference")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import pipeline:</b>\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        # broadcast=False → suppresses TelegramBot.send_signal_alert() pushes,
        # so the chat reply below is the ONLY delivery channel.
        report_html: str = await asyncio.to_thread(daily_inference, broadcast=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/suggest_buy: daily_inference crashed")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi khi chạy mô hình:</b>\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if not report_html or not report_html.strip():
        # Task 2.3: when the pool is empty, append the 5d top-5 probability
        # breakdown so the operator sees EXACTLY why the bot stayed quiet
        # (which tickers were close, and whether τ* or the meta-gate blocked).
        msg = EMPTY_BUY_RESULT_MESSAGE
        try:
            import main as _main  # noqa: PLC0415

            lines = list(getattr(_main, "_LATEST_5D_BREAKDOWN", []))
            if lines:
                body = "\n".join(html.escape(ln) for ln in lines)
                msg += (
                    "\n\n🔎 <b>Vì sao bot im lặng — Top 5 (5d):</b>\n"
                    f"<pre>{body}</pre>"
                )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to attach 5d breakdown to empty reply")
        await wait_msg.edit_text(msg, parse_mode=ParseMode.HTML)
        return

    await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))


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

    await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))


async def verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run ad-hoc 5d quant + LLM-sentiment analysis on a single ticker.

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

    await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))


async def audit_weekly_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """7-day post-mortem of /verify and /add commands."""
    await _run_audit_command(update, command_label="audit_weekly", days=7)


async def audit_monthly_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """30-day post-mortem of /verify and /add commands."""
    await _run_audit_command(update, command_label="audit_monthly", days=30)


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

# The canonical command list pushed to Telegram via set_my_commands on startup.
# This populates the "/" autocomplete menu in every chat.
_BOT_COMMANDS: list[BotCommand] = [
    BotCommand("suggest_buy", "Lấy khuyến nghị MUA (Top 3 Quant & Sentiment)"),
    BotCommand("suggest_sell", "Lấy khuyến nghị BÁN/HOLD cho danh mục cá nhân"),
    BotCommand("rebalance", "AI tư vấn cơ cấu danh mục hiện tại"),
    BotCommand("verify", "Kiểm định nhanh 1 cổ phiếu (VD: /verify HPG)"),
    BotCommand("add", "Thêm cổ phiếu vào danh mục (VD: /add VNE 1000 32.5)"),
    BotCommand("remove", "Xóa cổ phiếu khỏi danh mục (VD: /remove VNE)"),
    BotCommand("audit_weekly", "Hậu kiểm /verify & /add trong 7 ngày qua"),
    BotCommand("audit_monthly", "Hậu kiểm /verify & /add trong 30 ngày qua"),
    BotCommand("news", "Tổng hợp tin tức từ 20 nguồn gần nhất"),
    BotCommand("help", "Hiển thị danh sách lệnh"),
]


async def _set_bot_commands(app: Application) -> None:
    """Post-init hook: push command list to Telegram so the slash menu is current."""
    await app.bot.set_my_commands(_BOT_COMMANDS)
    LOGGER.info(
        "Telegram bot commands registered: %s",
        [c.command for c in _BOT_COMMANDS],
    )


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
    app.add_handler(CommandHandler("suggest_buy", suggest_buy_command))
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-buy(@\w+)?(\s|$)"),
            suggest_buy_command,
        )
    )

    # --- Phase 3 ---
    app.add_handler(CommandHandler("add", add_portfolio_command))
    app.add_handler(CommandHandler("remove", remove_portfolio_command))
    app.add_handler(CommandHandler("suggest_sell", suggest_sell_command))
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-sell(@\w+)?(\s|$)"),
            suggest_sell_command,
        )
    )
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("verify", verify_command))

    # Post-mortem audit — weekly/monthly review of /verify + /add accuracy.
    app.add_handler(CommandHandler("audit_weekly", audit_weekly_command))
    app.add_handler(CommandHandler("audit_monthly", audit_monthly_command))

    # --- Phase 4 ---
    app.add_handler(CommandHandler("rebalance", rebalance_command))

    app.add_error_handler(_on_error)
    return app


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
