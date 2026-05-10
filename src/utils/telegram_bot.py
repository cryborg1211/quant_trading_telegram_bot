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
    Phase 1: framework + /help (DONE)
    Phase 2: /suggest-buy — runs daily_inference() in a worker thread (DONE)
    Phase 3+: /suggest, /add, /news (TODO)
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
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

WAIT_MESSAGE = (
    "⏳ Đang chạy mô hình Stacking Quant &amp; LLM Sentiment... "
    "Vui lòng đợi khoảng 1-2 phút!"
)

EMPTY_RESULT_MESSAGE = (
    "📭 <b>Không có tín hiệu MUA nào sau bộ lọc Sentiment hôm nay.</b>\n"
    "<i>Mô hình đã chạy xong nhưng không có ticker nào vượt qua điều kiện "
    "(thanh khoản + Quant Top 6 + Sentiment).</i>"
)

# HTML body. Telegram supports a strict subset: <b>, <i>, <u>, <s>, <a>,
# <code>, <pre>, <blockquote>, <tg-spoiler>. The ampersand in "Quant & Sentiment"
# is escaped as &amp;. Square brackets do NOT need escaping (only <, >, & do).
HELP_TEXT = (
    "🤖 <b>Quant Trading V6 - Command Menu</b>\n"
    "\n"
    "<b>/suggest_buy</b> - Lấy khuyến nghị MUA "
    "(Dựa trên Top 3 Quant &amp; Sentiment).\n"
    "<b>/suggest</b> - Lấy khuyến nghị BÁN/HOLD cho danh mục hiện tại.\n"
    "<b>/add</b> <i>[Mã] [Khối lượng] [Giá]</i> - Thêm cổ phiếu vào danh mục "
    "quản lý (VD: <code>/add VNE 1000 32.5</code>).\n"
    "<b>/news</b> - Tổng hợp tin tức từ 20 nguồn gần nhất.\n"
    "<b>/help</b> - Hiển thị menu này."
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
        # Block on its own is too large — hard slice (rare).
        if len(block) > max_len:
            for i in range(0, len(block), max_len):
                chunks.append(block[i : i + max_len])
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Reply with the full command menu."""
    if update.message is None:
        return
    user = update.effective_user
    LOGGER.info(
        "/help requested by user_id=%s username=%s chat_id=%s",
        user.id if user else None,
        user.username if user else None,
        update.effective_chat.id if update.effective_chat else None,
    )
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def suggest_buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG001
    """Run the full daily inference pipeline on demand and reply with Top-3 BUY signals.

    Heavy ML work (`daily_inference()`) is dispatched via `asyncio.to_thread()`
    so it runs in the default thread pool without blocking the bot's event
    loop — `/help` and other commands stay responsive while the model runs.
    """
    if update.message is None:
        return

    user = update.effective_user
    LOGGER.info(
        "/suggest-buy requested by user_id=%s username=%s chat_id=%s",
        user.id if user else None,
        user.username if user else None,
        update.effective_chat.id if update.effective_chat else None,
    )

    # 1. Send the initial "please wait" acknowledgement immediately so the
    #    user knows the request was received, even though the model takes
    #    1–2 minutes.
    wait_msg = await update.message.reply_text(
        WAIT_MESSAGE,
        parse_mode=ParseMode.HTML,
    )

    # 2. Lazy-import `main.daily_inference` so the bot's startup time does NOT
    #    pay for joblib / catboost / xgboost imports unless someone actually
    #    invokes the heavy command.
    try:
        from main import daily_inference  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to import daily_inference")
        await wait_msg.edit_text(
            f"❌ <b>Lỗi import pipeline:</b>\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # 3. Run the synchronous, CPU-heavy pipeline off the event loop.
    try:
        report_html: str = await asyncio.to_thread(daily_inference)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("/suggest-buy: daily_inference crashed")
        await wait_msg.edit_text(
            "❌ <b>Lỗi khi chạy mô hình:</b>\n"
            f"<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # 4. Empty / no-signal case → edit the wait message in place.
    if not report_html or not report_html.strip():
        await wait_msg.edit_text(EMPTY_RESULT_MESSAGE, parse_mode=ParseMode.HTML)
        return

    # 5. Send the report. If short enough, edit the wait message in place;
    #    otherwise delete it and post chunks as fresh messages so each
    #    per-ticker block stays under Telegram's 4096-char hard limit.
    chunks = _split_html_report(report_html)
    if len(chunks) == 1:
        try:
            await wait_msg.edit_text(chunks[0], parse_mode=ParseMode.HTML)
            return
        except BadRequest as exc:
            LOGGER.warning("edit_text failed (%s); falling back to fresh message.", exc)

    # Multi-chunk OR edit failed: drop the wait message, send each chunk fresh.
    try:
        await wait_msg.delete()
    except Exception:  # noqa: BLE001
        pass
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except BadRequest as exc:
            LOGGER.warning("reply_text failed for chunk (%s); retrying without parse_mode.", exc)
            await update.message.reply_text(chunk)


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler — log the exception with the offending update for debugging."""
    LOGGER.exception("Unhandled exception while processing update: %r", update, exc_info=context.error)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def build_application() -> Application:
    """
    Build the python-telegram-bot Application with all command handlers wired up.

    Future phases extend this function (add more `CommandHandler`s) — every
    handler should be registered here so `main()` stays a thin entrypoint.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "YOUR_BOT_TOKEN":
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN env var is missing or placeholder. "
            "Set it in `.env` (see `.env.example`)."
        )

    app = ApplicationBuilder().token(token).build()

    # --- Phase 1 handlers ---
    app.add_handler(CommandHandler("help", help_command))
    # Bind /start to the help menu so first-time users see it immediately.
    app.add_handler(CommandHandler("start", help_command))

    # --- Phase 2 handlers ---
    # NOTE on naming: Telegram's bot_command entity parser only matches
    # `/[a-zA-Z0-9_]+`. Hyphens are NOT valid in slash commands — typing
    # `/suggest-buy` makes Telegram parse it as `/suggest` plus literal text
    # `-buy`. So our canonical command is `/suggest_buy` (underscore).
    app.add_handler(CommandHandler("suggest_buy", suggest_buy_command))
    # Fallback for users who manually type the hyphenated form: route
    # `/suggest-buy` (and `/suggest-buy@botname`) to the same handler via
    # plain-text regex match. This won't appear in Telegram's command
    # autocomplete but will still work if typed.
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/suggest-buy(@\w+)?(\s|$)"),
            suggest_buy_command,
        )
    )

    app.add_error_handler(_on_error)
    return app


def main() -> None:
    """Long-running entrypoint — starts polling and blocks until SIGINT/SIGTERM."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # python-telegram-bot's underlying httpx client is very chatty at INFO;
    # downgrade to WARNING so our own logs stay readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)

    try:
        app = build_application()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)

    LOGGER.info("Quant V6 Telegram bot starting (polling mode)...")
    # `drop_pending_updates=True` discards any messages queued while the bot
    # was offline — prevents replaying stale commands on restart.
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
