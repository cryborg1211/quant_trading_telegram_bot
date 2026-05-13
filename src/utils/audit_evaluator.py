"""Post-mortem audit of user commands.

Evaluates whether the user's past `/verify` and `/add` queries panned out —
fetches the T0 price (at command time) vs the latest price, computes %
return, then asks Gemini to explain WHY the price moved using recent news.

Public entry:
    run_post_mortem(user_id: str, days: int) -> str
        Returns a Telegram-ready HTML report string. Designed to be called
        from a `asyncio.to_thread(...)` wrapper so it doesn't block the
        bot's event loop.

Coverage caveat:
    `/suggest_buy` is in the user's spec but the audit_log row for that
    command has NO ticker (the result is Top-3 chosen at runtime, never
    persisted per-ticker). It is filtered out here by the `ticker IS NOT
    NULL` predicate. Future enhancement: have `suggest_buy_command` write
    one additional audit_log row per dispatched signal so they appear here.
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)

# Commands recorded with a per-row ticker — the only ones we can audit
# against historical OHLCV. `suggest_buy` is intentionally included for
# forward compatibility; the WHERE clause's `ticker IS NOT NULL` filter
# means today it contributes zero rows.
_AUDITABLE_COMMANDS: tuple[str, ...] = ("verify", "add", "suggest_buy")

# Hard cap so the report stays under Telegram's 4096-char limit and
# Gemini's per-prompt token budget. 10 tickers × ~300 chars each ≈ 3000.
_MAX_TICKERS_PER_REPORT: int = 10

# News-headline budget passed to Gemini for the "why did it move" prompt.
# More than ~5 headlines drowns out the signal in clickbait.
_MAX_HEADLINES_FOR_LLM: int = 5


# ─── Public entry ──────────────────────────────────────────────────────────

def run_post_mortem(user_id: str, days: int) -> str:
    """Build a Telegram-safe HTML post-mortem report for the given user.

    Never raises — every failure mode (no DB, no audit rows, no historical
    prices, Gemini outage) degrades to either an inline warning or a
    Vietnamese explanatory message so the caller can blindly send the
    return value to the user.
    """
    LOGGER.info("[/audit] starting for user_id=%s days=%s", user_id, days)
    if not user_id:
        return "<i>Không xác định được user_id.</i>"
    if days <= 0:
        return "<i>Số ngày phải lớn hơn 0.</i>"

    # 1. Pull audited tickers from audit_log
    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        audited = _fetch_audited_tickers(user_id, days, db)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/audit] DB read failed")
        return f"❌ <b>Lỗi DB:</b> <code>{html.escape(str(exc))}</code>"

    if not audited:
        return (
            f"<b>📊 BÁO CÁO HẬU KIỂM ({days} NGÀY QUA)</b>\n"
            f"══════════════════════\n\n"
            f"<i>Không có lệnh nào trong {days} ngày qua có thể hậu kiểm.</i>\n"
            f"<i>Chỉ các lệnh /verify và /add (có mã cụ thể) được tính.</i>"
        )

    # Cap so the report doesn't bloat.
    if len(audited) > _MAX_TICKERS_PER_REPORT:
        truncated = True
        audited = audited[:_MAX_TICKERS_PER_REPORT]
    else:
        truncated = False

    # 2. Build per-ticker evaluation rows
    rows: list[dict[str, Any]] = []
    for item in audited:
        rows.append(_evaluate_one_ticker(item, days, db))

    return _build_audit_report(rows, days, truncated=truncated)


# ─── DB helpers ───────────────────────────────────────────────────────────

def _fetch_audited_tickers(user_id: str, days: int, db: Any) -> list[dict[str, Any]]:
    """Return unique tickers the user invoked in the auditable commands.

    Each entry: {ticker, first_query_ts, commands (list[str])}.
    Ordered by first_query_ts ASC (oldest first — chronological review).
    """
    since = datetime.now() - timedelta(days=days)
    placeholders = ", ".join(["?"] * len(_AUDITABLE_COMMANDS))
    sql = (
        f"SELECT ticker, MIN(timestamp) AS first_ts, "
        f"       LIST(DISTINCT command) AS cmds "
        f"FROM audit_log "
        f"WHERE user_id = ? "
        f"  AND command IN ({placeholders}) "
        f"  AND ticker IS NOT NULL "
        f"  AND ticker <> '' "
        f"  AND timestamp >= ? "
        f"GROUP BY ticker "
        f"ORDER BY first_ts ASC"
    )
    params: list[Any] = [user_id, *_AUDITABLE_COMMANDS, since]
    rows = db.conn.execute(sql, params).fetchall()
    return [
        {
            "ticker": str(r[0]).upper().strip(),
            "first_query_ts": r[1],
            "commands": list(r[2] or []),
        }
        for r in rows
        if r and r[0]
    ]


def _get_t0_price(ticker: str, ts: Any, db: Any) -> float | None:
    """Close price at-or-just-before the user's query timestamp.

    Uses `stock_ohlcv`: the most recent completed daily candle BEFORE OR ON
    the query date is the closest defensible T0 reference (we cannot use
    the same-day close if the query happened mid-session — that close
    didn't exist yet).
    """
    if isinstance(ts, datetime):
        ts_date = ts.date()
    else:
        ts_date = ts
    row = db.conn.execute(
        "SELECT close FROM stock_ohlcv "
        "WHERE ticker = ? AND date <= ? "
        "ORDER BY date DESC LIMIT 1",
        [ticker, ts_date],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _get_current_price(ticker: str, db: Any) -> float | None:
    """Latest close price for the ticker."""
    row = db.conn.execute(
        "SELECT close FROM stock_ohlcv "
        "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        [ticker],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


# ─── Per-ticker pipeline ──────────────────────────────────────────────────

def _evaluate_one_ticker(
    item: dict[str, Any],
    days: int,
    db: Any,
) -> dict[str, Any]:
    """T0 + current prices, % return, LLM explanation. Never raises."""
    ticker: str = item["ticker"]
    first_ts = item["first_query_ts"]

    try:
        t0 = _get_t0_price(ticker, first_ts, db)
        t_now = _get_current_price(ticker, db)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[/audit] %s price fetch failed: %s", ticker, exc)
        return {"ticker": ticker, "first_ts": first_ts, "error": f"price lookup: {exc}"}

    if t0 is None or t_now is None:
        return {
            "ticker": ticker,
            "first_ts": first_ts,
            "error": "thiếu giá lịch sử trong DB",
        }
    if t0 <= 0:
        return {
            "ticker": ticker,
            "first_ts": first_ts,
            "error": f"giá T0 không hợp lệ ({t0})",
        }

    pct = (t_now - t0) / t0 * 100.0

    try:
        explanation = _explain_move(ticker, days, pct)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[/audit] %s Gemini explain failed: %s", ticker, exc)
        explanation = "(Không có lý giải từ LLM — mạng / quota / no news.)"

    return {
        "ticker": ticker,
        "first_ts": first_ts,
        "t0": t0,
        "t_now": t_now,
        "pct": pct,
        "explanation": explanation,
    }


def _explain_move(ticker: str, days: int, pct: float) -> str:
    """Ask Gemini why the price moved, using recent news as evidence.

    Returns a Vietnamese 1–3 sentence string. The caller wraps in try/except
    for graceful degradation, but most failure modes (no SDK, no API key,
    DNS block, no news) are handled inline here with a safe default string.
    """
    direction_vi = "tăng" if pct >= 0 else "giảm"
    pct_abs = abs(pct)

    # --- Pull recent news via the existing parallel multi-domain scraper ---
    headlines: list[str] = []
    try:
        from src.models.quant_agent_arbitrator import scrape_centralized_news  # noqa: PLC0415
        items = scrape_centralized_news(target_tickers=[ticker])
        for item in items[:_MAX_HEADLINES_FOR_LLM]:
            title = str(item.get("title", "")).strip()
            if title:
                headlines.append(f"- {title}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[/audit] news scrape for %s failed: %s", ticker, exc)
    headlines_text = "\n".join(headlines) if headlines else "(no recent headlines)"

    # --- Build the Gemini prompt per the user's spec ---
    prompt = (
        f"Người dùng đã đánh giá cổ phiếu {ticker} cách đây {days} ngày. "
        f"Từ đó đến nay, giá đã {direction_vi} {pct_abs:.1f}%.\n\n"
        f"Dựa vào các tiêu đề tin tức gần đây dưới đây, hãy giải thích NGẮN GỌN "
        f"(tối đa 3 câu, bằng tiếng Việt) nguyên nhân chính của biến động giá này. "
        f"Phân loại theo MỘT trong: Earnings (KQKD), Macro (vĩ mô), hoặc "
        f"Sentiment (tâm lý / tin tức). Bắt đầu câu trả lời bằng nhãn phân loại.\n\n"
        f"Tin tức gần đây:\n{headlines_text}"
    )

    # --- Call Gemini via the new google-genai SDK (same as arbitrator) ---
    try:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import types as genai_types  # type: ignore[import-not-found]
    except ImportError:
        return "(google-genai SDK chưa cài — bỏ qua phân tích AI.)"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "(GEMINI_API_KEY chưa set — bỏ qua phân tích AI.)"

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").removeprefix("models/")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
        ),
    )
    text = (response.text or "").strip() if hasattr(response, "text") else ""
    return text or "(LLM trả về rỗng.)"


# ─── HTML report builder ──────────────────────────────────────────────────

def _format_move(pct: float) -> str:
    """Visual label for a % return: green up / red down / yellow flat (±0.5%)."""
    if pct > 0.5:
        return f"🟢 TĂNG +{pct:.1f}%"
    if pct < -0.5:
        return f"🔴 GIẢM {pct:.1f}%"
    return f"🟡 ĐI NGANG {pct:+.1f}%"


def _build_audit_report(rows: list[dict[str, Any]], days: int, truncated: bool) -> str:
    """Build the final HTML report shown to the user."""
    header = (
        f"<b>📊 BÁO CÁO HẬU KIỂM ({days} NGÀY QUA)</b>\n"
        f"══════════════════════"
    )

    if not rows:
        return f"{header}\n\n<i>Không có mã nào để hậu kiểm.</i>"

    blocks: list[str] = [header]
    for r in rows:
        ticker = html.escape(r["ticker"])
        if r.get("error"):
            blocks.append(
                f"\n• <b>Mã:</b> {ticker}\n"
                f"• <b>Thực tế:</b> ⚠️ {html.escape(str(r['error']))}"
            )
            continue
        move_label = _format_move(r["pct"])
        explanation = html.escape(str(r.get("explanation", "—")))
        blocks.append(
            f"\n• <b>Mã:</b> {ticker}\n"
            f"• <b>Thực tế:</b> {move_label}\n"
            f"• <b>Nguyên nhân (AI):</b> {explanation}"
        )

    if truncated:
        blocks.append(
            f"\n<i>Đã giới hạn ở {_MAX_TICKERS_PER_REPORT} mã đầu tiên "
            f"(bạn có nhiều hơn trong khoảng thời gian này).</i>"
        )

    return "\n".join(blocks)
