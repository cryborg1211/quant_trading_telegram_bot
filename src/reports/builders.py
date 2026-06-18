"""Report-builder functions extracted from main.py (Phase 1 structural debt).

These are pure HTML-string builders with no orchestration logic.
Dependencies: config.settings (CONFIG), src.utils.telegram_alerter (TelegramBot,
format_source_links), numpy, html, re, datetime.
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any

import numpy as np

from config.settings import CONFIG
from src.utils.telegram_alerter import TelegramBot, format_source_links

# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

FEATURE_HUMAN_NAMES: dict[str, str] = {
    # --- Raw Alpha360 OHLCV base keys (used as lag-column base labels) ---
    "close": "Nền giá đóng cửa",
    "open": "Giá mở cửa",
    "high": "Mức đỉnh giá",
    "low": "Mức đáy giá",
    "vwap": "Giá trung bình gia quyền (VWAP)",  # kept for backward compat with old artifacts
    "hlc3": "Giá HLC3 (Trung bình Cao-Thấp-Đóng)",
    "volume": "Khối lượng giao dịch",
    # RSI variants
    "rsi_14": "RSI 14 ngày",
    "rsi_7": "RSI 7 ngày",
    "rsi_21": "RSI 21 ngày",
    # MACD
    "macd": "MACD",
    "macd_signal": "MACD Signal",
    "macd_hist": "MACD Histogram",
    # Bollinger
    "bb_upper": "Bollinger trên",
    "bb_lower": "Bollinger dưới",
    "bb_width": "Độ rộng Bollinger",
    "bb_pct": "% Bollinger",
    # Volume composites
    "volume_ratio_5": "Tỷ lệ khối lượng 5 ngày",
    "volume_ratio_20": "Tỷ lệ khối lượng 20 ngày",
    "volume_ma_5": "Khối lượng TB 5 ngày",
    "volume_ma_20": "Khối lượng TB 20 ngày",
    "obv": "OBV (Khối lượng cân bằng)",
    "obv_ma": "OBV trung bình",
    # Moving averages
    "sma_5": "Đường MA 5 ngày",
    "sma_10": "Đường MA 10 ngày",
    "sma_20": "Đường MA 20 ngày",
    "sma_50": "Đường MA 50 ngày",
    "ema_12": "Đường EMA 12 ngày",
    "ema_26": "Đường EMA 26 ngày",
    # Returns
    "return_1d": "Lợi nhuận 1 ngày",
    "return_5d": "Lợi nhuận 5 ngày",
    "return_20d": "Lợi nhuận 20 ngày",
    # Volatility
    "volatility_5": "Biến động giá 5 ngày",
    "volatility_20": "Biến động giá 20 ngày",
    "atr_14": "ATR 14 ngày",
    # Price ratios
    "close_to_high_52w": "Giá so với đỉnh 52 tuần",
    "close_to_low_52w": "Giá so với đáy 52 tuần",
    # Momentum
    "momentum_10": "Động lượng 10 ngày",
    "roc_10": "Tốc độ thay đổi giá (ROC 10)",
    "williams_r": "Williams %R",
    "cci_20": "CCI 20 ngày",
    # Stochastic
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    # Legacy macro keys (direct column names)
    "vnindex_return": "Biến động VN-Index",
    "usd_vnd_change": "Biến động tỷ giá USD/VND",
    "gold_change": "Biến động giá vàng",
    "oil_change": "Biến động giá dầu",
}

# Professional Vietnamese labels for macro_ prefixed inner keys
# e.g. macro_sp500_close -> inner "sp500_close" -> looked up here first
_MACRO_INNER_NAMES: dict[str, str] = {
    "sp500_close": "Diễn biến chứng khoán Mỹ (S&P 500)",
    "sp500_return": "Lợi nhuận S&P 500",
    "dxy_close": "Chỉ số sức mạnh đồng USD (DXY)",
    "dxy": "Chỉ số sức mạnh đồng USD (DXY)",
    "usd_vnd": "Tỷ giá USD/VND",
    "usd_vnd_change": "Biến động tỷ giá USD/VND",
    "gold_close": "Giá vàng thế giới",
    "gold_change": "Biến động giá vàng",
    "oil_close": "Giá dầu thô (WTI)",
    "oil_change": "Biến động giá dầu",
    "vix_close": "Chỉ số sợ hãi thị trường (VIX)",
    "vix": "Chỉ số biến động thị trường (VIX)",
    "fed_rate": "Lãi suất Fed",
    "cpi": "Lạm phát (CPI)",
    "vnindex_return": "Biến động VN-Index",
    "vnindex_close": "VN-Index",
    "interbank_rate": "Lãi suất liên ngân hàng",
    "interbank_on_rate": "Lãi suất liên ngân hàng qua đêm (ON)",
    "vnibor": "Lãi suất liên ngân hàng 1 tháng (VNIBOR 1M)",
    "inflation_yoy": "Lạm phát YoY (VN CPI)",
    "deposit_rate": "Lãi suất tiền gửi",
}

# Regex: match trailing numeric suffix (Alpha360 lag index, e.g. close_38, vwap_12)
_NUMERIC_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")
# Regex: match macro_ prefix (e.g. macro_sp500_close, macro_vnindex_return)
_MACRO_PREFIX_RE = re.compile(r"^macro_(.+)$", re.IGNORECASE)

_REPORT_SEPARATOR = "\n\n══════════════════════════════\n\n"

_SELL_DECISION = 0  # arbitrator class label for DOWN / SELL

_MR_SELL_VETO = (
    "⚠️ <b>[CẢNH BÁO BÁN ĐÚNG ĐÁY: Mã này đang rơi vào vùng hoảng loạn "
    "tột độ, xác suất cao sẽ có nhịp hồi chữ V. Hạn chế bán tháo lúc "
    "này!]</b>"
)

# SHORT horizon rendered in report copy ("Đánh giá xu hướng (N ngày tới)").
# The `_5d`-named vars/labels below mean "short horizon" — the artifact behind
# them is T+5 (recovered 16-06-26; the short model stays verify-only role).
SHORT_HORIZON_DAYS: int = 5

# Class-label -> display text mappings for the verify report.
_VERIFY_5D_PRED_LABELS: dict[int, str] = {
    0: "🔴 Giảm (DOWN)",
    1: "🟡 Đi ngang (SIDE)",
    2: "🟢 Tăng (UP)",
}
_VERIFY_20D_PRED_LABELS: dict[int, str] = {
    0: "Giảm",
    1: "Đi ngang",
    2: "Tăng",
}
_VERIFY_VERDICT_LABELS: dict[int, str] = {
    0: "🔴 <b>BÁN (SELL)</b>",
    1: "🟡 <b>GIỮ (HOLD)</b>",
    2: "🟢 <b>MUA / GIỮ (BUY/HOLD)</b>",
}

_REBALANCE_PRED_LABELS: dict[int, str] = {0: "🔴 Giảm", 1: "🟡 Đi ngang", 2: "🟢 Tăng"}

# ---------------------------------------------------------------------------
# Functions (moved from main.py, in source order)
# ---------------------------------------------------------------------------


def _humanize_feature(feat: str) -> str:
    """
    Convert raw Alpha360/stacking feature name to professional Vietnamese trading label.

    Resolution order:
    1. Exact match in FEATURE_HUMAN_NAMES          (e.g. rsi_14 -> "RSI 14 ngay")
    2. Macro prefix via _MACRO_INNER_NAMES          (e.g. macro_sp500_close -> "Dien bien CK My (S&P 500)")
    3. Explicit _lag_N suffix                       (e.g. rsi_14_lag_3 -> "RSI 14 ngay cach day 3 phien")
    4. Alpha360 numeric suffix (OHLCV lag columns)  (e.g. vwap_12 -> "Gia VWAP cach day 12 phien")
    5. Title-case fallback
    """
    # 1. Exact match
    if feat in FEATURE_HUMAN_NAMES:
        return FEATURE_HUMAN_NAMES[feat]

    # 2. Macro prefix — use professional Vietnamese names, not raw title-case
    m2 = _MACRO_PREFIX_RE.match(feat)
    if m2:
        inner = m2.group(1)
        # Try _MACRO_INNER_NAMES first, then FEATURE_HUMAN_NAMES, then title-case fallback
        label = _MACRO_INNER_NAMES.get(inner) or FEATURE_HUMAN_NAMES.get(inner)
        if label:
            return label
        return _MACRO_INNER_NAMES.get(inner, inner.replace("_", " ").title())

    # 3. Explicit _lag_N suffix (e.g. rsi_14_lag_3)
    if "_lag_" in feat:
        base_feat, lag = feat.rsplit("_lag_", 1)
        label = FEATURE_HUMAN_NAMES.get(base_feat, base_feat.replace("_", " ").title())
        return f"{label} cách đây {lag} phiên"

    # 4. Alpha360 numeric suffix (e.g. close_38, vwap_12, norm_vwap_5)
    m = _NUMERIC_SUFFIX_RE.match(feat)
    if m:
        base_feat, lag_idx = m.group(1), m.group(2)
        clean_base = base_feat.removeprefix("norm_")
        label = FEATURE_HUMAN_NAMES.get(clean_base) or FEATURE_HUMAN_NAMES.get(base_feat)
        if not label:
            label = clean_base.replace("_", " ").title()
        return f"{label} cách đây {lag_idx} phiên"

    # 5. Fallback
    return feat.replace("_", " ").title()


def _build_feature_explanation(model: Any, selected_features: list[str], top_k: int = 3) -> tuple[str, str]:
    """
    Use available tree-model feature_importances_ as SHAP-lite fallback for live alerts.
    Produces stable, human-readable drivers instead of raw column names.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "Không có feature importance từ mô hình", "Theo dõi rủi ro tin tức/vĩ mô"

    arr = np.asarray(importances, dtype=np.float64)
    if arr.size == 0 or len(selected_features) == 0:
        return "Không có feature importance từ mô hình", "Theo dõi rủi ro tin tức/vĩ mô"

    n = min(arr.size, len(selected_features))
    arr = arr[:n]
    feats = selected_features[:n]
    order = np.argsort(arr)[::-1]

    top_positive = [_humanize_feature(feats[i]) for i in order[:top_k] if arr[i] > 0]
    top_risk = [_humanize_feature(feats[i]) for i in order[-min(top_k, n):][::-1]]

    pos_text = ", ".join(top_positive) if top_positive else "Không có động lực nổi bật"
    risk_text = ", ".join(top_risk) if top_risk else "Không có yếu tố rủi ro nổi bật"
    return f"Động lực: {pos_text}", f"Rủi ro: {risk_text}"


def _format_sentiment_status(sentiment_data: dict[str, Any]) -> str:
    """Differentiate true neutral from no-news/timeout fallback."""
    source_urls = sentiment_data.get("source_urls", []) or []
    score = float(sentiment_data.get("sentiment_score", 0.0) or 0.0)
    reason = str(sentiment_data.get("reasoning_vi", ""))

    if len(source_urls) == 0 and score == 0.0:
        if "timeout" in reason.lower() or "không có tin" in reason.lower() or "no news" in reason.lower():
            return "Không có tin tức / Timeout"
        return "Không có tin tức / Timeout"

    if score > 0.2:
        label = "Tích cực"
    elif score < -0.2:
        label = "Tiêu cực"
    else:
        label = "Trung tính"
    return f"{label} ({score:+.2f})"


def _build_combined_report(signal_data_list: list[dict]) -> str:
    """Concatenate per-ticker Telegram messages into one HTML chat report.

    Reuses `TelegramBot._build_message()` (already HTML-escapes every dynamic
    field) so the combined string is safe to send with `parse_mode=HTML`.

    Returns "" when the list is empty so callers can short-circuit on
    "no signals" with a single truthiness check.
    """
    if not signal_data_list:
        return ""
    parts = [TelegramBot._build_message(sd) for sd in signal_data_list]
    return _REPORT_SEPARATOR.join(parts)


def _smart_truncate(text: str, limit: int = 300) -> str:
    """Word-aware truncation that never splits a word in half.

    Operates on RAW text — callers must `html.escape()` the RESULT, never the
    other way round — so it is impossible to sever an HTML entity (the
    `html.escape(...)[:500]` anti-pattern cut `&amp;` -> `&am`, which Telegram
    rejects with a parse error).  Returns `text` unchanged when within `limit`;
    otherwise trims to the last whole word and appends a single-glyph ellipsis.
    """
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    word_cut = cut.rsplit(" ", 1)[0].rstrip()   # back off to the last whole word
    return (word_cut if word_cut else cut.rstrip()) + "…"


def _build_fallback_observability_report_vi(
    fallback_tickers: list[str],
    stacking_predictions_5d: dict,
    all_sentiments: dict,
    fallback_reasons: dict,
    mr_scores: dict | None = None,
    live_prices: dict | None = None,
) -> str:
    """Vietnamese 'weak-market' observability report (HTML, the bot's
    Telegram parse mode).

    These tickers are MONITORING ONLY — not trade signals. The caller
    (`daily_inference`) returns this BEFORE `run_trade_execution`, so there
    is no portfolio / RL / dispatch side effect. The header and per-ticker
    flag make it impossible to mistake these for actionable BUYs.
    """
    out = [
        "<b>[⚠️ CẢNH BÁO: THỊ TRƯỜNG XẤU - KHÔNG CÓ ĐIỂM MUA AN TOÀN]</b>",
        "<i>⛔ KHÔNG GIAO DỊCH — danh sách dưới đây chỉ để theo dõi thị "
        "trường. Hôm nay không mã nào đủ tốt để vào lệnh.</i>",
        "",
    ]
    for i, t in enumerate(fallback_tickers, 1):
        p = stacking_predictions_5d.get(t, [0.0, 0.0, 0.0])
        p_dn, p_sd, p_up = p[0] * 100.0, p[1] * 100.0, p[2] * 100.0
        s = all_sentiments.get(t, {}) or {}
        try:
            score = float(s.get("sentiment_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        reason_vi = (
            s.get("reasoning_vi")
            or s.get("reasoning")
            or "Chưa có tin tức đáng chú ý."
        )
        why = fallback_reasons.get(
            t, "Cửa tăng quá thấp so với rủi ro thị trường chung."
        )
        # Trend gate rejected ALL of these (that's why we're in fallback),
        # so the only actionable tag here is whether the knife-catch
        # sub-model fired on extreme panic.
        mr = (mr_scores or {}).get(t) or {}
        tag = " [\U0001f52a BẮT ĐÁY]" if mr.get("fired") else ""
        _px = (live_prices or {}).get(t)
        price_str = f"{_px:,.0f} VND" if _px else "N/A"
        out += [
            f"<b>{i}. {html.escape(t)}{tag}</b>",
            f"   • <b>Giá hiện tại:</b> {html.escape(price_str)}",
            f"   • <b>Đánh giá xu hướng (5 ngày tới):</b> "
            f"Cửa Tăng <b>{p_up:.1f}%</b> | Đi Ngang {p_sd:.1f}% | "
            f"Cửa Giảm {p_dn:.1f}%",
            f"   • <b>Trạng thái:</b> ❌ HỦY BỎ TÍN HIỆU"
            + ("  →  \U0001f52a <b>nhưng MR phát hiện vùng bắt đáy!</b>"
               if mr.get("fired") else ""),
            f"   • <b>Lý do:</b> {html.escape(why)}",
            f"   • <b>Tin tức &amp; Tâm lý:</b> {html.escape(_smart_truncate(reason_vi, 800))}",
            f"   • {format_source_links(s.get('source_urls', []) or [])}",
            "",
        ]
    # Fix the footer-collision bug: guarantee exactly ONE blank line
    # (clean \n\n) between the last sentiment block and the footer.
    while out and out[-1] == "":
        out.pop()
    out.append("")
    out.append(
        "<i>Đây là chế độ theo dõi khi thị trường yếu — KHÔNG phải khuyến "
        "nghị MUA. Hệ thống sẽ tự động báo khi có điểm mua an toàn.</i>"
    )
    return "\n".join(out)


def _build_sell_hold_report(
    holding_tickers: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    live_exec_prices: dict,
    missing_tickers: list[str] | None = None,
    mr_scores: dict | None = None,
) -> str:
    """Build the HTML BAN/GIU digest for /suggest_sell.

    Every ticker passed in `holding_tickers` (that has a prediction) gets one
    block. `missing_tickers` lists holdings that could not be evaluated
    (delisted, illiquid, or absent from the live universe).
    """
    predictions_5d = stacking_predictions.get("5d", {})
    parts: list[str] = []

    for ticker in holding_tickers:
        if ticker not in predictions_5d:
            continue

        decision = final_decisions.get(ticker)
        sentiment = all_sentiments.get(ticker, {})
        sentiment_score = float(sentiment.get("sentiment_score", 0.0) or 0.0)
        sentiment_status = _format_sentiment_status(sentiment)
        reasoning = _smart_truncate(str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể.")), 400)
        confidence_5d = round(predictions_5d.get(ticker, [0, 0, 0])[2] * 100, 2)
        price = live_exec_prices.get(ticker)
        price_str = f"{price:,.0f} VND" if price else "N/A"

        if decision == _SELL_DECISION:
            verdict = "\U0001f534 <b>NÊN BÁN</b>"
        elif decision == 2:
            verdict = "\U0001f7e2 <b>GIỮ TIẾP (xu hướng còn tăng)</b>"
        else:
            verdict = "\U0001f7e1 <b>GIỮ THẬN TRỌNG (đang đi ngang)</b>"

        # MR VETO: the trend model says SELL, but the knife-catch model fired.
        mr = (mr_scores or {}).get(ticker) or {}
        veto_line = ""
        if decision == _SELL_DECISION and mr.get("fired"):
            veto_line = f"{_MR_SELL_VETO}\n"

        # Plain-language target / trailing-stop from the standing risk rules.
        tp = CONFIG.trading.take_profit_pct   # e.g. +0.15
        sl = CONFIG.trading.stop_loss_pct     # e.g. -0.07
        if price:
            target_str = f"{price * (1.0 + tp):,.0f} VND (+{tp * 100:.0f}%)"
            stop_str = f"{price * (1.0 + sl):,.0f} VND ({sl * 100:.0f}%)"
        else:
            target_str = stop_str = "N/A"

        source_urls = (sentiment.get("source_urls", []) or [])[:6]
        if source_urls:
            url_lines = "\n".join(f"  • {html.escape(u)}" for u in source_urls)
        else:
            url_lines = "  • Không có tin tức đáng kể."

        block = (
            f"\U0001f4cc <b>{html.escape(ticker)}</b> — giá hiện tại {html.escape(price_str)}\n"
            f"{veto_line}"
            f"• <b>Khuyến nghị:</b> {verdict}\n"
            f"• <b>Đánh giá xu hướng ({SHORT_HORIZON_DAYS} ngày tới):</b> Cửa Tăng "
            f"<b>{confidence_5d}%</b>\n"
            f"• \U0001f3af <b>Mục tiêu chốt lời:</b> {target_str}\n"
            f"• \U0001f6e1️ <b>Ngưỡng cắt lỗ:</b> {stop_str}\n"
            f"• <b>Tin tức &amp; Tâm lý:</b> {html.escape(sentiment_status)} — "
            f"{html.escape(reasoning)}\n"
            f"• <b>Nguồn tham khảo:</b>\n{url_lines}"
        )
        parts.append(block)

    header = (
        f"\U0001f4bc <b>[HỆ THỐNG] BÁN/GIỮ DANH MỤC</b>\n"
        f"\U0001f4c5 <b>Ngày:</b> {datetime.now().strftime('%d/%m/%Y')}\n"
        f"══════════════════════════════"
    )

    if not parts:
        body = "<i>Không có ticker nào trong danh mục được mô hình đánh giá.</i>"
    else:
        body = _REPORT_SEPARATOR.join(parts)

    footer = ""
    if missing_tickers:
        escaped_missing = html.escape(", ".join(missing_tickers))
        footer = (
            f"\n\n⚠️ <i>Không có dữ liệu live cho:</i> "
            f"<code>{escaped_missing}</code>"
        )

    return f"{header}\n\n{body}{footer}"


def _mr_state_line(mr_state: dict | None) -> str:
    """Plain-VN MR bottom-catch state for the /verify dual output."""
    if not mr_state:
        return "\U0001f52a <b>Trạng thái Bắt đáy:</b> Không khả dụng"
    if mr_state.get("fired"):
        return (
            "\U0001f52a <b>Trạng thái Bắt đáy:</b> "
            "\U0001f6a8 <b>CẢNH BÁO HOẢNG LOẠN</b> — cổ phiếu đang ở vùng bán tháo "
            "cực đoan, xác suất cao có nhịp hồi chữ V."
        )
    return (
        "\U0001f52a <b>Trạng thái Bắt đáy:</b> Chưa đạt "
        "(chưa rơi vào vùng hoảng loạn tột độ)"
    )


def _build_verify_report(
    ticker: str,
    decision: int | None,
    sentiment: dict,
    stacking_5d: list[float],
    stacking_20d: list[float],
    live_exec_price: float | None,
    mr_state: dict | None = None,
) -> str:
    """Build the HTML verification report for /verify.

    Every dynamic field is `html.escape`d. Structural tags (<b>, <code>, <i>)
    are constants. Output is safe to send with parse_mode=HTML.
    """
    # 5d distribution
    p_down, p_side, p_up = stacking_5d[0], stacking_5d[1], stacking_5d[2]
    pred_5d_idx = max(range(3), key=lambda i: stacking_5d[i])
    pred_5d_label = _VERIFY_5D_PRED_LABELS.get(pred_5d_idx, str(pred_5d_idx))
    confidence_5d = round(stacking_5d[pred_5d_idx] * 100, 2)

    # 20d distribution (compact)
    pred_20d_idx = max(range(3), key=lambda i: stacking_20d[i])
    pred_20d_label = _VERIFY_20D_PRED_LABELS.get(pred_20d_idx, str(pred_20d_idx))
    confidence_20d = round(stacking_20d[pred_20d_idx] * 100, 2)

    # Sentiment
    sent_score = float(sentiment.get("sentiment_score", 0.0) or 0.0)
    sent_status = _format_sentiment_status(sentiment)
    sent_reasoning = _smart_truncate(str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể.")), 600)

    # Final arbitrator verdict (decision integer 0/1/2)
    verdict_html = _VERIFY_VERDICT_LABELS.get(decision, "<i>(chưa có verdict)</i>")

    # Price (VN price normalization already applied by `_get_live_exec_prices`)
    price_str = f"{live_exec_price:,.0f} VND" if live_exec_price else "N/A"

    # Source URLs (already populated from ground-truth tracker in arbitrator)
    source_urls = (sentiment.get("source_urls", []) or [])[:3]
    if source_urls:
        url_lines = "\n".join(f"  - {html.escape(u)}" for u in source_urls)
    else:
        url_lines = "  Không có tin tức đáng kể"

    return (
        f"\U0001f50d <b>[KIỂM ĐỊNH] {html.escape(ticker)}</b>\n"
        f"\U0001f4c5 <b>Ngày:</b> {datetime.now().strftime('%d/%m/%Y')}\n"
        f"══════════════════════════════\n\n"
        f"\U0001f4b5 <b>Giá hiện tại:</b> {html.escape(price_str)}\n\n"
        f"\U0001f4ca <b>Đánh giá Xu hướng ({SHORT_HORIZON_DAYS} ngày tới)</b>\n"
        f"• Cửa Tăng: <b>{p_up * 100:.1f}%</b> | Đi Ngang: {p_side * 100:.1f}% "
        f"| Cửa Giảm: {p_down * 100:.1f}%\n"
        f"• Nhận định {SHORT_HORIZON_DAYS} ngày: {pred_5d_label}\n"
        f"• Nhận định 20 ngày: {pred_20d_label} ({confidence_20d}%)\n\n"
        f"{_mr_state_line(mr_state)}\n\n"
        f"\U0001f4f0 <b>Tin tức &amp; Tâm lý</b>\n"
        f"• Đánh giá: {html.escape(sent_status)} (điểm {sent_score:+.2f})\n"
        f"• Phân tích: {html.escape(sent_reasoning)}\n\n"
        f"\U0001f3af <b>Kết luận tổng hợp:</b> {verdict_html}\n\n"
        f"\U0001f517 <b>Nguồn tham khảo:</b>\n{url_lines}"
    )


def _build_rebalance_report(holdings_context: list[dict], advice: str) -> str:
    """Build Telegram-safe HTML for the /rebalance command output.

    Every dynamic field is html.escape'd. Structural tags are constants.
    """
    lines = []
    for h in holdings_context:
        ticker = html.escape(str(h.get("ticker", "?")))
        pnl_pct = float(h.get("pnl_pct", 0.0))
        pred_label = html.escape(str(h.get("pred_label", "N/A")))
        sign = "+" if pnl_pct >= 0 else ""
        icon = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
        lines.append(f"• {icon} <b>{ticker}</b>: {sign}{pnl_pct:.1f}% | {pred_label}")

    holdings_block = "\n".join(lines) if lines else "• Danh mục trống."

    return (
        "<b>⚖️ TƯ VẤN CƠ CẤU DANH MỤC</b>\n"
        "══════════════════════\n"
        f"<b>• Đang nắm giữ:</b>\n{holdings_block}\n\n"
        f"<b>\U0001f4ca Đề xuất AI:</b>\n{html.escape(advice)}"
    )
