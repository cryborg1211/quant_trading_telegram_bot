"""
Telegram alerter for Quant V6 trade signals.

signal_data expected keys:
- action          (str)       : "MUA" | "BÁN" | "GIỮ"
- ticker          (str)       : e.g. "FPT"
- price           (str)       : formatted price string, e.g. "136,000 VND"
- horizon         (str)       : "5 ngày" | "20 ngày"
- sentiment_score (float)     : -1.0 … 1.0
- sentiment_status(str)       : human-readable status
- gemini_summary  (str)       : Vietnamese LLM reasoning
- article_urls    (list[str]) : raw source URL list (up to 3); formatter loops explicitly
- model_class     (str)       : Stacking GBDT model label (e.g. "Stacking GBDT 5d: Tăng (UP)")
- confidence      (float)     : 0–100
- top_pos_features(str)       : human-readable positive drivers
- top_neg_features(str)       : human-readable risk factors
"""

import html
import logging
import os
import time
from datetime import datetime
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger(__name__)

_DOMAIN_LABELS: dict[str, str] = {
    "cafef.vn": "CafeF",
    "vietstock.vn": "VietStock",
    "tinnhanhchungkhoan.vn": "TinNhanhChungKhoan",
    "ndh.vn": "NDH",
    "vneconomy.vn": "VnEconomy",
    "baodautu.vn": "BaoDauTu",
    "theleader.vn": "TheLeader",
    "vnbusiness.vn": "VnBusiness",
    "dantri.com.vn": "DanTri",
    "tuoitre.vn": "TuoiTre",
    "vnexpress.net": "VnExpress",
}


def _domain_label(url: str) -> str:
    """Return a short display label for a URL domain."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        domain = ""
    return next((v for k, v in _DOMAIN_LABELS.items() if k in domain), domain or "Nguồn")


class TelegramBot:
    def __init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
        # Split-ID env (TELEGRAM_CHAT_ID_1 = Admin, _2 = User). Both receive
        # system broadcasts; legacy comma-separated TELEGRAM_CHAT_ID is kept
        # as a fallback for older deployments.
        ids: list[str] = []
        for _k in ("TELEGRAM_CHAT_ID_1", "TELEGRAM_CHAT_ID_2"):
            _v = (os.getenv(_k) or "").strip()
            if _v and _v != "YOUR_CHAT_ID":
                ids.append(_v)
        if not ids:
            ids = [
                c.strip()
                for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",")
                if c.strip() and c.strip() != "YOUR_CHAT_ID"
            ]
        self.chat_id_list = ids
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_signal_alert(self, signal_data: dict) -> None:
        """Build and dispatch an HTML-formatted trade alert."""
        msg = self._build_message(signal_data)
        self._dispatch(msg, ticker=signal_data.get("ticker", "N/A"))

    def send_text_alert(self, html_text: str, label: str = "alert") -> None:
        """Broadcast an arbitrary HTML-safe message to every chat ID in env.

        Used for system-level notifications (pipeline crash, manual ops alert)
        where no structured signal_data exists. The caller MUST ensure
        `html_text` is already HTML-escaped — this method does no escaping
        itself, only forwards to `_dispatch` with parse_mode=HTML.
        """
        self._dispatch(html_text, ticker=label)

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message(signal_data: dict) -> str:
        """Plain-Vietnamese trade card. NO technical jargon — a reader
        understands the trend odds and the news view instantly without
        knowing the model internals.
        """
        date_str = datetime.now().strftime("%d/%m/%Y")

        def esc(key: str, default: str = "") -> str:
            return html.escape(str(signal_data.get(key, default)))

        def pct(key: str) -> float:
            try:
                return float(signal_data.get(key))
            except (TypeError, ValueError):
                return float("nan")

        ticker = esc("ticker", "N/A")
        price = esc("price", "N/A")
        status_label = esc("status_label", "CHẤP NHẬN TÍN HIỆU")

        # Trend odds (5 trading days). Prefer the full 3-class split; fall
        # back to a single "Cửa Tăng" line if only confidence is available.
        p_up, p_side, p_dn = pct("prob_up"), pct("prob_side"), pct("prob_down")
        if p_up != p_up:  # NaN → no triple, use legacy single confidence
            try:
                p_up = float(signal_data.get("confidence", 0.0))
            except (TypeError, ValueError):
                p_up = 0.0
        trend_lines = [f"• Cửa Tăng:&nbsp;&nbsp;<b>{p_up:.1f}%</b>"]
        if p_side == p_side:
            trend_lines.append(f"• Đi Ngang:&nbsp;&nbsp;{p_side:.1f}%")
        if p_dn == p_dn:
            trend_lines.append(f"• Cửa Giảm:&nbsp;&nbsp;{p_dn:.1f}%")

        plus = esc("plus_points", "Không có yếu tố tích cực nổi bật.")
        minus = esc("minus_points", "Không có rủi ro nổi bật.")
        conclusion = esc(
            "conclusion",
            signal_data.get("gemini_summary", "Chưa có dữ liệu tin tức."),
        )

        article_urls: list[str] = signal_data.get("article_urls", []) or []
        if article_urls:
            url_lines = "\n".join(
                f"  • [{html.escape(_domain_label(u))}] {html.escape(u)}"
                for u in article_urls[:3]
            )
        else:
            url_lines = "  • Không có tin tức đáng kể."

        trend_block = "\n".join(trend_lines)
        return (
            f"🟢 <b>KHUYẾN NGHỊ MUA — {ticker}</b>\n"
            f"📅 {date_str}  •  Vùng giá: <b>{price}</b>\n"
            f"\n"
            f"📊 <b>Đánh giá xu hướng (5 ngày tới)</b>\n"
            f"{trend_block}\n"
            f"\n"
            f"✅ <b>Trạng thái: {status_label}</b>\n"
            f"\n"
            f"📰 <b>Điểm tin tức &amp; Tâm lý</b>\n"
            f"• 👍 <b>Điểm cộng:</b> {plus}\n"
            f"• 👎 <b>Điểm trừ:</b> {minus}\n"
            f"• 📌 <b>Kết luận:</b> {conclusion}\n"
            f"\n"
            f"🔗 <b>Nguồn tham khảo:</b>\n{url_lines}\n"
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, msg: str, ticker: str = "N/A") -> None:
        if not self.chat_id_list:
            LOGGER.warning("[TelegramBot] No chat IDs configured. Alert suppressed for %s.", ticker)
            return

        for chat_id in self.chat_id_list:
            payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}
            try:
                if self.bot_token == "YOUR_BOT_TOKEN":
                    LOGGER.info("[TelegramBot] MOCK ALERT → %s for %s", chat_id, ticker)
                    continue

                resp = requests.post(self.base_url, json=payload, timeout=10)
                if resp.status_code != 200:
                    LOGGER.warning(
                        "[TelegramBot] Failed to send to %s. HTTP %s: %s",
                        chat_id, resp.status_code, resp.text[:200],
                    )
                else:
                    LOGGER.info("[TelegramBot] Alert sent → %s for %s", chat_id, ticker)

                time.sleep(0.5)  # Telegram rate-limit guard

            except Exception as exc:  # noqa: BLE001
                LOGGER.error("[TelegramBot] Exception sending to %s: %s", chat_id, exc)