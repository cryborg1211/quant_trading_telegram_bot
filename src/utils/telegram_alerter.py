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
        chat_ids = os.getenv("TELEGRAM_CHAT_ID", "").split(",")
        self.chat_id_list = [
            c.strip() for c in chat_ids if c.strip() and c.strip() != "YOUR_CHAT_ID"
        ]
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_signal_alert(self, signal_data: dict) -> None:
        """Build and dispatch an HTML-formatted trade alert."""
        msg = self._build_message(signal_data)
        self._dispatch(msg, ticker=signal_data.get("ticker", "N/A"))

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message(signal_data: dict) -> str:
        date_str = datetime.now().strftime("%d/%m/%Y")

        def esc(key: str, default: str = "N/A") -> str:
            return html.escape(str(signal_data.get(key, default)))

        action = esc("action")
        ticker = esc("ticker")
        price = esc("price")
        horizon = esc("horizon", "N/A")
        sentiment_status = esc("sentiment_status", "Không rõ")
        gemini_summary = esc("gemini_summary", "Không có tin tức đáng kể.")
        model_class = esc("model_class")
        confidence = esc("confidence", "0.0")
        top_pos_features = esc("top_pos_features", "N/A")
        top_neg_features = esc("top_neg_features", "N/A")

        # Build source URL block: loop through raw list (cap 3), with domain label per link
        article_urls: list[str] = signal_data.get("article_urls", []) or []
        if article_urls:
            url_lines = "\n".join(
                f"  - [{html.escape(_domain_label(u))}] {html.escape(u)}"
                for u in article_urls[:3]
            )
        else:
            url_lines = "  Không có tin tức đáng kể"

        return (
            f"🚨 <b>[HỆ THỐNG] BÁO CÁO GIAO DỊCH</b>\n"
            f"📅 <b>Ngày:</b> {date_str}\n"
            f"══════════════════════════════\n\n"
            f"📌 <b>[1] TÍN HIỆU GIAO DỊCH</b>\n"
            f"• <b>Lệnh:</b> {action} <b>{ticker}</b>\n"
            f"• <b>Vùng giá vào:</b> {price}\n"
            f"• <b>Chân trời dự báo:</b> {horizon}\n\n"
            f"📰 <b>[2] PHÂN TÍCH TIN TỨC</b>\n"
            f"• <b>Tâm lý thị trường:</b> {sentiment_status}\n"
            f"• <b>Đánh giá:</b> {gemini_summary}\n"
            f"• <b>Nguồn trích dẫn:</b>\n{url_lines}\n\n"
            f"📈 <b>[3] PHÂN TÍCH ĐỊNH LƯỢNG</b>\n"
            f"• <b>Dự báo:</b> {model_class} (Độ tin cậy: <b>{confidence}%</b>)\n"
            f"• <b>Động lực tăng giá:</b> {top_pos_features}\n"
            f"• <b>Yếu tố rủi ro:</b> {top_neg_features}\n"
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