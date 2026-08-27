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


def format_source_links(urls: list[str] | None, limit: int = 6) -> str:
    """Clean single-line source attribution for the Telegram cards.

    Up to `limit` (default 6) URLs rendered as domain-labelled links:
    'Nguồn tham khảo: <a href="U1">VnExpress</a> · <a href="U2">CafeF</a> · …'
    href is attribute-escaped (quote=True). Shared by the /suggest_buy card,
    the combined report, and the fallback report (single source of truth).
    """
    clean = [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()][:limit]
    if not clean:
        return "Nguồn tham khảo: chưa có liên kết."
    parts = [
        f'<a href="{html.escape(u, quote=True)}">{html.escape(_domain_label(u))}</a>'
        for u in clean
    ]
    return "Nguồn tham khảo: " + " · ".join(parts)


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

    def send_text_to_chat(self, chat_id: str, html_text: str, label: str = "alert") -> None:
        """Send a pre-escaped HTML message to a SINGLE chat_id (not a broadcast).

        Unlike `send_text_alert` (which loops every id in `self.chat_id_list`),
        this targets exactly one recipient - used by the per-user portfolio-guard
        DM where `chat_id == user_id`. Reuses the same `_send_to_one` helper the
        broadcast loop uses, so mock-mode, rate-limit, status-code, and
        per-recipient exception handling are byte-for-byte identical. The caller
        MUST ensure `html_text` is already HTML-escaped.
        """
        self._send_to_one(chat_id, html_text, ticker=label)

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message(signal_data: dict) -> str:
        """Institutional trade card — clean HTML, NO icons / bullets / &nbsp;.

        The news view is a SINGLE integrated analytical paragraph (the LLM's
        `reasoning_vi`); the old Điểm cộng / Điểm trừ / Kết luận three-line block
        (which restated the same points) is gone.  Shared verbatim by the push
        alert AND the interactive /suggest_buy report (_build_combined_report).

        TWO LAYOUTS, ONE FUNCTION (backward compat by field presence):
          • legacy — signal dicts without attribution fields render the
            pre-attribution card byte-for-byte (old tests + old artifacts);
          • attribution — when `_dispatch_signals` supplies the overlay fields
            (garch_scalar / arb_note_vi / risk_tier_*), the card gains the
            unified 4-section contract: Header → Xác suất (model) →
            Lớp giám sát (regime/GARCH/arbitrator) → Triển khai danh mục
            (risk tier + sizing + hold rule).
        """
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")

        def esc(key: str, default: str = "") -> str:
            return html.escape(str(signal_data.get(key, default)))

        def pct(key: str) -> float:
            try:
                return float(signal_data.get(key))
            except (TypeError, ValueError):
                return float("nan")

        ticker = esc("ticker", "N/A")
        price = esc("price", "N/A")
        horizon_label = esc("horizon_label", "T+5")

        # Suggested half-Kelly position size (NAV fraction), shown at the top.
        sw = signal_data.get("suggested_weight")
        try:
            sizing_str = f"{float(sw) * 100:.1f}% NAV" if sw is not None else "N/A"
        except (TypeError, ValueError):
            sizing_str = "N/A"

        # Detected market regime — shown right under the sizing so the user sees WHY
        # the size is what it is (regime 0/7 → 0%, 1/6 → ≤10%, 3 → full Kelly).
        # Omitted entirely when absent (e.g. a pre-regime artifact).
        regime_label = signal_data.get("regime_label")
        regime_id = signal_data.get("market_regime")
        regime_line = ""
        if regime_label and regime_id is not None:
            regime_line = (f"Pha thị trường: <b>{html.escape(str(regime_label))}</b> "
                           f"(Regime {int(regime_id)})\n")

        # Tranche-strategy guidance (artifact `strategy` dict) — hold horizon +
        # optional PT/SL barrier rule.  Absent for legacy half-Kelly artifacts.
        hold_label = signal_data.get("hold_label")
        hold_line = ""
        if hold_label:
            hold_line = f"Nắm giữ: <b>{html.escape(str(hold_label))}</b>\n"
            exit_rule = signal_data.get("exit_rule")
            if exit_rule:
                hold_line += f"Quy tắc thoát: {html.escape(str(exit_rule))}\n"

        # Event-driven rescue banner (status + VN reason) — only for non-standard signals.
        _status = signal_data.get("status")
        _ly_do = signal_data.get("ly_do")
        event_line = ""
        if _status and str(_status) != "MUA":
            event_line = f"⚡ <b>{html.escape(str(_status))}</b>\n{html.escape(str(_ly_do))}\n"

        # Trend odds. Prefer the full 3-class split; fall back to a single
        # "Tăng" figure if only a scalar confidence is available.
        p_up, p_side, p_dn = pct("prob_up"), pct("prob_side"), pct("prob_down")
        if p_up != p_up:  # NaN → no triple, use legacy single confidence
            try:
                p_up = float(signal_data.get("confidence", 0.0))
            except (TypeError, ValueError):
                p_up = 0.0
        trend = [f"Tăng: <b>{p_up:.1f}%</b>"]
        if p_side == p_side:
            trend.append(f"Đi ngang: {p_side:.1f}%")
        if p_dn == p_dn:
            trend.append(f"Giảm: {p_dn:.1f}%")
        trend_line = "  |  ".join(trend)

        # SINGLE integrated analytical paragraph (LLM reasoning_vi). The separate
        # catalyst/risk lines were removed — the paragraph already synthesises them.
        analysis = esc(
            "conclusion",
            signal_data.get("gemini_summary", "Chưa có dữ liệu tin tức đáng kể."),
        )

        source_line = format_source_links(signal_data.get("article_urls", []) or [])

        # ── Attribution fields (unified 4-section contract) ──────────────
        # All optional: absent → legacy layout, no stray "N/A" lines.
        # `gate_margin_vi` replaced `base_decision_vi` on 27-08-26: the old field
        # printed the model's argmax, which is SELL on 361/361 names, so it never
        # varied and read as a per-name warning. The margin actually moves.
        # `base_decision_vi` is still honoured for any caller not yet migrated.
        gate_margin = signal_data.get("gate_margin_vi")
        base_decision = signal_data.get("base_decision_vi")
        garch_scalar = signal_data.get("garch_scalar")
        arb_note = signal_data.get("arb_note_vi")
        tier_label = signal_data.get("risk_tier_label_vi")
        tier_pct = signal_data.get("risk_tier_pct")

        garch_line = ""
        if garch_scalar is not None:
            try:
                _gs = float(garch_scalar)
                # Branch on the ROUNDED value so the label and the number it
                # prints agree by construction. The GARCH leg is a continuous
                # P(Bull) clip, so comparing against 1.0 exactly made a routine
                # 0.997 render as "Đang phanh — hạ tỷ trọng ×1.00" — a brake
                # claim sitting next to a no-brake number.
                _gs_txt = ("Không kích hoạt (×1.00)" if round(_gs, 2) >= 1.00
                           else f"Đang phanh — hạ tỷ trọng ×{_gs:.2f}")
                garch_line = f"Phanh biến động GARCH: <b>{_gs_txt}</b>"
            except (TypeError, ValueError):
                garch_line = ""

        has_attribution = bool(garch_line or arb_note or
                               (tier_label and tier_pct is not None))

        base_line = ""
        if gate_margin:
            base_line = (f"Biên qua cổng: "
                         f"<b>{html.escape(str(gate_margin))}</b>\n")
        elif base_decision:
            base_line = (f"Phân loại gốc của mô hình: "
                         f"<b>{html.escape(str(base_decision))}</b>\n")

        if not has_attribution:
            # Legacy layout — pre-attribution card, unchanged structure.
            return (
                f"<b>KHUYẾN NGHỊ MUA — {ticker}</b>\n"
                f"{horizon_label} Model  |  Khuyến nghị đi vốn: <b>{sizing_str}</b>\n"
                f"{regime_line}"
                f"{hold_line}"
                f"{event_line}"
                f"{date_str}  |  Vùng giá: <b>{price}</b>\n"
                f"\n"
                f"<b>Xác suất xu hướng ({horizon_label})</b>\n"
                f"{trend_line}\n"
                f"{base_line}"
                f"\n"
                f"<b>Nhận định</b>\n"
                f"{analysis}\n"
                f"\n"
                f"{source_line}"
            )

        # ── Operator-decision card (10-08-26 redesign) ────────────────────
        # Ordered to match how the reader actually decides: disqualifiers
        # first, then the two probabilities, then the context that explains
        # any size reduction, then a reference size, then the narrative.
        # Header says ỨNG VIÊN (candidate), not KHUYẾN NGHỊ (recommendation):
        # the operator decides every trade, and the live paperlog has BUY
        # calls averaging -0.69 pct over 43 settled rows, so a confident
        # recommendation header would overstate what the system knows.
        from src.reports.regime_explain import regime_explain_lines  # noqa: PLC0415
        from src.reports.signal_score import (  # noqa: PLC0415
            build_facts,
            score_line,
            warning_lines,
        )

        facts = build_facts(
            p_up_20d=signal_data.get("p_up_20d"),
            tau_20d=signal_data.get("tau_20d"),
            p_up_5d=signal_data.get("p_up_5d"),
            tau_5d=signal_data.get("tau_5d"),
            room_exhausted=signal_data.get("room_exhausted"),
            market_regime=regime_id,
            garch_scalar=garch_scalar,
            breadth=signal_data.get("breadth"),
            sentiment_score=signal_data.get("sentiment_score"),
            mr_fired=signal_data.get("mr_fired"),
        )

        warn_block = ""
        warns = warning_lines(facts, market_regime=regime_id)
        if warns:
            warn_block = ("⚠️ <b>CẢNH BÁO</b>\n"
                          + "\n".join(f"• {html.escape(w)}" for w in warns)
                          + "\n\n")

        # Probabilities, no inline commentary (the score block carries the
        # judgement instead).
        prob_lines = [trend_line]
        p5 = signal_data.get("p_up_5d")
        p20 = signal_data.get("p_up_20d")
        if p5 is not None and p20 is not None:
            try:
                prob_lines = [f"T+5: <b>{float(p5) * 100:.1f}%</b>  |  "
                              f"T+20: <b>{float(p20) * 100:.1f}%</b>"]
            except (TypeError, ValueError):
                pass
        prob_block = ("📈 <b>KHẢ NĂNG TĂNG</b>\n" + "\n".join(prob_lines) + "\n\n")

        # Context — expanded: each regime now explains what it IS and why it
        # forces its size action, plus the other exposure legs.
        ctx_lines: list[str] = []
        ctx_lines.extend(html.escape(ln) for ln in regime_explain_lines(regime_id))
        if garch_line:
            ctx_lines.append(garch_line)
        breadth = signal_data.get("breadth")
        if breadth is not None:
            try:
                ctx_lines.append(
                    f"Độ rộng thị trường: <b>{float(breadth) * 100:.0f}%</b> mã tăng")
            except (TypeError, ValueError):
                pass
        drift = signal_data.get("drift_scalar")
        if drift is not None:
            try:
                _d = float(drift)
                ctx_lines.append(
                    "Xu hướng chỉ số: "
                    + ("đang suy yếu — hạ tỷ trọng ×%.2f" % _d if _d < 1.0
                       else "bình thường"))
            except (TypeError, ValueError):
                pass
        if arb_note:
            ctx_lines.append(f"Trọng tài tin tức: {html.escape(str(arb_note))}")
        mr_fired = signal_data.get("mr_fired")
        if mr_fired is not None:
            ctx_lines.append("Bắt đáy (MR): "
                             + ("<b>đã kích hoạt</b>" if mr_fired else "chưa kích hoạt"))
        if gate_margin:
            ctx_lines.append(f"Biên qua cổng: <b>{html.escape(str(gate_margin))}</b>")
        elif base_decision:
            ctx_lines.append(f"Phân loại gốc của mô hình: "
                             f"<b>{html.escape(str(base_decision))}</b>")
        ctx_block = ""
        if ctx_lines:
            ctx_block = "🛡️ <b>BỐI CẢNH</b>\n" + "\n".join(ctx_lines) + "\n\n"

        # Reference sizing only — capital and max hold, nothing else. Framed
        # as THAM CHIẾU because the system is not sizing the operator's book.
        ref_bits = [f"Đi vốn: <b>{sizing_str}</b>"]
        hold_label_txt = signal_data.get("hold_label")
        if hold_label_txt:
            ref_bits.append(f"Nắm giữ tối đa: <b>{html.escape(str(hold_label_txt))}</b>")
        ref_block = "💰 <b>THAM CHIẾU</b>\n" + "  |  ".join(ref_bits) + "\n\n"

        return (
            f"📊 <b>{ticker} — {price}</b>  [ỨNG VIÊN {horizon_label}]\n"
            f"{date_str}\n"
            f"{event_line}"
            f"\n"
            f"{warn_block}"
            f"{prob_block}"
            f"{ctx_block}"
            f"{ref_block}"
            f"📰 <b>NHẬN ĐỊNH — {score_line(facts)}</b>\n"
            f"{analysis}\n"
            f"\n"
            f"{source_line}"
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, msg: str, ticker: str = "N/A") -> None:
        if not self.chat_id_list:
            LOGGER.warning("[TelegramBot] No chat IDs configured. Alert suppressed for %s.", ticker)
            return

        for chat_id in self.chat_id_list:
            self._send_to_one(chat_id, msg, ticker=ticker)

    def _send_to_one(self, chat_id: str, msg: str, ticker: str = "N/A") -> None:
        """Send ONE HTML message to ONE chat_id. Never raises.

        Extracted verbatim from `_dispatch`'s loop body — the mock-check,
        `requests.post`, status-code logging, `time.sleep(0.5)` rate-limit guard,
        and per-recipient exception logging are byte-for-byte preserved. Shared by
        the broadcast `_dispatch` loop and the single-recipient
        `send_text_to_chat`. The former loop's `continue` becomes an early
        `return` here (same effect: skip the post + skip the sleep in mock mode).
        """
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}
        try:
            if self.bot_token == "YOUR_BOT_TOKEN":
                LOGGER.info("[TelegramBot] MOCK ALERT → %s for %s", chat_id, ticker)
                return

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