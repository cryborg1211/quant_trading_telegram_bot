"""Telegram card contract — institutional, clean, sized.

Guards the exact regressions that shipped: the 'N/A' sizing line, emoji/&nbsp;
clutter, the repeating Điểm cộng/Điểm trừ/Kết luận block, and the source-link
limit.  Imports only the light alerter module (no ML stack), so it runs anywhere.
"""
import re

from src.utils.telegram_alerter import TelegramBot, format_source_links

# Emoji ranges (misc symbols/dingbats + the supplemental emoji planes).
_EMOJI = re.compile("[☀-➿\U0001F300-\U0001FAFF]")


def _card(**overrides) -> str:
    base = dict(
        ticker="HPG", price="25,000 VND",
        prob_up=66.1, prob_side=9.0, prob_down=24.9,
        horizon_label="T+5", suggested_weight=0.20,
        conclusion="Kết luận Tâm lý (Sentiment Score): +0.35. Dòng tin tích cực.",
        article_urls=["https://vnexpress.net/a", "https://cafef.vn/b"],
    )
    base.update(overrides)
    return TelegramBot._build_message(base)


def test_card_is_clean_institutional():
    c = _card()
    assert not _EMOJI.search(c), "emoji leaked into the card"
    assert "&nbsp;" not in c
    assert "•" not in c
    # the repeating pros/cons/conclusion block must be gone
    assert "Điểm cộng" not in c and "Điểm trừ" not in c
    assert "Nhận định" in c
    assert "KHUYẾN NGHỊ MUA — HPG" in c


def test_card_shows_real_sizing_not_na():
    # P(UP)≈0.42 → 6.5% NAV; this is the case that used to render 'N/A'.
    c = _card(suggested_weight=0.065)
    assert "Khuyến nghị đi vốn: <b>6.5% NAV</b>" in c
    assert "N/A" not in c


def test_card_horizon_label_is_dynamic():
    c20 = _card(horizon_label="T+20", suggested_weight=0.125)
    assert "T+20 Model" in c20
    assert "Khuyến nghị đi vốn: <b>12.5% NAV</b>" in c20
    assert "Xác suất xu hướng (T+20)" in c20


def test_card_na_only_when_weight_missing():
    c = _card(suggested_weight=None)
    assert "Khuyến nghị đi vốn: <b>N/A</b>" in c   # explicit, graceful


def test_source_links_clean_and_multiple():
    sl = format_source_links([f"https://s{i}.vn/a" for i in range(6)])
    assert sl.count("<a href") == 6        # >=5 sources, not capped at 2
    assert "🔗" not in sl and not _EMOJI.search(sl)
    assert "Nguồn tham khảo:" in sl


def test_source_links_empty_is_graceful():
    assert "chưa có" in format_source_links([]).lower()


# ── HTML-injection safety (migrated from the retired test_telegram_alerter.py) ──

def test_card_escapes_malicious_ticker():
    c = _card(ticker="<script>")
    assert "&lt;script&gt;" in c and "<script>" not in c


def test_card_escapes_conclusion_html():
    c = _card(conclusion="<b>x</b>")
    assert "&lt;b&gt;x&lt;/b&gt;" in c and "<b>x</b>" not in c


# ── Unified 4-section attribution card (02-07-26) ──────────────────────────

_ATTRIBUTION = dict(
    base_decision_vi="MUA (BUY)",
    regime_action_vi="Không điều chỉnh",
    garch_scalar=0.85,
    arb_note_vi="Không can thiệp",
    risk_tier="RISK_MID",
    risk_tier_pct=60,
    risk_tier_label_vi="TRUNG BÌNH — thận trọng",
    market_regime=3,
    regime_label="Strong Trend",
)


def test_attribution_card_renders_operator_sections():
    """Operator-decision layout (10-08-26 redesign).

    Header is ỨNG VIÊN, not KHUYẾN NGHỊ: the operator decides every trade, so
    the card presents a candidate rather than issuing a recommendation.
    """
    c = _card(**_ATTRIBUTION)
    assert "HPG — 25,000 VND" in c
    assert "[ỨNG VIÊN T+5]" in c
    assert "KHUYẾN NGHỊ MUA" not in c                       # old framing is gone
    assert "<b>KHẢ NĂNG TĂNG</b>" in c
    assert "<b>BỐI CẢNH</b>" in c
    assert "Phanh biến động GARCH: <b>Đang phanh — hạ tỷ trọng ×0.85</b>" in c
    assert "Trọng tài tin tức: Không can thiệp" in c
    assert "Phân loại gốc của mô hình: <b>MUA (BUY)</b>" in c
    assert "<b>THAM CHIẾU</b>" in c
    assert "Đi vốn: <b>20.0% NAV</b>" in c
    assert "ĐIỂM TỔNG:" in c                                # checklist score


def test_attribution_card_explains_the_regime():
    """Regime 3 must say what it IS and why it gets its size treatment —
    a bare label plus a multiplier told the operator nothing."""
    c = _card(**_ATTRIBUTION)
    assert "Xu Hướng Mạnh" in c and "Regime 3" in c
    assert "cho phép tỷ trọng tối đa" in c
    assert "hiệu suất xu hướng cao" in c                    # the detector's criterion


def test_attribution_card_has_no_stray_placeholders():
    c = _card(**_ATTRIBUTION)
    assert "&nbsp;" not in c
    assert "N/A" not in c


def test_attribution_garch_idle_renders_inactive():
    c = _card(**{**_ATTRIBUTION, "garch_scalar": 1.0})
    assert "Phanh biến động GARCH: <b>Không kích hoạt (×1.00)</b>" in c


def test_attribution_fields_are_escaped():
    c = _card(**{**_ATTRIBUTION, "arb_note_vi": "<script>"})
    assert "&lt;script&gt;" in c and "<script>" not in c


def test_legacy_card_unchanged_without_attribution_fields():
    # No attribution fields → legacy layout: sizing stays on the top line,
    # no section headers appear.
    c = _card()
    assert "Khuyến nghị đi vốn: <b>20.0% NAV</b>" in c
    assert "Lớp giám sát" not in c
    assert "Triển khai danh mục" not in c
    assert "Phân loại gốc" not in c


def test_attribution_card_keeps_hold_in_reference_block():
    c = _card(**_ATTRIBUTION, hold_label="30 phiên (đến ~24/07/2026)")
    assert "Nắm giữ tối đa: <b>30 phiên" in c
    # Reference block carries capital + max hold only.
    assert c.index("<b>THAM CHIẾU</b>") < c.index("Nắm giữ tối đa:")


# ── Warnings + score (10-08-26) ─────────────────────────────────────────────

def test_room_exhausted_raises_a_warning():
    c = _card(**_ATTRIBUTION, room_exhausted=True)
    assert "CẢNH BÁO" in c
    assert "room ngoại" in c


def test_t5_only_signal_is_flagged():
    """The T+20 gate does the quality filtering; every losing July-2026
    dispatch came through the T+5-only door, so it must be called out."""
    c = _card(**_ATTRIBUTION, p_up_5d=0.60, tau_5d=0.44,
              p_up_20d=0.40, tau_20d=0.46)
    assert "CẢNH BÁO" in c
    assert "T+5" in c and "T+20 chưa mở" in c


def test_clean_signal_has_no_warning_block():
    c = _card(**_ATTRIBUTION, room_exhausted=False,
              p_up_5d=0.60, tau_5d=0.44, p_up_20d=0.60, tau_20d=0.46)
    assert "CẢNH BÁO" not in c


def test_both_horizon_probabilities_shown_without_commentary():
    c = _card(**_ATTRIBUTION, p_up_5d=0.38, p_up_20d=0.45)
    assert "T+5: <b>38.0%</b>" in c
    assert "T+20: <b>45.0%</b>" in c
