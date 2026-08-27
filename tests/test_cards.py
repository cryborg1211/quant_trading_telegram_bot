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
    gate_margin_vi="+0.41pp trên cổng (42.4% vs 42.0%)",
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
    assert "Biên qua cổng: <b>+0.41pp trên cổng (42.4% vs 42.0%)</b>" in c
    assert "Phân loại gốc" not in c, (
        "the argmax line is retired — it read SELL on 361/361 names, so it "
        "never varied and read as a per-name warning")
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


# ── Gate margin replaces the argmax line (27-08-26) ────────────────────────
#
# `base_decision_vi` rendered the model's ARGMAX class and was CONSTANT: measured
# on the live artifact, argmax == SELL for 361 of 361 scored names (0 HOLD,
# 0 BUY). Even the strongest name in the market (p_up 0.4740) still has p_down
# 0.5169. Same fact that forced arbitrator_entry_mode "gate" -> "veto" on 12-08,
# where requiring argmax == UP produced ZERO buys in 920 days.
#
# A field that says "BÁN (SELL)" on every card every day carries no information
# and actively misleads - the operator reads it as a warning about that specific
# name and distrusts every candidate. The margin varies and answers the real
# question: comfortably through, or scraping the floor?


def test_gate_margin_formats_both_sides_and_the_delta():
    from main import _gate_margin_vi
    assert _gate_margin_vi(0.4241, 0.42) == "+0.41pp trên cổng (42.4% vs 42.0%)"


def test_gate_margin_marks_a_name_below_the_gate():
    from main import _gate_margin_vi
    s = _gate_margin_vi(0.4150, 0.42)
    assert s.startswith("-0.50pp")
    assert "DƯỚI cổng" in s


def test_gate_margin_never_prints_the_two_figures_identically():
    """Inherits `_fmt_pct_pair`'s anti-collision rule.

    A name 0.05pp over the gate would otherwise render "42.0% vs 42.0%" - the
    same self-contradiction fixed in the fallback reasons on 25-08.
    """
    from main import _gate_margin_vi
    s = _gate_margin_vi(0.4205, 0.42)
    assert "42.05% vs 42.00%" in s, s


def test_gate_margin_returns_none_when_an_input_is_missing():
    """No guessing from a half-known state.

    The secondary artifact can fail to load; inventing a margin would repeat the
    pinned-threshold class of bug fixed on 13-08 and 25-08.
    """
    from main import _gate_margin_vi
    assert _gate_margin_vi(0.4241, None) is None
    assert _gate_margin_vi(None, 0.42) is None


def test_card_falls_back_to_base_decision_for_unmigrated_callers():
    """Signal dicts built before 27-08 still render rather than losing a line."""
    c = _card(regime_action_vi="Không điều chỉnh", garch_scalar=0.85,
              arb_note_vi="Không can thiệp", risk_tier="RISK_MID",
              risk_tier_pct=60, risk_tier_label_vi="TRUNG BÌNH — thận trọng",
              market_regime=3, regime_label="Strong Trend",
              base_decision_vi="MUA (BUY)")
    assert "Phân loại gốc của mô hình: <b>MUA (BUY)</b>" in c


def test_gate_margin_wins_when_both_fields_are_present():
    c = _card(regime_action_vi="Không điều chỉnh", garch_scalar=0.85,
              arb_note_vi="Không can thiệp", risk_tier="RISK_MID",
              risk_tier_pct=60, risk_tier_label_vi="TRUNG BÌNH — thận trọng",
              market_regime=3, regime_label="Strong Trend",
              base_decision_vi="MUA (BUY)",
              gate_margin_vi="+0.41pp trên cổng (42.4% vs 42.0%)")
    assert "Biên qua cổng: <b>+0.41pp trên cổng (42.4% vs 42.0%)</b>" in c
    assert "Phân loại gốc" not in c
