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
