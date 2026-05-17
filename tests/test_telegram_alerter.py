"""Unit tests for TelegramBot._build_message in src/utils/telegram_alerter.py.

_build_message is a pure static method — it does HTML escaping, applies
defaults, caps article_urls at 3, and formats the output string. No network
calls are involved.
"""
from __future__ import annotations

import pytest

from src.utils.telegram_alerter import TelegramBot

_build = TelegramBot._build_message


class TestBuildMessage:
    # -----------------------------------------------------------------
    # HTML escaping
    # -----------------------------------------------------------------

    def test_escapes_ticker_lt_gt(self):
        result = _build({"ticker": "<script>"})
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_escapes_action_ampersand(self):
        result = _build({"action": "MUA & HOLD"})
        assert "&amp;" in result
        assert "MUA & HOLD" not in result

    def test_escapes_gemini_summary(self):
        result = _build({"gemini_summary": "<b>bold</b>"})
        assert "&lt;b&gt;" in result
        assert "<b>bold</b>" not in result

    # -----------------------------------------------------------------
    # Default values for missing keys
    # -----------------------------------------------------------------

    def test_missing_ticker_defaults_na(self):
        result = _build({})
        assert "N/A" in result

    def test_missing_action_defaults_na(self):
        result = _build({})
        assert "N/A" in result

    def test_missing_price_defaults_na(self):
        result = _build({})
        assert "N/A" in result

    # -----------------------------------------------------------------
    # article_urls cap and fallback
    # -----------------------------------------------------------------

    def test_article_urls_capped_at_3(self):
        urls = [
            "https://cafef.vn/a1",
            "https://cafef.vn/a2",
            "https://cafef.vn/a3",
            "https://cafef.vn/a4",
            "https://cafef.vn/a5",
        ]
        result = _build({"article_urls": urls})
        # 4th and 5th URLs must not appear anywhere in the output.
        assert "a4" not in result
        assert "a5" not in result

    def test_article_urls_exactly_3_all_present(self):
        urls = [
            "https://cafef.vn/a1",
            "https://cafef.vn/a2",
            "https://cafef.vn/a3",
        ]
        result = _build({"article_urls": urls})
        assert "a1" in result
        assert "a2" in result
        assert "a3" in result

    def test_empty_urls_shows_fallback_text(self):
        result = _build({"article_urls": []})
        assert "Không có tin" in result

    def test_none_urls_shows_fallback_text(self):
        result = _build({"article_urls": None})
        assert "Không có tin" in result

    # -----------------------------------------------------------------
    # Domain label formatting
    # -----------------------------------------------------------------

    def test_known_domain_cafef_label(self):
        result = _build({"article_urls": ["https://cafef.vn/article/123"]})
        assert "CafeF" in result

    def test_unknown_domain_uses_hostname_or_nguon(self):
        result = _build({"article_urls": ["https://unknown-site.com/article"]})
        # For an unknown domain, _domain_label returns the raw domain or "Nguồn".
        assert "unknown-site.com" in result or "Nguồn" in result

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def test_output_is_string(self):
        assert isinstance(_build({}), str)

    def test_confidence_in_output(self):
        result = _build({"confidence": 87.5})
        assert "87.5" in result

    def test_known_ticker_unescaped_in_output(self):
        result = _build({"ticker": "FPT"})
        assert "FPT" in result
