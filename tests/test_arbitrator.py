"""Unit tests for pure functions in src/models/quant_agent_arbitrator.py.

The arbitrator uses try/except for all its heavy optional deps (aiohttp, bs4,
gnews, google-genai), so the real module imports cleanly in a bare test env.
"""
from __future__ import annotations

import pytest

from unittest.mock import MagicMock, patch

from src.models.quant_agent_arbitrator import (
    REBALANCE_SYSTEM_PROMPT,
    _is_binary_content_type,
    _is_binary_url,
    get_rebalance_advice,
    make_final_decision,
)

# ---------------------------------------------------------------------------
# make_final_decision — all 6 decision paths + boundary conditions
# ---------------------------------------------------------------------------
# Decision logic (simplified):
#   pred_5d = argmax(pred_5d_probs)
#   pred_20d = argmax(pred_20d_probs) or pred_5d
#
#   5d==2, sentiment < -0.5  → 1  (safety override)
#   5d==2                    → 2  (BUY)
#   5d==1, 20d==2            → 2  (trend active)
#   5d==0, 20d==0, sent>0.5  → 1  (double-down veto)
#   5d==0, 20d==0            → 0  (full exit)
#   5d==0, sent>0.5          → 1  (partial veto)
#   otherwise                → pred_5d  (passthrough)

_UP   = [0.1, 0.2, 0.7]   # argmax = 2
_SIDE = [0.2, 0.5, 0.3]   # argmax = 1
_DOWN = [0.7, 0.2, 0.1]   # argmax = 0


class TestMakeFinalDecision:
    def test_5d_up_returns_buy(self):
        assert make_final_decision(_UP, sentiment_score=0.0) == 2

    def test_5d_up_very_negative_sentiment_safety_override(self):
        assert make_final_decision(_UP, sentiment_score=-0.6) == 1

    def test_boundary_sentiment_minus_0_5_not_overridden(self):
        # Condition is `< -0.5`; exactly -0.5 must NOT trigger the override.
        assert make_final_decision(_UP, sentiment_score=-0.5) == 2

    def test_5d_sideways_20d_up_trend_active(self):
        assert make_final_decision(_SIDE, sentiment_score=0.0, pred_20d_probs=_UP) == 2

    def test_double_down_full_exit(self):
        assert make_final_decision(_DOWN, sentiment_score=0.0, pred_20d_probs=_DOWN) == 0

    def test_double_down_sentiment_veto(self):
        assert make_final_decision(_DOWN, sentiment_score=0.6, pred_20d_probs=_DOWN) == 1

    def test_boundary_sentiment_0_5_not_vetoed_in_double_down(self):
        # Condition is `> 0.5`; exactly 0.5 must NOT trigger veto.
        assert make_final_decision(_DOWN, sentiment_score=0.5, pred_20d_probs=_DOWN) == 0

    def test_partial_veto_5d_down_good_sentiment(self):
        # pred_20d is SIDE (not DOWN), so double-down branch is skipped.
        # Falls through to `if pred_5d == 0 and sentiment_score > 0.5`.
        assert make_final_decision(_DOWN, sentiment_score=0.6, pred_20d_probs=_SIDE) == 1

    def test_passthrough_5d_sideways_20d_down(self):
        # pred_5d=1, pred_20d=0 — no special branch fires → passthrough pred_5d=1.
        assert make_final_decision(_SIDE, sentiment_score=0.0, pred_20d_probs=_DOWN) == 1

    def test_pred20d_none_defaults_to_pred5d(self):
        # pred_20d_probs=None → pred_20d = pred_5d = 0; double-down full exit.
        assert make_final_decision(_DOWN, sentiment_score=0.0, pred_20d_probs=None) == 0

    def test_log_detail_does_not_change_result(self):
        r1 = make_final_decision(_UP, sentiment_score=0.0, log_detail=False)
        r2 = make_final_decision(_UP, sentiment_score=0.0, log_detail=True)
        assert r1 == r2


# ---------------------------------------------------------------------------
# _is_binary_url
# ---------------------------------------------------------------------------

class TestIsBinaryUrl:
    def test_pdf_is_binary(self):
        assert _is_binary_url("https://example.com/report.pdf") is True

    def test_jpg_is_binary(self):
        assert _is_binary_url("https://example.com/photo.jpg") is True

    def test_zip_is_binary(self):
        assert _is_binary_url("https://example.com/archive.zip") is True

    def test_mp4_is_binary(self):
        assert _is_binary_url("https://example.com/clip.mp4") is True

    def test_html_path_not_binary(self):
        assert _is_binary_url("https://cafef.vn/tin-tuc/article") is False

    def test_html_extension_not_binary(self):
        assert _is_binary_url("https://cafef.vn/page.html") is False

    def test_uppercase_extension_is_binary(self):
        # Path is lowercased before comparison.
        assert _is_binary_url("https://example.com/REPORT.PDF") is True

    def test_extension_in_query_string_not_binary(self):
        # urlparse puts query params in .query, not .path.
        assert _is_binary_url("https://example.com/page?file=doc.pdf") is False


# ---------------------------------------------------------------------------
# _is_binary_content_type
# ---------------------------------------------------------------------------

class TestIsBinaryContentType:
    def test_text_html_not_binary(self):
        assert _is_binary_content_type("text/html") is False

    def test_text_html_with_charset_not_binary(self):
        assert _is_binary_content_type("text/html; charset=utf-8") is False

    def test_application_xhtml_xml_not_binary(self):
        assert _is_binary_content_type("application/xhtml+xml") is False

    def test_application_xml_not_binary(self):
        assert _is_binary_content_type("application/xml") is False

    def test_empty_string_not_binary(self):
        assert _is_binary_content_type("") is False

    def test_application_pdf_is_binary(self):
        assert _is_binary_content_type("application/pdf") is True

    def test_image_prefix_is_binary(self):
        assert _is_binary_content_type("image/png") is True

    def test_audio_prefix_is_binary(self):
        assert _is_binary_content_type("audio/mpeg") is True

    def test_video_prefix_is_binary(self):
        assert _is_binary_content_type("video/mp4") is True

    def test_application_vnd_prefix_is_binary(self):
        assert _is_binary_content_type("application/vnd.ms-excel") is True

    def test_application_octet_stream_is_binary(self):
        assert _is_binary_content_type("application/octet-stream") is True


# ---------------------------------------------------------------------------
# get_rebalance_advice
# ---------------------------------------------------------------------------

_SAMPLE_HOLDINGS = [
    {"ticker": "FPT", "pnl_pct": 12.5, "pred_label": "Tăng", "p_up": 0.7},
    {"ticker": "VNM", "pnl_pct": -3.0, "pred_label": "Giảm", "p_up": 0.15},
]

_SAMPLE_NEWS = {
    "FPT": ["Source URL: https://cafef.vn/fpt\nTitle: FPT công bố doanh thu tăng trưởng\nFull Article Body:\n..."],
}


class TestGetRebalanceAdvice:
    def test_no_api_key_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = get_rebalance_advice(_SAMPLE_HOLDINGS, _SAMPLE_NEWS)
        assert isinstance(result, str) and len(result) > 0

    def test_missing_genai_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        import src.models.quant_agent_arbitrator as arb
        original_genai = arb.genai
        arb.genai = None
        try:
            result = get_rebalance_advice(_SAMPLE_HOLDINGS, _SAMPLE_NEWS)
            assert isinstance(result, str) and len(result) > 0
        finally:
            arb.genai = original_genai

    def test_gemini_exception_returns_error_string(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        import src.models.quant_agent_arbitrator as arb
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")
        with patch.object(arb.genai, "Client", return_value=mock_client):
            result = get_rebalance_advice(_SAMPLE_HOLDINGS, _SAMPLE_NEWS)
        assert isinstance(result, str) and len(result) > 0

    def test_gemini_empty_response_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        import src.models.quant_agent_arbitrator as arb
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        with patch.object(arb.genai, "Client", return_value=mock_client):
            result = get_rebalance_advice(_SAMPLE_HOLDINGS, _SAMPLE_NEWS)
        assert isinstance(result, str) and len(result) > 0

    def test_gemini_success_returns_response_text(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        import src.models.quant_agent_arbitrator as arb
        mock_resp = MagicMock()
        mock_resp.text = "Nên chốt lời FPT vì đã tăng 12.5%."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        with patch.object(arb.genai, "Client", return_value=mock_client):
            result = get_rebalance_advice(_SAMPLE_HOLDINGS, _SAMPLE_NEWS)
        assert result == "Nên chốt lời FPT vì đã tăng 12.5%."

    def test_empty_holdings_does_not_crash(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = get_rebalance_advice([], {})
        assert isinstance(result, str)

    def test_rebalance_prompt_is_vietnamese(self):
        assert "Việt Nam" in REBALANCE_SYSTEM_PROMPT or "tiếng Việt" in REBALANCE_SYSTEM_PROMPT
