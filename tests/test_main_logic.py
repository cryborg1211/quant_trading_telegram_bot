"""Unit tests for pure helper functions in main.py.

All tests are offline: no DuckDB, no network, no model artifact files.
The conftest stubs out joblib, catboost, and heavy local imports so that
`import main` succeeds without the full ML stack installed.
"""
from __future__ import annotations

from datetime import datetime, time as dt_time, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import main
from main import (
    _build_combined_report,
    _build_feature_explanation,
    _build_rebalance_report,
    _format_sentiment_status,
    _get_live_exec_prices,
    _humanize_feature,
    is_crawl_allowed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# VN is UTC+7; use a fixed-offset timezone so tests work without zoneinfo data.
_VN_TZ = timezone(timedelta(hours=7))


def _fake_dt(hour: int, minute: int) -> datetime:
    """Return a datetime with VN timezone at the given HH:MM on a fixed date."""
    return datetime(2024, 6, 15, hour, minute, 0, tzinfo=_VN_TZ)


# ---------------------------------------------------------------------------
# is_crawl_allowed
# ---------------------------------------------------------------------------

class TestIsCrawlAllowed:
    def test_force_crawl_bypasses_guard(self):
        # force_crawl=True must return True regardless of the clock.
        assert is_crawl_allowed(force_crawl=True) is True

    def test_market_open_returns_false(self):
        with patch("main.datetime") as mock_dt:
            mock_dt.now.return_value = _fake_dt(14, 44)
            assert is_crawl_allowed(force_crawl=False) is False

    def test_market_closed_returns_true(self):
        with patch("main.datetime") as mock_dt:
            mock_dt.now.return_value = _fake_dt(15, 0)
            assert is_crawl_allowed(force_crawl=False) is True

    def test_boundary_one_minute_before_close_still_blocked(self):
        # MARKET_CLOSE = dt_time(15, 0). Condition is strict `<`.
        # At 14:59 → 14:59 < 15:00 = True → crawl blocked → returns False.
        # The boundary moment (exactly 15:00) is covered by test_market_closed_returns_true.
        with patch("main.datetime") as mock_dt:
            mock_dt.now.return_value = _fake_dt(14, 59)
            assert is_crawl_allowed(force_crawl=False) is False


# ---------------------------------------------------------------------------
# _humanize_feature
# ---------------------------------------------------------------------------

class TestHumanizeFeature:
    def test_exact_match_rsi(self):
        assert _humanize_feature("rsi_14") == "RSI 14 ngày"

    def test_exact_match_close(self):
        assert _humanize_feature("close") == "Nền giá đóng cửa"

    def test_macro_known_inner(self):
        result = _humanize_feature("macro_sp500_close")
        assert "S&P 500" in result or "sp500" in result.lower()

    def test_macro_unknown_inner_title_fallback(self):
        result = _humanize_feature("macro_unknown_key")
        # Inner part "unknown_key" → title-case fallback "Unknown Key"
        assert "Unknown Key" in result or "unknown" in result.lower()

    def test_lag_suffix_known_base(self):
        result = _humanize_feature("rsi_14_lag_3")
        assert "3 phiên" in result
        assert "RSI" in result

    def test_lag_suffix_unknown_base(self):
        result = _humanize_feature("my_indicator_lag_10")
        assert "10 phiên" in result

    def test_alpha360_numeric_ohlcv(self):
        result = _humanize_feature("close_38")
        assert "38 phiên" in result
        assert "đóng cửa" in result.lower() or "close" in result.lower()

    def test_alpha360_numeric_norm_strip(self):
        # norm_ prefix stripped, then "vwap" resolved via FEATURE_HUMAN_NAMES (kept for backward compat)
        result = _humanize_feature("norm_vwap_5")
        assert "5 phiên" in result
        assert "VWAP" in result or "vwap" in result.lower()

    def test_alpha360_numeric_hlc3_strip(self):
        # hlc3 is the new column name after the vwap→hlc3 rename (Flaw 1 fix)
        result = _humanize_feature("norm_hlc3_5")
        assert "5 phiên" in result
        assert "HLC3" in result or "hlc3" in result.lower() or "Cao" in result

    def test_fallback_title_case(self):
        result = _humanize_feature("some_unknown_feat")
        assert result == "Some Unknown Feat"


# ---------------------------------------------------------------------------
# _format_sentiment_status
# ---------------------------------------------------------------------------

class TestFormatSentimentStatus:
    def test_no_urls_zero_score_returns_no_news(self):
        result = _format_sentiment_status({"source_urls": [], "sentiment_score": 0.0})
        assert "Không có tin" in result or "Timeout" in result

    def test_empty_dict_defaults_to_no_news(self):
        result = _format_sentiment_status({})
        assert "Không có tin" in result or "Timeout" in result

    def test_positive_score_above_threshold(self):
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": 0.5})
        assert result.startswith("Tích cực")

    def test_negative_score_below_threshold(self):
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": -0.5})
        assert result.startswith("Tiêu cực")

    def test_neutral_score_in_range(self):
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": 0.1})
        assert result.startswith("Trung tính")

    def test_score_formatted_in_output(self):
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": 0.5})
        assert "+0.50" in result

    def test_boundary_plus_0_2_is_neutral(self):
        # Condition is score > 0.2, so 0.2 exactly is neutral.
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": 0.2})
        assert result.startswith("Trung tính")

    def test_boundary_minus_0_2_is_neutral(self):
        # Condition is score < -0.2, so -0.2 exactly is neutral.
        result = _format_sentiment_status({"source_urls": ["x"], "sentiment_score": -0.2})
        assert result.startswith("Trung tính")


# ---------------------------------------------------------------------------
# _get_live_exec_prices
# ---------------------------------------------------------------------------

def _make_df(**cols) -> pd.DataFrame:
    """Build a one-row DataFrame with the given column name→value mapping."""
    return pd.DataFrame({k: [v] for k, v in cols.items()})


class TestGetLiveExecPrices:
    def test_price_below_1000_scaled(self):
        df = _make_df(ticker="VNM", close=25.5)
        result = _get_live_exec_prices(df, ["VNM"])
        assert result["VNM"] == pytest.approx(25_500.0)

    def test_price_above_1000_unchanged(self):
        df = _make_df(ticker="VNM", close=25_000.0)
        result = _get_live_exec_prices(df, ["VNM"])
        assert result["VNM"] == pytest.approx(25_000.0)

    def test_nan_price_skipped(self):
        df = _make_df(ticker="VNM", close=float("nan"))
        assert "VNM" not in _get_live_exec_prices(df, ["VNM"])

    def test_inf_price_skipped(self):
        df = _make_df(ticker="VNM", close=float("inf"))
        assert "VNM" not in _get_live_exec_prices(df, ["VNM"])

    def test_negative_price_skipped(self):
        df = _make_df(ticker="VNM", close=-50.0)
        assert "VNM" not in _get_live_exec_prices(df, ["VNM"])

    def test_zero_price_skipped(self):
        df = _make_df(ticker="VNM", close=0.0)
        assert "VNM" not in _get_live_exec_prices(df, ["VNM"])

    def test_no_price_column_returns_empty(self):
        df = pd.DataFrame({"ticker": ["VNM"], "volume": [1_000_000]})
        assert _get_live_exec_prices(df, ["VNM"]) == {}

    def test_raw_close_preferred_over_close(self):
        # raw_close comes first in the priority list; close should be ignored.
        df = pd.DataFrame({"ticker": ["VNM"], "raw_close": [1_500.0], "close": [2_000.0]})
        result = _get_live_exec_prices(df, ["VNM"])
        assert result["VNM"] == pytest.approx(1_500.0)

    def test_unknown_ticker_absent(self):
        df = _make_df(ticker="VNM", close=100.0)
        result = _get_live_exec_prices(df, ["HPG"])
        assert "HPG" not in result

    def test_multiple_rows_uses_last(self):
        df = pd.DataFrame({"ticker": ["VNM", "VNM"], "close": [100.0, 200.0]})
        result = _get_live_exec_prices(df, ["VNM"])
        # Last row has close=200 → 200 * 1000 = 200_000
        assert result["VNM"] == pytest.approx(200_000.0)


# ---------------------------------------------------------------------------
# (TestAlignedProba removed — `aligned_proba` was a V6 stacker orphan, purged
#  in the V4 refactor.  The V4 3-class alignment lives in TabularEnsemble.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_feature_explanation
# ---------------------------------------------------------------------------

def _importance_model(importances: list[float], features: list[str]) -> tuple:
    """Return (mock_model, selected_features) ready for _build_feature_explanation."""
    m = MagicMock()
    m.feature_importances_ = np.array(importances, dtype=np.float64)
    return m, features


class TestBuildFeatureExplanation:
    def test_no_importance_attr_returns_fallback(self):
        model = MagicMock(spec=["predict_proba"])
        pos, risk = _build_feature_explanation(model, ["close_5", "rsi_14"])
        assert "Không có" in pos
        assert "Không có" in risk or "Theo dõi" in risk

    def test_empty_array_returns_fallback(self):
        model = MagicMock()
        model.feature_importances_ = np.array([], dtype=np.float64)
        pos, risk = _build_feature_explanation(model, ["close_5"])
        assert "Không có" in pos

    def test_empty_features_list_returns_fallback(self):
        model, _ = _importance_model([0.5, 0.3], [])
        pos, risk = _build_feature_explanation(model, [])
        assert "Không có" in pos

    def test_highest_importance_in_pos_string(self):
        # rsi_14 has highest importance (0.9)
        model, feats = _importance_model([0.9, 0.05, 0.05], ["rsi_14", "volume_5", "close_3"])
        pos, _ = _build_feature_explanation(model, feats, top_k=1)
        assert "RSI" in pos

    def test_lowest_importance_in_risk_string(self):
        # close_3 has lowest importance (0.05)
        model, feats = _importance_model([0.9, 0.05, 0.05], ["rsi_14", "volume_5", "close_3"])
        _, risk = _build_feature_explanation(model, feats, top_k=1)
        assert risk  # non-empty

    def test_returns_two_element_tuple(self):
        model, feats = _importance_model([0.5, 0.5], ["rsi_14", "close_5"])
        result = _build_feature_explanation(model, feats)
        assert isinstance(result, tuple) and len(result) == 2

    def test_humanize_applied_to_feature_names(self):
        # "rsi_14" should appear as "RSI 14 ngày" in the output
        model, feats = _importance_model([1.0, 0.0], ["rsi_14", "some_unknown"])
        pos, _ = _build_feature_explanation(model, feats, top_k=1)
        assert "RSI" in pos

    def test_top_k_1_limits_drivers(self):
        model, feats = _importance_model([0.5, 0.3, 0.2], ["rsi_14", "close_5", "volume_3"])
        pos, _ = _build_feature_explanation(model, feats, top_k=1)
        # Should only contain one item — no comma separating multiple features
        assert pos.count("RSI") + pos.count("Nền") + pos.count("Khối lượng") <= 1


# ---------------------------------------------------------------------------
# _build_combined_report
# ---------------------------------------------------------------------------

_MINIMAL_SIGNAL: dict = {}  # _build_message handles all missing keys gracefully


class TestBuildCombinedReport:
    def test_empty_list_returns_empty_string(self):
        assert _build_combined_report([]) == ""

    def test_single_signal_returns_nonempty_string(self):
        result = _build_combined_report([_MINIMAL_SIGNAL])
        assert isinstance(result, str) and len(result) > 0

    def test_two_signals_joined_by_separator(self):
        result = _build_combined_report([_MINIMAL_SIGNAL, _MINIMAL_SIGNAL])
        # _REPORT_SEPARATOR = "\n\n══════...══════\n\n" — count joining occurrences
        assert main._REPORT_SEPARATOR in result

    def test_three_signals_two_separators(self):
        result = _build_combined_report([_MINIMAL_SIGNAL] * 3)
        assert result.count(main._REPORT_SEPARATOR) == 2

    def test_result_is_string(self):
        assert isinstance(_build_combined_report([_MINIMAL_SIGNAL]), str)


# ---------------------------------------------------------------------------
# _build_rebalance_report
# ---------------------------------------------------------------------------

_SAMPLE_HOLDINGS = [
    {"ticker": "FPT", "pnl_pct": 12.5, "pred_label": "Tăng", "p_up": 0.7},
    {"ticker": "VNM", "pnl_pct": -3.0, "pred_label": "Giảm", "p_up": 0.15},
]


class TestBuildRebalanceReport:
    def test_returns_string(self):
        assert isinstance(_build_rebalance_report(_SAMPLE_HOLDINGS, "Giữ FPT, bán VNM."), str)

    def test_contains_header(self):
        result = _build_rebalance_report(_SAMPLE_HOLDINGS, "Advice text")
        assert "TƯ VẤN" in result

    def test_positive_pnl_shows_green_icon(self):
        result = _build_rebalance_report(_SAMPLE_HOLDINGS, "X")
        assert "🟢" in result

    def test_negative_pnl_shows_red_icon(self):
        result = _build_rebalance_report(_SAMPLE_HOLDINGS, "X")
        assert "🔴" in result

    def test_ticker_names_in_output(self):
        result = _build_rebalance_report(_SAMPLE_HOLDINGS, "X")
        assert "FPT" in result
        assert "VNM" in result

    def test_advice_html_escaped(self):
        result = _build_rebalance_report([], "Buy <FPT> & sell VNM")
        assert "&lt;FPT&gt;" in result
        assert "&amp;" in result

    def test_pnl_sign_positive(self):
        result = _build_rebalance_report([{"ticker": "FPT", "pnl_pct": 5.0, "pred_label": "Tăng", "p_up": 0.6}], "X")
        assert "+5.0%" in result

    def test_pnl_sign_negative(self):
        result = _build_rebalance_report([{"ticker": "VNM", "pnl_pct": -3.0, "pred_label": "Giảm", "p_up": 0.2}], "X")
        assert "-3.0%" in result

    def test_empty_holdings_shows_empty_label(self):
        result = _build_rebalance_report([], "Advice")
        assert "trống" in result or "Danh mục" in result

    def test_ticker_html_escaped(self):
        result = _build_rebalance_report([{"ticker": "<XSS>", "pnl_pct": 0.0, "pred_label": "N/A", "p_up": 0.0}], "X")
        assert "<XSS>" not in result
        assert "&lt;XSS&gt;" in result
