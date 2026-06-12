"""Integration tests for daily_inference — happy path, fallback path, rescue loop."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import polars as pl
import pytest


# ── shared fixtures ──────────────────────────────────────────────────────────

_TICKERS = ["VCB", "BID", "VHM"]

def _make_polars_df(tickers: list[str] | None = None) -> pl.DataFrame:
    tickers = tickers or _TICKERS
    rows = []
    for t in tickers:
        for i in range(5):
            rows.append({
                "ticker": t,
                "date": f"2024-06-{10+i:02d}",
                "open": 50000.0,
                "high": 51000.0,
                "low": 49000.0,
                "close": 50500.0,
                "volume": 1000000.0,
                "raw_close": 50500.0,
            })
    return pl.DataFrame(rows)


def _mock_alpha360():
    mock_gen = MagicMock()
    mock_gen.load_live_ohlcv_window.return_value = _make_polars_df()
    return mock_gen


# ── happy path ───────────────────────────────────────────────────────────────


@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_daily_inference_happy_path(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
):
    mock_a360_cls.return_value = _mock_alpha360()

    def predict_side_effect(df, horizon):
        if horizon == 20:  # PRIMARY dispatch horizon (T+3 secondary returns empty)
            return (
                {"VCB": [0.1, 0.2, 0.70], "BID": [0.1, 0.25, 0.65], "VHM": [0.15, 0.25, 0.60]},
                {"pnl_threshold_tau": 0.45},
                MagicMock(feature_importances_=None),
                ["feat_a", "feat_b"],
                {"VCB": True, "BID": True, "VHM": True},
            )
        return ({}, {}, MagicMock(), [], {})

    mock_predict.side_effect = predict_side_effect
    mock_eval.return_value = (
        {"VCB": 2, "BID": 2, "VHM": 2},
        {
            "VCB": {"sentiment_score": 0.4, "reasoning_vi": "good"},
            "BID": {"sentiment_score": 0.3, "reasoning_vi": "ok"},
            "VHM": {"sentiment_score": 0.2, "reasoning_vi": "neutral"},
        },
    )
    mock_rte.return_value = "<b>test report</b>"

    from main import daily_inference
    result = daily_inference(broadcast=False)

    assert result == "<b>test report</b>"
    mock_rte.assert_called_once()
    call_kwargs = mock_rte.call_args
    top_buy = call_kwargs.kwargs.get("top_buy_signals") or call_kwargs[1].get("top_buy_signals")
    if top_buy is None:
        top_buy = call_kwargs[0][0] if call_kwargs[0] else []
    assert len(top_buy) == 3
    assert set(top_buy) == {"VCB", "BID", "VHM"}
    mock_eval.assert_called_once()


# ── fallback path ────────────────────────────────────────────────────────────


@patch("main._build_fallback_observability_report_vi")
@patch("main._get_live_exec_prices")
@patch("main.mr_score_tickers")
@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_daily_inference_fallback_path(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
    mock_mr, mock_prices, mock_fb_report,
):
    mock_a360_cls.return_value = _mock_alpha360()

    def predict_side_effect(df, horizon):
        if horizon == 20:  # PRIMARY dispatch horizon
            return (
                {"VCB": [0.3, 0.4, 0.30], "BID": [0.35, 0.37, 0.28], "VHM": [0.4, 0.35, 0.25]},
                {"pnl_threshold_tau": 0.45},
                MagicMock(feature_importances_=None),
                ["feat_a"],
                {"VCB": False, "BID": False, "VHM": False},
            )
        return ({}, {}, MagicMock(), [], {})

    mock_predict.side_effect = predict_side_effect
    mock_eval.return_value = ({}, {})
    mock_mr.return_value = {}
    mock_prices.return_value = {"VCB": 50000.0}
    mock_fb_report.return_value = "<b>fallback report</b>"

    from main import daily_inference
    result = daily_inference(broadcast=False)

    assert result == "<b>fallback report</b>"
    mock_rte.assert_not_called()
    mock_fb_report.assert_called_once()


# ── rescue loop ──────────────────────────────────────────────────────────────


@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_daily_inference_rescue_loop_invoked(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
):
    mock_a360_cls.return_value = _mock_alpha360()

    def predict_side_effect(df, horizon):
        if horizon == 20:  # PRIMARY dispatch horizon
            return (
                {
                    "VCB": [0.1, 0.2, 0.70],
                    "BID": [0.1, 0.25, 0.65],
                    "FPT": [0.2, 0.37, 0.43],  # rescue range
                },
                {"pnl_threshold_tau": 0.45},
                MagicMock(feature_importances_=None),
                ["feat_a"],
                {"VCB": True, "BID": True, "FPT": False},
            )
        return ({}, {}, MagicMock(), [], {})

    mock_predict.side_effect = predict_side_effect

    call_count = [0]
    def eval_side_effect(horizon_preds, tickers):
        call_count[0] += 1
        if call_count[0] == 1:
            # Main evaluation for candidates
            return (
                {"VCB": 2, "BID": 2},
                {
                    "VCB": {"sentiment_score": 0.4, "reasoning_vi": "good"},
                    "BID": {"sentiment_score": 0.3, "reasoning_vi": "ok"},
                },
            )
        # Rescue sentiment fetch for FPT
        return (
            {},
            {"FPT": {"sentiment_score": 0.8, "reasoning_vi": "strong news"}},
        )

    mock_eval.side_effect = eval_side_effect
    mock_rte.return_value = "<b>rescue report</b>"

    from main import daily_inference
    result = daily_inference(broadcast=False)

    assert result == "<b>rescue report</b>"
    mock_rte.assert_called_once()
    call_kwargs = mock_rte.call_args
    top_buy = call_kwargs.kwargs.get("top_buy_signals") or call_kwargs[1].get("top_buy_signals")
    if top_buy is None:
        top_buy = call_kwargs[0][0] if call_kwargs[0] else []
    assert "FPT" in top_buy
    event_ov = call_kwargs.kwargs.get("event_overrides") or call_kwargs[1].get("event_overrides")
    if event_ov is None:
        # positional
        event_ov = call_kwargs[0][9] if len(call_kwargs[0]) > 9 else {}
    assert "FPT" in event_ov
    assert event_ov["FPT"]["weight"] == 0.05
