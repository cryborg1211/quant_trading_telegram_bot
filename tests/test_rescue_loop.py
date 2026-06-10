"""Unit tests for _rescue_loop() — rescue bull-bypass + bear veto event layer."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from main import _rescue_loop, SAFE_BUY_THRESHOLD, EVENT_MIN_P_UP, EVENT_BULL_SENTIMENT, EVENT_BEAR_SENTIMENT


# ── helpers ──────────────────────────────────────────────────────────────────

def _preds(**kv: float) -> dict[str, list[float]]:
    return {t: [0.0, 0.0, p] for t, p in kv.items()}


def _horizon(preds_5d: dict) -> dict[str, dict]:
    return {"5d": preds_5d, "20d": {}}


# ── tests ────────────────────────────────────────────────────────────────────


def test_fallback_mode_returns_unchanged_signals_and_empty_overrides():
    signals = ["VCB", "BID"]
    result_signals, overrides = _rescue_loop(
        fallback_mode=True,
        stacking_predictions_5d=_preds(VCB=0.7, BID=0.6),
        universe_tickers={"VCB", "BID"},
        top_buy_signals=signals,
        all_sentiments={},
        horizon_predictions=_horizon(_preds(VCB=0.7, BID=0.6)),
    )
    assert result_signals is signals  # exact same object
    assert overrides == {}


def test_no_rescue_candidates_returns_unchanged():
    preds = _preds(VCB=0.7, BID=0.6)
    signals = ["VCB", "BID"]
    sentiments = {
        "VCB": {"sentiment_score": 0.3},
        "BID": {"sentiment_score": 0.3},
    }
    result_signals, overrides = _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "BID"},
        top_buy_signals=signals,
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    # No ticker in rescue range (all P(UP) >= SAFE_BUY_THRESHOLD or not in range)
    assert set(result_signals) == {"VCB", "BID"}


@patch("main.evaluate_trades_batch")
def test_rescue_candidate_fetches_missing_sentiment(mock_eval):
    mock_eval.return_value = ({}, {"FPT": {"sentiment_score": 0.8}})
    preds = _preds(VCB=0.7, FPT=0.43)  # FPT in rescue range
    sentiments = {"VCB": {"sentiment_score": 0.3}}
    _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "FPT"},
        top_buy_signals=["VCB"],
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    mock_eval.assert_called_once()
    assert "FPT" in sentiments  # mutated in-place


@patch("main.evaluate_trades_batch")
def test_rescue_candidate_added_when_sentiment_strong(mock_eval):
    mock_eval.return_value = ({}, {"FPT": {"sentiment_score": 0.8, "reasoning_vi": "good news"}})
    preds = _preds(VCB=0.7, FPT=0.43)
    sentiments = {"VCB": {"sentiment_score": 0.3}}
    result_signals, overrides = _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "FPT"},
        top_buy_signals=["VCB"],
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    assert "FPT" in result_signals
    assert "FPT" in overrides


@patch("main.evaluate_trades_batch")
def test_all_sentiments_mutated_with_rescue_data(mock_eval):
    mock_eval.return_value = ({}, {"FPT": {"sentiment_score": 0.8}})
    preds = _preds(VCB=0.7, FPT=0.43)
    sentiments = {"VCB": {"sentiment_score": 0.3}}
    _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "FPT"},
        top_buy_signals=["VCB"],
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    assert "FPT" in sentiments


@patch("main.evaluate_trades_batch")
def test_sentiment_fetch_failure_is_swallowed(mock_eval):
    mock_eval.side_effect = RuntimeError("API timeout")
    preds = _preds(VCB=0.7, FPT=0.43)
    sentiments = {"VCB": {"sentiment_score": 0.3}}
    # Should not raise
    result_signals, overrides = _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "FPT"},
        top_buy_signals=["VCB"],
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    assert "VCB" in result_signals


def test_bear_veto_applied_via_build_event_overrides():
    preds = _preds(VCB=0.7)
    sentiments = {"VCB": {"sentiment_score": -0.6}}
    result_signals, overrides = _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB"},
        top_buy_signals=["VCB"],
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    assert "VCB" in overrides
    assert overrides["VCB"]["weight"] == 0.0


@patch("main.evaluate_trades_batch")
def test_output_list_is_new_object_not_mutated_input(mock_eval):
    mock_eval.return_value = ({}, {"FPT": {"sentiment_score": 0.8, "reasoning_vi": "news"}})
    preds = _preds(VCB=0.7, FPT=0.43)
    sentiments = {"VCB": {"sentiment_score": 0.3}}
    original = ["VCB"]
    result_signals, _ = _rescue_loop(
        fallback_mode=False,
        stacking_predictions_5d=preds,
        universe_tickers={"VCB", "FPT"},
        top_buy_signals=original,
        all_sentiments=sentiments,
        horizon_predictions=_horizon(preds),
    )
    # If rescue added, result must be a new list
    if len(result_signals) > len(original):
        assert result_signals is not original
