"""Capture-point wiring tests for daily_inference → signal_ledger.record_cancelled.

Mirrors tests/test_daily_inference_integration.py's @patch stack style: drive
daily_inference through the weak-market fallback branch (capture point 1) and the
post-arbitration no-buy monitor branch (capture point 2), asserting the rejected
Top-3 are logged with the right horizon/hold_days, that the two points are
mutually exclusive, and that the config kill-switch skips both.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import main


_TICKERS = ["VCB", "BID", "VHM"]


@pytest.fixture(autouse=True)
def _no_prod_paperlog(monkeypatch):
    # Same guard as test_daily_inference_integration: the fallback branch these
    # tests drive now writes the PROD paperlog via DuckDBEngine when
    # persist=True — stub it (incident 20-07-26: fake VCB/BID/VHM 'daily' rows
    # landed in the live experiment log during a pytest run).
    monkeypatch.setattr(main, "_paperlog_no_trade_day", lambda *a, **k: None)


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


def _mock_alpha360() -> MagicMock:
    mock_gen = MagicMock()
    mock_gen.load_live_ohlcv_window.return_value = _make_polars_df()
    return mock_gen


def _fallback_predict(df, horizon):
    """Low P(up) + meta-gate all-reject → weak-market fallback branch."""
    if horizon == 20:
        return (
            {"VCB": [0.3, 0.4, 0.30], "BID": [0.35, 0.37, 0.28], "VHM": [0.4, 0.35, 0.25]},
            {"pnl_threshold_tau": 0.45},
            MagicMock(feature_importances_=None),
            ["feat_a"],
            {"VCB": False, "BID": False, "VHM": False},
        )
    return ({}, {}, MagicMock(), [], {})


def _nobuy_predict(df, horizon):
    """High P(up) + meta-gate all-pass → NOT fallback; the no-buy monitor branch
    is reached instead once the arbitrator rejects every candidate."""
    if horizon == 20:
        return (
            {"VCB": [0.1, 0.2, 0.70], "BID": [0.1, 0.25, 0.65], "VHM": [0.15, 0.25, 0.60]},
            {"pnl_threshold_tau": 0.45},
            MagicMock(feature_importances_=None),
            ["feat_a"],
            {"VCB": True, "BID": True, "VHM": True},
        )
    return ({}, {}, MagicMock(), [], {})


# ── capture point 1: weak-market fallback ────────────────────────────────────


@patch("main.signal_ledger.record_cancelled")
@patch("main._build_fallback_observability_report_vi")
@patch("main._get_live_exec_prices")
@patch("main.mr_score_tickers")
@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_fallback_path_records_cancelled_signals(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
    mock_mr, mock_prices, mock_fb_report, mock_record,
):
    mock_a360_cls.return_value = _mock_alpha360()
    mock_predict.side_effect = _fallback_predict
    mock_eval.return_value = ({}, {})
    mock_mr.return_value = {}
    mock_prices.return_value = {"VCB": 50000.0}
    mock_fb_report.return_value = "<b>fallback report</b>"

    report_html, signal_data_list = main.daily_inference(broadcast=False)

    assert signal_data_list == []
    mock_rte.assert_not_called()             # fallback bypasses execution
    mock_record.assert_called_once()
    candidates, horizon, hold_days = mock_record.call_args.args
    assert horizon == 20                     # default horizon
    assert hold_days == 30                   # _cancelled_hold_days(20)
    assert {c["ticker"] for c in candidates} == {"VCB", "BID", "VHM"}
    # Each captured row carries the rejection reason + probabilities.
    by_ticker = {c["ticker"]: c for c in candidates}
    assert by_ticker["VCB"]["p_up"] == 0.30
    assert by_ticker["VCB"]["reason"]         # non-empty Vietnamese reason


# ── capture point 2: post-arbitration no-buy monitor ─────────────────────────


@patch("main.signal_ledger.record_cancelled")
@patch("main._build_fallback_observability_report_vi")
@patch("main._get_live_exec_prices")
@patch("main.mr_score_tickers")
@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_nobuy_monitor_path_records_cancelled_signals(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
    mock_mr, mock_prices, mock_fb_report, mock_record,
):
    mock_a360_cls.return_value = _mock_alpha360()
    mock_predict.side_effect = _nobuy_predict
    # Arbitrator rejects every candidate (non-BUY) → empty dispatch book.
    mock_eval.return_value = (
        {"VCB": 0, "BID": 1, "VHM": 1},
        {"VCB": {}, "BID": {}, "VHM": {}},
    )
    mock_rte.return_value = ("", [])          # empty post-arbitration book
    mock_mr.return_value = {}
    mock_prices.return_value = {}
    mock_fb_report.return_value = "<b>monitor</b>"

    report_html, signal_data_list = main.daily_inference(broadcast=False)

    mock_rte.assert_called_once()             # NOT fallback → execution ran
    mock_record.assert_called_once()
    candidates, horizon, hold_days = mock_record.call_args.args
    assert horizon == 20 and hold_days == 30
    assert {c["ticker"] for c in candidates} == {"VCB", "BID", "VHM"}


# ── mutual exclusivity ───────────────────────────────────────────────────────


@patch("main.signal_ledger.record_cancelled")
@patch("main._build_fallback_observability_report_vi")
@patch("main._get_live_exec_prices")
@patch("main.mr_score_tickers")
@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_capture_points_are_mutually_exclusive(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
    mock_mr, mock_prices, mock_fb_report, mock_record,
):
    # The fallback branch always `return`s before the no-buy monitor branch can
    # be reached, so at most ONE capture point fires per daily_inference call.
    mock_a360_cls.return_value = _mock_alpha360()
    mock_predict.side_effect = _fallback_predict
    mock_eval.return_value = ({}, {})
    mock_mr.return_value = {}
    mock_prices.return_value = {}
    mock_fb_report.return_value = "<b>fallback report</b>"

    main.daily_inference(broadcast=False)

    assert mock_record.call_count == 1


# ── config kill-switch ───────────────────────────────────────────────────────


def test_disabled_config_skips_both_capture_points(monkeypatch) -> None:
    monkeypatch.setattr(main.CONFIG.trading, "cancelled_signal_tracking_enabled", False)

    # Scenario A: weak-market fallback.
    with patch("main.Alpha360Generator") as m_a360, \
         patch("main.predict_v3_horizon") as m_pred, \
         patch("main.evaluate_trades_batch") as m_eval, \
         patch("main.run_trade_execution") as m_rte, \
         patch("main.mr_score_tickers") as m_mr, \
         patch("main._get_live_exec_prices") as m_px, \
         patch("main._build_fallback_observability_report_vi") as m_fb, \
         patch("main.signal_ledger.record_cancelled") as m_rec:
        m_a360.return_value = _mock_alpha360()
        m_pred.side_effect = _fallback_predict
        m_eval.return_value = ({}, {})
        m_mr.return_value = {}
        m_px.return_value = {}
        m_fb.return_value = "<b>fb</b>"
        main.daily_inference(broadcast=False)
        m_rte.assert_not_called()
        m_rec.assert_not_called()

    # Scenario B: post-arbitration no-buy monitor.
    with patch("main.Alpha360Generator") as m_a360, \
         patch("main.predict_v3_horizon") as m_pred, \
         patch("main.evaluate_trades_batch") as m_eval, \
         patch("main.run_trade_execution") as m_rte, \
         patch("main.mr_score_tickers") as m_mr, \
         patch("main._get_live_exec_prices") as m_px, \
         patch("main._build_fallback_observability_report_vi") as m_fb, \
         patch("main.signal_ledger.record_cancelled") as m_rec:
        m_a360.return_value = _mock_alpha360()
        m_pred.side_effect = _nobuy_predict
        m_eval.return_value = (
            {"VCB": 0, "BID": 1, "VHM": 1},
            {"VCB": {}, "BID": {}, "VHM": {}},
        )
        m_rte.return_value = ("", [])
        m_mr.return_value = {}
        m_px.return_value = {}
        m_fb.return_value = "<b>mon</b>"
        main.daily_inference(broadcast=False)
        m_rec.assert_not_called()
