"""Paperlog starvation fix (19-07-26): no-trade days must still log.

`sentiment_entry_paperlog` got zero `source='daily'` rows for 13 trading days
(last row 06-30) because only run_trade_execution's happy path wrote it — the
τ-gate kept the book empty all July, so every EOD run exited through one of
three paperlog-less paths: (1) weak-market fallback return, (2) empty-universe
fallback return, (3) empty-live-prices early return. All three now log via
`_paperlog_no_trade_day` / `_paperlog_snapshot_and_backfill`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

import main
from config.settings import CONFIG


_PREDS = {"5d": {"VCB": [0.2, 0.3, 0.5]}, "20d": {}}


# ---------------------------------------------------------------------------
# _paperlog_no_trade_day gating
# ---------------------------------------------------------------------------


def test_no_write_when_persist_false(monkeypatch):
    called = []
    monkeypatch.setattr(main, "_paperlog_snapshot_and_backfill",
                        lambda *a, **k: called.append(a))
    main._paperlog_no_trade_day(False, _PREDS, {}, {})
    assert called == []


def test_no_write_when_config_disabled(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "sentiment_entry_enabled", False)
    called = []
    monkeypatch.setattr(main, "_paperlog_snapshot_and_backfill",
                        lambda *a, **k: called.append(a))
    main._paperlog_no_trade_day(True, _PREDS, {}, {})
    assert called == []


def test_writes_via_engine_when_enabled(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "sentiment_entry_enabled", True)
    called = []
    monkeypatch.setattr(main, "_paperlog_snapshot_and_backfill",
                        lambda db, *a, **k: called.append(db))
    fake_engine = MagicMock()
    with patch("src.data.db_engine.DuckDBEngine", return_value=fake_engine):
        main._paperlog_no_trade_day(True, _PREDS, {}, {})
    assert called == [fake_engine]


def test_engine_failure_never_raises(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "sentiment_entry_enabled", True)
    with patch("src.data.db_engine.DuckDBEngine", side_effect=RuntimeError("locked")):
        main._paperlog_no_trade_day(True, _PREDS, {}, {})  # must not raise


# ---------------------------------------------------------------------------
# _paperlog_snapshot_and_backfill
# ---------------------------------------------------------------------------


def test_snapshot_logs_source_daily(monkeypatch):
    log_calls, backfill_calls = [], []
    monkeypatch.setattr(main, "_log_sentiment_entry_paperlog",
                        lambda **kw: log_calls.append(kw) or 1)
    monkeypatch.setattr(main, "_backfill_paperlog_outcomes",
                        lambda db: backfill_calls.append(db) or 0)
    db = MagicMock()
    main._paperlog_snapshot_and_backfill(db, _PREDS, {"VCB": 1}, {"VCB": {}})
    assert len(log_calls) == 1
    kw = log_calls[0]
    assert kw["source"] == "daily"
    assert kw["candidate_tickers"] == ["VCB"]
    assert kw["stacking_5d"] == _PREDS["5d"]
    assert backfill_calls == [db]


def test_snapshot_swallows_write_failure(monkeypatch):
    monkeypatch.setattr(
        main, "_log_sentiment_entry_paperlog",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("db locked")),
    )
    main._paperlog_snapshot_and_backfill(MagicMock(), _PREDS, {}, {})  # no raise


# ---------------------------------------------------------------------------
# Wiring: the weak-market fallback exit logs (the 13-day starvation path)
# ---------------------------------------------------------------------------


def _make_polars_df() -> pl.DataFrame:
    rows = []
    for t in ["VCB", "BID", "VHM"]:
        for i in range(5):
            rows.append({
                "ticker": t, "date": f"2024-06-{10+i:02d}", "open": 50000.0,
                "high": 51000.0, "low": 49000.0, "close": 50500.0,
                "volume": 1000000.0, "raw_close": 50500.0,
            })
    return pl.DataFrame(rows)


def _fallback_predict(df, horizon):
    if horizon == 20:
        return (
            {"VCB": [0.3, 0.4, 0.30], "BID": [0.35, 0.37, 0.28], "VHM": [0.4, 0.35, 0.25]},
            {"pnl_threshold_tau": 0.45},
            MagicMock(feature_importances_=None),
            ["feat_a"],
            {"VCB": False, "BID": False, "VHM": False},
        )
    return ({}, {}, MagicMock(), [], {})


@patch("main._paperlog_no_trade_day")
@patch("main._build_fallback_observability_report_vi", return_value="<b>fb</b>")
@patch("main._get_live_exec_prices", return_value={})
@patch("main.mr_score_tickers", return_value={})
@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch", return_value=({}, {}))
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_fallback_exit_writes_paperlog(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
    mock_mr, mock_prices, mock_fb, mock_pl,
):
    gen = MagicMock()
    gen.load_live_ohlcv_window.return_value = _make_polars_df()
    mock_a360_cls.return_value = gen
    mock_predict.side_effect = _fallback_predict

    main.daily_inference(broadcast=False)

    mock_rte.assert_not_called()
    mock_pl.assert_called_once()
    persist_arg = mock_pl.call_args.args[0]
    assert persist_arg is True  # cron path default


# ---------------------------------------------------------------------------
# Wiring: run_trade_execution's empty-live-prices early return logs
# ---------------------------------------------------------------------------


@patch("main._get_live_exec_prices", return_value={})
@patch("main.PortfolioManager")
def test_empty_live_prices_early_return_writes_paperlog(mock_pm_cls, mock_prices, monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "sentiment_entry_enabled", True)
    called = []
    monkeypatch.setattr(main, "_paperlog_snapshot_and_backfill",
                        lambda db, *a, **k: called.append(db))
    fake_db = MagicMock()
    mock_pm_cls.return_value = MagicMock(db=fake_db)

    html, dispatched = main.run_trade_execution(
        top_buy_signals=["VCB"],
        final_decisions={"VCB": 2},
        all_sentiments={"VCB": {}},
        stacking_predictions=_PREDS,
        latest_df=pd.DataFrame(),
        xgb_model_5d=MagicMock(),
        selected_features_5d=[],
        horizon=20,
        broadcast=False,
        persist=True,
    )

    assert html == "" and dispatched == []
    assert called == [fake_db]
