"""Persist-gate unit tests for the local-dashboard preview path (P2).

The dashboard renders live BUY signals by calling ``main.daily_inference`` —
but that path normally mutates the cron DuckDB book on every call (portfolio
update, RL prediction log, sentiment paperlog). P2 added a ``persist: bool =
True`` gate to ``run_trade_execution`` (and a passthrough on
``daily_inference``) so a dashboard "preview" can run inference WITHOUT any DB
write side-effects.

These tests pin that contract:
    * Test A — ``run_trade_execution(..., persist=False)`` performs ZERO writes
      (no ``update_live_performance`` / ``process_daily_trades`` / RL log /
      RL backfill / paperlog log / paperlog backfill).
    * Test B — ``run_trade_execution(...)`` with the DEFAULT (``persist=True``)
      performs all those writes — proving the gate is opt-in, not default-off.
    * Test C — ``daily_inference(..., persist=False)`` passes ``persist=False``
      straight through to ``run_trade_execution`` (default ``True`` otherwise).

All heavy serve dependencies are mocked. No real DuckDBEngine singleton, no
parquet shards, and no Telegram / Gemini calls are made — the assertions are on
mock call-counts, mirroring the existing
``tests/test_daily_inference_integration.py`` style.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl


# --------------------------------------------------------------------------- #
# Shared helpers / fixtures
# --------------------------------------------------------------------------- #

_TICKERS = ["VCB", "BID", "VHM"]


def _make_polars_df(tickers: list[str] | None = None) -> pl.DataFrame:
    """Minimal live OHLCV window frame (matches the integration-test shape)."""
    tickers = tickers or _TICKERS
    rows = []
    for t in tickers:
        for i in range(5):
            rows.append({
                "ticker": t,
                "date": f"2024-06-{10 + i:02d}",
                "open": 50000.0,
                "high": 51000.0,
                "low": 49000.0,
                "close": 50500.0,
                "volume": 1000000.0,
                "raw_close": 50500.0,
            })
    return pl.DataFrame(rows)


def _rte_kwargs() -> dict:
    """Build a valid ``run_trade_execution`` kwargs dict (sans ``persist``)."""
    horizon_predictions = {
        "5d": {"VCB": [0.1, 0.2, 0.70], "BID": [0.1, 0.25, 0.65]},
        "20d": {"VCB": [0.15, 0.25, 0.60], "BID": [0.2, 0.3, 0.50]},
    }
    return {
        "top_buy_signals": ["VCB", "BID"],
        "final_decisions": {"VCB": 2, "BID": 2},
        "all_sentiments": {
            "VCB": {"sentiment_score": 0.4, "reasoning_vi": "good"},
            "BID": {"sentiment_score": 0.3, "reasoning_vi": "ok"},
        },
        "stacking_predictions": horizon_predictions,
        "latest_df": _make_polars_df().to_pandas(),
        "xgb_model_5d": MagicMock(feature_importances_=None),
        "selected_features_5d": ["feat_a", "feat_b"],
        "horizon": 20,
        "broadcast": False,
    }


def _patched_run_trade_execution(persist_kwargs: dict):
    """Context-manager stack that mocks every write dependency of RTE.

    Returns a dict of the active mocks so the test can assert call-counts.
    Patches are applied to the ``main`` module namespace.
    """
    import main  # local import keeps collection cheap; main is heavy

    manager = MagicMock(name="PortfolioManager_instance")
    # `manager.db` is read for the RL/paperlog block — give it a sentinel.
    manager.db = MagicMock(name="db")

    patchers = {
        "PortfolioManager": patch.object(main, "PortfolioManager", return_value=manager),
        "_get_live_exec_prices": patch.object(
            main, "_get_live_exec_prices",
            return_value={"VCB": 50000.0, "BID": 27000.0},
        ),
        "_log_rl_predictions": patch.object(main, "_log_rl_predictions", return_value=0),
        "_backfill_rl_outcomes": patch.object(main, "_backfill_rl_outcomes", return_value=0),
        "_log_sentiment_entry_paperlog": patch.object(
            main, "_log_sentiment_entry_paperlog", return_value=0
        ),
        "_backfill_paperlog_outcomes": patch.object(
            main, "_backfill_paperlog_outcomes", return_value=0
        ),
        "_build_feature_explanation": patch.object(
            main, "_build_feature_explanation", return_value=("", ""),
        ),
        "_dispatch_signals": patch.object(
            main, "_dispatch_signals", return_value=[{"ticker": "VCB"}],
        ),
        "_load_v3_bot": patch.object(main, "_load_v3_bot", return_value=MagicMock(strategy=None)),
        "TelegramBot": patch.object(main, "TelegramBot", return_value=MagicMock()),
    }
    started = {name: p.start() for name, p in patchers.items()}
    # Expose the manager so the test can assert on its write methods.
    started["_manager"] = manager
    try:
        kwargs = _rte_kwargs()
        kwargs.update(persist_kwargs)
        main.run_trade_execution(**kwargs)
    finally:
        for p in patchers.values():
            p.stop()
    return started


# --------------------------------------------------------------------------- #
# Test A — persist=False writes nothing
# --------------------------------------------------------------------------- #

def test_persist_false_no_db_writes():
    """persist=False → every DB write side-effect is skipped."""
    mocks = _patched_run_trade_execution({"persist": False})
    manager = mocks["_manager"]

    # Portfolio write block skipped.
    manager.update_live_performance.assert_not_called()
    manager.process_daily_trades.assert_not_called()

    # RL prediction log + backfill skipped.
    mocks["_log_rl_predictions"].assert_not_called()
    mocks["_backfill_rl_outcomes"].assert_not_called()

    # Sentiment-entry paperlog log + backfill skipped.
    mocks["_log_sentiment_entry_paperlog"].assert_not_called()
    mocks["_backfill_paperlog_outcomes"].assert_not_called()

    # Dispatch still runs — caller still receives signals (no broadcast).
    mocks["_dispatch_signals"].assert_called_once()


# --------------------------------------------------------------------------- #
# Test B — persist=True (default) writes happen
# --------------------------------------------------------------------------- #

def test_persist_true_default_writes():
    """persist=True (the default) → all DB write side-effects run."""
    # Omit the kwarg entirely to prove the DEFAULT is True (cron-safe).
    mocks = _patched_run_trade_execution({})
    manager = mocks["_manager"]

    manager.update_live_performance.assert_called_once()
    manager.process_daily_trades.assert_called_once()

    mocks["_log_rl_predictions"].assert_called_once()
    mocks["_backfill_rl_outcomes"].assert_called_once()

    # Paperlog runs only when sentiment_entry_enabled; default config is True.
    from config.settings import CONFIG
    if CONFIG.trading.sentiment_entry_enabled:
        mocks["_log_sentiment_entry_paperlog"].assert_called_once()
        mocks["_backfill_paperlog_outcomes"].assert_called_once()

    mocks["_dispatch_signals"].assert_called_once()


# --------------------------------------------------------------------------- #
# Test C — daily_inference passes persist through to run_trade_execution
# --------------------------------------------------------------------------- #

@patch("main.run_trade_execution")
@patch("main.evaluate_trades_batch")
@patch("main.predict_v3_horizon")
@patch("main.Alpha360Generator")
def test_daily_inference_persist_passthrough(
    mock_a360_cls, mock_predict, mock_eval, mock_rte,
):
    """daily_inference(persist=False) → run_trade_execution(persist=False)."""
    mock_gen = MagicMock()
    mock_gen.load_live_ohlcv_window.return_value = _make_polars_df()
    mock_a360_cls.return_value = mock_gen

    def predict_side_effect(df, horizon):
        if horizon == 20:  # PRIMARY dispatch horizon
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
    mock_rte.return_value = ("<b>preview</b>", [{"ticker": "VCB"}])

    from main import daily_inference
    daily_inference(broadcast=False, persist=False)

    mock_rte.assert_called_once()
    passed = mock_rte.call_args.kwargs.get("persist")
    assert passed is False, f"expected persist=False forwarded, got {passed!r}"
