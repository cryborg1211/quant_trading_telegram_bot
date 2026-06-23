"""Wiring tests for `run_oos` + `equity_metrics` (V4.1 Structural Debt P3).

The hub node the context doc called `run_backtest.main` (degree 97) is actually
the pair `run_oos` / `_build_wf_config` — there is no `main` function at that
node. `_build_wf_config` is already covered by `test_run_backtest_config.py`;
this module fills the remaining gaps:

  * `run_oos` — engine wiring. `WalkForwardEngine.run` is mocked so no dataset
    materialization happens; we assert the equity curve is built and trimmed to
    `date >= cutoff`, and that the engine is invoked exactly once.
  * `equity_metrics` — pure PnL/return/drawdown math.

`make_ensemble_oracle` is invoked with a MagicMock ensemble: the oracle closure
is built but never called (the engine run is mocked), so no model is needed.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from run_backtest import _build_wf_config, equity_metrics, monthly_net_sharpe, run_oos
from src.backtest.pipeline import RunConfig
from src.execution.vn_cost_model import ExecutionConfig

_CUTOFF = date(2024, 1, 8)
_ENGINE_RUN = "src.backtest.walk_forward.WalkForwardEngine.run"


def _panel(cutoff: date = _CUTOFF, n: int = 12) -> pl.DataFrame:
    base = cutoff - timedelta(days=5)
    dates = [base + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": dates, "ticker": ["AAA"] * n, "feat": [0.5] * n})


def _mock_equity_curve(cutoff: date = _CUTOFF) -> pd.DataFrame:
    # Two rows before the cutoff, two on/after — exercises the trim.
    dts = [cutoff - timedelta(days=2), cutoff - timedelta(days=1), cutoff, cutoff + timedelta(days=1)]
    return pd.DataFrame({
        "date": dts,
        "nav": [1_000_000.0, 1_010_000.0, 1_020_000.0, 1_030_000.0],
        "daily_return": [0.0, 0.010, 0.0099, 0.0098],
    })


def _mock_result(cutoff: date = _CUTOFF) -> MagicMock:
    res = MagicMock()
    res.equity_curve = _mock_equity_curve(cutoff)
    return res


# --------------------------------------------------------------------------- #
# _build_wf_config — only the assertion NOT already in test_run_backtest_config
# --------------------------------------------------------------------------- #
class TestBuildWfConfigExec:
    def test_exec_config_is_execution_config(self) -> None:
        wf = _build_wf_config(["feat"], _CUTOFF, RunConfig())
        assert isinstance(wf.exec_config, ExecutionConfig)


# --------------------------------------------------------------------------- #
# run_oos — engine wiring (WalkForwardEngine.run mocked)
# --------------------------------------------------------------------------- #
class TestRunOosWiring:
    def test_run_oos_returns_dataframe(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()):
            out = run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig())
        assert isinstance(out, pd.DataFrame)

    def test_run_oos_columns_include_date_nav_return(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()):
            out = run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig())
        assert {"date", "nav", "daily_return"}.issubset(out.columns)

    def test_run_oos_equity_curve_trimmed_to_cutoff(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()):
            out = run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig())
        assert (pd.to_datetime(out["date"]).dt.date >= _CUTOFF).all()
        assert len(out) == 2  # only the cutoff + post-cutoff rows survive

    def test_run_oos_calls_engine_run_once(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()) as mrun:
            run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig())
        assert mrun.call_count == 1

    def test_run_oos_mode_tranche_passes_to_config(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()), \
                patch("run_backtest._build_wf_config", wraps=_build_wf_config) as spy:
            run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig(), mode="tranche")
        assert spy.call_args.args[3] == "tranche"

    def test_run_oos_mode_grid_passes_to_config(self) -> None:
        with patch(_ENGINE_RUN, return_value=_mock_result()), \
                patch("run_backtest._build_wf_config", wraps=_build_wf_config) as spy:
            run_oos(_panel(), ["feat"], MagicMock(), [], _CUTOFF, RunConfig(), mode="grid")
        assert spy.call_args.args[3] == "grid"


# --------------------------------------------------------------------------- #
# equity_metrics — pure math
# --------------------------------------------------------------------------- #
class TestEquityMetrics:
    def test_net_pnl_correct(self) -> None:
        eq = pd.DataFrame({"nav": [1_000_000.0, 1_100_000.0], "daily_return": [0.0, 0.1]})
        assert equity_metrics(eq, 1_000_000.0)["net_pnl"] == pytest.approx(100_000.0)

    def test_total_return_correct(self) -> None:
        eq = pd.DataFrame({"nav": [1_000_000.0, 1_100_000.0], "daily_return": [0.0, 0.1]})
        assert equity_metrics(eq, 1_000_000.0)["total_return"] == pytest.approx(0.1)

    def test_max_drawdown_negative(self) -> None:
        eq = pd.DataFrame({"nav": [100.0, 80.0, 120.0], "daily_return": [0.0, -0.2, 0.5]})
        assert equity_metrics(eq, 100.0)["max_drawdown"] < 0

    def test_empty_nav_returns_initial_capital(self) -> None:
        eq = pd.DataFrame({"nav": pd.Series([], dtype=float), "daily_return": pd.Series([], dtype=float)})
        assert equity_metrics(eq, 555.0)["final_nav"] == 555.0


class TestMonthlyNetSharpe:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(0)
        eq = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=40),
            "nav": np.linspace(1e6, 1.1e6, 40),
            "daily_return": rng.normal(0.0, 0.01, 40),
        })
        out = monthly_net_sharpe(eq)
        assert isinstance(out, pd.Series)
