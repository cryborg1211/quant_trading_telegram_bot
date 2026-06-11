"""Tests for the staggered-tranche rebalance mode in WalkForwardEngine.

Synthetic 4-ticker panel with constant prices; per-ticker constant feature
values double as oracle scores (oracle reads the first feature of the last
bar), so AAA=0.90 and BBB=0.80 are always the top-2 picks, CCC=0.10 never
qualifies, DDD=0.45 is third.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine

N_DAYS = 40
HOLD = 5
PRICE = 20.0          # thousand-VND scale → 20,000 VND after _prepare
SCORES = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.10, "DDD": 0.45}


def _panel() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=N_DAYS).date
    frames = []
    for tk, score in SCORES.items():
        frames.append(pd.DataFrame({
            "ticker": tk, "date": days,
            "open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE,
            "volume": 10_000_000,
            "feat": score,
        }))
    return pd.concat(frames, ignore_index=True)


def _oracle(X: np.ndarray) -> np.ndarray:
    return X[:, -1, 0].astype(np.float64)   # p_up = the feature value


def _engine(**overrides) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=2, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        **overrides,
    )
    return WalkForwardEngine(cfg, _oracle)


@pytest.fixture(scope="module")
def result_and_engine():
    eng = _engine()
    res = eng.run(_panel())
    return res, eng


class TestTrancheMechanics:
    def test_steady_state_tranche_count(self, result_and_engine) -> None:
        _, eng = result_and_engine
        # At the final close the live book is the last HOLD entry cohorts.
        assert len(eng._tranches) == HOLD

    def test_positions_held_exactly_hold_days(self, result_and_engine) -> None:
        res, _ = result_and_engine
        fills = pd.DataFrame(res.fills)
        fills["date"] = pd.to_datetime(fills["date"])
        cal = sorted(fills["date"].unique())
        idx = {d: i for i, d in enumerate(cal)}
        aaa = fills[fills["ticker"] == "AAA"].sort_values("date")
        first_buy = aaa[aaa["side"] == "buy"].iloc[0]
        first_sell = aaa[aaa["side"] == "sell"].iloc[0]
        held = idx[first_sell["date"]] - idx[first_buy["date"]]
        assert held == HOLD

    def test_only_qualifying_names_bought(self, result_and_engine) -> None:
        res, _ = result_and_engine
        fills = pd.DataFrame(res.fills)
        bought = set(fills[fills["side"] == "buy"]["ticker"])
        assert bought == {"AAA", "BBB"}          # top-2; CCC/DDD never picked

    def test_equal_weight_within_tranche(self, result_and_engine) -> None:
        res, _ = result_and_engine
        fills = pd.DataFrame(res.fills)
        buys = fills[fills["side"] == "buy"]
        first_day = buys["date"].min()
        day1 = buys[buys["date"] == first_day]
        notionals = (day1["qty"] * day1["price"]).to_numpy()
        assert len(notionals) == 2
        assert abs(notionals[0] - notionals[1]) / notionals.max() < 0.05

    def test_daily_deployment_is_nav_over_hold(self, result_and_engine) -> None:
        res, _ = result_and_engine
        fills = pd.DataFrame(res.fills)
        buys = fills[fills["side"] == "buy"]
        first_day_notional = -buys[buys["date"] == buys["date"].min()]["cash_flow"].sum()
        # First tranche budget = NAV/HOLD = 200M VND (within lot rounding + fees).
        assert first_day_notional == pytest.approx(200_000_000, rel=0.05)

    def test_steady_state_nearly_fully_invested(self, result_and_engine) -> None:
        res, _ = result_and_engine
        eq = res.equity_curve
        # After HOLD warm-up tranches, gross exposure should approach 1.
        steady = eq.iloc[HOLD + 3:]
        assert steady["gross_exposure"].mean() > 0.85

    def test_regime_zero_day_skips_new_tranche_only(self) -> None:
        days = pd.bdate_range("2024-01-02", periods=N_DAYS)
        zero_day = days[10]
        p_bull = pd.Series(1.0, index=days)
        p_bull.loc[zero_day] = 0.0

        eng = _engine()
        res = eng.run(_panel(), p_bull_series=p_bull)
        fills = pd.DataFrame(res.fills)
        fills["date"] = pd.to_datetime(fills["date"])
        on_zero = fills[fills["date"] == zero_day]
        assert len(on_zero[on_zero["side"] == "buy"]) == 0      # no new tranche
        assert len(on_zero[on_zero["side"] == "sell"]) > 0      # expiry still exits

    def test_grid_mode_unaffected(self) -> None:
        cfg = WalkForwardConfig(
            seq_len=1, feature_cols=["feat"],
            rebalance_mode="grid", rebalance_frequency=5,
            max_positions=2, signal_threshold=0.40,
            liquid_top_n=None, initial_capital=1_000_000_000.0,
        )
        eng = WalkForwardEngine(cfg, _oracle)
        res = eng.run(_panel())
        assert len(res.equity_curve) == N_DAYS    # smoke: grid path still runs
