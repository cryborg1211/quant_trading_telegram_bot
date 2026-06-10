"""Regression tests for the thousand-VND price-scale fix in WalkForwardEngine.

The parquet OHLCV shards store prices in thousands of VND (13.45 = 13,450 VND)
while VNCostModel's tick grid (10/50/100 VND) and share-quantity math assume
absolute VND.  Before the fix, the engine fed thousand-scale prices straight
into the cost model: a 13.45 BUY tick-rounded UP to 20 (+49%), a 9.8 SELL
rounded DOWN to 0 (total loss), and qty = w*NAV/13.45 inflated share counts
1000x.  The fix scales OHLC to absolute VND inside `_prepare`.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine
from src.execution.vn_cost_model import (
    Exchange,
    ExecutionConfig,
    Order,
    OrderSide,
    VNCostModel,
)


def _panel(n_days: int = 5, close: float = 13.45) -> pd.DataFrame:
    """Tiny single-ticker panel in thousand-VND scale with one feature col."""
    days = pd.bdate_range("2024-01-02", periods=n_days).date
    return pd.DataFrame({
        "ticker": ["AAA"] * n_days,
        "date": days,
        "open": [close] * n_days,
        "high": [close * 1.01] * n_days,
        "low": [close * 0.99] * n_days,
        "close": [close] * n_days,
        "volume": [1_000_000] * n_days,
        "feat": [0.0] * n_days,
    })


def _engine(price_unit_vnd: float = 1000.0) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"], price_unit_vnd=price_unit_vnd,
        liquid_top_n=None,
    )
    oracle = lambda X: np.full((X.shape[0],), 0.5)  # noqa: E731
    return WalkForwardEngine(cfg, oracle)


class TestPrepareScalesPrices:
    def test_ohlc_scaled_to_absolute_vnd(self) -> None:
        eng = _engine()
        eng._prepare(_panel(), None)
        row = eng._day_index[date(2024, 1, 3)]["AAA"]
        assert row["close"] == pytest.approx(13_450.0)
        assert row["open"] == pytest.approx(13_450.0)
        assert row["high"] == pytest.approx(13_450.0 * 1.01)
        assert row["low"] == pytest.approx(13_450.0 * 0.99)
        # ref_price (prior close) must be on the same scale as close.
        assert row["ref_price"] == pytest.approx(13_450.0)

    def test_unit_scale_is_noop(self) -> None:
        eng = _engine(price_unit_vnd=1.0)
        eng._prepare(_panel(), None)
        row = eng._day_index[date(2024, 1, 3)]["AAA"]
        assert row["close"] == pytest.approx(13.45)

    def test_feature_columns_untouched(self) -> None:
        eng = _engine()
        eng._prepare(_panel(), None)
        assert float(eng.ticker_frames["AAA"]["feat"].iloc[0]) == 0.0

    def test_returns_are_scale_invariant(self) -> None:
        scaled, unscaled = _engine(1000.0), _engine(1.0)
        panel = _panel()
        panel["close"] = [13.0, 13.5, 13.2, 13.8, 14.0]
        scaled._prepare(panel.copy(), None)
        unscaled._prepare(panel.copy(), None)
        r_s = scaled.ticker_frames["AAA"]["ret"].dropna().to_numpy()
        r_u = unscaled.ticker_frames["AAA"]["ret"].dropna().to_numpy()
        np.testing.assert_allclose(r_s, r_u)


class TestCostModelSanityAtAbsoluteVnd:
    """Demonstrates the bug magnitude: same order, two price scales."""

    @staticmethod
    def _fill(price: float):
        model = VNCostModel(ExecutionConfig())
        order = Order(
            ticker="AAA", side=OrderSide.BUY, quantity=10_000,
            target_price=price, reference_price=price,
            daily_volume=1_000_000, daily_volatility=0.02,
            exchange=Exchange.HOSE,
            timestamp=datetime.combine(date(2024, 1, 3), dtime(14, 35)),
            is_atc=True, atc_volume=150_000.0,
        )
        return model.simulate(order)

    def test_absolute_vnd_cost_is_sane(self) -> None:
        fill = self._fill(13_450.0)
        assert fill.is_filled
        notional = fill.filled_price * fill.filled_quantity
        # Explicit fees + at most one tick of rounding — well under 1%.
        assert fill.total_cost / notional < 0.01
        # ATC fill at the clearing price on the 50-VND grid: no rounding move.
        assert fill.filled_price == pytest.approx(13_450.0)

    def test_thousand_scale_order_is_pathological(self) -> None:
        # The OLD bug path: at 13.45 "VND" the ±1-VND ceiling tolerance spans
        # 7.4% of the price, so a plain at-market BUY is spuriously rejected
        # as PRICE_AT_CEILING_BUY.  (Orders that DID fill hit the 10-VND tick
        # grid instead: 13.45 → 20, +49% phantom slippage.)  Documents why
        # `_prepare` must scale prices to absolute VND.
        fill = self._fill(13.45)
        assert not fill.is_filled
        assert fill.rejection_reason.value == "price_at_ceiling_buy"
