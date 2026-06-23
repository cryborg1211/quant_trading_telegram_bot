"""Characterization tests for `VNCostModel.simulate` (V4.1 Structural Debt P3).

Hub node `VNCostModel.simulate` (degree-high, src/execution/vn_cost_model.py)
had ZERO direct coverage. These tests pin the cost-model math and rejection
paths so future refactors fail loudly.

Pure module — no DuckDB, no GPU, no mocks.

KEY CONTRACTS pinned here:
  * Fees are charged on the POST-slippage `gross_notional` (filled_price ×
    filled_quantity), NOT on the target price — so fee assertions are written
    relative to `fill.gross_notional`.
  * Prices are ABSOLUTE VND. Passing thousands-VND (e.g. 13.45) raw trips the
    band/ceiling guard — documenting the bug `WalkForwardEngine._prepare`
    prevents (see VN price-scale convention).
  * Participation policy lives on `SlippageModel`, not on the `Order`.
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from src.execution.vn_cost_model import (
    Exchange,
    ExecutionConfig,
    FeeSchedule,
    Fill,
    Order,
    OrderSide,
    ParticipationPolicy,
    RejectionReason,
    SlippageModel,
    VNCostModel,
    price_band_bounds,
    round_to_tick,
    tick_size_vnd,
)


def _make_order(
    *,
    target_price: float = 20_000.0,
    reference_price: float = 20_000.0,
    quantity: int = 1000,
    side: OrderSide = OrderSide.BUY,
    exchange: Exchange = Exchange.HOSE,
    daily_volume: float = 1_000_000.0,
    daily_volatility: float = 0.02,
    is_atc: bool = False,
    atc_volume: float | None = None,
    ticker: str = "VCB",
    timestamp: datetime | None = None,
) -> Order:
    """Factory mapping friendly kwargs to the real `Order` field names."""
    return Order(
        ticker=ticker,
        side=side,
        quantity=quantity,
        target_price=target_price,
        reference_price=reference_price,
        daily_volume=daily_volume,
        daily_volatility=daily_volatility,
        exchange=exchange,
        timestamp=timestamp,
        is_atc=is_atc,
        atc_volume=atc_volume,
    )


# --------------------------------------------------------------------------- #
# 1.2 — FeeSchedule math
# --------------------------------------------------------------------------- #
class TestFeeScheduleMath:
    def test_buy_fee_pct_formula(self) -> None:
        assert FeeSchedule().buy_fee_pct() == pytest.approx(0.0015 * 1.10)

    def test_sell_fee_pct_formula(self) -> None:
        assert FeeSchedule().sell_fee_pct() == pytest.approx(0.0015 * 1.10 + 0.0010)

    def test_round_trip_pct_equals_buy_plus_sell(self) -> None:
        fs = FeeSchedule()
        assert fs.round_trip_pct() == pytest.approx(fs.buy_fee_pct() + fs.sell_fee_pct())

    def test_custom_fees_propagate(self) -> None:
        fs = FeeSchedule(brokerage_per_side=0.002)
        assert fs.buy_fee_pct() == pytest.approx(0.002 * 1.10)


# --------------------------------------------------------------------------- #
# 1.3 — tick_size_vnd
# --------------------------------------------------------------------------- #
class TestTickSizeVnd:
    def test_hose_tier1_below_10k(self) -> None:
        assert tick_size_vnd(9_999.0, Exchange.HOSE) == 10

    def test_hose_tier1_boundary_10k(self) -> None:
        # 10_000 is NOT < 10_000 → falls to the next tier (50).
        assert tick_size_vnd(10_000.0, Exchange.HOSE) == 50

    def test_hose_tier2_below_50k(self) -> None:
        assert tick_size_vnd(49_999.0, Exchange.HOSE) == 50

    def test_hose_tier2_boundary_50k(self) -> None:
        assert tick_size_vnd(50_000.0, Exchange.HOSE) == 100

    def test_hose_tier3_above_50k(self) -> None:
        assert tick_size_vnd(70_000.0, Exchange.HOSE) == 100

    def test_hnx_flat_tick(self) -> None:
        assert tick_size_vnd(5_000.0, Exchange.HNX) == 100
        assert tick_size_vnd(80_000.0, Exchange.HNX) == 100

    def test_upcom_flat_tick(self) -> None:
        assert tick_size_vnd(5_000.0, Exchange.UPCOM) == 100
        assert tick_size_vnd(80_000.0, Exchange.UPCOM) == 100


# --------------------------------------------------------------------------- #
# 1.4 — price_band_bounds
# --------------------------------------------------------------------------- #
class TestPriceBandBounds:
    def test_hose_band_7pct(self) -> None:
        floor, ceiling = price_band_bounds(20_000.0, Exchange.HOSE)
        assert floor == pytest.approx(18_600.0)
        assert ceiling == pytest.approx(21_400.0)

    def test_hnx_band_10pct(self) -> None:
        floor, ceiling = price_band_bounds(20_000.0, Exchange.HNX)
        assert floor == pytest.approx(18_000.0)
        assert ceiling == pytest.approx(22_000.0)

    def test_upcom_band_15pct(self) -> None:
        floor, ceiling = price_band_bounds(20_000.0, Exchange.UPCOM)
        assert floor == pytest.approx(17_000.0)
        assert ceiling == pytest.approx(23_000.0)


# --------------------------------------------------------------------------- #
# 1.6 — simulate happy path
# --------------------------------------------------------------------------- #
class TestVNCostModelSimulateHappyPath:
    def test_buy_fill_absolute_vnd_fees(self) -> None:
        model = VNCostModel()
        fill = model.simulate(_make_order(side=OrderSide.BUY))
        assert fill.is_filled
        # Fees are on the POST-slippage gross_notional, not the target price.
        assert fill.brokerage_paid == pytest.approx(fill.gross_notional * 0.0015)
        assert fill.vat_paid == pytest.approx(fill.brokerage_paid * 0.10)
        assert fill.tax_paid == 0.0  # no transfer tax on a BUY

    def test_sell_fill_absolute_vnd_tax(self) -> None:
        model = VNCostModel()
        fill = model.simulate(_make_order(side=OrderSide.SELL))
        assert fill.is_filled
        assert fill.tax_paid == pytest.approx(fill.gross_notional * 0.0010)

    def test_lot_rounding_down(self) -> None:
        model = VNCostModel()
        fill = model.simulate(_make_order(quantity=150))
        assert fill.filled_quantity == 100  # 150 rounds down to one 100-lot

    def test_gross_notional_equals_price_times_qty(self) -> None:
        model = VNCostModel()
        fill = model.simulate(_make_order())
        assert fill.gross_notional == pytest.approx(
            fill.filled_price * fill.filled_quantity
        )

    def test_atc_fill_no_slippage(self) -> None:
        model = VNCostModel()
        fill = model.simulate(_make_order(is_atc=True, atc_volume=1_000_000))
        assert fill.is_filled
        assert fill.slippage_cost == 0.0
        # ATC clears at target price (already on the tick grid).
        assert fill.filled_price == pytest.approx(20_000.0)

    def test_simulate_batch_is_loop_over_simulate(self) -> None:
        model = VNCostModel()
        orders = [_make_order(side=OrderSide.BUY), _make_order(side=OrderSide.SELL)]
        batch = model.simulate_batch(orders)
        singles = [model.simulate(o) for o in orders]
        assert [f.is_filled for f in batch] == [f.is_filled for f in singles]
        assert [f.filled_price for f in batch] == [f.filled_price for f in singles]


# --------------------------------------------------------------------------- #
# 1.7 — rejection paths
# --------------------------------------------------------------------------- #
class TestVNCostModelRejectPaths:
    def test_reject_nan_price(self) -> None:
        fill = VNCostModel().simulate(_make_order(target_price=float("nan")))
        assert not fill.is_filled
        assert fill.rejection_reason == RejectionReason.INVALID_INPUT

    def test_reject_zero_volume(self) -> None:
        fill = VNCostModel().simulate(_make_order(daily_volume=0.0))
        assert fill.rejection_reason == RejectionReason.ZERO_VOLUME

    def test_reject_below_lot_size(self) -> None:
        fill = VNCostModel().simulate(_make_order(quantity=50))
        assert fill.rejection_reason == RejectionReason.BELOW_LOT_SIZE

    def test_reject_price_outside_band(self) -> None:
        # ref*1.10 = 22_000 > HOSE ceiling 21_400 → mathematically outside band.
        fill = VNCostModel().simulate(_make_order(target_price=22_000.0))
        assert fill.rejection_reason == RejectionReason.PRICE_OUTSIDE_BAND

    def test_reject_buy_at_ceiling(self) -> None:
        _, ceiling = price_band_bounds(20_000.0, Exchange.HOSE)
        fill = VNCostModel().simulate(
            _make_order(target_price=ceiling, side=OrderSide.BUY)
        )
        assert fill.rejection_reason == RejectionReason.PRICE_AT_CEILING_BUY

    def test_reject_sell_at_floor(self) -> None:
        floor, _ = price_band_bounds(20_000.0, Exchange.HOSE)
        fill = VNCostModel().simulate(
            _make_order(target_price=floor, side=OrderSide.SELL)
        )
        assert fill.rejection_reason == RejectionReason.PRICE_AT_FLOOR_SELL

    def test_reject_participation_exceed_policy_reject(self) -> None:
        # Participation policy lives on the SlippageModel, not the Order.
        cfg = ExecutionConfig(slippage=SlippageModel(policy=ParticipationPolicy.REJECT))
        # 200_000 / 1_000_000 = 20% > 10% max → reject.
        fill = VNCostModel(cfg).simulate(_make_order(quantity=200_000))
        assert fill.rejection_reason == RejectionReason.PARTICIPATION_REJECT


# --------------------------------------------------------------------------- #
# 1.8 — VN price-scale convention (the critical contract)
# --------------------------------------------------------------------------- #
class TestVnPriceScaleConvention:
    def test_absolute_vnd_fills_correctly(self) -> None:
        fill = VNCostModel().simulate(
            _make_order(target_price=13_450.0, reference_price=13_450.0)
        )
        assert fill.is_filled

    def test_thousand_scale_rejected_as_ceiling_buy(self) -> None:
        # 13.45 ("thousands-VND" passed raw): the 1.0 VND ceiling tolerance
        # dominates at this scale → rejected. This is the bug _prepare prevents.
        fill = VNCostModel().simulate(
            _make_order(target_price=13.45, reference_price=13.45)
        )
        assert not fill.is_filled
        assert fill.rejection_reason == RejectionReason.PRICE_AT_CEILING_BUY


# --------------------------------------------------------------------------- #
# 1.9 — tick rounding on fill
# --------------------------------------------------------------------------- #
class TestTickRoundingOnFill:
    def test_buy_tick_rounds_up_on_hose(self) -> None:
        fill = VNCostModel().simulate(_make_order(side=OrderSide.BUY))
        tick = tick_size_vnd(fill.filled_price, Exchange.HOSE)
        assert fill.filled_price % tick == 0  # on the legal grid
        # BUY impact pushes price up; round_to_tick rounds further up.
        assert fill.filled_price >= 20_000.0

    def test_sell_tick_rounds_down_on_hose(self) -> None:
        fill = VNCostModel().simulate(_make_order(side=OrderSide.SELL))
        tick = tick_size_vnd(fill.filled_price, Exchange.HOSE)
        assert fill.filled_price % tick == 0
        # SELL impact pushes price down; round_to_tick rounds further down.
        assert fill.filled_price <= 20_000.0

    def test_no_tick_rounding_when_disabled(self) -> None:
        order = _make_order(side=OrderSide.BUY)
        on_grid = VNCostModel(ExecutionConfig(enforce_tick=True)).simulate(order)
        raw = VNCostModel(ExecutionConfig(enforce_tick=False)).simulate(order)
        # Disabling tick enforcement yields the un-snapped continuous price.
        assert raw.filled_price != on_grid.filled_price
        assert on_grid.filled_price % tick_size_vnd(on_grid.filled_price, Exchange.HOSE) == 0
        # Sanity: the raw fill equals the explicit impact price (off-grid here).
        expected_raw = round_to_tick(raw.filled_price, Exchange.HOSE, side=OrderSide.BUY)
        assert not math.isclose(raw.filled_price, expected_raw)
