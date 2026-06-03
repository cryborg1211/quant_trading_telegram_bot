"""Targeted VN-microstructure tests for vn_cost_model.py.

Each test asserts ONE rule.  If any of these fail, the cost model is lying about
real VN execution and the LSTM will die a slow death.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ast
with open("src/execution/vn_cost_model.py", encoding="utf-8") as f:
    ast.parse(f.read())
print("AST parse OK")

import math
import numpy as np
import polars as pl

from src.execution.vn_cost_model import (
    Exchange, OrderSide, ParticipationPolicy, RejectionReason,
    FeeSchedule, SlippageModel, ExecutionConfig,
    Order, Fill, VNCostModel,
    round_down_to_lot, price_band_bounds, is_at_ceiling, is_at_floor,
    round_to_tick, tick_size_vnd, rejection_breakdown,
)

# ============================================================================
# TEST 1: Lot-size rounding (HOSE/HNX/UPCOM all 100)
# ============================================================================
assert round_down_to_lot(250) == 200
assert round_down_to_lot(99) == 0
assert round_down_to_lot(100) == 100
assert round_down_to_lot(199) == 100
assert round_down_to_lot(1_234_567) == 1_234_500
try:
    round_down_to_lot(-50)
except ValueError as e:
    pass
else:
    raise AssertionError("negative qty should raise")
print("TEST 1  round_down_to_lot       250→200  99→0  199→100  1234567→1234500  ok")


# ============================================================================
# TEST 2: Price band per exchange (HOSE±7, HNX±10, UPCOM±15)
# ============================================================================
ref = 50_000.0
floor_hose, ceil_hose = price_band_bounds(ref, Exchange.HOSE)
floor_hnx,  ceil_hnx  = price_band_bounds(ref, Exchange.HNX)
floor_upcm, ceil_upcm = price_band_bounds(ref, Exchange.UPCOM)
assert abs(ceil_hose - 53_500.0) < 1e-6, ceil_hose
assert abs(floor_hose - 46_500.0) < 1e-6, floor_hose
assert abs(ceil_hnx  - 55_000.0) < 1e-6, ceil_hnx
assert abs(ceil_upcm - 57_500.0) < 1e-6, ceil_upcm
print(f"TEST 2  price_band_bounds       HOSE[{floor_hose:.0f},{ceil_hose:.0f}]  "
      f"HNX[{floor_hnx:.0f},{ceil_hnx:.0f}]  UPCOM[{floor_upcm:.0f},{ceil_upcm:.0f}]  ok")


# ============================================================================
# TEST 3: Tick size schedule (HOSE three-tier)
# ============================================================================
assert tick_size_vnd(5_000, Exchange.HOSE) == 10
assert tick_size_vnd(25_000, Exchange.HOSE) == 50
assert tick_size_vnd(100_000, Exchange.HOSE) == 100
assert tick_size_vnd(5_000, Exchange.HNX) == 100
assert tick_size_vnd(5_000, Exchange.UPCOM) == 100
# round_to_tick conservatively (buy rounds up — worse for us)
assert round_to_tick(25_037.0, Exchange.HOSE, side=OrderSide.BUY) == 25_050
assert round_to_tick(25_037.0, Exchange.HOSE, side=OrderSide.SELL) == 25_000
print("TEST 3  tick_size + round_to_tick  HOSE 3-tier, BUY rounds up, SELL rounds down  ok")


# ============================================================================
# TEST 4: Asymmetric fees — sell tax only applied on sell leg
# ============================================================================
fs = FeeSchedule(brokerage_per_side=0.0015, sell_transfer_tax=0.001, vat_on_brokerage=0.10)
# Buy: brokerage * (1 + VAT) = 0.0015 * 1.10 = 0.00165
assert abs(fs.buy_fee_pct() - 0.00165) < 1e-9, fs.buy_fee_pct()
# Sell: brokerage * (1 + VAT) + tax = 0.00165 + 0.001 = 0.00265
assert abs(fs.sell_fee_pct() - 0.00265) < 1e-9, fs.sell_fee_pct()
# Round trip = buy + sell = 0.0043 (NOT 0.005 from a naive 0.0025 × 2)
assert abs(fs.round_trip_pct() - 0.00430) < 1e-9, fs.round_trip_pct()
# Sanity: institutional flat 0.8% V1 cost was BUY-symmetric; real VN is asymmetric.
# A naive flat-cost backtest would have undercounted sell-side churn by 0.001 / 0.004 = 25%.
print(f"TEST 4  asymmetric fees  buy_fee={fs.buy_fee_pct()*100:.3f}%  sell_fee={fs.sell_fee_pct()*100:.3f}%  "
      f"round_trip={fs.round_trip_pct()*100:.3f}%  ok")


# ============================================================================
# TEST 5: Square-root impact rule + participation penalty
# ============================================================================
sm = SlippageModel(alpha=1.0, max_participation=0.10, excess_exponent=2.0,
                   policy=ParticipationPolicy.PENALIZE)
# Q=1k, ADV=100k, σ=0.025 → participation=1%, impact = 1.0 * 0.025 * sqrt(0.01) = 0.0025
impact_1pct, part_1pct = sm.impact_pct(1_000, 100_000, 0.025)
assert abs(impact_1pct - 0.025 * 0.1) < 1e-9, impact_1pct
assert abs(part_1pct - 0.01) < 1e-9
# Q=10k, ADV=100k → participation=10% (at max), no penalty yet
# impact = 1.0 * 0.025 * sqrt(0.1) = 0.025 * 0.3162 = 0.00791
impact_at_max, part_at_max = sm.impact_pct(10_000, 100_000, 0.025)
assert abs(impact_at_max - 0.025 * math.sqrt(0.1)) < 1e-9
assert abs(part_at_max - 0.10) < 1e-9
# Q=20k, ADV=100k → participation=20% (2× max)
# base = 0.025 * sqrt(0.2) = 0.01118
# excess_ratio = (0.20 - 0.10) / 0.10 = 1.0
# penalty = (1 + 1.0)^2 = 4.0
# total = 0.01118 * 4.0 = 0.04472
impact_2x, part_2x = sm.impact_pct(20_000, 100_000, 0.025)
expected_2x = 0.025 * math.sqrt(0.2) * (1.0 + (0.20-0.10)/0.10) ** 2
assert abs(impact_2x - expected_2x) < 1e-9, (impact_2x, expected_2x)
assert abs(part_2x - 0.20) < 1e-9
print(f"TEST 5  sqrt impact  1%:{impact_1pct*100:.4f}%  10%:{impact_at_max*100:.4f}%  "
      f"20% w/penalty:{impact_2x*100:.4f}% (4× scale-up over 10%)  ok")


# ============================================================================
# TEST 6: BUY rejected at ceiling (Trắng bên bán)
# ============================================================================
model = VNCostModel(ExecutionConfig())

# Reference 50,000 → HOSE ceiling = 53,500
order_buy_ceiling = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=1000,
    target_price=53_500.0, reference_price=50_000.0,
    daily_volume=1_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill = model.simulate(order_buy_ceiling)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.PRICE_AT_CEILING_BUY, fill.rejection_reason
assert fill.signed_cash_flow == 0.0
print(f"TEST 6  BUY at ceiling rejected  reason={fill.rejection_reason.value}  ok")

# Same order as SELL → should fill (selling at ceiling is fine)
order_sell_ceiling = Order(
    ticker="VCB", side=OrderSide.SELL, quantity=1000,
    target_price=53_500.0, reference_price=50_000.0,
    daily_volume=1_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill_sell = model.simulate(order_sell_ceiling)
assert fill_sell.is_filled, f"sell at ceiling should fill, got {fill_sell.rejection_reason}"
print(f"         SELL at ceiling fills   filled_qty={fill_sell.filled_quantity}  ok")


# ============================================================================
# TEST 7: SELL rejected at floor (Trắng bên mua)
# ============================================================================
order_sell_floor = Order(
    ticker="ROS", side=OrderSide.SELL, quantity=1000,
    target_price=46_500.0, reference_price=50_000.0,  # at HOSE floor
    daily_volume=1_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill = model.simulate(order_sell_floor)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.PRICE_AT_FLOOR_SELL, fill.rejection_reason
print(f"TEST 7  SELL at floor rejected  reason={fill.rejection_reason.value}  ok")


# ============================================================================
# TEST 8: Order outside the band → PRICE_OUTSIDE_BAND
# ============================================================================
order_oob = Order(
    ticker="HVN", side=OrderSide.BUY, quantity=1000,
    target_price=60_000.0, reference_price=50_000.0,  # +20% — far above HOSE ceiling
    daily_volume=1_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill = model.simulate(order_oob)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.PRICE_OUTSIDE_BAND
print(f"TEST 8  Order outside band rejected  reason={fill.rejection_reason.value}  ok")


# ============================================================================
# TEST 9: Below lot size → rejected
# ============================================================================
order_small = Order(
    ticker="FPT", side=OrderSide.BUY, quantity=50,
    target_price=130_000.0, reference_price=130_000.0,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill = model.simulate(order_small)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.BELOW_LOT_SIZE
print(f"TEST 9  Below 100-share lot rejected  intended=50  ok")


# ============================================================================
# TEST 10: Participation policies — REJECT, PENALIZE, CAP
# ============================================================================
# Q = 50,000, ADV = 100,000 → participation 50% (5x max).
base_order = Order(
    ticker="DCM", side=OrderSide.BUY, quantity=50_000,
    target_price=20_000.0, reference_price=20_000.0,
    daily_volume=100_000, daily_volatility=0.025, exchange=Exchange.HOSE,
)
# Policy REJECT
model_rej = VNCostModel(ExecutionConfig(
    slippage=SlippageModel(alpha=1.0, max_participation=0.10,
                           policy=ParticipationPolicy.REJECT)))
f_rej = model_rej.simulate(base_order)
assert not f_rej.is_filled
assert f_rej.rejection_reason == RejectionReason.PARTICIPATION_REJECT, f_rej.rejection_reason
print(f"TEST 10  REJECT policy   participation={f_rej.participation_pct*100:.1f}%  rejected  ok")

# Policy PENALIZE — fills but at brutal impact
model_pen = VNCostModel(ExecutionConfig(
    slippage=SlippageModel(alpha=1.0, max_participation=0.10,
                           policy=ParticipationPolicy.PENALIZE, excess_exponent=2.0)))
f_pen = model_pen.simulate(base_order)
assert f_pen.is_filled
# Sanity: cost as % should be enormous (penalised square-root → ~10%+ impact alone)
print(f"         PENALIZE policy participation={f_pen.participation_pct*100:.1f}%  "
      f"filled @ {f_pen.filled_price:.0f}  slippage={f_pen.slippage_cost:.0f}  "
      f"cost_pct={f_pen.cost_pct*100:.2f}%")
assert f_pen.cost_pct > 0.05, f"50% participation should cost > 5% of notional, got {f_pen.cost_pct}"

# Policy CAP — qty hard-capped to max·ADV = 10,000
model_cap = VNCostModel(ExecutionConfig(
    slippage=SlippageModel(alpha=1.0, max_participation=0.10,
                           policy=ParticipationPolicy.CAP)))
f_cap = model_cap.simulate(base_order)
assert f_cap.is_filled
assert f_cap.filled_quantity == 10_000, f"expected cap at 10k, got {f_cap.filled_quantity}"
assert abs(f_cap.participation_pct - 0.10) < 1e-9
print(f"         CAP policy      filled_qty={f_cap.filled_quantity} (intended {base_order.quantity})  ok")


# ============================================================================
# TEST 11: Zero-volume → rejected (illiquid day)
# ============================================================================
order_dry = Order(
    ticker="ABC", side=OrderSide.BUY, quantity=1000,
    target_price=10_000.0, reference_price=10_000.0,
    daily_volume=0, daily_volatility=0.025, exchange=Exchange.HOSE,
)
fill = model.simulate(order_dry)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.ZERO_VOLUME
print(f"TEST 11  Zero-volume rejected  reason={fill.rejection_reason.value}  ok")


# ============================================================================
# TEST 12: NaN / non-finite inputs → INVALID_INPUT
# ============================================================================
nan_order = Order(
    ticker="X", side=OrderSide.BUY, quantity=1000,
    target_price=float("nan"), reference_price=10_000.0,
    daily_volume=100_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
fill = model.simulate(nan_order)
assert not fill.is_filled
assert fill.rejection_reason == RejectionReason.INVALID_INPUT
print(f"TEST 12  NaN price → INVALID_INPUT  ok")


# ============================================================================
# TEST 13: End-to-end fill — verify cash flow math
# ============================================================================
# Tight order: BUY 1,000 shares at 30,000 VND on HOSE.
# ADV 500,000 (0.2% participation — well within band).
# σ = 2% → impact = 1.0 * 0.02 * sqrt(0.002) = 0.000894 = 8.94bps
order = Order(
    ticker="MWG", side=OrderSide.BUY, quantity=1000,
    target_price=30_000.0, reference_price=30_000.0,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
f = model.simulate(order)
assert f.is_filled
expected_impact_pct = 1.0 * 0.02 * math.sqrt(1000/500_000)
expected_raw_fill = 30_000.0 * (1 + expected_impact_pct)  # buy → impact above target
# Tick-rounded UP to nearest 50 (price band 10k–50k)
expected_tick_fill = math.ceil(expected_raw_fill / 50) * 50

print(f"TEST 13  end-to-end BUY fill")
print(f"         target={order.target_price:.0f}  raw_fill={expected_raw_fill:.2f}  "
      f"tick_fill={expected_tick_fill}  actual_fill={f.filled_price:.0f}")
assert f.filled_price == expected_tick_fill, (f.filled_price, expected_tick_fill)

expected_brokerage = f.gross_notional * 0.0015
expected_vat = expected_brokerage * 0.10
expected_tax = 0.0  # buy leg, no transfer tax
assert abs(f.brokerage_paid - expected_brokerage) < 1e-6
assert abs(f.vat_paid - expected_vat) < 1e-6
assert f.tax_paid == 0.0  # buy, no transfer tax
print(f"         gross_notional={f.gross_notional:,.0f}  brokerage={f.brokerage_paid:,.2f}  "
      f"vat={f.vat_paid:,.2f}  tax={f.tax_paid:,.2f}  slippage={f.slippage_cost:,.2f}")
print(f"         signed_cash_flow={f.signed_cash_flow:,.2f}  total_cost_pct={f.cost_pct*100:.3f}%  ok")


# ============================================================================
# TEST 14: SELL leg → tax IS charged
# ============================================================================
sell_order = Order(
    ticker="MWG", side=OrderSide.SELL, quantity=1000,
    target_price=30_000.0, reference_price=30_000.0,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
)
f = model.simulate(sell_order)
assert f.is_filled
# tax = gross * 0.001
expected_tax = f.gross_notional * 0.001
assert abs(f.tax_paid - expected_tax) < 1e-6
assert f.tax_paid > 0
print(f"TEST 14  SELL leg: tax={f.tax_paid:,.2f}  brokerage={f.brokerage_paid:,.2f}  "
      f"vat={f.vat_paid:,.2f}  ok")


# ============================================================================
# TEST 15: Batch + Polars apply_to_signals + rejection_breakdown
# ============================================================================
# Build a small panel of mixed orders: some fill, some get rejected.
records = [
    {"ticker": "VCB", "side": "buy",  "quantity": 1_000,
     "target_price": 80_000, "reference_price": 80_000, "daily_volume": 500_000,
     "daily_volatility": 0.018, "exchange": "HOSE"},
    {"ticker": "FLC", "side": "buy",  "quantity": 5_000,
     "target_price": 10_700, "reference_price": 10_000, "daily_volume": 800_000,
     "daily_volatility": 0.045, "exchange": "HOSE"},   # at HOSE ceiling — REJECT
    {"ticker": "SHB", "side": "sell", "quantity": 30_000,
     "target_price": 14_500, "reference_price": 14_500, "daily_volume": 100_000,
     "daily_volatility": 0.025, "exchange": "HNX"},     # 30% participation — penalised
    {"ticker": "XYZ", "side": "buy",  "quantity": 50,
     "target_price": 12_000, "reference_price": 12_000, "daily_volume": 50_000,
     "daily_volatility": 0.025, "exchange": "HOSE"},    # < 100 lot — REJECT
    {"ticker": "ABC", "side": "buy",  "quantity": 1_000,
     "target_price": 8_500, "reference_price": 10_000, "daily_volume": 100_000,
     "daily_volatility": 0.02, "exchange": "HOSE"},     # -15% — outside HOSE ±7 band → REJECT
]
df = pl.DataFrame(records)
out = model.apply_to_signals(df)
print(f"TEST 15  apply_to_signals  cols added: {[c for c in out.columns if c not in df.columns]}")
filled = out["is_filled"].to_list()
reasons = out["rejection_reason"].to_list()
print(f"         filled mask:  {filled}")
print(f"         reasons:      {reasons}")
assert filled == [True, False, True, False, False]
assert reasons[1] == "price_at_ceiling_buy"
assert reasons[3] == "below_lot_size"
assert reasons[4] == "price_outside_band"

# Re-derive Fill objects to feed into rejection_breakdown
orders = [
    Order(
        ticker=r["ticker"], side=OrderSide(r["side"]),
        quantity=int(r["quantity"]),
        target_price=float(r["target_price"]),
        reference_price=float(r["reference_price"]),
        daily_volume=float(r["daily_volume"]),
        daily_volatility=float(r["daily_volatility"]),
        exchange=Exchange(r["exchange"]),
    ) for r in records
]
fills = model.simulate_batch(orders)
breakdown = rejection_breakdown(fills)
print(f"         breakdown:    {breakdown}")
assert breakdown["filled"] == 2
assert breakdown["price_at_ceiling_buy"] == 1
assert breakdown["below_lot_size"] == 1
assert breakdown["price_outside_band"] == 1


# ============================================================================
# TEST 16: round_trip_cost_pct — Phase 4 drop-in
# ============================================================================
m_default = VNCostModel(ExecutionConfig())
# Default: alpha=1.0, sigma=0.02, participation=1.0 (1.0 share / 1.0 ADV)
cost = m_default.round_trip_cost_pct(notional_share=0.01, daily_vol=0.025)
# Asymmetric fees: 0.0043 + 2 * 0.025 * sqrt(0.01) = 0.0043 + 0.005 = 0.0093 = 0.93%
expected = 0.0043 + 2 * 0.025 * math.sqrt(0.01)
assert abs(cost - expected) < 1e-9, (cost, expected)
print(f"TEST 16  round_trip_cost_pct (1% participation, σ=2.5%)  cost={cost*100:.3f}%  ok")


print()
print("ALL TESTS PASSED  —  the VN cost model refuses every impossible fill.")
