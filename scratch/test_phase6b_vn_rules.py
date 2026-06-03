"""
Targeted tests for the three VN-specific patches:
  1. T+2.5 settlement queue (inventory tracker)
  2. Tick tiers (HOSE 3-tier, HNX/UPCOM flat 100)
  3. ATC (At-The-Close) auction semantics

PLUS: the prior 16-test suite must still pass (regression).
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ast
with open("src/execution/vn_cost_model.py", encoding="utf-8") as f:
    ast.parse(f.read())
print("AST parse OK")

import math
from datetime import date, datetime, time as dtime, timedelta

from src.execution.vn_cost_model import (
    Exchange, OrderSide, RejectionReason,
    FeeSchedule, SlippageModel, ExecutionConfig,
    Order, Fill, VNCostModel,
    InventoryTracker, PendingLot,
    tick_size_vnd, round_to_tick, settlement_datetime, add_business_days,
)


# ============================================================================
# CORRECTION 1: T+2.5 SETTLEMENT QUEUE
# ============================================================================
print("\n────────────────────────────────────────────────────────────────────")
print(" CORRECTION 1 — T+2.5 settlement queue")
print("────────────────────────────────────────────────────────────────────")

# 1a. add_business_days math (skip Sat/Sun)
assert add_business_days(date(2026, 5, 18), 2) == date(2026, 5, 20)    # Mon → Wed
assert add_business_days(date(2026, 5, 22), 2) == date(2026, 5, 26)    # Fri → Tue (skip Sat/Sun)
assert add_business_days(date(2026, 5, 19), 2) == date(2026, 5, 21)    # Tue → Thu
print("TEST 1a  add_business_days  Mon→Wed=ok  Fri→Tue=ok  Tue→Thu=ok")

# 1b. settlement_datetime always at 13:00 ICT
s = settlement_datetime(date(2026, 5, 18))
assert s == datetime(2026, 5, 20, 13, 0), f"expected Wed 13:00, got {s}"
print(f"TEST 1b  settlement_datetime  buy on Mon 2026-05-18 → settles {s.isoformat()}  ok")


# 1c. InventoryTracker — the headline T+1 sell rejection
inv = InventoryTracker()
# Monday 09:30 ICT — buy 1,000 MWG @ 30,050
inv.record_buy(ticker="MWG", trade_date=date(2026, 5, 18), quantity=1000, fill_price=30_050.0)

mon_afternoon = datetime(2026, 5, 18, 14, 30)
tue_morning   = datetime(2026, 5, 19, 9, 30)
wed_morning   = datetime(2026, 5, 20, 9, 30)
wed_1259      = datetime(2026, 5, 20, 12, 59)
wed_1300      = datetime(2026, 5, 20, 13, 0)
wed_1330      = datetime(2026, 5, 20, 13, 30)
thu_morning   = datetime(2026, 5, 21, 9, 30)

assert inv.available_at("MWG", mon_afternoon) == 0,  "same-day sell forbidden"
assert inv.available_at("MWG", tue_morning)   == 0,  "T+1 sell forbidden"
assert inv.available_at("MWG", wed_morning)   == 0,  "T+2 09:30 still pre-13:00 — forbidden"
assert inv.available_at("MWG", wed_1259)      == 0,  "T+2 12:59 still pre-13:00 — forbidden"
assert inv.available_at("MWG", wed_1300)      == 1000, "T+2 13:00 sharp — settled"
assert inv.available_at("MWG", wed_1330)      == 1000, "T+2 13:30 — settled"
assert inv.available_at("MWG", thu_morning)   == 1000, "T+3 — settled"
assert inv.pending_at("MWG", mon_afternoon)   == 1000
assert inv.pending_at("MWG", wed_1330)        == 0
print("TEST 1c  Inventory at each timestamp:")
for label, t in (("Mon 14:30", mon_afternoon), ("Tue 09:30", tue_morning),
                 ("Wed 09:30", wed_morning), ("Wed 12:59", wed_1259),
                 ("Wed 13:00", wed_1300),   ("Wed 13:30", wed_1330),
                 ("Thu 09:30", thu_morning)):
    print(f"          {label}:  settled={inv.available_at('MWG', t):>4}  pending={inv.pending_at('MWG', t):>4}")
print("          ok")


# 1d. THE HEADLINE TEST — T+1 sell gets REJECTED by the engine
print("\nTEST 1d  T+1 SELL REJECTION (the headline)")
model = VNCostModel(ExecutionConfig())
inv2 = InventoryTracker()

# Monday morning: BUY 1,000 MWG
buy_mon = Order(
    ticker="MWG", side=OrderSide.BUY, quantity=1000,
    target_price=30_000, reference_price=30_000,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
)
fill_buy = model.simulate(buy_mon, inventory=inv2)
assert fill_buy.is_filled
print(f"          T+0 Mon 09:30  BUY 1000 MWG  →  filled @ {fill_buy.filled_price:.0f}")
print(f"          Inventory:  settled={inv2.available_at('MWG', buy_mon.timestamp):>4}  "
      f"pending={inv2.pending_at('MWG', buy_mon.timestamp):>4}")

# Tuesday morning: try to SELL → MUST REJECT
sell_tue = Order(
    ticker="MWG", side=OrderSide.SELL, quantity=1000,
    target_price=30_500, reference_price=30_050,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 19, 9, 30),
)
fill_t1 = model.simulate(sell_tue, inventory=inv2)
assert not fill_t1.is_filled, "T+1 sell MUST be rejected"
assert fill_t1.rejection_reason == RejectionReason.INVENTORY_NOT_SETTLED, \
    f"expected INVENTORY_NOT_SETTLED, got {fill_t1.rejection_reason}"
print(f"          T+1 Tue 09:30  SELL 1000 MWG → REJECTED  reason={fill_t1.rejection_reason.value}  ✓")

# Wednesday morning: still pre-13:00 → REJECT
sell_wed_am = Order(
    ticker="MWG", side=OrderSide.SELL, quantity=1000,
    target_price=30_500, reference_price=30_050,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 9, 30),
)
fill_wed_am = model.simulate(sell_wed_am, inventory=inv2)
assert not fill_wed_am.is_filled
assert fill_wed_am.rejection_reason == RejectionReason.INVENTORY_NOT_SETTLED
print(f"          T+2 Wed 09:30  SELL 1000 MWG → REJECTED  reason={fill_wed_am.rejection_reason.value}  ✓")

# Wednesday afternoon (13:30): NOW it fills
sell_wed_pm = Order(
    ticker="MWG", side=OrderSide.SELL, quantity=1000,
    target_price=30_500, reference_price=30_050,
    daily_volume=500_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 13, 30),
)
fill_wed_pm = model.simulate(sell_wed_pm, inventory=inv2)
assert fill_wed_pm.is_filled, f"T+2 13:30 sell should fill, got {fill_wed_pm.rejection_reason}"
print(f"          T+2 Wed 13:30  SELL 1000 MWG → FILLED @ {fill_wed_pm.filled_price:.0f}  ✓")

# After the sell, inventory at the same instant should be 0
assert inv2.available_at("MWG", sell_wed_pm.timestamp) == 0
print(f"          Post-sell inventory at Wed 13:30: settled={inv2.available_at('MWG', sell_wed_pm.timestamp)}  ok")


# 1e. Partial settlement — buy 500 Monday, buy 500 Tuesday, sell 1000 Friday
inv3 = InventoryTracker()
b1 = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=500,
    target_price=90_000, reference_price=90_000,
    daily_volume=300_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 10, 0),    # Mon
)
b2 = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=500,
    target_price=90_000, reference_price=90_000,
    daily_volume=300_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 19, 10, 0),    # Tue
)
sell_fri_target = Order(
    ticker="VCB", side=OrderSide.SELL, quantity=1000,
    target_price=90_500, reference_price=90_000,
    daily_volume=300_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 22, 14, 0),    # Fri afternoon — both lots settled
)
assert model.simulate(b1, inventory=inv3).is_filled
assert model.simulate(b2, inventory=inv3).is_filled
f = model.simulate(sell_fri_target, inventory=inv3)
assert f.is_filled and f.filled_quantity == 1000
print(f"TEST 1e  Mon+Tue buys (500 each), Fri 14:00 sell 1000 → filled (both settled by Fri)  ok")

# Same logic but sell on THURSDAY 13:30: only Mon's lot has settled (Wed 13:00),
# Tue's lot settles Thu 13:00.  Available = 1000 → sell of 1000 should fill.
inv4 = InventoryTracker()
model.simulate(b1, inventory=inv4)
model.simulate(b2, inventory=inv4)
# Try to sell 1000 on Thu 09:30 — only Mon's 500 has settled (Wed 13:00)
sell_thu_am = Order(
    ticker="VCB", side=OrderSide.SELL, quantity=1000,
    target_price=90_500, reference_price=90_000,
    daily_volume=300_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 21, 9, 30),    # Thu morning — only 500 settled
)
f = model.simulate(sell_thu_am, inventory=inv4)
assert not f.is_filled and f.rejection_reason == RejectionReason.INVENTORY_NOT_SETTLED
print(f"TEST 1f  Thu 09:30 sell 1000 (only 500 settled) → REJECTED  ok")

# But selling 500 on Thu 09:30 should succeed
sell_thu_am_500 = Order(
    ticker="VCB", side=OrderSide.SELL, quantity=500,
    target_price=90_500, reference_price=90_000,
    daily_volume=300_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 21, 9, 30),
)
f = model.simulate(sell_thu_am_500, inventory=inv4)
assert f.is_filled and f.filled_quantity == 500
print(f"TEST 1g  Thu 09:30 sell 500 (matches settled half) → FILLED  ok")


# ============================================================================
# CORRECTION 2: TICK TIERS
# ============================================================================
print("\n────────────────────────────────────────────────────────────────────")
print(" CORRECTION 2 — Tick tiers (Bước giá)")
print("────────────────────────────────────────────────────────────────────")

# HOSE 3-tier: <10k=10, <50k=50, ≥50k=100
hose_cases = [
    (5_000,   10),
    (9_999,   10),
    (10_000,  50),
    (25_000,  50),
    (49_950,  50),
    (50_000, 100),
    (100_000, 100),
    (500_000, 100),
]
for price, expected in hose_cases:
    got = tick_size_vnd(price, Exchange.HOSE)
    assert got == expected, f"HOSE@{price}: expected tick={expected}, got {got}"
print(f"TEST 2a  HOSE tick tiers: {[(p, tick_size_vnd(p, Exchange.HOSE)) for p, _ in hose_cases]}")

# HNX & UPCOM flat 100
for px in (1_000, 5_000, 10_000, 50_000, 100_000, 500_000):
    assert tick_size_vnd(px, Exchange.HNX) == 100, f"HNX@{px}"
    assert tick_size_vnd(px, Exchange.UPCOM) == 100, f"UPCOM@{px}"
print("TEST 2b  HNX & UPCOM all flat 100 VND across the price range  ok")

# Rounding direction at each tier — BUY rounds UP, SELL rounds DOWN
# Tier 1 (<10k): tick=10
assert round_to_tick(5_037.0, Exchange.HOSE, side=OrderSide.BUY) == 5_040
assert round_to_tick(5_037.0, Exchange.HOSE, side=OrderSide.SELL) == 5_030
# Tier 2 (10-50k): tick=50
assert round_to_tick(25_037.0, Exchange.HOSE, side=OrderSide.BUY) == 25_050
assert round_to_tick(25_037.0, Exchange.HOSE, side=OrderSide.SELL) == 25_000
# Tier 3 (≥50k): tick=100
assert round_to_tick(100_037.0, Exchange.HOSE, side=OrderSide.BUY) == 100_100
assert round_to_tick(100_037.0, Exchange.HOSE, side=OrderSide.SELL) == 100_000
# Exact boundaries
assert round_to_tick(10_000.0, Exchange.HOSE, side=OrderSide.BUY) == 10_000   # already on tier-2 grid (10,000 % 50 == 0)
assert round_to_tick(50_000.0, Exchange.HOSE, side=OrderSide.BUY) == 50_000   # already on tier-3 grid (50,000 % 100 == 0)
print("TEST 2c  Round_to_tick across all three tiers (BUY=up, SELL=down)  ok")


# ============================================================================
# CORRECTION 3: ATC (At-The-Close) AUCTION
# ============================================================================
print("\n────────────────────────────────────────────────────────────────────")
print(" CORRECTION 3 — ATC (At-The-Close)")
print("────────────────────────────────────────────────────────────────────")

model = VNCostModel(ExecutionConfig())

# 3a. ATC fills at exact target_price (no impact slippage)
atc_order = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=10_000,
    target_price=80_000, reference_price=80_000,
    daily_volume=500_000, daily_volatility=0.025, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 14, 35),    # mid-ATC session
    is_atc=True,
    atc_volume=50_000,    # plenty of matched volume
)
f = model.simulate(atc_order)
assert f.is_filled
assert f.filled_quantity == 10_000
# Target=80,000 is already on tier-3 (100 VND) grid; impact==0 → exact fill
assert f.filled_price == 80_000, f"ATC fill should be at clearing price, got {f.filled_price}"
assert f.slippage_cost == 0.0, f"ATC must NOT carry slippage, got {f.slippage_cost}"
print(f"TEST 3a  ATC BUY 10k @ 80,000  →  filled @ {f.filled_price:.0f}  slippage={f.slippage_cost}  ok")

# 3b. ATC partial fill when intended_qty > atc_volume
atc_big = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=80_000,
    target_price=80_000, reference_price=80_000,
    daily_volume=500_000, daily_volatility=0.025, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 14, 35),
    is_atc=True,
    atc_volume=25_000,    # only 25k matched in ATC — partial fill
)
f = model.simulate(atc_big)
assert f.is_filled, f"should partial-fill, got {f.rejection_reason}"
assert f.filled_quantity == 25_000, f"expected 25k (atc cap), got {f.filled_quantity}"
assert f.filled_price == 80_000
assert f.slippage_cost == 0.0
print(f"TEST 3b  ATC BUY 80k requested, ATC matched only 25k  →  partial fill {f.filled_quantity}  ok")

# 3c. ATC rejected when atc_volume < lot
atc_tiny = Order(
    ticker="VCB", side=OrderSide.BUY, quantity=10_000,
    target_price=80_000, reference_price=80_000,
    daily_volume=500_000, daily_volatility=0.025, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 14, 35),
    is_atc=True,
    atc_volume=50,       # less than one lot — uncrossable
)
f = model.simulate(atc_tiny)
assert not f.is_filled
assert f.rejection_reason == RejectionReason.ATC_VOLUME_EXCEEDED
print(f"TEST 3c  ATC BUY 10k requested, ATC matched only 50 (<lot)  →  REJECTED  reason={f.rejection_reason.value}  ok")

# 3d. ATC honours T+2.5 — selling at ATC same day you bought is still rejected
inv_atc = InventoryTracker()
buy_mon_morning = Order(
    ticker="HPG", side=OrderSide.BUY, quantity=1_000,
    target_price=25_000, reference_price=25_000,
    daily_volume=2_000_000, daily_volatility=0.022, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
)
model.simulate(buy_mon_morning, inventory=inv_atc)

sell_atc_same_day = Order(
    ticker="HPG", side=OrderSide.SELL, quantity=1_000,
    target_price=25_300, reference_price=25_000,
    daily_volume=2_000_000, daily_volatility=0.022, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 14, 35),    # SAME DAY ATC
    is_atc=True,
    atc_volume=100_000,
)
f = model.simulate(sell_atc_same_day, inventory=inv_atc)
assert not f.is_filled
assert f.rejection_reason == RejectionReason.INVENTORY_NOT_SETTLED, \
    f"ATC sell same-day must hit T+2.5 wall, got {f.rejection_reason}"
print(f"TEST 3d  ATC SELL on SAME DAY as buy → REJECTED by T+2.5  ok")


# ============================================================================
# REGRESSION: prior 16 tests still pass
# ============================================================================
print("\n────────────────────────────────────────────────────────────────────")
print(" REGRESSION — prior 16-test suite")
print("────────────────────────────────────────────────────────────────────")

import subprocess
result = subprocess.run(
    [sys.executable, "scratch/test_phase6_costs.py"],
    capture_output=True, text=True, timeout=120,
)
last_line = result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr
print(f"  prior suite final line: {last_line}")
assert result.returncode == 0, f"regression suite failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
assert "ALL TESTS PASSED" in result.stdout
print("  regression PASS")


print()
print("════════════════════════════════════════════════════════════════════")
print(" ALL TESTS PASSED — VN microstructure model is uncompromising.")
print("════════════════════════════════════════════════════════════════════")
