"""
Phase 1.5 (anti-FOMO over-extension features) + Phase 6.5 (corporate-action ledger).

Headline proofs:
  • A 1:1 stock split (factor 2.0) does NOT register as a loss.
  • A cash dividend neutralizes the ex-date price drop in total wealth.
  • Over-extension features flag a +7% ceiling pump as cross-sectionally extreme.
PLUS regression of the prior Phase-6/6b suites.
"""
import sys, io, os, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ast
for f in ("src/data/tensor_builder.py", "src/execution/vn_cost_model.py"):
    with open(f, encoding="utf-8") as fh:
        ast.parse(fh.read())
print("AST parse OK (both files)")

import numpy as np
import polars as pl
from datetime import date, datetime

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1.5 — Over-extension (anti-FOMO) features
# ════════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────────────")
print(" PHASE 1.5 — anti-FOMO over-extension features")
print("──────────────────────────────────────────────────────────")

from src.data.tensor_builder import add_overextension_features

# Build a panel: 30 tickers, 40 days. One ticker (PUMP) ramps to a +7% ceiling
# blow-off at the end; the rest drift normally.
rng = np.random.default_rng(3)
days = pl.date_range(date(2026, 1, 1), date(2026, 2, 9), interval="1d", eager=True).filter(
    pl.date_range(date(2026, 1, 1), date(2026, 2, 9), interval="1d", eager=True).dt.weekday() <= 5
)
days = days[:40]
records = []
for i in range(30):
    tk = f"T{i:02d}" if i > 0 else "PUMP"
    px = 20_000.0
    for d in days:
        px *= (1.0 + rng.normal(0.0, 0.01))
        records.append({"ticker": tk, "date": d, "close": px})
# Force PUMP into a parabolic blow-off on the last 5 days
df = pl.DataFrame(records)
pump_mask = df["ticker"] == "PUMP"
pump_rows = df.filter(pump_mask).sort("date")
boosted = pump_rows["close"].to_list()
for k in range(1, 6):
    boosted[-k] = boosted[-6] * (1.07 ** (6 - k))   # ramp to +7%/day blow-off
df = df.with_columns(
    pl.when(pump_mask)
    .then(pl.Series(
        # rebuild close for PUMP rows in date order; others unchanged
        "close",
        # map back: easier to just recompute via join
        df.filter(pump_mask).sort("date")["close"]
    ))
    .otherwise(pl.col("close"))
    .alias("close")
) if False else df  # (skip the convoluted in-place edit; rebuild cleanly below)

# Cleaner rebuild: bump PUMP's last 5 closes directly.
pdf = df.to_pandas()
pump_idx = pdf.index[pdf["ticker"] == "PUMP"].tolist()
pump_idx_sorted = pdf.loc[pump_idx].sort_values("date").index.tolist()
base = pdf.loc[pump_idx_sorted[-6], "close"]
for k, idx in enumerate(pump_idx_sorted[-5:], start=1):
    pdf.loc[idx, "close"] = base * (1.07 ** k)
df = pl.from_pandas(pdf)

out = add_overextension_features(df, ma_windows=(5, 20), cross_sectional=True)

# Columns present
assert "overext_5" in out.columns and "overext_20" in out.columns
assert "overext_5_xsz" in out.columns and "overext_20_xsz" in out.columns
print(f"TEST 1.5a  columns added: overext_5, overext_20, overext_5_xsz, overext_20_xsz  ok")

# On the LAST date, PUMP's overext_5 should be the highest raw value AND its
# cross-sectional z-score should be near the top of the universe.
last_day = out["date"].max()
last = out.filter(pl.col("date") == last_day).sort("overext_5", descending=True)
top_ticker = last["ticker"][0]
pump_row = last.filter(pl.col("ticker") == "PUMP")
pump_raw = pump_row["overext_5"][0]
pump_xsz = pump_row["overext_5_xsz"][0]
print(f"TEST 1.5b  on {last_day}:  most over-extended = {top_ticker}")
print(f"           PUMP overext_5={pump_raw:.4f}  overext_5_xsz={pump_xsz:.3f}")
assert top_ticker == "PUMP", f"PUMP should be most over-extended, got {top_ticker}"
assert pump_raw > 0.05, f"PUMP raw over-extension should be large, got {pump_raw}"
assert pump_xsz > 1.5, f"PUMP cross-sectional z should be extreme (>1.5σ), got {pump_xsz}"
print("           → the attention head can now SEE the ceiling pump as exhausted  ok")

# Leak-safety: over-extension at t uses close[t] only (no future). Verify the
# raw value matches a hand recomputation for one row.
chk = out.filter((pl.col("ticker") == "T05")).sort("date")
closes = chk["close"].to_numpy()
ma5_manual = np.mean(closes[-5:])
oe5_manual = closes[-1] / ma5_manual - 1.0
oe5_feature = chk["overext_5"][-1]
assert abs(oe5_manual - oe5_feature) < 1e-9, f"{oe5_manual} vs {oe5_feature}"
print(f"TEST 1.5c  leak-safe MA math verified (manual {oe5_manual:.5f} == feature {oe5_feature:.5f})  ok")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 6.5 — Corporate-action ledger
# ════════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────────────")
print(" PHASE 6.5 — corporate-action ledger (split + dividend)")
print("──────────────────────────────────────────────────────────")

from src.execution.vn_cost_model import (
    Exchange, OrderSide, RejectionReason,
    ExecutionConfig, Order, VNCostModel,
    InventoryTracker, CorporateActionEvent, CorporateActionType,
)

model = VNCostModel(ExecutionConfig())

# ── THE HEADLINE: 1:1 stock split must NOT register as a loss ────────────────
print("\nTEST 6.5a  1:1 STOCK SPLIT (factor 2.0) — the headline")
inv = InventoryTracker()
# Buy 1,000 shares @ 100,000 VND on Monday (cost basis 100,000,000 VND)
buy = Order(
    ticker="VND", side=OrderSide.BUY, quantity=1000,
    target_price=100_000, reference_price=100_000,
    daily_volume=2_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
)
fb = model.simulate(buy, inventory=inv)
assert fb.is_filled
# Mark BEFORE the split, at the cum price (use settled instant Wed 13:30)
t_pre = datetime(2026, 5, 20, 13, 30)
pv_pre = inv.position_value("VND", mark_price=100_000, t=t_pre)
print(f"           PRE-split:  shares={pv_pre['net_shares']}  price=100,000  "
      f"wealth={pv_pre['total_wealth']:,.0f}")

# 1:1 bonus issue → factor 2.0. Ex-date reference price halves to ~50,000.
inv.apply_corporate_action(CorporateActionEvent(
    ticker="VND", ex_date=date(2026, 5, 21),
    action_type=CorporateActionType.SPLIT, split_factor=2.0,
))
# Post-split mark at the ADJUSTED price 50,000
t_post = datetime(2026, 5, 21, 13, 30)
pv_post = inv.position_value("VND", mark_price=50_000, t=t_post)
print(f"           POST-split: shares={pv_post['net_shares']}  price=50,000   "
      f"wealth={pv_post['total_wealth']:,.0f}")

# THE PROOF: wealth is preserved — the 50% "price crash" is purely mechanical.
assert pv_post["net_shares"] == 2000, f"shares should double, got {pv_post['net_shares']}"
assert abs(pv_post["total_wealth"] - pv_pre["total_wealth"]) < 1e-6, \
    f"split changed wealth! pre={pv_pre['total_wealth']} post={pv_post['total_wealth']}"
print(f"           ✓ wealth unchanged across a 50% price drop — system DID NOT die")

# A naive engine WITHOUT the patch would see 100,000 → 50,000 and book −50%.
naive_pnl_pct = (50_000 - 100_000) / 100_000
print(f"           (naive engine would have booked {naive_pnl_pct:+.0%} — catastrophic false loss)")


# ── Cash-dividend trap fix ───────────────────────────────────────────────────
print("\nTEST 6.5b  CASH DIVIDEND — neutralize the ex-date drop")
inv2 = InventoryTracker()
buy2 = Order(
    ticker="VNM", side=OrderSide.BUY, quantity=1000,
    target_price=80_000, reference_price=80_000,
    daily_volume=1_000_000, daily_volatility=0.018, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
)
assert model.simulate(buy2, inventory=inv2).is_filled
t_pre2 = datetime(2026, 5, 20, 13, 30)
wealth_pre = inv2.position_value("VNM", 80_000, t_pre2)["total_wealth"]
# Big special dividend of 8,000 VND/share (10% of price) on ex-date.
# Ex-price drops to 72,000.
inv2.apply_corporate_action(CorporateActionEvent(
    ticker="VNM", ex_date=date(2026, 5, 21),
    action_type=CorporateActionType.CASH_DIVIDEND, cash_per_share=8_000,
))
t_post2 = datetime(2026, 5, 21, 13, 30)
pv2 = inv2.position_value("VNM", mark_price=72_000, t=t_post2)
print(f"           PRE:  shares=1000 price=80,000  wealth={wealth_pre:,.0f}")
print(f"           POST: shares={pv2['net_shares']} price=72,000  "
      f"cash={pv2['cash_balance']:,.0f}  wealth={pv2['total_wealth']:,.0f}")
assert abs(pv2["cash_balance"] - 8_000_000) < 1e-6, "dividend cash should be 1000×8000"
assert abs(pv2["total_wealth"] - wealth_pre) < 1e-6, \
    f"dividend changed wealth! pre={wealth_pre} post={pv2['total_wealth']}"
print(f"           ✓ 10% ex-date price drop fully offset by dividend cash — wealth flat")


# ── Split preserves cost basis & T+2.5 chronology ───────────────────────────
print("\nTEST 6.5c  split rescales shares AND keeps T+2.5 consistent")
inv3 = InventoryTracker()
# Buy 200 Monday, sell 100 Wednesday afternoon (settled), then 2:1 split Thursday.
# (Quantities are lot-valid multiples of 100 — the cost model rejects sub-lots.)
model.simulate(Order(
    ticker="HPG", side=OrderSide.BUY, quantity=200,
    target_price=25_000, reference_price=25_000,
    daily_volume=5_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
), inventory=inv3)
# Wed 13:30: 200 settled → sell 100
sell = model.simulate(Order(
    ticker="HPG", side=OrderSide.SELL, quantity=100,
    target_price=25_500, reference_price=25_000,
    daily_volume=5_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 20, 13, 30),
), inventory=inv3)
assert sell.is_filled
t_chk = datetime(2026, 5, 20, 14, 0)
net_before = inv3.net_shares_at("HPG", t_chk)
assert net_before == 100, f"net should be 100 (200−100), got {net_before}"
# 2:1 split on Thursday
inv3.apply_corporate_action(CorporateActionEvent(
    ticker="HPG", ex_date=date(2026, 5, 21),
    action_type=CorporateActionType.SPLIT, split_factor=2.0,
))
t_after = datetime(2026, 5, 21, 14, 0)
net_after = inv3.net_shares_at("HPG", t_after)
print(f"           net before split={net_before}  after 2:1 split={net_after}")
assert net_after == 200, f"100 shares should become 200 post-split, got {net_after}"
print(f"           ✓ (held 100) × 2 = 200, sold-record correctly rescaled too")


# ── parse_corporate_actions adapter (Phase-5 store format) ──────────────────
print("\nTEST 6.5d  parse_corporate_actions from a Phase-5-style DataFrame")
ca_df = pl.DataFrame({
    "ticker": ["VND", "VNM", "HPG"],
    "event_date": [date(2026, 5, 21), date(2026, 6, 1), date(2026, 6, 15)],
    "action_type": ["split", "dividend", "stock_dividend"],
    "factor": [2.0, 1.0, 1.5],
    "cash_amount": [0.0, 8_000.0, 0.0],
})
events = InventoryTracker.parse_corporate_actions(ca_df)
assert len(events) == 3
by_ticker = {e.ticker: e for e in events}
assert by_ticker["VND"].action_type == CorporateActionType.SPLIT
assert by_ticker["VND"].split_factor == 2.0
assert by_ticker["VNM"].action_type == CorporateActionType.CASH_DIVIDEND
assert by_ticker["VNM"].cash_per_share == 8_000.0
assert by_ticker["HPG"].action_type == CorporateActionType.STOCK_DIVIDEND
assert by_ticker["HPG"].split_factor == 1.5
print(f"           parsed {len(events)} events: "
      f"VND split×2.0, VNM div 8000/sh, HPG bonus×1.5  ok")

# ingest_corporate_actions (batch, ex-date order)
inv4 = InventoryTracker()
model.simulate(Order(
    ticker="VND", side=OrderSide.BUY, quantity=1000,
    target_price=100_000, reference_price=100_000,
    daily_volume=2_000_000, daily_volatility=0.02, exchange=Exchange.HOSE,
    timestamp=datetime(2026, 5, 18, 9, 30),
), inventory=inv4)
log = inv4.ingest_corporate_actions([e for e in events if e.ticker == "VND"])
assert len(log) == 1 and log[0]["shares_after"] == 2000
print(f"           ingest_corporate_actions audit log: {log[0]}")
print("           ok")


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION — prior Phase 6 / 6b suites still pass
# ════════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────────────")
print(" REGRESSION — Phase 6 + Phase 6b")
print("──────────────────────────────────────────────────────────")
_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
for suite in ("scratch/test_phase6_costs.py", "scratch/test_phase6b_vn_rules.py"):
    r = subprocess.run(
        [sys.executable, suite], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=_env, timeout=180,
    )
    out = r.stdout or ""
    ok = r.returncode == 0 and "ALL TESTS PASSED" in out
    print(f"  {suite}: {'PASS' if ok else 'FAIL'}")
    assert ok, f"{suite} regression failed:\n{out[-800:]}\n{(r.stderr or '')[-400:]}"

print()
print("════════════════════════════════════════════════════════════")
print(" ALL TESTS PASSED — split/dividend cannot kill the book.")
print("════════════════════════════════════════════════════════════")
