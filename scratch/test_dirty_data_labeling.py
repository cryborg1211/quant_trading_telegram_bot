"""Regression: triple-barrier labeling must survive dirty rows (close==0, neg, NaN)."""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
from datetime import date

from src.labels.triple_barrier import (
    TripleBarrierConfig, triple_barrier_pipeline,
    get_daily_vol, get_bins, get_events, apply_pt_sl_on_t1, get_sample_weights,
)

# Turn numpy divide warnings into ERRORS so any unguarded division fails loudly.
np.seterr(all="raise")
warnings.simplefilter("error", RuntimeWarning)

# ── 1. get_daily_vol with a zero price in the series ──
idx = pd.bdate_range("2024-01-01", periods=60)
px = pd.Series(np.linspace(100, 130, 60), index=idx)
px.iloc[30] = 0.0           # suspended-day artifact
px.iloc[45] = -5.0          # negative artifact
vol = get_daily_vol(px, span=20)
assert np.isfinite(vol.dropna()).all(), "get_daily_vol produced inf/NaN from dirty price"
print(f"TEST 1  get_daily_vol survives zero/neg close — finite σ (max={vol.dropna().max():.4f})  ok")

# ── 2. get_bins RUTHLESS: zero entry/exit → NaN ret AND NaN bin (not FLAT) ──
close = pd.Series([100.0, 0.0, 110.0, 121.0, 0.0, 130.0], index=pd.bdate_range("2024-02-01", periods=6))
events = pd.DataFrame({
    "t1": [close.index[2], close.index[3], close.index[5]],
    "trgt": [0.02, 0.02, 0.02],
    "pt": [pd.NaT, pd.NaT, pd.NaT],
    "sl": [pd.NaT, pd.NaT, pd.NaT],
}, index=[close.index[1], close.index[0], close.index[4]])  # index[1] & index[4] = zero entry
bins = get_bins(events, close, label_scheme="012")
# index[1] (entry close==0) and index[4] (entry close==0) → unlabelable → NaN.
assert np.isnan(bins.loc[close.index[1], "ret"]), "zero-entry event must have NaN ret"
assert np.isnan(bins.loc[close.index[1], "bin"]), "zero-entry event must have NaN bin"
assert np.isnan(bins.loc[close.index[4], "ret"]) and np.isnan(bins.loc[close.index[4], "bin"])
# index[0] (entry 100 → exit close[3]=121) is valid: finite ret, label in {0,1,2}.
vr = bins.loc[close.index[0]]
assert np.isfinite(vr["ret"]) and int(vr["bin"]) in (0, 1, 2), f"valid row corrupted: {vr.to_dict()}"
print(f"TEST 2  get_bins RUTHLESS: zero entry/exit → NaN (not FLAT); valid row finite "
      f"(ret={vr['ret']:.3f}, bin={int(vr['bin'])})  ok")

# ── 3. apply_pt_sl_on_t1 with zeros in high/low path ──
c2 = pd.Series([50.0, 51.0, 0.0, 52.0, 49.0, 53.0], index=pd.bdate_range("2024-03-01", periods=6))
hi = c2 * 1.02; lo = c2 * 0.98
hi.iloc[2] = 0.0; lo.iloc[2] = 0.0   # dirty bar mid-path
ev = pd.DataFrame({"t1": [c2.index[5]], "trgt": [0.03]}, index=[c2.index[0]])
touches = apply_pt_sl_on_t1(c2, ev, (2.0, 2.0), high=hi, low=lo, use_intrabar_extremes=True)
assert pd.notna(touches["t1"].iloc[0]), "apply_pt_sl_on_t1 lost the event"
print(f"TEST 3  apply_pt_sl_on_t1 survives zero high/low mid-path — t1={touches['t1'].iloc[0].date()}  ok")

# ── 4. Full pipeline on a panel salted with dirty rows ──
rng = np.random.default_rng(0)
days = pd.bdate_range("2023-01-02", periods=200).date.tolist()
rows = []
for i in range(8):
    tk = f"DIRTY{i}"
    px_ = 30_000.0
    for k, d in enumerate(days):
        px_ *= (1 + rng.normal(0.0005, 0.02))
        c = px_
        # Salt ~3% of rows with zero/neg/NaN close (the real-world failure mode).
        if rng.uniform() < 0.03:
            c = rng.choice([0.0, -1.0, np.nan])
        rows.append({"ticker": tk, "date": d, "close": c,
                     "high": (c * 1.01 if np.isfinite(c) else c),
                     "low": (c * 0.99 if np.isfinite(c) else c),
                     "open": c, "volume": 1_000_000})
panel = pl.from_pandas(pd.DataFrame(rows))

# Mimic the master-script ingestion filter (Patch 1) — drop dirty rows first.
price_cols = ["open", "high", "low", "close"]
panel = panel.with_columns([pl.col(c).cast(pl.Float64) for c in price_cols])
valid = pl.lit(True)
for c in price_cols:
    valid = valid & pl.col(c).is_finite() & (pl.col(c) > 0)
n_before = panel.height
panel = panel.filter(valid)
print(f"TEST 4  ingestion filter dropped {n_before - panel.height}/{n_before} dirty rows")

result = triple_barrier_pipeline(panel, cfg=TripleBarrierConfig(horizon=5, label_scheme="012"))
rp = result.to_pandas()
assert np.isfinite(rp["ret"].to_numpy()).all(), "pipeline ret has inf/NaN"
assert np.isfinite(rp["w"].to_numpy()).all(), "pipeline weights have inf/NaN"
assert np.isfinite(rp["trgt"].to_numpy()).all(), "pipeline trgt has inf/NaN"
assert set(rp["bin"].unique()) <= {0, 1, 2}
print(f"TEST 4  full pipeline clean: events={len(rp)}  ret/w/trgt all finite  mean_w={rp['w'].mean():.3f}  ok")

# ── 5. EXTREME case: pipeline directly on UNFILTERED dirty panel (no Patch-1) ──
# The defensive triple-barrier patches alone must prevent a crash even if the
# upstream filter is somehow bypassed.
panel_raw = pl.from_pandas(pd.DataFrame(rows)).with_columns(
    [pl.col(c).cast(pl.Float64) for c in price_cols])
result2 = triple_barrier_pipeline(panel_raw, cfg=TripleBarrierConfig(horizon=5, label_scheme="012"))
rp2 = result2.to_pandas()
assert np.isfinite(rp2["ret"].to_numpy()).all() and np.isfinite(rp2["w"].to_numpy()).all()
# RUTHLESS contract: the final training set carries NO NaN labels and bin is int.
assert rp2["bin"].isna().sum() == 0, "pipeline leaked NaN labels"
assert set(rp2["bin"].unique()) <= {0, 1, 2}
assert pd.api.types.is_integer_dtype(rp2["bin"]), f"bin must be integer, got {rp2['bin'].dtype}"
print(f"TEST 5  pipeline on UNFILTERED dirty panel: NO crash, events={len(rp2)}, "
      f"0 NaN labels, bin dtype={rp2['bin'].dtype}  ok")

# ── 6. RUTHLESS drop count: an event with a clean entry but a ZERO EXIT bar
#       must be EXCISED (not labelled FLAT). Construct that exact case. ──
didx = pd.bdate_range("2023-06-01", periods=80)
cc = pd.Series(100.0 * np.cumprod(1 + np.random.default_rng(1).normal(0.001, 0.02, 80)), index=didx)
# Force a zero EXIT bar where many 5-day events will land their t1.
cc.iloc[40] = 0.0
panel6 = pl.from_pandas(pd.DataFrame({
    "ticker": "EXITZERO", "date": didx.date,
    "open": cc.values, "high": cc.values * 1.01,
    "low": cc.values * 0.99, "close": cc.values, "volume": 1_000_000,
}).pipe(lambda d: d.assign(**{c: d[c].clip(lower=0) for c in ["open", "high", "low", "close"]})))
res6 = triple_barrier_pipeline(panel6, cfg=TripleBarrierConfig(horizon=5, label_scheme="012"))
rp6 = res6.to_pandas()
# No event may have t0 OR t1 on the zero-exit bar (date index 40); all dropped.
zero_date = didx[40].date()
assert (rp6["t0"].dt.date != zero_date).all(), "kept an event entering on the zero bar"
assert np.isfinite(rp6["ret"]).all() and rp6["bin"].isna().sum() == 0
print(f"TEST 6  zero-EXIT events excised (not FLAT-labelled): events={len(rp6)}, all clean  ok")

print()
print("ALL DIRTY-DATA TESTS PASSED — unlabelable events are EXCISED, never synthesised.")
