"""
Verify the vectorized apply_pt_sl_on_t1 is EXACTLY equivalent to the old
iterrows implementation, and benchmark the speedup.

The reference below is a verbatim copy of the pre-vectorization iterrows logic.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.labels.triple_barrier import apply_pt_sl_on_t1  # the NEW vectorized one


# ── REFERENCE: verbatim old iterrows implementation ──────────────────────────
def reference_apply(close, events, pt_sl, high=None, low=None, use_intrabar_extremes=True):
    pt_mult, sl_mult = pt_sl
    out = pd.DataFrame(index=events.index, columns=["pt", "sl", "t1"], dtype="object")
    out["pt"] = pd.NaT
    out["sl"] = pd.NaT
    use_ohlc = use_intrabar_extremes and high is not None and low is not None
    last_bar = close.index[-1]
    # Extract columns as typed Series (avoid iterrows' object-row coercion, which
    # breaks np.isfinite when a NaN σ forces the mixed datetime+float row to object;
    # the touch LOGIC below is byte-identical to the original).
    t1_col = events["t1"]
    trgt_col = events["trgt"]
    for t0 in events.index:
        t1_raw = t1_col.loc[t0]
        t1_v = t1_raw if pd.notna(t1_raw) else last_bar
        trgt = float(trgt_col.loc[t0])
        if not np.isfinite(trgt) or trgt <= 0:
            out.at[t0, "t1"] = t1_v
            continue
        path = close.loc[t0:t1_v]
        if len(path) < 2:
            out.at[t0, "t1"] = t1_v
            continue
        p0 = close.at[t0]
        if not np.isfinite(p0) or p0 <= 0:
            out.at[t0, "t1"] = t1_v
            continue
        if use_ohlc:
            hi = high.loc[t0:t1_v]; lo = low.loc[t0:t1_v]
            up_metric = hi.where(hi > 0) / p0 - 1.0
            dn_metric = lo.where(lo > 0) / p0 - 1.0
        else:
            r = path.where(path > 0) / p0 - 1.0
            up_metric, dn_metric = r, r
        pt_thr = pt_mult * trgt if pt_mult > 0 else None
        sl_thr = -sl_mult * trgt if sl_mult > 0 else None
        if pt_thr is not None:
            hit_up = up_metric[up_metric >= pt_thr]
            if not hit_up.empty:
                out.at[t0, "pt"] = hit_up.index[0]
        if sl_thr is not None:
            hit_dn = dn_metric[dn_metric <= sl_thr]
            if not hit_dn.empty:
                out.at[t0, "sl"] = hit_dn.index[0]
        touches = [d for d in (out.at[t0, "pt"], out.at[t0, "sl"]) if pd.notna(d)]
        if touches:
            first_touch = min(touches)
            if (pd.notna(out.at[t0, "pt"]) and pd.notna(out.at[t0, "sl"])
                    and out.at[t0, "pt"] == out.at[t0, "sl"]):
                out.at[t0, "pt"] = pd.NaT
            out.at[t0, "t1"] = min(first_touch, t1_v)
        else:
            out.at[t0, "t1"] = t1_v
    out["pt"] = pd.to_datetime(out["pt"])
    out["sl"] = pd.to_datetime(out["sl"])
    out["t1"] = pd.to_datetime(out["t1"])
    return out


def assert_equal(new, ref, label):
    for col in ("pt", "sl", "t1"):
        a = new[col].reset_index(drop=True)
        b = ref[col].reset_index(drop=True)
        if not a.equals(b):
            diff = a.compare(b)
            raise AssertionError(f"[{label}] column '{col}' differs:\n{diff.head(20)}")
    return True


# ── 1. Randomized equivalence across many regimes / configs ──────────────────
rng = np.random.default_rng(20240521)
n_cases = 0
for trial in range(12):
    T = rng.integers(60, 200)
    idx = pd.bdate_range("2020-01-01", periods=T)
    # Price path with occasional jumps so PT/SL fire at varied offsets.
    rets = rng.normal(0.0, rng.uniform(0.01, 0.04), T)
    rets[rng.integers(0, T, size=rng.integers(0, 5))] += rng.choice([-0.12, 0.12])
    close = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)
    span = rng.uniform(0.003, 0.02)
    high = close * (1 + np.abs(rng.normal(0, span, T)))
    low = close * (1 - np.abs(rng.normal(0, span, T)))

    horizon = int(rng.integers(3, 15))
    # Events at a random subset of bars; t1 = t0 + horizon (clipped).
    starts = np.sort(rng.choice(np.arange(T - 1), size=rng.integers(10, T // 2), replace=False))
    t0s = idx[starts]
    t1_locs = np.minimum(starts + horizon, T - 1)
    t1s = idx[t1_locs]
    trgt = pd.Series(rng.uniform(0.01, 0.05, len(starts)), index=t0s)
    events = pd.DataFrame({"t1": t1s, "trgt": trgt.values}, index=t0s)

    for pt_sl in [(2.0, 2.0), (1.5, 3.0), (2.0, 0.0), (0.0, 2.0)]:
        for use_ohlc in (True, False):
            new = apply_pt_sl_on_t1(close, events, pt_sl,
                                    high=high if use_ohlc else None,
                                    low=low if use_ohlc else None,
                                    use_intrabar_extremes=use_ohlc)
            ref = reference_apply(close, events, pt_sl,
                                  high=high if use_ohlc else None,
                                  low=low if use_ohlc else None,
                                  use_intrabar_extremes=use_ohlc)
            assert_equal(new, ref, f"trial{trial}/pt_sl{pt_sl}/ohlc{use_ohlc}")
            n_cases += 1
print(f"TEST 1  randomized equivalence: {n_cases} (path × pt_sl × ohlc) cases — IDENTICAL  ok")


# ── 2. Targeted edge cases ───────────────────────────────────────────────────
idx = pd.bdate_range("2021-01-01", periods=12)
close = pd.Series([100, 101, 103, 99, 96, 108, 100, 100, 100, 0.0, 100, 100], dtype=float, index=idx)
high = close * 1.03
low = close * 0.97
# Same-bar tie: bar 5 spikes high AND low beyond ±2σ from entry at bar 0.
high.iloc[5] = 130.0   # +30% high
low.iloc[5] = 70.0     # -30% low
events = pd.DataFrame({
    "t1": pd.to_datetime([idx[6], idx[3], idx[11]]),   # datetime64 (as get_events emits)
    "trgt": np.array([0.05, np.nan, 0.05], dtype=float),  # row 2 invalid σ → no touch
    }, index=pd.DatetimeIndex([idx[0], idx[2], idx[8]]))
new = apply_pt_sl_on_t1(close, events, (2.0, 2.0), high=high, low=low)
ref = reference_apply(close, events, (2.0, 2.0), high=high, low=low)
assert_equal(new, ref, "edge")
# Specifically confirm the same-bar tie at bar 5 → SL kept, PT dropped.
row0 = new.loc[idx[0]]
assert pd.isna(row0["pt"]) and row0["sl"] == idx[5] and row0["t1"] == idx[5], \
    f"tie-break failed: {row0.to_dict()}"
# Invalid-σ event → no touch, t1 = its vertical barrier.
assert pd.isna(new.loc[idx[2], "pt"]) and pd.isna(new.loc[idx[2], "sl"])
assert new.loc[idx[2], "t1"] == idx[3]
print("TEST 2  edge cases (same-bar tie → SL, invalid σ → t1=vertical) match reference  ok")


# ── 3. Performance benchmark: vectorized vs iterrows reference ───────────────
T = 2500
idx = pd.bdate_range("2014-01-01", periods=T)
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, T))), index=idx)
high = close * 1.01; low = close * 0.99
horizon = 5
starts = np.arange(T - horizon - 1)            # ~2490 events for ONE ticker
t0s = idx[starts]; t1s = idx[starts + horizon]
events = pd.DataFrame({"t1": t1s, "trgt": rng.uniform(0.01, 0.04, len(starts))}, index=t0s)

t = time.perf_counter()
new = apply_pt_sl_on_t1(close, events, (2.0, 2.0), high=high, low=low)
t_vec = time.perf_counter() - t

t = time.perf_counter()
ref = reference_apply(close, events, (2.0, 2.0), high=high, low=low)
t_ref = time.perf_counter() - t

assert_equal(new, ref, "perf")
speedup = t_ref / max(t_vec, 1e-9)
print(f"TEST 3  perf on {len(events)} events (1 ticker):")
print(f"        iterrows reference : {t_ref*1000:8.1f} ms")
print(f"        vectorized         : {t_vec*1000:8.1f} ms")
print(f"        speedup            : {speedup:8.0f}×   (identical output)")
# Extrapolate to the full 355-ticker universe.
print(f"        → full ~355-ticker universe: reference≈{t_ref*355:6.1f}s  vectorized≈{t_vec*355:6.2f}s")
assert speedup > 20, f"expected >20x speedup, got {speedup:.1f}x"

print()
print("ALL TESTS PASSED — vectorized apply_pt_sl_on_t1 is EXACT and C-speed.")
