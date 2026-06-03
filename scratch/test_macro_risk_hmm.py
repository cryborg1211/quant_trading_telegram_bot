"""Standalone test for the Macro Risk Oracle (2-state Gaussian HMM)."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
from src.models.macro_risk_hmm import (
    build_market_proxy_returns, train_macro_risk_hmm, MacroRiskHMM,
)

rng = np.random.default_rng(0)

# Regime-diverse timeline: bull/bear/bull/bear so the TRAIN split sees BOTH.
# Regimes are cleanly separable on the MEAN (as real VN bull vs the sustained
# 2022 bear are), which is what an unsupervised return-HMM keys on.
idx = pd.bdate_range("2018-01-01", periods=420)
segments = (
    list(rng.normal(0.0016, 0.008, 110)) +   # bull
    list(rng.normal(-0.0045, 0.013, 100)) +   # bear
    list(rng.normal(0.0016, 0.008, 110)) +   # bull
    list(rng.normal(-0.0045, 0.013, 100))     # bear
)
mret = pd.Series(np.array(segments), index=idx)
cutoff = idx[270]   # train on first 270 days (covers bull + bear + bull)

# ── 1. Train on in-sample only; identify Bull state ──
hmm = train_macro_risk_hmm(mret[mret.index < cutoff], seed=42)
print(f"TEST 1  bull_state={hmm.bull_state}  means={[round(m,5) for m in hmm.state_means]}  "
      f"vars={[round(v,6) for v in hmm.state_vars]}")
# Bull state must have the higher mean and a sane (non-degenerate) variance.
assert hmm.state_means[hmm.bull_state] == max(hmm.state_means)
assert max(hmm.state_vars) < 0.01, f"degenerate state variance: {hmm.state_vars}"
print("        bull = higher-mean state, no degenerate variance  ok")

# ── 2. Filtered (leak-free) P(Bull): high in bull windows, low in bear ──
pb = hmm.p_bull_series(mret, filtered=True, min_obs=20)
# Windows clearly inside each regime (avoid transition edges):
bull1 = pb.iloc[40:90].mean()
bear1 = pb.iloc[120:180].mean()
bull2 = pb.iloc[210:290].mean()      # spans cutoff into OOS bull
bear2 = pb.iloc[320:390].mean()      # OOS bear
print(f"TEST 2  P(Bull): bull1={bull1:.2f}  bear1={bear1:.2f}  bull2={bull2:.2f}  bear2(OOS)={bear2:.2f}")
assert bull1 > 0.6 and bull2 > 0.6, "should be bullish in bull windows"
assert bear1 < 0.4 and bear2 < 0.4, "should de-risk in bear windows"
print("        regime de-risking confirmed (soft, continuous)  ok")

# ── 3. Leak-free: filtered[t] unchanged when future bars are appended ──
pb_trunc = hmm.p_bull_series(mret.iloc[:300], filtered=True, min_obs=20)
assert abs(pb_trunc.iloc[290] - pb.iloc[290]) < 1e-9, "filtered value changed with future → LEAK"
print(f"TEST 3  leak-free filtered: day290 identical truncated vs full "
      f"({pb_trunc.iloc[290]:.4f})  ok")

# Smoothed (leaky) differs from filtered — proves predict_proba peeks ahead.
pb_smooth = hmm.p_bull_series(mret, filtered=False)
diff = (pb_smooth - pb).abs().max()
print(f"        smoothed vs filtered max|Δ|={diff:.3f} (smoothing DOES peek ahead → filtered used for OOS)")

# ── 4. Live latest + joblib round-trip (persistence for deployment) ──
import joblib, tempfile
latest = hmm.p_bull_latest(mret)
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "hmm.joblib")
    joblib.dump(hmm, p)
    hmm2 = joblib.load(p)
    assert isinstance(hmm2, MacroRiskHMM)
    assert abs(hmm2.p_bull_latest(mret) - latest) < 1e-12
print(f"TEST 4  latest live P(Bull)={latest:.3f}  joblib round-trip identical  ok")

# ── 5. proxy builder from a polars panel ──
days = pd.bdate_range("2021-01-01", periods=30).date.tolist()
rows = []
for i in range(5):
    px = 100.0
    for d in days:
        px *= (1 + rng.normal(0.001, 0.01)); rows.append({"ticker": f"T{i}", "date": d, "close": px})
proxy = build_market_proxy_returns(pl.from_pandas(pd.DataFrame(rows)))
assert len(proxy) == 29 and proxy.notna().all()   # 30 days → 29 returns
print(f"TEST 5  market proxy returns: {len(proxy)} obs, all finite  ok")

print()
print("MACRO RISK HMM OK")
