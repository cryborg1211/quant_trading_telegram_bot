"""Tests for portfolio/construction.py — LW shrinkage, vol targeting, Kelly, MV constraints."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ast
with open("src/portfolio/construction.py", encoding="utf-8") as f:
    ast.parse(f.read())
print("AST parse OK")

import numpy as np
from src.portfolio.construction import (
    PortfolioConstraints,
    get_ledoit_wolf_cov,
    volatility_target_weights,
    kelly_scalar,
    kelly_optimize,
    mean_variance_optimize,
    PortfolioConstructor,
    TRADING_DAYS,
)

rng = np.random.default_rng(7)


# ============================================================================
# TEST 1: Ledoit-Wolf shrinkage vs sklearn (must match closely)
# ============================================================================
N, T = 12, 180
# Build a true covariance with sector structure, then sample returns.
A = rng.standard_normal((N, N))
true_cov = A @ A.T / N + np.eye(N) * 0.5
X = rng.multivariate_normal(np.zeros(N), true_cov, size=T)

Sigma, delta = get_ledoit_wolf_cov(X)

from sklearn.covariance import LedoitWolf
lw = LedoitWolf(assume_centered=False).fit(X)
Sigma_sk, delta_sk = lw.covariance_, lw.shrinkage_

# PSD check
eigs = np.linalg.eigvalsh(Sigma)
assert eigs.min() > -1e-10, f"Sigma not PSD: min eig {eigs.min()}"
# Shrinkage intensity should match sklearn within a small tolerance
assert abs(delta - delta_sk) < 0.05, f"delta mismatch: ours={delta:.4f} sklearn={delta_sk:.4f}"
# Covariance matrices should be close (Frobenius relative error)
rel_err = np.linalg.norm(Sigma - Sigma_sk) / np.linalg.norm(Sigma_sk)
assert rel_err < 0.05, f"cov Frobenius rel-err too high: {rel_err:.4f}"
print(f"TEST 1  Ledoit-Wolf  ours δ={delta:.4f}  sklearn δ={delta_sk:.4f}  "
      f"cov_rel_err={rel_err:.4f}  PSD min_eig={eigs.min():.4f}  ok")


# ============================================================================
# TEST 2: LW handles the N >> T degenerate case (sample cov is singular)
# ============================================================================
N2, T2 = 50, 20   # fewer obs than assets — sample cov rank-deficient
X2 = rng.standard_normal((T2, N2))
Sigma2, delta2 = get_ledoit_wolf_cov(X2)
eigs2 = np.linalg.eigvalsh(Sigma2)
# Shrinkage MUST pull it to full rank (all eigenvalues > 0)
assert eigs2.min() > 1e-8, f"shrinkage failed to regularize: min eig {eigs2.min()}"
assert delta2 > 0.2, f"expect heavy shrinkage when N>>T, got δ={delta2:.4f}"
print(f"TEST 2  LW with N({N2})>>T({T2})  δ={delta2:.4f}  min_eig={eigs2.min():.6f} (full rank)  ok")


# ============================================================================
# TEST 3: Volatility targeting hits the target exactly
# ============================================================================
w_raw = np.array([0.3, 0.3, 0.4])
cov3 = np.array([
    [0.0004, 0.0001, 0.00005],
    [0.0001, 0.0009, 0.0002],
    [0.00005, 0.0002, 0.0016],
])  # daily covariance
target = 0.15
w_scaled = volatility_target_weights(w_raw, cov3, target, periods_per_year=252)
achieved_vol = np.sqrt(w_scaled @ cov3 @ w_scaled * 252)
assert abs(achieved_vol - target) < 1e-9, f"vol target miss: {achieved_vol} vs {target}"
# Direction preserved (scaling is positive)
assert np.all(np.sign(w_scaled) == np.sign(w_raw))
print(f"TEST 3  vol targeting  target={target:.2%}  achieved={achieved_vol:.6f}  "
      f"scale={w_scaled[0]/w_raw[0]:.3f}  ok")

# Zero-variance edge case → zeros
w_zero = volatility_target_weights(np.zeros(3), cov3, 0.15)
assert np.all(w_zero == 0.0)
print("        zero-variance book → zeros (no division blow-up)  ok")


# ============================================================================
# TEST 4: kelly_scalar closed form
# ============================================================================
# W=0.6, PF=2.0 → f* = 0.6·(1 - 1/2) = 0.30
assert abs(kelly_scalar(np.array([0.6]), np.array([2.0]))[0] - 0.30) < 1e-12
# W=0.5, PF=1.0 (no edge) → 0
assert abs(kelly_scalar(np.array([0.5]), np.array([1.0]))[0] - 0.0) < 1e-12
# PF<1 (losing) → negative
assert kelly_scalar(np.array([0.5]), np.array([0.5]))[0] < 0
# Vectorized
W = np.array([0.55, 0.60, 0.45, 0.70])
PF = np.array([1.5, 2.0, 0.9, 3.0])
ks = kelly_scalar(W, PF)
expected = W * (1 - 1/PF)
assert np.allclose(ks, expected)
print(f"TEST 4  kelly_scalar  W={W.tolist()}  PF={PF.tolist()}")
print(f"        f*={np.round(ks, 4).tolist()}  (PF<1 → negative)  ok")


# ============================================================================
# TEST 5: kelly_optimize uses covariance (correlated winners get less)
# ============================================================================
# Two assets with identical edge; one pair highly correlated, the other not.
W5 = np.array([0.6, 0.6, 0.6])
PF5 = np.array([2.0, 2.0, 2.0])
# Assets 0 and 1 highly correlated (0.9); asset 2 independent.
cov5 = np.array([
    [0.04, 0.036, 0.0],
    [0.036, 0.04, 0.0],
    [0.0, 0.0, 0.04],
])
k = kelly_optimize(W5, PF5, cov5, fraction=0.5)
print(f"TEST 5  kelly_optimize  weights={np.round(k, 4).tolist()}")
# The independent asset (2) should get MORE than each correlated one (0,1),
# because the correlated pair is effectively one bet split two ways.
assert k[2] > k[0] and k[2] > k[1], f"independent asset should get more: {k}"
assert abs(k[0] - k[1]) < 1e-9, "symmetric correlated pair should be equal"
print(f"        independent asset (2) weighted higher than correlated pair (0,1)  ok")

# Half-Kelly is exactly half of full-Kelly
k_full = kelly_optimize(W5, PF5, cov5, fraction=1.0)
assert np.allclose(k, 0.5 * k_full)
print("        half-Kelly == 0.5 × full-Kelly  ok")


# ============================================================================
# TEST 6: mean_variance_optimize — long-only + per-ticker cap respected
# ============================================================================
N6 = 8
A6 = rng.standard_normal((N6, N6))
cov6 = A6 @ A6.T / N6 + np.eye(N6) * 0.3
mu6 = rng.uniform(0.0, 0.05, N6)
tickers6 = [f"T{i}" for i in range(N6)]

cons6 = PortfolioConstraints(max_weight=0.20, long_only=True, target_leverage=1.0)
res6 = mean_variance_optimize(mu6, cov6, tickers6, cons6, risk_aversion=5.0)
w6 = res6["weights"]
print(f"TEST 6  MV long-only + 20% cap  converged={res6['converged']}")
print(f"        weights={np.round(w6, 4).tolist()}")
print(f"        sum={w6.sum():.6f}  max={w6.max():.4f}  min={w6.min():.6f}")
assert res6["converged"], res6["message"]
assert w6.min() >= -1e-7, f"long-only violated: min {w6.min()}"
assert w6.max() <= 0.20 + 1e-6, f"per-ticker cap violated: max {w6.max()}"
assert abs(w6.sum() - 1.0) < 1e-6, f"budget violated: sum {w6.sum()}"
print("        long-only ✓  per-ticker cap ✓  budget ✓")


# ============================================================================
# TEST 7: Sector caps enforced (Banking + Real Estate ≤ 30% each)
# ============================================================================
N7 = 9
A7 = rng.standard_normal((N7, N7))
cov7 = A7 @ A7.T / N7 + np.eye(N7) * 0.2
# Make banking names have the highest edge → optimizer WANTS to overweight them
mu7 = np.array([0.10, 0.10, 0.10,   # 0-2 BANKS (high edge)
                0.08, 0.08, 0.08,   # 3-5 REAL_ESTATE
                0.02, 0.02, 0.02])  # 6-8 OTHER
tickers7 = ["VCB", "BID", "CTG", "VHM", "NVL", "DXG", "FPT", "MWG", "HPG"]
sector_map = {
    "VCB": "BANKS", "BID": "BANKS", "CTG": "BANKS",
    "VHM": "REAL_ESTATE", "NVL": "REAL_ESTATE", "DXG": "REAL_ESTATE",
    "FPT": "OTHER", "MWG": "OTHER", "HPG": "OTHER",
}
cons7 = PortfolioConstraints(
    max_weight=0.15,
    sector_caps={"BANKS": 0.30, "REAL_ESTATE": 0.30},
    ticker_to_sector=sector_map,
    long_only=True,
    target_leverage=1.0,
)
res7 = mean_variance_optimize(mu7, cov7, tickers7, cons7, risk_aversion=1.0)
w7 = res7["weights"]
banks = w7[[0, 1, 2]].sum()
realest = w7[[3, 4, 5]].sum()
print(f"TEST 7  sector caps  converged={res7['converged']}")
print(f"        BANKS exposure={banks:.4f} (cap 0.30)  REAL_ESTATE={realest:.4f} (cap 0.30)")
print(f"        active_sector_caps={res7['active_sector_caps']}")
assert res7["converged"], res7["message"]
assert banks <= 0.30 + 1e-6, f"BANKS cap violated: {banks}"
assert realest <= 0.30 + 1e-6, f"REAL_ESTATE cap violated: {realest}"
assert w7.max() <= 0.15 + 1e-6
# Despite banks having the highest edge, the cap binds (should be near 0.30)
assert banks > 0.25, f"expected banking near its cap given high edge, got {banks}"
print("        sector caps bind correctly even when edge wants more  ok")


# ============================================================================
# TEST 8: Vol-target as a hard QP constraint
# ============================================================================
# Build a REALISTICALLY-scaled daily covariance (per-asset daily vol ~2%):
#   diag(cov) ≈ 0.0004  → single-asset annualized vol ≈ sqrt(0.0004·252) ≈ 0.32
# A 15% portfolio target is then feasible via diversification.
A8 = rng.standard_normal((N6, N6))
cov8 = (A8 @ A8.T / N6 + np.eye(N6)) * 0.0002
mu8 = rng.uniform(0.0, 0.03, N6)
cons8 = PortfolioConstraints(
    max_weight=0.40, long_only=True, target_leverage=1.0,
    target_vol=0.15, periods_per_year=252,
)
res8 = mean_variance_optimize(mu8, cov8, tickers6, cons8, risk_aversion=0.5)
ann_vol = res8["annualized_vol"]
print(f"TEST 8  vol-ceiling QP  converged={res8['converged']}  ann_vol={ann_vol:.4f} (ceiling 0.15)")
assert res8["converged"], res8["message"]
assert ann_vol <= 0.15 + 1e-4, f"vol ceiling violated: {ann_vol}"
# Confirm the constraint actually binds (optimizer pushed up to the ceiling)
assert ann_vol > 0.10, f"expected vol near the ceiling, got {ann_vol}"
print(f"        annualized vol within ceiling AND binding (>{0.10})  ok")


# ============================================================================
# TEST 9: End-to-end PortfolioConstructor
# ============================================================================
N9, T9 = 9, 252
A9 = rng.standard_normal((N9, N9))
true_cov9 = A9 @ A9.T / N9 * 0.0004 + np.eye(N9) * 0.0004
returns9 = rng.multivariate_normal(np.zeros(N9), true_cov9, size=T9)
win_rates9 = rng.uniform(0.45, 0.65, N9)
profit_factors9 = rng.uniform(0.9, 2.5, N9)

constructor = PortfolioConstructor(
    constraints=PortfolioConstraints(
        max_weight=0.15,
        sector_caps={"BANKS": 0.30, "REAL_ESTATE": 0.30},
        ticker_to_sector=sector_map,
        long_only=True,
        target_leverage=1.0,
        target_vol=0.15,
    ),
    kelly_fraction=0.5,
    risk_aversion=2.0,
)
out = constructor.construct(returns9, tickers7, win_rates=win_rates9, profit_factors=profit_factors9)
w9 = out["weights"]
print(f"TEST 9  PortfolioConstructor end-to-end")
print(f"        converged={out['converged']}  shrinkage_δ={out['shrinkage_delta']:.4f}")
print(f"        weights={np.round(w9, 4).tolist()}")
print(f"        sector_exposures={ {k: round(v, 4) for k, v in out['sector_exposures'].items()} }")
print(f"        ann_vol={out['annualized_vol']:.4f}  gross_lev={out['gross_leverage']:.4f}")
assert out["converged"]
assert w9.min() >= -1e-7
assert w9.max() <= 0.15 + 1e-6
assert abs(w9.sum() - 1.0) < 1e-6
assert out["sector_exposures"]["BANKS"] <= 0.30 + 1e-6
assert out["sector_exposures"]["REAL_ESTATE"] <= 0.30 + 1e-6
assert out["annualized_vol"] <= 0.15 + 1e-4
print("        all constraints simultaneously satisfied  ok")


# ============================================================================
# TEST 10: Infeasible config raises (cap too small to cover budget)
# ============================================================================
try:
    bad = PortfolioConstraints(max_weight=0.05, long_only=True, target_leverage=1.0)
    mean_variance_optimize(mu6, cov6, tickers6, bad)  # 0.05 × 8 = 0.40 < 1.0
    raise AssertionError("should have raised infeasibility")
except ValueError as e:
    print(f"TEST 10  infeasible cap detected  →  {str(e)[:60]}...  ok")


print()
print("ALL TESTS PASSED — risk engine produces feasible, constrained, vol-targeted books.")
