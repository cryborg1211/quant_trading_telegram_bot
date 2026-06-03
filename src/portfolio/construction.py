"""
src/portfolio/construction.py — Quant Engine V2.0, Phase 7

Risk & sizing engine: LSTM directional signals → constrained portfolio weights.

Pipeline
────────
    1. get_ledoit_wolf_cov        Σ̂ = δ·F + (1−δ)·S   (shrunk, well-conditioned)
    2. kelly_scalar / kelly_optimize   per-asset & multi-asset fractional Kelly
    3. mean_variance_optimize     long-only QP with per-ticker + sector caps
    4. volatility_target_weights  scale to a target annualized ex-ante vol

References
    Ledoit & Wolf (2004) — well-conditioned shrinkage to scaled identity.
    Kelly (1956); MacLean, Thorp, Ziemba (2011) — fractional-Kelly safety.
    Markowitz (1952) — mean-variance; constraints via SLSQP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.optimize as opt

LOGGER = logging.getLogger("portfolio.construction")

__all__ = [
    "PortfolioConstraints",
    "get_ledoit_wolf_cov",
    "volatility_target_weights",
    "kelly_scalar",
    "kelly_optimize",
    "mean_variance_optimize",
    "PortfolioConstructor",
]

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioConstraints:
    """Hard bounds for the mean-variance optimizer."""
    max_weight: float = 0.10
    """Per-ticker cap (e.g. 0.10 = no name above 10% of book)."""

    sector_caps: dict[str, float] = field(default_factory=dict)
    """{sector_name: max_fraction}, e.g. {'BANKS': 0.30, 'REAL_ESTATE': 0.30}."""

    ticker_to_sector: dict[str, str] = field(default_factory=dict)
    """{ticker: sector_name}.  Tickers absent here are uncapped by sector."""

    long_only: bool = True
    """If True, w_i ≥ 0 (HOSE retail cannot easily short)."""

    target_leverage: float = 1.0
    """Σ w_i — gross book leverage (1.0 = fully invested, no leverage)."""

    target_vol: float | None = None
    """Annualized ex-ante vol ceiling as a QP constraint; None disables it."""

    periods_per_year: int = TRADING_DAYS
    """Annualization factor for vol (252 for daily returns)."""

    min_weight_floor: float = 0.0
    """Lower bound when long_only (default 0). Set >0 to force min position size."""

    def sector_mask(self, tickers: Sequence[str], sector: str) -> np.ndarray:
        """Boolean mask of which tickers belong to `sector`."""
        return np.array(
            [self.ticker_to_sector.get(t) == sector for t in tickers],
            dtype=bool,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ledoit-Wolf shrinkage covariance
# ─────────────────────────────────────────────────────────────────────────────

def get_ledoit_wolf_cov(
    returns: np.ndarray,
    *,
    assume_centered: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) shrinkage covariance to a scaled-identity target.

        Σ̂ = δ·F + (1−δ)·S

        S = (1/T)·Xᵀ·X                      sample covariance (MLE divisor T)
        F = (tr(S)/N)·I                      scaled-identity target
        δ ∈ [0, 1]                           optimal shrinkage under Frobenius loss

    Optimal δ (their Eq. for the scaled-identity target):

        π̂  = (1/T²) Σ_t ‖xₜ xₜᵀ − S‖²_F      dispersion of the sample estimator
        d̂² = ‖S − F‖²_F                       distance of S to the target
        δ̂  = clip( π̂ / d̂² , 0, 1 )

    The π̂ inner sum is computed WITHOUT materialising the (T, N, N) outer
    products, using the identity:

        ‖xₜ xₜᵀ − S‖²_F = ‖xₜ‖⁴ − 2·xₜᵀ S xₜ + ‖S‖²_F

    Output is guaranteed PSD (convex combination of two PSD matrices).

    Args:
        returns:         (T, N) returns matrix.
        assume_centered: skip mean-removal if already centered.

    Returns:
        (Sigma, shrinkage_delta)
    """
    X = np.asarray(returns, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N); got shape {X.shape}")
    T, N = X.shape
    if T < 2:
        raise ValueError(f"need T >= 2 observations, got {T}")

    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)

    S = (X.T @ X) / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # π̂ via the no-outer-product identity:
    sq_norm = np.einsum("ti,ti->t", X, X)            # ‖xₜ‖²            (T,)
    xtx_fro_sq = sq_norm ** 2                          # ‖xₜ xₜᵀ‖²_F = ‖xₜ‖⁴
    xt_S_x = np.einsum("ti,ij,tj->t", X, S, X)         # xₜᵀ S xₜ        (T,)
    S_fro_sq = float((S * S).sum())                    # ‖S‖²_F
    pi_per_t = xtx_fro_sq - 2.0 * xt_S_x + S_fro_sq
    pi_hat = float(pi_per_t.sum()) / (T ** 2)

    d_sq = float(((S - F) ** 2).sum())

    delta = 0.0 if d_sq < 1e-15 else float(np.clip(pi_hat / d_sq, 0.0, 1.0))
    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, delta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Volatility targeting
# ─────────────────────────────────────────────────────────────────────────────

def volatility_target_weights(
    raw_weights: np.ndarray,
    cov: np.ndarray,
    target_annual_vol: float,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> np.ndarray:
    """
    Linearly scale raw weights so ex-ante annualized portfolio vol == target.

        σ_p (per period) = sqrt(wᵀ Σ w)
        σ_p (annual)     = σ_p · sqrt(periods_per_year)
        k                = target_annual_vol / σ_p_annual
        w_scaled         = k · w_raw

    A zero-variance portfolio (all-cash or degenerate) returns zeros — there is
    no finite scale that produces positive vol from a zero-risk book.

    Args:
        raw_weights:        (N,) un-scaled weights (direction + relative size).
        cov:                (N, N) per-period covariance (e.g. Ledoit-Wolf).
        target_annual_vol:  desired annualized vol (0.15 = 15%).
        periods_per_year:   annualization factor.

    Returns:
        Scaled (N,) weights. Multiply book notional to get position sizes.
    """
    w = np.asarray(raw_weights, dtype=np.float64)
    Sigma = np.asarray(cov, dtype=np.float64)

    var_p = float(w @ Sigma @ w)
    if var_p <= 0.0:
        LOGGER.warning("volatility_target_weights: non-positive ex-ante variance; returning zeros")
        return np.zeros_like(w)

    vol_p_annual = np.sqrt(var_p * periods_per_year)
    scale = target_annual_vol / vol_p_annual
    return w * scale


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fractional Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def kelly_scalar(
    win_rate: np.ndarray,
    profit_factor: np.ndarray,
) -> np.ndarray:
    """
    Per-asset binary Kelly fraction from win-rate W and profit-factor PF.

    With PF = (total wins) / (total losses) and win:loss size ratio
    b = PF·(1−W)/W, the binary Kelly criterion f* = (W·b − (1−W)) / b
    simplifies to a clean closed form:

        f*  =  W · ( 1 − 1/PF )

    • PF > 1  ⇒ f* > 0   (edge present)
    • PF = 1  ⇒ f* = 0   (no edge — flat)
    • PF < 1  ⇒ f* < 0   (negative edge — would short if allowed)

    Args:
        win_rate:      (N,) historical hit-rate per asset, ∈ [0, 1].
        profit_factor: (N,) gross-profit / gross-loss per asset, > 0.

    Returns:
        (N,) raw Kelly fractions (can be negative; clamp downstream if long-only).
    """
    W = np.asarray(win_rate, dtype=np.float64)
    PF = np.asarray(profit_factor, dtype=np.float64)
    PF_safe = np.where(PF <= 1e-9, 1e-9, PF)
    return W * (1.0 - 1.0 / PF_safe)


def kelly_optimize(
    win_rates: np.ndarray,
    profit_factors: np.ndarray,
    cov: np.ndarray,
    *,
    fraction: float = 0.5,
    expected_return_scale: float = 1.0,
) -> np.ndarray:
    """
    Multi-asset fractional-Kelly allocation using the covariance matrix.

        μ   = kelly_scalar(W, PF) · expected_return_scale     per-asset "edge"
        w*  = fraction · Σ⁻¹ · μ                              growth-optimal mix

    The Σ⁻¹ couples the per-asset edges through their correlations: two highly
    correlated winners are NOT each given full size (that would double the bet
    on one factor). `fraction` ∈ (0, 1] applies the fractional-Kelly haircut
    (0.5 = half-Kelly) which sharply reduces drawdown for a modest growth cost.

    The raw result may contain negatives; pass it to `mean_variance_optimize`
    (long_only=True) to enforce the no-short VN constraint.

    Args:
        win_rates:             (N,) hit-rates.
        profit_factors:        (N,) profit factors.
        cov:                   (N, N) covariance (use Ledoit-Wolf).
        fraction:              fractional-Kelly multiplier ∈ (0, 1].
        expected_return_scale: scales the per-asset edge into return units.

    Returns:
        (N,) fractional-Kelly weights (unconstrained; may be negative).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    mu = kelly_scalar(win_rates, profit_factors) * expected_return_scale
    Sigma = np.asarray(cov, dtype=np.float64)
    try:
        inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        LOGGER.warning("kelly_optimize: singular covariance; falling back to pseudo-inverse")
        inv = np.linalg.pinv(Sigma)
    return fraction * (inv @ mu)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Constrained mean-variance optimization
# ─────────────────────────────────────────────────────────────────────────────

def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    tickers: Sequence[str],
    constraints: PortfolioConstraints,
    *,
    risk_aversion: float = 1.0,
    w0: np.ndarray | None = None,
    max_iter: int = 500,
) -> dict:
    """
    Long-only mean-variance QP with per-ticker and sector caps, solved by SLSQP.

        minimize    −μᵀw + ½·λ·wᵀΣw
        subject to  Σ wᵢ = target_leverage                       (budget)
                    0 ≤ wᵢ ≤ max_weight        ∀ i               (long-only + cap)
                    Σ_{i∈s} wᵢ ≤ sector_cap[s] ∀ sectors s        (sector caps)
                    wᵀΣw · periods_per_year ≤ target_vol²         (optional vol ceiling)

    Analytic gradients are supplied for both the objective and the linear
    constraints to keep SLSQP fast and well-conditioned.

    Args:
        expected_returns: (N,) μ — e.g. fractional-Kelly edges.
        cov:              (N, N) Σ — Ledoit-Wolf recommended.
        tickers:          length-N ticker list (for sector mapping).
        constraints:      PortfolioConstraints.
        risk_aversion:    λ ≥ 0 — higher ⇒ more weight on variance.
        w0:               (N,) warm-start; default = feasible equal-weight.
        max_iter:         SLSQP iteration cap.

    Returns:
        dict: weights, expected_return, portfolio_vol, annualized_vol,
              converged, message, n_iter, active_sector_caps.
    """
    mu = np.asarray(expected_returns, dtype=np.float64)
    Sigma = np.asarray(cov, dtype=np.float64)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"cov shape {Sigma.shape} != ({n}, {n})")
    if len(tickers) != n:
        raise ValueError(f"len(tickers)={len(tickers)} != n={n}")

    lev = constraints.target_leverage
    lo = constraints.min_weight_floor if constraints.long_only else -constraints.max_weight
    hi = constraints.max_weight
    bounds = [(lo, hi)] * n

    # Feasibility guard: caps must be able to absorb the budget.
    if hi * n < lev - 1e-9:
        raise ValueError(
            f"infeasible: max_weight {hi} × {n} names = {hi*n:.2f} < target_leverage {lev}"
        )

    def objective(w: np.ndarray) -> float:
        return float(-mu @ w + 0.5 * risk_aversion * (w @ Sigma @ w))

    def objective_grad(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (Sigma @ w)

    cons: list[dict] = [{
        "type": "eq",
        "fun": lambda w: float(w.sum() - lev),
        "jac": lambda w: np.ones_like(w),
    }]

    active_caps: list[str] = []
    for sector, cap in constraints.sector_caps.items():
        mask = constraints.sector_mask(tickers, sector).astype(np.float64)
        if mask.sum() == 0:
            continue
        active_caps.append(sector)
        cons.append({
            "type": "ineq",
            "fun": (lambda w, m=mask, c=cap: float(c - m @ w)),   # ≥ 0 feasible
            "jac": (lambda w, m=mask: -m),
        })

    if constraints.target_vol is not None:
        target_var_pp = (constraints.target_vol ** 2) / constraints.periods_per_year
        cons.append({
            "type": "ineq",
            "fun": (lambda w, tv=target_var_pp: float(tv - w @ Sigma @ w)),
            "jac": (lambda w: -2.0 * (Sigma @ w)),
        })

    # SLSQP is sensitive to the starting point, especially when a non-linear
    # (vol-ceiling) constraint is active.  Try several deterministic warm
    # starts and keep the first that converges (or the best objective if none).
    starts: list[np.ndarray] = []
    if w0 is not None:
        starts.append(np.clip(np.asarray(w0, dtype=np.float64), lo, hi))
    starts.append(np.clip(np.full(n, lev / n), lo, hi))           # equal weight
    inv_var = 1.0 / (np.diag(Sigma) + 1e-12)                      # min-var-ish
    starts.append(np.clip(inv_var / inv_var.sum() * lev, lo, hi))
    if np.any(mu > 0):                                            # edge-tilted
        edge = np.clip(mu, 0, None)
        if edge.sum() > 0:
            starts.append(np.clip(edge / edge.sum() * lev, lo, hi))

    result = None
    for s in starts:
        cand = opt.minimize(
            objective, s, jac=objective_grad, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"ftol": 1e-10, "maxiter": max_iter, "disp": False},
        )
        if cand.success:
            result = cand
            break
        if result is None or cand.fun < result.fun:
            result = cand  # keep best-objective fallback if none converge

    w = result.x
    var_pp = float(w @ Sigma @ w)
    return {
        "weights": w,
        "expected_return": float(mu @ w),
        "portfolio_vol": float(np.sqrt(max(var_pp, 0.0))),
        "annualized_vol": float(np.sqrt(max(var_pp, 0.0) * constraints.periods_per_year)),
        "converged": bool(result.success),
        "message": str(result.message),
        "n_iter": int(result.nit),
        "active_sector_caps": active_caps,
        "gross_leverage": float(np.abs(w).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioConstructor:
    """
    End-to-end: historical returns + signal stats → constrained, vol-targeted weights.

    Steps:
        1. Σ̂ ← Ledoit-Wolf(returns)
        2. μ ← fractional-Kelly edge from (win_rate, profit_factor)
        3. w ← mean_variance_optimize(μ, Σ̂, constraints)        (long-only, caps)
        4. if constraints.target_vol set and the QP vol-ceiling was NOT used as a
           hard constraint, optionally re-scale via volatility_target_weights.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        *,
        kelly_fraction: float = 0.5,
        risk_aversion: float = 1.0,
    ) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        self.constraints = constraints
        self.kelly_fraction = kelly_fraction
        self.risk_aversion = risk_aversion

    def construct(
        self,
        returns: np.ndarray,
        tickers: Sequence[str],
        *,
        win_rates: np.ndarray | None = None,
        profit_factors: np.ndarray | None = None,
        expected_returns: np.ndarray | None = None,
    ) -> dict:
        """
        Build target weights.

        Provide EITHER (win_rates, profit_factors) — Kelly edge is derived — OR
        an explicit `expected_returns` vector (e.g. LSTM-implied μ).

        Returns the `mean_variance_optimize` dict plus `shrinkage_delta`,
        `mu`, and `sector_exposures`.
        """
        Sigma, delta = get_ledoit_wolf_cov(returns)

        if expected_returns is not None:
            mu = np.asarray(expected_returns, dtype=np.float64)
        elif win_rates is not None and profit_factors is not None:
            mu = self.kelly_fraction * kelly_scalar(win_rates, profit_factors)
        else:
            raise ValueError(
                "provide either expected_returns or (win_rates AND profit_factors)"
            )

        out = mean_variance_optimize(
            mu, Sigma, tickers, self.constraints,
            risk_aversion=self.risk_aversion,
        )
        out["shrinkage_delta"] = delta
        out["mu"] = mu

        # Sector exposure report
        w = out["weights"]
        exposures: dict[str, float] = {}
        for sector in set(self.constraints.ticker_to_sector.values()):
            mask = self.constraints.sector_mask(tickers, sector)
            exposures[sector] = float(w[mask].sum())
        out["sector_exposures"] = exposures
        return out
