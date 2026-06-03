"""
src/models/statistical_gates.py — Quant Engine V3.0

Production-gate statistical utilities, extracted from the V2-era
`train_lstm_v2.py` (which is now deleted along with the LSTM stack).
Pure functions, no torch / no LSTM dependencies — safe to import from any
backtest or live-bot context.

Contents
────────
    EULER_MASCHERONI         constant γ ≈ 0.5772 used by DSR
    deflated_sharpe(...)     Bailey & de Prado (2014) DSR + p-value
    cscv_pbo(...)            Bailey, Borwein, de Prado, Zhu (2014) PBO via CSCV

Both gates are unchanged from the V2 implementation — only relocated, so the
DSR/PBO numbers in any historical teardown report are byte-for-byte reproducible.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

LOGGER = logging.getLogger("models.statistical_gates")

EULER_MASCHERONI = 0.5772156649015329

__all__ = ["EULER_MASCHERONI", "deflated_sharpe", "cscv_pbo"]


# ─────────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio (Bailey & de Prado 2014)
# ─────────────────────────────────────────────────────────────────────────────
def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    *,
    annualisation: float = 1.0,
) -> dict[str, Any]:
    """
    Bailey & de Prado (2014) Deflated Sharpe Ratio.

    Math
    ────
    Let SR̂ = T-period Sharpe of `returns` (per-period units).
    Expected max-Sharpe under the null of independent random skill,
    given N trials, with Euler-Mascheroni correction (their Eq. 4):

        E[max SR] ≈ (1 − γ) · Φ⁻¹(1 − 1/N)  +  γ · Φ⁻¹(1 − 1/(N·e))

    where γ ≈ 0.5772 is Euler-Mascheroni and e is the natural log base.
    This is the per-period 'noise floor' a single best-of-N config must clear.

    DSR z-score (their Eq. 9), accounting for non-Gaussian moments:

        DSR̂ = (SR̂ − E[max SR]) · √(T − 1)
                 ─────────────────────────────────────────
                 √( 1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂² )

    where γ₃ = skew, γ₄ = kurtosis (raw, not excess).

    p-DSR = Φ(DSR̂).  The model is 'significant' iff p-DSR ≥ 1 − α.

    Args:
        returns:       1-D array of per-period net returns.
        n_trials:      number of distinct hyper-parameter / model configurations
                       tested (the 'N' in the Bonferroni-like correction).
        annualisation: multiply SR by sqrt(annualisation) for reporting
                       (does NOT enter the DSR z-score itself; that operates
                       in per-period units).

    Returns:
        dict with sr_per_period, sr_annualised, sr0_per_period,
        sr0_annualised, dsr_z, p_dsr, skew, kurtosis, n_obs, n_trials, valid.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    T = r.size

    if T < 30:
        return {
            "sr_per_period": float("nan"),
            "sr_annualised": float("nan"),
            "sr0_per_period": float("nan"),
            "sr0_annualised": float("nan"),
            "dsr_z": float("nan"),
            "p_dsr": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
            "n_obs": T,
            "n_trials": int(n_trials),
            "valid": False,
            "warning": f"insufficient observations (T={T} < 30)",
        }

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 1e-12:
        return {
            "sr_per_period": 0.0,
            "sr_annualised": 0.0,
            "sr0_per_period": 0.0,
            "sr0_annualised": 0.0,
            "dsr_z": 0.0,
            "p_dsr": 0.5,
            "skew": 0.0,
            "kurtosis": 3.0,
            "n_obs": T,
            "n_trials": int(n_trials),
            "valid": False,
            "warning": "zero return variance",
        }

    sr_pp = mu / sigma                                  # per-period Sharpe
    skew = float(pd.Series(r).skew())
    excess_kurt = float(pd.Series(r).kurtosis())
    kurt_raw = excess_kurt + 3.0

    # Expected max-Sharpe under N trials (per-period)
    if n_trials >= 2:
        sr0_pp = (
            (1.0 - EULER_MASCHERONI) * scipy_norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_MASCHERONI * scipy_norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        ) / np.sqrt(T)
    else:
        sr0_pp = 0.0  # single-trial case — no multiplicity penalty

    # DSR denominator: account for non-normal moments.
    inner = 1.0 - skew * sr_pp + ((kurt_raw - 1.0) / 4.0) * (sr_pp ** 2)
    if inner <= 0 or not np.isfinite(inner):
        return {
            "sr_per_period": float(sr_pp),
            "sr_annualised": float(sr_pp * np.sqrt(annualisation)),
            "sr0_per_period": float(sr0_pp),
            "sr0_annualised": float(sr0_pp * np.sqrt(annualisation)),
            "dsr_z": float("nan"),
            "p_dsr": float("nan"),
            "skew": skew,
            "kurtosis": kurt_raw,
            "n_obs": T,
            "n_trials": int(n_trials),
            "valid": False,
            "warning": f"DSR denominator ill-defined (1 − γ₃·SR + (γ₄−1)/4·SR² = {inner:.3e})",
        }

    dsr_z = (sr_pp - sr0_pp) * np.sqrt((T - 1) / inner)
    p_dsr = float(scipy_norm.cdf(dsr_z))

    return {
        "sr_per_period": float(sr_pp),
        "sr_annualised": float(sr_pp * np.sqrt(annualisation)),
        "sr0_per_period": float(sr0_pp),
        "sr0_annualised": float(sr0_pp * np.sqrt(annualisation)),
        "dsr_z": float(dsr_z),
        "p_dsr": p_dsr,
        "skew": skew,
        "kurtosis": kurt_raw,
        "n_obs": T,
        "n_trials": int(n_trials),
        "valid": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PBO via CSCV (Bailey, Borwein, de Prado, Zhu 2014)
# ─────────────────────────────────────────────────────────────────────────────
def cscv_pbo(M: np.ndarray, *, S: int = 16) -> dict[str, Any]:
    """
    Probability of Backtest Overfitting via Combinatorial Symmetric CV.

    Algorithm (Bailey, Borwein, de Prado, Zhu 2014)
    ────────────────────────────────────────────────
    Input:    M ∈ ℝ^(T × N)   per-period performance matrix
              (rows = T sub-periods, columns = N model configurations)
              S    even number of equal-time partitions (default 16)

    For every C(S, S/2) split of partitions into IS (S/2) and OOS (S/2):
        1.  IS_perf  = mean over IS-rows of M  →  vector of length N
        2.  OOS_perf = mean over OOS-rows of M →  vector of length N
        3.  n*    = argmax(IS_perf)                  (best IS model)
        4.  r̄_n* = relative rank of OOS_perf[n*]
                    = (rank of OOS_perf[n*] in OOS_perf, 1-indexed) / (N + 1)
        5.  λ    = logit(r̄_n*) = log(r̄_n* / (1 − r̄_n*))

    PBO = Pr(λ ≤ 0) = empirical fraction of splits in which the IS-best
    configuration ranks at or below the OOS median.

    A high PBO ⇒ in-sample winners are NOT systematically out-of-sample winners,
    i.e., selection itself is overfit.  Threshold 10% is the AFML book default.

    Args:
        M: per-period × per-config performance matrix.
        S: number of partitions (even, ≥ 2).

    Returns:
        dict with pbo, n_combinations, lambda_mean, lambda_std, lambda_q05/50/95,
        n_periods, n_configs, valid.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"cscv_pbo expects 2-D matrix, got shape {M.shape}")
    T, N = M.shape

    if N < 2:
        return {
            "pbo": float("nan"),
            "n_combinations": 0,
            "n_periods": T,
            "n_configs": N,
            "valid": False,
            "warning": "PBO requires N ≥ 2 model configurations",
        }
    if S < 2 or S % 2 != 0:
        raise ValueError(f"S must be even and ≥ 2, got {S}")
    if T < S:
        # Auto-downgrade to the largest even S ≤ T.
        original_S = S
        S = max(2, (T // 2) * 2)
        LOGGER.warning(
            "cscv_pbo: T=%d < requested S=%d. Auto-downgrading to S=%d.",
            T, original_S, S,
        )

    partitions = np.array_split(np.arange(T), S)
    half = S // 2
    lambdas: list[float] = []

    for is_parts in combinations(range(S), half):
        oos_parts = tuple(s for s in range(S) if s not in is_parts)
        is_rows = np.concatenate([partitions[s] for s in is_parts])
        oos_rows = np.concatenate([partitions[s] for s in oos_parts])

        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)

        n_star = int(np.argmax(is_perf))

        # OOS rank of n_star in oos_perf (1-indexed; higher = better).
        # average ties to avoid pathological logit jumps.
        ranks = pd.Series(oos_perf).rank(method="average").to_numpy()
        rank_n_star = float(ranks[n_star])
        relative_rank = rank_n_star / (N + 1.0)

        # Avoid logit divergence at exact 0 or 1.
        relative_rank = float(np.clip(relative_rank, 1e-6, 1.0 - 1e-6))
        lambdas.append(float(np.log(relative_rank / (1.0 - relative_rank))))

    lambdas_arr = np.array(lambdas, dtype=np.float64)
    pbo = float((lambdas_arr <= 0).mean())

    return {
        "pbo": pbo,
        "n_combinations": int(lambdas_arr.size),
        "n_periods": T,
        "n_configs": N,
        "lambda_mean": float(lambdas_arr.mean()),
        "lambda_std": float(lambdas_arr.std(ddof=1)) if lambdas_arr.size > 1 else 0.0,
        "lambda_q05": float(np.quantile(lambdas_arr, 0.05)),
        "lambda_q50": float(np.quantile(lambdas_arr, 0.50)),
        "lambda_q95": float(np.quantile(lambdas_arr, 0.95)),
        "valid": True,
    }
