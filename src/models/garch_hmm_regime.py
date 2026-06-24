"""
src/models/garch_hmm_regime.py — GARCH(1,1) + Multi-D Gaussian HMM Regime Overlay

Combines GARCH conditional-volatility extraction with a multi-dimensional
HMM to produce a dynamic P(Bull) exposure brake.

Pipeline
────────
  1. Fit GARCH(1,1) on market return → extract σ_t (conditional volatility).
  2. Form 5-D emission matrix: [market_ret, sp500_ret, dxy_ret, usdvnd_ret,
     log(σ_t)].  The log-transform is critical: raw σ_t is always-positive and
     right-skewed (especially under IGARCH persistence≈1 where it random-walks);
     log(σ_t) is approximately Gaussian → satisfies the GaussianHMM emission
     assumption.  Winsorize at 99th percentile in log-space as safety net.
  3. Per-column z-score standardize (heterogeneous scales).
  4. Fit N-state GaussianHMM on the standardized 5-D matrix.
  5. Identify Bull state (highest market-return mean, lowest variance tiebreak).
  6. Expose filtered P(Bull) series (expanding-window, leak-free).

Look-ahead discipline
─────────────────────
  - GARCH parameters fit strictly on train split (no OOS contamination).
  - At inference time, GARCH σ_t at bar t uses only returns ≤ t (GARCH is
    inherently causal: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}).
  - Z-score params (mean/std per column) are frozen from the TRAIN split —
    inference applies the same transform, no OOS leakage.
  - HMM inference uses expanding-window filtered posterior (smoothed posterior
    at the last timestep of X[:t+1] ≡ forward-only estimate at t).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("models.garch_hmm_regime")

__all__ = [
    "GarchHmmRegime",
    "train_garch_hmm",
]

# Column ordering for the 5-D emission matrix.
# The 5th dim is log(σ_t), not raw σ_t — see module docstring for rationale.
_EMISSION_COLS: tuple[str, ...] = (
    "market_ret",
    "sp500_ret",
    "dxy_ret",
    "usdvnd_ret",
    "log_garch_vol",
)

# Winsorize percentile for log(σ_t) — caps the IGARCH explosive tail without
# creating the flat density spike that raw clipping would introduce.
_VOL_WINSOR_PCTL: float = 99.0


@dataclass
class GarchHmmRegime:
    """Fitted GARCH(1,1) + N-state Gaussian HMM regime model."""

    hmm_model: object          # hmmlearn GaussianHMM (picklable)
    garch_params: dict         # {omega, alpha, beta, mu} for reconstructing σ_t
    bull_state: int            # index of the identified Bull regime
    n_states: int
    scale: float               # legacy (kept for compat); actual conditioning via z-score
    state_means: list          # per-state mean vector (raw units, de-standardized)
    state_covars: list         # per-state covariance diagonal (raw units)
    emission_mean: np.ndarray  # per-column mean from train (5,)
    emission_std: np.ndarray   # per-column std from train (5,)
    log_vol_cap: float = np.inf  # winsorize cap for log(σ_t), from train p99
    emission_cols: list = field(default_factory=lambda: list(_EMISSION_COLS))
    train_end: str | None = None

    # ── GARCH conditional volatility ─────────────────────────────────────

    def _garch_vol(self, returns: np.ndarray) -> np.ndarray:
        """Reconstruct GARCH(1,1) conditional volatility from stored params.

        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

        Causal by construction: σ_t depends only on returns up to t-1.
        """
        omega = self.garch_params["omega"]
        alpha = self.garch_params["alpha"]
        beta = self.garch_params["beta"]
        mu = self.garch_params["mu"]

        T = len(returns)
        sigma2 = np.empty(T, dtype=np.float64)
        persistence = alpha + beta
        if persistence < 1.0:
            sigma2_unc = omega / (1.0 - persistence)
        else:
            sigma2_unc = np.var(returns)
        sigma2[0] = sigma2_unc

        eps = returns - mu
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-12)

        return np.sqrt(sigma2)

    # ── Build 5-D emission matrix (log-vol + z-score standardized) ─────

    def _build_emission(self, obs: pd.DataFrame) -> np.ndarray:
        """Build the z-score standardized emission matrix from a macro DataFrame.

        The vol dimension uses log(σ_t), winsorized at the train-frozen cap,
        so the GaussianHMM emission assumption holds even under IGARCH drift.
        """
        market_ret = obs["market_ret"].to_numpy(dtype=np.float64)
        sigma_t = self._garch_vol(market_ret)
        log_vol = np.log(np.maximum(sigma_t, 1e-12))
        log_vol = np.minimum(log_vol, self.log_vol_cap)

        cols = []
        for c in _EMISSION_COLS:
            if c == "log_garch_vol":
                cols.append(log_vol)
            else:
                cols.append(obs[c].to_numpy(dtype=np.float64))

        X_raw = np.column_stack(cols)
        return (X_raw - self.emission_mean) / self.emission_std

    # ── Filtered posterior (leak-free) ───────────────────────────────────

    def _filtered_last(self, Xs: np.ndarray) -> float:
        """Smoothed posterior at the last timestep == filtered (no future)."""
        return float(self.hmm_model.predict_proba(Xs)[-1, self.bull_state])

    def p_bull_series(
        self,
        obs: pd.DataFrame,
        *,
        filtered: bool = True,
        min_obs: int = 30,
    ) -> pd.Series:
        """P(Bull) for each day, date-indexed.

        filtered=True → expanding-window forward filter (leak-free, use for OOS).
        filtered=False → smoothed (leaky, diagnostics only).
        """
        required = set(_EMISSION_COLS) - {"log_garch_vol"}
        missing = required - set(obs.columns)
        if missing:
            raise ValueError(f"obs missing columns: {missing}")

        obs = obs.dropna(subset=list(required))
        if obs.empty:
            return pd.Series(dtype=np.float64, name="p_bull")

        idx = obs.index
        Xs = self._build_emission(obs)

        if not filtered:
            post = self.hmm_model.predict_proba(Xs)[:, self.bull_state]
            return pd.Series(post, index=idx, name="p_bull")

        out = np.full(len(obs), np.nan, dtype=np.float64)
        for t in range(min_obs - 1, len(obs)):
            out[t] = self._filtered_last(Xs[: t + 1])

        return pd.Series(out, index=idx, name="p_bull").ffill().fillna(0.5)

    def p_bull_latest(self, obs: pd.DataFrame) -> float:
        """Leak-free filtered P(Bull) for the most recent bar."""
        required = set(_EMISSION_COLS) - {"log_garch_vol"}
        obs = obs.dropna(subset=list(required))
        if obs.empty:
            return 0.5
        Xs = self._build_emission(obs)
        return self._filtered_last(Xs)

    def regime_labels(self, obs: pd.DataFrame) -> pd.Series:
        """Most-likely regime index per day (Viterbi decode)."""
        required = set(_EMISSION_COLS) - {"log_garch_vol"}
        obs = obs.dropna(subset=list(required))
        if obs.empty:
            return pd.Series(dtype=np.int64, name="regime")
        Xs = self._build_emission(obs)
        labels = self.hmm_model.predict(Xs)
        return pd.Series(labels, index=obs.index, name="regime")

    def exposure_brake(
        self,
        obs: pd.DataFrame,
        *,
        threshold: float = 0.5,
        filtered: bool = True,
    ) -> pd.Series:
        """Binary exposure signal: 1.0 if P(Bull) >= threshold, else 0.0.

        Use as a multiplicative gate on portfolio weights:
            final_weight = raw_weight × exposure_brake
        """
        p = self.p_bull_series(obs, filtered=filtered)
        return (p >= threshold).astype(np.float64).rename("exposure")


def train_garch_hmm(
    obs: pd.DataFrame,
    *,
    n_states: int = 3,
    seed: int = 42,
    garch_p: int = 1,
    garch_q: int = 1,
    n_iter: int = 300,
    n_restarts: int = 20,
    scale: float = 100.0,
) -> GarchHmmRegime:
    """Fit GARCH(1,1) on market returns, then N-state HMM on the 5-D emission.

    Parameters
    ----------
    obs : DataFrame with columns [market_ret, sp500_ret, dxy_ret, usdvnd_ret],
          date-indexed, strictly the TRAIN split.
    n_states : number of HMM hidden states (3 or 4 recommended).
    scale : legacy param (kept for API compat); actual conditioning uses
            per-column z-score standardization.

    Returns
    -------
    GarchHmmRegime — fitted model ready for inference.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "arch package required: pip install arch"
        ) from exc
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn required: pip install hmmlearn"
        ) from exc

    required = {"market_ret", "sp500_ret", "dxy_ret", "usdvnd_ret"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"obs missing columns: {missing}")

    obs = obs.dropna(subset=list(required)).sort_index()
    if len(obs) < max(2 * n_states, 50):
        raise ValueError(f"train obs too short for GARCH+HMM: {len(obs)}")

    # ── Step 1: GARCH(1,1) on market returns ─────────────────────────────
    market_ret = obs["market_ret"].to_numpy(dtype=np.float64)
    am = arch_model(
        market_ret * 100,
        vol="Garch",
        p=garch_p,
        q=garch_q,
        mean="Constant",
        dist="Normal",
    )
    res = am.fit(disp="off", show_warning=False)

    garch_params = {
        "omega": float(res.params.get("omega", 0.0)),
        "alpha": float(res.params.get("alpha[1]", 0.0)),
        "beta": float(res.params.get("beta[1]", 0.0)),
        "mu": float(res.params.get("mu", 0.0)) / 100.0,
    }
    garch_params["omega"] /= 100.0 ** 2

    persistence = garch_params["alpha"] + garch_params["beta"]
    LOGGER.info(
        "GARCH(1,1) fit | ω=%.2e  α=%.4f  β=%.4f  μ=%.6f  persistence=%.4f",
        garch_params["omega"],
        garch_params["alpha"],
        garch_params["beta"],
        garch_params["mu"],
        persistence,
    )

    # ── Step 2: Build 5-D raw emission matrix ────────────────────────────
    # log(σ_t) instead of raw σ_t: GaussianHMM assumes Gaussian emissions;
    # raw vol is always-positive / right-skewed (especially under IGARCH where
    # it random-walks). log-transform makes it approximately symmetric.
    sigma_t = _reconstruct_garch_vol(market_ret, garch_params)
    log_vol = np.log(np.maximum(sigma_t, 1e-12))

    # Winsorize at p99 in log-space: caps the explosive IGARCH tail without
    # creating a flat density spike (unlike raw clipping which concentrates
    # mass at the cap → artificial HMM regime).
    log_vol_cap = float(np.percentile(log_vol, _VOL_WINSOR_PCTL))
    log_vol = np.minimum(log_vol, log_vol_cap)

    X_raw = np.column_stack([
        market_ret,
        obs["sp500_ret"].to_numpy(dtype=np.float64),
        obs["dxy_ret"].to_numpy(dtype=np.float64),
        obs["usdvnd_ret"].to_numpy(dtype=np.float64),
        log_vol,
    ])

    LOGGER.info(
        "Emission raw | vol: raw_range=[%.6f, %.6f]  log_cap=%.4f  "
        "log_range=[%.4f, %.4f]",
        float(sigma_t.min()), float(sigma_t.max()), log_vol_cap,
        float(log_vol.min()), float(log_vol.max()),
    )

    # ── Step 2b: Per-column z-score standardization ──────────────────────
    emission_mean = X_raw.mean(axis=0)
    emission_std = X_raw.std(axis=0)
    emission_std = np.where(emission_std < 1e-12, 1.0, emission_std)
    Xs = (X_raw - emission_mean) / emission_std

    LOGGER.info(
        "Emission z-score | means=%s  stds=%s",
        np.round(emission_mean, 6).tolist(),
        np.round(emission_std, 6).tolist(),
    )

    # ── Step 3: Fit N-state GaussianHMM on standardized 5-D emissions ───
    # Degenerate filter: per-dimension ceiling (not global var which is
    # meaningless after standardization — each dim has unit variance).
    degen_ceiling = 100.0  # z-scored dims have var ≈ 1; 100× is generous

    best_model = None
    best_ll = -np.inf
    n_accepted = 0
    for k in range(n_restarts):
        m = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-4,
            random_state=seed + k,
            min_covar=1e-3,
        )
        try:
            m.fit(Xs)
            ll = float(m.score(Xs))
        except Exception:
            LOGGER.debug("HMM restart %d failed", k)
            continue
        if not np.isfinite(ll):
            continue
        max_var = float(np.max(m.covars_.ravel()))
        if max_var > degen_ceiling:
            LOGGER.debug("HMM restart %d rejected (max_var=%.1f > %.1f)", k, max_var, degen_ceiling)
            continue
        n_accepted += 1
        if ll > best_ll:
            best_ll, best_model = ll, m

    if best_model is None:
        raise RuntimeError(
            f"GarchHmmRegime: all {n_restarts} restarts degenerate — "
            f"train split may lack regime diversity. "
            f"(accepted={n_accepted}, degen_ceiling={degen_ceiling})"
        )

    LOGGER.info("HMM fit | accepted %d/%d restarts, best logL=%.1f", n_accepted, n_restarts, best_ll)

    model = best_model
    means_z = np.asarray(model.means_)            # (n_states, 5) in z-score space
    covars = np.asarray(model.covars_)
    if covars.ndim == 3:
        vars_z = np.diagonal(covars, axis1=1, axis2=2)
    elif covars.ndim == 2:
        vars_z = covars
    else:
        vars_z = covars.reshape(n_states, -1)

    # ── Step 4: Identify Bull state ──────────────────────────────────────
    # De-standardize means for dim 0 (market_ret) to identify Bull by
    # highest raw mean return, tiebreak by lowest raw variance.
    market_means_raw = means_z[:, 0] * emission_std[0] + emission_mean[0]
    market_vars_raw = vars_z[:, 0] * emission_std[0] ** 2
    order = np.lexsort((market_vars_raw, -market_means_raw))
    bull_state = int(order[0])

    # De-standardize all dims for diagnostics storage.
    means_raw = (means_z * emission_std + emission_mean).tolist()
    vars_raw = (vars_z * emission_std ** 2).tolist()

    LOGGER.info(
        "GarchHmmRegime fit | states=%d  bull=%d  market_means(raw)=%s  "
        "train=%d obs  logL=%.1f",
        n_states, bull_state,
        np.round(market_means_raw, 6).tolist(),
        len(obs), best_ll,
    )

    return GarchHmmRegime(
        hmm_model=model,
        garch_params=garch_params,
        bull_state=bull_state,
        n_states=n_states,
        scale=scale,
        state_means=means_raw,
        state_covars=vars_raw,
        emission_mean=emission_mean,
        emission_std=emission_std,
        log_vol_cap=log_vol_cap,
        train_end=str(pd.Timestamp(obs.index.max()).date()),
    )


def _reconstruct_garch_vol(returns: np.ndarray, params: dict) -> np.ndarray:
    """Standalone GARCH(1,1) vol reconstruction (used during training)."""
    omega, alpha, beta, mu = (
        params["omega"], params["alpha"], params["beta"], params["mu"],
    )
    T = len(returns)
    sigma2 = np.empty(T, dtype=np.float64)
    persistence = alpha + beta
    sigma2[0] = omega / (1.0 - persistence) if persistence < 1.0 else np.var(returns)
    eps = returns - mu
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-12)
    return np.sqrt(sigma2)
