"""
src/models/macro_risk_hmm.py — Quant Engine V2.0, Macro Risk Oracle

Unsupervised 2-state Gaussian HMM over a market-proxy return series, used to
SOFT-SCALE portfolio exposure by P(Bull) instead of a hard cash kill-switch.

Why a soft HMM overlay (vs. the old hard `min_bull_prob` threshold)
──────────────────────────────────────────────────────────────────
A hard threshold creates a non-differentiable risk cliff: one basis point of
signal flips the whole book from 100% invested to 100% cash. The HMM instead
emits a continuous P(Bull) ∈ [0, 1]; multiplying target weights by it scales
exposure smoothly (P(Bull)=0.2 → 20% invested / 80% cash). Regimes are learned
UNSUPERVISED from the data — no hand-set thresholds.

Look-ahead discipline (two distinct leaks, both handled)
─────────────────────────────────────────────────────────
  1. PARAMETER leak — fitting the HMM on OOS data. Avoided: `train_macro_risk_hmm`
     is fit STRICTLY on the in-sample (train) split.
  2. INFERENCE leak — `predict_proba` runs forward-BACKWARD (smoothing), so the
     posterior at time t peeks at future observations. Avoided by the
     `filtered=True` default: the smoothed posterior at the LAST timestep of a
     sequence has no future to peek at, so `predict_proba(X[:t+1])[-1]` equals
     the FILTERED (forward-only) estimate at t. An expanding window yields a
     leak-free per-day series. `filtered=False` returns the (leaky) smoothed
     series — diagnostics only, never for the OOS backtest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

LOGGER = logging.getLogger("models.macro_risk_hmm")

__all__ = [
    "MacroRiskHMM",
    "build_market_proxy_returns",
    "build_regime_observation",
    "train_macro_risk_hmm",
]

# Macro return columns (from src/data/macro_crawler.py → data/macro_daily.parquet)
# appended as extra HMM emission dims when macro is enabled.
_MACRO_RET_COLS: tuple[str, ...] = ("sp500_ret", "dxy_ret", "usdvnd_ret")


def build_market_proxy_returns(
    panel: pl.DataFrame | pd.DataFrame,
    *,
    close_col: str = "close",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.Series:
    """
    Cross-sectional mean of daily simple returns across the active universe.

    A breadth-style market proxy (equal-weighted), date-indexed. This is the
    1-D observation series the HMM regimes are inferred from.
    """
    if isinstance(panel, pl.DataFrame):
        df = panel.select([ticker_col, date_col, close_col]).to_pandas()
    else:
        df = panel[[ticker_col, date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])
    df["_r"] = df.groupby(ticker_col, sort=False)[close_col].pct_change()
    mret = df.groupby(date_col)["_r"].mean().sort_index()
    return mret.replace([np.inf, -np.inf], np.nan).dropna().rename("market_ret")


def build_regime_observation(
    panel: pl.DataFrame | pd.DataFrame,
    *,
    use_macro: bool = False,
    macro_parquet: object = None,
    close_col: str = "close",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> "pd.Series | pd.DataFrame":
    """The HMM observation: price-breadth proxy, optionally widened with macro dims.

    Returns the 1-D ``market_ret`` Series (identical to ``build_market_proxy_returns``)
    when ``use_macro`` is off or the macro parquet is unavailable — so the legacy
    price-only regime is the exact, unchanged fallback. When macro is on AND the
    parquet exists, returns a DataFrame whose FIRST column is ``market_ret`` followed
    by the macro return dims (``sp500_ret``/``dxy_ret``/``usdvnd_ret``).

    Leak discipline: macro is left-joined AS-OF each proxy date (``reindex`` on the
    proxy index, no future rows), small calendar gaps forward-filled ≤3 days. Rows
    where any dim is still NaN (warm-up) are dropped so the emission matrix is dense.
    """
    market_ret = build_market_proxy_returns(
        panel, close_col=close_col, ticker_col=ticker_col, date_col=date_col
    )
    if not use_macro or macro_parquet is None:
        return market_ret

    from pathlib import Path  # noqa: PLC0415 — local; keeps module import light

    path = Path(str(macro_parquet))
    if not path.exists():
        LOGGER.warning("[macro-hmm] %s absent — price-only proxy.", path)
        return market_ret
    try:
        macro = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — unreadable parquet → price-only
        LOGGER.warning("[macro-hmm] could not read %s (%s) — price-only.", path, exc)
        return market_ret

    cols = [c for c in _MACRO_RET_COLS if c in macro.columns]
    if "date" not in macro.columns or not cols:
        LOGGER.warning("[macro-hmm] %s missing date/macro-ret cols — price-only.", path)
        return market_ret

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()
    obs = pd.DataFrame({"market_ret": market_ret})
    aligned = macro[cols].reindex(obs.index).ffill(limit=3)  # as-of, gap-bridge only
    for c in cols:
        obs[c] = aligned[c]
    obs = obs.dropna()
    if obs.shape[1] < 2 or obs.empty:
        LOGGER.warning("[macro-hmm] macro join yielded no usable dims — price-only.")
        return market_ret
    LOGGER.info("[macro-hmm] regime observation widened to %d dims: %s",
                obs.shape[1], list(obs.columns))
    return obs


@dataclass
class MacroRiskHMM:
    """A fitted 2-state Gaussian HMM + the identified Bull-state index."""
    model: object                       # hmmlearn GaussianHMM (picklable)
    bull_state: int
    scale: float                        # returns are fit/inferred in `scale`×units
    state_means: list                   # per-state mean return (raw units, diag)
    state_vars: list                    # per-state variance (raw units, diag)
    train_end: str | None = None        # ISO date of the last in-sample bar

    def _X(self, obs: "pd.Series | pd.DataFrame") -> np.ndarray:
        arr = np.asarray(obs.to_numpy(dtype=np.float64))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr * self.scale

    # ── internal: leak-free filtered posterior at the final timestep ─────────
    def _filtered_last(self, Xs: np.ndarray) -> float:
        # Smoothed posterior at the LAST step == filtered (no future) → leak-free.
        return float(self.model.predict_proba(Xs)[-1, self.bull_state])

    def p_bull_series(
        self,
        obs: "pd.Series | pd.DataFrame",
        *,
        filtered: bool = True,
        min_obs: int = 20,
    ) -> pd.Series:
        """
        P(Bull) for each day, date-indexed.

        `obs` is the regime observation — a 1-D market-return Series, or the N-D
        DataFrame from `build_regime_observation` (must match the dimensionality
        the HMM was trained on). filtered=True → expanding-window forward filter
        (leak-free; use for OOS). filtered=False → smoothed (leaky; diagnostics).
        """
        r = obs.dropna()
        idx = r.index
        if len(r) == 0:
            return pd.Series(dtype=np.float64, name="p_bull")
        Xs = self._X(r)

        if not filtered:
            post = self.model.predict_proba(Xs)[:, self.bull_state]
            return pd.Series(post, index=idx, name="p_bull")

        out = np.full(len(r), np.nan, dtype=np.float64)
        for t in range(min_obs - 1, len(r)):
            out[t] = self._filtered_last(Xs[: t + 1])
        # Warm-up bars (< min_obs) → neutral 0.5 then forward-fill.
        return pd.Series(out, index=idx, name="p_bull").ffill().fillna(0.5)

    def p_bull_latest(self, obs: "pd.Series | pd.DataFrame") -> float:
        """Leak-free filtered P(Bull) for the most recent bar (live inference)."""
        r = obs.dropna()
        if len(r) == 0:
            return 0.5
        return self._filtered_last(self._X(r))


def train_macro_risk_hmm(
    returns_train: pd.Series,
    *,
    n_states: int = 2,
    seed: int = 42,
    n_iter: int = 300,
    n_restarts: int = 12,
    scale: float = 100.0,
) -> MacroRiskHMM:
    """
    Fit a Gaussian HMM on IN-SAMPLE market returns and identify the Bull state.

    Bull = the state with the HIGHER mean return; ties broken by LOWER variance.
    Fit is strictly on `returns_train` (the temporal train split) so the regime
    parameters never see out-of-sample data.

    Robustness (HMM EM is init-sensitive on tiny-variance return data):
      • returns are scaled to PERCENT (×100) for numerical conditioning so the
        emission variances are O(1) rather than O(1e-4);
      • `n_restarts` random initialisations are fit, DEGENERATE solutions
        (an unused state whose variance blows up to the covar prior) are
        rejected, and the highest log-likelihood survivor is kept.

    Requires `hmmlearn` (pip install hmmlearn).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover
        raise ImportError("hmmlearn is required for the Macro Risk Oracle: "
                          "pip install hmmlearn") from exc

    r = returns_train.dropna()
    if len(r) < max(2 * n_states, 30):
        raise ValueError(f"train returns too short for HMM: {len(r)} obs")
    Xs = np.asarray(r.to_numpy(dtype=np.float64))
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)
    Xs = Xs * scale
    n_dims = Xs.shape[1]
    data_var = float(np.var(Xs))
    degen_ceiling = 50.0 * data_var          # an unused state's var >> data var

    best_model = None
    best_ll = -np.inf
    for k in range(n_restarts):
        m = GaussianHMM(
            n_components=n_states, covariance_type="diag",
            n_iter=n_iter, tol=1e-4, random_state=seed + k, min_covar=1e-3,
        )
        try:
            m.fit(Xs)
            ll = float(m.score(Xs))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("HMM restart %d failed: %s", k, exc)
            continue
        if not np.isfinite(ll):
            continue
        if float(np.max(m.covars_.ravel())) > degen_ceiling:
            continue                          # degenerate (unused state) → reject
        if ll > best_ll:
            best_ll, best_model = ll, m

    if best_model is None:
        raise RuntimeError(
            "MacroRiskHMM: all restarts produced degenerate/failed fits — "
            "the train split may lack regime diversity.")

    model = best_model
    means_scaled = np.asarray(model.means_)                  # (n_states, n_dims)
    covars = np.asarray(model.covars_)
    if covars.ndim == 3:                                     # full matrices → diagonal
        vars_scaled = np.diagonal(covars, axis1=1, axis2=2)  # (n_states, n_dims)
    elif covars.ndim == 2:                                   # diag variances
        vars_scaled = covars
    else:                                                    # 1-D edge
        vars_scaled = covars.reshape(n_states, -1)
    # Bull = the state with the HIGHER market-breadth return (column 0); ties →
    # lower variance on that same dim. Macro dims (columns 1..) only sharpen the
    # regime split — they never redefine which state is "Bull".
    market_means = means_scaled[:, 0]
    market_vars = vars_scaled[:, 0]
    order = np.lexsort((market_vars, -market_means))
    bull_state = int(order[0])

    # Report diagnostics in RAW return units (nested per-state lists for N-D).
    means_raw = (means_scaled / scale).tolist()
    vars_raw = (vars_scaled / (scale ** 2)).tolist()
    LOGGER.info(
        "MacroRiskHMM fit | dims=%d states=%d  market_means(raw)=%s  bull_state=%d  "
        "train_obs=%d  logL=%.1f",
        n_dims, n_states, np.round(market_means / scale, 6).tolist(),
        bull_state, len(r), best_ll,
    )
    return MacroRiskHMM(
        model=model, bull_state=bull_state, scale=scale,
        state_means=means_raw, state_vars=vars_raw,
        train_end=str(pd.Timestamp(r.index.max()).date()),
    )
