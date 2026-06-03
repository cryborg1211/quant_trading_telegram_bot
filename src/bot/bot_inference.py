"""
src/bot/bot_inference.py — V3 V1-faithful live-trading inference.

The bot's ONE responsibility: load the GOLDEN-config joblib bundle written by
`run_backtest.py` (the V4.0 Fast Evaluator), take live cross-sectional features
(the tabular `_xsz` columns) for a daily decision bar, and emit a strict
BUY / HOLD signal per ticker.

Train/serve parity is enforced by:
  • loading the EXACT `tabular_features` column order from the bundle
    (the order is load-bearing for the GBM stack);
  • wrapping the live feature matrix as a NAMED pandas DataFrame before passing
    it to `TabularEnsemble.predict_proba` (so LightGBM/XGBoost/CatBoost match
    the `feature_names_in_` recorded at fit time — no warning, no skew);
  • thresholding with the EXACT `up_threshold` chosen by the GOLDEN sweep.

Usage:

    from src.bot.bot_inference import V3BotInference

    bot = V3BotInference.from_artifact("models/saved/v3_ensemble.joblib")
    # `live` is a pandas DataFrame indexed by ticker, columns include the 9 features
    signals = bot.signals(live)
    # → {'HPG': ('BUY', 0.612), 'VNM': ('HOLD', 0.418), ...}

No torch, no heavy ML stack at runtime; joblib pulls the pickled GBMs straight
in via lightgbm / xgboost / catboost / sklearn.  Optional HMM regime overlay
is loaded but not auto-applied — the caller decides when to soft-scale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("bot.inference")

# Importing TabularEnsemble is required so joblib can unpickle it.
from src.models.tabular_ensemble import TabularEnsemble       # noqa: E402

BUY = "BUY"
HOLD = "HOLD"
REQUIRED_BUNDLE_KEYS = ("ensemble", "tabular_features", "up_threshold")


@dataclass
class V3BotInference:
    """
    Production inference wrapper for the V3 V1-faithful TabularEnsemble.

    Construct via `V3BotInference.from_artifact(path)`; never instantiate
    directly in production code — the artifact is the contract.
    """
    ensemble: TabularEnsemble
    tabular_features: list[str]
    up_threshold: float
    signal_threshold: float
    macro_hmm: Any = None
    metadata: dict = field(default_factory=dict)
    schema_version: str = "v3.0"

    # ── loaders ───────────────────────────────────────────────────────────
    @classmethod
    def from_artifact(cls, path: str | Path) -> "V3BotInference":
        """Load the GOLDEN-config bundle written by run_backtest.py."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"V3 artifact not found: {p}")
        bundle = joblib.load(p)
        missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
        if missing:
            raise ValueError(f"Artifact {p} missing required keys: {missing}")

        feats = list(bundle["tabular_features"])
        up_thr = float(bundle["up_threshold"])
        sig_thr = float(bundle.get("signal_threshold", up_thr - 0.05))
        bot = cls(
            ensemble=bundle["ensemble"],
            tabular_features=feats,
            up_threshold=up_thr,
            signal_threshold=sig_thr,
            macro_hmm=bundle.get("macro_hmm"),
            metadata=dict(bundle.get("metadata", {})),
            schema_version=str(bundle.get("schema_version", "v3.0")),
        )
        LOGGER.info(
            "V3BotInference loaded | schema=%s  features=%d  up_threshold=%.2f  "
            "signal_threshold=%.2f  trained_at=%s  oos_sharpe=%+.3f",
            bot.schema_version, len(feats), up_thr, sig_thr,
            bot.metadata.get("trained_at", "?"),
            bot.metadata.get("oos_sharpe", float("nan")),
        )
        return bot

    # ── feature normalisation (train/serve parity) ────────────────────────
    def _as_named_df(self, live: pd.DataFrame | Mapping | np.ndarray,
                     tickers: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Coerce whatever the caller hands us into a (n_tickers, n_features)
        DataFrame with columns in the EXACT order the ensemble was trained on.

        Accepts:
          • pd.DataFrame: columns must be a superset of self.tabular_features;
            extras dropped, missing → ValueError.
          • dict / Mapping: keys = ticker, values = dict of feature → value.
          • np.ndarray (n, n_features): columns assumed to be in
            self.tabular_features order; tickers (optional) become the index.

        This is the bot-side mirror of `_as_named_df` inside `TabularEnsemble`
        — by pre-wrapping here we get a consistent named index for the signal
        dict AND we make the lightgbm/xgboost/catboost feature-name validation
        unambiguous.
        """
        if isinstance(live, pd.DataFrame):
            missing = [c for c in self.tabular_features if c not in live.columns]
            if missing:
                raise ValueError(
                    f"Live features missing columns required by the model: {missing}")
            return live[self.tabular_features].copy()

        if isinstance(live, Mapping):
            # dict of {ticker: {feature: value}}
            df = pd.DataFrame.from_dict(live, orient="index")
            missing = [c for c in self.tabular_features if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Live features missing columns required by the model: {missing}")
            return df[self.tabular_features].copy()

        arr = np.asarray(live, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.tabular_features):
            raise ValueError(
                f"Expected (n, {len(self.tabular_features)}) numpy input; got {arr.shape}")
        idx = list(tickers) if tickers is not None else list(range(arr.shape[0]))
        if len(idx) != arr.shape[0]:
            raise ValueError(f"tickers length {len(idx)} != rows {arr.shape[0]}")
        return pd.DataFrame(arr, index=idx, columns=self.tabular_features)

    # ── public API ────────────────────────────────────────────────────────
    def predict_proba(self, live: pd.DataFrame | Mapping | np.ndarray,
                      tickers: Sequence[str] | None = None) -> pd.Series:
        """
        Return P(UP) per row, indexed by ticker.

        The ensemble itself re-applies the train/serve `_as_named_df` guard
        internally; we do it here too so the index propagates to the caller.
        """
        X = self._as_named_df(live, tickers)
        if len(X) == 0:
            return pd.Series([], dtype=np.float64, name="p_up")
        p_up = np.asarray(self.ensemble.predict_proba(X), dtype=np.float64)
        if not np.all(np.isfinite(p_up)):
            LOGGER.warning("Non-finite P(UP) detected — replacing with 0.0")
            p_up = np.where(np.isfinite(p_up), p_up, 0.0)
        return pd.Series(p_up, index=X.index, name="p_up")

    def predict_proba_3class(self, live: pd.DataFrame | Mapping | np.ndarray,
                             tickers: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Return (n, 3) probability matrix as a DataFrame indexed by ticker,
        columns = ['p_down', 'p_flat', 'p_up'].  Drop-in compatible with the
        legacy V6 stacker's 3-class output — used by `main.py` to swap V3 into
        the existing arbitrator / verify / rebalance flows with NO downstream
        code change.
        """
        X = self._as_named_df(live, tickers)
        if len(X) == 0:
            return pd.DataFrame(columns=["p_down", "p_flat", "p_up"])
        proba = np.asarray(self.ensemble.predict_proba_3class(X), dtype=np.float64)
        # Defensive: replace any non-finite (shouldn't happen post-renormalize)
        proba = np.where(np.isfinite(proba), proba, 0.0)
        return pd.DataFrame(proba, index=X.index, columns=["p_down", "p_flat", "p_up"])

    def signals(self, live: pd.DataFrame | Mapping | np.ndarray,
                tickers: Sequence[str] | None = None) -> dict[str, tuple[str, float]]:
        """
        Strict per-ticker trading signal at the GOLDEN threshold.

        Returns a dict mapping ticker → (signal, p_up) where:
            signal = "BUY"  iff P(UP) >= self.up_threshold
            signal = "HOLD" otherwise
        """
        p_up = self.predict_proba(live, tickers)
        return {
            str(idx): (BUY if float(p) >= self.up_threshold else HOLD, float(p))
            for idx, p in p_up.items()
        }

    def buy_list(self, live: pd.DataFrame | Mapping | np.ndarray,
                 tickers: Sequence[str] | None = None,
                 top_k: int | None = None) -> list[tuple[str, float]]:
        """
        Convenience: return ONLY the BUY signals sorted by descending P(UP).

        `top_k` caps the list size (use this to honour the portfolio's
        `max_positions` constraint — pass cfg.max_positions from training time).
        """
        p_up = self.predict_proba(live, tickers)
        buys = p_up[p_up >= self.up_threshold].sort_values(ascending=False)
        if top_k is not None:
            buys = buys.head(int(top_k))
        return [(str(idx), float(p)) for idx, p in buys.items()]

    # ── informational ─────────────────────────────────────────────────────
    def card(self) -> str:
        """One-paragraph human-readable model card for ops dashboards."""
        m = self.metadata
        return (
            f"V3BotInference[{self.schema_version}]  "
            f"trained_at={m.get('trained_at', '?')}  "
            f"seed={m.get('best_seed', '?')}  "
            f"features={len(self.tabular_features)} ({self.tabular_features})  "
            f"thresholds: up={self.up_threshold:.2f}  signal_gate={self.signal_threshold:.2f}  "
            f"labels: T+{m.get('tb_horizon', '?')} ±{m.get('tb_pt', '?')}σ  "
            f"OOS: Sharpe={m.get('oos_sharpe', float('nan')):+.3f}  "
            f"NetPnL={m.get('oos_net_pnl_vnd', float('nan')):+,.0f} VND  "
            f"DD={m.get('oos_max_dd', float('nan')):+.2%}  "
            f"days={m.get('oos_days', '?')}  "
            f"UP-precision={m.get('golden_mean_up_precision', float('nan')):.4f}"
        )


__all__ = ["V3BotInference", "BUY", "HOLD"]
