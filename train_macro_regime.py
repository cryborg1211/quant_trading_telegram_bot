"""
train_macro_regime.py — Fit the GARCH(1,1) + Multi-D HMM regime overlay.

Pipeline
────────
    1. Load OHLCV panel → build market-breadth proxy returns
    2. Load macro parquet → extract sp500_ret / dxy_ret / usdvnd_ret
    3. Join into a 4-column observation DataFrame (date-indexed)
    4. Chronological train split (default 80%)
    5. Fit GARCH(1,1) + N-state HMM on the TRAIN split
    6. Serialize to models/saved/garch_hmm_v4_weights.joblib
    7. Reload + verify inference on a dummy row

Run
───
    python train_macro_regime.py
    python train_macro_regime.py --n-states 4 --train-frac 0.75
    python train_macro_regime.py --out models/saved/custom.joblib
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.backtest.pipeline import RunConfig, configure_logging, load_ohlcv
from src.models.garch_hmm_regime import GarchHmmRegime, train_garch_hmm
from src.models.macro_risk_hmm import build_market_proxy_returns

LOGGER = logging.getLogger("quant.train_macro_regime")

_DEFAULT_OUT = Path("models/saved/garch_hmm_v4_weights.joblib")
_DEFAULT_MACRO_PARQUET = Path("data/macro_daily.parquet")


def _build_obs(
    cfg: RunConfig,
    macro_parquet: Path,
) -> pd.DataFrame:
    """Build the 4-column observation DataFrame: market_ret + 3 macro returns.

    market_ret = cross-sectional mean of daily simple returns across the OHLCV
    panel (same breadth proxy the existing MacroRiskHMM uses).
    """
    LOGGER.info("Loading OHLCV panel for market-breadth proxy...")
    panel = load_ohlcv(cfg)
    market_ret = build_market_proxy_returns(panel)
    LOGGER.info("Market proxy: %d days (%s .. %s)",
                len(market_ret), market_ret.index.min().date(), market_ret.index.max().date())

    if not macro_parquet.exists():
        raise FileNotFoundError(
            f"Macro parquet not found: {macro_parquet}. "
            f"Run 'python main.py --task crawl_macro' first."
        )
    macro = pd.read_parquet(macro_parquet)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    required_macro = ["sp500_ret", "dxy_ret", "usdvnd_ret"]
    missing = [c for c in required_macro if c not in macro.columns]
    if missing:
        raise ValueError(f"Macro parquet missing columns: {missing}")

    obs = pd.DataFrame({"market_ret": market_ret})
    aligned = macro[required_macro].reindex(obs.index).ffill(limit=3)
    for c in required_macro:
        obs[c] = aligned[c]
    obs = obs.dropna()

    LOGGER.info("Observation matrix: %d rows × %d cols (%s .. %s)",
                len(obs), obs.shape[1], obs.index.min().date(), obs.index.max().date())
    return obs


def _verify(path: Path) -> None:
    """Reload the saved model and run inference on a dummy row."""
    LOGGER.info("Verification: reloading %s ...", path)
    loaded: GarchHmmRegime = joblib.load(path)

    n_cols = len(loaded.emission_cols) - 1  # minus garch_vol (reconstructed)
    dummy = pd.DataFrame(
        np.random.RandomState(0).randn(60, n_cols) * 0.01,
        columns=["market_ret", "sp500_ret", "dxy_ret", "usdvnd_ret"],
        index=pd.bdate_range("2025-01-01", periods=60),
    )

    p = loaded.p_bull_latest(dummy)
    brake = loaded.exposure_brake(dummy, threshold=0.5)
    labels = loaded.regime_labels(dummy)

    LOGGER.info("  p_bull_latest = %.4f", p)
    LOGGER.info("  exposure_brake mean = %.2f", brake.mean())
    LOGGER.info("  regime distribution = %s", dict(labels.value_counts().sort_index()))
    LOGGER.info("Verification PASSED — model is functional.")


def main(
    cfg: RunConfig,
    *,
    macro_parquet: Path = _DEFAULT_MACRO_PARQUET,
    n_states: int = 3,
    n_restarts: int = 12,
    train_frac: float = 0.80,
    out_path: Path = _DEFAULT_OUT,
) -> GarchHmmRegime:
    configure_logging()
    t0 = time.perf_counter()

    LOGGER.info("=" * 72)
    LOGGER.info(" TRAIN_MACRO_REGIME | states=%d  restarts=%d  train_frac=%.0f%%",
                n_states, n_restarts, train_frac * 100)
    LOGGER.info("=" * 72)

    # ── 1-3. Build observation matrix ────────────────────────────────────
    obs = _build_obs(cfg, macro_parquet)

    # ── 4. Chronological train split ─────────────────────────────────────
    split_idx = int(len(obs) * train_frac)
    obs_train = obs.iloc[:split_idx]
    obs_oos = obs.iloc[split_idx:]
    cutoff = obs_train.index.max()
    LOGGER.info("Train/OOS split at %s: train=%d  OOS=%d",
                cutoff.date(), len(obs_train), len(obs_oos))

    # ── 5. Fit GARCH + HMM ──────────────────────────────────────────────
    model = train_garch_hmm(
        obs_train,
        n_states=n_states,
        seed=cfg.seed,
        n_restarts=n_restarts,
    )

    # OOS diagnostics
    if len(obs_oos) > 0:
        p_oos = model.p_bull_series(obs, filtered=True)
        oos_mask = p_oos.index > cutoff
        if oos_mask.any():
            oos_p = p_oos[oos_mask]
            LOGGER.info("OOS P(Bull): mean=%.3f  min=%.3f  max=%.3f  std=%.3f",
                        oos_p.mean(), oos_p.min(), oos_p.max(), oos_p.std())

    # ── 6. Serialize ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path, compress=3)
    size_kb = out_path.stat().st_size / 1024
    LOGGER.info("Model saved → %s  (%.1f KB)", out_path, size_kb)

    # ── 7. Verify ────────────────────────────────────────────────────────
    _verify(out_path)

    LOGGER.info("=" * 72)
    LOGGER.info(" DONE | wall=%.1fs  states=%d  bull=%d  persistence=%.4f",
                time.perf_counter() - t0, model.n_states, model.bull_state,
                model.garch_params["alpha"] + model.garch_params["beta"])
    LOGGER.info("=" * 72)
    return model


def _cli() -> dict:
    p = argparse.ArgumentParser(
        description="Fit GARCH(1,1) + Multi-D HMM regime overlay and save weights.")
    p.add_argument("--n-states", type=int, default=3,
                   help="HMM hidden states (default: 3)")
    p.add_argument("--n-restarts", type=int, default=12,
                   help="HMM random restarts (default: 12)")
    p.add_argument("--train-frac", type=float, default=0.80,
                   help="Chronological train fraction (default: 0.80)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--macro-parquet", type=Path, default=_DEFAULT_MACRO_PARQUET)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help="Output path (default: models/saved/garch_hmm_v4_weights.joblib)")
    a = p.parse_args()

    cfg = RunConfig()
    cfg.seed = a.seed
    return {
        "cfg": cfg,
        "macro_parquet": a.macro_parquet,
        "n_states": a.n_states,
        "n_restarts": a.n_restarts,
        "train_frac": a.train_frac,
        "out_path": a.out,
    }


if __name__ == "__main__":
    try:
        main(**_cli())
    except Exception as exc:
        LOGGER.error("FATAL: %s", exc, exc_info=True)
        sys.exit(1)
