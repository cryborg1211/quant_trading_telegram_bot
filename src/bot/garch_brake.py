"""
src/bot/garch_brake.py — live GARCH-HMM macro exposure brake (serve path).

Computes a single market-wide exposure multiplier ∈ [floor, 1.0] from the
fitted GARCH(1,1)+HMM overlay (``models/saved/garch_hmm_v4_weights.joblib``),
applied to every MUA weight in ``main._dispatch_signals``.

Benchmark (seed 0, T+5, bear OOS): regime_policy + this brake was the best
defense (Sharpe −0.36 → +0.005, timing_α +0.37). The two layers are
complementary — price micro-regime × macro breadth — so they STACK.

FAIL-OPEN CONTRACT
──────────────────
This runs on the daily live cron. ANY failure (missing/stale macro parquet,
incompatible model pickle, empty OHLCV, import error) returns 1.0 — full
exposure, no brake — and logs a warning. The brake can only ever REDUCE
exposure when everything is healthy; it can never break serve.

Leak discipline is inherited: GARCH is causal, the HMM posterior is filtered
(``p_bull_latest`` reads the leak-free last-bar estimate).
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import CONFIG

LOGGER = logging.getLogger("bot.garch_brake")

_WEIGHTS_PATH = Path("models/saved/garch_hmm_v4_weights.joblib")
_MACRO_RET_COLS = ("sp500_ret", "dxy_ret", "usdvnd_ret")

# Module-level model cache (loaded once per process).
_MODEL = None
_MODEL_TRIED = False


def _load_model():
    """Load + cache the fitted GarchHmmRegime. Returns None on any failure."""
    global _MODEL, _MODEL_TRIED
    if _MODEL is not None or _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        import joblib  # noqa: PLC0415

        if not _WEIGHTS_PATH.exists():
            LOGGER.warning("[garch-brake] weights absent: %s", _WEIGHTS_PATH)
            return None
        _MODEL = joblib.load(_WEIGHTS_PATH)
        LOGGER.info("[garch-brake] loaded %s (bull_state=%s)",
                    _WEIGHTS_PATH.name, getattr(_MODEL, "bull_state", "?"))
    except Exception:  # noqa: BLE001 — fail-open
        LOGGER.warning("[garch-brake] model load failed — full exposure", exc_info=True)
        _MODEL = None
    return _MODEL


def _build_live_obs():
    """Assemble the live macro observation frame (market proxy + macro returns).

    Returns a date-indexed DataFrame with columns [market_ret, sp500_ret,
    dxy_ret, usdvnd_ret], or None if anything is unavailable.
    """
    import pandas as pd  # noqa: PLC0415

    from src.backtest.pipeline import RunConfig, load_ohlcv  # noqa: PLC0415
    from src.models.macro_risk_hmm import build_market_proxy_returns  # noqa: PLC0415

    panel = load_ohlcv(RunConfig())
    market_ret = build_market_proxy_returns(panel)
    if market_ret is None or len(market_ret) == 0:
        LOGGER.warning("[garch-brake] empty market proxy — full exposure")
        return None

    macro_path = Path(str(CONFIG.paths.macro_parquet))
    if not macro_path.exists():
        LOGGER.warning("[garch-brake] macro parquet absent: %s — full exposure", macro_path)
        return None

    macro = pd.read_parquet(macro_path)
    if "date" not in macro.columns or not all(c in macro.columns for c in _MACRO_RET_COLS):
        LOGGER.warning("[garch-brake] macro parquet missing cols — full exposure")
        return None
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    obs = pd.DataFrame({"market_ret": market_ret})
    aligned = macro[list(_MACRO_RET_COLS)].reindex(obs.index).ffill(limit=3)
    for c in _MACRO_RET_COLS:
        obs[c] = aligned[c]
    obs = obs.dropna()
    if obs.empty:
        LOGGER.warning("[garch-brake] macro join empty after dropna — full exposure")
        return None
    return obs


def live_exposure_scalar() -> float:
    """Market-wide exposure multiplier ∈ [floor, 1.0] for today's dispatch.

    Returns 1.0 (no brake / full exposure) when disabled OR on ANY failure.
    """
    if not getattr(CONFIG.trading, "garch_brake_enabled", False):
        return 1.0
    try:
        import numpy as np  # noqa: PLC0415

        model = _load_model()
        if model is None:
            return 1.0
        obs = _build_live_obs()
        if obs is None:
            return 1.0
        p_bull = float(model.p_bull_latest(obs))
        floor = float(getattr(CONFIG.trading, "garch_brake_floor", 0.2))
        scalar = float(np.clip(p_bull, floor, 1.0))
        LOGGER.info("[garch-brake] P(Bull)=%.3f → exposure ×%.3f (floor %.2f)",
                    p_bull, scalar, floor)
        return scalar
    except Exception:  # noqa: BLE001 — fail-open: never break the live pipeline
        LOGGER.warning("[garch-brake] scalar computation failed — full exposure",
                       exc_info=True)
        return 1.0
