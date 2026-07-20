"""
src/bot/garch_brake.py — live market-wide exposure meta-controller (serve path).

``live_exposure_scalar()`` combines THREE independent market-wide legs into
one exposure multiplier ∈ (0, 1.0], applied to every MUA weight in
``main._dispatch_signals``:
  • GARCH-HMM  — macro/vol regime (``models/saved/garch_hmm_v4_weights.joblib``)
  • drift      — trailing index-return slow-bleed (added 19-07-26; the GARCH
                 leg is vol-triggered and read ~0.99 through the whole
                 low-vol July grind-down)
  • breadth    — fraction of the liquid universe with positive trailing
                 returns (added 20-07-26; check_drift.py found July's bleed
                 was a breadth collapse, 29.5% vs 41.5% train — the direct
                 signal none of the price/vol proxies above actually read)

Combined = min(all three). Every call logs the FULL per-leg breakdown plus
which leg is binding — the attribution the July post-mortem was missing
(before this, nobody could tell which layer cost how much on a given day).

Benchmark (seed 0, T+5, bear OOS): regime_policy + the GARCH leg was the
best defense (Sharpe −0.36 → +0.005, timing_α +0.37). regime_policy is
per-name price micro-structure; these three legs are market-wide — they
STACK, not compete.

FAIL-OPEN CONTRACT
──────────────────
This runs on the daily live cron. ANY failure in ANY leg (missing/stale
macro parquet, incompatible model pickle, empty OHLCV, import error) drops
that leg to 1.0 (full exposure, no brake) and logs a warning — the other
legs are unaffected. The controller can only ever REDUCE exposure when
everything is healthy; a failure can never break serve or over-brake.

Leak discipline is inherited: GARCH is causal, the HMM posterior is filtered
(``p_bull_latest`` reads the leak-free last-bar estimate); drift and breadth
are both strictly backward-looking (no forward return, no lookahead).
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


def _build_live_obs(panel=None):
    """Assemble the live macro observation frame (market proxy + macro returns).

    `panel` — an already-loaded OHLCV panel to reuse (the drift/breadth legs
    load one for the same call); loads its own when omitted so this function
    still works standalone. Returns a date-indexed DataFrame with columns
    [market_ret, sp500_ret, dxy_ret, usdvnd_ret], or None if anything is
    unavailable.
    """
    import pandas as pd  # noqa: PLC0415

    from src.models.macro_risk_hmm import build_market_proxy_returns  # noqa: PLC0415

    if panel is None:
        from src.backtest.pipeline import RunConfig, load_ohlcv  # noqa: PLC0415

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


def drift_scalar_from_returns(
    returns: "list[float]",
    window: int,
    trigger: float,
    full: float,
    floor: float,
) -> float:
    """Pure drift-brake core: trailing `window`-session cum return → scalar.

    Piecewise linear: cum ≥ trigger → 1.0; cum ≤ full → floor; linear ramp
    between. Guards: too little history / degenerate knob ordering → 1.0
    (fail-open, matching the module contract).
    """
    if window <= 0 or len(returns) < window or full >= trigger or not 0.0 < floor <= 1.0:
        return 1.0
    cum = 1.0
    for r in returns[-window:]:
        cum *= 1.0 + float(r)
    cum -= 1.0
    if cum >= trigger:
        return 1.0
    if cum <= full:
        return floor
    # Linear interpolation between (trigger → 1.0) and (full → floor).
    frac = (trigger - cum) / (trigger - full)
    return 1.0 - frac * (1.0 - floor)


def _drift_scalar(market_ret) -> float:
    """Config-driven drift scalar from the live market-proxy return series."""
    if not getattr(CONFIG.trading, "drift_brake_enabled", False):
        return 1.0
    scalar = drift_scalar_from_returns(
        [float(r) for r in market_ret],
        int(getattr(CONFIG.trading, "drift_brake_window", 10)),
        float(getattr(CONFIG.trading, "drift_brake_trigger", -0.03)),
        float(getattr(CONFIG.trading, "drift_brake_full", -0.06)),
        float(getattr(CONFIG.trading, "drift_brake_floor", 0.5)),
    )
    return scalar


def _breadth_leg_scalar(panel) -> float:
    """Config-driven breadth scalar from the live OHLCV panel."""
    if not getattr(CONFIG.trading, "breadth_brake_enabled", False):
        return 1.0
    from src.trading.breadth import breadth_from_panel  # noqa: PLC0415
    from src.trading.breadth import breadth_scalar as _breadth_scalar_fn  # noqa: PLC0415

    window = int(getattr(CONFIG.trading, "breadth_brake_window", 20))
    breadth = breadth_from_panel(panel, window=window)
    if breadth is None:
        return 1.0  # insufficient history — fail open
    scalar = _breadth_scalar_fn(
        breadth,
        float(getattr(CONFIG.trading, "breadth_brake_trigger", 0.40)),
        float(getattr(CONFIG.trading, "breadth_brake_floor_level", 0.25)),
        float(getattr(CONFIG.trading, "breadth_brake_floor", 0.5)),
    )
    if scalar < 1.0:
        LOGGER.info("[breadth-brake] trailing breadth=%.3f (window=%d) → exposure ×%.3f",
                    breadth, window, scalar)
    return scalar


def live_exposure_scalar() -> float:
    """Market-wide exposure multiplier ∈ (0, 1.0] for today's dispatch.

    min(GARCH-HMM, drift, breadth) — three independent legs reading three
    different signals (macro/vol regime, index-return drift, universe-wide
    breadth). Each leg fails open to 1.0 independently; all disabled → 1.0.
    Every call logs the full breakdown + the binding leg (module docstring
    has the "why" for each leg).
    """
    garch_scalar = 1.0
    drift_scalar = 1.0
    breadth_scalar = 1.0

    need_panel = (getattr(CONFIG.trading, "drift_brake_enabled", False)
                  or getattr(CONFIG.trading, "breadth_brake_enabled", False)
                  or getattr(CONFIG.trading, "garch_brake_enabled", False))
    panel = None
    if need_panel:
        try:
            from src.backtest.pipeline import RunConfig, load_ohlcv  # noqa: PLC0415

            panel = load_ohlcv(RunConfig())
        except Exception:  # noqa: BLE001 — fail-open
            LOGGER.warning("[meta-controller] OHLCV panel load failed — "
                           "drift/breadth legs full exposure", exc_info=True)
            panel = None

    if getattr(CONFIG.trading, "drift_brake_enabled", False):
        try:
            from src.models.macro_risk_hmm import build_market_proxy_returns  # noqa: PLC0415

            market_ret = build_market_proxy_returns(panel)
            if market_ret is not None and len(market_ret) > 0:
                drift_scalar = _drift_scalar(market_ret)
        except Exception:  # noqa: BLE001 — fail-open
            LOGGER.warning("[drift-brake] computation failed — full exposure", exc_info=True)
            drift_scalar = 1.0

    if getattr(CONFIG.trading, "breadth_brake_enabled", False):
        try:
            breadth_scalar = _breadth_leg_scalar(panel)
        except Exception:  # noqa: BLE001 — fail-open
            LOGGER.warning("[breadth-brake] computation failed — full exposure", exc_info=True)
            breadth_scalar = 1.0

    if getattr(CONFIG.trading, "garch_brake_enabled", False):
        try:
            import numpy as np  # noqa: PLC0415

            model = _load_model()
            # Shared panel first; _build_live_obs self-loads if it's None
            # (e.g. the shared load failed but this leg is still enabled).
            obs = _build_live_obs(panel) if model is not None else None
            if model is not None and obs is not None:
                p_bull = float(model.p_bull_latest(obs))
                floor = float(getattr(CONFIG.trading, "garch_brake_floor", 0.2))
                garch_scalar = float(np.clip(p_bull, floor, 1.0))
        except Exception:  # noqa: BLE001 — fail-open: never break the live pipeline
            LOGGER.warning("[garch-brake] scalar computation failed — full exposure",
                           exc_info=True)
            garch_scalar = 1.0

    legs = {"garch": garch_scalar, "drift": drift_scalar, "breadth": breadth_scalar}
    combined = min(legs.values())
    binding = min(legs, key=legs.get)
    LOGGER.info(
        "[meta-controller] legs garch=%.3f drift=%.3f breadth=%.3f → "
        "combined=×%.3f (binding=%s)",
        garch_scalar, drift_scalar, breadth_scalar, combined,
        binding if combined < 1.0 else "none",
    )
    return combined
