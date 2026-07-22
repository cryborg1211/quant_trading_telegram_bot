"""Market breadth — pure functions (meta-controller leg, 2026-07-20).

Born from the July-2026 post-mortem: `scripts/check_drift.py` found the
July bleed was a BREADTH collapse (trailing realized 20d UP rate 29.5% vs
41.5% train), not model decay — yet nothing in the exposure stack read
breadth. Every existing brake watches a proxy (price micro-regime, macro
vol, index drift) one step removed from "how many names are actually
working right now." This is the direct measure.

Breadth here = fraction of the liquid universe with a POSITIVE trailing
`window`-session return, as of today. Purely backward-looking (no forward
return, no lookahead) — safe to compute live at serve time, unlike
`check_drift.py`'s forward-return diagnostic which needs the window to have
already matured.
"""
from __future__ import annotations

import logging

import polars as pl

LOGGER = logging.getLogger(__name__)


def breadth_from_panel(
    panel: pl.DataFrame,
    window: int = 20,
    min_tickers: int = 30,
) -> float | None:
    """Fraction of tickers with positive trailing `window`-session return.

    `panel` needs columns [ticker, date, close] (the standard OHLCV panel
    shape used across this module — see `src.backtest.pipeline.load_ohlcv`).
    Returns None (fail-open signal for the caller) when fewer than
    `min_tickers` names have enough history — never raises.
    """
    if panel is None or panel.is_empty() or window <= 0:
        return None
    positives = 0
    total = 0
    for _ticker, sub in panel.select(["ticker", "date", "close"]).group_by(
        "ticker", maintain_order=True
    ):
        closes = sub.sort("date")["close"].to_list()
        if len(closes) < window + 1:
            continue
        c0, c1 = closes[-window - 1], closes[-1]
        if c0 is None or c1 is None or c0 <= 0:
            continue
        total += 1
        if (c1 / c0 - 1.0) > 0.0:
            positives += 1
    if total < min_tickers:
        return None
    return positives / total


def breadth_scalar(
    breadth: float,
    trigger: float,
    floor_level: float,
    floor: float,
) -> float:
    """Piecewise-linear exposure scalar from a breadth reading.

    breadth ≥ trigger → 1.0; breadth ≤ floor_level → floor; linear ramp
    between. Mirrors `garch_brake.drift_scalar_from_returns`'s shape for
    consistency. Degenerate knob ordering → 1.0 (fail-open).
    """
    if floor_level >= trigger or not 0.0 < floor <= 1.0:
        return 1.0
    if breadth >= trigger:
        return 1.0
    if breadth <= floor_level:
        return floor
    frac = (trigger - breadth) / (trigger - floor_level)
    return 1.0 - frac * (1.0 - floor)


def breadth_delta_from_panel(
    panel: pl.DataFrame,
    window: int = 20,
    delta_window: int = 5,
    min_tickers: int = 30,
) -> tuple[float, float] | None:
    """(breadth_now, breadth_delta) as of the panel's latest date.

    `breadth_delta` = breadth_now minus breadth as of `delta_window` sessions
    earlier — the "inflection" from the 22-07-26 knife-catch research
    (scripts/analyze_mr_breadth_inflection.py): positive = breadth turning
    UP. Reuses `breadth_from_panel` on two date-truncated slices of the SAME
    panel — no new math, no lookahead (both endpoints strictly ≤ their own
    date). Returns None if either endpoint lacks enough history/tickers.
    """
    if panel is None or panel.is_empty():
        return None
    dates = panel.select("date").unique().sort("date")["date"].to_list()
    if len(dates) <= delta_window:
        return None
    now_breadth = breadth_from_panel(panel, window=window, min_tickers=min_tickers)
    if now_breadth is None:
        return None
    earlier_cutoff = dates[-1 - delta_window]
    earlier_panel = panel.filter(pl.col("date") <= earlier_cutoff)
    earlier_breadth = breadth_from_panel(earlier_panel, window=window, min_tickers=min_tickers)
    if earlier_breadth is None:
        return None
    return now_breadth, now_breadth - earlier_breadth


def live_breadth_inflection(
    window: int = 20,
    delta_window: int = 5,
    low_cut: float = 0.41,
    rising_threshold: float = 0.03,
) -> dict | None:
    """Live {breadth, breadth_delta, favorable} for today. Fail-open → None.

    `favorable` = the "Low breadth + Rising delta" bucket from the
    22-07-26 knife-catch research: OOF precision 0.667 (n=63 fires) vs 0.542
    for Low+Falling (n=264) on a large purged-OOF sample — a REAL but
    UNCONFIRMED gap (the strict 1-year hold-out was too thin, 6 vs 8 fires,
    to independently verify; direction agreed but wasn't statistically
    meaningful at that n). Informational only — this NEVER gates a trade or
    changes the MR-LGBM fire decision; it only annotates the existing
    knife-catch display so a human can weigh current breadth context. Any
    failure (missing OHLCV, insufficient history) returns None and the
    caller must degrade to "context unavailable", never block on it.
    """
    try:
        from src.backtest.pipeline import RunConfig, load_ohlcv  # noqa: PLC0415

        panel = load_ohlcv(RunConfig())
        result = breadth_delta_from_panel(panel, window=window, delta_window=delta_window)
        if result is None:
            return None
        breadth, delta = result
        favorable = breadth < low_cut and delta > rising_threshold
        return {"breadth": breadth, "breadth_delta": delta, "favorable": favorable}
    except Exception:  # noqa: BLE001 — fail-open, never break the caller
        LOGGER.warning("[breadth-inflection] live computation failed — "
                       "context unavailable", exc_info=True)
        return None
