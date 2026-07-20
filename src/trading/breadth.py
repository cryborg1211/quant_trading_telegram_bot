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

import polars as pl


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
