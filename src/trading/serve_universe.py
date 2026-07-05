"""
src/trading/serve_universe.py — dynamic top-N trailing-ADV serve universe.

Single pure-function policy module (mirrors the `regime_policy.py` / `risk_tier.py`
precedent: one small config-agnostic module per serve-path policy concern). It
mirrors the backtest engine's liquidity gate
(`src/backtest/walk_forward.py::_apply_liquidity_filter`, N=50, window=20) so
serve resolves the SAME candidate universe its validated backtest evidence was
produced under, instead of the static hardcoded `_VN30_UNIVERSE` frozenset.

DELIBERATE NO-`shift(1)` DESIGN — READ BEFORE "FIXING" THIS
──────────────────────────────────────────────────────────
The backtest computes `adv20 = mean(close*volume, window).shift(1)`. The
`.shift(1)` there is essential: `_prepare` walks a HISTORICAL calendar
day-by-day, so day D's liquidity gate must NOT see day D's own volume (that
would leak forward-looking information into the historical simulation).

Serve has NO such concern. It computes ONE live snapshot as of the most
recently COMPLETED trading session and asks "which names are liquid RIGHT
NOW?". Using that session's own volume in its own trailing mean is standard
current-ADV — not a leak. Adding a `.shift(1)` here would silently lag the
live universe by one trading day for no benefit. `test_no_shift_uses_latest_session`
locks this behavior in: mutating the LAST row's volume MUST change the ranked
ADV (the opposite assertion of the backtest's leak-check).

Purity contract: no I/O, no CONFIG import, no DB access. The caller
(`main._resolve_candidate_universe`) reads config, passes the params, and owns
the degrade-to-VN30 fallback decision — this function only computes the ranking
and signals "insufficient history" by returning an empty frozenset.
"""
from __future__ import annotations

import polars as pl

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "date", "close", "volume"}
)


def liquid_universe(
    ohlcv_pl: pl.DataFrame,
    top_n: int = 50,
    adv_window: int = 20,
    min_valid_names: int = 5,
) -> frozenset[str]:
    """Return the top-`top_n` tickers by trailing-`adv_window` dollar-volume ADV.

    Mirrors the backtest's liquidity gate but on ONE live snapshot (no
    `.shift(1)` — see module docstring). ADV is computed per ticker over its
    OWN trailing window, then each ticker's LATEST available ADV value is ranked
    across the cross-section; the top `top_n` survive.

    Args:
        ohlcv_pl: Live OHLCV panel with at least `ticker`, `date`, `close`,
            `volume` columns (the exact frame `daily_inference` already loads
            via `load_live_ohlcv_window`). Multiple rows per ticker, one per
            trading session; the last row per ticker is the most recently
            completed session.
        top_n: Number of names to keep (the backtest's validated `liquid_top_n`).
        adv_window: Trailing window length in trading rows (the backtest's
            validated `adv_window`). A ticker needs at least this many rows to
            produce a non-null ADV.
        min_valid_names: Minimum number of tickers that must have a valid
            (non-null) ADV for the universe to be considered usable. Below this
            → return an empty frozenset so the CALLER's degrade path takes over.
            The function does NOT silently fall back to "all tickers" — that
            fail-safe decision (all predictions vs. static VN30) belongs to the
            caller.

    Returns:
        A frozenset of the surviving ticker symbols (up to `top_n`), or an
        EMPTY frozenset when fewer than `min_valid_names` tickers have a valid
        ADV (insufficient history — thin window, delisted/new tickers, etc.).

    Raises:
        ValueError: When required columns are missing. This is a LOUD schema
            break (real upstream contract violation), deliberately NOT treated
            as a silent data-sparsity degrade — mirrors the engine's own
            `_prepare` "panel missing columns" distinction.
    """
    missing = _REQUIRED_COLUMNS - set(ohlcv_pl.columns)
    if missing:
        raise ValueError(
            f"liquid_universe: OHLCV panel missing required columns: "
            f"{sorted(missing)} (have: {sorted(ohlcv_pl.columns)})"
        )

    if ohlcv_pl.is_empty():
        return frozenset()

    # Per-ticker trailing dollar-volume ADV. Sort by (ticker, date) so the
    # rolling mean walks each ticker's OWN chronological window; `over("ticker")`
    # keeps the rolling window strictly within each group (no cross-ticker leak).
    # min_periods == adv_window → tickers with < adv_window rows get null ADV.
    ranked = (
        ohlcv_pl.select(["ticker", "date", "close", "volume"])
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") * pl.col("volume")).alias("dvol")
        )
        .with_columns(
            pl.col("dvol")
            .rolling_mean(window_size=adv_window, min_periods=adv_window)
            .over("ticker")
            .alias("adv")
        )
    )

    # Each ticker's LATEST available ADV = the last row of its per-ticker
    # series (most recently completed session — no `.shift(1)`).
    latest = (
        ranked.group_by("ticker")
        .agg(pl.col("adv").last().alias("adv"))
        .filter(pl.col("adv").is_not_null())
    )

    if latest.height < min_valid_names:
        return frozenset()

    # Rank by ADV descending; deterministic ties broken alphabetically by
    # ticker (determinism over byte-parity with the engine's rank(method="first")
    # insertion-order tie-break — ties are rare in $-ADV and serve only needs
    # the same TOP-N SET most days, not identical tie ordering).
    survivors = (
        latest.sort(["adv", "ticker"], descending=[True, False])
        .head(int(top_n))
        .get_column("ticker")
        .to_list()
    )
    return frozenset(str(t) for t in survivors)


__all__ = ["liquid_universe"]
