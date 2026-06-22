"""Macro daily crawler (Macro-Integration P1).

Pulls the three market-level macro series via yfinance and maintains
``data/macro_daily.parquet`` (the retired-but-still-configured
``CONFIG.paths.macro_parquet`` path):

    ^GSPC      -> sp500     (S&P 500 index)
    DX-Y.NYB   -> dxy       (US Dollar Index)
    VND=X      -> usdvnd    (USD/VND FX)

Stored columns: ``date, sp500, dxy, usdvnd`` + per-series ``*_ret`` (pct_change
over the full stored series so boundary returns stay correct across incremental
updates). Only the ``*_ret`` ratios feed models downstream — absolute levels are
kept for reference.

vnstock cannot serve these (VN-equities only); yfinance is the source. Every
symbol fetch is isolated so one feed outage degrades that column to NaN instead
of aborting the EOD pipeline (mirrors the ``sentiment_crawler`` guard pattern).
``yfinance`` is imported lazily so the module stays importable (and unit-testable
via a mocked ``_fetch_one``) without the network dependency.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("quant.macro_crawler")

# yfinance symbol -> our column name.
_MACRO_SYMBOLS: dict[str, str] = {
    "^GSPC": "sp500",
    "DX-Y.NYB": "dxy",
    "VND=X": "usdvnd",
}
_LEVEL_COLS: list[str] = list(_MACRO_SYMBOLS.values())

# US (^GSPC/DXY) vs VN/FX trading calendars differ — forward-fill only small gaps.
_MACRO_FFILL_LIMIT: int = 3

_DEFAULT_PARQUET = Path("data/macro_daily.parquet")
_HISTORY_START = "2015-01-01"


def _fetch_one(symbol: str, start: str, end: str | None) -> pd.Series:
    """Daily Close series for one yfinance symbol, indexed by tz-naive date."""
    import yfinance as yf  # noqa: PLC0415 — lazy: keep module importable + mockable

    hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype="float64")
    s = hist["Close"].copy()
    # Normalise any tz-aware index to naive midnight (robust across yfinance versions).
    s.index = pd.to_datetime(s.index, utc=True).tz_convert(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def fetch_macro_history(start: str = _HISTORY_START, end: str | None = None) -> pd.DataFrame:
    """Outer-join the macro symbols on date → ``date, sp500, dxy, usdvnd`` + ``*_ret``.

    A symbol that fails to fetch is logged and left absent (column all-NaN), so a
    single feed outage never aborts the caller.
    """
    cols: dict[str, pd.Series] = {}
    for symbol, name in _MACRO_SYMBOLS.items():
        try:
            s = _fetch_one(symbol, start, end)
            if s.empty:
                LOGGER.warning("[macro] %s (%s) returned no rows.", name, symbol)
            cols[name] = s.rename(name)
        except Exception as exc:  # noqa: BLE001 — degrade column to NaN, never crash
            LOGGER.warning("[macro] %s (%s) fetch failed: %s", name, symbol, exc)
            cols[name] = pd.Series(dtype="float64", name=name)

    frame = pd.DataFrame(cols).sort_index()
    frame = frame.ffill(limit=_MACRO_FFILL_LIMIT)   # bridge calendar-mismatch gaps only
    frame.index.name = "date"
    out = frame.reset_index()
    for name in _LEVEL_COLS:
        if name in out.columns:
            out[f"{name}_ret"] = out[name].pct_change()
    return out


def update_macro_daily(
    parquet_path: str | Path | None = None, days_back: int | None = None
) -> int:
    """Build / refresh the macro parquet. Backfill (``days_back=None``) or incremental.

    Idempotent: merges fresh rows with the existing parquet on ``date`` (fresh wins
    on overlap), re-sorts, and recomputes ``*_ret`` over the COMBINED level series so
    boundary returns are correct. Returns the total stored row count.
    """
    path = Path(parquet_path) if parquet_path is not None else _DEFAULT_PARQUET
    if days_back is not None:
        # +5-day buffer so the short pull always overlaps the last stored vintage.
        start = (
            pd.Timestamp.today().normalize() - pd.Timedelta(days=int(days_back) + 5)
        ).strftime("%Y-%m-%d")
    else:
        start = _HISTORY_START

    fresh = fetch_macro_history(start=start)
    combined = fresh[["date", *_LEVEL_COLS]].copy()

    if path.exists():
        try:
            prev = pd.read_parquet(path)
            combined = pd.concat(
                [prev[["date", *_LEVEL_COLS]], combined], ignore_index=True
            )
        except Exception as exc:  # noqa: BLE001 — corrupt/old-schema parquet → rebuild
            LOGGER.warning("[macro] could not merge existing parquet (%s) — rebuilding.", exc)

    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = (
        combined.drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    for name in _LEVEL_COLS:
        combined[f"{name}_ret"] = combined[name].pct_change()

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    LOGGER.info(
        "[macro] wrote %d rows → %s (%s .. %s)",
        len(combined), path,
        combined["date"].min().date() if len(combined) else "-",
        combined["date"].max().date() if len(combined) else "-",
    )
    return len(combined)
