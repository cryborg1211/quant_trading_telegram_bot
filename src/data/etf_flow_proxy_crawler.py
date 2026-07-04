"""VN-focused foreign ETF activity proxy crawler.

Historical, free, real alternative to per-ticker foreign/proprietary flow
(see `foreign_flow_crawler.py`, which can only ever see "today" -- SSI
iBoard has no date-indexed history). This module tracks two internationally
-listed ETFs whose whole purpose is foreign capital access to Vietnamese
equities:

    VNM      (NYSEArca) -- VanEck Vietnam ETF
    00885.TW (TWSE)     -- Fubon FTSE Vietnam ETF

TICKER COLLISION WARNING -- READ THIS
──────────────────────────────────────
The VanEck ETF's ticker is literally "VNM" -- IDENTICAL to the Vietnamese
stock VNM (Vinamilk) used everywhere else in this codebase
(`data/ohlcv_VNM.parquet`, `_VN30_UNIVERSE`, etc.). They are NOT the same
instrument. This module never writes bare "VNM" as a ticker value -- every
row is tagged with an explicit `venue` column (`NYSEARCA` / `TWSE`) and the
output parquet is namespaced (`data/etf_flow_proxy.parquet`), separate from
`data/ohlcv_*.parquet`. Do not join this table to the main OHLCV/feature
pipeline on bare ticker string -- always join on (etf_symbol, venue).

WHAT THIS ACTUALLY MEASURES (be honest about the gap)
──────────────────────────────────────────────────────
The clean version of this proxy would be creation/redemption flow (ETF
shares outstanding delta x NAV), which directly measures authorized-
participant-level foreign capital moving in/out. That requires historical
shares-outstanding data. VERIFIED THIS SESSION: `yfinance`'s
`Ticker.get_shares_full()` returns None for both VNM and 00885.TW -- not
available for these tickers via this free source. So this module instead
stores daily Close + Volume (both DO work via `yfinance`, verified) and
derives `dollar_volume` (Close x Volume, native currency) and `ret_1d` as
an ACTIVITY/SENTIMENT proxy, not a true flow decomposition. A volume spike
on VNM/00885.TW is suggestive of foreign interest, not proof of net
buying/selling direction the way a real creation/redemption number would
be. Treat this as weaker evidence than the SSI foreign-room numbers,
market-level only (not per-VN-ticker), and re-derive `net_flow`-style
features from it only after the Phase-2 lead-lag correlation check (same
gate as everything else in this feature line).

RETRY / EMPTY CONTRACT
────────────────────────
Mirrors `macro_crawler.py`: each symbol fetch is isolated -- one ETF's feed
failing degrades that column to absent rows, never aborts the other.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("quant.etf_flow_proxy_crawler")

# yfinance symbol -> (our etf_symbol label, venue).
_ETF_SYMBOLS: dict[str, tuple[str, str]] = {
    "VNM": ("VNM", "NYSEARCA"),          # VanEck Vietnam ETF -- NOT the VN stock VNM
    "00885.TW": ("00885", "TWSE"),       # Fubon FTSE Vietnam ETF
}

_DEFAULT_PARQUET = Path("data/etf_flow_proxy.parquet")
_HISTORY_START = "2015-01-01"


def _fetch_one(yf_symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Daily Close+Volume for one yfinance symbol. Empty frame on failure --
    never raises, matching `macro_crawler._fetch_one`'s per-symbol isolation.
    """
    import yfinance as yf  # noqa: PLC0415 -- lazy: keep module importable + mockable

    try:
        hist = yf.Ticker(yf_symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 -- degrade to empty, never crash the caller
        LOGGER.warning("[etf_flow] %s fetch raised %s: %s", yf_symbol, type(exc).__name__, exc)
        return pd.DataFrame(columns=["date", "close", "volume"])

    if hist is None or hist.empty or "Close" not in hist.columns or "Volume" not in hist.columns:
        LOGGER.warning("[etf_flow] %s returned no usable rows.", yf_symbol)
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = hist[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"}).copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(None).normalize()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "date"
    return out.reset_index()


def fetch_etf_flow_proxy_history(start: str = _HISTORY_START, end: str | None = None) -> pd.DataFrame:
    """One row per (date, etf_symbol, venue) across both tracked ETFs.

    Columns: date, etf_symbol, venue, close, volume, dollar_volume, ret_1d.
    `ret_1d` is computed PER SYMBOL (never across the concat boundary) so a
    symbol with a shorter history never produces a spurious cross-symbol
    return on its first row.
    """
    frames: list[pd.DataFrame] = []
    for yf_symbol, (etf_symbol, venue) in _ETF_SYMBOLS.items():
        raw = _fetch_one(yf_symbol, start, end)
        if raw.empty:
            continue
        raw["etf_symbol"] = etf_symbol
        raw["venue"] = venue
        raw = raw.sort_values("date")
        raw["ret_1d"] = raw["close"].pct_change()
        raw["dollar_volume"] = raw["close"] * raw["volume"]
        frames.append(raw[["date", "etf_symbol", "venue", "close", "volume", "dollar_volume", "ret_1d"]])

    if not frames:
        return pd.DataFrame(columns=["date", "etf_symbol", "venue", "close", "volume", "dollar_volume", "ret_1d"])
    return pd.concat(frames, ignore_index=True).sort_values(["etf_symbol", "date"]).reset_index(drop=True)


def update_etf_flow_proxy(
    parquet_path: str | Path | None = None, days_back: int | None = None,
) -> int:
    """Build/refresh the ETF proxy parquet. Backfill (`days_back=None`,
    default) or incremental. Idempotent merge on (date, etf_symbol),
    fresh-wins-on-overlap -- same policy as `macro_crawler.update_macro_daily`.
    """
    path = Path(parquet_path) if parquet_path is not None else _DEFAULT_PARQUET
    if days_back is not None:
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=int(days_back) + 5)).strftime("%Y-%m-%d")
    else:
        start = _HISTORY_START

    fresh = fetch_etf_flow_proxy_history(start=start)
    if fresh.empty:
        LOGGER.warning("[etf_flow] nothing fetched this run -- parquet left unchanged.")
        return pd.read_parquet(path).shape[0] if Path(path).exists() else 0

    combined = fresh
    if path.exists():
        try:
            prev = pd.read_parquet(path)
            combined = pd.concat([prev, fresh], ignore_index=True)
        except Exception as exc:  # noqa: BLE001 -- corrupt/old-schema parquet -> rebuild from fresh
            LOGGER.warning("[etf_flow] could not merge existing parquet (%s) -- rebuilding.", exc)
            combined = fresh

    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = (
        combined.drop_duplicates(subset=["date", "etf_symbol"], keep="last")
        .sort_values(["etf_symbol", "date"])
        .reset_index(drop=True)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    LOGGER.info(
        "[etf_flow] wrote %d rows -> %s (symbols=%s)",
        len(combined), path, sorted(combined["etf_symbol"].unique().tolist()),
    )
    return len(combined)


__all__ = ["fetch_etf_flow_proxy_history", "update_etf_flow_proxy"]
