"""SSI FastConnect as the primary OHLCV source for the EOD crawl (10-08-26).

WHY THIS EXISTS
───────────────
The 15:30 ICT pipeline spent 26 of its 29.5 minutes crawling OHLCV, and
almost all of that was deliberate sleeping: vnstock's guest quota (~20
req/min kill-switch) forces `crawlers._throttle_request` to pace at 4.25s
per ticker even though a single `Quote.history` call measures ~0.42s.

FastConnect is an authenticated feed with its own 60 req/min budget, so the
same 359-ticker incremental costs 6.0 min when requests run concurrently
inside that budget. Measured 10-08-26:

    vnstock     359 tickers x max(0.42s latency, 4.25s throttle) = 26.0 min
    FastConnect 359 tickers, 4 workers paced to 60 req/min       =  6.0 min

Serial FastConnect is NOT a win (2.77s median latency x 359 = 16.6 min) —
the concurrency is the point, and the shared rate limiter is what makes the
concurrency safe. Do not "simplify" this to a serial loop.

VALUE PARITY (verified before wiring, same date)
────────────────────────────────────────────────
O/H/L/C matched vnstock to 0.0000% across VCB/HPG/FPT/SSI/VNM x 9 sessions.
Volume matched exactly on 4 of 5; VCB differed 0.372% (block-deal/put-through
accounting). Prices are what every feature and label is built on, so the
price identity is the reason this swap is safe at all. FastConnect also
respects the requested window exactly, where vnstock over-returns earlier
dates.

SCOPE — deliberately narrow
───────────────────────────
This module only covers the DAILY INCREMENTAL case: a short trailing window
per ticker, one request each. Cold starts (a brand-new listing needing
2016->now = ~128 chunked requests) still go through vnstock via
`crawlers.fetch_ohlcv`'s existing fallback path. Splitting it this way keeps
the fast path free of chunking/paging logic, and the slow path is rare.

Fail-open by construction: no credentials, network down, or a malformed
response all yield an empty result, and `crawl_hose_overnight` then behaves
exactly as it did before this module existed.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

_FC_DAILY_OHLC_URL = "https://fc-data.ssi.com.vn/api/v2/Market/DailyOhlc"
_REQUEST_TIMEOUT_S = 20

# FastConnect community tier. Kept a little under the documented 60 so a
# clock skew between our pacing and theirs can't trip a 429 mid-crawl.
FC_REQUESTS_PER_MIN = 55
FC_MAX_WORKERS = 4

# One request per ticker covers this trailing window. 30 is FastConnect's
# hard per-request cap (past it the API returns 0 rows with status="Success"
# — silently, which is how an earlier probe in this project drew a wrong
# conclusion). A month of calendar days comfortably covers any weekend,
# holiday cluster, or a couple of missed cron runs.
PREFETCH_LOOKBACK_DAYS = 30

_OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]


class _RateLimiter:
    """Sliding-window limiter shared across worker threads.

    A plain per-thread sleep would let N workers each pace themselves to the
    full budget and collectively run N times over it. The window must be
    shared to mean anything.
    """

    def __init__(self, max_calls: int, per_seconds: float = 60.0) -> None:
        self._max_calls = max(1, int(max_calls))
        self._per = float(per_seconds)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self._per:
                    self._calls.popleft()
                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self._per - (now - self._calls[0])
            time.sleep(max(0.01, sleep_for))


def _access_token() -> str | None:
    """Reuse the foreign-flow module's token call — same credentials, same
    never-log-the-value contract. Returns None when unavailable."""
    from src.data.foreign_flow_crawler import _fastconnect_access_token  # noqa: PLC0415

    return _fastconnect_access_token()


def is_available() -> bool:
    """True when FastConnect credentials produce a token.

    Called once per crawl, not per ticker — a failure here means the whole
    fast path is skipped and vnstock handles the crawl as before.
    """
    return bool(_access_token())


def _parse_rows(symbol: str, rows: list[dict]) -> pd.DataFrame:
    """FastConnect rows -> the normalized frame `crawlers.fetch_ohlcv` expects.

    FastConnect returns numbers as STRINGS in absolute VND; the local parquet
    convention is thousands of VND (see the VN price-scale convention — every
    downstream cost/tick calculation depends on it), hence the /1000.
    """
    out: list[dict] = []
    for row in rows:
        try:
            trading_date = str(row.get("TradingDate") or "").strip()
            if not trading_date:
                continue
            dd, mm, yy = trading_date.split("/")
            close = float(row["Close"]) / 1000.0
            rec = {
                "ticker": symbol,
                "date": date(int(yy), int(mm), int(dd)),
                "open": float(row["Open"]) / 1000.0,
                "high": float(row["High"]) / 1000.0,
                "low": float(row["Low"]) / 1000.0,
                "close": close,
                "volume": float(row["Volume"]),
                "adj_close": close,
            }
        except (KeyError, TypeError, ValueError):
            LOGGER.debug("[fc-ohlcv] %s: unparseable row skipped", symbol)
            continue
        # A zero/blank price row is malformed, not a real halted session —
        # letting it through would poison returns and every rolling feature.
        if rec["open"] <= 0 or rec["close"] <= 0:
            continue
        out.append(rec)

    if not out:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    return pd.DataFrame(out)[_OHLCV_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_daily_ohlc(
    symbol: str,
    from_date: date,
    to_date: date,
    token: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """One ticker, one request, <=`PREFETCH_LOOKBACK_DAYS` span.

    Returns an empty frame on any failure. Callers treat empty as "fall back
    to vnstock for this ticker", never as "this ticker has no data".
    """
    symbol = symbol.upper().strip()
    span_days = (to_date - from_date).days + 1
    if span_days > PREFETCH_LOOKBACK_DAYS:
        # Guard the silent-empty trap rather than letting the API swallow it.
        LOGGER.warning(
            "[fc-ohlcv] %s: %sd span exceeds the %sd cap — clamping the start.",
            symbol, span_days, PREFETCH_LOOKBACK_DAYS,
        )
        from_date = to_date - timedelta(days=PREFETCH_LOOKBACK_DAYS - 1)

    http = session or requests
    try:
        resp = http.get(
            _FC_DAILY_OHLC_URL,
            params={
                "Symbol": symbol,
                "FromDate": from_date.strftime("%d/%m/%Y"),
                "ToDate": to_date.strftime("%d/%m/%Y"),
                "PageIndex": 1,
                "PageSize": 100,
                "ascending": "true",
            },
            headers=None if session else {"Authorization": f"Bearer {token}"},
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _parse_rows(symbol, resp.json().get("data") or [])
    except Exception:  # noqa: BLE001 — per-ticker failure must not kill the batch
        LOGGER.warning("[fc-ohlcv] %s: fetch failed, will fall back.", symbol, exc_info=True)
        return pd.DataFrame(columns=_OHLCV_COLUMNS)


def bulk_prefetch(
    tickers: list[str],
    end_date: date,
    lookback_days: int = PREFETCH_LOOKBACK_DAYS,
    max_workers: int = FC_MAX_WORKERS,
    requests_per_min: int = FC_REQUESTS_PER_MIN,
) -> tuple[dict[str, pd.DataFrame], date]:
    """Concurrently fetch a trailing window for every ticker.

    Returns `({ticker: frame}, window_start)`. Only non-empty frames are
    included, so a missing key means "FastConnect had nothing for this one"
    and the caller's fallback applies. `window_start` lets the caller check
    whether a given ticker's needed history actually fits inside what was
    prefetched — a ticker that is further behind than the window must not be
    served from this cache or it would silently keep a permanent gap.
    """
    window_start = end_date - timedelta(days=lookback_days - 1)
    if not tickers:
        return {}, window_start

    token = _access_token()
    if not token:
        LOGGER.warning("[fc-ohlcv] no FastConnect token — skipping the fast path.")
        return {}, window_start

    limiter = _RateLimiter(requests_per_min)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    def one(ticker: str) -> tuple[str, pd.DataFrame]:
        limiter.acquire()
        return ticker, fetch_daily_ohlc(ticker, window_start, end_date, token, session=session)

    started = time.monotonic()
    result: dict[str, pd.DataFrame] = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="fc-ohlcv") as pool:
            for ticker, frame in pool.map(one, tickers):
                if not frame.empty:
                    result[ticker] = frame
    finally:
        session.close()

    elapsed = time.monotonic() - started
    LOGGER.info(
        "[fc-ohlcv] prefetched %s/%s tickers for %s..%s in %.1fs (%.0f req/min).",
        len(result), len(tickers), window_start, end_date, elapsed,
        len(tickers) / elapsed * 60 if elapsed > 0 else 0.0,
    )
    return result, window_start
