"""Foreign (Khoi ngoai) + Proprietary (Tu doanh) daily flow crawler.

Pulls daily net buy/sell value+volume per ticker and maintains
``data/foreign_flow_daily.parquet``. Mirrors the isolation/degrade philosophy
of ``macro_crawler.py``: an adapter failing (timeout, schema drift, empty
200) degrades that run to a no-op, never raises past the caller.

SOURCE HONESTY (verified live this session — 2026-07-01)
──────────────────────────────────────────────────────────
  • SSI iBoard (`_fetch_ssi_hose_snapshot`) is REAL and CONFIRMED working:
        GET https://iboard-query.ssi.com.vn/stock/exchange/HOSE
    returns a LIVE SNAPSHOT of the whole HOSE (~400+ tickers) in one call,
    with real per-ticker fields: buyForeignQtty, buyForeignValue,
    sellForeignQtty, sellForeignValue, remainForeignQtty. This is the ONLY
    adapter here that has been hit live and parsed successfully.
  • CRITICAL LIMITATION: this is a LIVE BOARD SNAPSHOT, not a historical
    archive. There is no confirmed date parameter -- every call returns
    "as of right now" (embedded ``tradingDate`` reflects the exchange's
    current session, not a caller-chosen date). Multi-day HISTORY only
    exists by running this once per trading day, going forward, and letting
    ``update_foreign_flow_daily`` accumulate it via idempotent append.
    Live-tested alternatives that did NOT pan out this session: vnstock's
    Trading.foreign_trade/prop_trade (NotImplementedError, any source/key),
    TCBS public API (404 on all guessed paths), Fireant (401/404, needs a
    paid registered key), VNDirect finfo-api (timeout/404), CafeF AJAX
    price-history (real endpoint, rejects every param name tried), HOSE
    website (pure JS SPA, no discoverable REST surface via guessing).
  • Proprietary desk flow (tu doanh) is CONFIRMED ABSENT from the SSI
    payload above -- no prop/proprietary field of any kind in the real
    response. `_hose_eod_bulletin_adapter` remains an unverified stub for
    HOSE's official EOD bulletin (CSV), the only plausible free tu-doanh
    source, untested -- do not trust it without live verification first.
  • `_vndirect_adapter` remains a stub -- the finfo-api host either timed
    out or 404'd on every guessed path tested this session.

RETRY / EMPTY-200 CONTRACT
───────────────────────────
`_fetch_ssi_hose_snapshot` retries transient errors only (timeout,
connection, and 5xx server errors via `_is_retryable_5xx`) -- never 4xx or
a malformed/empty 200 body, since retrying those just burns the rate limit
for no benefit. A 5xx matters here specifically because this source has NO
historical backfill (see SOURCE HONESTY): a transient 503 that isn't
retried becomes a permanent one-day hole in the accumulating dataset. A
200 with an empty payload is a DISTINCT, logged case (not an exception) --
e.g. an exchange holiday.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any

import polars as pl
import requests
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

LOGGER = logging.getLogger("quant.foreign_flow_crawler")

_DEFAULT_PARQUET = Path("data/foreign_flow_daily.parquet")
_REQUEST_TIMEOUT_S = 15
_SSI_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_SSI_EXCHANGE_URL = "https://iboard-query.ssi.com.vn/stock/exchange/{exchange}"

# HOSE ATC (at-the-close) session end. Live-verified this session: a crawl
# run at 14:30 (before this) produced 406/406 rows flagged as leaked by the
# Phase-2 EDA leakage check (scripts/eda_flow_features.py::check_no_leakage)
# -- the "daily" foreign-flow numbers were a partial-session snapshot, not
# the settled EOD figure. A buffer past the literal close time is used
# (15:00, not 14:45) since SSI's own internal settlement/aggregation lag
# past the auction close hasn't been characterized.
_SAFE_CRAWL_HOUR = 15
_SAFE_CRAWL_MINUTE = 0

# Output schema. `source`/`fetched_at` exist specifically for the Phase-2
# leakage audit (compare `fetched_at` wall-clock against the market-close
# timestamp for `date` -- a same-day fetch before ATC close is a leak).
_SCHEMA: dict[str, pl.PolarsDataType] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "foreign_buy_val": pl.Float64,     # VND, thousands (matches OHLCV convention)
    "foreign_sell_val": pl.Float64,
    "foreign_net_val": pl.Float64,
    "foreign_buy_vol": pl.Float64,
    "foreign_sell_vol": pl.Float64,
    "foreign_remain_room_vol": pl.Float64,
    "prop_buy_val": pl.Float64,        # always NULL for now -- no verified tu-doanh source
    "prop_sell_val": pl.Float64,
    "prop_net_val": pl.Float64,
    "source": pl.Utf8,
    "fetched_at": pl.Datetime,
}

_TRANSIENT_EXC = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _is_retryable_5xx(exc: BaseException) -> bool:
    """True for a 5xx server-side HTTPError -- transient, worth retrying.

    `raise_for_status()` raises `requests.exceptions.HTTPError` for any 4xx/5xx.
    Only 5xx (server fault, typically transient) should be retried; 4xx
    (client fault -- bad path, auth, rate limit) is permanent for this call
    and must propagate immediately so the caller degrades this run to no-op.
    """
    resp = getattr(exc, "response", None)
    return (
        isinstance(exc, requests.exceptions.HTTPError)
        and resp is not None
        and resp.status_code >= 500
    )


@retry(
    retry=retry_if_exception_type(_TRANSIENT_EXC) | retry_if_exception(_is_retryable_5xx),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_ssi_hose_snapshot(exchange: str = "HOSE") -> list[dict[str, Any]]:
    """One live bulk call -- every HOSE ticker's current-session snapshot.

    Raises only on transient errors (retried) or a malformed non-200 (not
    retried, propagates to the caller who degrades to "no data this run").
    A 200 with an empty `data` array is returned as `[]`, not raised --
    the caller logs that distinctly (see `crawl_today`).
    """
    r = requests.get(
        _SSI_EXCHANGE_URL.format(exchange=exchange),
        headers=_SSI_HEADERS, timeout=_REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    payload = r.json()
    return payload.get("data") or []


def _hose_eod_bulletin_adapter() -> None:
    """HOSE official EOD bulletin (CSV) -- the only plausible free tu-doanh
    source. UNVERIFIED -- hosx.vn's own site is a JS SPA with no REST surface
    discoverable by guessing (confirmed this session). Left as a documented
    non-adapter rather than a guessed URL that would silently return nothing
    useful. Wire this only after finding the real bulletin download path.
    """
    raise NotImplementedError("HOSE EOD bulletin path not found/verified live yet.")


def crawl_today(tickers: list[str] | None = None, exchange: str = "HOSE") -> pl.DataFrame:
    """Fetch the CURRENT live snapshot, optionally filtered to `tickers`.

    `tickers=None` keeps the whole exchange -- the bulk call costs the same
    either way, so there is no reason to throw data away at crawl time; a
    caller wanting VN30-only can filter downstream at read time instead.

    Never raises past this point: a transient-exhausted or malformed
    response degrades to an empty frame (logged), matching the
    macro_crawler "single feed outage never aborts the pipeline" policy.
    """
    fetched_at = datetime.now()
    if fetched_at.time() < dtime(_SAFE_CRAWL_HOUR, _SAFE_CRAWL_MINUTE):
        LOGGER.warning(
            "[foreign_flow] crawling at %s, before the %02d:%02d safe-crawl buffer -- "
            "today's foreign-flow values will be a PARTIAL-SESSION snapshot, not the "
            "settled EOD figure. Schedule this crawler to run after %02d:%02d.",
            fetched_at.time().strftime("%H:%M"), _SAFE_CRAWL_HOUR, _SAFE_CRAWL_MINUTE,
            _SAFE_CRAWL_HOUR, _SAFE_CRAWL_MINUTE,
        )
    try:
        raw_items = _fetch_ssi_hose_snapshot(exchange)
    except Exception as exc:  # noqa: BLE001 -- exhausted retries or malformed response
        LOGGER.warning("[foreign_flow] SSI iBoard snapshot fetch failed: %s", exc)
        return pl.DataFrame(schema=_SCHEMA)

    if not raw_items:
        LOGGER.info("[foreign_flow] SSI iBoard returned an empty snapshot (holiday?) -- not an error.")
        return pl.DataFrame(schema=_SCHEMA)

    wanted = set(tickers) if tickers else None
    rows: list[dict[str, Any]] = []
    for item in raw_items:
        symbol = item.get("stockSymbol")
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        trading_date_raw = str(item.get("tradingDate") or "")
        if len(trading_date_raw) != 8:
            continue  # malformed date on this row -- skip it, don't fail the whole snapshot
        row_date = date(
            int(trading_date_raw[0:4]), int(trading_date_raw[4:6]), int(trading_date_raw[6:8]),
        )
        buy_val = item.get("buyForeignValue")
        sell_val = item.get("sellForeignValue")
        # SSI reports absolute VND -- this project's on-disk convention is
        # THOUSANDS of VND (see vn_price_scale_convention) -- scale down to match.
        rows.append({
            "date": row_date,
            "ticker": symbol,
            "foreign_buy_val": (buy_val / 1000.0) if buy_val is not None else None,
            "foreign_sell_val": (sell_val / 1000.0) if sell_val is not None else None,
            "foreign_net_val": (
                ((buy_val or 0) - (sell_val or 0)) / 1000.0
                if buy_val is not None or sell_val is not None else None
            ),
            "foreign_buy_vol": item.get("buyForeignQtty"),
            "foreign_sell_vol": item.get("sellForeignQtty"),
            "foreign_remain_room_vol": item.get("remainForeignQtty"),
            "prop_buy_val": None,
            "prop_sell_val": None,
            "prop_net_val": None,
            "source": "ssi_iboard_live_snapshot",
            "fetched_at": fetched_at,
        })

    if not rows:
        LOGGER.warning(
            "[foreign_flow] snapshot had %d tickers but none matched the requested set.",
            len(raw_items),
        )
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(rows, schema=_SCHEMA)


def update_foreign_flow_daily(
    tickers: list[str] | None = None,
    parquet_path: str | Path | None = None,
    exchange: str = "HOSE",
) -> int:
    """Crawl the CURRENT snapshot and idempotently merge into the parquet store.

    Idempotent on (date, ticker): fresh wins on overlap (same merge policy
    as `macro_crawler.update_macro_daily`), so re-running on an
    already-crawled date (e.g. calling this twice in one session) is safe
    and just overwrites that day's row with the latest fetch.

    Intended use: run this ONCE PER TRADING DAY, after close, going
    forward -- there is no historical backfill path for this source (see
    module docstring). `date` is taken from the snapshot's own
    `tradingDate`, never from the caller's wall-clock, so a run kicked off
    slightly early/late still tags the row with the exchange's actual
    session date.
    """
    path = Path(parquet_path) if parquet_path is not None else _DEFAULT_PARQUET
    fresh = crawl_today(tickers, exchange)

    if fresh.is_empty():
        LOGGER.warning("[foreign_flow] nothing fetched this run -- parquet left unchanged.")
        return pl.read_parquet(path).height if path.exists() else 0

    if path.exists():
        try:
            prev = pl.read_parquet(path)
            combined = pl.concat([prev, fresh], how="diagonal_relaxed")
        except Exception as exc:  # noqa: BLE001 -- corrupt/old-schema parquet -> rebuild from fresh
            LOGGER.warning("[foreign_flow] could not merge existing parquet (%s) -- rebuilding.", exc)
            combined = fresh
    else:
        combined = fresh

    combined = (
        combined.sort(["date", "ticker", "fetched_at"])
        .unique(subset=["date", "ticker"], keep="last")
        .sort(["date", "ticker"])
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)
    LOGGER.info(
        "[foreign_flow] wrote %d total rows -> %s (this run added up to %d rows for %s)",
        combined.height, path, fresh.height,
        fresh.get_column("date").max() if not fresh.is_empty() else "?",
    )
    return combined.height


__all__ = [
    "crawl_today",
    "update_foreign_flow_daily",
]
