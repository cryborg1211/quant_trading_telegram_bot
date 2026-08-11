"""Official exchange-computed market breadth from SSI FastConnect DailyIndex.

WHY A NEW SOURCE WHEN breadth.py ALREADY COMPUTES BREADTH
────────────────────────────────────────────────────────
`src/trading/breadth.py` derives breadth itself — fraction of the liquid
universe with a positive trailing-N-session return. That is a PROXY. The
exchange publishes the real thing per session, and two of its fields have no
equivalent anywhere in this system:

    Advances / Declines / NoChanges  — same thing breadth.py approximates
    Ceilings / Floors                — count of limit-UP and limit-DOWN names

`Floors` is the one that motivated this. A limit-down count is a direct
capitulation measure, which is exactly what the MR knife-catch signal is trying
to time and what the (promising-but-unconfirmed) breadth-inflection work was
inferring indirectly from rolling returns. Advances/Declines is expected to
correlate strongly with the existing proxy; Ceilings/Floors is genuinely new
information.

Also captured: `TotalDealVol` / `TotalDealVal`, the put-through (thoa thuan)
block-deal totals, which sit outside the order-book matching the OHLCV panel
sees. VNINDEX carried 107.9M shares / 2.17T VND of it on 2026-08-10.

WHAT THIS IS NOT
────────────────
Not proprietary-desk (tu doanh) data. FastConnect has none — every field of all
five working endpoints was dumped on 2026-08-11 and none is prop-related, and
eight guessed endpoint names all 404. The `prop_*` columns elsewhere in this repo
come from the unofficial iBoard scrape and remain TODAY-ONLY.

Index-level, so one row per (date, index) — deliberately a separate module and
parquet from `foreign_flow_crawler`, which is per-ticker.

TWO FIELD-COVERAGE FACTS, both live-verified 2026-08-11
───────────────────────────────────────────────────────
  * `Advances` / `Declines` / `NoChanges` are populated for VNINDEX ONLY. VN30
    and VN100 return 0 for all three. Their `Ceilings`/`Floors` and deal
    volumes ARE populated, so they stay in the default set, but never read A/D
    off anything but VNINDEX or the breadth will silently read as zero.
  * An UNSETTLED session returns `IndexValue = 0` with the count fields already
    partially filled. A crawl at 14:18 (before the 14:45 ATC close) produced
    exactly that. Those rows are dropped on parse — see `parse_daily_index_rows`.
    This is the same hazard `foreign_flow_crawler._SAFE_CRAWL_HOUR` guards
    against, handled here off the DATA rather than off the clock so it holds
    regardless of when the crawl runs.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import requests

LOGGER = logging.getLogger("quant.market_breadth")

_FC_BASE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"
_FC_DAILY_INDEX_URL = f"{_FC_BASE_URL}/DailyIndex"
_REQUEST_TIMEOUT_S = 20
_DEFAULT_PARQUET = Path("data/market_breadth_daily.parquet")

# Same hard cap the other FastConnect endpoints enforce: past this the API
# returns 0 rows with status="Success" — silently, which is how an earlier probe
# in this project drew a wrong conclusion about missing data.
_FC_MAX_RANGE_DAYS = 30

# FastConnect's own community rate limit is 60/min; stay under it.
_PACE_SECONDS = 1.1

DEFAULT_INDICES = ("VNINDEX", "VN30", "VN100")

_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "index_id": pl.Utf8,
    "index_value": pl.Float64,
    "change": pl.Float64,
    "ratio_change": pl.Float64,
    "advances": pl.Int64,
    "no_changes": pl.Int64,
    "declines": pl.Int64,
    "ceilings": pl.Int64,
    "floors": pl.Int64,
    "total_match_vol": pl.Float64,
    "total_match_val": pl.Float64,
    "total_deal_vol": pl.Float64,
    "total_deal_val": pl.Float64,
    "total_vol": pl.Float64,
    "total_val": pl.Float64,
    "trading_session": pl.Utf8,
}


def _access_token() -> str | None:
    """Reuse the foreign-flow module's token call — same credentials, same
    never-log-the-value contract."""
    from src.data.foreign_flow_crawler import _fastconnect_access_token  # noqa: PLC0415

    return _fastconnect_access_token()


def _chunk_date_range(from_date: date, to_date: date,
                      max_days: int = _FC_MAX_RANGE_DAYS) -> list[tuple[date, date]]:
    """Split into consecutive <=max_days windows (inclusive spans)."""
    if from_date > to_date:
        return []
    out: list[tuple[date, date]] = []
    cursor = from_date
    step = timedelta(days=max_days - 1)
    while cursor <= to_date:
        end = min(cursor + step, to_date)
        out.append((cursor, end))
        cursor = end + timedelta(days=1)
    return out


def _f(row: dict, key: str) -> float | None:
    """FastConnect returns every number as a STRING; blank/None → None."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(row: dict, key: str) -> int | None:
    v = _f(row, key)
    return None if v is None else int(v)


def parse_daily_index_rows(rows: list[dict]) -> pl.DataFrame:
    """Map raw DailyIndex payload rows onto `_SCHEMA`.

    Values are left in EXCHANGE units — counts are counts, volumes are shares,
    values are absolute VND. No thousands-of-VND rescaling here: unlike the
    OHLCV shards this is not a price series, so the price-unit convention that
    governs `VNCostModel` does not apply and applying it would be wrong.

    UNSETTLED rows are DROPPED. A session still in progress comes back with
    `IndexValue = 0` and count fields already partly filled — a crawl at 14:18,
    before the 14:45 ATC close, returned exactly that for all three indices.
    Keeping such a row would persist a mid-session snapshot as if it were the
    settled EOD figure.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        raw_date = str(row.get("TradingDate") or "").strip()
        if not raw_date:
            continue
        try:
            dd, mm, yy = raw_date.split("/")
            d = date(int(yy), int(mm), int(dd))
        except (ValueError, TypeError):
            LOGGER.debug("[breadth] unparseable TradingDate %r skipped", raw_date)
            continue
        idx_value = _f(row, "IndexValue")
        if idx_value is None or idx_value <= 0.0:
            LOGGER.info("[breadth] %s %s: IndexValue=%s — session not settled, "
                        "row dropped.", row.get("IndexId"), raw_date, idx_value)
            continue
        out.append({
            "date": d,
            "index_id": str(row.get("IndexId") or "").strip().upper() or None,
            "index_value": idx_value,
            "change": _f(row, "Change"),
            "ratio_change": _f(row, "RatioChange"),
            "advances": _i(row, "Advances"),
            "no_changes": _i(row, "NoChanges"),
            "declines": _i(row, "Declines"),
            "ceilings": _i(row, "Ceilings"),
            "floors": _i(row, "Floors"),
            "total_match_vol": _f(row, "TotalMatchVol"),
            "total_match_val": _f(row, "TotalMatchVal"),
            "total_deal_vol": _f(row, "TotalDealVol"),
            "total_deal_val": _f(row, "TotalDealVal"),
            "total_vol": _f(row, "TotalVol"),
            "total_val": _f(row, "TotalVal"),
            "trading_session": (str(row.get("TradingSession")).strip()
                                if row.get("TradingSession") is not None else None),
        })
    if not out:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(out, schema=_SCHEMA).sort(["index_id", "date"])


def fetch_daily_index(
    index_id: str,
    from_date: date,
    to_date: date,
    *,
    token: str | None = None,
    session: requests.Session | None = None,
    page_size: int = 100,
) -> pl.DataFrame:
    """All DailyIndex rows for one index over an arbitrary range.

    Chunks the range to `_FC_MAX_RANGE_DAYS` and pages within each window using
    the response's own `totalRecord`. Never raises: a failed window is logged
    and skipped, not fatal to the rest of the backfill.
    """
    tok = token or _access_token()
    if not tok:
        LOGGER.warning("[breadth] no FastConnect token — nothing fetched.")
        return pl.DataFrame(schema=_SCHEMA)

    http = session or requests.Session()
    owns_session = session is None
    if owns_session:
        http.headers.update({"Authorization": f"Bearer {tok}"})

    frames: list[pl.DataFrame] = []
    try:
        for win_from, win_to in _chunk_date_range(from_date, to_date):
            page_index = 1
            total_record: int | None = None
            while True:
                try:
                    resp = http.get(
                        _FC_DAILY_INDEX_URL,
                        params={
                            "IndexId": index_id,
                            "FromDate": win_from.strftime("%d/%m/%Y"),
                            "ToDate": win_to.strftime("%d/%m/%Y"),
                            "PageIndex": page_index,
                            "PageSize": page_size,
                        },
                        headers=None if not owns_session else None,
                        timeout=_REQUEST_TIMEOUT_S,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as exc:  # noqa: BLE001 — skip the window, keep going
                    LOGGER.warning("[breadth] %s %s..%s page %s failed: %s",
                                   index_id, win_from, win_to, page_index, exc)
                    break

                rows = payload.get("data") or []
                if total_record is None:
                    try:
                        total_record = int(payload.get("totalRecord") or 0)
                    except (TypeError, ValueError):
                        total_record = None
                if not rows:
                    break
                frames.append(parse_daily_index_rows(rows))
                if total_record is not None and page_index * page_size >= total_record:
                    break
                if len(rows) < page_size:
                    break
                page_index += 1
    finally:
        if owns_session:
            http.close()

    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").unique(
        subset=["index_id", "date"], keep="last").sort(["index_id", "date"])


def _merge_and_persist(fresh: pl.DataFrame, path: Path) -> int:
    """Idempotent merge on (index_id, date) — fresh wins — then write.

    Mirrors `foreign_flow_crawler._merge_and_persist` so the two feeds cannot
    drift apart in on-disk merge policy.
    """
    combined = fresh
    if path.exists():
        try:
            prev = pl.read_parquet(path)
            combined = pl.concat([prev, fresh], how="diagonal_relaxed")
        except Exception as exc:  # noqa: BLE001 — corrupt/old schema → rebuild
            LOGGER.warning("[breadth] could not merge existing parquet (%s) — rebuilding.", exc)
            combined = fresh
    combined = (combined
                .unique(subset=["index_id", "date"], keep="last")
                .sort(["index_id", "date"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)
    return combined.height


def backfill_market_breadth(
    from_date: date,
    to_date: date,
    indices: tuple[str, ...] = DEFAULT_INDICES,
    parquet_path: str | Path | None = None,
) -> int:
    """Backfill DailyIndex breadth for `indices` over [from_date, to_date].

    Returns the TOTAL row count on disk afterwards. One index failing does not
    abort the others.
    """
    import time  # noqa: PLC0415

    path = Path(parquet_path) if parquet_path is not None else _DEFAULT_PARQUET
    token = _access_token()
    if not token:
        LOGGER.warning("[breadth] FastConnect auth failed — backfill aborted.")
        return pl.read_parquet(path).height if path.exists() else 0

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    frames: list[pl.DataFrame] = []
    try:
        for idx in indices:
            LOGGER.info("[breadth] fetching %s %s..%s ...", idx, from_date, to_date)
            df = fetch_daily_index(idx, from_date, to_date, token=token, session=session)
            LOGGER.info("[breadth]   %s → %d rows", idx, df.height)
            if not df.is_empty():
                frames.append(df)
            time.sleep(_PACE_SECONDS)
    finally:
        session.close()

    if not frames:
        LOGGER.warning("[breadth] backfill produced no rows — parquet unchanged.")
        return pl.read_parquet(path).height if path.exists() else 0

    fresh = pl.concat(frames, how="diagonal_relaxed")
    total = _merge_and_persist(fresh, path)
    LOGGER.info("[breadth] wrote %d total rows → %s (this run fetched %d)",
                total, path, fresh.height)
    return total


def update_market_breadth_daily(
    indices: tuple[str, ...] = DEFAULT_INDICES,
    parquet_path: str | Path | None = None,
    lookback_days: int = 5,
) -> int:
    """Daily refresh — re-fetches a short trailing window so late exchange
    corrections land. Safe to call from the EOD pipeline; never raises."""
    to_d = datetime.now().date()
    from_d = to_d - timedelta(days=max(1, lookback_days))
    try:
        return backfill_market_breadth(from_d, to_d, indices, parquet_path)
    except Exception:  # noqa: BLE001 — observability feed must not break the pipeline
        LOGGER.warning("[breadth] daily update failed — parquet left unchanged.",
                       exc_info=True)
        path = Path(parquet_path) if parquet_path is not None else _DEFAULT_PARQUET
        return pl.read_parquet(path).height if path.exists() else 0
