import io
import logging
import os
import re
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Allow `python src/data/crawlers.py` (the user-facing dry-run command in the
# `if __name__ == "__main__":` block at the bottom) by ensuring the project
# root is on sys.path BEFORE the `from config.settings import CONFIG` line.
# Harmless when imported normally — the path is already on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cloudscraper
import numpy as np
import pandas as pd
import requests
import yfinance as yf  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from vnstock import Vnstock  # type: ignore[import-untyped]
from vnstock.api.listing import Listing  # type: ignore[import-untyped]
from vnstock.api.quote import Quote  # type: ignore[import-untyped]
from vnstock.api.trading import Trading  # type: ignore[import-untyped]

from config.settings import CONFIG

LOGGER = logging.getLogger(__name__)


class StockCrawler:
    """Crawler for Vietnamese stock market data using the unified vnstock 4.x API."""

    # TD-12 circuit breaker: hard wall-clock cap per ticker fetch.
    # vnstock's underlying client has a 30s read timeout but does NOT cap
    # total wall-clock time across retries — a single stuck ticker has been
    # observed stalling 2+ minutes in production logs. 45s gives one read
    # timeout + buffer for the parquet write.
    PER_TICKER_HARD_TIMEOUT_SECONDS = 45

    def __init__(self):
        """Initializes the unified API components."""
        self.listing_api = Listing()
        self.fallback_sources = ["VCI"]
        self.last_request_ts = 0.0

    def _fetch_ohlcv_with_timeout(self, **kwargs) -> pd.DataFrame:
        """Run `fetch_ohlcv` under a hard wall-clock timeout (TD-12).

        On timeout, logs an error, records the failure to crawler_errors.txt,
        and returns an empty DataFrame so the outer loop moves on.

        Limitation: Python threads cannot be killed mid-flight. The orphaned
        worker continues running (eventually finishing its vnstock retry
        loop and exiting) — we just stop waiting for it. Memory cost is
        ~1 stack frame per orphan; for a 355-ticker overnight crawl with a
        few % timeout rate this is negligible.
        """
        ticker = str(kwargs.get("ticker", "?"))
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hose-{ticker}")
        try:
            future = executor.submit(self.fetch_ohlcv, **kwargs)
            try:
                return future.result(timeout=self.PER_TICKER_HARD_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                LOGGER.error(
                    "[%s] Hard wall-clock timeout after %ss — skipping. "
                    "Orphan worker thread will finish in background.",
                    ticker, self.PER_TICKER_HARD_TIMEOUT_SECONDS,
                )
                self._log_crawler_error(
                    ticker,
                    TimeoutError(f"wall-clock {self.PER_TICKER_HARD_TIMEOUT_SECONDS}s exceeded"),
                    "circuit-breaker",
                )
                return pd.DataFrame()
        finally:
            # `wait=False` so we don't block here waiting for the orphan;
            # `cancel_futures=True` cancels anything not yet started (no-op
            # for our single-task case but documented for clarity).
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _log_crawler_error(ticker: str, error: BaseException, context: str = "") -> None:
        """Persist ticker-level crawler errors without stopping overnight batches.

        TD-26: writes go through a RotatingFileHandler (10 MiB × 5 backups)
        attached to the `crawler.errors` named logger. Disk usage capped at
        ~60 MiB even under a year of overnight batches with thousands of
        per-ticker errors.
        """
        from src.utils.logging_utils import get_crawler_error_logger  # noqa: PLC0415
        timestamp = datetime.now().isoformat(timespec="seconds")
        get_crawler_error_logger().info(
            f"{timestamp}\t{ticker}\t{context}\t{type(error).__name__}: {error}"
        )

    def _throttle_request(self, min_interval_seconds: float = CONFIG.crawler.throttle_min_interval_seconds) -> None:
        """Guest-safe throttle: hard <= ~14 requests/min pacing, below vnstock 20/min kill-switch."""
        jitter = random.uniform(1.0, 3.0)
        elapsed = time.monotonic() - self.last_request_ts
        wait_seconds = max(jitter, min_interval_seconds - elapsed)

        if wait_seconds > 0:
            LOGGER.info("Sleeping %.2fs before request (throttle).", wait_seconds)
            time.sleep(wait_seconds)

        self.last_request_ts = time.monotonic()

    @staticmethod
    def _normalize_symbols(values) -> List[str]:
        symbols = []
        for value in values:
            if pd.isna(value):
                continue
            symbol = str(value).strip().upper()
            if symbol:
                symbols.append(symbol)
        return sorted(set(symbols))

    @staticmethod
    def _is_common_stock_symbol(symbol: str) -> bool:
        """Filter obvious covered warrants, derivatives, indices, and malformed symbols."""
        if not symbol or not re.fullmatch(r"[A-Z]{2,4}", symbol):
            return False

        warrant_prefixes = ("C", "CW")
        warrant_patterns = (
            r"^C[A-Z]{2,}\d",
            r"^CW",
            r".*W\d*$",
        )

        if symbol.startswith(warrant_prefixes):
            return False
        if any(re.fullmatch(pattern, symbol) for pattern in warrant_patterns):
            return False

        blocked = {
            "VNINDEX",
            "VN30",
            "VNXALL",
            "HNX",
            "UPCOM",
            "VNI",
            "VNALL",
        }
        return symbol not in blocked

    def _extract_hose_symbols_from_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Extract HOSE common-stock symbols from a vnstock listing dataframe."""
        if df is None or df.empty:
            return []

        normalized_cols = {str(c).lower(): c for c in df.columns}

        symbol_col = None
        for candidate in ("ticker", "symbol", "code", "organ_code"):
            if candidate in normalized_cols:
                symbol_col = normalized_cols[candidate]
                break
        if symbol_col is None:
            symbol_col = df.columns[0]

        filtered = df.copy()

        exchange_col = None
        for candidate in ("exchange", "comgroupcode", "board", "floor", "exchange_name"):
            if candidate in normalized_cols:
                exchange_col = normalized_cols[candidate]
                break

        if exchange_col is not None:
            exchange_values = filtered[exchange_col].astype(str).str.upper()
            filtered = filtered[exchange_values.str.contains("HOSE|HSX", regex=True, na=False)]

        status_col = None
        for candidate in ("status", "listed_status", "listing_status", "trading_status"):
            if candidate in normalized_cols:
                status_col = normalized_cols[candidate]
                break

        if status_col is not None:
            status_values = filtered[status_col].astype(str).str.upper()
            dead_mask = status_values.str.contains(
                "DELIST|DELISTED|SUSPEND|INACTIVE|HUY|NGUNG", regex=True, na=False
            )
            filtered = filtered[~dead_mask]

        type_col = None
        for candidate in ("type", "stock_type", "security_type", "instrument_type"):
            if candidate in normalized_cols:
                type_col = normalized_cols[candidate]
                break

        if type_col is not None:
            type_values = filtered[type_col].astype(str).str.upper()
            warrant_mask = type_values.str.contains(
                "WARRANT|CW|CHUNG QUYEN|COVERED", regex=True, na=False
            )
            filtered = filtered[~warrant_mask]

        name_col = None
        for candidate in ("name", "short_name", "organ_name", "company_name"):
            if candidate in normalized_cols:
                name_col = normalized_cols[candidate]
                break

        if name_col is not None:
            name_values = filtered[name_col].astype(str).str.upper()
            warrant_mask = name_values.str.contains(
                "WARRANT|CHUNG QUYEN|COVERED WARRANT|CW", regex=True, na=False
            )
            filtered = filtered[~warrant_mask]

        symbols = self._normalize_symbols(filtered[symbol_col].tolist())
        return [s for s in symbols if self._is_common_stock_symbol(s)]

    def get_universe(self) -> List[str]:
        """Discovers the VN100 universe using the Listing API."""
        try:
            res = self.listing_api.symbols_by_group("VN100", source="VCI")
            if res is not None:
                if isinstance(res, pd.Series):
                    return self._normalize_symbols(res.tolist())
                if isinstance(res, pd.DataFrame):
                    col = "ticker" if "ticker" in res.columns else ("symbol" if "symbol" in res.columns else res.columns[0])
                    return self._normalize_symbols(res[col].tolist())
        except Exception as e:
            LOGGER.warning("Failed to fetch VN100 universe: %s", e)
        return []

    def get_hose_universe(self) -> List[str]:
        """Discover active HOSE common-stock tickers; excludes obvious CW/warrants/dead symbols."""
        attempts = []

        for method_name, args, kwargs in [
            ("symbols_by_exchange", ("HOSE",), {"source": "VCI"}),
            ("symbols_by_exchange", ("HSX",), {"source": "VCI"}),
            ("all_symbols", tuple(), {"source": "VCI"}),
            ("all_symbols", tuple(), {}),
        ]:
            if hasattr(self.listing_api, method_name):
                attempts.append((self.listing_api, method_name, args, kwargs))

        for obj, method_name, args, kwargs in attempts:
            try:
                res = getattr(obj, method_name)(*args, **kwargs)
                if isinstance(res, pd.Series):
                    symbols = self._normalize_symbols(res.tolist())
                    symbols = [s for s in symbols if self._is_common_stock_symbol(s)]
                elif isinstance(res, pd.DataFrame):
                    symbols = self._extract_hose_symbols_from_dataframe(res)
                else:
                    symbols = self._normalize_symbols(res)
                    symbols = [s for s in symbols if self._is_common_stock_symbol(s)]

                if symbols:
                    LOGGER.info("HOSE universe discovered via %s: %s tickers.", method_name, len(symbols))
                    return symbols
            except TypeError:
                try:
                    res = getattr(obj, method_name)(*args)
                    if isinstance(res, pd.DataFrame):
                        symbols = self._extract_hose_symbols_from_dataframe(res)
                    elif isinstance(res, pd.Series):
                        symbols = self._normalize_symbols(res.tolist())
                        symbols = [s for s in symbols if self._is_common_stock_symbol(s)]
                    else:
                        symbols = self._normalize_symbols(res)
                        symbols = [s for s in symbols if self._is_common_stock_symbol(s)]
                    if symbols:
                        LOGGER.info("HOSE universe discovered via %s: %s tickers.", method_name, len(symbols))
                        return symbols
                except Exception as e:
                    self._log_crawler_error("HOSE_UNIVERSE", e, f"{method_name} fallback")
            except Exception as e:
                self._log_crawler_error("HOSE_UNIVERSE", e, method_name)

        LOGGER.error("Failed to discover HOSE universe.")
        return []

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: str = CONFIG.crawler.stock_start_date,
        end_date: Optional[str] = None,
        file_path: Optional[str] = None,
        sleep_before_request: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV with incremental local parquet append."""
        ticker = ticker.upper().strip()
        df_old = pd.DataFrame()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        fetch_start = start_date

        if file_path and os.path.exists(file_path):
            try:
                if file_path.endswith(".parquet"):
                    df_old = pd.read_parquet(file_path)
                else:
                    df_old = pd.read_csv(file_path)

                if not df_old.empty and "date" in df_old.columns:
                    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
                    max_date = df_old["date"].max()
                    fetch_start = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    LOGGER.info("%s: existing latest=%s. Fetching from %s...", ticker, max_date, fetch_start)
            except Exception as e:
                self._log_crawler_error(ticker, e, f"read {file_path}")
                LOGGER.warning("Error reading %s for %s: %s. Full fetch from %s.", file_path, ticker, e, start_date)
                df_old = pd.DataFrame()
                fetch_start = start_date
        elif file_path:
            LOGGER.info("%s: no local file. Full fetch from %s...", ticker, start_date)

        if fetch_start > end_date:
            LOGGER.info("%s: up to date through %s.", ticker, end_date)
            return df_old

        if sleep_before_request:
            self._throttle_request()

        df_new = pd.DataFrame()

        for src in self.fallback_sources:
            try:
                q = Quote(symbol=ticker, source=src)
                df = q.history(start=fetch_start, end=end_date)

                if df is not None and not df.empty:
                    df = df.reset_index()
                    df = df.rename(
                        columns={
                            "time": "date",
                            "tradingDate": "date",
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume",
                        }
                    )

                    if "date" not in df.columns:
                        raise ValueError(f"{ticker}: source {src} returned no date column")

                    if "ticker" not in df.columns:
                        df["ticker"] = ticker
                    if "adj_close" not in df.columns:
                        df["adj_close"] = df["close"]

                    required_cols = ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        raise ValueError(f"{ticker}: source {src} missing columns {missing}")

                    df = df[required_cols].dropna(subset=["open", "close", "volume"])
                    df["ticker"] = ticker
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df_new = df
                    break
            except BaseException as e:
                self._log_crawler_error(ticker, e, f"fetch {src}")
                error_text = str(e).lower()
                if "429" in error_text or "limit" in error_text or "rate" in error_text:
                    cooldown = CONFIG.crawler.rate_limit_cooldown_seconds
                    LOGGER.warning("%s: rate limit from %s. %ss hard cooldown, then continue...", ticker, src, cooldown)
                    time.sleep(cooldown)
                continue

        if df_new.empty:
            LOGGER.warning("%s: no new records fetched.", ticker)
            df_final = df_old
        else:
            if df_old.empty:
                df_final = df_new
            else:
                df_final = pd.concat([df_old, df_new], ignore_index=True)
                df_final.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=True)

            df_final = df_final.sort_values(by=["ticker", "date"]).reset_index(drop=True)

        if file_path and not df_final.empty:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if file_path.endswith(".parquet"):
                    df_final.to_parquet(file_path, index=False)
                else:
                    df_final.to_csv(file_path, index=False)
                LOGGER.info("%s: saved %s rows -> %s", ticker, len(df_final), file_path)
            except Exception as e:
                self._log_crawler_error(ticker, e, f"save {file_path}")
                LOGGER.error("%s: save failed: %s", ticker, e)

        return df_final

    def crawl_hose_overnight(
        self,
        start_date: str = CONFIG.crawler.stock_start_date,
        end_date: Optional[str] = None,
        data_dir: str = str(CONFIG.paths.data_dir),
    ) -> dict:
        """Crawl entire HOSE common-stock universe safely for overnight runs."""
        from tqdm import tqdm

        tickers = self.get_hose_universe()
        summary = {
            "total": len(tickers),
            "success": 0,
            "failed": 0,
            "skipped_empty": 0,
            "rows": 0,
        }

        if not tickers:
            raise RuntimeError("Could not discover HOSE universe.")

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Starting overnight HOSE crawl: %s tickers.", len(tickers))

        for ticker in tqdm(tickers, desc="Crawling HOSE", unit="ticker"):
            try:
                file_path = str(Path(data_dir) / f"ohlcv_{ticker}.parquet")
                # TD-12: each ticker fetch runs under a wall-clock circuit
                # breaker so a single stuck request can't stall the batch.
                df = self._fetch_ohlcv_with_timeout(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    file_path=file_path,
                    sleep_before_request=True,
                )
                if df.empty:
                    summary["skipped_empty"] += 1
                else:
                    summary["success"] += 1
                    summary["rows"] += len(df)
            except BaseException as e:
                summary["failed"] += 1
                self._log_crawler_error(ticker, e, "crawl_hose_overnight")
                error_text = str(e).lower()
                if "429" in error_text or "limit" in error_text or "rate" in error_text:
                    cooldown = CONFIG.crawler.rate_limit_cooldown_seconds
                    LOGGER.warning("%s: batch-level rate limit. %ss hard cooldown, then continue...", ticker, cooldown)
                    time.sleep(cooldown)
                else:
                    LOGGER.error("%s: failed, continuing. Error logged.", ticker)
                continue

        LOGGER.info(
            "HOSE crawl completed: total=%s success=%s empty=%s failed=%s rows=%s",
            summary["total"], summary["success"], summary["skipped_empty"], summary["failed"], summary["rows"],
        )
        return summary


def build_retry_session(
    total: int = CONFIG.crawler.request_retry_total,
    backoff_factor: float = CONFIG.crawler.request_backoff_factor,
) -> requests.Session:
    """Build a retry-capable HTTP session for unstable macro/news APIs."""
    session = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class MacroCrawler:
    """Crawler for global macro and local money market with VN-local fallbacks.

    Source-priority strategy
    ────────────────────────
    Many VN ISPs DNS-block `markets.tradingeconomics.com`, so a TE-only
    pipeline returns empty in production. Each fetcher therefore tries:

        1. SBV (State Bank of Vietnam) public rate page — `sbv.gov.vn`.
           Vietnamese government domain, rarely blocked from VN ISPs.
        2. TradingEconomics — kept for non-VN deploys and future ISP relief.

    `vnstock` 4.x note
    ──────────────────
    `Trading.money_market_historical_data(...)` was REMOVED in vnstock 4.x
    (verified via introspection — no method named `money*`/`interbank*`/
    `vnibor*`/`liquid*` exists anywhere in the package). The previous
    codebase was making a call that always raised `AttributeError` and
    then sleeping 42s through retries before swallowing it.
    """

    # ─── Primary source: SBV public interbank-rate page ───────────────────
    # "Lãi suất thị trường tiền tệ liên ngân hàng" — interbank money-market
    # rates, published daily by the State Bank of Vietnam.
    #
    # KNOWN LIMITATION (verified empirically): The SBV portal is built on
    # Liferay 7.x with React portlets. The rate table is rendered CLIENT-SIDE
    # via AJAX after page load — the static HTML returned by `requests.get`
    # contains the page chrome (nav, scripts, CSS) but NO `<table>` elements.
    #
    # Reaching the actual data therefore requires one of:
    #   (a) A headless browser (Playwright / Selenium) that runs the JS.
    #   (b) The portlet's underlying REST endpoint, which is not exposed in
    #       the static HTML and must be reverse-engineered from DevTools on
    #       the rendered page.
    #
    # Until either of those is wired in, this scraper exits gracefully with
    # an empty DataFrame — the caller's TradingEconomics fallback fires next,
    # and `_integrate_macro` in alpha360_generator drops the column rather
    # than poisoning training data.
    _SBV_INTERBANK_URL = (
        "https://www.sbv.gov.vn/webcenter/portal/vi/menu/trangchu/tk/laisuat/lstttlnh"
    )

    # Maps the column header strings SBV may use (Vietnamese or English) to
    # canonical tenor codes. Case-insensitive lookup happens after upper().
    _SBV_TENOR_NORMALIZER: dict[str, str] = {
        "ON": "ON", "QUA ĐÊM": "ON", "QUA DEM": "ON", "OVERNIGHT": "ON",
        "1W": "1W", "1 TUẦN": "1W", "1 TUAN": "1W",
        "2W": "2W", "2 TUẦN": "2W", "2 TUAN": "2W",
        "1M": "1M", "1 THÁNG": "1M", "1 THANG": "1M",
        "3M": "3M", "3 THÁNG": "3M", "3 THANG": "3M",
        "6M": "6M", "6 THÁNG": "6M", "6 THANG": "6M",
        "9M": "9M", "9 THÁNG": "9M", "9 THANG": "9M",
        "12M": "12M", "12 THÁNG": "12M", "12 THANG": "12M",
    }

    # ─── Secondary source: TradingEconomics symbol candidates ─────────────
    _OVERNIGHT_SYMBOL_CANDIDATES: tuple[tuple[str, str], ...] = (
        ("vietnaminterate", "VN_POLICY_RATE"),
        ("vnminterate", "VN_INTEREST_RATE"),
    )
    _VNIBOR_1M_SYMBOL_CANDIDATES: tuple[tuple[str, str], ...] = (
        ("vnmibor1m", "VN_INTERBANK_1M"),
        ("vnmibor", "VN_INTERBANK_GENERIC"),
        ("vietnaminterate", "VN_POLICY_RATE"),
    )
    _CPI_SYMBOL_CANDIDATES: tuple[tuple[str, str], ...] = (
        ("vnmcpip", "VN_CPI_INFLATION_YOY"),
    )

    def __init__(self):
        self.trading_api = Trading()
        # Single TradingEconomics provider shared across calls (reuses HTTP session).
        self._te_provider = MacroProvider()
        # SBV scrape is cached per-instance: the same HTML page contains both
        # ON and 1M tenors, so a single fetch serves multiple methods.
        # `None` = "not yet attempted"; an empty DataFrame = "tried, failed".
        self._sbv_cache: pd.DataFrame | None = None
        # cloudscraper handles basic JS-challenge / WAF cases for SBV.
        self._sbv_scraper = cloudscraper.create_scraper()

    # ────────────────────────────────────────────────────────────────────
    # Primary fetcher: SBV public interbank-rate page
    # ────────────────────────────────────────────────────────────────────
    def _fetch_sbv_interbank_rates(self) -> pd.DataFrame:
        """Scrape the SBV interbank-rate HTML page once per instance lifetime.

        Returns:
            Wide DataFrame with `date` plus any tenor columns the page
            exposed (e.g. ON, 1W, 1M, 3M, ...). Empty on any failure
            (DNS, HTTP non-200, missing table, parse error). Never raises.
        """
        if self._sbv_cache is not None:
            return self._sbv_cache

        try:
            response = self._sbv_scraper.get(self._SBV_INTERBANK_URL, timeout=20)
            if response.status_code != 200:
                LOGGER.warning("SBV interbank page returned HTTP %s", response.status_code)
                self._sbv_cache = pd.DataFrame()
                return self._sbv_cache

            # pandas.read_html is forgiving and handles row/colspans.
            # IMPORTANT: pandas >= 2.1 requires HTML strings to be wrapped in
            # io.StringIO — passing the raw string makes pandas interpret it
            # as a file path (FileNotFoundError on the first `<` of the body).
            tables = self._extract_html_tables(response.text)
            if not tables:
                LOGGER.warning(
                    "SBV: 0 <table> elements in static HTML (page is Liferay/React, "
                    "rate data loads via AJAX after page render). Returning empty — "
                    "headless-browser scrape required for live SBV data. "
                    "See class docstring for context."
                )
                self._dump_sbv_debug(response.text, reason="no-tables")
                self._sbv_cache = pd.DataFrame()
                return self._sbv_cache

            # Find the rate matrix: needs ≥3 columns and a date-like first column.
            best: pd.DataFrame | None = None
            for tbl in tables:
                if tbl.shape[1] < 3:
                    continue
                first_col_str = tbl.iloc[:, 0].astype(str)
                # Date format on SBV is typically "DD/MM/YYYY".
                if first_col_str.str.contains(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", regex=True, na=False).any():
                    best = tbl
                    break

            if best is None:
                LOGGER.warning("SBV: no rate-matrix table identified across %s candidate tables.", len(tables))
                self._dump_sbv_debug(response.text, reason="no-rate-matrix")
                self._sbv_cache = pd.DataFrame()
                return self._sbv_cache

            best = best.copy()

            # Normalize column headers via the tenor map; preserve the date col.
            new_cols: list[str] = []
            for raw in best.columns:
                key = str(raw).strip().upper()
                new_cols.append(self._SBV_TENOR_NORMALIZER.get(key, str(raw)))
            best.columns = new_cols

            # First column → "date"
            best = best.rename(columns={best.columns[0]: "date"})
            best["date"] = pd.to_datetime(best["date"], dayfirst=True, errors="coerce").dt.date
            best = best.dropna(subset=["date"])

            # Coerce tenor columns to numeric. SBV uses "," as decimal sep in some pages.
            for col in best.columns:
                if col == "date":
                    continue
                cleaned = best[col].astype(str).str.replace(",", ".", regex=False).str.strip()
                best[col] = pd.to_numeric(cleaned, errors="coerce")

            best = best.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            LOGGER.info(
                "SBV interbank scrape OK: %s rows, columns=%s",
                len(best), list(best.columns),
            )
            self._sbv_cache = best
            return self._sbv_cache

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SBV interbank scrape failed: %s: %s", type(exc).__name__, exc)
            self._sbv_cache = pd.DataFrame()
            return self._sbv_cache

    @staticmethod
    def _extract_html_tables(html_text: str) -> list[pd.DataFrame]:
        """Parse every <table> in `html_text` to a DataFrame using BeautifulSoup.

        Avoids the lxml/html5lib dependency that `pd.read_html` requires.
        Returns an empty list if BeautifulSoup is unavailable or no tables exist.
        """
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError:
            LOGGER.warning("BeautifulSoup4 missing — cannot parse SBV tables.")
            return []

        soup = BeautifulSoup(html_text, "html.parser")
        out: list[pd.DataFrame] = []
        for tbl in soup.find_all("table"):
            rows: list[list[str]] = []
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if len(rows) < 2:
                continue
            # First non-empty row → headers; pad any short trailing rows.
            ncols = max(len(r) for r in rows)
            header = rows[0] + [""] * (ncols - len(rows[0]))
            body = [r + [""] * (ncols - len(r)) for r in rows[1:]]
            try:
                out.append(pd.DataFrame(body, columns=header))
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def _dump_sbv_debug(html_text: str, reason: str) -> None:
        """Persist the raw SBV HTML to logs/ on parse failure so a human can
        inspect why the table wasn't found (Liferay portlet / JS-loaded body /
        markup changed). Logged path makes the location easy to grep.

        TD-26: keeps ONLY the latest dump. Previous `sbv_html_dump_*.html`
        files are unlinked before writing the new one. Caps disk usage at
        one file (~400 KB) regardless of how often the parser fails.
        """
        try:
            log_dir = CONFIG.paths.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            # Cap disk at one SBV dump — delete predecessors first.
            for old in log_dir.glob("sbv_html_dump_*.html"):
                try:
                    old.unlink()
                except Exception:  # noqa: BLE001
                    pass
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"sbv_html_dump_{reason}_{ts}.html"
            path.write_text(html_text, encoding="utf-8")
            LOGGER.warning("SBV debug HTML dumped to %s (%s bytes).", path, len(html_text))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SBV debug dump itself failed: %s", exc)

    def _sbv_extract_tenor(self, tenor: str, out_column: str) -> pd.DataFrame:
        """Slice [date, <tenor>] from the cached SBV scrape and rename to `out_column`."""
        sbv_df = self._fetch_sbv_interbank_rates()
        if sbv_df.empty or tenor not in sbv_df.columns:
            return pd.DataFrame()
        result = sbv_df[["date", tenor]].rename(columns={tenor: out_column}).dropna(subset=[out_column])
        return result.reset_index(drop=True)

    def _fetch_te_first_match(
        self,
        candidates: tuple[tuple[str, str], ...],
        out_column: str,
    ) -> pd.DataFrame:
        """Try each (symbol, label) until one returns non-empty; pivot to wide [date, out_column].

        Returns an empty DataFrame if every candidate fails or returns no data.
        Never raises — the crawler loop must continue even when external APIs
        are blocked (DNS error, rate limit, structural change).
        """
        for symbol, label in candidates:
            try:
                long_df = self._te_provider.fetch_historical_series(symbol, label)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("TradingEconomics %s (%s) raised: %s", symbol, label, exc)
                continue
            if long_df is None or long_df.empty:
                continue
            wide = long_df.rename(columns={"value": out_column})[["date", out_column]].copy()
            wide["date"] = pd.to_datetime(wide["date"]).dt.date
            wide = (
                wide.dropna(subset=[out_column])
                    .sort_values("date")
                    .drop_duplicates(subset=["date"], keep="last")
                    .reset_index(drop=True)
            )
            if wide.empty:
                continue
            LOGGER.info(
                "TradingEconomics fetch matched: symbol=%s label=%s rows=%s -> column=%s",
                symbol, label, len(wide), out_column,
            )
            return wide
        LOGGER.warning(
            "TradingEconomics: no candidate symbol returned data for column=%s (tried: %s)",
            out_column, [s for s, _ in candidates],
        )
        return pd.DataFrame()

    # The start_date/end_date arguments are accepted for API compatibility
    # with the previous vnstock-based signature, but neither SBV nor
    # TradingEconomics filters server-side on a date range — the merge logic
    # in `fetch_macro` handles alignment downstream.
    def _fetch_overnight_rate(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Fetch VN overnight interbank rate.

        Order: SBV (primary, VN-local) → TradingEconomics policy rate
        (secondary, often DNS-blocked from VN ISPs).
        Returns wide DataFrame [date, interbank_on_rate]; empty if both fail.
        """
        primary = self._sbv_extract_tenor("ON", "interbank_on_rate")
        if not primary.empty:
            LOGGER.info("Overnight rate from SBV: %s rows.", len(primary))
            return primary
        LOGGER.info("Overnight rate: SBV returned empty; trying TradingEconomics.")
        return self._fetch_te_first_match(self._OVERNIGHT_SYMBOL_CANDIDATES, "interbank_on_rate")

    def _fetch_vnibor_1m(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Fetch VN 1-month interbank rate.

        Order: SBV (primary, VN-local) → TradingEconomics (secondary).
        Returns wide DataFrame [date, vnibor]; empty if both fail.
        """
        primary = self._sbv_extract_tenor("1M", "vnibor")
        if not primary.empty:
            LOGGER.info("VNIBOR 1M from SBV: %s rows.", len(primary))
            return primary
        LOGGER.info("VNIBOR 1M: SBV returned empty; trying TradingEconomics.")
        return self._fetch_te_first_match(self._VNIBOR_1M_SYMBOL_CANDIDATES, "vnibor")

    def _fetch_inflation_yoy(self) -> pd.DataFrame:
        """Fetch VN CPI YoY%. Returns wide DataFrame [date, inflation_yoy]; empty on failure.

        TradingEconomics-only: I do not currently have a confirmed Vietnamese
        local URL for monthly CPI YoY% time-series in machine-readable form
        (GSO publishes press-release HTML/PDFs that aren't reliably parseable
        and SBV doesn't expose CPI on a public page). The TE call is wrapped
        in defensive try/except so a DNS block returns an empty DataFrame
        rather than crashing — the model can still train without inflation
        if necessary, just with one fewer feature.

        TODO(future): if a stable machine-readable VN CPI source is found,
        add an `_fetch_local_inflation_yoy()` and prepend it as primary here.
        """
        try:
            return self._fetch_te_first_match(self._CPI_SYMBOL_CANDIDATES, "inflation_yoy")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Inflation YoY fetch failed (defensive): %s", exc)
            return pd.DataFrame()

    def fetch_macro(
        self,
        start_date: str = CONFIG.crawler.macro_start_date,
        end_date: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetches DXY, S&P 500, USD/VND, and Interbank Rate since 2014, with optional incremental load."""
        df_old = pd.DataFrame()

        if file_path:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith(".parquet"):
                        df_old = pd.read_parquet(file_path)
                    else:
                        df_old = pd.read_csv(file_path)

                    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
                    max_date = df_old["date"].max()
                    start_date = max_date.strftime("%Y-%m-%d")
                    LOGGER.info("Found existing macro data. Latest date=%s. Fetching delta...", start_date)
                except Exception as e:
                    LOGGER.warning("Error reading macro %s: %s. Proceeding with full fetch.", file_path, e)
                    df_old = pd.DataFrame()
            else:
                LOGGER.info("Macro file %s not found. Fetching full history from %s...", file_path, start_date)

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date >= end_date:
            LOGGER.info("Macro data is up to date (%s). Skipping network call.", start_date)
            return df_old

        jitter = random.uniform(3.5, 5.0)
        LOGGER.info("Cooling down for %.2fs before fetching Macro data...", jitter)
        time.sleep(jitter)

        tickers = {"DX-Y.NYB": "dxy_close", "^GSPC": "sp500_close", "VND=X": "usd_vnd"}
        macro_dfs = []
        for yf_ticker, col_name in tickers.items():
            try:
                df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df[["Close"]].rename(columns={"Close": col_name})
                    macro_dfs.append(df)
            except Exception as e:
                LOGGER.warning("Error fetching %s: %s", yf_ticker, e)

        macro_combined = pd.concat(macro_dfs, axis=1) if macro_dfs else pd.DataFrame()

        # === Overnight interbank / SBV policy rate ===
        # Was: vnstock.Trading.money_market_historical_data — REMOVED in vnstock 4.x.
        # Now: TradingEconomics scrape via _fetch_overnight_rate (never raises,
        # returns empty on failure so the pipeline keeps moving).
        try:
            ir_df = self._fetch_overnight_rate(start_date=start_date, end_date=end_date)
            if not ir_df.empty:
                ir_df = ir_df.set_index("date")[["interbank_on_rate"]]
                macro_combined = macro_combined.merge(ir_df, left_index=True, right_index=True, how="outer")
                LOGGER.info("Merged %s overnight-rate rows into macro frame.", len(ir_df))
            else:
                LOGGER.warning("Overnight rate unavailable from TradingEconomics; column will be NaN.")
        except Exception as e:
            LOGGER.warning("Overnight rate merge failed: %s", e)

        # === NEW: 1-month VNIBOR ===
        try:
            vnibor_df = self._fetch_vnibor_1m(start_date=start_date, end_date=end_date)
            if not vnibor_df.empty:
                vnibor_df = vnibor_df.copy()
                vnibor_df["date"] = pd.to_datetime(vnibor_df["date"]).dt.date
                vnibor_df = vnibor_df.set_index("date")[["vnibor"]]
                macro_combined = macro_combined.merge(vnibor_df, left_index=True, right_index=True, how="outer")
                LOGGER.info("Merged %s VNIBOR 1M rows into macro frame.", len(vnibor_df))
        except Exception as e:
            LOGGER.warning("VNIBOR 1M merge failed: %s", e)

        # === NEW: VN CPI YoY (monthly) ===
        try:
            cpi_df = self._fetch_inflation_yoy()
            if not cpi_df.empty:
                cpi_df = cpi_df.set_index("date")[["inflation_yoy"]]
                macro_combined = macro_combined.merge(cpi_df, left_index=True, right_index=True, how="outer")
                LOGGER.info("Merged %s CPI YoY rows into macro frame.", len(cpi_df))
        except Exception as e:
            LOGGER.warning("VN CPI YoY merge failed: %s", e)

        if macro_combined.empty:
            LOGGER.info("Data is up to date. No new records fetched.")
            df_final = df_old
        else:
            expected_cols = ["dxy_close", "sp500_close", "usd_vnd", "interbank_on_rate", "vnibor", "inflation_yoy"]
            for col in expected_cols:
                if col not in macro_combined.columns:
                    macro_combined[col] = np.nan

            macro_combined.index.name = "date"
            macro_combined = macro_combined.reset_index()
            macro_combined["date"] = pd.to_datetime(macro_combined["date"]).dt.date
            # `.ffill()` here is the FIRST forward-fill pass: it propagates
            # monthly CPI values across every business day until the next
            # release, so each row in macro_daily has a non-null inflation
            # reading once the first publication has occurred.
            df_new = macro_combined[["date", *expected_cols]].sort_values("date").ffill()

            if df_old.empty:
                df_final = df_new
            else:
                df_final = pd.concat([df_old, df_new], ignore_index=True)
                df_final.drop_duplicates(subset=["date"], keep="last", inplace=True)

            df_final = df_final.sort_values(by=["date"]).reset_index(drop=True)

        if file_path and not df_final.empty:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if file_path.endswith(".parquet"):
                    df_final.to_parquet(file_path, index=False)
                else:
                    df_final.to_csv(file_path, index=False)
                LOGGER.info("Successfully saved %s macro records to %s.", len(df_final), file_path)
            except Exception as e:
                LOGGER.error("Error saving to %s: %s", file_path, e)

        return df_final


class MacroProvider:
    """Qlib-style Provider with fail-safe handling."""

    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.session = build_retry_session()
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36"}

    def fetch_historical_series(self, symbol: str, indicator_name: str) -> pd.DataFrame:
        """Fetches from TradingEconomics Chart API with graceful failure."""
        urls = [
            f"https://markets.tradingeconomics.com/chart?s={symbol}&span=20y",
            f"https://markets.tradingeconomics.com/chart?s={symbol.lower()}&span=20y",
        ]
        for url in urls:
            try:
                response = self.session.get(url, headers=self.headers, timeout=20)
                if response.status_code != 200:
                    response = self.scraper.get(url, headers=self.headers, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        payload = data[0].get("data", []) if data and isinstance(data[0], dict) else []
                    elif isinstance(data, dict):
                        payload = data.get("data", [])
                    else:
                        payload = []
                    df = pd.DataFrame(payload)
                    if not df.empty and {"x", "y"}.issubset(df.columns):
                        df["date"] = pd.to_datetime(df["x"], unit="ms").dt.date
                        df["indicator_name"] = indicator_name
                        df["value"] = pd.to_numeric(df["y"], errors="coerce")
                        return df[["date", "indicator_name", "value"]].dropna(subset=["value"])
            except Exception as exc:
                LOGGER.warning("%s fetch failed at %s: %s", indicator_name, url, exc)
        LOGGER.warning("Skipping %s; all retries/fallback URLs failed.", indicator_name)
        return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        indicators = {"vnmcpip": "VN_CPI_MONTHLY", "vietnamdepintrat": "VN_DEPOSIT_RATE_12M"}
        dfs = [self.fetch_historical_series(s, n) for s, n in indicators.items()]
        valid_dfs = [d for d in dfs if not d.empty]
        return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Dry-run verification — `python src/data/crawlers.py`
# ---------------------------------------------------------------------------
# This block exercises the rate fetchers and confirms:
#   1. No `AttributeError` from the long-removed vnstock money-market endpoint.
#   2. Each fetcher returns a `pd.DataFrame` (empty is acceptable when the
#      machine has no outbound network or TradingEconomics blocks the agent).
#
# Run:
#     python src/data/crawlers.py
#
# Exit code 0 = no AttributeError, all return values are DataFrame.
# Exit code 1 = a structural failure that needs a code fix.
if __name__ == "__main__":
    import traceback

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print("MacroCrawler rate-fetcher dry-run")
    print("=" * 70)

    crawler = MacroCrawler()
    failures: list[str] = []

    # First, exercise the SBV primary source directly so you can see exactly
    # what was scraped (or what failed). Useful for diagnosing whether SBV
    # changed page structure vs. a network-level block.
    print("\n--- SBV primary scrape ---")
    try:
        sbv_df = crawler._fetch_sbv_interbank_rates()
        if sbv_df.empty:
            print("⚠️  SBV scrape returned empty (likely network/structure issue, see WARNING logs above).")
        else:
            print(f"✅ SBV scrape: {len(sbv_df)} rows; columns={list(sbv_df.columns)}")
            print(sbv_df.head(3).to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ SBV scrape raised: {type(exc).__name__}: {exc}")
        failures.append("_fetch_sbv_interbank_rates")

    print("\n--- Public fetcher API ---")
    for fn_name in ("_fetch_overnight_rate", "_fetch_vnibor_1m", "_fetch_inflation_yoy"):
        try:
            fn = getattr(crawler, fn_name)
            df = fn()
        except AttributeError as exc:
            # Treated as a hard failure — this is exactly what the previous
            # vnstock-based code would crash with, and the whole point of
            # this dry-run is to prove it no longer happens.
            print(f"❌ {fn_name}: AttributeError — {exc}")
            traceback.print_exc()
            failures.append(fn_name)
            continue
        except Exception as exc:  # noqa: BLE001
            # Any other exception is also a code-shape failure (the methods
            # are advertised as never-raise / return-empty-on-failure).
            print(f"❌ {fn_name}: unexpected exception — {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append(fn_name)
            continue

        if not isinstance(df, pd.DataFrame):
            print(f"❌ {fn_name}: did not return DataFrame (got {type(df).__name__})")
            failures.append(fn_name)
            continue

        status = "OK (data)" if not df.empty else "OK (empty — likely network blocked)"
        cols = list(df.columns) if not df.empty else "n/a"
        print(f"✅ {fn_name}: rows={len(df)} cols={cols} -> {status}")

    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} fetcher(s) raised structural errors: {failures}")
        sys.exit(1)
    print("PASSED: all fetchers returned DataFrame without AttributeError.")
    sys.exit(0)