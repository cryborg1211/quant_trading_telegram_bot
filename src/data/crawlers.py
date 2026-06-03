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

# Ensure the project root is on sys.path so `from config.settings import CONFIG`
# resolves when this module is imported (e.g. via the EOD crawl entrypoints).
# Harmless when imported normally — the path is already on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cloudscraper
import numpy as np
import pandas as pd
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
