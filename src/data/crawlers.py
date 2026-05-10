import logging
import os
import re
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

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

    def __init__(self):
        """Initializes the unified API components."""
        self.listing_api = Listing()
        self.fallback_sources = ["VCI"]
        self.last_request_ts = 0.0

    @staticmethod
    def _log_crawler_error(ticker: str, error: BaseException, context: str = "") -> None:
        """Persist ticker-level crawler errors without stopping overnight batches."""
        log_dir = CONFIG.paths.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        msg = f"{timestamp}\t{ticker}\t{context}\t{type(error).__name__}: {error}\n"
        with (log_dir / "crawler_errors.txt").open("a", encoding="utf-8") as f:
            f.write(msg)

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
                df = self.fetch_ohlcv(
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
    """Crawler for global macro and local money market using Unified API."""

    def __init__(self):
        self.trading_api = Trading()

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

        ir_df = pd.DataFrame()
        for source in ["VCI", "TCBS", "MSN"]:
            for attempt in range(1, 4):
                try:
                    ir_df = self.trading_api.money_market_historical_data(
                        symbol="ON",
                        source=source,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if ir_df is not None and not ir_df.empty:
                        break
                except Exception as e:
                    wait = min(30, 2**attempt)
                    LOGGER.warning("Interbank Rate fetch failed source=%s attempt=%s: %s. Retry in %ss.", source, attempt, e, wait)
                    time.sleep(wait)
            if ir_df is not None and not ir_df.empty:
                break

        if ir_df is not None and not ir_df.empty:
            try:
                ir_df = ir_df.reset_index()
                t_col = "time" if "time" in ir_df.columns else ("date" if "date" in ir_df.columns else ir_df.columns[0])
                v_col = "value" if "value" in ir_df.columns else ir_df.columns[1]
                ir_df["date"] = pd.to_datetime(ir_df[t_col]).dt.date
                ir_df = ir_df.set_index("date")[[v_col]].rename(columns={v_col: "interbank_on_rate"})
                macro_combined = macro_combined.merge(ir_df, left_index=True, right_index=True, how="outer")
            except Exception as e:
                LOGGER.warning("Interbank Rate normalization failed: %s", e)
        else:
            LOGGER.warning("Interbank Rate fetch exhausted all sources.")

        if macro_combined.empty:
            LOGGER.info("Data is up to date. No new records fetched.")
            df_final = df_old
        else:
            for col in ["dxy_close", "sp500_close", "usd_vnd", "interbank_on_rate"]:
                if col not in macro_combined.columns:
                    macro_combined[col] = np.nan

            macro_combined.index.name = "date"
            macro_combined = macro_combined.reset_index()
            macro_combined["date"] = pd.to_datetime(macro_combined["date"]).dt.date
            df_new = macro_combined[["date", "dxy_close", "sp500_close", "usd_vnd", "interbank_on_rate"]].sort_values("date").ffill()

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