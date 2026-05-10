import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=False,
    )
    return logging.getLogger(__name__)


LOGGER = setup_logging()


@contextmanager
def timed_step(message: str):
    start = time.perf_counter()
    LOGGER.info("%s started...", message)
    try:
        yield
    finally:
        LOGGER.info("%s finished in %.2fs.", message, time.perf_counter() - start)


class Alpha360Generator:
    """Senior Quant Feature Generator following Microsoft Qlib Alpha360 architecture.

    Uses Polars for high-performance vectorized operations and zero-copy integration
    with DuckDB via Arrow.
    """

    def __init__(self, db_path="data/quant_v6_core.duckdb", output_path="data/alpha360_features.parquet"):
        self.db_path = db_path
        self.output_path = output_path
        self.lookback = 60

    def run(self):
        LOGGER.info("Starting Alpha360 Feature Generation...")
        total_start = time.perf_counter()

        with timed_step("Loading stock/macro/sentiment data"):
            stock_df, macro_df, sentiment_df = self._load_data()
        LOGGER.info(
            "Loaded %s stock rows x %s cols, %s macro rows x %s cols, %s sentiment rows x %s cols.",
            stock_df.height,
            stock_df.width,
            macro_df.height,
            macro_df.width,
            sentiment_df.height,
            sentiment_df.width,
        )

        with timed_step("Preprocessing OHLCV data and approximating VWAP"):
            stock_df = self._preprocess_stock(stock_df)
        LOGGER.info("Preprocessed shape: %s", stock_df.shape)

        with timed_step(f"Applying rolling Z-score normalization (window={self.lookback})"):
            norm_df = self._normalize_features(stock_df)
        LOGGER.info("Normalized shape: %s", norm_df.shape)

        with timed_step("Constructing Alpha360 lagged matrix (60 lags x 6 features)"):
            alpha360_df = self._generate_lags(norm_df)
        LOGGER.info("Alpha360 lagged shape: %s", alpha360_df.shape)

        with timed_step("Generating target labels"):
            labeled_df = self._generate_targets(alpha360_df, stock_df)

        with timed_step("Cleaning infinite values"):
            labeled_df = self._clean_infinite(labeled_df)

        with timed_step("Integrating shifted Macro features"):
            final_df = self._integrate_macro(labeled_df, macro_df)

        with timed_step("Integrating lagged Sentiment features"):
            final_df = self._integrate_sentiment(final_df, sentiment_df)
        LOGGER.info("Final integrated shape: %s", final_df.shape)

        if final_df.height > 0:
            with timed_step(f"Saving Alpha360 features to {self.output_path}"):
                final_df.write_parquet(self.output_path)
        else:
            LOGGER.warning("Final DataFrame is empty. Skipping save.")

        LOGGER.info("Alpha360 Generation Completed. Final shape=%s total_time=%.2fs", final_df.shape, time.perf_counter() - total_start)

    def _load_data(self):
        """Loads every data/ohlcv_*.parquet dynamically; no fixed VN100 universe."""
        parquet_files = sorted(Path("data").glob("ohlcv_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No data/ohlcv_*.parquet files found. Run crawler first.")

        LOGGER.info("Discovered %s OHLCV parquet files. Collecting full history; this might take a minute.", len(parquet_files))
        stock_df = (
            pl.scan_parquet([str(p) for p in parquet_files])
            .with_columns([
                pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
                pl.col("date").cast(pl.Date),
            ])
            .sort(["ticker", "date"])
            .collect()
        )
        ticker_count = stock_df.select(pl.col("ticker").n_unique()).item()
        min_date = stock_df.select(pl.col("date").min()).item()
        max_date = stock_df.select(pl.col("date").max()).item()
        LOGGER.info("Stock history shape=%s tickers=%s date_range=%s..%s", stock_df.shape, ticker_count, min_date, max_date)

        if ticker_count < 300:
            raise ValueError(
                f"Alpha360 universe too small: {ticker_count} tickers. "
                "Expected full HOSE universe from data/ohlcv_*.parquet."
            )

        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            macro_query = "SELECT * FROM macro_daily ORDER BY date"
            macro_df = pl.from_arrow(conn.execute(macro_query).arrow()).with_columns(pl.col("date").cast(pl.Date))
            sentiment_df = self._query_sentiment(conn)
        finally:
            conn.close()
        LOGGER.info("Macro shape=%s", macro_df.shape)
        LOGGER.info("Sentiment shape=%s", sentiment_df.shape)
        return stock_df, macro_df, sentiment_df

    def build_live_features(self, tickers: Iterable[str] | None = None, window_rows: int = 120) -> pl.DataFrame:
        """Build one latest Alpha360 row per ticker using only a small parquet tail window.

        This path is for daily inference only: no target generation, no full 10-year load.
        """
        if window_rows < self.lookback + 5:
            raise ValueError(f"window_rows must be >= {self.lookback + 5}")

        with timed_step(f"Loading live OHLCV tail window ({window_rows} rows/ticker max)"):
            stock_df = self._load_live_stock_window(tickers=tickers, window_rows=window_rows)
        LOGGER.info("Live stock window shape=%s tickers=%s", stock_df.shape, stock_df.select(pl.col("ticker").n_unique()).item())

        min_date = stock_df.select(pl.col("date").min()).item()
        with timed_step(f"Loading macro/sentiment window from {min_date}"):
            macro_df = self._load_macro_since(min_date)
            sentiment_df = self._load_sentiment_since(min_date)
        LOGGER.info("Live macro window shape=%s", macro_df.shape)
        LOGGER.info("Live sentiment window shape=%s", sentiment_df.shape)

        with timed_step("Building live Alpha360 features"):
            stock_df = self._preprocess_stock(stock_df)
            norm_df = self._normalize_features(stock_df)
            alpha_df = self._generate_lags(norm_df)
            alpha_df = self._clean_infinite(alpha_df)
            joined_df = self._integrate_macro(alpha_df, macro_df)
            joined_df = self._integrate_sentiment(joined_df, sentiment_df)
            latest_df = (
                joined_df.sort(["ticker", "date"])
                .group_by("ticker")
                .tail(1)
                .sort("ticker")
            )

        LOGGER.info("Live Alpha360 feature shape=%s latest_date=%s", latest_df.shape, latest_df.select(pl.col("date").max()).item())
        return latest_df

    def _load_live_stock_window(self, tickers: Iterable[str] | None, window_rows: int) -> pl.DataFrame:
        ticker_filter = {t.upper() for t in tickers} if tickers else None
        parquet_files = sorted(Path("data").glob("ohlcv_*.parquet"))
        if ticker_filter:
            parquet_files = [p for p in parquet_files if p.stem.replace("ohlcv_", "").upper() in ticker_filter]
        if not parquet_files:
            raise FileNotFoundError("No matching data/ohlcv_*.parquet files found.")

        frames = []
        for idx, path in enumerate(parquet_files, start=1):
            if idx == 1 or idx % 50 == 0 or idx == len(parquet_files):
                LOGGER.info("Reading live parquet tails %s/%s...", idx, len(parquet_files))
            frame = (
                pl.scan_parquet(str(path))
                .with_columns([
                    pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
                    pl.col("date").cast(pl.Date),
                ])
                .sort("date")
                .tail(window_rows)
                .collect()
            )
            if frame.height:
                frames.append(frame)

        if not frames:
            raise ValueError("All live OHLCV tail windows are empty.")
        return pl.concat(frames, how="diagonal").sort(["ticker", "date"])

    def _load_macro_since(self, min_date) -> pl.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            query = "SELECT * FROM macro_daily WHERE date >= ? ORDER BY date"
            return pl.from_arrow(conn.execute(query, [min_date]).arrow()).with_columns(pl.col("date").cast(pl.Date))
        finally:
            conn.close()

    def _load_sentiment_since(self, min_date) -> pl.DataFrame:
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            query = """
                SELECT
                    CAST(date AS DATE) AS date,
                    AVG(sentiment_score) AS sentiment_score_mean,
                    AVG(magnitude) AS sentiment_magnitude_mean,
                    AVG(sentiment_nlp) AS sentiment_nlp_mean,
                    AVG(impact_force) AS sentiment_impact_force_mean,
                    COUNT(*) AS sentiment_news_count,
                    SUM(CASE WHEN is_market_wide THEN 1 ELSE 0 END) AS sentiment_market_wide_count
                FROM hist_sentiment_llm_labeled
                WHERE CAST(date AS DATE) >= ?
                GROUP BY CAST(date AS DATE)
                ORDER BY date
            """
            return pl.from_arrow(conn.execute(query, [min_date]).arrow()).with_columns(pl.col("date").cast(pl.Date))
        finally:
            conn.close()

    def _query_sentiment(self, conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
        query = """
            SELECT
                CAST(date AS DATE) AS date,
                AVG(sentiment_score) AS sentiment_score_mean,
                AVG(magnitude) AS sentiment_magnitude_mean,
                AVG(sentiment_nlp) AS sentiment_nlp_mean,
                AVG(impact_force) AS sentiment_impact_force_mean,
                COUNT(*) AS sentiment_news_count,
                SUM(CASE WHEN is_market_wide THEN 1 ELSE 0 END) AS sentiment_market_wide_count
            FROM hist_sentiment_llm_labeled
            GROUP BY CAST(date AS DATE)
            ORDER BY date
        """
        return pl.from_arrow(conn.execute(query).arrow()).with_columns(pl.col("date").cast(pl.Date))

    def _preprocess_stock(self, df):
        """Calculates VWAP approximation and ensures proper types."""
        return df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap")
        ]).sort(["ticker", "date"])

    def _normalize_features(self, df):
        """Vectorized normalization partitioned by ticker."""
        price_cols = ["open", "high", "low", "close", "vwap"]

        expressions = []
        for col in price_cols:
            mean = pl.col(col).rolling_mean(window_size=self.lookback).over("ticker")
            std = pl.col(col).rolling_std(window_size=self.lookback).over("ticker")
            expressions.append(((pl.col(col) - mean) / (std + 1e-8)).alias(f"norm_{col}"))

        vol_col = "volume"
        log_vol = pl.col(vol_col).log1p()
        vol_mean = log_vol.rolling_mean(window_size=self.lookback).over("ticker")
        vol_std = log_vol.rolling_std(window_size=self.lookback).over("ticker")
        expressions.append(((log_vol - vol_mean) / (vol_std + 1e-8)).alias(f"norm_{vol_col}"))

        return df.with_columns(expressions)

    def _generate_lags(self, df):
        """Dynamically generates 60 lagged columns for each normalized feature."""
        norm_cols = ["norm_open", "norm_high", "norm_low", "norm_close", "norm_volume", "norm_vwap"]

        lag_exprs = []
        for col in norm_cols:
            base_name = col.replace("norm_", "")
            for i in range(self.lookback):
                lag_exprs.append(pl.col(col).shift(i).over("ticker").alias(f"{base_name}_{i}"))

        # Preserve raw close and volume for live inference (liquidity filter + portfolio/alerts need unscaled values)
        base_cols = ["ticker", "date"]
        for _raw_col in ("close", "volume"):
            if _raw_col in df.columns:
                base_cols.append(_raw_col)
        
        return df.select(base_cols + lag_exprs)

    def _generate_targets(self, alpha_df, stock_df):
        """Calculates 5-day classification targets (UP, SIDEWAY, DOWN)."""
        df = alpha_df.join(
            stock_df.select(["ticker", "date", "close"]),
            on=["ticker", "date"],
            how="left"
        )

        df = df.with_columns([
            ((pl.col("close").shift(-5).over("ticker") / pl.col("close")) - 1)
            .clip(-0.25, 0.25)
            .alias("target_return_5d"),
            ((pl.col("close").shift(-20).over("ticker") / pl.col("close")) - 1)
            .clip(-0.40, 0.40)
            .alias("target_return_20d"),
            pl.col("close").alias("raw_close"),
        ])

        return df.with_columns([
            pl.when(pl.col("target_return_5d") > 0.03).then(2)
            .when(pl.col("target_return_5d") < -0.03).then(0)
            .otherwise(1)
            .alias("target_class_5d")
        ]).drop(["close"])

    def _clean_infinite(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64]
        ])

    def _integrate_macro(self, alpha_df, macro_df):
        """Joins shifted macro data to prevent look-ahead bias."""
        macro_df = macro_df.select([
            pl.col(c) for c in macro_df.columns if macro_df[c].null_count() < macro_df.height
        ])

        macro_df = macro_df.sort("date").fill_null(strategy="forward")
        macro_cols = [c for c in macro_df.columns if c != "date"]

        shifted_macro = macro_df.select([
            pl.col("date"),
            *[pl.col(c).shift(1).alias(f"macro_{c}") for c in macro_cols]
        ])

        alpha_df = alpha_df.with_columns(pl.col("date").cast(pl.Date))
        shifted_macro = shifted_macro.with_columns(pl.col("date").cast(pl.Date))

        joined_df = alpha_df.join(shifted_macro, on="date", how="left")

        target_cols = ["target_class_5d"]
        feature_cols = [c for c in joined_df.columns if c not in target_cols]

        return joined_df.drop_nulls(subset=feature_cols)

    def _integrate_sentiment(self, alpha_df: pl.DataFrame, sentiment_df: pl.DataFrame) -> pl.DataFrame:
        """Joins T-1 shifted daily sentiment aggregates to prevent look-ahead bias."""
        if sentiment_df.is_empty():
            LOGGER.warning("Sentiment table empty. Filling sentiment features with zeros.")
            return alpha_df.with_columns([
                pl.lit(0.0).alias("sentiment_score_mean_lag1"),
                pl.lit(0.0).alias("sentiment_magnitude_mean_lag1"),
                pl.lit(0.0).alias("sentiment_nlp_mean_lag1"),
                pl.lit(0.0).alias("sentiment_impact_force_mean_lag1"),
                pl.lit(0.0).alias("sentiment_news_count_lag1"),
                pl.lit(0.0).alias("sentiment_market_wide_count_lag1"),
            ])

        sentiment_df = sentiment_df.sort("date").fill_null(0)
        sentiment_cols = [c for c in sentiment_df.columns if c != "date"]
        shifted_sentiment = sentiment_df.select([
            pl.col("date"),
            *[pl.col(c).shift(1).fill_null(0).cast(pl.Float32).alias(f"{c}_lag1") for c in sentiment_cols],
        ])

        alpha_df = alpha_df.with_columns(pl.col("date").cast(pl.Date))
        shifted_sentiment = shifted_sentiment.with_columns(pl.col("date").cast(pl.Date))
        joined_df = alpha_df.join(shifted_sentiment, on="date", how="left")

        sentiment_feature_cols = [c for c in joined_df.columns if c.startswith("sentiment_") and c.endswith("_lag1")]
        return joined_df.with_columns([pl.col(c).fill_null(0).cast(pl.Float32).alias(c) for c in sentiment_feature_cols])


if __name__ == "__main__":
    generator = Alpha360Generator()
    generator.run()