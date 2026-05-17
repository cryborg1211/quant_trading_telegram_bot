import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

from src.features.triple_barrier import add_triple_barrier_labels


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

        # No `read_only=True` — DuckDB rejects mixed-config connections to the
        # same file in one process (see db_engine.py docstring). This is a
        # read-only query in practice; the SQL itself is `SELECT`-only.
        conn = duckdb.connect(self.db_path)
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
            # _generate_targets (which drops the transient high/low) is NOT run
            # on the live path, so drop them here to mirror the training schema.
            alpha_df = alpha_df.drop([c for c in ("high", "low") if c in alpha_df.columns])
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
        # No `read_only=True` — DuckDB rejects mixed-config connections to the
        # same file in one process (see db_engine.py docstring). This is a
        # read-only query in practice; the SQL itself is `SELECT`-only.
        conn = duckdb.connect(self.db_path)
        try:
            query = "SELECT * FROM macro_daily WHERE date >= ? ORDER BY date"
            return pl.from_arrow(conn.execute(query, [min_date]).arrow()).with_columns(pl.col("date").cast(pl.Date))
        finally:
            conn.close()

    def _load_sentiment_since(self, min_date) -> pl.DataFrame:
        # No `read_only=True` — DuckDB rejects mixed-config connections to the
        # same file in one process (see db_engine.py docstring). This is a
        # read-only query in practice; the SQL itself is `SELECT`-only.
        conn = duckdb.connect(self.db_path)
        try:
            # Per-ticker aggregation: each ticker gets its own sentiment signal.
            query = """
                SELECT
                    ticker,
                    CAST(date AS DATE) AS date,
                    AVG(sentiment_score) AS sentiment_score_mean,
                    AVG(magnitude) AS sentiment_magnitude_mean,
                    AVG(sentiment_nlp) AS sentiment_nlp_mean,
                    AVG(impact_force) AS sentiment_impact_force_mean,
                    COUNT(*) AS sentiment_news_count,
                    SUM(CASE WHEN is_market_wide THEN 1 ELSE 0 END) AS sentiment_market_wide_count
                FROM hist_sentiment_llm_labeled
                WHERE CAST(date AS DATE) >= ?
                GROUP BY ticker, CAST(date AS DATE)
                ORDER BY ticker, date
            """
            return pl.from_arrow(conn.execute(query, [min_date]).arrow()).with_columns(pl.col("date").cast(pl.Date))
        finally:
            conn.close()

    def _query_sentiment(self, conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
        # Per-ticker aggregation so each stock gets its own sentiment signal rather
        # than a market-wide average that is identical for every ticker on a given day.
        query = """
            SELECT
                ticker,
                CAST(date AS DATE) AS date,
                AVG(sentiment_score) AS sentiment_score_mean,
                AVG(magnitude) AS sentiment_magnitude_mean,
                AVG(sentiment_nlp) AS sentiment_nlp_mean,
                AVG(impact_force) AS sentiment_impact_force_mean,
                COUNT(*) AS sentiment_news_count,
                SUM(CASE WHEN is_market_wide THEN 1 ELSE 0 END) AS sentiment_market_wide_count
            FROM hist_sentiment_llm_labeled
            GROUP BY ticker, CAST(date AS DATE)
            ORDER BY ticker, date
        """
        return pl.from_arrow(conn.execute(query).arrow()).with_columns(pl.col("date").cast(pl.Date))

    def _preprocess_stock(self, df):
        """Calculates HLC3 (typical price = (H+L+C)/3) and ensures proper types.

        NOTE: This is NOT VWAP. True VWAP requires transaction-level volume data.
        HLC3 is a valid price-action feature but is named accurately here.
        """
        return df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("hlc3")
        ]).sort(["ticker", "date"])

    def _normalize_features(self, df):
        """Vectorized normalization partitioned by ticker."""
        price_cols = ["open", "high", "low", "close", "hlc3"]

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
        norm_cols = ["norm_open", "norm_high", "norm_low", "norm_close", "norm_volume", "norm_hlc3"]

        lag_exprs = []
        for col in norm_cols:
            base_name = col.replace("norm_", "")
            for i in range(self.lookback):
                lag_exprs.append(pl.col(col).shift(i).over("ticker").alias(f"{base_name}_{i}"))

        # Preserve raw close/volume for live inference (liquidity filter +
        # portfolio/alerts need unscaled values). high/low are preserved ONLY
        # so triple-barrier can detect intrabar PT/SL touches in
        # _generate_targets; they are dropped there before the feature matrix
        # is built so raw price levels never leak in as model features.
        base_cols = ["ticker", "date"]
        for _raw_col in ("close", "volume", "high", "low"):
            if _raw_col in df.columns:
                base_cols.append(_raw_col)

        return df.select(base_cols + lag_exprs)

    def _generate_targets(self, alpha_df, stock_df):
        """Volatility-scaled Triple-Barrier labels (Lopez de Prado, AFML Ch. 3).

        Replaces the legacy fixed ±3% thresholds. For each horizon emits:
            target_class_{h}  {0:DOWN, 1:SIDE, 2:UP}   (de Prado bin + 1)
            target_bin_{h}    {-1, 0, 1}                (raw de Prado label)
            target_return_{h} realized close-to-close return at the touched barrier
            t1_{h}            event-end date  ← REQUIRED by PurgedKFold (purging)

        `alpha_df` already carries the raw, unscaled `close` (preserved by
        `_generate_lags`), so triple-barrier runs directly on it — no join is
        needed, which also removes the legacy `close`/`close_right` join
        collision. `stock_df` is intentionally unused now (close-path
        barriers); kept in the signature for call-site compatibility.
        """
        del stock_df  # legacy arg; raw close lives on alpha_df

        # use_intrabar_extremes=True: full OHLCV (high/low) is now loaded into
        # data/ohlcv_*.parquet, so barrier touches are detected intraday — a 2σ
        # profit-take is realistically hit on the bar's high, not at its close.
        # The close-only path systematically under-counts PT/SL hits and inflates
        # the SIDEWAYS class.
        df = add_triple_barrier_labels(
            alpha_df,
            horizon=5,
            pt_mult=2.0,
            sl_mult=2.0,
            suffix="5d",
            vol_span=20,
            use_intrabar_extremes=True,
        )
        df = add_triple_barrier_labels(
            df,
            horizon=20,
            pt_mult=2.0,
            sl_mult=2.0,
            suffix="20d",
            vol_span=20,
            use_intrabar_extremes=True,
        )

        # raw close → raw_close (kept for liquidity/portfolio); drop close and
        # the transient high/low (only needed above for intrabar barrier touch)
        # so non-stationary raw price levels never enter the feature matrix.
        drop_cols = [c for c in ("close", "high", "low") if c in df.columns]
        return df.with_columns(pl.col("close").alias("raw_close")).drop(drop_cols)

    def _clean_infinite(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64]
        ])

    def _integrate_macro(self, alpha_df, macro_df):
        """Join shifted macro data to alpha features without leakage or NaN bombs.

        FREQUENCY MISMATCH HANDLING (the critical concern for monthly CPI):
        ───────────────────────────────────────────────────────────────────
        Daily series   (DXY, S&P 500, USD/VND, interbank_on_rate, vnibor):
            Already have ~1 row per business day. Forward-fill patches
            US/global holidays that don't coincide with VN trading days.

        Monthly series (inflation_yoy from VN CPI release):
            Has ~12 rows/year — one per release date. Forward-fill carries
            the most recent published reading through every subsequent
            business day until the next monthly release. This is the
            standard "as-of" macro feature treatment.

        Two passes of forward-fill are applied:
          (1) ON THE MACRO FRAME before join — propagates values across the
              dates that already have macro_daily rows.
          (2) ON THE JOINED FRAME, per-ticker — patches VN trading days
              where macro_daily had no row at all (e.g., a VN trading day
              that was a US market holiday).

        NORMALIZATION DECISION:
        ───────────────────────
        Macro features are PASSED THROUGH AS RAW PERCENTAGES — they are NOT
        Z-score normalized via `_normalize_features`. Reasoning:

          • Interest rates and inflation already live on stable, mean-reverting
            scales (~0–15% historically for VN). The 60-day rolling Z-score
            would obscure the actual policy/economic regime — a 4% rate vs
            10% rate is genuinely different state, not noise to standardize.
          • 1st-order differences could capture surprises but would lose the
            level signal that drives risk-on/risk-off rotation.

        LEAK PREVENTION:
        ────────────────
        Every macro column is shifted by 1 day before the join, so the
        feature value seen on date D is the macro reading from D-1.
        """
        # 1. Drop columns that are 100% null (defensive — if the crawler
        # hasn't populated a new column yet, exclude it rather than letting
        # downstream `drop_nulls` wipe the entire training set).
        macro_df = macro_df.select([
            pl.col(c) for c in macro_df.columns if macro_df[c].null_count() < macro_df.height
        ])

        # 2. PASS 1 forward-fill on the macro frame's own date axis.
        # This is what propagates the monthly inflation_yoy values from
        # release date through every subsequent macro_daily row.
        macro_df = macro_df.sort("date").fill_null(strategy="forward")

        # 3. Convert non-stationary price-level series to log-returns to eliminate
        # covariate shift. S&P 500 (2200→5200), DXY (89→115), and USD/VND have
        # long-term trends that put inference data OOD vs. training splits.
        # log().diff() = log(p_t/p_{t-1}) = log-return (stationary).
        # The first row becomes NaN → fill with 0.0 (no change at start of history).
        # Rate series (interbank_on_rate, vnibor, inflation_yoy) are already on
        # stationary scales (0–15%) and are kept as raw levels.
        _PRICE_LEVEL_COLS = {"sp500_close", "dxy_close", "usd_vnd"}
        _LOGRET_RENAME = {
            "sp500_close": "sp500_logret",
            "dxy_close": "dxy_logret",
            "usd_vnd": "usd_vnd_logret",
        }
        level_cols_present = [c for c in _PRICE_LEVEL_COLS if c in macro_df.columns]
        if level_cols_present:
            macro_df = macro_df.with_columns([
                pl.col(c).log().diff().fill_null(0.0).alias(_LOGRET_RENAME[c])
                for c in level_cols_present
            ]).drop(level_cols_present)

        macro_cols = [c for c in macro_df.columns if c != "date"]

        # 4. T-1 shift to prevent look-ahead bias.
        shifted_macro = macro_df.select([
            pl.col("date"),
            *[pl.col(c).shift(1).alias(f"macro_{c}") for c in macro_cols]
        ])

        alpha_df = alpha_df.with_columns(pl.col("date").cast(pl.Date))
        shifted_macro = shifted_macro.with_columns(pl.col("date").cast(pl.Date))

        # 5. Left-join macro onto (ticker, date) feature rows.
        joined_df = alpha_df.join(shifted_macro, on="date", how="left")

        # 6. PASS 2 forward-fill on the joined frame — per ticker, ordered
        # by date. This patches VN trading days where macro_daily simply
        # has no row (US holiday, source outage, etc.) so monthly CPI in
        # particular never appears as NaN once the first publication has
        # been observed.
        macro_feature_cols = [f"macro_{c}" for c in macro_cols]
        if macro_feature_cols:
            joined_df = joined_df.sort(["ticker", "date"]).with_columns([
                pl.col(c).forward_fill().over("ticker") for c in macro_feature_cols
            ])

        target_cols = ["target_class_5d"]
        feature_cols = [c for c in joined_df.columns if c not in target_cols]

        return joined_df.drop_nulls(subset=feature_cols)

    def _integrate_sentiment(self, alpha_df: pl.DataFrame, sentiment_df: pl.DataFrame) -> pl.DataFrame:
        """Joins T-1 shifted per-ticker sentiment aggregates to prevent look-ahead bias.

        Sentiment is now joined on (ticker, date) so each stock receives its own
        sentiment signal instead of a market-wide daily average. The T-1 shift is
        applied within each ticker's time series via `.over("ticker")`.
        """
        _SENTIMENT_ZERO_COLS = [
            "sentiment_score_mean_lag1",
            "sentiment_magnitude_mean_lag1",
            "sentiment_nlp_mean_lag1",
            "sentiment_impact_force_mean_lag1",
            "sentiment_news_count_lag1",
            "sentiment_market_wide_count_lag1",
        ]
        if sentiment_df.is_empty():
            LOGGER.warning("Sentiment table empty. Filling sentiment features with zeros.")
            return alpha_df.with_columns([pl.lit(0.0).alias(c) for c in _SENTIMENT_ZERO_COLS])

        sentiment_cols = [c for c in sentiment_df.columns if c not in ("ticker", "date")]
        # Sort per ticker so shift(1).over("ticker") gives correct T-1 values.
        sentiment_df = sentiment_df.sort(["ticker", "date"]).with_columns([
            pl.col(c).fill_null(0) for c in sentiment_cols
        ])
        shifted_sentiment = sentiment_df.select([
            pl.col("ticker"),
            pl.col("date"),
            *[
                pl.col(c).shift(1).over("ticker").fill_null(0).cast(pl.Float32).alias(f"{c}_lag1")
                for c in sentiment_cols
            ],
        ])

        alpha_df = alpha_df.with_columns(pl.col("date").cast(pl.Date))
        shifted_sentiment = shifted_sentiment.with_columns(pl.col("date").cast(pl.Date))
        # Join on (ticker, date): each ticker gets its own sentiment row.
        joined_df = alpha_df.join(shifted_sentiment, on=["ticker", "date"], how="left")

        sentiment_feature_cols = [c for c in joined_df.columns if c.startswith("sentiment_") and c.endswith("_lag1")]
        return joined_df.with_columns([pl.col(c).fill_null(0).cast(pl.Float32).alias(c) for c in sentiment_feature_cols])


if __name__ == "__main__":
    generator = Alpha360Generator()
    generator.run()