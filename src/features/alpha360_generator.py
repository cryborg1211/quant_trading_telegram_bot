import logging
from pathlib import Path
from typing import Iterable

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


class Alpha360Generator:
    """Live RAW-OHLCV window loader (parquet-first).

    HISTORICAL NOTE — this class used to be a full Microsoft-Qlib-style Alpha360
    feature factory (60-lag rolling-Z-scored price/volume matrix + macro +
    sentiment integration, written to ``data/alpha360_features.parquet``). That
    ENTIRE build path was retired in the parquet-first migration:

      * V4 inference recomputes ALL of its own features from raw prices via
        ``src/backtest/pipeline.build_features`` (train/serve parity), so it
        never reads an Alpha360 feature matrix;
      * ``data/alpha360_features.parquet`` was deleted (2.25 GB dead artifact);
      * ``macro_daily`` was dropped (V4 uses cross-sectional price alphas + an
        HMM price proxy, not macro features), so the macro integration is gone;
      * point price lookups moved to ``src/data/price_lookup.py``.

    What remains is the only piece still used live: loading a RAW multi-row
    OHLCV tail window per ticker from the ``data/ohlcv_*.parquet`` shards, for
    the V3/V4 tabular pipeline and the mean-reversion sub-model.

    The class name and the ``db_path`` / ``output_path`` constructor args are
    retained only for call-site compatibility (main.py + tests construct it as
    ``Alpha360Generator()``); they are otherwise vestigial. Renaming/relocating
    this to e.g. ``src/data/ohlcv_window.py`` is the remaining #8 polish.
    """

    def __init__(self, db_path: str | None = None, output_path: str | None = None):
        # Vestigial: the live loader reads parquet shards directly
        # (Path("data").glob("ohlcv_*.parquet")), not DuckDB. Args kept so the
        # historical Alpha360Generator(...) constructor signature still works.
        self.db_path = db_path or "data/quant_v6_core.duckdb"
        self.output_path = output_path

    def load_live_ohlcv_window(self, tickers: Iterable[str] | None = None,
                               window_rows: int = 120) -> pl.DataFrame:
        """RAW multi-row OHLCV tail window for the V3/V4 tabular pipeline.

        The V3/V4 `pipeline.build_features` recomputes ALL of its own features
        from RAW prices, so it needs the full time series WITH the complete
        OHLCV suite (open/high/low are required by hl_range_ratio + gap_risk,
        and the multi-row window is required for FracDiff + the rolling/
        cross-sectional stats).

        Returns columns: ticker, date, open, high, low, close, volume.
        """
        raw = self._load_live_stock_window(tickers=tickers, window_rows=window_rows)
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise ValueError(
                f"load_live_ohlcv_window: OHLCV parquet tail missing {missing}; "
                f"available columns={raw.columns}")
        return raw.select(cols)

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
