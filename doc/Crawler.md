# Crawler Data Ingestion & Processing Logic Backup

This document serves as a comprehensive reference for the data ingestion, processing, and saving logic implemented in `price_macro_crawler.py`. This is intended to facilitate the migration to a new Qlib-style architecture.

## 1. DATA SOURCES

The system fetches data from multiple providers using Python libraries and specific API wrappers:

### 1.1. Vietnamese Stock Market Data (`vnstock`)
- **Universe Discovery**:
    - Function: `self.vn.stock(symbol='SSI', source=src).listing.all_symbols()`
    - Sources attempted: `VCI`, `TCBS`, `KBS` (fallback chain).
    - Logic: Fetches all tickers, filters for HOSE exchange, and ensures symbols are 3 letters long.
- **Historical Price Data (OHLCV)**:
    - Function: `self.vn.stock(symbol=ticker, source=self.source).quote.history(start=start_date, end=end_date)`
    - Default Source: `KBS`.
    - Fields: `open`, `high`, `low`, `close`, `volume`.
- **Money Market (Interbank Rates)**:
    - Function: `money_market_historical_data(symbol='ON', ...)` or `self.vn.money_market_historical_data(...)`.
    - Symbol: `ON` (Overnight Rate).

### 1.2. Global Macro Data (`yfinance`)
- **Data Points**:
    - **DXY (US Dollar Index)**: Ticker `DX-Y.NYB`.
    - **S&P 500 Index**: Ticker `^GSPC`.
    - **USD/VND Exchange Rate**: Ticker `VND=X`.
- **Processing**: Fetches daily closing prices and calculates percentage changes aligned with the Vietnamese trading calendar.

### 1.3. Local Storage (`sqlite3`)
- **Database**: `master_quant_database.db`.
- **Table**: `UNIFIED_RESEARCH_MATRIX`.

---

## 2. RAW VARIABLES DETAILED

These variables are fetched directly from sources before any secondary technical calculations:

| Variable Group | Name | Description | Source |
| :--- | :--- | :--- | :--- |
| **Identity** | `ticker` | Stock symbol (e.g., ACB, FPT) | vnstock |
| | `date` | Trading date (YYYY-MM-DD) | vnstock / yf |
| **Price (Raw)** | `open` | Opening price | vnstock |
| | `high` | Highest price | vnstock |
| | `low` | Lowest price | vnstock |
| | `close` | Closing price | vnstock |
| | `volume` | Trading volume | vnstock |
| **Macro** | `DXY` | US Dollar Index price | yfinance |
| | `SP500` | S&P 500 Index price | yfinance |
| | `USDVND` | USD/VND exchange rate | yfinance |
| | `IR` | Interbank Overnight Interest Rate | vnstock |
| **Sentiment** | `Sentiment_NLP` | Sentiment score (usually from NLP pipeline) | DB (historical) |
| | `Impact_Force` | Impact force of news/sentiment | DB (historical) |

---

## 3. CALCULATED VARIABLES (Features)

The following indicators are derived from the raw variables for use in quantitative models:

### 3.1. Technical Indicators
- **`pct`**: Daily return of adjusted close price (`adj_close`).
- **`rsi`**: Relative Strength Index (14-period).
- **`sma_50`**: Simple Moving Average (50-period).
- **`bb_width`**: Bollinger Bands Width (`(4 * std20) / ma20`).
- **`atr_pct`**: Average True Range as percentage of price (`(ATR14 / adj_close) * 100`).
- **`vol_ma20`**: 20-day Volume Moving Average (used for liquidity filtering).
- **`vol_pct`**: Daily percentage change in volume.
- **`ema_200`**: Exponential Moving Average (200-period).
- **`dist_ema200`**: Relative distance to EMA 200 (`(adj_close - ema_200) / ema_200`).
- **`MACD_hist`**: Moving Average Convergence Divergence Histogram (`MACD(12,26) - Signal(9)`).

### 3.2. Macro Indicators
- **`DXY_pct`**: Daily percentage change of DXY.
- **`SP500_pct`**: Daily percentage change of S&P 500.
- **`USDVND_pct`**: Daily percentage change of USD/VND rate.
- **`IR_pct`**: Daily percentage change of Interbank Overnight Rate.

---

## 4. SAVING LOGIC & SCHEMA

### 4.1. Data Integrity & Sync Logic
- **WAL Mode**: Uses `PRAGMA journal_mode=WAL` for concurrent read/write support.
- **Pre-sync Cleanup**:
    - Deletes rows where critical price/volume data is `NULL`.
    - Deletes existing data for the "today" date to prevent duplication on re-runs.
- **Forward Filling (ffill)**:
    - Sentiment columns (`Sentiment_NLP`, `Impact_Force`) and Macro columns are forward-filled from the last available database entry for each ticker.
    - Historical "padding" (last 5 rows) is loaded from the DB to ensure continuity.
- **Liquidity Filter**: Only tickers with `vol_ma20 >= 100,000` are committed to the research matrix.

### 4.2. Database Schema: `UNIFIED_RESEARCH_MATRIX`

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `ticker` | TEXT | Stock ticker symbol |
| `date` | TEXT | Date string (YYYY-MM-DD) |
| `open` | REAL | Raw opening price |
| `high` | REAL | Raw high price |
| `low` | REAL | Raw low price |
| `close` | REAL | Raw close price |
| `volume` | REAL | Raw trading volume |
| `pct` | REAL | Daily return |
| `vol_pct` | REAL | Volume change % |
| `ratio_event` | REAL | Event ratio (Placeholder/Metadata) |
| `adj_factor` | REAL | Adjustment factor (Placeholder/Metadata) |
| `adj_close` | REAL | Adjusted closing price |
| `adj_open` | REAL | Adjusted opening price |
| `adj_high` | REAL | Adjusted high price |
| `adj_low` | REAL | Adjusted low price |
| `DXY_pct` | REAL | % Change in DXY |
| `USDVND_pct` | REAL | % Change in USDVND |
| `SP500_pct` | REAL | % Change in S&P 500 |
| `IR_pct` | REAL | % Change in Interbank Rate |
| `Sentiment_Score_orig`| REAL | Original sentiment score |
| `Target_Buy` | REAL | Target buy flag/signal |
| `Sentiment_NLP` | REAL | NLP Sentiment Score (ffill-ed) |
| `Magnitude_NLP` | REAL | NLP Magnitude Score |
| `Impact_Force` | REAL | Calculated Impact Force (ffill-ed) |
| `rsi` | REAL | Relative Strength Index (14) |
| `sma_50` | REAL | Simple Moving Average (50) |
| `bb_width` | REAL | Bollinger Band Width |
| `atr_pct` | REAL | ATR Percentage (14) |
| `vol_ma20` | REAL | Volume Moving Average (20) |
| `ema_200` | REAL | Exponential Moving Average (200)|
| `dist_ema200` | REAL | Distance to EMA 200 |
| `MACD_hist` | REAL | MACD Histogram |
| `proba_up` | REAL | Probability signal (added dynamically) |

---
*Created by Antigravity AI for Khương - Quant System Migration Project.*
