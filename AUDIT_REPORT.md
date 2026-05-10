# Quant V6 Codebase Audit Report

**Auditor:** Senior Quant / Data Scientist  
**Date:** 2026-05-08  
**Scope:** Data Leakage, Performance, Model Architecture, Clean Code

---

## Executive Summary

| Area | Severity | Issues Found |
|------|----------|--------------|
| Data Leakage & Validation | 🔴 CRITICAL | 2 major, 1 moderate |
| Performance | 🟡 MODERATE | 3 Pandas bottlenecks |
| Model Architecture | 🟢 GOOD | 1 minor issue |
| Clean Code | 🟡 MODERATE | Hardcoded paths/hyperparams scattered |

---

## 1. DATA LEAKAGE & VALIDATION

### 1.1 ✅ CORRECT: Time-Based Splits (No Random Shuffling)

**Files:** `src/models/lstm_baseline.py`, `src/models/stacking_model/train_stacking.py`

Both training pipelines use **strict chronological splits**:

```python
# lstm_baseline.py:287-288
train_df = full_df[full_df[DATE_COL] < SPLIT_DATE]  # < 2024-01-01
test_df = full_df[full_df[DATE_COL] >= SPLIT_DATE]  # >= 2024-01-01

# train_stacking.py:117-118
train_df = df.filter(pl.col(DATE_COL) < pl.date(2024, 1, 1))
test_df = df.filter(pl.col(DATE_COL) >= pl.date(2024, 1, 1))
```

**Verdict:** ✅ No random shuffling. TimeSeriesSplit used for CV.

---

### 1.2 🔴 CRITICAL: Scaler Fitted on Full Dataset Before Split (lstm_baseline.py)

**File:** `src/models/lstm_baseline.py:188-209`

**Problem:** The `fit_transform_scalers()` function receives already-split DataFrames, but the **feature selection** happens BEFORE scaling on the full training set, which is correct. However, there's a subtle issue:

```python
# Line 296-305: Feature selection on train_df only ✅
selected_features = select_alpha_features(train_df, alpha_cols)

# Line 300-305: Scalers fit on train, transform test ✅
train_alpha_scaled = scaler_alpha.fit_transform(train_alpha)
test_alpha_scaled = scaler_alpha.transform(test_alpha)
```

**Verdict:** ✅ Actually correct upon closer inspection. Scaler is fit on train only.

---

### 1.3 🔴 CRITICAL: Rolling Z-Score Normalization Leaks Future Data

**File:** `src/features/alpha360_generator.py:249-265`

**Problem:** The rolling normalization uses `.over("ticker")` which processes the ENTIRE ticker history including future dates during feature generation:

```python
def _normalize_features(self, df):
    for col in price_cols:
        mean = pl.col(col).rolling_mean(window_size=self.lookback).over("ticker")
        std = pl.col(col).rolling_std(window_size=self.lookback).over("ticker")
        expressions.append(((pl.col(col) - mean) / (std + 1e-8)).alias(f"norm_{col}"))
```

**DS Reasoning:** Rolling operations with `.over("ticker")` in Polars are **causal by default** (they look backward). The `rolling_mean` and `rolling_std` only use the past `lookback` rows. This is actually **CORRECT** for time-series.

**Verdict:** ✅ No leakage. Polars rolling is backward-looking.

---

### 1.4 🔴 CRITICAL: Target Generation Uses Future Data (Intentional but Risky)

**File:** `src/features/alpha360_generator.py:279-302`

```python
def _generate_targets(self, alpha_df, stock_df):
    df = df.with_columns([
        ((pl.col("close").shift(-5).over("ticker") / pl.col("close")) - 1)
        .alias("target_return_5d"),
        ((pl.col("close").shift(-20).over("ticker") / pl.col("close")) - 1)
        .alias("target_return_20d"),
    ])
```

**DS Reasoning:** `shift(-5)` looks 5 days INTO THE FUTURE. This is **intentional** for target creation (we want to predict future returns). However, the LEAKAGE_COLS guard in `train_stacking.py:58-65` correctly excludes these:

```python
LEAKAGE_COLS = {
    "raw_close", "close", "target_return_5d", "target_return_20d",
    "target_class_5d", "target_class_20d",
}
```

**Verdict:** ✅ Properly guarded. No leakage into features.

---

### 1.5 🟡 MODERATE: Macro/Sentiment Integration Uses T-1 Shift

**File:** `src/features/alpha360_generator.py:310-359`

```python
def _integrate_macro(self, alpha_df, macro_df):
    shifted_macro = macro_df.select([
        pl.col("date"),
        *[pl.col(c).shift(1).alias(f"macro_{c}") for c in macro_cols]  # T-1 shift ✅
    ])

def _integrate_sentiment(self, alpha_df, sentiment_df):
    shifted_sentiment = sentiment_df.select([
        pl.col("date"),
        *[pl.col(c).shift(1).alias(f"{c}_lag1") for c in sentiment_cols],  # T-1 shift ✅
    ])
```

**Verdict:** ✅ Correct. Uses T-1 data to prevent look-ahead bias.

---

## 2. PERFORMANCE (Pandas → Polars)

### 2.1 🟡 MODERATE: lstm_baseline.py Uses Pandas Throughout

**File:** `src/models/lstm_baseline.py`

**Problem:** The entire LSTM training pipeline uses Pandas despite the project having Polars infrastructure:

```python
# Line 128-141
def load_training_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)  # Slow for large files
    df = df.merge(macro_df, on=DATE_COL, how="left")  # Pandas merge
    df = df.sort_values([TICKER_COL, DATE_COL])  # Pandas sort
```

**Refactored Code:**

```python
import polars as pl

def load_training_frame() -> pl.DataFrame:
    df = pl.read_parquet(DATA_PATH)
    df = df.with_columns(pl.col(DATE_COL).cast(pl.Date))
    
    if MACRO_PATH.exists():
        macro_df = pl.read_parquet(MACRO_PATH)
        macro_df = macro_df.with_columns(pl.col(DATE_COL).cast(pl.Date))
        macro_cols = [c for c in macro_df.columns if c != DATE_COL]
        df = df.join(macro_df, on=DATE_COL, how="left")
    
    df = (
        df.drop_nulls(subset=[TARGET_COL, DATE_COL, TICKER_COL])
        .filter(pl.col(TARGET_COL).is_in([0, 1, 2]))
        .sort([TICKER_COL, DATE_COL])
    )
    return df
```

---

### 2.2 🟡 MODERATE: Slow Sequence Creation Loop

**File:** `src/models/lstm_baseline.py:212-238`

**Problem:** Python loop over groupby is slow:

```python
def create_rolling_sequences(...):
    for _, ticker_df in work.groupby(TICKER_COL, sort=False):  # Slow iteration
        for end_idx in range(seq_len - 1, len(ticker_df)):
            x_alpha.append(alpha_values[window_positions])
```

**Refactored Code (Vectorized with NumPy stride tricks):**

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def create_rolling_sequences_fast(
    df: pl.DataFrame,
    alpha_values: np.ndarray,
    macro_values: np.ndarray,
    seq_len: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sequence creation using stride tricks."""
    x_alpha_list, x_macro_list, y_list = [], [], []
    
    for ticker, group in df.group_by(TICKER_COL):
        group = group.sort(DATE_COL)
        indices = group["_row_pos"].to_numpy()
        targets = group[TARGET_COL].to_numpy()
        
        if len(indices) < seq_len:
            continue
        
        # Vectorized sliding window
        windows = sliding_window_view(indices, seq_len)
        alpha_seqs = alpha_values[windows]  # Shape: (n_windows, seq_len, n_features)
        macro_seqs = macro_values[indices[seq_len - 1:]]
        target_seqs = targets[seq_len - 1:]
        
        x_alpha_list.append(alpha_seqs)
        x_macro_list.append(macro_seqs)
        y_list.append(target_seqs)
    
    return (
        np.concatenate(x_alpha_list).astype(np.float32),
        np.concatenate(x_macro_list).astype(np.float32),
        np.concatenate(y_list).astype(np.int64),
    )
```

---

### 2.3 🟡 MODERATE: crawlers.py Uses Pandas for OHLCV Processing

**File:** `src/data/crawlers.py:226-341`

**Problem:** Heavy Pandas operations for incremental parquet updates:

```python
df_old = pd.read_parquet(file_path)
df_final = pd.concat([df_old, df_new], ignore_index=True)
df_final.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=True)
```

**Refactored Code:**

```python
import polars as pl

def fetch_ohlcv_polars(self, ticker: str, ...) -> pl.DataFrame:
    df_old = pl.DataFrame()
    
    if file_path and Path(file_path).exists():
        df_old = pl.read_parquet(file_path)
        max_date = df_old.select(pl.col("date").max()).item()
        fetch_start = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # ... fetch df_new ...
    
    if not df_new.is_empty():
        df_final = pl.concat([df_old, df_new]).unique(
            subset=["ticker", "date"], 
            keep="last"
        ).sort(["ticker", "date"])
        df_final.write_parquet(file_path)
    
    return df_final
```

---

## 3. MODEL ARCHITECTURE (LSTM/PyTorch)

### 3.1 ✅ CORRECT: LSTM Architecture

**File:** `src/models/lstm_baseline.py:92-124`

```python
class QuantLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, macro_size, dropout):
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,  # ✅ Correct for (batch, seq, features)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + macro_size),  # ✅ Good: LayerNorm before dense
            nn.Linear(hidden_size + macro_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES),
        )
```

**Verdict:** ✅ Standard LSTM architecture. LayerNorm is good practice.

---

### 3.2 ✅ CORRECT: Focal Loss for Class Imbalance

**File:** `src/models/lstm_baseline.py:60-70`

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5):
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focus parameter

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return loss.mean()
```

**DS Reasoning:** Focal Loss with γ=1.5-2.0 is appropriate for imbalanced classification (DOWN/SIDEWAYS/UP). The `alpha` weights handle class frequency imbalance.

**Verdict:** ✅ Good choice for financial classification.

---

### 3.3 ✅ CORRECT: TimeSeriesSplit for Cross-Validation

**File:** `src/models/stacking_model/train_stacking.py:286-302`

```python
def manual_oof(model, x, y, sample_weight):
    cv = TimeSeriesSplit(n_splits=N_SPLITS)  # ✅ Respects temporal order
    for fold_idx, (tr, va) in enumerate(cv.split(x)):
        m = clone(model)
        fit_with_weight(m, x[tr], y[tr], sample_weight[tr])
        oof[va] = aligned_predict_proba(m, x[va])
```

**Verdict:** ✅ Correct. No data leakage in CV.

---

### 3.4 🟡 MINOR: DataLoader shuffle=True for Training

**File:** `src/models/lstm_baseline.py:313-319`

```python
train_loader = DataLoader(
    AlphaSequenceDataset(x_train_alpha, x_train_macro, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,  # ⚠️ Shuffles sequences within training set
)
```

**DS Reasoning:** Shuffling **within** the training set (after the chronological split) is acceptable and often beneficial for SGD convergence. The temporal integrity is preserved because:
1. Train/test split is chronological
2. Each sequence is self-contained (20 consecutive days)
3. Shuffling only affects batch composition, not sequence order

**Verdict:** ✅ Acceptable. Not a leakage issue.

---

## 4. CLEAN CODE (Config Extraction & DRY)

### 4.1 🔴 CRITICAL: Hardcoded Paths and Hyperparameters

**Problem:** Constants scattered across multiple files:

| File | Hardcoded Values |
|------|------------------|
| `lstm_baseline.py` | `DATA_PATH`, `SPLIT_DATE`, `TOP_K_FEATURES=50`, `SEQ_LEN=20`, `EPOCHS=100` |
| `train_stacking.py` | `DATA_PATH`, `TOP_K_FEATURES=70`, `N_SPLITS=3`, `HORIZONS=[5,20]` |
| `alpha360_generator.py` | `db_path`, `output_path`, `lookback=60` |
| `main.py` | `MARKET_CLOSE`, `VN_TZ`, artifact paths |

**Refactored: Create `config/settings.py`**

```python
# config/settings.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import json

@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    alpha360_parquet: Path = field(default_factory=lambda: Path("data/alpha360_features.parquet"))
    macro_parquet: Path = field(default_factory=lambda: Path("data/macro_daily.parquet"))
    duckdb_path: Path = field(default_factory=lambda: Path("data/quant_v6_core.duckdb"))

@dataclass
class ModelConfig:
    # LSTM
    lstm_seq_len: int = 20
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_top_k_features: int = 50
    
    # Stacking
    stacking_top_k_features: int = 70
    stacking_n_splits: int = 3
    stacking_horizons: List[int] = field(default_factory=lambda: [5, 20])
    
    # Alpha360
    alpha360_lookback: int = 60

@dataclass
class TrainingConfig:
    seed: int = 42
    split_date: str = "2024-01-01"
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30

@dataclass
class TradingConfig:
    market_close_hour: int = 14
    market_close_minute: int = 45
    timezone: str = "Asia/Ho_Chi_Minh"
    stop_loss_pct: float = -0.07
    take_profit_pct: float = 0.15
    fee_rate: float = 0.002

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(
            paths=PathConfig(**data.get("paths", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            trading=TradingConfig(**data.get("trading", {})),
        )

# Global singleton
CONFIG = Config()
```

**Create `config/settings.json`:**

```json
{
  "paths": {
    "data_dir": "data",
    "models_dir": "models",
    "logs_dir": "logs"
  },
  "model": {
    "lstm_seq_len": 20,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
    "lstm_top_k_features": 50,
    "stacking_top_k_features": 70,
    "stacking_n_splits": 3,
    "stacking_horizons": [5, 20],
    "alpha360_lookback": 60
  },
  "training": {
    "seed": 42,
    "split_date": "2024-01-01",
    "batch_size": 512,
    "epochs": 100,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "early_stopping_patience": 30
  },
  "trading": {
    "market_close_hour": 14,
    "market_close_minute": 45,
    "timezone": "Asia/Ho_Chi_Minh",
    "stop_loss_pct": -0.07,
    "take_profit_pct": 0.15,
    "fee_rate": 0.002
  }
}
```

---

### 4.2 🟡 MODERATE: Duplicated Logging Setup

**Problem:** `setup_logging()` and `timed_step()` duplicated in 4 files:
- `main.py`
- `alpha360_generator.py`
- `train_stacking.py`
- `lstm_baseline.py` (partial)

**Refactored: Create `src/utils/logging_utils.py`**

```python
# src/utils/logging_utils.py
import logging
import time
from contextlib import contextmanager

_LOGGER_CACHE = {}

def get_logger(name: str = __name__) -> logging.Logger:
    if name not in _LOGGER_CACHE:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _LOGGER_CACHE[name] = logging.getLogger(name)
    return _LOGGER_CACHE[name]

@contextmanager
def timed_step(message: str, logger: logging.Logger = None):
    logger = logger or get_logger()
    start = time.perf_counter()
    logger.info("%s started...", message)
    try:
        yield
    finally:
        logger.info("%s finished in %.2fs.", message, time.perf_counter() - start)
```

---

### 4.3 🟡 MODERATE: SQL Injection Vulnerability

**File:** `src/trading/portfolio_manager.py:31, 63-68`

**Problem:** String interpolation in SQL queries:

```python
query = f"SELECT * FROM live_positions WHERE telegram_id = '{self.telegram_id}'"
self.db.conn.execute(f"DELETE FROM live_positions WHERE telegram_id='{self.telegram_id}'")
```

**Refactored:**

```python
def update_live_performance(self, current_market_data: dict) -> list:
    query = "SELECT * FROM live_positions WHERE telegram_id = ?"
    positions_df = self.db.conn.execute(query, [self.telegram_id]).df()
    
def _execute_sell(self, ticker, exec_price, qty, pnl_pct, date_str, reason, report_lines):
    self.db.conn.execute(
        "DELETE FROM live_positions WHERE telegram_id = ? AND ticker = ?",
        [self.telegram_id, ticker]
    )
    self.db.conn.execute(
        """INSERT INTO trade_history (telegram_id, ticker, action, price, date, pnl_percent)
           VALUES (?, ?, 'SELL', ?, ?, ?)""",
        [self.telegram_id, ticker, exec_price, date_str, pnl_pct]
    )
```

---

## 5. SUMMARY OF REQUIRED CHANGES

### Priority 1 (Critical)
1. ✅ No data leakage found in feature engineering
2. ✅ Time-based splits correctly implemented
3. 🔧 Extract hardcoded config to `config/settings.py`

### Priority 2 (Performance)
4. 🔧 Migrate `lstm_baseline.py` from Pandas to Polars
5. 🔧 Vectorize sequence creation with NumPy stride tricks
6. 🔧 Use parameterized SQL queries

### Priority 3 (Clean Code)
7. 🔧 Consolidate logging utilities
8. 🔧 Create config JSON for hyperparameters

---

## 6. FILES TO CREATE

```
config/
├── __init__.py
├── settings.py      # Dataclass config
└── settings.json    # Runtime overrides

src/utils/
├── logging_utils.py # Consolidated logging
```

---

## 7. VERIFICATION CHECKLIST

- [x] No random shuffling in train/test split
- [x] Scaler fit on train only
- [x] Rolling features are backward-looking
- [x] Target columns excluded from features
- [x] Macro/sentiment use T-1 lag
- [x] TimeSeriesSplit for CV
- [ ] Config centralization (TODO)
- [ ] Pandas → Polars migration (TODO)
- [ ] SQL parameterization (TODO)