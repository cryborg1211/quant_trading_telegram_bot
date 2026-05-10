# Quant V6 Architecture

**System:** Automated quantitative research + trading platform  
**Market:** VN100 / liquid Vietnam-listed equities  
**Version:** V6 hybrid Parquet + DuckDB + Polars + Stacking GBDT architecture  
**Production stance:** point-in-time safe, leakage-aware, artifact-driven, GPU reproducible

---

## 1. System Overview

Quant V6 is an automated quantitative trading system focused on the **VN100 universe**. Its purpose is to:

- maintain daily OHLCV data for active tickers,
- build Alpha360-style technical features,
- enrich features with macro-economic and NLP sentiment information,
- train a robust short-horizon classification model,
- generate production-ready directional probabilities,
- support portfolio and trading components under strict no-lookahead constraints.

Core flow:

```text
Per-ticker OHLCV Parquet
        |
        v
Polars Alpha360 feature generation
        |
        v
Point-in-time DuckDB macro/sentiment enrichment
        |
        v
Stacking GBDT classifier
        |
        v
Signal / portfolio / execution layer
```

Quant V6 is designed around one mandatory production principle:

> **Every feature used at inference time must be reproducible using only information available before the trading decision timestamp.**

---

## 2. Data Architecture: Hybrid Parquet + DuckDB

Quant V6 uses a hybrid storage strategy:

| layer | path | role |
|---|---|---|
| Active OHLCV files | `data/ohlcv_*.parquet` | fast daily ticker updates |
| Feature matrix | `data/alpha360_features.parquet` | model-ready Alpha360 table |
| DuckDB OLAP core | `data/quant_v6_core.duckdb` | macro, sentiment, historical archive, analytics |
| Portfolio/app state | `data/portfolio.json` + DuckDB app tables | trading state and logs |

Observed current state:

```text
data/ohlcv_*.parquet count: 100
data/alpha360_features.parquet rows: 254,978
data/alpha360_features.parquet columns: 366
DuckDB core: data/quant_v6_core.duckdb
```

---

### 2.1 Parquet Responsibility

`data/ohlcv_*.parquet` is the active price update layer.

Example:

```text
data/ohlcv_ACB.parquet
data/ohlcv_FPT.parquet
data/ohlcv_HPG.parquet
data/ohlcv_MBB.parquet
data/ohlcv_VCB.parquet
...
```

Responsibilities:

- active daily OHLCV refresh,
- ticker-isolated update path,
- fast Polars scans,
- Alpha360 feature input,
- live inference price source.

Why Parquet here:

- columnar,
- compressed,
- portable,
- fast for Polars,
- avoids DB-level write locks during ticker refresh,
- simple file-per-ticker operational model.

---

### 2.2 DuckDB Responsibility

`data/quant_v6_core.duckdb` is the analytical OLAP and archive layer.

Known tables include:

```text
stock_ohlcv
macro_daily
macro_economic_raw
sentiment_score
live_positions
trade_history
rl_mistake_logs
hist_unified_research_matrix
hist_sentiment_llm_labeled
hist_macro_features_10y
```

Responsibilities:

- OLAP queries,
- historical archive storage,
- macro-economic data,
- NLP sentiment data,
- research joins,
- audit trails,
- migration target for legacy SQLite data.

DuckDB is **not** the primary active per-ticker OHLCV update mechanism. Active V6 price updates remain in Parquet.

---

### 2.3 Why Polars + DuckDB + Parquet

```text
Parquet = durable columnar storage
Polars  = high-speed feature engineering
DuckDB  = embedded analytical SQL / OLAP
```

This stack is optimized for quant research because:

- Polars lazily scans Parquet and computes rolling/window features efficiently.
- DuckDB performs fast SQL joins across macro, sentiment, and archive tables.
- Parquet keeps active ticker updates simple and portable.
- No heavyweight server infrastructure is required.
- The research loop remains local, reproducible, and fast.

---

## 3. Production Data Integration: Point-in-Time Only

### 3.1 Lookahead Bias Policy

**Same-day macro/sentiment joins are forbidden by default.**

Incorrect pattern:

```sql
-- FORBIDDEN: same-day join can leak data unavailable at trading time
SELECT *
FROM price_features px
LEFT JOIN sentiment_features s
    ON px.date = s.date;
```

Reason: a sentiment or macro record dated `T` may be published after the trading decision for day `T`. Joining on the same date can inject future information into the model.

Mandatory rule:

> **Macro and sentiment data MUST be lagged by at least one trading day before joining with current-day price data. Current-day price row `T` may only use macro/sentiment records known at `T-1` or earlier.**

---

### 3.2 Safe T-1 Join Pattern

Preferred daily-bar join:

```sql
WITH sentiment_daily AS (
    SELECT
        CAST(date AS DATE) AS signal_date,
        AVG(sentiment_score) AS market_sentiment_score,
        AVG(sentiment_nlp) AS market_sentiment_nlp,
        AVG(impact_force) AS market_impact_force,
        COUNT(*) AS market_news_count
    FROM hist_sentiment_llm_labeled
    WHERE is_market_wide = TRUE
    GROUP BY 1
),
macro_daily_lagged AS (
    SELECT
        CAST(date AS DATE) AS signal_date,
        dxy_pct,
        usdvnd_pct,
        sp500_pct,
        sentiment_score AS macro_sentiment_score
    FROM hist_macro_features_10y
)
SELECT
    px.*,
    m.dxy_pct,
    m.usdvnd_pct,
    m.sp500_pct,
    m.macro_sentiment_score,
    s.market_sentiment_score,
    s.market_sentiment_nlp,
    s.market_impact_force,
    s.market_news_count
FROM alpha360_price_features px
LEFT JOIN macro_daily_lagged m
    ON CAST(px.date AS DATE) = m.signal_date + INTERVAL 1 DAY
LEFT JOIN sentiment_daily s
    ON CAST(px.date AS DATE) = s.signal_date + INTERVAL 1 DAY;
```

This enforces:

```text
price/features at T
    uses macro/sentiment from T-1
```

---

### 3.3 Safer Latest-Known-As-Of Pattern

When publication timestamps are available, use an as-of join:

```sql
SELECT
    px.*,
    s.market_sentiment_score,
    s.market_sentiment_nlp,
    s.market_impact_force
FROM alpha360_price_features px
LEFT JOIN sentiment_events s
    ON s.published_at < px.decision_timestamp
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY px.ticker, px.date
    ORDER BY s.published_at DESC
) = 1;
```

Production requirement:

> Prefer `published_at < decision_timestamp` over date-only joins whenever source data supports publication timestamps.

---

## 4. Feature Engineering Contract

### 4.1 Alpha360 Is the Canonical Technical Feature Engine

All live technical indicators must be generated by the Polars Alpha360 pipeline.

Canonical path:

```text
data/ohlcv_*.parquet
        |
        v
src/features/alpha360_generator.py
        |
        v
data/alpha360_features.parquet
```

### 4.2 Legacy Technical Indicators Are Forbidden in Live Inference

Hard production rule:

> **Legacy technical indicators from `hist_unified_research_matrix` are STRICTLY FORBIDDEN for the live inference pipeline. All technical features MUST be computed dynamically via the Polars `Alpha360` pipeline to ensure consistency and prevent Data Drift.**

Forbidden live-inference fields from legacy archive include:

```text
rsi
sma_50
bb_width
atr_pct
vol_ma20
ema_200
dist_ema200
MACD_hist
proba_up
Target_Buy
```

Reason:

- legacy indicators may use different formulas,
- window definitions may differ,
- adjusted/unadjusted price handling may differ,
- missing-value handling may differ,
- computation timestamp is unknown,
- reuse creates training/inference skew,
- live production can drift from backtest assumptions.

Allowed use of `hist_unified_research_matrix`:

- historical audit,
- research comparison,
- migration validation,
- offline exploratory analysis,
- non-production benchmark only.

Forbidden use:

- direct live inference features,
- direct replacement for Alpha360 technical columns,
- training data mixed with live Alpha360 features without strict feature-contract validation.

---

### 4.3 Allowed DuckDB Enrichment Fields

Allowed enrichment candidates:

```text
dxy_pct
usdvnd_pct
sp500_pct
macro_sentiment_score
market_sentiment_score
market_sentiment_nlp
market_impact_force
market_news_count
```

All must be point-in-time safe:

```text
feature_date <= price_date - 1 trading day
```

Recommended output:

```text
data/alpha360_features_enriched.parquet
```

This output must contain only:

1. Alpha360 technical features computed from active Parquet OHLCV,
2. lagged macro/sentiment features from DuckDB,
3. identifiers and labels required by training.

---

## 5. Modeling Pipeline: Stacking GBDT

Implementation:

```text
src/models/stacking_model/train_stacking.py
```

Input:

```python
DATA_PATH = Path("data/alpha360_features.parquet")
```

Current target:

```text
target_class_5d
```

Current return horizon:

```text
5 trading days
```

---

### 5.1 Shift From Deep Learning to Tree-Based Ensemble

V6 moves from a deep-learning-first baseline toward a tabular tree ensemble.

Reason:

- Alpha360 features are tabular,
- VN equity samples are noisy,
- labeled data is limited versus feature count,
- GBDTs handle non-linear feature interactions well,
- GBDTs are robust to mixed feature scales and missing-value patterns,
- stacked probability outputs are easier to audit than opaque sequence embeddings.

Architecture:

```text
Layer 1: XGBoost + LightGBM + CatBoost
Layer 2: Logistic Regression meta-model
```

---

### 5.2 Layer 1: GPU-Accelerated Base Learners

#### XGBoost

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
)
```

#### LightGBM

```python
LGBMClassifier(
    objective="multiclass",
    num_class=3,
    device="gpu",
)
```

#### CatBoost

```python
CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    task_type="GPU",
)
```

Each model emits probabilities for:

```text
0 = DOWN
1 = SIDEWAYS
2 = UP
```

---

### 5.3 Layer 2: OOF Logistic Regression Meta-Model

The meta-model is trained on out-of-fold probabilities:

```text
xgboost_p0,  xgboost_p1,  xgboost_p2
lightgbm_p0, lightgbm_p1, lightgbm_p2
catboost_p0, catboost_p1, catboost_p2
```

Current meta-model:

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=2000,
)
```

OOF design:

```text
1. Chronological train/test split.
2. TimeSeriesSplit within training.
3. Base model predicts validation fold probabilities.
4. OOF probabilities become meta-features.
5. Meta-model trains only on OOF predictions.
6. Base models refit on full train set.
7. Test set evaluated on future data only.
```

This prevents meta-model leakage.

---

## 6. Labeling Strategy: Hybrid Quantile 33-33-33

The target is based on 5-day forward return:

```python
target_return_5d = close_0.shift(-5).over(ticker) / close_0 - 1.0
```

Training thresholds:

```python
q33 = train_returns.quantile(0.3333333333)
q66 = train_returns.quantile(0.6666666667)
```

Label mapping:

```text
return <= q33  -> 0 = DOWN
return <= q66  -> 1 = SIDEWAYS
return >  q66  -> 2 = UP
```

Why this matters in Vietnam equities:

- volatile return regimes,
- liquidity shocks,
- market-wide policy sensitivity,
- long sideways periods,
- abrupt risk-on/risk-off transitions,
- severe imbalance under fixed thresholds.

The 33-33-33 scheme:

- balances training classes,
- adapts to historical volatility,
- reduces majority-class collapse,
- makes macro-F1 meaningful,
- supports relative ranking rather than fixed absolute return labels.

---

## 7. Artifact and MLOps Contract

### 7.1 Current Artifacts

Current training artifacts:

```text
models/stacking/selected_features.json
models/stacking/scaler.joblib
models/stacking/xgboost_model.joblib
models/stacking/lightgbm_model.joblib
models/stacking/catboost_model.cbm
models/stacking/meta_model.joblib
models/stacking/classification_report.json
models/stacking/confusion_matrix.json
```

### 7.2 Mandatory Quantile Threshold Artifact

Production requirement:

> The calculated quantile thresholds `q33` and `q66` MUST be exported during training to `models/stacking/quantile_thresholds.json`.

Required file:

```text
models/stacking/quantile_thresholds.json
```

Required schema:

```json
{
  "target": "target_return_5d",
  "horizon_days": 5,
  "label_mapping": {
    "0": "DOWN",
    "1": "SIDEWAYS",
    "2": "UP"
  },
  "q33_return": -0.012345,
  "q66_return": 0.018765,
  "q33_percent": -1.2345,
  "q66_percent": 1.8765,
  "train_start_date": "YYYY-MM-DD",
  "train_end_date": "YYYY-MM-DD",
  "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

Rationale:

- inference must know the exact training-time label regime,
- backtests must map probabilities to the same thresholds used in training,
- retraining must produce a versioned threshold artifact,
- production signals must not silently change semantics.

Current note:

```text
train_stacking.py currently stores threshold info inside classification_report.json.
Production requires a dedicated quantile_thresholds.json artifact.
```

---

### 7.3 MLOps Automation Roadmap

Target `gsd` workflow:

```bash
gsd data:update-ohlcv
gsd data:build-alpha360
gsd data:enrich-alpha360-point-in-time
gsd model:train-stacking
gsd model:validate-artifacts
gsd signal:generate
```

Mandatory validations:

- no same-day macro/sentiment joins,
- all technical features from Alpha360 only,
- no forbidden legacy technical columns,
- `quantile_thresholds.json` exists,
- selected feature list matches inference feature order,
- model artifacts exist,
- class mapping is stable,
- train/test split is chronological,
- inference data contains no unseen missing-column drift.

---

## 8. Infrastructure Requirements: GPU GBDT Stack

The V6 model stack is GPU-heavy.

Required GPU paths:

| model | required acceleration |
|---|---|
| XGBoost | `device="cuda"` |
| LightGBM | `device="gpu"` / OpenCL GPU build |
| CatBoost | `task_type="GPU"` |

### 8.1 Strict Environment Pinning Required

Production and reproducible research must use a pinned environment file:

```text
environment.yml
```

or:

```text
requirements.txt
```

The file must pin:

- Python version,
- `xgboost` version with CUDA support,
- `lightgbm` GPU-compatible build,
- `catboost` version,
- CUDA runtime compatibility,
- NVIDIA driver minimum version,
- `scikit-learn`,
- `polars`,
- `duckdb`,
- `joblib`,
- `numpy`.

Example environment contract:

```yaml
name: quant-v6
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.11
  - numpy
  - scikit-learn
  - polars
  - duckdb
  - joblib
  - pip
  - pip:
      - xgboost
      - lightgbm
      - catboost
```

This example is not enough for production by itself; exact CUDA/driver/build constraints must be pinned after target hardware is finalized.

---

### 8.2 Windows/Linux Build Risk

GPU GBDT packages are sensitive to:

- CUDA version mismatch,
- NVIDIA driver mismatch,
- LightGBM OpenCL build availability,
- compiler toolchains,
- Windows path/build issues,
- conda vs pip binary differences.

Production rule:

> Model training environments must be recreated from a locked dependency file before benchmark results are accepted.

CPU fallback:

```text
Allowed: local dev/debug only
Forbidden: claiming production parity without rerunning full validation
```

---

## 9. Recent SQLite to DuckDB Migration Log

Legacy source:

```text
old-data/master_quant_database.db
```

Destination:

```text
data/quant_v6_core.duckdb
```

Migration script:

```text
scripts/migrate_sqlite_to_duckdb.py
```

DuckDB SQLite scanner flow:

```sql
INSTALL sqlite;
LOAD sqlite;
ATTACH 'old-data/master_quant_database.db' AS old_sqlite (TYPE SQLITE);
```

Migrated tables:

| SQLite source | DuckDB destination | rows |
|---|---|---:|
| `UNIFIED_RESEARCH_MATRIX` | `hist_unified_research_matrix` | 871,150 |
| `sentiment_LLM_labeled` | `hist_sentiment_llm_labeled` | 6,745 |
| `macro_features_10y_macro_data` | `hist_macro_features_10y` | 3,200 |

Verified counts:

```text
hist_unified_research_matrix,871150
hist_sentiment_llm_labeled,6745
hist_macro_features_10y,3200
```

### 9.1 Migration Casting Rules

Date normalization:

```sql
COALESCE(
    TRY_CAST(date_col AS TIMESTAMP),
    try_strptime(date_col, '%a, %d %b %Y %H:%M:%S GMT')
)
```

Numeric null/NaN handling:

```sql
CASE
    WHEN CAST(col AS VARCHAR) IN ('NaN','nan','NULL','')
    THEN NULL
    ELSE TRY_CAST(col AS DOUBLE)
END
```

SQLite detach:

```sql
DETACH old_sqlite;
```

---

### 9.2 Sentiment Table Semantics

`hist_sentiment_llm_labeled` includes:

```text
date
title
sentiment_score
magnitude
reason
url
sentiment_nlp
impact_force
is_market_wide
```

Migration added:

```sql
TRUE AS is_market_wide
```

Meaning:

> Legacy sentiment rows are market-wide unless future ticker extraction assigns instrument-level relevance.

Therefore, sentiment must be aggregated at market/date level and lagged before joining.

---

## 10. Reinforcement Learning Component

DuckDB includes:

```text
rl_mistake_logs
```

Current status:

> **Reinforcement Learning (RL) is currently in the R&D/Incubation phase. The `rl_mistake_logs` table is designed to capture incorrect predictions from the Stacking ensemble, acting as a feedback loop for future RL-based dynamic portfolio sizing.**

Production constraints:

- RL must not override stacker signals in live trading yet.
- RL outputs must not alter position sizing without separate validation.
- `rl_mistake_logs` is observational feedback infrastructure.
- Future RL experiments may use this table to learn dynamic sizing, regime filters, or risk throttles.

Intended future loop:

```text
Stacking prediction
        |
        v
Realized outcome
        |
        v
Mistake classification
        |
        v
rl_mistake_logs
        |
        v
R&D RL sizing/regime model
```

---

## 11. Production Guardrails

### 11.1 Point-in-Time Data

Required:

```text
macro/sentiment feature timestamp < decision timestamp
```

Minimum fallback:

```text
feature_date <= price_date - 1 trading day
```

Forbidden:

```text
same-day date equality joins without timestamp availability proof
```

---

### 11.2 Feature Consistency

Required:

- Alpha360 computes all technical features.
- Inference feature list equals `selected_features.json`.
- Feature order is deterministic.
- Median/imputation/scaler behavior matches training.
- No legacy technical columns enter live inference.

---

### 11.3 Artifact Completeness

Required before live inference:

```text
selected_features.json
scaler.joblib
xgboost_model.joblib
lightgbm_model.joblib
catboost_model.cbm
meta_model.joblib
quantile_thresholds.json
classification_report.json
confusion_matrix.json
```

---

### 11.4 Reproducible Environment

Required:

```text
locked environment.yml or requirements.txt
GPU availability check
CUDA/driver validation
package version dump
training command log
artifact checksum
```

---

## 12. Leakage-Safe End-to-End Flow

```text
[Daily OHLCV Fetch]
        |
        v
data/ohlcv_<TICKER>.parquet
        |
        v
[Polars Alpha360 Technical Features]
        |
        v
[DuckDB Macro/Sentiment T-1 Enrichment]
        |
        v
data/alpha360_features_enriched.parquet
        |
        v
[Stacking GBDT Training / Inference]
        |
        v
[Signal Generation]
        |
        v
[Portfolio / Execution Layer]
        |
        v
[Prediction Outcome Logging]
        |
        v
rl_mistake_logs
```

---

## 13. Near-Term Roadmap

| priority | deliverable | production requirement |
|---:|---|---|
| P0 | `scripts/build_enriched_alpha360.py` | must enforce T-1/as-of joins |
| P0 | remove legacy technical features from live path | prevents data drift |
| P0 | `models/stacking/quantile_thresholds.json` export | required for inference contract |
| P0 | feature validation script | block forbidden columns + missing features |
| P1 | pinned `environment.yml` / `requirements.txt` | reproducible GPU training |
| P1 | `gsd` pipeline commands | automated workflow |
| P1 | stacking inference script | daily signal generation |
| P2 | ticker-specific sentiment extraction | improves granularity |
| P2 | RL sizing R&D notebook/pipeline | uses `rl_mistake_logs` only after validation |

---

## 14. Final Architecture Summary

Quant V6 now follows a production-safe architecture:

- **Parquet** stores active per-ticker OHLCV for fast daily updates.
- **Polars Alpha360** computes all live technical features.
- **DuckDB** stores macro, sentiment, OLAP tables, and historical archives.
- **Macro/sentiment enrichment is point-in-time safe** using T-1 or stricter as-of joins.
- **Legacy technical indicators are forbidden** in live inference to prevent data drift.
- **Stacking GBDT** combines GPU XGBoost, LightGBM, and CatBoost with an OOF Logistic Regression meta-model.
- **Quantile thresholds must be persisted** to `models/stacking/quantile_thresholds.json`.
- **GPU dependencies must be pinned** for Windows/Linux reproducibility.
- **RL remains R&D/incubation**, with `rl_mistake_logs` serving as future feedback infrastructure.

The architecture is now aligned with production trading requirements: no lookahead, no training/inference skew, strict artifacts, reproducible infrastructure, and explicit separation between live inference features and historical research archives.