# Quant V6 — Full Project Context Dump

> **Purpose of this file:** Self-contained brain-dump for handing off to a new LLM
> session with zero prior context. Read top-to-bottom for orientation, then jump
> to the relevant section. Last updated after the TD-44/50/31 sprint.

---

## TL;DR

**Quant V6** is an automated quantitative trading platform for the **VN100 / HOSE**
(Vietnamese stock exchange) universe, ~355 tickers. It combines:

- A **dual-horizon (5d + 20d) Stacking GBDT** ensemble (XGBoost + LightGBM +
  CatBoost → logistic meta-learner) for directional class predictions (DOWN/SIDE/UP).
- **Alpha360**-style feature engineering: 6 raw OHLCV fields × 60 rolling-Z-score
  lags per ticker = 360 lag columns, plus T-1-shifted macro (DXY, S&P 500,
  USD/VND, interbank rate) and lagged daily sentiment aggregates.
- **Parallel multi-domain news scraping** (cafef, vietstock, tinnhanhchungkhoan,
  vneconomy, vietnambiz) + **Gemini 2.5 Flash** for per-ticker sentiment scoring
  and reasoning.
- A long-running **interactive Telegram bot** (`run_bot.py`) with 9 commands,
  per-user portfolio isolation, audit log, and post-mortem reviews.
- A daily **cron pipeline** (`main.py --task daily_inference`) that runs after
  market close, generates Top-3 BUY signals, manages the portfolio, and dispatches
  HTML alerts.

**Status:** production, multi-user, runs unattended overnight on a single VPS via
systemd. ~12 sprints of incremental hardening completed. Operational substrate
(log rotation, DuckDB backups, supervisor, rate limiting) is in place.
Phase-3 RL training is the next strategic milestone (currently blocked only by
adding a test suite).

---

## 1. Technology stack

| Layer | Tech |
|---|---|
| Language | Python 3.11 |
| Storage | DuckDB (single file: `data/quant_v6_core.duckdb`) + per-ticker parquet (`data/ohlcv_<TICKER>.parquet`) |
| DataFrames | Polars (training / feature engineering), pandas (crawler / bot) |
| ML | XGBoost, LightGBM, CatBoost (GPU), scikit-learn (logistic meta, scaler, splits) |
| LLM | Google Gemini 2.5 Flash via **`google-genai`** SDK (NOT the deprecated `google.generativeai`) |
| Telegram | `python-telegram-bot[job-queue]>=20.7` (v20+ async API) |
| Market data | `vnstock` 4.x (HOSE OHLCV), `yfinance` (US macro) |
| News | `gnews`, `googlenewsdecoder`, `aiohttp` (async per-domain scraping), `BeautifulSoup4`, `feedparser` (RSS) |
| HTTP | `cloudscraper` (for SBV / TradingEconomics fallbacks), `requests` (retry-capable Session) |
| Persistence | `joblib` (sklearn artifacts), parquet (features), `.cbm` (CatBoost) |
| Process supervisor | `systemd` (unit at `deploy/quant-v6-bot.service`) |

Python 3.11 is required (uses 3.10+ union syntax `str | None`, `match`).

There is **no `requirements.txt` at project root** (open tech-debt). Install set
documented in `README.md` and reproduced at the bottom of this file.

---

## 2. Repository layout

```text
stock_price_v3/
├── main.py                       # CLI: --task crawl_hose|build_alpha360|full_pipeline|daily_inference
├── run_bot.py                    # Thin wrapper → src.utils.telegram_bot.main()
├── README.md                     # Project entry point
├── CONTEXT_DUMP.md               # ← this file
├── AUDIT_REPORT.md               # Historical tech-debt audits
├── .env                          # Live secrets (gitignored)
├── .env.example                  # Template for required env vars
├── .gitignore                    # Created during Phase 1 cleanup
│
├── config/
│   ├── settings.py               # CONFIG dataclasses (Config.from_json)
│   └── settings.json             # User overrides (optional)
│
├── src/
│   ├── data/
│   │   ├── db_engine.py          # DuckDBEngine singleton, schema init, migrations
│   │   └── crawlers.py           # StockCrawler (vnstock) + MacroCrawler + MacroProvider
│   │
│   ├── features/
│   │   └── alpha360_generator.py # Polars rolling-Z-score lag matrix
│   │
│   ├── models/
│   │   ├── stacking_model/
│   │   │   └── train_stacking.py # XGB+LGBM+CatBoost → logistic meta, 5d + 20d
│   │   └── quant_agent_arbitrator.py  # Parallel news scrape + Gemini + Top-6→3 selection
│   │
│   ├── crawlers/
│   │   └── sentiment_crawler.py  # Historical LLM-labelled sentiment (RSS + GNews + Gemini)
│   │
│   ├── trading/
│   │   └── portfolio_manager.py  # Unified `portfolio` table; cron + bot users share
│   │
│   ├── rl/
│   │   └── trading_env.py        # Phase-3 RL env scaffolding (incomplete)
│   │
│   └── utils/
│       ├── telegram_bot.py       # ~950 lines: handlers + rate limiter + bootstrap
│       ├── telegram_alerter.py   # One-shot HTML push alerter (cron path)
│       ├── audit_evaluator.py    # /audit_weekly + /audit_monthly post-mortem engine
│       ├── logging_utils.py      # setup_rotating_logging + get_crawler_error_logger
│       └── version.py            # get_version() — git SHA / VERSION file
│
├── data/
│   ├── quant_v6_core.duckdb      # ⚠ SINGLE SOURCE OF TRUTH for all relational data
│   ├── ohlcv_<TICKER>.parquet    # Per-ticker historical OHLCV (~355 files)
│   ├── macro_daily.parquet       # Wide-format global + VN macro
│   └── alpha360_features.parquet # Full training matrix (~360 lag cols × 10y × 355 tickers)
│
├── models/stacking/
│   ├── 5d/
│   │   ├── selected_features.json
│   │   ├── scaler.joblib
│   │   ├── xgboost_model.joblib
│   │   ├── lightgbm_model.joblib
│   │   ├── catboost_model.cbm
│   │   ├── meta_model.joblib                # Logistic regression
│   │   ├── classification_report.json
│   │   ├── confusion_matrix.json
│   │   └── quantile_thresholds.json
│   └── 20d/   (same shape)
│
├── logs/                         # RotatingFileHandler: quant_v6.log + crawler_errors.txt (10 MiB × 5)
├── backups/                      # Daily DuckDB cp backups, 14-day retention
│
├── deploy/
│   └── quant-v6-bot.service      # systemd unit
│
├── scripts/
│   ├── backup_db.sh              # Cron-friendly daily DB backup
│   ├── cleanup_legacy_rl_stubs.py # One-off TD-44 purge
│   └── migrate_sqlite_to_duckdb.py # Historical migration helper
│
├── docs/
│   └── ARCHITECTURE_V6.md        # 983-line system design (predates the bot work; stale on bot/audit/RL)
│
├── doc/
│   ├── .antigravity_rules.md     # Internal planning
│   ├── .plan.md                  # Internal planning
│   └── Crawler.md                # Crawler-specific deeper dive
│
├── scratch/                      # Throwaway exploration scripts (not in production paths)
├── old-data/                     # Legacy V2 codebase, kept for reference (open tech debt: delete)
└── TradingAgents-main/           # Vendored upstream repo (open tech debt: pin or remove)
```

---

## 3. DuckDB schema — all 9 tables

Single file: **`data/quant_v6_core.duckdb`**. All tables created/migrated at
`DuckDBEngine.__init__` time. Schema migrations are idempotent (ALTER TABLE ADD
COLUMN with information_schema check).

| Table | Columns | PK | Purpose |
|---|---|---|---|
| **`stock_ohlcv`** | ticker VARCHAR, date DATE, open/high/low/close/volume/adj_close DOUBLE | (ticker, date) | Daily HOSE OHLCV. ~10 years of history. Source: `vnstock.Trading` |
| **`macro_daily`** | date DATE, dxy_close, sp500_close, usd_vnd, interbank_on_rate, **vnibor**, **inflation_yoy** DOUBLE | (date) | Wide-format global + VN macro. `vnibor` and `inflation_yoy` columns exist but are **unpopulated** (vnstock 4.x removed money_market API; TradingEconomics DNS-blocked from VN ISPs) |
| **`sentiment_score`** | ticker VARCHAR, date DATE, sentiment_nlp DOUBLE, impact_force DOUBLE | (ticker, date) | Aggregated daily sentiment per ticker (legacy, used by Alpha360 lag features) |
| **`macro_economic_raw`** | date DATE, indicator_name VARCHAR, value DOUBLE | (date, indicator_name) | Long-format historical macro (VN_CPI_MONTHLY, VN_DEPOSIT_RATE_12M) from TradingEconomics. Currently orphaned — not joined into Alpha360 |
| **`live_positions`** ⚠ DEPRECATED | telegram_id VARCHAR, ticker VARCHAR, quantity INT, entry_price DOUBLE, entry_date DATE | (telegram_id, ticker) | Old cron-trade table. **No writes post-TD-33**. Kept for rollback safety. |
| **`trade_history`** | id INTEGER (seq), telegram_id VARCHAR, ticker VARCHAR, action VARCHAR ('BUY'|'SELL'), price DOUBLE, date DATE, pnl_percent DOUBLE | (id) | Append-only trade log. Column name `telegram_id` is now a misnomer — stores `user_id` (which may be `'cron'`). |
| **`rl_mistake_logs`** | ticker VARCHAR, predicted_date DATE, predicted_action VARCHAR, actual_t5_outcome DOUBLE, features_snapshot VARCHAR | none | Phase-3 RL training data. Two-phase logger (TD-05 fix): INSERT at T0 with `actual_t5_outcome = NULL`, UPDATE at T+5 with real return from `stock_ohlcv`. Table name is now a misnomer — holds ALL high-confidence predictions, not just mistakes. |
| **`portfolio`** | user_id VARCHAR, ticker VARCHAR, volume INTEGER, price DOUBLE, added_at TIMESTAMP | none | **Unified portfolio** (TD-33). Bot `/add` writes with telegram user_id; cron `PortfolioManager._execute_buy` writes with `user_id='cron'`. Bot `/suggest_sell` filters by user_id (multi-user isolation). |
| **`audit_log`** | user_id VARCHAR, command VARCHAR, ticker VARCHAR, details VARCHAR, timestamp TIMESTAMP | none (append-only) | Per-user command audit. Every `/verify`, `/add`, `/remove`, `/suggest_buy`, `/suggest_sell`, `/audit_*` invocation. Used by `/audit_weekly` to find auditable history. |

Auxiliary table outside the engine's schema init:
| Table | Created by | Purpose |
|---|---|---|
| `hist_sentiment_llm_labeled` | `sentiment_crawler.py::SentimentCrawler._append_rows` | Long-form daily news sentiment scored by Gemini. Aggregated into Alpha360 via `_load_sentiment_since`. |

**DuckDB connection rule:** ALL callers must use bare `duckdb.connect(path)` —
no `read_only=True`, no extra `config={}`. DuckDB refuses mismatched-config
connections to the same file in the same process. (Bug class: see TD audit
history.) `DuckDBEngine` is a thread-safe singleton; secondary modules
(`alpha360_generator`, `sentiment_crawler`) open their own connections but
with identical config.

---

## 4. Telegram bot — all 9 commands

Run as `python run_bot.py`. Long-running polling service. Required env:
`TELEGRAM_BOT_TOKEN`. Replies use `ParseMode.HTML` everywhere (Markdown is
forbidden — VN news contains too many `_*[]` to be safe).

| Command | Action | Audit log | Rate-limit | Wait msg | Heavy? |
|---|---|---|---|---|---|
| `/help`, `/start` | Show menu | no | no | no | no |
| `/suggest_buy` | Run full pipeline → Top-3 BUY signals | yes (`suggest_buy`) | **30s per user** | yes | ~1–2 min |
| `/suggest_sell` | BÁN/HOLD per ticker in user's portfolio | yes | no | yes | ~10–30 s |
| `/verify <ticker>` | Ad-hoc 5d Quant + LLM verdict on one ticker | yes (`verify`, ticker=) | **30s per user** | yes | ~3–8 s |
| `/add <ticker> <vol> <price>` | INSERT into `portfolio` | yes (`add`, ticker=, details=) | no | no | <100 ms |
| `/remove <ticker>` | DELETE from `portfolio` (matching user_id) | yes (`remove`, ticker=) | no | no | <100 ms |
| `/audit_weekly` | Post-mortem of last 7d `/verify`+`/add` (% return + Gemini explanation) | yes (`audit_weekly`) | no (TD-53 open) | yes | ~10–30 s |
| `/audit_monthly` | Same, 30d | yes | no (TD-53 open) | yes | ~10–30 s |
| `/news` | 20 most recent RSS items from cafef/vietstock/tnck | no | no | no | ~2–5 s |

**Hyphen-form fallback:** `/suggest-buy` and `/suggest-sell` are routed via
`MessageHandler` regex since Telegram's `bot_command` entity parser doesn't
accept hyphens (so `CommandHandler("suggest-buy", ...)` would never fire).

**HTML safety:** Every dynamic field (LLM output, ticker, URL, sentiment text)
passes through `html.escape()` before assembly. Structural tags (`<b>`, `<i>`,
`<code>`, `<a href="...">`) are constants. `disable_web_page_preview=True` on
`/news` to avoid 20 link cards.

**Telegram chunking:** Reports over 4000 chars split via `_split_html_report()`
on the `══════` visual separator (used by `_build_combined_report`,
`_build_sell_hold_report`, and the verify/audit report builders). `/news` uses
`_chunk_lines()` (no separator — packs lines under 4000 chars).

---

## 5. CLI tasks — `python main.py --task <name>`

| Task | Purpose | When | Notes |
|---|---|---|---|
| `daily_inference` *(default)* | Live inference: load features → 5d+20d predict → liquidity filter → Top-6 candidates → arbitrator → Top-3 sentiment-ranked → dispatch | After market close (16:00 ICT) | Reads parquet + DuckDB; no crawling |
| `crawl_hose` | Crawl all HOSE OHLCV from `vnstock` | Cold-start or weekly refresh | 15:00 ICT time guard blocks intraday; bypass with `--force-crawl` |
| `build_alpha360` | Full rebuild of `data/alpha360_features.parquet` | After a historical re-crawl | ~5–10 min (reads all parquet files) |
| `full_pipeline` | crawl_hose → build_alpha360 → daily_inference | Weekly retrain | Most expensive path |

Options: `--start-date`, `--end-date`, `--data-dir`, `--window-rows N` (per-ticker
tail for live inference, default 120), `--max-candidates N` (Top-N pool sent to
arbitrator, default 6), `--force-crawl` (bypass 15:00 guard).

The whole `main()` is wrapped in a `try/except` that catches every `Exception`
(not `KeyboardInterrupt`/`SystemExit`) and dispatches a Telegram crash alert with:
- `🔖 Version: <git SHA>` (TD-31)
- Exception type + message
- Last 1500 chars of traceback in `<pre>`
- Then re-raises so cron sees a non-zero exit code.

---

## 6. Configuration

### `.env` (runtime secrets — template in `.env.example`)

```
TELEGRAM_BOT_TOKEN=...        # bot token from @BotFather
TELEGRAM_CHAT_ID=123,456      # comma-separated — broadcast for cron alerts
GEMINI_API_KEY=...            # from https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.5-flash # optional override
```

### `config/settings.py` — typed dataclasses

```python
@dataclass
class PathConfig:        data_dir, models_dir, logs_dir, alpha360_parquet, macro_parquet, duckdb_path
@dataclass
class ModelConfig:       lstm_seq_len=20, lstm_top_k_features=50,
                          stacking_top_k_features=70, stacking_n_splits=3,
                          stacking_horizons=[5,20], stacking_baseline_macro_f1=0.20,
                          alpha360_lookback=60
@dataclass
class TrainingConfig:    seed=42, split_date="2024-01-01", batch_size=512,
                          epochs=100, learning_rate=1e-4, weight_decay=1e-4,
                          early_stopping_patience=30
@dataclass
class TradingConfig:     stop_loss_pct=-0.07, take_profit_pct=0.15,
                          fee_rate=0.002, virtual_allocation_per_ticker=10_000_000,
                          default_telegram_id="default_user"  # legacy, mostly unused now
@dataclass
class CrawlerConfig:     stock_start_date="2016-01-01", macro_start_date="2014-01-01",
                          throttle_min_interval_seconds=4.25,
                          rate_limit_cooldown_seconds=75,
                          request_retry_total=3, request_backoff_factor=1.5
@dataclass
class SentimentConfig:   gemini_model="models/gemini-2.5-flash",
                          rss_lookback_weekday_days=1, rss_lookback_monday_days=3,
                          max_tickers=30, gnews_max_results=8,
                          gnews_sleep_seconds=1.25, article_char_limit=4000
```

`CONFIG = Config.from_json("config/settings.json")` at module load. Missing
`settings.json` → defaults. Many magic numbers still inline (open tech debt TD-15).

---

## 7. Data flow

### 7.1 Daily cron — `main.py --task daily_inference`

```text
1. main() — wrap whole body in try/except + crash alerter
2. Set up rotating logger (TD-26)
3. Resolve version (TD-31)
4. Log startup banner: "Quant V6 starting | task=daily_inference pid=... version=..."

5. Alpha360Generator.build_live_features(window_rows=120)
   • Reads data/ohlcv_*.parquet tails (one per ticker)
   • Preprocesses (VWAP), normalizes (60-day rolling Z-score per ticker),
     generates 60 lag columns × 6 features
   • Loads macro_daily + sentiment from DuckDB
   • T-1 shifts macro + sentiment to prevent leakage
   • Forward-fills macro/sentiment across business days
   • Returns latest row per ticker as Polars frame

6. predict_stacking_horizon(latest_df, 5)
   • Loads models/stacking/5d/{xgboost, lightgbm, catboost, meta_model}.joblib + scaler + selected_features
   • Computes p_down, p_side, p_up per ticker
7. predict_stacking_horizon(latest_df, 20) — same for 20d

8. Liquidity filter: ADDV_20d ≥ 15_000_000_000 VND (close × volume × 1000-if-needed)
9. Top-6 candidate pool: ticker filter ∩ liquid, sorted by 5d UP probability DESC

10. evaluate_trades_batch(horizon_predictions, candidate_tickers):
    • scrape_centralized_news(target_tickers=[...])  ← parallel × 5 domains
    • map_tickers_to_news() — dedupe, regex-extract tickers from article body
    • get_batch_sentiment_scores() — single Gemini call with all candidate news
    • For each ticker: make_final_decision(5d_probs, 20d_probs, sentiment_score)
      Returns 0/1/2 (SELL/HOLD/BUY) with safety override: sentiment < -0.5 vetoes 5d BUY

11. Top-3 selection: from candidate_tickers WHERE decision == 2,
    sort by (sentiment_score DESC, 5d_p_up DESC), take 3.

12. run_trade_execution():
    • PortfolioManager.update_live_performance() — hard SL/TP enforcement
    • PortfolioManager.process_daily_trades(top_buy_signals, next_day_open_prices)
      = SELL holdings predicted DOWN by model, BUY new signals not already held
    • _log_rl_predictions() — INSERT per high-confidence UP prediction (NULL outcome)
    • _backfill_rl_outcomes() — UPDATE old NULL rows with real T+5 return
    • For each dispatched ticker: TelegramBot.send_signal_alert(signal_data)
      = HTML per-ticker message → POST to api.telegram.org/bot.../sendMessage
      for every chat_id in TELEGRAM_CHAT_ID

13. Return combined HTML report (used by /suggest_buy bot path; broadcast=False suppresses cron alerts)
14. Log completion + duration
```

### 7.2 Bot `/suggest_buy` — on-demand

Same pipeline as the cron, with one critical difference:

```python
# In telegram_bot.py::suggest_buy_command:
report_html = await asyncio.to_thread(daily_inference, broadcast=False)
                                                       ^^^^^^^^^^^^^^^
# broadcast=False ⇒ TelegramBot.send_signal_alert pushes ARE SUPPRESSED.
# Only the bot reply (this user's chat) gets the report. Prevents duplicate
# spam to TELEGRAM_CHAT_ID recipients when a user manually triggers.
```

Rate-limited at 30 s per user (TD-30). `asyncio.to_thread` means the bot's
event loop stays responsive to `/help`, `/news`, etc. during the 1–2 minute
inference.

### 7.3 Bot `/verify <ticker>` — single-ticker ad-hoc

```text
1. Validate ticker regex /^[A-Z0-9]{2,6}$/, rate-limit check, audit-log
2. asyncio.to_thread(verify_single_ticker, ticker)
3. verify_single_ticker():
   a. Alpha360Generator.build_live_features(tickers=[ticker], window_rows=120)
      ← Filters parquet glob to ONE file → 350× faster than full universe load
   b. predict_stacking_horizon(latest_df, 5) and (latest_df, 20)
   c. evaluate_trades_batch({"5d": ..., "20d": ...}, [ticker])
      ← News scrape + Gemini + decision for THIS ticker only
   d. _get_live_exec_prices(latest_df, [ticker])
   e. _build_verify_report() → HTML with:
      • Price
      • 5d distribution (UP=X%, SIDE=Y%, DOWN=Z%) + dominant label
      • 20d compact label
      • Sentiment status + score + LLM reasoning
      • Verdict (BUY/HOLD/SELL from arbitrator)
      • Top 3 source URLs
4. Bot edits wait-msg in place (or chunks if >4000 chars)
```

### 7.4 Bot `/audit_weekly` / `/audit_monthly` — post-mortem

```text
1. SQL: SELECT DISTINCT ticker, MIN(timestamp) FROM audit_log
        WHERE user_id=? AND command IN ('verify','add','suggest_buy')
              AND ticker IS NOT NULL AND timestamp >= now - INTERVAL N DAY
        GROUP BY ticker ORDER BY first_ts ASC
2. For each ticker (capped at 10):
   a. t0_close = SELECT close FROM stock_ohlcv WHERE ticker=? AND date<=? ORDER DESC LIMIT 1
   b. t_now_close = SELECT close FROM stock_ohlcv WHERE ticker=? ORDER date DESC LIMIT 1
   c. pct = (t_now - t0) / t0 * 100
   d. _explain_move(ticker, days, pct):
      • scrape_centralized_news([ticker])  ← 5-domain parallel
      • Build Vietnamese Gemini prompt with last 5 headlines
      • Gemini returns 1-3 sentence Earnings/Macro/Sentiment explanation
3. Build HTML report:
   "📊 BÁO CÁO HẬU KIỂM (N NGÀY QUA)"
   ══════════════════════
   • Mã: HPG
   • Thực tế: 🟢 TĂNG +4.6%
   • Nguyên nhân (AI): Earnings: ...
```

Coverage limitation: `/suggest_buy` rows in `audit_log` have no per-ticker
breakdown (only timestamp + user). Future enhancement: write one row per
dispatched signal so they appear in post-mortem.

---

## 8. Eight architectural constraints (the "must never violate" list)

From the original system design — every PR must respect these:

1. **Point-in-time safe.** All macro features T-1 shifted. All news features
   `_lag1`. No same-day close in features for same-day labels.
2. **Liquidity filter.** ADDV_20d ≥ 15 B VND threshold. Below this, ticker is
   excluded from candidate pool (mid/small caps don't get traded by signal).
3. **Top-6 → Top-3 funnel.** Quant Top-6 by 5d UP prob → arbitrator scores
   sentiment per ticker → sort by (sentiment DESC, quant DESC) → Top-3 dispatch.
4. **Sentiment safety override.** If Gemini sentiment < -0.5, vetoes any 5d BUY
   regardless of model confidence. Class 1 (HOLD) returned instead.
5. **Quantile-based labels.** UP/SIDE/DOWN labels computed from train-set
   q33/q66 of 5d (and 20d) return distribution. NOT fixed thresholds. Stored in
   `models/stacking/{5d|20d}/quantile_thresholds.json`.
6. **Per-ticker Z-score normalization.** Alpha360 lag features computed via
   60-day rolling mean/std PER TICKER. Cross-ticker normalization is forbidden
   (would leak future market regime).
7. **Macro raw, not normalized.** Interest rates and CPI YoY are raw percentages,
   NOT Z-scored. They live on stable mean-reverting scales; Z-score would
   obscure the regime signal.
8. **No `read_only=True` on DuckDB connections.** DuckDB refuses mismatched-config
   connections to the same file in one process. Singleton engine + secondary
   modules all use bare `duckdb.connect(path)`.

---

## 9. Critical design decisions worth knowing

| Decision | Rationale | Location |
|---|---|---|
| **Stacking GBDT, not deep learning** | 10-year VN equity data is ~900k samples × 360 features. Stacking outperforms LSTM here; XGB/LGBM/CatBoost cover different non-linearities; logistic meta is robust. | `src/models/stacking_model/train_stacking.py` |
| **Polars, not pandas, for Alpha360** | 360 lag columns × 350 tickers × 10y → polars is 10–100× faster for `rolling().over("ticker")` partitioned operations. | `src/features/alpha360_generator.py` |
| **DuckDB, not Postgres** | Single-file embedded, zero ops overhead, columnar (fast macro joins), supports parquet directly via `pl.scan_parquet`. Multi-process write contention exists but bot is single-process. | `src/data/db_engine.py` |
| **15:00 ICT crawl guard** | VN market closes at 15:00; the daily candle is not finalized until then. Crawling mid-day pollutes training data and double-writes the in-progress candle. | `main.py::is_crawl_allowed()` |
| **45 s per-ticker circuit breaker** | vnstock has 30s read timeout but no wall-clock cap across retries. One stuck ticker stalled overnight crawls 2+ min in production. `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=45)` plus `shutdown(wait=False)` to release orphan threads. | `src/data/crawlers.py::_fetch_ohlcv_with_timeout` |
| **5-domain parallel news scraper** | Originally 2 domains (cafef + vietstock), sequential. Cafef dominates volume so the Top-3 cap-by-domain was 100% cafef-biased. Now 5 domains in `asyncio.gather`, brand-prefix-stripped title fingerprint dedup, diversity-preserving selector picks 1 article per host before allowing repeats. | `src/models/quant_agent_arbitrator.py::_scrape_news_async` |
| **HTML, not Markdown, for Telegram** | Markdown v1 silently breaks on news bodies containing `_*[]`. HTML mode requires only `<>&` escaping, and `html.escape()` already handles that. | `src/utils/telegram_bot.py`, `src/utils/telegram_alerter.py` |
| **`asyncio.to_thread` for heavy commands** | `daily_inference` is sync + CPU-bound (joblib, XGBoost, Polars). Running in the bot's event loop would freeze `/help` for 1–2 min. `to_thread` offloads to the default thread pool. | `src/utils/telegram_bot.py` |
| **Per-(user, command) rate limiter in-memory** | `dict[tuple[str,str], float]` with one `threading.Lock`. Restart-loss is fine (bot restart effectively expires cooldowns). 30 s window. | `src/utils/telegram_bot.py::_check_and_record_rate_limit` |
| **Two-phase RL outcome logging** | At T0 INSERT with NULL outcome (can't know yet). At T+5+ UPDATE with `(t5_close - t0_close)/t0_close` from `stock_ohlcv`. Both phases run every daily_inference. | `main.py::_log_rl_predictions` + `_backfill_rl_outcomes` |
| **Unified `portfolio` table (TD-33)** | Previously: bot wrote to `portfolio`, cron wrote to `live_positions`, no sync. Now: both write to `portfolio` with `user_id` (`"cron"` for cron, `"<telegram_id>"` for users). Multi-user isolation enforced at SQL level. | `src/trading/portfolio_manager.py` |
| **`google-genai`, not `google.generativeai`** | The old SDK was deprecated and emitted FutureWarnings; Google sunsetted it. New SDK: `from google import genai; client = genai.Client(api_key=...); client.models.generate_content(...)`. | `src/models/quant_agent_arbitrator.py`, `src/crawlers/sentiment_crawler.py`, `src/utils/audit_evaluator.py` |
| **Crash alerts with traceback + version stamp (TD-09 + TD-31)** | Last 1500 chars of `traceback.format_exc()` in `<pre>` tag, prefixed with git SHA. Re-raises so cron sees non-zero exit. Wrapped in inner try/except so alerter failure can't mask the original crash. | `main.py::_send_crash_alert` |

---

## 10. Sprint history (chronological)

Recent sprints in conversation order:

1. **Onboarding** — architecture read + 8-constraint acknowledgement.
2. **Gap 3 (Liquidity filter)** — ADDV ≥ 15 B VND in `daily_inference` before candidate selection. Required fixing `_generate_lags` to preserve `volume` column.
3. **Gap 4 (Remove general queries)** — Arbitrator news scraper used to issue 6 general market queries (e.g., `"thị trường chứng khoán"`); now strictly `"{ticker}" site:{domain}`.
4. **Gap 5 (Pre-fetch cap)** — Enforce 3 URL/ticker cap BEFORE `AsyncNewsScraper.fetch_many`, not after.
5. **Feature humanization + missing source URLs** — Vietnamese labels for macro/lag features; ground-truth URL tracker (`ticker_urls_dict`) so empty Gemini source_urls field doesn't lose attribution.
6. **Top-6 → Top-3 sentiment filter** — Pool of 6 candidates, arbitrator scores sentiment, Top-3 selected by (sentiment DESC, quant DESC).
7. **Tech debt audit #1** — 24 items identified (TD-01 through TD-24, plus TD-25 macro).
8. **Phase 1 critical fixes** — `.gitignore`, `.env.example`, logging conversion (`print()` → `logging.LOGGER.info()`), removed dead `_format_source_urls`, updated `--max-candidates` help text, split DuckDB multi-statement execute, `train_stacking.py` split-date from CONFIG, renamed `lstm_class` → `model_class`.
9. **`google-genai` SDK migration** — Critical: arbitrator was crashing with `AttributeError`. Migrated arbitrator + sentiment_crawler. (Phase A in original numbering.)
10. **Telegram bot Phase 1** — Framework with `python-telegram-bot[job-queue]>=20.7`, `/help` + `/start`, idempotent setup, error handler.
11. **Bot Phase 2** — HTML parse mode everywhere, `/suggest_buy` with `asyncio.to_thread` + `broadcast=False`, refactored `daily_inference` to return HTML string.
12. **Bot Phase 3** — `/add`, `/remove`, `/suggest_sell`, `/news`. `portfolio` table created with `user_id` column. Multi-user isolation.
13. **Multi-user portfolio refinement** — Schema migration for existing `portfolio` table (add `user_id` if missing, tag legacy rows `'legacy'`).
14. **Macro feature attempt** — Added `vnibor` and `inflation_yoy` columns to `macro_daily`. Tried to populate via vnstock 4.x (API removed), then TradingEconomics (DNS-blocked from VN ISPs), then SBV (Liferay/React requires headless browser). **Shelved as TD-25.**
15. **Stability sprint** — TD-09 global crash alerter (Telegram-dispatched HTML with traceback). TD-12 45 s per-ticker circuit breaker via `ThreadPoolExecutor`.
16. **`/verify` command** — Single-ticker ad-hoc 5d + 20d + sentiment + verdict report.
17. **News scraper diversity** — Expanded from 2 → 5 VN financial portals. Parallel async via `asyncio.gather` + thread pool wrapping sync `GNews.get_news`. Brand-prefix-stripped title fingerprint dedup. Domain-diversity-preserving cap-of-3.
18. **Audit log** — `audit_log` table + `log_user_action` + `_audit_log_async` (writes wrapped in `asyncio.to_thread`).
19. **DuckDB crash hotfix** — `read_only=True` connections in `alpha360_generator` and `sentiment_crawler` crashed against the engine's read-write connection. Stripped `read_only=True` from all secondary modules; hardened singleton init under lock.
20. **15:00 time guard refresh** — Raised from 14:45 to 15:00 ICT; new log format matching ops spec.
21. **Tech debt audit #2** — Status of 24 items + 5 new TD-26 through TD-30 critical operational risks.
22. **Phase A safety net** — TD-26 log rotation (`RotatingFileHandler` 10 MiB × 5), TD-27 daily DuckDB backup (`scripts/backup_db.sh` + 14-day retention), TD-28 systemd unit (`deploy/quant-v6-bot.service`), TD-30 rate limiter (30 s per-user on `/suggest_buy` + `/verify`).
23. **Post-mortem audit** — `/audit_weekly` and `/audit_monthly`. `src/utils/audit_evaluator.py` with `run_post_mortem(user_id, days)`. Multi-user SQL filter, Gemini-based "explain the move" prompt.
24. **TD-33 + TD-05** — Portfolio table merge (deprecate `live_positions`, unified `portfolio` with `user_id='cron'` for automated path). RL outcome backfill (two-phase: T0 INSERT NULL + T+5 UPDATE with real return).
25. **`README.md`** — Project entry point.
26. **TD-44 + TD-50 + TD-31** — Purge legacy `-0.05` stub rows (`scripts/cleanup_legacy_rl_stubs.py`). Lock RL writes under `db._audit_lock`. Version stamp in logs + crash alerts via `src/utils/version.py`.

---

## 11. Open tech debt (post-current sprint)

Latest audit produced ~22 open items. Highest-priority remaining:

### Critical (do this week)
| ID | Item | Effort | Risk |
|---|---|---|---|
| **TD-04** | `DuckDBEngine.query()` accepts raw SQL — public method, no parameterization. No production callers but the surface exists. | 15 min | injection |
| **TD-29** | Crash alerter has no fallback channel — if Telegram is rate-limited/down during a crash, the alert silently fails. SMTP fallback ~30 lines. | 30 min | silent crashes |

### High (this sprint)
| ID | Item | Effort | Note |
|---|---|---|---|
| **TD-17** | **Zero test coverage.** ~4000 lines of new code shipped without a single regression test. | 4–6 h | Highest long-term leverage |
| **TD-49** | `_RL_HORIZON_DAYS = 5` hardcoded — should derive from same CONFIG as model | 5 min | Couple to `RETURN_COLS` |
| **TD-46/47** | Retention crons for `rl_mistake_logs` (365 d) + `audit_log` (180 d) | 30 min | Unbounded growth |
| **TD-53** | Add 30 s rate-limit gate to `/audit_weekly` + `/audit_monthly` | 10 min | Gemini quota risk |
| **TD-32** | No CI / pre-commit | 2 h | GitHub Actions |
| **TD-39** | `docs/RUNBOOK.md` doesn't exist (install steps live in systemd-unit comments only) | 1 h | Sunday-night ops |

### Medium
| ID | Item | Effort |
|---|---|---|
| **TD-19** | Delete duplicate `setup_logging` / `timed_step` in 3 modules (canonical now in `logging_utils.py`) | 1 h |
| **TD-23** | `DuckDBEngine.__init__` default path duplicates `CONFIG.paths.duckdb_path` | 15 min |
| **TD-18** | `train_stacking.py` `DATA_PATH`/`ARTIFACT_ROOT` duplicate `CONFIG.paths` | 15 min |
| **TD-15** | Magic numbers (`-0.5`, `0.5`, `0.6`, `15_000_000_000`, etc.) → CONFIG | 2 h |
| **TD-52** | `_log_rl_predictions` / `_backfill_rl_outcomes` belong in `src/rl/rl_logger.py` | 1 h |
| **TD-37** | `evaluate_trades_batch` over-iterates all predictions vs candidate pool | 1 h |
| **TD-38** | No bot health-check endpoint (deadlock detection) | 3 h |
| **TD-14** | Delete `old-data/`, decide on `TradingAgents-main/` | 30 min |
| **TD-48** | `audit_evaluator._explain_move` builds its own Gemini client (duplicates arbitrator) | 1 h |
| **TD-45** | Historical `live_positions` rows → migrate to `portfolio` with `user_id='cron'` | 30 min |

### Low
| ID | Item | Note |
|---|---|---|
| **TD-35** | 3 near-duplicate text splitters in `telegram_bot.py` | Consolidate |
| **TD-36** | Lazy `from main import …` inside async handlers | Tradeoff |
| **TD-40** | `telegram_bot.py` is 950+ lines | Split |
| **TD-41** | `RSS_FEEDS` hardcoded in `sentiment_crawler.py` | → CONFIG |
| **TD-42** | `/verify` runs 20d model unnecessarily | Cheap, leave |
| **TD-43** | `_humanize_feature` map in `main.py` | Move to `formatters.py` |
| **TD-51** | `trade_history.telegram_id` column name now misnomer | Rename |
| **TD-54** | `rl_mistake_logs` table name now misnomer (holds all predictions) | Rename |

### Closed since original audit
TD-01, TD-02, TD-03, TD-05, TD-06, TD-07, TD-08, TD-09, TD-10, TD-12, TD-13, TD-16,
TD-24, TD-26, TD-27, TD-28, TD-30, TD-31, TD-33, TD-44, TD-50.

### Shelved by decision
**TD-25** — VN macro feature population (vnibor, inflation_yoy).
- `vnstock` 4.x removed `Trading.money_market_historical_data` (verified via introspection).
- `markets.tradingeconomics.com` DNS-blocked from VN ISPs.
- `sbv.gov.vn` interbank rate page is Liferay 7.x + React (table loads via AJAX after page render); static HTML scrape returns 0 `<table>` elements.
- Resolution: ship without these columns. Model retraining ignores them via the `null_count() < height` guard in `_integrate_macro`.
- Re-open trigger: Playwright integration OR DNS-over-HTTPS resolver for TradingEconomics OR a working VN local API discovery.

---

## 12. Operations

### Production deployment

```bash
# 1. (One-time) Create unprivileged user
sudo useradd -r -s /bin/false quantbot
sudo chown -R quantbot:quantbot /opt/stock_price_v3
sudo touch /var/log/quant-v6-bot.log /var/log/quant-v6-bot.err.log
sudo chown quantbot:quantbot /var/log/quant-v6-bot*.log

# 2. Install systemd unit
sudo cp deploy/quant-v6-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-v6-bot.service
sudo systemctl status quant-v6-bot.service

# 3. Cron entries (operator crontab)
# Daily inference at 16:00 ICT (after 15:00 close + buffer)
0 16 * * 1-5 /opt/stock_price_v3/.venv/bin/python -X utf8 /opt/stock_price_v3/main.py --task daily_inference >> /var/log/quant-v6-inference.log 2>&1
# Daily backup at 23:00
0 23 * * * /opt/stock_price_v3/scripts/backup_db.sh >> /var/log/quant-v6-backup.log 2>&1
```

### Common ops commands

```bash
# Tail bot logs (rotating Python log + systemd-captured stdout)
tail -f logs/quant_v6.log
journalctl -u quant-v6-bot.service -f
tail -f /var/log/quant-v6-bot.log

# Manual inference run
python -X utf8 main.py --task daily_inference

# Manual full retrain
python -X utf8 main.py --task full_pipeline --force-crawl

# DuckDB sanity check (READ ONLY — no writes from CLI tools!)
python -X utf8 -c "
from src.data.db_engine import DuckDBEngine
db = DuckDBEngine()
print(db.conn.execute('SELECT COUNT(*) FROM stock_ohlcv').fetchone())
print(db.conn.execute('SELECT COUNT(*) FROM audit_log').fetchone())
print(db.conn.execute(\"SELECT COUNT(*) FROM portfolio WHERE user_id != 'cron'\").fetchone())
"

# Restart bot (graceful — SIGINT respects 30s timeout)
sudo systemctl restart quant-v6-bot.service

# Restore from backup (stop bot first!)
sudo systemctl stop quant-v6-bot.service
cp backups/quant_v6_core_YYYYMMDD.duckdb data/quant_v6_core.duckdb
sudo systemctl start quant-v6-bot.service

# Purge legacy RL stub rows (one-off, idempotent)
python -X utf8 scripts/cleanup_legacy_rl_stubs.py

# Check audit log for a user's history
python -X utf8 -c "
from src.data.db_engine import DuckDBEngine
db = DuckDBEngine()
rows = db.conn.execute('''
  SELECT command, ticker, details, timestamp
  FROM audit_log WHERE user_id = ?
  ORDER BY timestamp DESC LIMIT 20
''', ['<TELEGRAM_USER_ID>']).fetchall()
for r in rows: print(r)
"
```

### Logs

| Path | Contents | Rotation |
|---|---|---|
| `logs/quant_v6.log` | Main app log (cron + bot) | 10 MiB × 5 backups (RotatingFileHandler) |
| `logs/crawler_errors.txt` | Per-ticker crawler failures (TSV: ts\tticker\tcontext\terror) | 10 MiB × 5 backups |
| `logs/sbv_html_dump_*.html` | SBV HTML dump on parse failure | Keep latest 1 only |
| `/var/log/quant-v6-bot.log` | systemd-captured stdout (when running under unit) | None — needs `/etc/logrotate.d/quant-v6-bot` if you want it bounded |
| `journalctl -u quant-v6-bot.service` | systemd-level events (restart, crash) | systemd-controlled |

### Backups

- `scripts/backup_db.sh` — daily `cp data/quant_v6_core.duckdb backups/quant_v6_core_YYYYMMDD.duckdb`, 14-day retention via `find -mtime +14 -delete`.
- Cron at 23:00 daily.
- Limitation: plain `cp` may capture a torn write if the bot is mid-INSERT. Move to DuckDB's `.backup` SQL via a Python wrapper if torn writes ever materialize.

---

## 13. Recent file edits (state at the time of this dump)

**Recently modified** (TD-44 / TD-50 / TD-31 sprint):
- `main.py` — added `_log_rl_predictions` + `_backfill_rl_outcomes`, version stamp in startup banner + crash alerter, lock wrap on RL writes.
- `src/utils/telegram_bot.py` — version stamp in startup banner.
- `src/utils/version.py` — **new** — git SHA / VERSION file resolver.
- `scripts/cleanup_legacy_rl_stubs.py` — **new** — one-off TD-44 purge.

**Earlier sprint** (TD-33 / TD-05):
- `src/trading/portfolio_manager.py` — refactored to use `portfolio` table, `user_id` instead of `telegram_id`, default `user_id='cron'`.
- `main.py::run_trade_execution` — RL block replaced with two-phase logger calls.

**Even earlier**:
- `src/utils/audit_evaluator.py` — **new** — `run_post_mortem` for `/audit_weekly`/`/audit_monthly`.
- `src/utils/telegram_bot.py` — `audit_weekly_command`, `audit_monthly_command`, `_run_audit_command`.
- `src/utils/logging_utils.py` — extended with `setup_rotating_logging` + `get_crawler_error_logger`.
- `deploy/quant-v6-bot.service` — **new** — systemd unit.
- `scripts/backup_db.sh` — **new** — daily DB backup.

---

## 14. Where to look for X

| If you want to... | Look at |
|---|---|
| Change the bot's reply for a command | `src/utils/telegram_bot.py` — find `<command>_command` async function |
| Add a new Telegram command | `src/utils/telegram_bot.py` — add handler + register in `build_application()` + extend `HELP_TEXT` |
| Change feature engineering | `src/features/alpha360_generator.py` — `_preprocess_stock`, `_normalize_features`, `_generate_lags`, `_integrate_macro`, `_integrate_sentiment` |
| Change model architecture | `src/models/stacking_model/train_stacking.py` — `build_base_models()`, `build_oof_meta_features`, `train_horizon` |
| Tune sentiment veto / Top-N | `src/models/quant_agent_arbitrator.py` — `make_final_decision`, `NEWS_MAX_ARTICLES_PER_TICKER`, `NEWS_DOMAINS` |
| Adjust crawl rate limit / cooldown | `config/settings.py::CrawlerConfig` |
| Add a macro indicator | `src/data/crawlers.py::MacroCrawler.fetch_macro` (yfinance) + `_init_macro_daily_table` in `db_engine.py` (column migration) |
| Change LLM prompt | `src/models/quant_agent_arbitrator.py::NEWS_ANALYST_SYSTEM_PROMPT` (sentiment) or `src/utils/audit_evaluator.py::_explain_move` (post-mortem) |
| Modify the per-ticker alert format | `src/utils/telegram_alerter.py::_build_message` |
| Modify the combined/sell/verify report format | `main.py::_build_combined_report`, `_build_sell_hold_report`, `_build_verify_report` |
| See per-user audit history | `audit_log` table — `SELECT * FROM audit_log WHERE user_id=?` |
| See actual cron trades | `trade_history` table — column is `telegram_id` (legacy) but stores `user_id` |
| Reset a single user's portfolio | `DELETE FROM portfolio WHERE user_id = ?` |
| Force a model retrain | `python main.py --task full_pipeline --force-crawl` (or just `python -m src.models.stacking_model.train_stacking` if data already exists) |
| Debug a crash | journalctl + `logs/quant_v6.log`; the Telegram crash alert has version + traceback |
| Verify what version is running | `LOGGER.info` startup banner OR call `from src.utils.version import get_version; get_version()` |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **VN100 / HOSE** | Vietnamese stock exchange; ~355 active common-stock tickers as of 2026 |
| **OHLCV** | Open / High / Low / Close / Volume — daily candle data |
| **ADDV** | Average Daily Dollar Volume = mean(close × volume) over 20 days. Liquidity filter threshold = 15 B VND |
| **VWAP** | Volume-Weighted Average Price ≈ (High + Low + Close) / 3 in our approximation |
| **Alpha360** | Microsoft Qlib-style feature scheme: 6 fields × 60 lag bins = 360 columns per ticker |
| **Stacking GBDT** | Meta-ensemble of 3 base GBDT models (XGB + LGBM + CatBoost) → logistic regression meta-learner |
| **Quantile thresholds** | Train-set q33 and q66 of 5d returns define DOWN/SIDE/UP labels. Stored in `quantile_thresholds.json` per horizon |
| **5d / 20d horizon** | Forward-return windows; we predict both, arbitrator uses both in `make_final_decision` |
| **Sentiment veto / Safety override** | If Gemini sentiment < -0.5, 5d BUY is downgraded to HOLD. Class label 1 returned |
| **Top-6 → Top-3 funnel** | Quant Top-6 by p_up → arbitrator scores sentiment → Top-3 by (sentiment DESC, quant DESC) |
| **Arbitrator** | The combined news-scrape + Gemini + decision-rule layer. Module: `quant_agent_arbitrator.py` |
| **T0 / T+5** | T0 = day a prediction was made; T+5 = 5 trading days later (label horizon for 5d model) |
| **`cron` user_id** | Sentinel string for automated cron trades in the unified `portfolio` table. Real users have numeric Telegram IDs |
| **`-0.05` stub** | Pre-TD-05 hardcoded `actual_t5_outcome` value. Now purged (TD-44) |
| **VNIBOR** | Vietnam Interbank Offered Rate. 1-month tenor is the standard reference; we have an `interbank_on_rate` (overnight) column from when vnstock supported it, plus an unpopulated `vnibor` (1-month) column |
| **SBV** | State Bank of Vietnam — would be the canonical source for VN interbank rates if their portal weren't Liferay/React-rendered |

---

## 16. Dependencies (verbatim install)

```bash
pip install \
  "python-telegram-bot[job-queue]>=20.7" \
  duckdb \
  polars \
  pandas \
  numpy \
  scikit-learn \
  xgboost \
  lightgbm \
  catboost \
  joblib \
  vnstock \
  yfinance \
  cloudscraper \
  requests \
  beautifulsoup4 \
  feedparser \
  gnews \
  googlenewsdecoder \
  aiohttp \
  google-genai \
  python-dotenv \
  tqdm
```

GPU caveat: `train_stacking.py` configures XGBoost / LightGBM / CatBoost for CUDA.
On a CPU-only host, edit `build_base_models()` to swap `device="cuda"` → `device="cpu"`
and `task_type="GPU"` → `task_type="CPU"` (TD-10, low priority since training is
operator-driven, not on the hot path).

---

## 17. Things a new chat session should know FIRST

If you're handing this off to another LLM and they only read one section, make
it this one:

1. **Don't open DuckDB read-only.** Bare `duckdb.connect(path)` everywhere. The
   singleton (`DuckDBEngine`) holds an open RW connection; any secondary read-only
   open crashes with `Can't open a connection to same database file with a different configuration`.

2. **Use `google-genai`, not `google.generativeai`.** The old SDK is deprecated.
   API surface: `client = genai.Client(api_key=...)` then `client.models.generate_content(model=..., contents=..., config=genai_types.GenerateContentConfig(...))`. Model name has no `models/` prefix.

3. **Telegram replies must be HTML mode** (`parse_mode=ParseMode.HTML`). Markdown
   silently breaks on VN news bodies. Every dynamic field needs `html.escape()`.

4. **The cron path** (`main.py --task daily_inference`) and the **bot path**
   (`/suggest_buy`) both go through `daily_inference()`. The bot calls it with
   `broadcast=False` to suppress the per-ticker push alerts (otherwise the user
   gets the report twice). The function returns a combined HTML string for the
   bot to send.

5. **Multi-user portfolio isolation** is enforced at the SQL level via `WHERE
   user_id = ?` in every bot query against `portfolio` and `audit_log`. Cron
   trades land with `user_id='cron'` and are invisible to user `/suggest_sell`.

6. **RL outcome backfill is two-phase** (`_log_rl_predictions` writes NULL at T0,
   `_backfill_rl_outcomes` UPDATEs with real return after 5 trading days). DO NOT
   reintroduce a hardcoded outcome value — TD-44 just purged 3 legacy `-0.05` stubs
   in production.

7. **15:00 ICT crawl guard** — `main.py::is_crawl_allowed()` returns False before
   15:00. Bypass with `--force-crawl` only for operator-initiated rebuilds.

8. **Per-ticker 45s wall-clock circuit breaker** wraps every `fetch_ohlcv` call
   in the overnight crawler. Don't remove it — vnstock 30s timeouts without it
   stalled production overnight runs.

9. **Rate limiter is in-memory + monotonic clock.** Restart-loss is intentional
   (restart = cooldown expires). Currently applies to `/suggest_buy` + `/verify`
   only; TD-53 (extending to `/audit_*`) is the next obvious gate.

10. **Version stamp is `git rev-parse --short HEAD` with fallback to `./VERSION`
    file, then `"unknown"`.** Memoized. Available everywhere via
    `from src.utils.version import get_version`.

11. **`rl_mistake_logs` is a misnomer** — it holds ALL high-confidence
    predictions, not just mistakes. Training-time code must filter to negative
    outcomes if the goal is mistake-learning. TD-54 will rename to
    `rl_prediction_logs` eventually.

12. **`trade_history.telegram_id` is a misnomer** — stores `user_id` (may be
    `'cron'` or a Telegram digit string).

---

*End of context dump. ~600 lines, self-contained. Ask the new session to load this and they should be ready to work on any layer of the system without further onboarding.*
