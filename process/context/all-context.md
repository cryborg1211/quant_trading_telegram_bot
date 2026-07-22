# Quant Engine V4.0 - All Context

Last updated: 2026-07-22

This file is the root context entrypoint for the repo.

Use it for two things:

1. quick routing to the right context pack or root file
2. broad architecture and repository understanding

Start here before loading deeper context files.

---

## How This File Works (the `all-*.md` Convention)

Every `process/context/` directory has one `all-*.md` entrypoint that acts as an attachable quick router for that domain. This root file (`all-context.md`) is the top-level router. Context groups each have their own `all-{group}.md` entrypoint.

**The pattern:**

```
process/context/
  all-context.md                      <-- THIS FILE: root router
  tests/
    all-tests.md                      <-- group router for tests
  planning/
    all-planning.md                   <-- group router for planning
```

**How agents use it:**

1. Agent reads `all-context.md` first (this file)
2. Finds the relevant context group from the routing tables below
3. Reads that group's `all-{group}.md` entrypoint
4. Only then loads the specific deep doc needed

This layered routing keeps context windows small. Never load the whole `process/context/` tree.

---

## Quick Start

For most substantial tasks:

1. read this file first
2. choose the smallest relevant root file or context group from the tables below
3. only then load deeper files

---

## Current Root Entry Points

| File | Read when |
|---|---|
| `process/context/all-context.md` | any substantial planning, research, review, or implementation task |
| `process/context/tests/all-tests.md` | testing, verification, debugging test failures, execution planning |
| `process/context/planning/all-planning.md` | plan-shape calibration, planning examples, SIMPLE vs COMPLEX reference docs |

## Current Context Groups

| Group | Entry point | Scope |
|---|---|---|
| `planning/` | `process/context/planning/all-planning.md` | plan-shape calibration, planning examples, SIMPLE vs COMPLEX reference docs |
| `tests/` | `process/context/tests/all-tests.md` | pytest runner, 895 tests (72 files), in-memory DuckDB stubs, debugging quick-ref |

## Task Routing Table

| If the task involves... | Start with |
|---|---|
| architecture or stack questions | this file |
| testing or verification | `process/context/tests/all-tests.md` |
| creating a new plan | `process/context/planning/all-planning.md` |
| codebase dependency tracing | code-review-graph MCP tools (get_hub_nodes, get_impact_radius, query_graph) |
| blast radius assessment | code-review-graph MCP tools (get_impact_radius, get_affected_flows) |

## Context Group Lifecycle

Context groups are durable knowledge domains, not feature folders.

Create a group when:

- a topic has 3+ durable docs
- a single doc exceeds roughly 800 lines with separable subtopics
- multiple agents repeatedly need only one slice of a large context file
- the topic maps to a stable operational domain (tests, infra, database, auth, UI, workflows, etc.)

Do not create a group when:

- the content is a temporary report
- the content is a plan or execution artifact
- the topic is feature-specific and belongs in `process/features/...`

Move or split one group at a time. Use `all-{group}.md` entrypoints. Run the `audit-context` skill after every context organization change.

## Naming Convention

There are no `README.md` files inside `process/context/`.

Canonical entrypoints use `all-*.md`:

- root: `process/context/all-context.md`
- group: `process/context/{group}/all-{group}.md`

## Context Update Protocol

When durable project knowledge changes:

1. update the smallest relevant context file
2. update this file if routing, ownership, naming, or groups changed
3. update the owning `all-{group}.md` entrypoint when a group exists
4. run `audit-context`

---

## Repository Structure

```
stock_price_v3/
  main.py                 -- Pipeline orchestrator + serving (report builders extracted to src/reports/)
  run_bot.py              -- Telegram bot entry (continuous via systemd)
  run_backtest.py         -- Walk-forward backtest runner
  train_models.py         -- Model training entry (tabular ensemble + MR-LGBM)
  conftest.py             -- Root pytest fixtures
  pytest.ini              -- Pytest configuration
  requirements.txt        -- Pinned Python 3.11 dependencies
  config/
    settings.py           -- Dataclass-based config (PathConfig, ModelConfig, TradingConfig, etc.)
    settings.json          -- JSON overrides loaded by Config.from_json()
  src/
    backtest/
      pipeline.py         -- Feature pipeline (build_features, FEATURE_RECIPE_VERSION)
      walk_forward.py     -- Walk-forward engine: "tranche" mode (AFML staggered cohorts, evaluator default) + legacy "grid" mode; scales thousand-VND parquet prices to absolute VND (price_unit_vnd)
    bot/
      bot_inference.py    -- V3BotInference (serve-path model loading + prediction)
      sizing.py           -- Half-Kelly position sizing (20% NAV cap)
    crawlers/
      sentiment_crawler.py -- News sentiment crawler
    data/
      crawlers.py         -- OHLCV data ingestion
      db_engine.py        -- DuckDB engine management
      price_lookup.py     -- Fresh-parquet price lookups
      tensor_builder.py   -- Feature tensor construction
    execution/
      vn_cost_model.py    -- Vietnamese market cost model (fees, slippage)
    features/
      alpha360_generator.py  -- DEPRECATED (gutted in V4.0)
      market_regime.py    -- 8-regime HMM classifier + regime feature builder
      mr_features.py      -- Mean-reversion (knife-catch) features
    labels/
      triple_barrier.py   -- Triple barrier labeling (T+5, T+20 horizons)
    models/
      tabular_ensemble.py -- LightGBM + XGBoost + CatBoost → LogisticRegression meta
      macro_risk_hmm.py   -- HMM macro-regime overlay (2-state Gaussian)
      garch_hmm_regime.py -- GARCH(1,1)+multi-D Gaussian HMM exposure-brake overlay (log-vol space, persistence-capped)
      quant_agent_arbitrator.py -- Gemini-powered sentiment arbitrator + bear veto
      statistical_gates.py -- Statistical pre-filters
      train_mr_lgbm.py    -- Mean-reversion LGBM trainer
      stacking_model/
        purged_kfold.py   -- Purged K-fold cross-validation
    portfolio/
      construction.py     -- Mean-variance optimization
    trading/
      portfolio_manager.py -- Portfolio state management
      intraday_scanner.py  -- Intraday attack scanner (pure module, monitoring-only, kill-switch OFF default)
      portfolio_guard.py   -- EOD alert-only protective scan of human /add holdings (pure + thin I/O, kill-switch ON default)
      sector_map.py       -- HOSE ticker→sector map + apply_sector_cap (dispatch guard, 20-07-26)
      candidate_hysteresis.py -- Admission hysteresis: N-consecutive-day qualify streak (20-07-26)
      cohort_weights.py    -- prob_scaled_weights: shared serve/backtest cohort weight formula (A/B, REJECTED)
      breadth.py           -- Market breadth: exposure scalar leg + knife-catch inflection annotation + rank_breadth admission time series (20/22-07-26)
    reports/
      __init__.py         -- Re-exports report builders
      builders.py         -- 10 report builder functions + 11 constants (extracted from main.py)
    utils/
      telegram_alerter.py -- Telegram message formatting + delivery
      telegram_bot.py     -- PTB application builder (commands, handlers)
      logging_utils.py    -- Centralized logging setup
      audit_evaluator.py  -- Trade audit evaluation
      version.py          -- Version string
  tests/                  -- 72 test files, 895 tests (pytest)
  train_macro_regime.py   -- Train + serialize the GARCH-HMM regime overlay
  scripts/
    migrate_sqlite_to_duckdb.py -- Legacy SQLite → DuckDB migration
    backup_db.sh          -- Database backup script
    cleanup_legacy_rl_stubs.py  -- Dead code cleanup
    analyze_sentiment_paperlog.py -- T+3 / T+20 return stats for sentiment-entry treatment vs control
    validate_garch_hmm_brake.py -- A/B backtest: T+5 signals with vs without GARCH-HMM exposure brake
    sweep_garch_hmm_brake.py    -- Robustness grid (min_exposure floor × max_persistence cap)
    walk_forward_macro_pipeline.py -- Rolling-window regime diagnostic (T+5/T+20, 504d train)
    check_drift.py        -- Read-only drift/sampling-bias monitor (prediction dist, UP base-rate, regime pulse)
    analyze_mr_regime_conditioning.py -- MR-LGBM regime-gate + RSI-direction knife-catch research (both REJECTED)
    analyze_mr_breadth_inflection.py -- MR-LGBM breadth-inflection knife-catch research (promising, unconfirmed)
    retrain_all.ps1       -- Full 3-model retrain (MR+T20+T5); scheduled weekly via Windows Task Scheduler
    ab_rank_breadth.ps1   -- rank_breadth admission A/B runner (REJECTED)
  deploy/
    quant-v6-bot.service  -- Systemd unit for run_bot.py
  doc/
    SYSTEM_DESIGN.md      -- System design notes (partially stale)
  data/                   -- Runtime data directory (DuckDB, Parquet shards)
  models/                 -- Trained model artifacts (.joblib)
  logs/                   -- Runtime log output
  backups/                -- Auto-saved previous model artifacts (do not index)
  process/                -- Agent harness operational workspace
```

## Technology Stack

- **Language:** Python 3.11.9
- **ML Framework:** Pure-tabular stacking ensemble — LightGBM 4.6 + XGBoost 3.2 + CatBoost 1.2 → CalibratedClassifierCV LogisticRegression meta-learner
- **Regime Model:** hmmlearn 0.3 (2-state Gaussian HMM macro-regime overlay); arch 8.0 GARCH(1,1) + multi-D Gaussian HMM exposure-brake overlay (`garch_hmm_regime.py`)
- **Feature Pipeline:** Polars 1.40 native (fast columnar operations), Pandas 3.0 for legacy compat
- **Storage:** DuckDB 1.5 + PyArrow 24 Parquet shards (OHLCV ingestion, feature caching)
- **Numerics:** NumPy 2.3, SciPy 1.16, scikit-learn 1.8
- **Sentiment:** Google GenAI SDK 1.70 (Gemini Flash) — conditional soft overlay + hard bear veto
- **News:** GNews 0.4 + googlenewsdecoder 0.1 + BeautifulSoup 4.14
- **Bot:** python-telegram-bot 22.7 (async PTB framework)
- **HTTP:** aiohttp 3.13, requests 2.33
- **Config:** python-dotenv 1.2 (.env), dataclass-based settings with JSON overrides
- **Testing:** pytest (895 tests, in-memory DuckDB stubs)
- **Deployment:** Bare metal VPS — systemd (bot), cron (daily pipeline at 15:30 ICT Mon–Fri); weekly full retrain via Windows Task Scheduler (`quant_weekly_retrain`, every Saturday 09:00) on the dev box

## Key Patterns and Conventions

**Typing:** Strict Python 3.10+ type hints everywhere. All function signatures annotated.

**Data processing:** Polars-native feature pipelines. Avoid reverting to Pandas unless interfacing with legacy ML libraries that require it.

**Storage pattern:** DuckDB for analytical queries, Parquet shards for data at rest. `db_engine.py` manages connections.

**Config pattern:** Nested dataclasses (`Config` → `PathConfig`, `ModelConfig`, `TradingConfig`, `CrawlerConfig`, `SentimentConfig`). JSON overrides via `config/settings.json`. Singleton `CONFIG` instance.

**Architecture style:** Pure functions + procedural orchestration. No deep OOP inheritance trees. Prefer extracting pure functions over adding class methods.

**Feature recipe versioning:** `FEATURE_RECIPE_VERSION` in `src/backtest/pipeline.py` (computed via `compute_feature_schema_hash(...)`, currently `v2-sha8:53b5bd85`). Hard gate — serve-path checks match at load time. Any feature engineering change requires: bump version (auto via the schema hash) → full retrain.

**Backtest portfolio construction:** `run_backtest.py` defaults to `--mode tranche --hold-days 30` (staggered AFML cohort book: daily deploy NAV/H into top-`max_positions` names, hold exactly H trading days). Legacy `--mode grid` (concentrated delta-rebalance) is structurally unfit for this signal — its ~45 correlated entry dates let market beta dominate. Price-scale rule: parquet OHLCV is in thousands of VND; the engine converts to absolute VND via `WalkForwardConfig.price_unit_vnd` — any new code feeding parquet prices into `VNCostModel` must do the same. Bot payload carries a `strategy` dict (mode/hold_days/signal_threshold); serve consumes it via `_tranche_signal_fields` (tranche cohort weight `1/(hold_days×n_picks)`).

**Regime-conditional sizing (DD control):** Both backtest and serve apply the market-regime policy from `src/trading/regime_policy.py` — the single source of truth for `NO_TRADE_REGIMES {0,7}` (skip the name, weight stays cash), `PENALTY_REGIMES {1,6}` (× `REGIME_PENALTY_FACTOR` = 0.5 = `REGIME_PENALTY_CAP/DEFAULT_NAV_CAP`), and `STRONG_TREND_REGIME {3}`; imported by both `src/bot/sizing.py` (serve) and `src/backtest/walk_forward.py` (backtest). **Backtest:** opt-in via `--regime-sizing` / `WalkForwardConfig.use_regime_sizing` (default OFF). **Serve:** `main._dispatch_signals` applies it in the non-event-override branch (regime read per-ticker from the `_LATEST_REGIME_BY_TICKER` cache; event overrides keep precedence), gated by `CONFIG.trading.regime_sizing_enabled` (default **ON**, settings.json kill-switch). A/B (2026-06-14, T+20 GOLDEN): MaxDD −23.3%→−16.9%, Sharpe +0.73→+0.88, Net +46%→+42%, DSR 0.35→0.45 (still <0.95 → stays paper-only). No feature-recipe change, no retrain. PENALTY uses a 0.5× *multiplier* (not the absolute `REGIME_PENALTY_CAP`) because tranche per-name (~0.7% NAV) is far below the 10% cap.

**Serve-path horizons:** PRIMARY = T+20 (`v3_ensemble_20d.joblib`). SHORT = T+5 (`v3_ensemble_5d.joblib`), used by `/verify` for intraday confirmation — `SHORT_HORIZON_DAYS = 5` in `src/bot/bot_inference.py` (recovered 18-06-26; was briefly T+3 on 12-06-26, reverted because the 5d artifact was already gate-verified `v2-sha8:53b5bd85` and no retrain was required). Known cleanup debt: `src/reports/builders.py:326` hardcodes literal `5 ngày tới` instead of `{SHORT_HORIZON_DAYS}` — fix pending in the Telegram-work effort. **Latest retrain gates (17-07-26, full 3-model retrain, finer sweep grid):** T+20 up_threshold 0.45→**0.46** (UP-precision 0.5698), T+5 up_threshold 0.40→**0.44** — both tighter than before, both still fail DSR (paper-only). See the promote-gate paragraph below: future retrains compare against these before overwriting.

**Sentiment-entry paper-log:** `sentiment_entry_paperlog` DuckDB table (+ `seq_sentiment_entry_id` sequence) captures the full candidate cross-section on every daily pipeline run (`source='daily'`) and every `/verify` invocation (`source='verify'`). Columns: per-horizon model probabilities, `decision_5d` argmax, `sentiment_score`, `entry_close`, `ret_3d`, `ret_20d`, `outcome_filled`. Backfill is PROGRESSIVE (fixed 2026-06-22, `_backfill_paperlog_outcomes`): `ret_3d` fills once the T+3 window matures (scan gate `_PAPERLOG_SHORT_MATURE_DAYS`=4 calendar days), `ret_20d` at `_PAPERLOG_MATURE_DAYS`=21 days; `outcome_filled` flips TRUE only when the terminal T+20 return lands (rows with only `ret_3d` stay pending). Uses `price_lookup`; see `doc/audit_paperlog_temporal_flaw.md`. Config knobs: `CONFIG.trading.sentiment_entry_enabled` (default True) / `sentiment_entry_threshold` (default 0.7, analysis-time reference only — all rows are logged regardless). Analysis: `scripts/analyze_sentiment_paperlog.py`. Tests: `tests/test_sentiment_paperlog.py` (10 tests). Shipped 2026-06-16; `source='daily'` row not yet confirmed in production (requires one live cron run at 15:30 ICT).

**Model accuracy auditor (2026-06-29):** `src/utils/accuracy_audit.py` — READ-ONLY confusion-matrix analytics layered on top of `sentiment_entry_paperlog` (no new table, no serve-path write). `classify_outcome` maps each settled row to TP/FP/TN/FN (BUY=positive class: BUY+R>0=TP, BUY+R≤0=FP, SELL/HOLD+R≤0=TN, SELL/HOLD+R>0=FN), `summarize_accuracy` yields BUY Precision = TP/(TP+FP) and Defensive Recall = TN/(TN+FN), `build_accuracy_report` renders a 🟢🔴🔵 last-15 table. Realized return = `ret_20d`; settled = `outcome_filled=TRUE`; prefers arbitrated `final_decision`, falls back to `decision_5d`. Decision encoding (shared with the arbitrator): `0=SELL/EXIT, 1=HOLD, 2=BUY`. Wired to a NEW bot command `/audit_accuracy` (system-wide — paperlog is global, no user_id); `/audit_weekly` is unchanged (post-mortem + engine-picks). Tests: `tests/test_accuracy_audit.py` (16). A consolidated spec to build a parallel `model_predictions_audit` table was deliberately REJECTED as redundant with the paperlog + temporal-flaw-prone.

**Gemini 503 resilience (2026-06-29):** `sentiment_crawler._generate_content` is the single Gemini call, `tenacity @retry` wrapped — `wait_exponential(multiplier=1, min=1, max=60)`, `stop_after_attempt(5)`, retry ONLY on transient errors via `_is_transient_gemini_error` (503/502/504/UNAVAILABLE/OVERLOADED by `.code` or message); 4xx + JSON-parse errors are never retried. Reusable scoring core extracted to `SentimentCrawler.score_payload` (shared by `_score_item` + the backfill). On exhausted retries it falls back to a neutral row with `reason='Gemini fallback: ...'`. Backfill of historically corrupted rows: `scripts/backfill_sentiment_503.py` (re-scores `reason LIKE 'Gemini fallback%'` from stored `title` — NO `text_raw` column exists — UPDATEs by `url`; dry-run default, `--commit` to write). `tenacity==9.1.4` pinned. Tests: `tests/test_sentiment_resilience.py` (15).

**Telegram formatting:** Strict 4096-char limit. HTML mode with careful tag closure. Long reports split into multiple messages.

**Hub nodes (highest blast radius):**
1. `build_regime_features` (market_regime.py) — degree 142
2. `run_backtest.run_oos` / `_build_wf_config` — degree 97 (no standalone `main`; the hub is these two extracted functions)
3. `daily_inference` (main.py) — degree 84 (active decomposition target)
4. `triple_barrier_pipeline` — degree 82
5. `TabularEnsemble.fit` — degree 75
6. `build_application` (telegram_bot.py) — degree 72

**Active refactoring (V4.1 Structural Debt program):**
- Phase 1 COMPLETE: `daily_inference` decomposed (271→169 lines) into `_select_candidates()`, `_rescue_loop()`, `_dispatch_signals()`. Report builders (10 functions + 11 constants) extracted to `src/reports/builders.py`. 21 new tests added.
- Phase 2 COMPLETE (2026-06-13): Automated feature-schema hashing live (`src/utils/schema_hash.py`); `FEATURE_RECIPE_VERSION` computed via `compute_feature_schema_hash(...)` (recipe `v2-sha8:53b5bd85`), replacing the manual `"v1.1"` string.
- Phase 3 COMPLETE (2026-06-21): Hub-node test coverage added for `VNCostModel.simulate`, `triple_barrier_pipeline`, `TabularEnsemble.fit`, `run_backtest.run_oos`/`_build_wf_config` — new `tests/test_vn_cost_model.py`, `test_triple_barrier.py`, `test_tabular_ensemble.py`, `test_run_backtest_wiring.py` (95 tests).

**Deprecated:** `alpha360_generator.py` is gutted in V4.0 — system is purely tabular.

**Local dashboard (new program, 2026-06-19):** Streamlit dashboard for a single Windows laptop user. Package root: `dashboard/`. No-polling architecture: send-only Telegram alerter, no PTB ApplicationBuilder anywhere in `dashboard/`. Reuses `main.daily_inference(broadcast=False)`, `verify_single_ticker`, `inference_for_holdings`, `PortfolioManager`, `signal_ledger`, `audit_evaluator.run_post_mortem`. Tabs: `mua/ban/giu/verify/audit/settings/tam_nhin`. Launch: `streamlit run dashboard/app.py` — `app.py` prepends repo-root to `sys.path` (streamlit puts only the script dir on path). Dashboard reads of serve-built signal dicts must coerce the Telegram-display `price` string via `headless._parse_price` (serve stores `"22,600 VND"` text). Installer: Inno Setup `setup.exe` (P4 scope) — no PyInstaller.

**Intraday attack scanner (2026-07-06):** `src/trading/intraday_scanner.py` — pure module, monitoring-only. Every 10–30 min during HOSE hours (09:15–11:30 / 13:00–14:45 ICT) one SSI iBoard bulk snapshot builds provisional daily bars (absolute VND ÷1000 to the parquet thousands convention), spliced in-memory onto each ticker's 120-row parquet tail, dual-horizon (T+5 + T+20) rescore, ADV-top-50 gate (`_resolve_candidate_universe`), event-only alert card (new top-3 entrant / τ crossing / |Δ|≥2pp) sent to BOTH chats (ADMIN + USER). **Hard constraints:** zero writes to parquet/DuckDB/`sentiment_entry_paperlog` (protects the running item-1 pre-registered experiment), no arbitrator/Gemini call in the loop, no `signal_ledger` writes, no BUY dispatch. Config: `TradingConfig.intraday_scanner_enabled` (default **False** — kill-switch), `intraday_scan_interval_min` (default 15, clamped 10–30), `intraday_alert_delta_pp` (default 0.02). Wiring: `telegram_bot.py::_intraday_scan_job` via PTB `JobQueue`, config-gated; degrades to an ERROR log (bot still builds/runs) when `app.job_queue is None`. `requirements.txt` pins `python-telegram-bot[job-queue]==22.7` (new dependency — PTB's JobQueue extra was not previously installed). Tests: `tests/test_intraday_scanner.py` (39). Gate 5 (live market-open smoke, 09:15 ICT) still pending as of ship date — plan stays active until that manual verification lands.

**Tầm Nhìn fan-chart (2026-06-29):** `dashboard/tabs/tam_nhin.py` + `dashboard/utils/fan_chart.py` — TradingView-style pure-technical forecast: `go.Candlestick` history + 12 Monte Carlo GBM paths + neon median (no shaded bands), rangeslider/rangeselector, crosshair spikes (`spikemode="toaxis+across"`). `project_fan` still computes analytic bands (unit-tested) but the figure draws MC paths instead. MC seed is per-ticker (`_ticker_seed` = `crc32(ticker)`, NOT `hash()` which is process-salted) — a single shared seed made every ticker render one cloned wiggle shape. Needs OHLC, so `price_lookup.ohlc_history(ticker, n)` was added alongside `close_history`. Tests: `tests/test_dashboard_fan_chart.py`.

**Portfolio guard (2026-07-14):** `src/trading/portfolio_guard.py` — pure module (+ one thin I/O function), EOD alert-only protective scan of human users' `/add`-tracked holdings (never the `user_id='cron'` automated book — the triggering incident was a user riding a position −7% with no alert). Stage 1 is five deterministic, zero-LLM triggers per lot: hard stop-loss (`CONFIG.trading.stop_loss_pct`), take-profit (`take_profit_pct`), trailing-stop (new `portfolio_guard_trailing_pct`, default `0.08`), model-flip (OR across T+5/T+20 argmax==SELL), NO_TRADE regime warning (`{0,7}`); a corporate-action-gap shield downgrades hard-stop/trailing-stop wording only (take-profit stays confident, per approved scope). Stage 2 is config-gated: at most ONE `evaluate_trades_batch` arbitrator call + ONE `mr_score_tickers` call, across the union of triggered tickers only (never per-user, never per-trigger). **Hard constraints:** alert-only (zero writes to parquet/DuckDB/`signal_ledger`/`sentiment_entry_paperlog`, no auto-sell), never-raise (`main.notify_portfolio_guard` mirrors `notify_tranche_exits`'s try/except→log→`0` shape), event-only delivery (silent when nothing fires), and `portfolio_guard.py` never imports `main` (one-directional — `main.py` imports it). Wiring: `main._run_guard_for_users` + `main.notify_portfolio_guard()` called from both `full_pipeline` and `inference_only` immediately after each's `notify_tranche_exits()`; optional on-demand `/guard` command (mirrors `/suggest_sell`). Delivery is per-user Telegram DM via new `TelegramBot.send_text_to_chat(chat_id, html_text, label)` (extracted `_send_to_one` from `_dispatch`, single-recipient not broadcast). Config: `CONFIG.trading.portfolio_guard_enabled` (default **True**, kill-switch), `portfolio_guard_trailing_pct` (default `0.08`), `portfolio_guard_llm_enabled` (default **True**). **Price-scale resolution (the highest-risk defect this feature guards against):** `portfolio.price` (a user's `/add` entry price) had two conflicting scale assumptions already live in the codebase — the bot's own `/add` help example (`/add VNE 1000 32.5`) implies thousands-VND, while `dashboard/utils/headless.py::_pnl_ratio` assumes absolute VND unconditionally. `portfolio_guard.normalize_entry_price_vnd` resolves it with the same `< 1000.0 ⇒ ×1000` rule already canonical in `main._VN_PRICE_SCALE_THRESHOLD` (main.py:89); confirmed correct on real rows (entries `27.5`/`152.5` normalize to `27,500`/`152,500` VND). **Flagged, not fixed:** `headless.py::_pnl_ratio`'s one-sided absolute-VND assumption is a latent-bug candidate for the dashboard's GIỮ tab if a user ever followed the bot's own thousands-VND `/add` example literally — out of scope for this feature by design; tracked in `process/features/local-dashboard/backlog/dashboard-pnl-ratio-price-scale_14-07-26.md` (recommended fix reuses `normalize_entry_price_vnd` rather than a third inline copy of the threshold rule). Tests: `tests/test_portfolio_guard.py` (40). Status: code-complete, 653/653 full suite green (independently re-confirmed via a full `pytest -q` run), live-DB read-only dry run completed (no write paths invoked by construction; empirical DB diff not run) — the general plan (`process/general-plans/active/portfolio-guard_PLAN_13-07-26.md`) stays ACTIVE pending confirmation of the first production EOD run at the 15:30 ICT cron (same Gate-5-style precedent as the intraday scanner).

**Manual verification (20-07-26, 00:06 ICT):** an operator-triggered `full_pipeline` run (not the scheduled cron) fired portfolio_guard for real against live holdings — alert sent, confirming end-to-end wiring. Does not satisfy the plan's own stated gate (the scheduled 15:30 ICT cron); the plan stays ACTIVE.

**Sentiment append dedup fix (2026-07-19/20):** `src/crawlers/sentiment_crawler.py::_append_rows` crashed the ENTIRE EOD `full_pipeline` on 16-07 and 17-07 — re-crawled Vietstock warrant articles (crawl window overlaps yesterday) violated `hist_sentiment_llm_labeled`'s PK (ticker, date, title), killing the pipeline before inference ran two days straight (no signals, no reports). Fix: batch `drop_duplicates` + SQL anti-join before INSERT (existing rows win, paid Gemini scores preserved). Tests: `tests/test_sentiment_append_dedup.py` (4). Live-verified 20-07 00:01: pipeline re-crawled the exact offending articles and completed cleanly.

**Sentiment-entry paperlog starvation ROOT-CAUSED and FIXED (2026-07-19/20):** `source='daily'` rows stopped 2026-06-30 (13 trading days silent) — root cause was NOT the scheduler (an earlier 07-07 fix attempt did not actually work): the meta-labeler τ-gate rejects all 50 liquid names on most EOD runs ("Top-0 survivors → arbitrator pool: []"), and ALL THREE of `daily_inference`'s no-trade exit paths (weak-market fallback, empty-universe fallback, empty-live-prices early return in `run_trade_execution`) returned before the paperlog write. Fix: extracted `_paperlog_snapshot_and_backfill` (shared write+backfill core) + `_paperlog_no_trade_day` (DuckDBEngine-singleton path, mirrors `verify_single_ticker`, for exits holding no `PortfolioManager`) — all 3 exit paths now log. No-trade days are data, not noise; the item-1 experiment needs exactly this slice.

**July-2026 loss root cause + dispatch guards (2026-07-20):** 12 of 13 July dispatches closed/ran red, all via the T+5 side-door (the T+20 τ-gate stayed shut all month) — BSR was re-dispatched 4 consecutive days into a falling knife, and 8 of 13 picks were one correlated PetroVietnam energy/fertilizer complex. Two guards shipped in `main._select_candidates` (applied before the arbitrator-pool slice, so a skipped name frees its slot for the next-best; the monitoring-only fallback branch stays unfiltered by design):
- **Open-cohort dedup** — `signal_ledger.open_tickers()`: any ticker with an OPEN `dispatched_signals` cohort (any horizon) is excluded from new candidate pools. Kill-switch `CONFIG.trading.dispatch_open_cohort_dedup_enabled` (default ON).
- **Sector cap** — `src/trading/sector_map.py`: ~100 liquid HOSE tickers mapped to 13 sectors (the whole July cluster — BSR/PVD/GAS/DPM/DCM — is deliberately ONE `OIL_GAS` sector, since they share one gas-price driver). Max `CONFIG.trading.arbitrator_sector_cap` (default 2) names per sector in the arbitrator pool; unmapped tickers (`OTHER`) are uncapped.
- **Admission hysteresis** — `src/trading/candidate_hysteresis.py`: requires `CONFIG.trading.hysteresis_min_qualify_days` (default 2) CONSECUTIVE raw-qualifying days (tracked independent of dedup/sector-cap, so a long-open cohort doesn't reset a name's streak) before admission. Own-connection DuckDB table `candidate_qualify_streak` (mirrors `signal_ledger.py`'s pattern); read is always safe, write is `persist`-gated (new `persist` param threaded through `_select_candidates`). Kill-switch `CONFIG.trading.hysteresis_enabled` (default ON). **Process note:** a pre-existing test file (`test_select_candidates.py`) had no awareness of this new knob — fixed via a GLOBAL autouse fixture in root `conftest.py` defaulting new risky DB-touching config knobs OFF for the whole suite, rather than patching every call site individually; apply this pattern proactively for future default-on knobs that open a DB connection from a commonly-exercised path.

**Meta-controller unification + drift/breadth exposure brakes (2026-07-20):** `src/bot/garch_brake.py` evolved from a single GARCH-HMM leg into a 3-leg market-wide exposure combiner (`live_exposure_scalar() = min(garch, drift, breadth)`), sharing ONE OHLCV panel load per call (was 2) and emitting one consolidated attribution log line (`legs garch=X drift=Y breadth=Z → combined=W (binding=leg)`) — the "which layer cost how much" question the July post-mortem couldn't answer before this. New legs:
- **Drift** — `drift_scalar_from_returns`: trailing `CONFIG.trading.drift_brake_window` (default 10)-session cumulative market-proxy return, piecewise-linear ramp from `drift_brake_trigger` (−3%) to `drift_brake_full` (−6%), floor `drift_brake_floor` (0.5). Covers the slow-bleed shape the vol-triggered GARCH leg missed all of July (read ~0.99 through a 12-of-13-red month).
- **Breadth** — `src/trading/breadth.py::breadth_from_panel` + `breadth_scalar`: fraction of the liquid universe with a positive trailing `breadth_brake_window` (default 20)-session return, same ramp shape (`breadth_brake_trigger`=0.40 / `floor_level`=0.25 / `floor`=0.5).
Each leg fails open to 1.0 independently. Kill-switches: `garch_brake_enabled` / `drift_brake_enabled` / `breadth_brake_enabled` (all default ON).

**Drift + sampling-bias monitor (2026-07-20):** `scripts/check_drift.py` — read-only, on-demand: (1) serve P(UP) distribution vs OOS reference percentiles, (2) trailing realized 20d UP base rate vs train (sampling bias), (3) regime defensive-share pulse. First run found prediction calibration OK but **trailing UP base rate SEVERE: 29.5% vs 41.5% train** — the July bleed was a market-wide breadth collapse, not model decay. `REF_*` constants must be re-pinned after every retrain.

**Weekly auto-retrain (2026-07-20):** `scripts/retrain_all.ps1` (MR-LGBM + T+20 + T+5, full walk-forward + save) is scheduled via Windows Task Scheduler task `quant_weekly_retrain`, **every Saturday 09:00** (user-requested, upgraded from an initial monthly cadence). Auto-persists new GOLDEN artifacts; old ones auto-backed-up under `models/saved/backups/`.

**Promote-gate for the weekly auto-retrain (2026-07-20):** `run_backtest._persist_bot_payload` compares a new candidate's embedded OOS metrics (`oos_sharpe`, `oos_max_dd`, `golden_mean_up_precision` — already carried by every bundle) against the CURRENT on-disk artifact's own metadata before overwriting. Reject conditions: Sharpe regression beyond `CONFIG.trading.promote_gate_min_sharpe_delta` (default −0.10), MaxDD regression beyond `promote_gate_max_dd_regression_pp` (default 3.0pp), or UP-precision below the absolute floor `promote_gate_min_up_precision` (default 0.35). On reject, the incumbent stays live untouched and the candidate is stashed under `models/saved/rejected/*_REJECTED.joblib`. No incumbent (first run) always promotes. Kill-switch `promote_gate_enabled` (default ON) — protects the now-unattended weekly schedule from a single bad walk-forward run auto-deploying.

**Knife-catch timing research + breadth-context annotation (2026-07-22):** Three hypotheses tested for timing the MR-LGBM (`🔪 BẮT ĐÁY`) capitulation-bounce signal, reusing `train_mr_lgbm.py`'s exact purged-OOF/CV/feature machinery (read-only research scripts, zero model changes):
1. **Regime-gate** (`scripts/analyze_mr_regime_conditioning.py`) — REJECTED. Gating on `market_regime==5` (Mean-Reversion/RSI-extreme) underperformed (precision 0.563) both the overall rate (0.576) and Choppy/regime-6 (0.601, 59% of all rows).
2. **RSI-direction split** (same script, round 2) — REJECTED, and corrected round 1's own root-cause theory: regime-5's overbought half has ZERO fires (MR-LGBM's setup features are structurally bearish-only, so there was never any "dilution" to begin with). The more direct test — all rows split by raw RSI14 direction — found deep-oversold fires (0.560 precision) are LESS reliable than moderate-RSI fires (0.597), opposite of the "wait for max panic" intuition.
3. **Breadth inflection** (`scripts/analyze_mr_breadth_inflection.py`) — the one promising result: OOF (n=59,508/133,258) showed Low-breadth-and-Rising precision 0.667 vs Low-breadth-and-Falling 0.542, clearing the model's own 0.60 target. BUT an extended ~3-year out-of-training check (still historical parquets, no new data needed — a user challenge correctly caught an earlier "needs paperlog" mis-framing) found Low+Falling ties Low+Rising (both 0.833) once it clears the reliability floor — the effect does NOT replicate on a larger holdout. **Verdict: promising, unconfirmed.**
Wired anyway (explicit user call, despite the "unconfirmed" verdict) as a DISPLAY-ONLY annotation — `src/trading/breadth.py::live_breadth_inflection` + `breadth_delta_from_panel`, consumed by `main.mr_score_tickers` (computed once per call, only when ≥1 ticker fires) and rendered via `src/reports/builders.py::_mr_breadth_context_line` on all 3 existing MR display sites (fallback report, sell/hold veto, `/verify`). Always labeled "tín hiệu nghiên cứu, chưa xác nhận" (research signal, not confirmed) — never gates the fire decision itself. Kill-switch `mr_breadth_context_enabled` (default ON).

**Backtest admission/weighting A/B results, all REJECTED (2026-07-20/22)** — same acceptance gate as the 14-06-26 regime-sizing A/B (Sharpe up AND MaxDD not worse):
- **Prob-scaled cohort weights** (`src/trading/cohort_weights.py::prob_scaled_weights`) — equal-weight beat it (Sharpe 0.545 vs 0.533; edge differences within a top-10 cohort are too small/noisy to size on).
- **Hold-days sweep** — confirmed hold=30 (the existing default) as a genuine hump-peak optimum: hold=20 Sharpe 0.497, hold=30 **0.545**, hold=40 0.503.
- **Rank-based admission with breadth-conditioned K** (`WalkForwardConfig.admission_mode="rank_breadth"`, `src/trading/breadth.py::breadth_time_series`) — no absolute P(UP) floor, top-K ranked names where K scales with market breadth. mean Sharpe 0.524→0.518, mean DD ~−13.5%→~−14.2% (worse both ways) on the 2022-11→2026-07 OOS window with default knobs.
All three stay in-tree behind their flags (default OFF/unchanged), fully tested, cheap to revisit with different tuning later.

## Environment and Configuration

**Config files:**
- `config/settings.py` — dataclass definitions + `CONFIG` singleton
- `config/settings.json` — JSON overrides (runtime knobs)
- `.env` — secrets (git-ignored)
- `.env.example` — template for required env vars
- `pytest.ini` — test runner config

**Env var groups (names only, never values):**
- Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Sentiment LLM: `GEMINI_API_KEY`, `GEMINI_MODEL` (optional, defaults to `gemini-flash-latest`)

**Runtime paths (from PathConfig defaults):**
- Data: `data/` (DuckDB at `data/quant_v6_core.duckdb`, Parquet at `data/alpha360_features.parquet`, `data/macro_daily.parquet`)
- Models: `models/` (trained .joblib artifacts)
- Logs: `logs/`

## Current Features

| Feature | Folder | Status |
|---|---|---|
| V4.1 Structural Debt | `process/features/v4-1-structural-debt/` | COMPLETE (2026-06-21, all 3 phases) |
| Local Dashboard | `process/features/local-dashboard/` | in-progress (P0–P2 done; P3 launcher next) |
| Macro Integration (A/B) | `process/features/macro-integration/` | COMPLETE (2026-06-23): P1 crawler + P2 macro-HMM overlay kept; GBM-macro A/B KILLED (worse DD+PBO), `use_macro_features` default OFF |
| GARCH-HMM Regime Overlay | `process/features/macro-integration/` | shipped (2026-06-24): `garch_hmm_regime.py` GARCH(1,1)+5-D HMM exposure brake; log-vol space, persistence-capped 0.96, linear scaler clip(P(Bull),0.2,1.0). OOS T+5 KEEP (loss-mitigation, still −Sharpe). **Superseded/absorbed 2026-07-20**: `src/bot/garch_brake.py` is now a 3-leg meta-controller (GARCH + drift + breadth) — see the "Meta-controller unification" paragraph above |

## Code-Review-Graph MCP

A local MCP server (`code-review-graph`) maintains a live graph database of the entire codebase. Use its tools for dependency tracing, blast radius analysis, and hub detection instead of broad grep scans. Key tools:
- `list_graph_stats_tool` — graph health check
- `get_hub_nodes_tool` — architectural hotspots
- `get_impact_radius_tool` — blast radius for a specific node
- `get_affected_flows_tool` — execution flows touching a node
- `query_graph_tool` — arbitrary graph queries

## Scan Metadata

- Generated: 2026-06-09
- Last content update: 2026-07-22
- HEAD: main (70070dc)
- Mode: fresh scaffold + study
- Package manager: pip (requirements.txt, Python 3.11)
