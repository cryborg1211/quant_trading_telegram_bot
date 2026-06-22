# Quant Engine V4.0 - All Context

Last updated: 2026-06-19

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
| `tests/` | `process/context/tests/all-tests.md` | pytest runner, 238 tests, in-memory DuckDB stubs, debugging quick-ref |

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
      quant_agent_arbitrator.py -- Gemini-powered sentiment arbitrator + bear veto
      statistical_gates.py -- Statistical pre-filters
      train_mr_lgbm.py    -- Mean-reversion LGBM trainer
      stacking_model/
        purged_kfold.py   -- Purged K-fold cross-validation
    portfolio/
      construction.py     -- Mean-variance optimization
    trading/
      portfolio_manager.py -- Portfolio state management
    reports/
      __init__.py         -- Re-exports report builders
      builders.py         -- 10 report builder functions + 11 constants (extracted from main.py)
    utils/
      telegram_alerter.py -- Telegram message formatting + delivery
      telegram_bot.py     -- PTB application builder (commands, handlers)
      logging_utils.py    -- Centralized logging setup
      audit_evaluator.py  -- Trade audit evaluation
      version.py          -- Version string
  tests/                  -- 21 test files, 238 tests (pytest)
  scripts/
    migrate_sqlite_to_duckdb.py -- Legacy SQLite → DuckDB migration
    backup_db.sh          -- Database backup script
    cleanup_legacy_rl_stubs.py  -- Dead code cleanup
    analyze_sentiment_paperlog.py -- T+3 / T+20 return stats for sentiment-entry treatment vs control
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
- **Regime Model:** hmmlearn 0.3 (2-state Gaussian HMM macro-regime overlay)
- **Feature Pipeline:** Polars 1.40 native (fast columnar operations), Pandas 3.0 for legacy compat
- **Storage:** DuckDB 1.5 + PyArrow 24 Parquet shards (OHLCV ingestion, feature caching)
- **Numerics:** NumPy 2.3, SciPy 1.16, scikit-learn 1.8
- **Sentiment:** Google GenAI SDK 1.70 (Gemini Flash) — conditional soft overlay + hard bear veto
- **News:** GNews 0.4 + googlenewsdecoder 0.1 + BeautifulSoup 4.14
- **Bot:** python-telegram-bot 22.7 (async PTB framework)
- **HTTP:** aiohttp 3.13, requests 2.33
- **Config:** python-dotenv 1.2 (.env), dataclass-based settings with JSON overrides
- **Testing:** pytest (238 tests, in-memory DuckDB stubs)
- **Deployment:** Bare metal VPS — systemd (bot), cron (daily pipeline at 15:30 ICT Mon–Fri)

## Key Patterns and Conventions

**Typing:** Strict Python 3.10+ type hints everywhere. All function signatures annotated.

**Data processing:** Polars-native feature pipelines. Avoid reverting to Pandas unless interfacing with legacy ML libraries that require it.

**Storage pattern:** DuckDB for analytical queries, Parquet shards for data at rest. `db_engine.py` manages connections.

**Config pattern:** Nested dataclasses (`Config` → `PathConfig`, `ModelConfig`, `TradingConfig`, `CrawlerConfig`, `SentimentConfig`). JSON overrides via `config/settings.json`. Singleton `CONFIG` instance.

**Architecture style:** Pure functions + procedural orchestration. No deep OOP inheritance trees. Prefer extracting pure functions over adding class methods.

**Feature recipe versioning:** `FEATURE_RECIPE_VERSION` in `src/backtest/pipeline.py` (computed via `compute_feature_schema_hash(...)`, currently `v2-sha8:53b5bd85`). Hard gate — serve-path checks match at load time. Any feature engineering change requires: bump version (auto via the schema hash) → full retrain.

**Backtest portfolio construction:** `run_backtest.py` defaults to `--mode tranche --hold-days 30` (staggered AFML cohort book: daily deploy NAV/H into top-`max_positions` names, hold exactly H trading days). Legacy `--mode grid` (concentrated delta-rebalance) is structurally unfit for this signal — its ~45 correlated entry dates let market beta dominate. Price-scale rule: parquet OHLCV is in thousands of VND; the engine converts to absolute VND via `WalkForwardConfig.price_unit_vnd` — any new code feeding parquet prices into `VNCostModel` must do the same. Bot payload carries a `strategy` dict (mode/hold_days/signal_threshold); serve consumes it via `_tranche_signal_fields` (tranche cohort weight `1/(hold_days×n_picks)`).

**Regime-conditional sizing (DD control):** Both backtest and serve apply the market-regime policy from `src/trading/regime_policy.py` — the single source of truth for `NO_TRADE_REGIMES {0,7}` (skip the name, weight stays cash), `PENALTY_REGIMES {1,6}` (× `REGIME_PENALTY_FACTOR` = 0.5 = `REGIME_PENALTY_CAP/DEFAULT_NAV_CAP`), and `STRONG_TREND_REGIME {3}`; imported by both `src/bot/sizing.py` (serve) and `src/backtest/walk_forward.py` (backtest). **Backtest:** opt-in via `--regime-sizing` / `WalkForwardConfig.use_regime_sizing` (default OFF). **Serve:** `main._dispatch_signals` applies it in the non-event-override branch (regime read per-ticker from the `_LATEST_REGIME_BY_TICKER` cache; event overrides keep precedence), gated by `CONFIG.trading.regime_sizing_enabled` (default **ON**, settings.json kill-switch). A/B (2026-06-14, T+20 GOLDEN): MaxDD −23.3%→−16.9%, Sharpe +0.73→+0.88, Net +46%→+42%, DSR 0.35→0.45 (still <0.95 → stays paper-only). No feature-recipe change, no retrain. PENALTY uses a 0.5× *multiplier* (not the absolute `REGIME_PENALTY_CAP`) because tranche per-name (~0.7% NAV) is far below the 10% cap.

**Serve-path horizons:** PRIMARY = T+20 (`v3_ensemble_20d.joblib`). SHORT = T+5 (`v3_ensemble_5d.joblib`), used by `/verify` for intraday confirmation — `SHORT_HORIZON_DAYS = 5` in `src/bot/bot_inference.py` (recovered 18-06-26; was briefly T+3 on 12-06-26, reverted because the 5d artifact was already gate-verified `v2-sha8:53b5bd85` and no retrain was required). Known cleanup debt: `src/reports/builders.py:326` hardcodes literal `5 ngày tới` instead of `{SHORT_HORIZON_DAYS}` — fix pending in the Telegram-work effort.

**Sentiment-entry paper-log:** `sentiment_entry_paperlog` DuckDB table (+ `seq_sentiment_entry_id` sequence) captures the full candidate cross-section on every daily pipeline run (`source='daily'`) and every `/verify` invocation (`source='verify'`). Columns: per-horizon model probabilities, `decision_5d` argmax, `sentiment_score`, `entry_close`, `ret_3d`, `ret_20d`, `outcome_filled`. Backfill is PROGRESSIVE (fixed 2026-06-22, `_backfill_paperlog_outcomes`): `ret_3d` fills once the T+3 window matures (scan gate `_PAPERLOG_SHORT_MATURE_DAYS`=4 calendar days), `ret_20d` at `_PAPERLOG_MATURE_DAYS`=21 days; `outcome_filled` flips TRUE only when the terminal T+20 return lands (rows with only `ret_3d` stay pending). Uses `price_lookup`; see `doc/audit_paperlog_temporal_flaw.md`. Config knobs: `CONFIG.trading.sentiment_entry_enabled` (default True) / `sentiment_entry_threshold` (default 0.7, analysis-time reference only — all rows are logged regardless). Analysis: `scripts/analyze_sentiment_paperlog.py`. Tests: `tests/test_sentiment_paperlog.py` (10 tests). Shipped 2026-06-16; `source='daily'` row not yet confirmed in production (requires one live cron run at 15:30 ICT).

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

**Local dashboard (new program, 2026-06-19):** Streamlit dashboard for a single Windows laptop user. Package root: `dashboard/` (new, not yet created — P1 scope). No-polling architecture: send-only Telegram alerter, no PTB ApplicationBuilder anywhere in `dashboard/`. Reuses `main.daily_inference(broadcast=False)`, `verify_single_ticker`, `inference_for_holdings`, `PortfolioManager`, `signal_ledger`, `audit_evaluator.run_post_mortem`. Installer: Inno Setup `setup.exe` (P4 scope) — no PyInstaller.

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
| Macro Integration (A/B) | `process/features/macro-integration/` | planned (2026-06-22) |

## Code-Review-Graph MCP

A local MCP server (`code-review-graph`) maintains a live graph database of the entire codebase. Use its tools for dependency tracing, blast radius analysis, and hub detection instead of broad grep scans. Key tools:
- `list_graph_stats_tool` — graph health check
- `get_hub_nodes_tool` — architectural hotspots
- `get_impact_radius_tool` — blast radius for a specific node
- `get_affected_flows_tool` — execution flows touching a node
- `query_graph_tool` — arbitrary graph queries

## Scan Metadata

- Generated: 2026-06-09
- HEAD: main (d2d0a56)
- Mode: fresh scaffold + study
- Package manager: pip (requirements.txt, Python 3.11)
