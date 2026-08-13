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
| `tests/` | `process/context/tests/all-tests.md` | pytest runner, 894 tests (75 files), in-memory DuckDB stubs, debugging quick-ref |

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
      walk_forward.py     -- Walk-forward engine: "tranche" mode (AFML staggered cohorts, evaluator default) + legacy "grid" mode; scales thousand-VND parquet prices to absolute VND (price_unit_vnd); burst-budget divisor (`tranche_budget_days`, opt-in, SHIPPED) decouples the daily budget from hold length, optional per-day `budget_days_series` override (vol-scaled variant REJECTED as calibrated)
    bot/
      bot_inference.py    -- V3BotInference (serve-path model loading + prediction)
      sizing.py           -- Half-Kelly position sizing (20% NAV cap)
    crawlers/
      sentiment_crawler.py -- News sentiment crawler
    data/
      crawlers.py         -- OHLCV data ingestion (FastConnect prefetch + throttled vnstock fallback)
      fastconnect_ohlcv.py -- SSI FastConnect concurrent OHLCV prefetch (PRIMARY EOD source, 10-08-26)
      market_breadth_crawler.py -- official exchange breadth via FastConnect DailyIndex (A/D + ceilings/floors + block deals, 12-08-26)
      db_engine.py        -- DuckDB engine management
      price_lookup.py     -- Fresh-parquet price lookups
      tensor_builder.py   -- Feature tensor construction
    execution/
      vn_cost_model.py    -- Vietnamese market cost model (fees, slippage)
    features/
      alpha360_generator.py  -- DEPRECATED (gutted in V4.0)
      market_regime.py    -- 8-regime HMM classifier + regime feature builder
      mr_features.py      -- Mean-reversion (knife-catch) features
      mr_context_features.py -- MR volume-exhaustion + sector-relative oversold (research, PROMISING-UNCONFIRMED)
      impulse_features.py -- Fast-attack/momentum-ignition features (research, REJECTED as a sub-model, kept for reuse)
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
      vol_sizing.py         -- Vol-scaled burst-budget divisor (research, REJECTED as calibrated — miscalibrated triggers, kept for reuse)
    reports/
      __init__.py         -- Re-exports report builders
      builders.py         -- 10 report builder functions + 11 constants (extracted from main.py)
    utils/
      telegram_alerter.py -- Telegram message formatting + delivery
      telegram_bot.py     -- PTB application builder (commands, handlers)
      logging_utils.py    -- Centralized logging setup
      audit_evaluator.py  -- Trade audit evaluation
      version.py          -- Version string
  tests/                  -- 75 test files, 894 tests (pytest)
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
    analyze_ranking_objective.py -- LGBMRanker vs pointwise admission research (REJECTED -- 2x DD, PBO 66.7%)
    analyze_confluence_signal.py -- T+5/T+20 agreement paperlog research (INCONCLUSIVE -- confluence-UP bucket empty)
    analyze_confluence_backtest.py -- T+5/T+20 agreement OOS backtest (confluence REJECTED; found T+20 gate does all the quality work)
    analyze_concentration_ab.py -- Absolute-gate concentration A/B (barbell wins on risk, loses on PnL -- motivated burst sizing)
    analyze_burst_sizing_ab.py -- Burst-budget divisor A/B (nav/10 SHIPPED opt-in, ~3x concentration PnL inside the DD comfort band)
    analyze_vol_scaled_burst_ab.py -- Vol-scaled burst-divisor A/B (REJECTED -- miscalibrated, worse than flat nav/10)
    analyze_impulse_attack.py -- Fast-attack/momentum-ignition sub-model research (REJECTED -- OOF precision misses both bars)
    analyze_mr_context_features.py -- MR volume-exhaustion + sector-relative oversold research (PROMISING, UNCONFIRMED -- clears 0.60 OOF, holdout too thin)
    analyze_dsr_calibration.py -- DSR/PBO calibration diagnostic (read-only -- failure is ROBUST, not inflated trials or small-sample noise)
    analyze_serve_stack_ab.py -- serve-vs-backtest divergence A/B (argmax gate = off switch; ranking/cohort vacuous)
    analyze_defensive_layers_ab.py -- the 4 production defensive layers measured (all cost Sharpe, 8/8 cells)
    analyze_gate_level_sweep.py -- gate 0.41..0.46 x mode x layers (modes identical; 0.43-0.44 beats serve's 0.46)
    analyze_mr_limit_down_context.py -- limit-down knife-catch timing (UNTESTABLE: floors is a ~1.5-year series)
    analyze_argmax_admission_ab.py -- argmax admission A/B (REJECTED: 0 buys in 920 days)
    backfill_market_breadth.py -- DailyIndex breadth backfill (2016-01-04..now, 7200 rows)
    retrain_all.ps1       -- Full 3-model retrain (MR+T20+T5); scheduled weekly via Windows Task Scheduler.
                             Runs `--serve-parity` since 12-08-26 (gate-offset 0 + 4 defensive layers, 6-level grid)
    ab_rank_breadth.ps1   -- rank_breadth admission A/B runner (REJECTED)
  deploy/
    quant-v6-bot.service  -- Systemd unit for run_bot.py
  doc/
    SYSTEM_DESIGN.md      -- SUPERSEDED (flagged 08-08-26, banner added) -- describes the pre-V4.0 Alpha360-stacking architecture; kept as historical design-rationale, not current-state truth
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
- **Testing:** pytest (894 tests, in-memory DuckDB stubs)
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

**"Smarter system" research marathon (2026-07-22/24) — 9 experiments, 1 shipped win, 2 promising-unconfirmed, rest REJECTED/INCONCLUSIVE:**
- **Ranking objective** (`scripts/analyze_ranking_objective.py`) — REJECTED. `LGBMRanker` (lambdarank) vs pointwise: Sharpe +0.554 vs +0.545 (noise-level) but MaxDD −26.56% vs −13.31% (nearly double) and PBO 66.7% (massive overfit). Caveat: minimal single-model test, not a fully clean kill of the idea itself.
- **Confluence** (`scripts/analyze_confluence_signal.py` paperlog version, then `scripts/analyze_confluence_backtest.py` full-OOS version) — confluence itself REJECTED (Confluence-UP bucket empty in the paperlog; backtest version shows adding the T+5 gate on top of T+20 makes picks slightly WORSE, not better). **Real finding: the T+20 gate does ALL the quality work — T+20-gated ≈ 3x base rate, T+5-gated ≈ noise.** Directly explains why all 13 July-2026 losses came via the T+5 side-door.
- **Concentration A/B** (`scripts/analyze_concentration_ab.py`) — the `absolute_gate≥0.46` barbell matches cross_sectional's Sharpe at 1/6.6 the DD but only 1/8.4 the PnL, because the gate opens ~5% of days and the calendar budget (nav/30) barely deploys. Motivated burst sizing below.
- **Burst-budget divisor** (`scripts/analyze_burst_sizing_ab.py`) — **SHIPPED opt-in** (`WalkForwardConfig.tranche_budget_days`, default unchanged). nav/10 (3x calendar budget) on gate-open days: NetPnL +1.95B vs the nav/30 reference's +661M (~3x), Sharpe flat (0.594 vs 0.600), DD −12.86% lands just inside the ~−13% production comfort band. nav/5 (6x) overshoots the band for a worse Sharpe.
- **Vol-scaled burst divisor** (`scripts/analyze_vol_scaled_burst_ab.py`, `src/trading/vol_sizing.py`) — REJECTED. Dynamic divisor (calm→smaller, stressed→bigger, via `market_vol_time_series`) landed at mean divisor 17.7 (not 10) — trigger levels were guessed absolutes, not calibrated to VN market's real vol distribution, so it sat conservative and lost to flat nav/10 on both PnL and Sharpe.
- **Fast-attack / impulse sub-model** (`scripts/analyze_impulse_attack.py`, `src/features/impulse_features.py`) — REJECTED. Tail-event label (setup ∧ 3-bar return >+3%) mirroring MR-LGBM's exact scaffold: OOF precision 0.520 misses both the 0.60 target and MR-LGBM's own 0.578 bar. Volume-confirmed momentum doesn't predict continuation the way oversold+volume predicts a bounce.
- **MR context features** (`scripts/analyze_mr_context_features.py`, `src/features/mr_context_features.py`) — **PROMISING, UNCONFIRMED** (same bucket as breadth-inflection). Volume-exhaustion + sector-relative-oversold (reuses `sector_map.sector_of`) added to the shipped 11 MR features: OOF precision 0.642 (+6.3pp, first config to clear 0.60), holdout 0.800 — but holdout fires collapse to n=5, too thin to trust. NOT wired into serve.
- **DSR/PBO calibration diagnostic** (`scripts/analyze_dsr_calibration.py`, read-only, no formula changed) — the production sweep grid is genuinely diverse (mean daily-return correlation 0.420, NOT near-clone configs) so `n_trials` is a fair multiplicity count, not inflated. Block-bootstrap (2000 resamples): only 0.4% pass DSR even though bootstrap Sharpe ranges up to +1.42 at p95. **DSR failure is ROBUST, not a small-sample artifact** — the edge itself needs to be structurally bigger; further sizing/admission knobs are unlikely to close this gap.
- **TWAP execution-cost** — rejected analytically, no script needed. Tranche fills are always ATC (`walk_forward.py`'s order construction sets `is_atc=True` unconditionally); the cost model's ATC branch has zero size-based slippage (flat clearing price, only a volume cap) — no impact exists to reduce.

**Foreign-flow data via SSI FastConnect — researched, blocked on user action (2026-07-24, see `process/general-plans/backlog/foreign-flow-fastconnect-integration_PLAN_24-07-26.md`):** SSI's OFFICIAL documented API (distinct from the unofficial `iboard-query.ssi.com.vn` scrape already in `foreign_flow_crawler.py`) exposes historical, date-range-queryable foreign buy/sell volume + foreign room via its `DailyStockPrice` endpoint — free, but needs an SSI trading account (fully online eKYC, ~3min) PLUS a separate FastConnect API registration (branch visit or mailed docs, not self-serve). The existing `foreign_flow_crawler.py`/`flow_features.py`/`eda_flow_features.py` trio is confirmed schema-source-agnostic — zero rework needed once a backfill adapter exists; only new work is that one adapter (token auth + `DailyStockPrice` calls). Not started — user has no SSI account yet.

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

**EOD OHLCV crawl — FastConnect primary, vnstock fallback (2026-08-10):** the 15:30 ICT pipeline spent **25.4 of its 29.5 minutes** crawling OHLCV, and nearly all of it was deliberate sleeping — vnstock guest is capped ~20 req/min so `crawlers._throttle_request` paces at 4.25s/ticker even though one `Quote.history` call measures 0.42s. The network was never the bottleneck; the rate limit was. `src/data/fastconnect_ohlcv.py::bulk_prefetch` runs one concurrent pass (4 workers, shared sliding-window limiter at 55 req/min) before the existing serial loop, which then does disk work only. **Live-measured 10-08-26: crawl 1523s → 381s (4.0x), 354/357 prefetched, 4 throttle sleeps instead of 357.** Full pipeline 29.5 min → ~10 min.
- **Serial FastConnect is NOT a win** (2.77s median latency × 359 = 16.6 min) — the concurrency is the point, and the *shared* limiter is what makes it safe. A per-thread limiter would let 4 workers each take the full budget.
- **Value parity verified before wiring** (a faster source that disagrees on price would be worthless — every feature and label is built on these numbers): O/H/L/C matched vnstock to **0.0000%** across VCB/HPG/FPT/SSI/VNM × 9 sessions; volume matched exactly on 4 of 5 (VCB differed 0.372%, block-deal accounting). FastConnect also respects the requested window exactly where vnstock over-returns earlier dates.
- **Window-sufficiency guard:** a ticker is served from the prefetch only when the window provably reaches back past its local max date. Serving a 30-day prefetch to a ticker 60 days behind would advance its parquet and orphan the gap forever (the next run reads only the new max). Verified live — `DMX: gap starts 2016-01-01, before the prefetch window 2026-07-12 — using the slow path`. Cold starts deliberately stay on vnstock so the fast path carries no chunking/paging logic.
- **Dtype hazard (caused a production crash the same day, fixed in `ded6e4c`):** vnstock returns int64 volume, FastConnect float64 — one crawl split the 359 shards into two parquet schemas (345 double / 15 int64) and `alpha360_generator._load_live_stock_window`'s `pl.concat(how="diagonal")` died with `SchemaError: type Int64 is incompatible with expected type Float64`. Fixed at BOTH ends: `crawlers._merge_and_save` casts numerics to float64 on every write, AND the loader casts per shard on read — the writer alone is insufficient because several int64 shards belong to inactive tickers that will never be rewritten. **Any new OHLCV source must go through `_merge_and_save`.**
- Kill-switch `CONFIG.crawler.fastconnect_ohlcv_enabled` (default ON) restores the pure-vnstock path with no other change. Credentials are the same `Consumer_Key`/`ConsumerSecret_Key` as the foreign-flow module. Tests: `tests/test_fastconnect_ohlcv.py` (24).
- **Remaining 15:30 cost is external-API-bound, deliberately not optimized:** sentiment ~2.4–3.3 min (GNews per-ticker loop has an intentional politeness sleep — concurrency risks a Google block that would kill sentiment entirely; the 5 Gemini calls are paid and were already cost-tuned) and inference ~1.3 min (dominated by the arbitrator's news scrape + Gemini). Chasing those ~2 min means risking the paid path and the scrape-sensitive path for marginal gain.

**⚠️ Paperlog horizon-column swap — invalidates four earlier conclusions (2026-08-10, fixed in `7df162e`):** `sentiment_entry_paperlog`'s probability columns were named by horizon but always held PRIMARY/SECONDARY. `daily_inference` sets `stacking_predictions_5d = predict_v3_horizon(latest_df, horizon)` — the PRIMARY, i.e. **T+20** on the daily cron — and `stacking_predictions_20d` the SECONDARY (T+5); the writer mapped those straight onto `p_up_5d`/`p_up_20d`, so **`p_up_20d` held T+5 values and `p_up_5d` held T+20**. `/verify` compounds it: its PRIMARY is `SHORT_HORIZON`, so `source='verify'` rows carried a different horizon in the same column with nothing in the data to distinguish them. Columns are now `*_primary`/`*_secondary` plus a per-row `primary_horizon_days`; migration is an idempotent `ALTER TABLE` (pure relabelling — no value reinterpreted, 4690 live rows preserved).
- **`check_drift.py` was the live casualty:** it read `p_up_20d` (T+5) against `REF_PUP`, documented as the T+20 teardown reference, and reported no drift. The calibration monitor was not measuring calibration.
- **These four documented conclusions are now UNVERIFIED pending a re-run** — do not cite them as settled: (1) `analyze_confluence_signal.py`'s "the T+20 gate does all the quality work, T+5 ≈ noise" compared the horizons with labels swapped and **may be inverted**; (2) `analyze_probability_calibration.py`'s "T+20 ~15.7pp optimistic, T+5 well calibrated" paired T+5 probabilities with 20d returns; (3) `analyze_rank_sleeve.py` ranked on the wrong horizon; (4) `backfill_dispatched_signals_from_paperlog.py` derived `decision_20d` from the wrong horizon **and writes to `dispatched_signals`** — check that table if it was ever run.
- **Paperlog measures the wrong population for strategy evaluation:** it logs the full cross-section including the monitoring-only fallback branch, which is deliberately NOT liquidity-gated. Of 44 settled BUY rows only **1** is in the ADV top-50; the τ≥0.46 liquid count over 7 weeks is **n=1**. Any tradeable-performance claim read off this table without a liquidity filter is an artifact — a decile table showed d10 `+0.56%` on all rows but **−1.84%** on liquid names only, with d1 the best decile.

**⚠️ GATE STARVATION — τ is calibrated on a different population than it is applied to (2026-08-10):** `check_drift.py` now carries a tail-vs-gate check, because an absolute-threshold strategy is decided by the UPPER TAIL, not the median, and the median-only check reported OK while the gate was shut:

```
serve  p10=0.343 median=0.393 p90=0.423      median shift -0.015 → OK
OOSref p10=0.321 median=0.408 p90=0.463      p90 shift -0.040
🔴 GATE STARVED: p90=0.423 is BELOW tau=0.46  (1.1% of scores reach tau)
```

τ=0.46 was swept on a distribution whose p90 was 0.463; serve's p90 is now 0.423, so **the gate sits above the 90th percentile of the model's own output** and the book stays empty regardless of signal quality. **It is NOT a population mismatch** — `run_backtest.py`'s sweep already runs with `liquid_top_n=50`, so τ was calibrated on the tradeable universe, and the 08-08 retrain's sweep still picked 0.46 on fresh data. It is **distribution drift against a frozen artifact**, and the chain is:

**PROMOTE-GATE DEADLOCK — the live T+20 artifact has been frozen since 17-07 (2026-08-10):** every retrain since has been rejected (`models/saved/rejected/`): 26-07 both horizons, 08-08 T+20 (`MaxDD regression: new=19.26% old=12.60%, max +3.0pp`) and T+5 (`Sharpe regression: new=0.489 old=0.590`). Four consecutive rejections, so `v3_ensemble_20d.joblib` is `trained_at=2026-07-17` and `v3_ensemble_5d.joblib` 19-07 while the market moved underneath them. **The gate compares the candidate's fresh OOS metrics against the incumbent's OWN stored metadata — metrics measured on a DIFFERENT, earlier OOS window.** The incumbent's −12.60% MaxDD was measured on a window ending 17-07, before the worst of the decline; any candidate trained now is scored on a window that includes it, so it looks worse by construction. Left alone the gate rejects every future retrain indefinitely and the system stays frozen on 17-07 forever. **Fix: evaluate the incumbent on the CANDIDATE's OOS window before comparing (or compare both against a same-window benchmark) — a like-for-like comparison, not frozen-vintage vs current.** The gate's intent is right; its comparison basis is not.

Serve-side extra gates that the backtest never modelled compound the low dispatch count — arbitrator argmax+sentiment (killed 69% of τ-clearing rows), sector cap, open-cohort dedup, admission hysteresis: the validated strategy and the deployed strategy are not the same strategy.

**⚠️ SERVE ↔ BACKTEST DIVERGENCE — the validated numbers did not describe the deployed system (2026-08-11/12).** Eleven differences were found between the rule the OOS figures were earned with and the rule production runs. Measured with `scripts/analyze_serve_stack_ab.py`, `analyze_defensive_layers_ab.py`, `analyze_gate_level_sweep.py`.

**The threshold conflation (the root cause).** The GOLDEN artifact stores `up_threshold` (0.46) and `signal_threshold` (0.41). Inside `run_backtest.py`, `up_threshold` feeds ONLY the UP-precision / confusion-matrix metrics — the engine trades on `signal_threshold`, historically pinned 5pp below via `sig_thr = thr - 0.05`. **Serve gates on `up_threshold`** (`predict_v3_horizon`'s `meta_gate` reads `bot.up_threshold`), so the number production admits on was never the number the sweep optimised. Not a 5pp slip — a category error.

**`admission_mode` was never a real variable.** `absolute_gate` and `cross_sectional` produce byte-identical results at every gate level, because `admission_pool_cap` (6) ≥ `max_positions` (5) so the pool cap never binds. Earlier framings of "0.41 cross_sectional vs 0.46 absolute_gate" were really just "0.41 vs 0.46".

**Every defensive layer costs Sharpe — 8 of 8 cells, both thresholds:**

| threshold | layers | NetPnL | Sharpe | MaxDD | buys |
|---|---|---|---|---|---|
| 0.41 (validated) | none | +5.19B | +0.600 | −31.00% | 4555 |
| 0.41 | brake | +4.18B | +0.557 | −29.17% | 4555 |
| 0.41 | filters | +2.08B | +0.412 | −21.86% | 1248 |
| 0.41 | all four | +1.79B | +0.393 | −22.48% | 1248 |
| 0.46 (serve) | none | +644M | **+0.620** | −4.44% | 46 |
| 0.46 | brake | +444M | +0.480 | −4.19% | 46 |
| 0.46 | filters | +97.6M | +0.355 | −0.91% | 5 |
| **0.46 all four = SERVE** | | **+87.6M** | **+0.340** | −0.87% | **5** |

The deployed stack retains **1.7%** of the validated PnL and **0.11%** of the bets — five trades in 920 days. The best Sharpe of all eight is 0.46 with NO layers, so **the τ-gate is the only defensive layer that pays for itself**; the four added later (each after a real loss, none ever measured) all trade Sharpe for drawdown roughly one-for-one.

**Gate-level sweep 0.41–0.46 (`analyze_gate_level_sweep.py`) — SUPERSEDED, do not cite.** It reported 0.44 → Sharpe 0.693 / 0.43 → 0.633 / a 0.45 collapse, on ONE seed of the FROZEN 17-07 GOLDEN ensemble. The real production path disagrees on the levels, the ranking, and the shape (below).

**⚠️ THE THRESHOLD DOES NOT AFFECT SHARPE (12-08-26, full parity sweep).** `run_backtest.py --serve-parity --sweep-thresholds 0.46,0.45,0.44,0.43,0.42,0.41 --no-save`, 4 seeds each, 922 OOS days (`logs/parity_sweep_full_20260812_154135.log`):

| up_thr | mean NetPnL | mean Sharpe | mean DD | predUP | UPprec |
|---|---|---|---|---|---|
| 0.46 (serve) | +907M | +0.427 | **−9.66%** | 139,561 | 0.4674 |
| 0.45 | +1.218B | +0.470 | −12.20% | 207,649 | 0.4616 |
| 0.44 | +1.252B | +0.419 | −13.17% | 295,511 | 0.4568 |
| **0.43 ← GOLDEN** | **+1.583B** | +0.475 | **−13.88%** | 399,754 | 0.4543 |
| 0.42 | +1.554B | +0.454 | −14.83% | 511,392 | 0.4513 |
| 0.41 | +1.453B | +0.425 | −16.20% | 620,486 | 0.4477 |

Sharpe spans **0.056** across all six levels and zigzags; a single level's 4-seed spread is **0.30**. The entire curve fits inside one level's noise, so any Sharpe-based ranking of these levels — including every earlier one in this file — was reading noise. **DD and precision ARE cleanly monotone** (−9.66%→−16.20%, 0.4674→0.4477), so the real trade is PnL against drawdown and Sharpe is flat *because* the two rise together. This also retires "0.46 is past the peak": there is no peak, and 0.46 posts the best DD and best precision of the six. What 0.46 costs is deployment — 57% of the PnL — and its live problem is gate starvation (serve p90 0.423 < 0.46), i.e. drift against a frozen artifact, not a bad level.

**Serve-parity does NOT rescue the statistical gates.** GOLDEN teardown (best seed 45): Sharpe +0.535, DD −12.54%, +18.26%/922d, **DSR p=0.1743 (FAIL <0.95), PBO 85.0% (FAIL >10%) → UNFIT FOR PRODUCTION.** Consistent with the 24-07 diagnostic that DSR failure is ROBUST. Paper-only stands.

**⚠️ UNGUARDED — GOLDEN selection has no drawdown constraint.** `run_backtest.py` picks GOLDEN as max mean OOS Net PnL, no DD term. Here that chose 0.43 at mean DD **−13.88%**, already outside the ~−13% comfort band, with 0.42 only 1.8% behind on PnL (inside noise) at −14.83%. Since Sharpe carries no signal and PnL/DD rise together, a max-PnL objective **systematically pushes toward maximum drawdown**; another seed draw picks 0.42 or 0.41 (−16.20%). The promote-gate cannot catch it — `max_abs_dd_pp` is 25%, deliberately loose so it only blocks runaways. Fix belongs in SELECTION (max PnL *subject to* a mean-DD budget), not promotion. NOT implemented — it changes what the unattended Saturday run optimises.

**FIX SHIPPED:** `run_backtest.py --serve-parity` (`--gate-offset 0` + the four layers) makes the sweep optimise the threshold under production conditions, so the stored `up_threshold` becomes the value actually optimised. Wired into `scripts/retrain_all.ps1` with a 6-level grid `0.46..0.41` (deliberately no wider than the 5 it replaces — DSR penalises trial count). Artifacts now carry `metadata.sweep_conditions` recording gate offset, admission mode, layer flags, regime sizing and max_positions; its absence in every earlier artifact is how this went unnoticed for months.

**Promote-gate needed a second repair for any of this to land (`11adbad`).** The 10-08 fix stopped comparing MaxDD across nested OOS windows but left Sharpe on a bare relative check — and `--serve-parity` changed the *strategy* measured, not just the window: the four layers cost ~0.2 Sharpe, double the −0.10 delta allowed. The live incumbent's 0.645 was earned BARE, so it rejected every parity candidate on arrival (needs ≥0.545; dry sweep delivered 0.540). Now the relative Sharpe check runs ONLY when both artifacts share a `_sweep_basis` (gate offset, the four layer flags, regime sizing, max_positions — `engine_gate` excluded because it IS the optimised output, `admission_mode` excluded because it was measured vacuous); otherwise absolute floors decide and it logs at WARNING. Two UNSTAMPED artifacts count as the SAME basis (both predate the stamp, both swept bare) which is what keeps the genuine 08-08 T+5 regression a reject. One-time widening by construction. Replay tool: `scripts/replay_promote_gate_from_sweep_log.py` (read-only, prints old-gate vs new-gate verdicts side by side).

**Also fixed at the entry gate (`4081965`):** `CONFIG.trading.arbitrator_entry_mode = "veto"` (new default). The old behaviour required `make_final_decision` to return BUY, which needs the primary horizon's ARGMAX to be UP — measured over 920 days that takes the validated config from 46 buys to **ZERO** (p_up p99 0.4523 < p_down median 0.5957, so UP cannot win the argmax). It was an off switch, not a filter, and it forced every real dispatch through the event-rescue path.

**STILL NOT MODELLED — the event-rescue path.** `main.build_event_overrides`: a name the τ-gate REJECTED (`0.42 ≤ P(UP) < 0.45`) is force-dispatched at **5% NAV** (`_EVENT_CAP`, ~7× the tranche per-name weight) when Gemini sentiment ≥ 0.60, and the `if _ov:` branch **skips regime sizing, the PENALTY multiplier and the whole 3-leg exposure brake**. It admits below the model's own threshold, in the P(UP) band measured as anti-informative (Platt slope −0.274). Not backtestable — sentiment has no point-in-time history. Deferred until the paperlog has enough settled rows.

**Ranking and cohort size are vacuous, not divergences.** Serve sorts the arbitrated pool by SENTIMENT with p_up as a tiebreak and slices top-3 vs the backtest's top-5. Random-ranking arms across 3 seeds and a top-3 arm came back byte-identical to the baseline: under the τ-gate the admitted set never exceeds 3 names (46 buys over ~46 gate-open days ≈ 1/day), so there is nothing to reorder or truncate. This changes if the gate is ever loosened.

## Scan Metadata

- Generated: 2026-06-09
- Last content update: 2026-08-12
- HEAD: main (fba7459)
- Mode: fresh scaffold + study
- Package manager: pip (requirements.txt, Python 3.11)
