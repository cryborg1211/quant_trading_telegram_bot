# Quant Engine V4.0 — Full Context Dump

Vietnamese-equity quant trading assistant. Pure **tabular stacking ensemble**
(LightGBM + XGBoost + CatBoost → calibrated LogisticRegression meta) on a VN50
dynamic universe, triple-barrier labels, **T+5 / T+20** dual horizon, half-Kelly
sizing with an 8-regime structural overlay, an HMM macro-risk gate, and a
mean-reversion ("knife-catch") sub-model. Served to users via a Telegram bot.

Pure tabular by design — **no deep learning** (overfitting risk on VN's short,
noisy history). Everything is Polars/numpy + gradient-boosted trees.

---

## 0. Architecture at a glance

```
data/ohlcv_*.parquet  ── SINGLE SOURCE OF TRUTH for OHLCV (per-ticker shards)
        │
        ├─▶ train_models.py   (HEAVY ~40min)  ─▶ models/saved/v3_training_checkpoint.joblib
        │      ingest → features → labels → split → iron-fist select → HMM → 4-seed ensemble
        │
        └─▶ run_backtest.py   (FAST)          ─▶ models/saved/v3_ensemble_{5,20}d.joblib
               load checkpoint → threshold sweep → DSR/PBO → FIT gate → persist live payload
               (--export-only: skip sweep, repackage artifact in seconds)
        │
run_bot.py ─▶ main.py (daily_inference / serve) ─▶ Telegram cards
```

Two decoupled scripts communicate ONLY through the checkpoint joblib. Train/serve
parity is **structural**: both build features through the one shared module
`src/backtest/pipeline.py`.

---

## 1. Repo tree (key modules)

```
main.py                         Orchestration + live serve: daily_inference, predict_v3_horizon,
                                run_trade_execution, verify/suggest_sell/rebalance report builders,
                                full_pipeline (EOD), crawl_hose, RL logging, _smart_truncate
run_bot.py                      Telegram polling bot entrypoint (long-running service)
train_models.py                 HEAVY trainer → checkpoint
run_backtest.py                 FAST evaluator + artifact packager (--export-only)
conftest.py                     repo-root sys.path shim for pytest

src/backtest/pipeline.py        SHARED dataset library (load_ohlcv, build_features, labels, align,
                                chronological_split, select_features, subset_features, RunConfig,
                                FEATURE_RECIPE_VERSION, CATEGORICAL_FEATURES)
src/backtest/walk_forward.py    WalkForwardEngine (OOS sim + inference cache)
src/features/market_regime.py   8-regime rule-based Polars classifier (build_regime_features)
src/features/mr_features.py     mean-reversion/oversold features (RSI/BB/ATR/Williams, Wilder)
src/features/triple_barrier.py  (also src/labels/triple_barrier.py) AFML triple-barrier labeller
src/data/tensor_builder.py      FracDiff + cross-sectional Gaussian-rank Z + alpha factors + adv stats
src/data/price_lookup.py        parquet-first point price/volume lookups (read_parquet)
src/data/crawlers.py            StockCrawler (vnstock) → writes data/ohlcv_*.parquet
src/data/db_engine.py           DuckDB singleton + schema init (live tables only)
src/crawlers/sentiment_crawler.py  daily LLM news sentiment → hist_sentiment_llm_labeled
src/models/tabular_ensemble.py  TabularEnsemble (LGB+XGB+CAT → LogReg meta, calibrated, categorical)
src/models/macro_risk_hmm.py    2-state HMM on a PRICE market proxy → P(Bull) gate
src/models/quant_agent_arbitrator.py  Gemini news-analyst + final BUY/HOLD/SELL arbitration
src/models/train_mr_lgbm.py     trains the MR sub-model → models/mr/
src/models/stacking_model/purged_kfold.py  PurgedKFold (de Prado §7, embargo=horizon)
src/bot/sizing.py               half-Kelly + regime overrides (pure functions)
src/bot/bot_inference.py        V3BotInference — loads v3_ensemble_{H}d.joblib, predict_proba_3class
src/trading/portfolio_manager.py  per-user portfolio (DuckDB `portfolio`)
src/utils/telegram_alerter.py   TelegramBot._build_message (signal card), format_source_links
src/utils/telegram_bot.py       command handlers (suggest_buy/sell, verify, audit_*, msg_id2, ...)
src/utils/audit_evaluator.py    /audit_weekly + /audit_monthly post-mortem (price_lookup)
src/execution/vn_cost_model.py  VN T+2.5 cost/inventory model
src/portfolio/construction.py   MV/Kelly portfolio construction (backtest side)
config/settings.py              CONFIG (paths, crawler, sentiment/Gemini knobs)
tests/                          130 pytest (sizing, cards, serve resilience, feature serve,
                                market regime, main logic, arbitrator, telegram split)

data/ohlcv_<TICKER>.parquet     ⚠ OHLCV SINGLE SOURCE OF TRUTH (~355 shards, gitignored)
data/quant_v6_core.duckdb       live bot state ONLY — no price data
models/saved/v3_ensemble_{5,20}d.joblib   LIVE bot payloads
models/saved/v3_training_checkpoint.joblib  transient train→backtest handoff
models/mr/                      MR sub-model (mr_lgbm.joblib + mr_threshold.json)
```

(`.claude/worktrees/*` is a stale throwaway worktree — ignore.)

---

## 2. Data layer — Parquet-first

**Single source of truth for OHLCV = `data/ohlcv_*.parquet`** (per-ticker shards).
`StockCrawler` writes them every EOD; *everything* reads them:

- **Training / backtest** — `pipeline.load_ohlcv` globs the shards (`pl.scan_parquet`).
- **Live serve features** — `Alpha360Generator.load_live_ohlcv_window` reads the same shards.
- **Point price / volume lookups** (RL outcome backfill, `/audit_*`, sentiment-crawl
  liquidity ranking) — `src/data/price_lookup.py` via DuckDB `read_parquet('data/ohlcv_*.parquet')`.

The legacy DuckDB `stock_ohlcv` + `macro_daily` tables were **DROPPED** (a DB audit
found `stock_ohlcv` ~18 days stale — the crawler only ever wrote the parquet). Also
dropped: `sentiment_score`, `macro_economic_raw`, `live_positions` (dead/deprecated).
`data/alpha360_features.parquet` (2.25 GB) deleted; the whole Alpha360 feature factory
was retired.

**DuckDB `data/quant_v6_core.duckdb` now holds 5 live tables only:**

| Table | Purpose |
|---|---|
| `hist_sentiment_llm_labeled` | Gemini-scored daily news sentiment (read by the arbitrator). ⚠ thin (~1wk backfilled). |
| `portfolio` | Per-user live holdings (Telegram user_id; `'cron'` for the bot). |
| `trade_history` | Append-only trade log. |
| `rl_mistake_logs` | Phase-3 RL data: two-phase (INSERT NULL outcome at T0 → UPDATE real return via `price_lookup` at T+5). |
| `audit_log` | Per-user command audit (drives `/audit_*`). |

DuckDB rule: bare `duckdb.connect(path)` everywhere (no `read_only=True`, no
mismatched `config={}` — DuckDB rejects mixed-config connections in one process).

---

## 3. Feature pipeline (`pipeline.build_features`)

From RAW OHLCV, per-ticker + cross-sectional. All features are **cross-sectional
Gaussian-rank Z-scores** (`_xsz`), so the GBMs see no raw price levels:

- FracDiff(d=0.4) of close & volume (stationary, memory-preserving) → `_xsz`
- `mom20` (20-bar momentum) → `_xsz`
- anti-FOMO over-extension (ma 5/20) → `_xsz`
- alpha factors: relative strength (rs 10/20), smart-money, vol-squeeze → `_xsz`
- advanced stats candidates: amihud_liquidity, realized_skewness_20d, vol_of_vol_20d,
  hl_range_ratio, gap_risk → `_xsz`
- **`market_regime`** — integer 0–7 categorical (see §4), forced-survive.

**Pool → selection:** 9 baseline originals + 5 candidates + 1 categorical. The
iron-fist `select_features` (collinearity |r|>0.65 → mutual-info → top-3) runs on
the TRAIN split only and caps candidates at 3 → **final = 13 features** (9 + 3 +
`market_regime`). `market_regime` BYPASSES corr/MI (always survives, declared
categorical). Seasonality + macro features are OFF (calendar-memorisation / macro
handled by the HMM).

**FEATURE_RECIPE_VERSION = "v1.1"** (bumped from v1.0 when `market_regime` was added).
The trained artifact stamps it; the live bot asserts a match at load and refuses a
drifted model (loud `RuntimeError`). Any v1.0 artifact is rejected → must rebuild.

`frac_diff_d` is the only config-driven recipe knob — stamped in the artifact and read
back at serve (never a library default).

---

## 4. Market regime (`src/features/market_regime.py`)

`build_regime_features(lf: pl.LazyFrame) -> pl.LazyFrame` — pure rule-based Polars
(`pl.when().then()`), zero ML. Classifies each row into `market_regime` ∈ {0..7}
from raw-OHLCV indicators (ATR, RSI, Bollinger bandwidth/%B, Efficiency-Ratio as the
ADX proxy, volume-z, wick/body ratio), all relative/per-ticker → scale-free, leak-free
(`.over("ticker")`, windows end at t). Warm-up rows default to Choppy (non-null).

| id | regime | VN label | trigger (heuristic) |
|----|--------|----------|---------------------|
| 0 | Freeze | Đóng Băng | low ATR & low volume |
| 1 | Squeeze | Tích Lũy (Nén) | BB bandwidth < 0.5× its 60d mean |
| 2 | Early Trend | Khởi Đầu Xu Hướng | band break + volume |
| 3 | Strong Trend | Xu Hướng Mạnh | ER>0.5 + MA aligned |
| 4 | Climax | Cao Trào | above upper band + volume spike + hot RSI |
| 5 | Mean Reversion | Hồi Quy Trung Bình | RSI<30 or >70 |
| 6 | Choppy | Đi Ngang (Nhiễu) | ER<0.3 (also the default) |
| 7 | Liquidity Sweep | Quét Thanh Khoản | big-range bar + tiny body (long wicks) |

`REGIME_LABELS_VI` + `regime_label_vi()` provide the Telegram labels.

---

## 5. Models

**Stacking ensemble (`TabularEnsemble`):**
- Level 1: XGBoost + LightGBM + CatBoost (multiclass 3-class {DOWN,FLAT,UP}).
- Level 2: LogisticRegression meta (C=5.0, class_weight balanced) on the OOF 9-prob matrix.
- OOF via **PurgedKFold** (n_splits=5, embargo = label horizon — AFML §7).
- **Calibration**: `CalibratedClassifierCV(sigmoid, cv=5)` on the OOF meta-matrix —
  squashes over-confident raw probs (~0.70–0.78) to realistic ~0.51–0.55 so half-Kelly
  doesn't pin at the cap. Falls back to raw meta if classes too thin.
- **Categorical**: `categorical_features=["market_regime"]` threaded to LightGBM
  (`categorical_feature=`) + CatBoost (`cat_features=`), int-coerced (round/clip) at
  every fit/predict; XGBoost treats it as a numeric ordinal. Pickled with the ensemble
  → serve needs no special handling.
- Missing-class force-fit: tiny-weighted synth rows at the feature-mean satisfy the
  XGB/LGBM `sum_weight` validator when a fold lacks a class.

**Macro Risk HMM (`macro_risk_hmm.py`):** 2-state HMM on a PRICE market proxy
(`build_market_proxy_returns`) → leak-free filtered P(Bull) overlay. NOT macro data.

**MR sub-model (`train_mr_lgbm.py`):** separate LightGBM on oversold/capitulation
features (`mr_features.py`) → `models/mr/`. Live `mr_score_tickers` flags knife-catch
panic (🔪) — a parallel signal, deliberately NOT mixed into the main stack.

---

## 6. Train → backtest → deploy

```bash
# HEAVY (data/feature/label/arch change). Per horizon:
python train_models.py --tb-horizon 20 --n-configs 4   # → v3_training_checkpoint.joblib
# FAST (iterate eval/sizing freely; FIT gate = DSR p≥0.95 AND PBO≤10%):
python run_backtest.py                                  # → v3_ensemble_20d.joblib
# Artifact-only refresh after retrain (seconds, no sweep):
python run_backtest.py --export-only                    # alias --skip-backtest
```

Dual horizon = run the pair twice (`--tb-horizon 5` then `20`). Checkpoint is
single-horizon + transient; the bot loads BOTH `v3_ensemble_5d.joblib` +
`v3_ensemble_20d.joblib`. `run_backtest.py` auto-backs-up the existing artifact to
`models/saved/backups/` before overwrite. `--no-save` for dry runs. `--export-only`
stamps OOS metrics NaN + preserves the existing artifact's tuned thresholds.

**RunConfig V4 defaults:** parquet_glob `data/ohlcv_*.parquet`, train_frac 0.70,
frac_diff_d 0.4, tb_horizon 20, tb_pt 3.0σ, tb_sl 2.0σ, n_configs 4, liquid_top_n 50
(VN50 gate), max_weight 0.20, max_positions 5, signal_threshold 0.35, target_vol 0.15,
kelly_fraction 0.5, cscv_S 12, seed 42, use_macro_hmm True, hmm_n_states 2.

---

## 7. Serve path (`main.py`)

`daily_inference` (no crawl): `load_live_ohlcv_window` (120-bar tails) →
`predict_v3_horizon(5)` + `(20)` → liquidity gate (VN50 ADV) → Top-6 candidates →
arbitrator + Gemini sentiment → Top-3 (sentiment DESC, then P(UP)) → `run_trade_execution`
(portfolio + RL log + Telegram dispatch).

- `predict_v3_horizon(latest_df, h)` loads the per-horizon `V3BotInference`, builds
  features via the SHARED `build_features` (recipe parity), projects to the artifact's
  `tabular_features`, returns `{ticker:[p_down,p_flat,p_up]}`. Secondary horizon
  failures are non-fatal (`except FileNotFoundError/RuntimeError → {}`).
- `_compute_v3_features` also stashes per-ticker `market_regime` into
  `_LATEST_REGIME_BY_TICKER` (panel always carries it) for sizing + the card.
- **Weak-market fallback**: if no candidate passes the gates, returns a monitoring-only
  observability report (NO trades) — now includes per-ticker live price.

`/verify`, `/suggest_sell`, `/rebalance` reuse `predict_v3_horizon` (single/holdings).

---

## 8. Sizing (`src/bot/sizing.py`)

Half-Kelly, R=2.0 (locked), **20% NAV cap**, top-5 names ⇒ 100% gross, long-only,
unlevered. `w = min(max(0, 0.5·(p − (1−p)/R)), cap)`. Cap binds at p≥0.60; calibrated
band p∈[0.50,0.55] → 12.5–16.25% NAV.

**Regime overrides** (`suggested_weight(p_up, market_regime=...)`, backward-compatible
default None):
- 0 Freeze / 7 Liquidity Sweep → **0.0** (stand aside)
- 1 Squeeze / 6 Choppy → cap shrunk to **REGIME_PENALTY_CAP=0.10**
- 3 Strong Trend → full half-Kelly to the 20% cap
- 2/4/5 + None → unmodified

---

## 9. Telegram surfaces

- **Signal card** (`telegram_alerter._build_message`): clean HTML, no emoji/banners.
  Shows `T+{h} Model | Khuyến nghị đi vốn: X% NAV`, `Pha thị trường: <label> (Regime N)`,
  price, trend split (Tăng/Đi ngang/Giảm), single `Nhận định` paragraph, source links.
- Commands (`telegram_bot.py`): `/suggest_buy{5,20}`, `/suggest_sell`, `/verify`,
  `/add` `/remove`, `/news`, `/rebalance`, `/audit_weekly` `/audit_monthly`, `/msg_id2`,
  start/help. Group oversight gate mirrors responses to Admin.
- **Gemini news-analyst** (`quant_agent_arbitrator.py`): scrapes article bodies → JSON
  sentiment. Default model `gemini-2.5-flash` (GA pin; the floating `flash-latest` 503s),
  `max_retries=5` + exponential backoff. Missing key → "Không có API Key" (logged WARN,
  distinct from the polite "hệ thống nguồn đang bận" busy-fallback after exhausted retries).

---

## 10. EOD pipeline & CLI

`python main.py --task {daily_inference | crawl_hose | full_pipeline}`.
`full_pipeline` (15:30 ICT cron) = `crawl_hose` (15:00 ICT guard; `--days-back 1` =
previous-day incremental, `--force-crawl` bypass) → LLM sentiment → `daily_inference`.
There is **no `build_alpha360` step** (retired — V4 recomputes features from raw OHLCV).

---

## 11. Key invariants & gotchas

- **Train/serve parity is structural** — both sides call `pipeline.build_features`.
  Never duplicate feature math. Change it → bump `FEATURE_RECIPE_VERSION` + retrain.
- **Recipe tripwire** blocks serving a model whose recipe drifted from the live code.
- **HTML safety**: truncate RAW text THEN `html.escape` (never `escape(...)[:N]` — it
  severs `&amp;`/`&#x27;` → Telegram parse error). `_smart_truncate` is word-aware.
- **Price units**: parquet `close` is in thousand-VND (e.g. 13.7); `_get_live_exec_prices`
  normalises to VND for display (13,700 VND).
- **Boundary purge** in `chronological_split` (AFML §7): `train_mask = (dates<cutoff) &
  (t1<cutoff)` — drops train labels whose barrier spills into OOS.
- **PowerShell stdin encoding** mangles non-ASCII piped to `python -`; use `chr(0x...)`
  in throwaway test scripts. (`reconfigure(encoding="utf-8")` fixes OUTPUT only.)
- backtest portfolio uses an MV/target-vol optimiser; the LIVE bot uses pure per-name
  half-Kelly — a known, documented difference.

---

## 12. Tests / CI

`pytest` → **130 passing**. conftest stubs heavy deps FALLBACK-ONLY (real libs when
installed). Coverage: sizing (incl regime rules), Telegram cards, serve resilience
(secondary-horizon non-fatal), feature serve parity, market_regime (range/leak-free),
main logic, arbitrator, telegram split. GitHub Actions CI runs compile + pytest.

---

## 13. This-cycle changelog (V3 → V4.0 hardening)

1. Split monolith → `train_models.py` + `run_backtest.py` + shared `pipeline.py`.
2. Train/serve feature parity routed through `build_features`; recipe-version tripwire.
3. Dual-horizon T+5/T+20; calibrated probabilities; 20% cap + top-5.
4. **Parquet-first migration**: dropped 5 dead DuckDB tables; `price_lookup.py` for all
   point price lookups; deleted dead alpha360 parquet (2.25 GB).
5. **Dead-code purge**: gutted the Alpha360 feature factory (kept only
   `load_live_ohlcv_window`); removed `MacroCrawler`/`MacroProvider`/`build_retry_session`
   + `build_alpha360` task.
6. **8-market-regime** feature: rule-based Polars, GBM-native categorical (recipe→v1.1),
   regime-aware sizing, shown on the card.
7. `run_backtest.py --export-only` fast artifact repackage.
8. Telegram HTML fixes: `_smart_truncate` (word-aware, escape-after) across fallback
   report + `/verify` + `/suggest_sell` + msg_id2; fallback report gained price line.
9. Gemini robustness: default → `gemini-2.5-flash`, retries 3→5, clearer exhausted-log.
