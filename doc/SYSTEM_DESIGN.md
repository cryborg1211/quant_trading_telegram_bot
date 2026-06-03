# Quant V6 — System Design & Continuation Brief

> **For the next agent / chat:** this is the canonical context-transfer doc.
> Read §1 (TL;DR) and §11 (Honest caveats) first, then §3 (Repo map) before
> touching code. Numbers are real-OOS, sample sizes included. Nothing here
> is aspirational — only what is actually built and verified.

**Generated:** end of the dual-model sprint · `main` HEAD `c906618` ·
24 commits this sprint pushed to `origin/main`.

---

## 1. TL;DR — what this system IS today

A production Telegram trading assistant for HOSE/VN equities with a
**dual-model architecture**:

- **Alpha360 5d Stacking** — trend / momentum continuation (XGB + LGBM +
  CatBoost → LGBM meta + meta-labeler), de Prado volatility-scaled
  triple-barrier targets, cost-aware Net Sharpe selection. Deployed at
  operator-chosen `τ* = 0.48` (the Sharpe-optimal was 0.58; lowered for
  ~15 trades/mo livability). OOS hit ~73%, +3.25 Net Sharpe @ 0.8% RT
  cost. **n ≈ 54 trades / 2 years — small sample.**
- **MR Knife-Catcher** — single LightGBM on dedicated oversold features
  (RSI, BB %B, DMA, Williams %R, ATR, gap). Strict `τ* = 0.96`. OOS
  precision 64% on **n = 14 fires** (~1/month) — a rare alert, not a
  strategy.
- **Gemini sentiment** (model: `gemini-2.5-flash`, GA) provides
  catalyst/risk/reasoning per ticker. JSON schema auto-normalized
  (handles dict-keyed/list/wrapped shapes). Transient-only exponential
  backoff. Polite VN fallback when unavailable.
- **Telegram bot** — multi-tenant (Admin ID1 / User ID2), plain-VN
  jargon-free UI, invisible oversight gate (every ID2 command request +
  response mirrored to Admin), dual-audit routing (User: post-mortem;
  Admin: post-mortem + bounded rolling re-fit).

---

## 2. Quick-start for the next agent

```bash
# inspect state
git log --oneline -25                # this sprint's 24 commits
cat doc/SYSTEM_DESIGN.md             # this file
cat README.md                        # user-facing summary

# restart the bot cleanly (psutil ghost sweep included in the snippet
# in any of the recent "deploy" Bash blocks — see commit history)
python run_bot.py                    # foreground OR via nohup

# retrain (lightweight)
python -m src.models.train_mr_lgbm                    # MR, ~35s on 850k
python src/scripts/audit_and_retrain.py               # same, with summary

# retrain (heavy — full stack, ~10+ min). Features are recomputed in-pipeline
# from raw OHLCV — there is no build_alpha360 step (that path was retired).
python train_models.py && python run_backtest.py

# refresh context
python backfill_context_data.py      # macro (yfinance) + Gemini sentiment

# strict 2yr OOS backtest of the 5d model
python backtest_recent_2yrs.py
```

**Editing rules of thumb (learned the hard way this sprint):**
- Vietnamese text in `print()` will crash a Windows cp1252 console.
  Write to a UTF-8 file and read it back. **Never** let a console
  encoding bug silently corrupt logic via an `except` fallback.
- The Telegram message is **HTML parse-mode** (`<b>`, `<a href>`, `&amp;`).
  Use `html.escape(..., quote=True)` for href attributes.
- `git add .` is unsafe here — `.env` contains live keys. Stage by name.
- Model artifacts (`models/stacking/`, `models/mr/`) are gitignored —
  regenerable, not version-controlled. Same for `logs/`, `scratch/`,
  parquet, duckdb.

---

## 3. Repository map (critical files only)

```
main.py                                      CLI orchestrator + LIVE inference + Telegram report builders
                                             (_build_verify_report, _build_sell_hold_report,
                                              _build_fallback_observability_report_vi,
                                              mr_score_tickers, predict_stacking_horizon,
                                              load_stacking_artifacts, daily_inference, run_trade_execution)
run_bot.py                                   Bot entrypoint (long-polling); python-telegram-bot
config/settings.{py,json}                    Typed Config dataclass + JSON overrides
                                             (PathConfig, ModelConfig, TrainingConfig, TradingConfig,
                                              CrawlerConfig, SentimentConfig, UniverseFilterConfig)

src/features/alpha360_generator.py           Alpha360 features (Polars): 60-bar rolling-Z OHLCV lags
                                             + macro (T-1 shifted) + per-ticker sentiment join
                                             + triple-barrier labels per horizon (5d, 20d)
src/features/triple_barrier.py               Vol-scaled triple-barrier (intrabar; same-bar PT+SL → DOWN)
src/features/mr_features.py                  Mean-reversion oversold features (pandas/numpy, no ta-lib)
                                             MR_FEATURE_COLUMNS = [dma_sma10/20/50, bb_pctb,
                                             bb_below_lower, rsi_9, rsi_14, williams_r_14,
                                             atr_norm_14, gap_pct, gap_down]

src/models/stacking_model/train_stacking.py  Dual-horizon stacking trainer: PurgedKFold OOF,
                                             cost-aware τ* selection, meta-labeler training
src/models/stacking_model/purged_kfold.py    de Prado PurgedKFold (purge [start,t1] overlap + embargo)
src/models/stacking_model/economic_metrics.py Net Sharpe / select_pnl_threshold / meta_label_feature_matrix
src/models/train_mr_lgbm.py                  MR sub-model trainer (single LGBM, scale_pos_weight)
src/models/quant_agent_arbitrator.py         News scrape (GNews) + Gemini sentiment
                                             (schema-normalized, transient backoff, polite fallback)

src/utils/telegram_bot.py                    Handlers (suggest_buy/sell, verify, audit_*, feedback,
                                             msg_id2, add/remove, news, rebalance, help, start)
                                             + group=-1 oversight gate (TypeHandler)
                                             + role helpers (_role_for, _extract_user_id)
                                             + _send_or_reply_chunks (response mirror to Admin)
src/utils/telegram_alerter.py                Plain-VN signal card formatter (TelegramBot._build_message)
                                             + format_source_links() helper
src/utils/audit_evaluator.py                 /audit_* engine: run_post_mortem(user_id, days)
src/scripts/audit_and_retrain.py             Bounded admin rolling re-fit (MR ~35s) + VN summary

src/data/db_engine.py                        DuckDB singleton + schema migrations (live tables only)
src/data/crawlers.py                         StockCrawler (vnstock) -> data/ohlcv_*.parquet shards
src/data/price_lookup.py                     Parquet-first point price/volume lookups (read_parquet)
src/crawlers/sentiment_crawler.py            Daily LLM-labeled sentiment crawler + RSS feeds

backfill_context_data.py                     Standalone Macro + LLM-sentiment backfill
backtest_recent_2yrs.py                      Strict 2yr OOS backtest harness for the 5d model
```

---

## 4. Data layer

**Single Source of Truth for OHLCV — Parquet-first.** Per-ticker shards
`data/ohlcv_<TICKER>.parquet` (~355 files, **gitignored**) are the authoritative
price store. `StockCrawler` writes them every EOD; *everything* reads them:

* **Training / backtest** — `pipeline.load_ohlcv` globs the shards via
  `pl.scan_parquet("data/ohlcv_*.parquet")`.
* **Live serve features** — `Alpha360Generator.load_live_ohlcv_window` reads the
  same shards, so train/serve parity is structural.
* **Point price / volume lookups** (RL outcome backfill, `/audit_*`, and
  sentiment-crawl liquidity ranking) — `src/data/price_lookup.py` via DuckDB
  `read_parquet('data/ohlcv_*.parquet')`.

The legacy DuckDB `stock_ohlcv` and `macro_daily` tables were **DROPPED** (the DB
audit found `stock_ohlcv` ~18 days stale — the crawler only ever wrote the
parquet; `macro_daily` was unused by V4). `data/alpha360_features.parquet` and
the entire Alpha360 feature factory were deleted too.

**DuckDB:** `data/quant_v6_core.duckdb` (gitignored) now holds only live bot
state — **no price data**:

| Table | Status | Notes |
|---|---|---|
| `hist_sentiment_llm_labeled` | ⚠️ ~180 rows | **Only ~1 week of history backfilled**; the rest of the training span had no sentiment → effectively zero-filled in training. Authoritative schema is created by `backfill_context_data.py` (includes the `ticker` column the legacy crawler DDL forgot). PK(ticker, date, title) for idempotent upsert. |
| `portfolio` | ✅ per-user live positions | Multi-user via Telegram `user_id` |
| `audit_log`, `trade_history`, `rl_mistake_logs` | ✅ | Append-only audit trails |

**Crawl pipeline:** `StockCrawler` (vnstock) → `data/ohlcv_*.parquet` is the EOD
OHLCV ingestion; `SentimentCrawler` → GNews per-ticker + Vietnamese RSS →
`_score_item` → `hist_sentiment_llm_labeled`. (The macro crawler was removed —
V4 uses cross-sectional price alphas + an HMM price proxy, not macro features.)

---

## 5. Feature pipelines

### 5.1 Alpha360 (trend / 5d model) — `src/features/alpha360_generator.py`

- Per-ticker 60-bar rolling-Z normalization of OHLCV + HLC3.
- **Lag flattening:** 60 lags × 6 features = 360 `close_i / open_i / ...`
  columns (these are the values trees see; `close_0..close_19` are also
  reused as inputs to the **meta-labeler features**).
- **Macro integration:** T-1 shift, two-pass forward-fill, S&P500/DXY/
  USD-VND converted to **log-returns** (stationarity); join on `date`.
- **Sentiment integration:** per-ticker (`GROUP BY ticker, date`),
  T-1 shifted via `shift(1).over("ticker")`, lagged aggregates
  (mean score, magnitude, NLP, impact_force, news_count,
  market_wide_count).
- **Targets:** `add_triple_barrier_labels(horizon, pt_mult=2.0,
  sl_mult=2.0, vol_span=20, use_intrabar_extremes=True)` → emits
  `target_class_{5d,20d}` (0/1/2), `target_return_{}`, `t1_{}`
  (event-end date — required by PurgedKFold).
- **Leak safety:** intrabar high/low threaded through `_generate_lags`
  for triple-barrier touch, then **dropped** before final feature
  matrix; raw `close` saved as `raw_close` (preserved for liquidity
  filter / portfolio) and removed from model features.

### 5.2 MR (capitulation) — `src/features/mr_features.py`

- Pure pandas/numpy, vectorized, **ta-lib is intentionally not a
  dependency** (RSI/ATR use Wilder `ewm(alpha=1/n, adjust=False)` —
  numerically identical).
- `build_mr_features(df)` is leak-audited (built-in smoke test
  mutates future bars and asserts past features are byte-identical).
- Used at training time by `train_mr_lgbm.py` and at serve time by
  `main.mr_score_tickers(tickers)` (loads 80-bar OHLCV tails,
  computes features, scores latest row).

---

## 6. Models

### 6.1 5d Stacking — `src/models/stacking_model/`

- **Base:** XGBoost + LightGBM + CatBoost (GPU-trained — TD-10).
- **OOF meta features:** PurgedKFold (5 folds, embargo=horizon).
- **Meta-learner:** LightGBM on the 9 OOF probability columns.
- **Cost-aware τ\* selection:** `select_pnl_threshold` sweeps τ on
  leak-free meta-OOF probabilities, picks the **Net Sharpe-maximizing**
  τ (subject to a min-trades floor). Cost model: `DEFAULT_FEE_RATE =
  0.002` + `DEFAULT_SLIPPAGE_PER_SIDE = 0.002` → **0.8% round-trip**.
- **Meta-labeler:** secondary binary LGBM trained on the *primary's
  bullish bets only* (de Prado AFML 3.6). Features built by
  `meta_label_feature_matrix()`: `[p_down, p_side, p_up, conviction,
  spread, trend, vol]` (last two from `close_0..close_19` so train/serve
  is row-local).
- **Selection metric:** `net_sharpe` (macro-F1 is a secondary diagnostic
  only); deployment gate is `beats_baseline = net_sharpe > 0`.
- **Artifacts** (`models/stacking/{5d,20d}/`, gitignored):
  `xgboost_model.joblib`, `lightgbm_model.joblib`, `catboost_model.cbm`,
  `meta_model.joblib`, `meta_labeler.joblib`,
  `selected_features.json` (70 features per horizon),
  `quantile_thresholds.json` (loaded at inference — contains
  `pnl_threshold_tau`, `round_trip_cost`, audit keys).

**5d τ\* operator override:** `pnl_threshold_tau` was patched from the
Sharpe-optimal **0.58** to **0.48** (`models/stacking/5d/quantile_thresholds.json`)
to lift live frequency from ~2/mo → ~15/mo. The autoselected value
(`pnl_threshold_tau_autoselected: 0.58`) and override reason are stored
in the same JSON for audit.

### 6.2 MR LightGBM — `src/models/train_mr_lgbm.py`

- **Single LGBM**, no stacking — light inference, less overfit surface.
- **Label:** `y = 1` iff `(3-bar fwd return > +3%)` **AND** at least one
  panic condition at *t* (`mr_rsi_9 < 20` OR `mr_bb_below_lower == 1` OR
  `mr_dma_sma20 < -0.05`). The setup-gate is what makes it a
  knife-catcher, not a momentum-chaser. **Positive rate: 4.0% train /
  2.5% holdout.**
- **Imbalance:** `scale_pos_weight = n_neg/n_pos` (per fold).
- **Validation:** PurgedKFold (5 folds, embargo=3) + strict
  chronological 1-year hold-out.
- **τ\* selection:** lowest τ with OOF precision ≥ 0.60 and ≥ 40 fires;
  fallback to max-precision (this sprint hit fallback at **0.96**).
- **Artifacts** (`models/mr/`, gitignored): `mr_lgbm.joblib`,
  `mr_threshold.json`, `mr_report.json`, `last_retrain_summary.txt`.

---

## 7. Telegram bot architecture

### 7.1 Roles (`.env`)

`TELEGRAM_CHAT_ID_1` = Admin (full access).
`TELEGRAM_CHAT_ID_2` = User (analytical commands only; `/add` and
`/remove` are blocked with a polite-VN denial).
Unknown ids: denied by the oversight gate.

### 7.2 Oversight gate (`group=-1` `TypeHandler`)

`_oversight_gate` in `telegram_bot.py` runs before *every* command:
- unknown → polite VN denial + `ApplicationHandlerStop`.
- ID2 → shadow REQUEST to ID1 (`👁️ [GIÁM SÁT] ...`); block `/add`,
  `/remove`; otherwise allow.
- ID1 → no shadowing.

`_send_or_reply_chunks` additionally **mirrors the full RESPONSE** to
ID1 when the requester is ID2 (covers the analytical command outputs
funneled through that one helper).

### 7.3 Commands

| Command | Path | Notes |
|---|---|---|
| `/suggest_buy` | `main.daily_inference` → `predict_stacking_horizon(5)` + `mr_score_tickers` → arbitrator (Gemini) | Universe filter (`CONFIG.universe_filter`): `enabled`, `exclude_vn30`, `min_price_vnd`, `exclude_tickers`. **Fallback observability mode** if no candidates clear the gate: top-3 by P(UP) with `[⚠️ CẢNH BÁO: THỊ TRƯỜNG XẤU]` header and `HỦY BỎ TÍN HIỆU` per ticker, MR-fired tagged `[🔪 BẮT ĐÁY]`. `run_trade_execution` BYPASSED in fallback (no portfolio/RL side effects). |
| `/suggest_sell` | `main.inference_for_holdings` → `_build_sell_hold_report` | **MR Veto**: if 5d says SELL but MR fired → banner `⚠️ [CẢNH BÁO BÁN ĐÚNG ĐÁY ...]` above the verdict. Includes 🎯 take-profit / 🛡️ stop-loss from `CONFIG.trading`. |
| `/verify <T>` | `main.verify_single_ticker` → `_build_verify_report` (with `mr_state`) | Dual output: trend (Cửa Tăng/Đi Ngang/Giảm) + `🔪 Trạng thái Bắt đáy` ("Chưa đạt" / "🚨 CẢNH BÁO HOẢNG LOẠN"). |
| `/audit_weekly`, `/audit_monthly` | `_run_audit_command` → `audit_evaluator.run_post_mortem` | Dual-routing: both roles get the same report. Admin **additionally** triggers `src/scripts/audit_and_retrain.py` non-blocking via `asyncio.to_thread(subprocess.run, …)`; completion summary pushed to Admin. |
| `/feedback <msg>` | `feedback_command` | User → Admin direct channel; admin sees `📩 [FEEDBACK TỪ USER]`. |
| `/msg_id2 <msg>` | `msg_id2_command` | **Admin-only; hidden from menu/help.** Sends `📢 [THÔNG BÁO TỪ ADMIN]` to ID2. Silent ignore for everyone else (including ID2). |
| `/add`, `/remove` | `add_portfolio_command`, `remove_portfolio_command` | Admin-only via the gate; blocked for ID2 with polite VN message. |
| `/news`, `/rebalance`, `/help` | RSS digest / AI rebalance / menu | `/help` includes the privacy disclosure: `💡 Lưu ý: ... có thể được Admin ghi nhận.` |

### 7.4 Message formatting

**One shared signal card** (`TelegramBot._build_message` in
`telegram_alerter.py`), powered by `signal_data` keys: `ticker`, `price`,
`prob_up/side/down`, `status_label`, `plus_points`, `minus_points`,
`conclusion`, `article_urls`. Output is plain-VN, jargon-free
("Cửa Tăng / Đi Ngang / Cửa Giảm", "Điểm cộng / Điểm trừ / Kết luận").

**Source attribution:** `format_source_links(urls, limit=2)` →
`🔗 <b>Nguồn báo:</b> <a href="U1">Bài viết 1</a> | <a href="U2">Bài viết 2</a>`.
Reused by the fallback report. **Always ends `\n\n`** so it cannot
collide with appended footers (a real bug we fixed).

---

## 8. Sentiment engine

`src/models/quant_agent_arbitrator.get_batch_sentiment_scores`:

1. **API call:** `google-genai` `client.models.generate_content`
   (model from `GEMINI_MODEL` env, currently GA `gemini-2.5-flash`).
2. **Response handling — the real bug-fix:** `_normalize_news_json`
   coerces dict-keyed / list-of-objects / wrapped / single-object
   shapes to canonical `{TICKER: {...}}`. The historical "Lỗi gọi API"
   was a deterministic `list indices must be integers or slices, not
   dict` from assuming a dict shape — *not* an API failure.
3. **Retry policy:** transient-only (429/5xx/timeout/connection),
   exponential backoff (2 → 3 → 5s); permanent (4xx, bad key) **fail
   fast**.
4. **Logging:** every attempt logs type, HTTP status, transient flag
   (`LOGGER.error`). Exhausted retries return the polite-VN fallback:
   `⚠️ Không thể tải tin tức lúc này (hệ thống nguồn đang bận). Vui
   lòng thử lại sau.` (never the raw exception text to the user).

Per-ticker enrichment writes `catalyst`, `risk`, `reasoning_vi`,
`source_urls` into the result. Downstream formatters render those.

---

## 9. Config & environment

### 9.1 `.env` (gitignored — never commit)

```
GEMINI_API_KEY=<key>
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID_1=<admin id>
TELEGRAM_CHAT_ID_2=<user id>
GEMINI_MODEL=gemini-2.5-flash         # GA — preferred over gemini-flash-latest (intermittent 503)
```

### 9.2 `config/settings.json` (tracked)

Notable knobs:
- `training.split_date: "2024-01-01"` — chronological train/OOS boundary.
- `trading.fee_rate: 0.002` · `stop_loss_pct: -0.07` · `take_profit_pct: 0.15`.
- `sentiment.gemini_model: "models/gemini-flash-latest"` — overridden by
  `.env` `GEMINI_MODEL` for the arbitrator/audit paths.
- `universe_filter: { enabled: true, exclude_vn30: false,
  min_price_vnd: 20000.0, exclude_tickers: [] }` — current live config:
  drops sub-20k-VND penny/smallcaps. `vn30_tickers` default list is in
  `UniverseFilterConfig`.

### 9.3 `CONFIG` dataclasses (`config/settings.py`)

`PathConfig`, `ModelConfig`, `TrainingConfig`, `TradingConfig`,
`CrawlerConfig`, `SentimentConfig`, `UniverseFilterConfig`,
+ `Config.from_json("config/settings.json")` overlay.

---

## 10. Operational runbook

**Restart the bot cleanly** (ghost-poller-safe — there are **always**
zombies on Windows):

```python
# psutil sweep (avoids the `pkill -f` self-kill footgun)
import os, psutil, time
me=os.getpid(); parent=psutil.Process(me).ppid()
def is_bot(cl): return ("run_bot.py" in cl) or cl.rstrip().endswith("-m src.utils.telegram_bot")
def is_tool(cl): return any(b in cl for b in ("pkill","psutil","shell-snapshot","claude","process_iter"))
for p in psutil.process_iter(["pid","cmdline"]):
    cl=" ".join(p.info.get("cmdline") or [])
    if p.pid in (me,parent) or not cl or is_tool(cl) or not is_bot(cl): continue
    try: p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
time.sleep(1)
```

then `nohup python run_bot.py > logs/bot_run_<ts>.log 2>&1 &` and
verify clean start (search log for `Application started` and the
absence of `telegram.error.Conflict`).

**Verify deploy:** the bot logs its commit SHA on boot
(`version=<sha>`). Note: that SHA is the **last commit at process
start** — uncommitted working-tree edits *are* loaded by Python
(modules read from disk) but the SHA label won't reflect them. If
exact-sha matters, commit before restart.

**Critical safety patterns this sprint:**
- Always exclude the executor and parent shell from psutil kills.
- `git add .` is banned — `.env` would be staged. Use named files.
- All Vietnamese / emoji **must** be written via UTF-8 file or
  HTML-escaped — never `print()`ed to a Windows console without a
  best-effort try/except wrap (we fixed exactly this bug in
  `src/scripts/audit_and_retrain.py`).

---

## 11. Validated numbers & honest caveats

**These are the only numbers that have been verified end-to-end.
Everything else is engineering, not edge.**

| Model | Validation | Result | Sample |
|---|---|---|---|
| 5d Stacking (τ\*=0.58, Sharpe-optimal) | 2024-01→2026-05 OOS backtest (`backtest_recent_2yrs.py`) | **+4.57 Net Sharpe · 85% hit · 54 trades / 2 yrs** | n=54 |
| 5d Stacking (τ\*=0.48, **deployed**) | OOF τ sweep at conservative 0.8% cost | **+3.25 Net Sharpe · ~73% hit · ~15 trades/mo** | OOF-projected |
| MR LightGBM (τ\*=0.96) | strict 1-year chronological hold-out | **64% precision · 14 fires** · avg_precision 0.33 | **n=14** |
| Reversal audit (5d at crash bottoms) | event study on 2 OOS troughs (Apr-25 −18%, Mar-26 −10%) | mean P(UP) 25–38% across ±5 days — model is **blind to V-bottoms**; gate barely opens (peak 3.4% at +3) | n=2 troughs |

**Caveats that the next agent MUST internalize:**
1. **Both models have institutionally small samples.** 54 trades and
   14 fires would fail a Deflated Sharpe / PBO test. Treat the
   numbers above as *suggestive*, not proven.
2. **Sentiment is shallow.** Only ~1 week of historical sentiment was
   backfilled (GNews isn't retroactive). So in the 5d model's *trained
   weights* sentiment is near-zero — the edge is **macro-driven**.
   Sentiment mainly helps *live* inference.
3. **Macro is partial.** `interbank_on_rate` / `vnibor` are NULL
   (SBV+TradingEconomics DNS-blocked from VN ISPs).
4. **No cross-sectional features.** Everything is per-name time-series
   rolling-Z. No ranking, no factor neutralization. This is the single
   biggest feature gap (see §13).
5. **No portfolio construction.** The bot emits Top-3 signals; there is
   no covariance, vol-targeting, sizing logic, drawdown overlay, or
   capacity model.
6. **No survivorship-bias control.** Backtests use the *current*
   ticker universe — delisted names absent → upward-biased OOS.

---

## 12. Known bugs / tech debt

| ID | Issue | Status |
|---|---|---|
| TD-10 | XGBoost/LightGBM/CatBoost hardcoded GPU device — `device="cuda"` (XGB), `device_type="gpu"` (LGBM), `task_type="GPU"` (CatBoost). Inference falls back to CPU automatically with a warning. | Open |
| TD-25 | SBV + TradingEconomics interbank-rate scrapers DNS-blocked from VN ISPs. **Resolved by removal** — `macro_daily` + `MacroCrawler` were dropped (V4 uses no macro features). | Closed |
| — | Gemini news-analyst **on `gemini-flash-latest`** intermittently 503s. Pinned to GA `gemini-2.5-flash`; backoff handles residual. | Mitigated |
| — | `_domain_label` in `telegram_alerter.py` is now unused (kept to avoid a risky remove). | Cosmetic |
| — | `quant_agent_arbitrator.NEWS_ANALYST_SYSTEM_PROMPT` is referenced inside the function but does not declare ARRAY-vs-OBJECT shape policy — normalization in code handles either, but the prompt could be tightened to force OBJECT-keyed. | Cleanup |
| — | The cron `daily_inference` path's return value for the fallback observability mode is currently consumed only by the interactive `/suggest_buy`; the cron broadcast path is not exercised for fallback. | Open scoping |

---

## 13. V2.0 Roadmap (from the Principal-Quant audit)

**Ordered by PnL impact, not by glamour. The model is #4.**

### 13.1 Statistical-rigor engine (highest leverage)
- **Combinatorial Purged Cross-Validation (CPCV)** → distribution of
  Sharpes instead of a point estimate.
- **Deflated Sharpe Ratio** + **Probability of Backtest Overfitting
  (PBO)** as mandatory deployment gates. Will likely classify current
  edges as not significant; that is the point.
- Minimum effective-trade-count thresholds; sample-weight labels by
  uniqueness (de Prado AFML Ch. 4); **sequential bootstrap** for
  training sets to remove overlap bias.

### 13.2 Data integrity & point-in-time discipline
- **Survivorship-bias-free** universe (include delisted/halted names
  as-of).
- Bitemporal store (knowledge date vs. event date) for fundamentals
  / index membership / corporate actions.
- Realistic **cost model**: spread + market-impact (participation-rate
  aware) + VN microstructure (ATC/price band) + liquidity-conditioned
  tradeable filter — replace the flat 0.8%.

### 13.3 Risk & portfolio-construction engine (institutional dividing line)
- Covariance estimation (Ledoit-Wolf shrinkage); **volatility
  targeting**; risk-parity / MV optimizer with turnover + sector caps;
  fractional-Kelly sizing; drawdown-control overlay; regime kill-switch.
- Capacity & turnover analysis at deployment size.

### 13.4 Model & feature architecture
- **Cross-sectional layer:** daily rank-transform, sector/size
  factor-neutralization (model the *residual*), cross-sectional
  z-scores. **Currently entirely absent.**
- **Regime features:** market breadth, dispersion, realized-vol
  regime, term-structure of vol, drawdown state.
- **Fractional differentiation** (AFML Ch. 5) to replace rolling-Z —
  preserve memory while achieving stationarity.
- **MR uniform with 5d:** port volatility-scaled Triple-Barrier to MR;
  meta-label it; sample-uniqueness weighting on both.
- **Regime-gated mixture-of-experts** (trend / MR / flat) before any
  sequence-model migration. Sequence models (LSTM/Transformer) are
  premature until §13.1–§13.3 are done and feature engineering plateaus.

### 13.5 Research process / MLOps
- Feature store + experiment tracking.
- **Research-vs-production firewall:** the backtest is a *falsification*
  tool, never a search tool. Automate PBO on every research iteration.
- Walk-forward paper trading before capital.

---

## 14. Sprint commit history (this session)

24 commits on `main`, pushed to `origin/main`. Newest first:

```
c906618  docs: professional README overhaul + sprint wrap-up
f5c2ef6  chore(bot): align /msg_id2 empty-usage example to latest spec
251a82a  feat(ui): clickable source attribution + fix sentiment/footer collision
e3f8495  fix(arbitrator): real cause of 'Lỗi gọi API' was a JSON-shape bug
5fc095e  feat(bot): dual-audit routing — user report-only vs admin report+retrain
ed0ca1e  feat(bot): dual-model integration — MR knife-catch into verify/sell/buy
8e81ee1  fix(models): MR label = bounce AND panic-setup (knife-catch, not chaser)
61ebc85  feat(models): MR sub-model trainer — single LightGBM (train_mr_lgbm.py)
a7ba5ec  feat(features): mean-reversion / capitulation feature engineering
7bb6fe0  feat(bot): /help privacy disclosure + /feedback user->admin channel
2bc2af5  feat(bot): /msg_id2 admin->user one-way announcement channel
c507508  feat(bot): split-ID access control + 1-way admin oversight
3335187  feat(bot): Vietnamese Fallback Observability Mode for /suggest_buy
6ca481b  feat(ui): plain-Vietnamese Telegram formatter — strip all model jargon
1b3f7d7  feat(inference): config-driven Universe Filter
b5503a6  feat(bot): 5d tau* 0.58->0.48 (operator-approved) + top-5 prob logging
44041a4  fix(llm): gemini-3.5-flash does not exist -> standardize on gemini-flash-latest
777bacc  fix(data): load .env in backfill so GEMINI_API_KEY reaches SentimentCrawler
613a3bf  feat(data): standalone Macro + LLM-sentiment backfill script
f023691  fix(features): Alpha360 graceful degradation on missing macro/sentiment
51e3d81  feat(ml): Task 3 — de Prado meta-labeling secondary classifier
0c6bed3  feat(ml): de Prado triple-barrier intrabar + cost-aware Net Sharpe gate
```

Run `git log -p <sha>` on any of these for the exact diff and rationale
(commit messages are dense).

---

## 15. Common edit patterns (for the next agent)

| You want to … | Touch this |
|---|---|
| Change a Telegram message wording | `src/utils/telegram_alerter._build_message` (normal card) or the per-builder in `main.py` (`_build_verify_report`, `_build_sell_hold_report`, `_build_fallback_observability_report_vi`) |
| Add/remove a command | Handler in `src/utils/telegram_bot.py` + `app.add_handler(CommandHandler(...))` in `build_application` + optionally `_BOT_COMMANDS` for the `/` menu |
| Change role permissions | `_role_for`, `_EDIT_ONLY_COMMANDS`, `_oversight_gate` |
| Change cost / threshold for 5d | `models/stacking/5d/quantile_thresholds.json` (live patch; record audit keys) OR `economic_metrics.DEFAULT_FEE_RATE/SLIPPAGE_PER_SIDE` (full re-fit needed) |
| Change MR strictness | `models/mr/mr_threshold.json` `tau` (live patch) OR `TARGET_PRECISION`/`MIN_FIRES` in `train_mr_lgbm.py` (re-fit) |
| Add an MR feature | `MR_FEATURE_COLUMNS` + the indicator block in `build_mr_features`; re-train via `python -m src.models.train_mr_lgbm` (~35s) |
| Add a 5d feature | Add it to `pipeline.build_features` (bump `FEATURE_RECIPE_VERSION`) → re-train via `python train_models.py && python run_backtest.py`; heavy path (~10+ min) |
| Change Gemini model | `.env` `GEMINI_MODEL=...` (arbitrator/audit) and/or `CONFIG.sentiment.gemini_model` (sentiment crawler) |
| Backfill data | `python backfill_context_data.py [--macro-csv ...]` |
| Restrict the universe | `config/settings.json` → `universe_filter` (no code change) |

---

**End of design doc.** If you're a new agent reading this: re-read
§1, §11, §13 before proposing changes. The previous sprint's biggest
mistakes were always made when an honest caveat in §11 was forgotten.
