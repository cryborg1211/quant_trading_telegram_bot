# Quant V6 — Dual-Model Vietnamese Equity Trading Assistant

A production Telegram trading assistant for the HOSE/VN universe, built on a
**dual-model architecture**: a trend-following stacking ensemble and a
specialized mean-reversion "knife-catcher", arbitrated by an LLM sentiment
engine and delivered through a multi-tenant Telegram bot with role-based
access control and admin oversight.

> **Status:** research/production hybrid. Read **§7 Known Limitations &
> Honest Caveats** before trusting any number in this document — every
> headline metric there is annotated with its sample size and regime.

---

## 1. System Overview

```
 OHLCV (vnstock) ┐
 Macro (yfinance) ├─► DuckDB ─► Feature Pipelines ─┬─► 5d Stacking (trend)
 News (GNews/RSS) ┘                                 └─► MR LightGBM (capitulation)
                                                            │
                                          Gemini sentiment (catalyst/risk)
                                                            │
                                   Telegram bot · role routing · oversight
```

Two models run **in parallel** on every relevant command. They are
deliberately **not blended** — they fire in opposite regimes:

| | When it acts | Role |
|---|---|---|
| **Alpha360 5d Stacking** | Established up-trends | Trend / momentum continuation |
| **MR Knife-Catcher** | Extreme panic / capitulation | Mean-reversion V-bottom alert |

---

## 2. Core Models

### 2.1 Alpha360 — 5d Stacking Ensemble (Trend-Following)

- **Architecture:** XGBoost + LightGBM + CatBoost → LightGBM meta-learner.
- **Features:** Microsoft-Qlib-style Alpha360 (60-bar rolling-Z OHLCV lags)
  + real macro (S&P500, DXY, USD/VND log-returns, 2014→2026) + per-ticker
  LLM sentiment.
- **Labels:** de Prado **volatility-scaled triple-barrier** with intrabar
  touch detection and a conservative same-bar tie-break (ambiguous → DOWN).
- **Validation:** **PurgedKFold** (purge + embargo ≥ horizon) — zero
  label-window leakage. Selection metric is **cost-aware Net Sharpe**, not
  macro-F1 (0.8% round-trip VN friction baked in).
- **Decision:** primary side + a **meta-labeler** that must independently
  predict the trade is profitable net of cost.
- **OOS backtest (2024-01-01 → 2026-05-15, leak-free):** at the
  Sharpe-optimal threshold ≈ **+4.6 Net Sharpe, ~85% hit, ~54 trades / 2y**.
  **Deployed** at the operator-chosen `τ* = 0.48` (≈15 trades/mo) →
  **+3.25 Net Sharpe @ the conservative 0.8% cost, ~73% hit**. *(Sample is
  small — see §7.)*

### 2.2 MR Knife-Catcher — Single LightGBM (Mean-Reversion)

- **Why separate:** a reversal audit proved the 5d stack is ~81%
  price-momentum and structurally blind to V-bottoms. The MR model uses a
  **dedicated oversold feature set** (`src/features/mr_features.py`):
  distance-to-MA, Bollinger %B / lower-band pierce, RSI(9/14), Williams %R,
  normalized ATR, overnight gap — pure-vectorized, leak-audited.
- **Label:** capitulation = `+3% / 3-bar bounce` **AND** a panic setup at
  *t* (RSI9<20 ∨ below-lower-BB ∨ DMA20<−5%). Genuinely rare (~4% train).
- **Imbalance:** `scale_pos_weight`, shallow regularized trees.
- **Threshold:** intentionally **strict `τ* = 0.96`** (high precision, low
  recall — fire only on absolute panic).
- **Hold-out (untouched last year):** **64% precision** on **14 fires**
  (~1 signal/month universe-wide), `avg_precision` ≈ 0.33 (~13× base-rate
  lift). A rare high-conviction safety-net, **not** a continuous strategy
  (see §7).

---

## 3. Telegram Features & Roles

Two tenants, configured in `.env` (`TELEGRAM_CHAT_ID_1` = Admin,
`TELEGRAM_CHAT_ID_2` = User):

| Capability | Admin (ID 1) | User (ID 2) |
|---|---|---|
| Analytical commands | ✅ | ✅ |
| Portfolio mutation (`/add`, `/remove`) | ✅ | ❌ blocked (polite VN) |
| Unknown sender | — | ❌ denied |

### Commands

| Command | Description |
|---|---|
| `/suggest_buy` | Dual-model: candidates tagged `[📈 ĐÁNH TREND]` (5d) / `[🔪 BẮT ĐÁY]` (MR). Weak-market fallback returns a flagged monitoring-only report. |
| `/suggest_sell` | Per-holding sell/hold + 🎯 target / 🛡️ stop. **MR Veto:** if trend says SELL but MR fires → prominent *"đừng bán đúng đáy"* warning. |
| `/verify <TICKER>` | Dual output: trend odds (Cửa Tăng/Đi Ngang/Giảm) **and** `🔪 Trạng thái Bắt đáy` (Chưa đạt / CẢNH BÁO HOẢNG LOẠN). |
| `/feedback <msg>` | User → Admin direct channel. |
| `/msg_id2 <msg>` | **Admin-only**, hidden from menu/help. Broadcast `📢 [THÔNG BÁO TỪ ADMIN]` to the User. Silent-ignore for anyone else. |
| `/news`, `/rebalance`, `/help` | Market news / AI rebalance / menu (with privacy disclosure). |

All user-facing text is **plain Vietnamese, jargon-free** (no `τ*`,
"meta-labeler", "Stacking", etc.).

---

## 4. Oversight & Retraining

- **Invisible oversight gate** (`group=-1` pre-dispatch handler): every
  ID2 command is shadow-copied to ID1 (request + the full response), with
  a disclosure line in `/help` (consented monitoring of a known user on an
  owner-operated financial bot).
- **`/audit_weekly` · `/audit_monthly` dual-routing:**
  - **User:** identical plain-VN performance post-mortem (wins/losses with
    LLM-written catalyst explanations). **No** retrain.
  - **Admin:** same report **+** instant *"đang retrain dưới nền…"* reply
    **+** non-blocking background **rolling re-fit** of the MR sub-model
    (`src/scripts/audit_and_retrain.py`, ~35s — bounded; the heavy 5d
    stacking retrain is intentionally *not* per-audit) **+** a completion
    summary pushed to Admin. Runs via `asyncio.to_thread` — the event loop
    never blocks.

---

## 5. Sentiment Engine

- **Gemini LLM** (`GEMINI_MODEL`, default GA `gemini-2.5-flash`) produces
  per-ticker **catalyst / risk / Vietnamese reasoning** + sentiment score.
- **Schema auto-normalization:** tolerates dict-keyed, list-of-objects,
  wrapped, and single-object JSON shapes (the original `"Lỗi gọi API"`
  outage was a deterministic list-vs-dict parse bug, now fixed).
- **Transient-only exponential backoff** (2→3→5s) on 429/5xx/timeout;
  permanent errors (bad key/4xx) fail fast. Exhausted retries → a polite
  VN fallback, never a raw stack trace.
- **Source attribution:** top-2 article links rendered inline
  (`🔗 Nguồn báo: <a>Bài viết 1</a> | <a>Bài viết 2</a>`).

---

## 6. Setup

```bash
pip install python-telegram-bot lightgbm xgboost catboost polars duckdb \
            pandas numpy scikit-learn google-genai gnews feedparser \
            vnstock yfinance joblib python-dotenv psutil
cp .env.example .env   # set GEMINI_API_KEY, TELEGRAM_BOT_TOKEN,
                       # TELEGRAM_CHAT_ID_1 (Admin), TELEGRAM_CHAT_ID_2 (User)
python run_bot.py      # starts the polling bot
```

Offline pipeline: `python train_models.py` → `python run_backtest.py`
(5d/20d stack; features recomputed in-pipeline from raw OHLCV) ·
`python -m src.models.train_mr_lgbm` (MR) ·
`python backfill_context_data.py` (sentiment backfill).

---

## 7. Known Limitations & Honest Caveats

*This section is deliberate. Numbers above are real but small-sample —
deploy with eyes open.*

1. **5d hit-rate is regime- and threshold-dependent.** "~85%" is the
   Sharpe-optimal OOS config over only **~54 trades / 2 years**. Deployed
   `τ*=0.48` trades ~15×/month at ~73% hit. Confidence intervals are wide;
   it is a **selective, low-frequency** strategy, not an always-on engine.
2. **MR fires ~1×/month universe-wide** (hold-out precision 64% on
   **n=14**). Statistically fragile; treat as a rare capitulation *alert*,
   not a return source. It will **not** ride most V-recoveries.
3. **Sentiment is shallow historically.** The historical sentiment
   backfill covers only ~1 week — the 5d model's edge is **macro-driven**;
   sentiment mainly improves *live* inference, not the trained weights.
4. **VN macro gaps.** `interbank_on_rate`/`vnibor` are NULL (SBV +
   TradingEconomics DNS-blocked from VN ISPs); the model runs on
   S&P500/DXY/USD-VND log-returns.
5. **Upstream API capacity.** `gemini-flash-latest` showed intermittent
   503 "high demand"; pinned to GA `gemini-2.5-flash`. Sustained Google
   spikes still surface the polite fallback (by design — sentiment is
   never faked).
6. **Model artifacts are git-ignored** (`models/stacking/`, `models/mr/`)
   — regenerable from the training scripts; not version-controlled.

---

## 8. Repository Layout

```
main.py                              CLI orchestrator + live inference + Telegram report builders
run_bot.py                           Bot entrypoint (polling)
config/settings.{py,json}            Typed config (split-date, costs, universe filter, Gemini model)
src/features/alpha360_generator.py   Alpha360 + macro/sentiment integration (leak-guarded)
src/features/triple_barrier.py       Vol-scaled triple-barrier (intrabar, conservative tie-break)
src/features/mr_features.py          Mean-reversion oversold feature set
src/models/stacking_model/           5d stack: train_stacking, purged_kfold, economic_metrics
src/models/train_mr_lgbm.py          MR knife-catcher trainer
src/models/quant_agent_arbitrator.py News scrape + Gemini sentiment (schema-normalized, backoff)
src/utils/telegram_bot.py            Handlers, role routing, oversight gate, dual-audit
src/utils/telegram_alerter.py        Plain-VN message formatter + source links
src/scripts/audit_and_retrain.py     Bounded background rolling re-fit (admin)
backfill_context_data.py             Macro + LLM-sentiment backfill
```
