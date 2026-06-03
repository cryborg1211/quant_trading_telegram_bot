# Quant Engine V4.0 — Production Runbook

Operational guide for the V4.0 system: a **pure-tabular stacking ensemble**
(LightGBM + XGBoost + CatBoost → calibrated LogisticRegression meta) on a
**VN50 dynamic universe**, T+5 and T+20 horizons, half-Kelly sizing, with an
HMM macro-regime overlay and a mean-reversion (knife-catch) sub-model.

> **Pipeline at a glance**
> ```
> OHLCV ingest (crawl_hose / external) ─▶ DuckDB + data/ohlcv_*.parquet
>        │
>        ├─▶ train_models.py  ──▶ models/saved/v3_training_checkpoint.joblib   (HEAVY: features+labels+HMM+4-seed ensemble)
>        │                              │
>        │                              ▼
>        └─▶ run_backtest.py  ──▶ models/saved/v3_ensemble_{5,20}d.joblib       (FAST: sweep + DSR/PBO + FIT gate)
>                                       │
>                                       ▼
>             run_bot.py / main.py --task full_pipeline ──▶ Telegram cards
> ```
> The two-script split lets you re-tune walk-forward / sizing parameters in
> minutes (`run_backtest.py`) without the ~40-min GBM retrain (`train_models.py`).

---

## 0. Prerequisites

```bash
cd /opt/quant                          # ← adjust to your install path
python3.11 -m venv .venv               # Python 3.11 (code uses 3.10+ typing)
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # PINNED + complete (incl. xgboost/catboost/telegram/genai)

mkdir -p data models/saved models/mr logs
```

Verify the import surface (all three base learners MUST be present — a missing
one silently degrades the ensemble via graceful-import fallbacks):

```bash
python -c "import lightgbm, xgboost, catboost, sklearn, hmmlearn, polars, duckdb, telegram; print('deps OK')"
python -c "import main, train_models, run_backtest; print('entry points import OK')"
```

---

## 1. Credentials — `.env`

Create `/opt/quant/.env` (auto-loaded via `python-dotenv`; git-ignored — never commit real keys):

```dotenv
# Telegram (read by telegram_bot.py / telegram_alerter.py)
TELEGRAM_BOT_TOKEN=123456789:AAExampleReplaceMe
TELEGRAM_CHAT_ID_1=-1001234567890     # Admin   — receives all broadcasts + oversight mirror
TELEGRAM_CHAT_ID_2=-1009876543210     # User    — receives signals
# (legacy comma-separated TELEGRAM_CHAT_ID is still honoured as a fallback)

# Gemini / GenAI (read by quant_agent_arbitrator.py, sentiment_crawler.py, audit_evaluator.py)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-flash-latest      # optional; this is the default
```

```bash
chmod 600 /opt/quant/.env
```

All env reads are lazy with safe defaults — a missing key disables that feature
(e.g. no `GEMINI_API_KEY` ⇒ sentiment scores neutral) rather than crashing.

**Channel smoke-test:**

```bash
source /opt/quant/.env
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID_2}" -d text="V4.0 runbook connectivity check"
```

---

## 2. Data ingestion

OHLCV lands in the DuckDB store + `data/ohlcv_*.parquet` (the V4 feature pipeline
reads parquet/DuckDB). Two supported sources:

```bash
# (a) Built-in EOD crawler — guarded to run only AFTER the 15:00 ICT close.
python main.py --task crawl_hose --days-back 1    # INCREMENTAL: previous day only (daily refresh)
python main.py --task crawl_hose                  # FULL backfill from 2016 (first run / gap repair)
#   --days-back N  → fetch only the last N calendar days (1 = previous day; 3–5 adds correction overlap)
#   --force-crawl  → bypass the 15:00 ICT time guard (operator-initiated rebuild)

# (b) External pipeline — populate data/ohlcv_*.parquet yourself (the OHLCV
#     single source of truth), then skip crawl_hose. The model/backtest never
#     crawl; they only read the parquet shards.
```

Macro + LLM-sentiment refresh happens inside `--task full_pipeline` (§4). The
mean-reversion sub-model artifacts live in `models/mr/` (§3d).

---

## 3. Train + validate the models

### 3a. Train (HEAVY — run only on data/feature/label/architecture change)

```bash
python train_models.py --tb-horizon 20 --n-configs 4 --start-date 2018-01-01 \
  2>&1 | tee logs/train_20d_$(date +%F).log
# → writes models/saved/v3_training_checkpoint.joblib
#   {ensembles (4 seeds), macro_hmm, tabular_features (iron-fist selected), cutoff, train_cfg}
```
Key flags: `--tb-horizon {5,20}`, `--n-configs` (seeds; ≥2 for PBO), `--tb-pt`/`--tb-sl`
(σ barriers, default 3.0/2.0), `--train-frac`, `--no-hmm`.

### 3b. Backtest + persist the bot artifact (FAST — iterate freely)

```bash
python run_backtest.py 2>&1 | tee logs/backtest_$(date +%F).log
# reads v3_training_checkpoint.joblib → threshold sweep + DSR + PBO →
# writes models/saved/v3_ensemble_{tb_horizon}d.joblib  (the live-bot payload)
#   + models/saved/prob_distribution.png  (calibrated P(UP) vs Kelly cap-onset)
```
Tunable WITHOUT retraining (refreshed from current defaults each run): `--liquid-top-n`
(VN50 gate, 50), `--max-positions` (5), `--max-weight` (0.20), `--sweep-thresholds`,
`--no-save` (skip writing the live artifact — **use for any dry/smoke run**).

**`--export-only`** (alias `--skip-backtest`) — repackage the live-bot artifact
straight from the checkpoint in **seconds**: it bypasses the walk-forward sweep +
DSR/PBO **and** the dataset rebuild. Use after a retrain when you only need a fresh
`v3_ensemble_{H}d.joblib`, not a new backtest. Thresholds are preserved from the
existing artifact (else defaults `up=0.50`); OOS metrics are stamped `NaN` to mark
the artifact export-only; backup-on-write still applies. ⚠️ No `FIT` gate runs —
only export a checkpoint you have already validated with a full §3b run.

> **Deployment gate:** promote to live ONLY if the teardown verdict is `FIT`
> (DSR p ≥ 0.95 **and** PBO ≤ 10%). `--n-configs ≥ 2` is required for PBO.

### 3c. Dual-horizon = run the pair twice

```bash
python train_models.py --tb-horizon 5  && python run_backtest.py   # → v3_ensemble_5d.joblib
python train_models.py --tb-horizon 20 && python run_backtest.py   # → v3_ensemble_20d.joblib
```
The checkpoint is transient (overwritten per horizon); the bot loads both
`v3_ensemble_5d.joblib` and `v3_ensemble_20d.joblib`.

Artifact-only refresh after a retrain (no re-backtest) — seconds per horizon:
```bash
python train_models.py --tb-horizon 5  && python run_backtest.py --export-only
python train_models.py --tb-horizon 20 && python run_backtest.py --export-only
```

### 3d. Mean-reversion (knife-catch) sub-model

```bash
python -m src.models.train_mr_lgbm    # → models/mr/mr_lgbm.joblib + mr_threshold.json
```
The bot loads this at inference (`mr_score_tickers`). Retrain on the same cadence
as the main ensemble, or when MR precision drifts.

### Verify artifacts before scheduling

```bash
python -c "from src.bot.bot_inference import V3BotInference; \
b=V3BotInference.from_artifact('models/saved/v3_ensemble_20d.joblib'); print(b.card())"
```

---

## 4. Run the bot

### Interactive bot (long-running service)

```bash
python run_bot.py            # or: python -m src.utils.telegram_bot
```
Commands: `/suggest_buy5`, `/suggest_buy20` (dual-horizon BUY cards with sizing),
`/verify <TICKER>`, `/suggest_sell`, `/rebalance`, `/news`, `/help`. Run it under
systemd / pm2 / supervisord so it restarts on crash.

### EOD cron — `--task full_pipeline` at 15:30 ICT

`full_pipeline` = `crawl_hose` (15:00 guard) → LLM sentiment →
`daily_inference` (broadcasts the **T+5** Top-3 to `TELEGRAM_CHAT_ID_1/_2`).
(No `build_alpha360` step — V4 recomputes features from raw OHLCV in-pipeline.)
15:30 ICT gives a safe margin after the 15:00 close for the day's bar to land.

```cron
CRON_TZ=Asia/Ho_Chi_Minh
# V4.0 EOD ingest + inference, 15:30 ICT, Mon–Fri.  --days-back 1 = previous-day-only
# incremental crawl (NOT a full 2016 re-crawl every evening).
30 15 * * 1-5 cd /opt/quant && /opt/quant/.venv/bin/python main.py --task full_pipeline --days-back 1 >> logs/eod.log 2>&1
```
> If `cron` lacks `CRON_TZ` (older Vixie/busybox): set the server to UTC and use `30 8 * * 1-5` (same command).
> T+20 broadcasts aren't part of the EOD cron — get them on demand via `/suggest_buy20`.

Manual dry-run (same code path; with a placeholder token it logs instead of sends):
```bash
python main.py --task daily_inference        # inference only, no crawl
```

---

## 5. Monitoring & troubleshooting

| Symptom | Check / cause |
|---|---|
| No Telegram message | `tail -n 80 logs/eod.log`; `.env` loaded? run the §1 `curl`. |
| `FileNotFoundError: v3_ensemble_{H}d.joblib` | Model not built. Run §3a+§3b for that horizon. |
| `FEATURE-RECIPE MISMATCH` (RuntimeError) | `build_features` changed since training → bot refuses stale model. **Retrain both horizons** (§3c) after bumping the recipe version (§6). |
| `/suggest_buy5` works but no T+5 model | The bot needs the **primary** horizon's artifact; the secondary (T+20) is an optional arbitrator cross-check and degrades gracefully if absent. |
| Card shows `Khuyến nghị đi vốn: N/A` | `suggested_weight` couldn't be computed (ticker P(UP) missing) — check the prediction dict. |
| Ensemble trained with < 3 base learners | `xgboost`/`catboost` missing from the venv — `pip install -r requirements.txt`. |

Key paths:
```
data/ohlcv_*.parquet                live OHLCV window source (read by the serve path)
models/saved/v3_training_checkpoint.joblib   transient train→backtest hand-off
models/saved/v3_ensemble_{5,20}d.joblib      LIVE bot payloads (source of truth)
models/saved/prob_distribution.png           calibrated P(UP) vs Kelly cap-onset (R-tuning)
models/mr/mr_lgbm.joblib + mr_threshold.json mean-reversion sub-model
logs/eod.log                                 daily EOD pipeline log
```

Log rotation:
```bash
sudo tee /etc/logrotate.d/quant >/dev/null <<'EOF'
/opt/quant/logs/*.log { weekly rotate 8 compress missingok notifempty }
EOF
```

---

## 6. Cadence, sizing config & operational gotchas

**Cadence.** Daily: the 15:30 EOD cron (§4). Periodic (weekly / on drift):
re-run §3c (both horizons) + §3d (MR), promote only on a `FIT` verdict.

**Sizing (locked in `src/bot/sizing.py`):** R=2.0, half-Kelly, **20% NAV cap**,
**top-5** names ⇒ 100% gross, long-only, unlevered. The cap binds at p ≥ 0.60,
so the calibrated 0.50–0.55 band sizes smoothly at 12.5–16.25% NAV. The backtest
mirrors this (`RunConfig.max_weight=0.20`, `max_positions=5`).

**Regime overrides** (`market_regime` 0–7, rule-based — `src/features/market_regime.py`):
**0 Freeze / 7 Liquidity Sweep → 0%** (stand aside), **1 Squeeze / 6 Choppy → ≤10% cap**,
**3 Strong Trend → full Kelly to the 20% cap**; 2/4/5 size normally. The detected
regime renders on the card: `Pha thị trường: <nhãn> (Regime N)`.

**Feature-recipe tripwire.** Whenever you change the hardcoded feature logic in
`build_features` (add/remove/retune a feature, window, or ordering), **bump
`src/backtest/pipeline.FEATURE_RECIPE_VERSION`** and **retrain both horizons**.
The trained artifact stamps the version; the bot asserts it at load and refuses a
drifted model (loud `RuntimeError`). **Current = `"v1.1"`** — bumped from `v1.0`
when the `market_regime` categorical feature was added, so any artifact trained at
`v1.0` is rejected and MUST be rebuilt (full retrain, or `--export-only` re-export
from a `v1.1` checkpoint).

**⚠️ Artifact registry.** `models/saved/` is git-untracked and unversioned, but
`run_backtest.py` now **auto-backs-up** the existing `v3_ensemble_{H}d.joblib` to
`models/saved/backups/v3_ensemble_{H}d_<UTC-timestamp>.joblib` before each
overwrite — so a bad run is recoverable (a synthetic smoke run once clobbered the
live model with no rollback; that gap is now closed). Still:
- Any dry/smoke run SHOULD use `run_backtest.py --no-save` to avoid backup churn.
- `models/saved/backups/` accumulates over time — prune old copies periodically.
- The MR artifacts (`models/mr/`) have NO auto-backup — copy them before retraining (§3d).

---

## 7. Tests & CI

```bash
pytest -q                              # 120 tests, ~3s (serve-path + unit)
python -m compileall -q main.py run_bot.py train_models.py run_backtest.py src
```
`.github/workflows/ci.yml` runs the pinned install + byte-compile + `pytest` on
every push/PR. The serve-path tests (`tests/test_cards.py`, `test_sizing.py`,
`test_serve_resilience.py`, `test_feature_serve.py`) guard the sizing/formatting,
dual-horizon resilience, and OHLCV feature path.

> **Paper-trading only.** The bot emits target weights and signal cards; it does
> NOT route live orders. Promote to capital only after sustained paper-trading
> with a `FIT` verdict.
