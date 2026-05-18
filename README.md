# Quant V6 — Vietnamese Equity Trading Platform

Automated quantitative trading system for the **VN100 / HOSE** universe. Combines a
**dual-horizon Stacking GBDT** model (XGBoost + LightGBM + CatBoost → logistic meta)
with **Gemini-driven news sentiment** to generate daily 5-day and 20-day directional
signals, executed via an **interactive Telegram bot** with multi-user portfolios.

> **Status:** production · multi-user · runs unattended overnight on a single VPS
> **Stack:** Python 3.11 · DuckDB · Polars · XGBoost/LightGBM/CatBoost · Gemini 2.5 Flash · python-telegram-bot v20+

---

## What this does

- **Ingests** daily OHLCV for ~355 HOSE tickers via `vnstock` 4.x, plus US macro (DXY, S&P 500, USD/VND) via `yfinance`.
- **Engineers** Alpha360-style features — 6 raw fields × 60 rolling-Z-score lags per ticker = 360 lag columns, plus VWAP, T-1-shifted macro, and lagged sentiment aggregates.
- **Trains** a Stacking GBDT ensemble (3 base learners + logistic meta) for both 5-day and 20-day classification horizons. Quantile-thresholded UP/SIDE/DOWN labels.
- **Scrapes** Vietnamese financial news in parallel across 5 portals (cafef, vietstock, tinnhanhchungkhoan, vneconomy, vietnambiz) and routes the diverse top-3 articles per ticker to **Gemini 2.5 Flash** for sentiment + reasoning.
- **Dispatches** Top-3 BUY signals per session via Telegram, with a Vietnamese HTML report including the LLM's reasoning, top model drivers, and source URLs.
- **Manages** per-user portfolios via 9 interactive bot commands (`/suggest_buy`, `/verify`, `/add`, `/remove`, `/suggest_sell`, `/audit_weekly`, `/audit_monthly`, `/news`, `/help`).
- **Logs** every command for post-mortem accuracy review (`/audit_weekly` correlates past `/verify` and `/add` calls against actual price moves, with Gemini explaining the catalysts).

See [`docs/ARCHITECTURE_V6.md`](docs/ARCHITECTURE_V6.md) for the deeper system design.

---

## Quick start (< 5 minutes)

```bash
# 1. Clone and enter the project
git clone <repo-url> stock_price_v3
cd stock_price_v3

# 2. Create venv and install dependencies (see Dependencies section below)
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt     # see "Dependencies" if file is missing

# 3. Configure env — copy template and fill in your keys
cp .env.example .env
# Edit .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY

# 4. (Cold-start only) crawl historical OHLCV — outside market hours
python -X utf8 main.py --task crawl_hose --force-crawl

# 5. (Cold-start only) build Alpha360 features
python -X utf8 main.py --task build_alpha360

# 6. (Cold-start only) train the Stacking GBDT model
python -X utf8 -m src.models.stacking_model.train_stacking

# 7. Run daily inference (no crawling, uses existing data)
python -X utf8 main.py --task daily_inference

# 8. (Long-running) Start the interactive Telegram bot
python -X utf8 run_bot.py
```

> **Windows users:** always use `python -X utf8` to avoid `UnicodeEncodeError` from
> emoji in Vietnamese output.

---

## Telegram bot commands

Once `run_bot.py` is running and you've messaged the bot:

| Command | What it does |
|---|---|
| `/help` (or `/start`) | Show the command menu |
| `/suggest_buy` | Run the full pipeline; reply with Top-3 BUY signals (Quant + Sentiment). 30s per-user cooldown. |
| `/suggest_sell` | BÁN / GIỮ recommendation for every ticker in your `portfolio` |
| `/verify <ticker>` | Ad-hoc 5d Quant + LLM Sentiment verdict on one ticker (e.g. `/verify HPG`). 30s cooldown. |
| `/add <ticker> <volume> <price>` | Add a position to your personal portfolio |
| `/remove <ticker>` | Remove every lot of that ticker from your portfolio |
| `/audit_weekly` / `/audit_monthly` | Post-mortem: actual % return + Gemini-explained catalyst for every `/verify` and `/add` you made in the last 7 / 30 days |
| `/news` | The 20 most recent items from the configured Vietnamese RSS feeds |

All replies use Telegram HTML mode (`<b>`, `<i>`, `<code>`). Per-user data isolation
is enforced at the DB layer: each user only ever sees their own portfolio + audit
history.

---

## CLI tasks (`main.py`)

```bash
python -X utf8 main.py --task <task-name> [options]
```

| Task | Purpose | When to run |
|---|---|---|
| `daily_inference` *(default)* | Live pipeline: load features → predict → arbitrate → dispatch Telegram alerts | Daily, ~16:00 ICT via cron |
| `crawl_hose` | Crawl OHLCV for the full HOSE universe | Cold-start only, or `--force-crawl` after market close |
| `build_alpha360` | Rebuild `data/alpha360_features.parquet` from the full DuckDB | After a full historical re-crawl |
| `full_pipeline` | All of the above in sequence | Weekly / monthly retrain cycle |

Useful options: `--window-rows N` (per-ticker tail rows for live inference, default 120),
`--max-candidates N` (Top-N pool sent to the arbitrator, default 6),
`--force-crawl` (bypass the 15:00 market-close guard).

---

## Architecture at a glance

```text
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│ vnstock 4.x │───>│  DuckDB      │───>│ Alpha360      │───>│ Stacking GBDT  │
│ yfinance    │    │  + parquet   │    │ (Polars +     │    │ (5d + 20d)     │
│             │    │              │    │  60 lags)     │    │                │
└─────────────┘    └──────────────┘    └───────────────┘    └────────┬───────┘
                          ▲                                          │
                          │                                          v
                   ┌──────┴───────┐    ┌────────────────┐    ┌───────────────┐
                   │ Audit log    │    │ News scraper   │───>│ Gemini 2.5    │
                   │ Portfolio    │    │ (5 VN portals  │    │ Flash         │
                   │ Trade history│    │  in parallel)  │    │ (Sentiment +  │
                   │ RL log       │    │                │    │  Reasoning)   │
                   └──────────────┘    └────────────────┘    └───────┬───────┘
                          ▲                                          │
                          │            ┌────────────────┐            │
                          └────────────┤  Arbitrator    │<───────────┘
                                       │  (Top-6 → 3)   │
                                       └────────┬───────┘
                                                │
                                                v
                                       ┌─────────────────┐
                                       │ Telegram bot    │
                                       │ (run_bot.py)    │
                                       └─────────────────┘
```

Full sequencing, leak-prevention rules, and quant decisions are in
[`docs/ARCHITECTURE_V6.md`](docs/ARCHITECTURE_V6.md).

---

## Directory layout

```text
stock_price_v3/
├── main.py                       # CLI orchestration + crash alerter
├── run_bot.py                    # Telegram bot entrypoint (systemd-friendly)
├── config/
│   ├── settings.py               # CONFIG dataclasses
│   └── settings.json             # User overrides
├── src/
│   ├── data/
│   │   ├── db_engine.py          # DuckDB singleton, schema, migrations
│   │   └── crawlers.py           # StockCrawler (vnstock) + MacroCrawler (yfinance/TE)
│   ├── features/
│   │   └── alpha360_generator.py # Polars rolling-Z-score lag matrix builder
│   ├── models/
│   │   ├── stacking_model/
│   │   │   └── train_stacking.py # 3-base + logistic-meta trainer
│   │   └── quant_agent_arbitrator.py  # Async news scrape + Gemini + Top-6→3
│   ├── crawlers/
│   │   └── sentiment_crawler.py  # Historical LLM-labelled sentiment archive
│   ├── trading/
│   │   └── portfolio_manager.py  # Unified `portfolio` table (cron + bot users)
│   ├── rl/
│   │   └── trading_env.py        # RL env scaffolding (Phase-3, WIP)
│   └── utils/
│       ├── telegram_bot.py       # Long-running bot: handlers, rate limiter
│       ├── telegram_alerter.py   # One-shot push alerter (cron path)
│       ├── audit_evaluator.py    # /audit_weekly + /audit_monthly engine
│       └── logging_utils.py      # RotatingFileHandler factories (10 MiB × 5)
├── data/
│   ├── quant_v6_core.duckdb      # Single source of truth (OHLCV, macro, portfolio, audit, RL)
│   ├── ohlcv_<TICKER>.parquet    # Per-ticker historical parquet (one per HOSE symbol)
│   ├── macro_daily.parquet       # Wide-format global + VN macro
│   └── alpha360_features.parquet # Full training matrix
├── models/stacking/
│   ├── 5d/                       # selected_features.json, scaler.joblib, xgboost/lightgbm/catboost/meta artifacts
│   └── 20d/
├── logs/                         # Rotating: quant_v6.log, crawler_errors.txt (10 MiB × 5)
├── backups/                      # Daily DuckDB cp backups (14-day retention)
├── deploy/
│   └── quant-v6-bot.service      # systemd unit
├── scripts/
│   └── backup_db.sh              # Cron-friendly daily DB backup
├── docs/
│   ├── ARCHITECTURE_V6.md        # System design + leak-prevention rules
│   └── RUNBOOK.md                # Operational procedures (planned, see TD-39)
├── AUDIT_REPORT.md               # Historical tech-debt audits
└── .env.example                  # Required env var template
```

---

## Configuration

### Required environment variables (`.env`)

```bash
# Telegram bot (single bot, comma-separated chat IDs for multi-recipient pushes)
TELEGRAM_BOT_TOKEN=123456:ABCdef...
TELEGRAM_CHAT_ID=123456789,987654321

# Google Gemini (https://aistudio.google.com/apikey)
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-flash-latest      # optional override
```

A template lives in [`.env.example`](.env.example).

### Code-level configuration (`config/settings.py`)

Trading thresholds (`stop_loss_pct`, `take_profit_pct`, `virtual_allocation_per_ticker`),
crawler throttles, model hyperparameters, sentiment LLM settings — all defined as
typed dataclasses in `config/settings.py`. Override at runtime by editing
`config/settings.json` (auto-loaded by `Config.from_json`).

---

## Operations

### Production deployment

The bot is designed to run as a systemd service. Unit file in
[`deploy/quant-v6-bot.service`](deploy/quant-v6-bot.service). Quick install
(operator runbook will live at `docs/RUNBOOK.md`):

```bash
sudo cp deploy/quant-v6-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-v6-bot.service
sudo systemctl status quant-v6-bot.service
journalctl -u quant-v6-bot.service -f
```

### Daily cron schedule (recommended)

| Time (ICT) | Command | Purpose |
|---|---|---|
| **16:00** | `python -X utf8 main.py --task daily_inference` | Live inference after market close (15:00 guard ensures the daily candle is closed) |
| **23:00** | `scripts/backup_db.sh` | DuckDB backup, 14-day retention |
| Weekly | Manual full_pipeline + retrain | Recommend Sunday morning before market opens Monday |

### Logs

- `logs/quant_v6.log` — Main application log, rotated (10 MiB × 5 backups → max 60 MiB)
- `logs/crawler_errors.txt` — Per-ticker crawler failures (rotated, same policy)
- `/var/log/quant-v6-bot.log` (when running under systemd) — stdout/stderr capture
- `journalctl -u quant-v6-bot.service` — systemd-level events

### Backups & restore

```bash
# Backup (automated daily)
scripts/backup_db.sh

# Restore (stop the bot first)
sudo systemctl stop quant-v6-bot.service
cp backups/quant_v6_core_YYYYMMDD.duckdb data/quant_v6_core.duckdb
sudo systemctl start quant-v6-bot.service
```

---

## Dependencies

Python 3.11 is required. There is **no `requirements.txt` at the project root yet**
(open tech-debt item) — install the dependency set manually:

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

> **GPU note:** XGBoost / LightGBM / CatBoost are configured for CUDA in
> `train_stacking.py`. If you don't have a GPU, edit the `build_base_models()`
> block to swap `device="cuda"` → `device="cpu"` (tracked as TD-10).

---

## Development

### Code style

No formatter is currently enforced. The codebase follows informal Black + isort
conventions with type hints throughout. **No CI / pre-commit hooks yet** (open
tech-debt item TD-32 — planned).

### Tests

**Currently no test suite** (open tech-debt item TD-17). Verification has been
done via sandbox dry-runs inside development sessions. Adding `tests/` with
pytest is on the immediate roadmap — covering the rate limiter, RL backfill,
portfolio isolation, audit evaluator, news diversity selector, and DuckDB
schema migrations is the highest-leverage starting set.

### Running a smoke check

```bash
# 1. Syntax sanity across the main modules
python -X utf8 -c "
import ast
for p in ['main.py', 'run_bot.py', 'src/data/db_engine.py',
          'src/utils/telegram_bot.py', 'src/models/quant_agent_arbitrator.py',
          'src/features/alpha360_generator.py']:
    ast.parse(open(p, encoding='utf-8').read())
    print('OK', p)
"

# 2. Crash-alert dry-run (mock token, no network)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN python -X utf8 -c "
import traceback
from main import _send_crash_alert
try: raise ValueError('smoke-check')
except ValueError as e: _send_crash_alert('smoke', e, traceback.format_exc())
"

# 3. DuckDB schema + migration smoke
python -X utf8 -c "
from src.data.db_engine import DuckDBEngine
db = DuckDBEngine()
print('tables:', db.conn.execute('SHOW TABLES').fetchall())
"
```

---

## Further reading

| Doc | Purpose |
|---|---|
| [`docs/ARCHITECTURE_V6.md`](docs/ARCHITECTURE_V6.md) | Full system design — leak-prevention rules, model decisions, data flow |
| `docs/RUNBOOK.md` *(planned)* | Operational procedures: deploy, restore, incident response, escalation |
| [`AUDIT_REPORT.md`](AUDIT_REPORT.md) | Historical tech-debt audits |
| [`.env.example`](.env.example) | Required env vars (copy to `.env` and fill in) |
| [`deploy/quant-v6-bot.service`](deploy/quant-v6-bot.service) | systemd unit |

---

## Known gaps / tech debt

These items are tracked in audit reports and recent work is ongoing. Major open
items at the time of this README:

- **No `requirements.txt`** — install list above must be kept in sync manually.
- **No automated tests** — TD-17, highest long-term risk.
- **VN macro features (`vnibor`, `inflation_yoy`)** — columns exist in `macro_daily`
  but are not populated. Both upstream sources (vnstock 4.x money-market API, and
  `markets.tradingeconomics.com` DNS) are unavailable from VN ISPs. Tracked as TD-25.
- **No CI / pre-commit** — TD-32.
- **Architecture doc** is from before the Telegram bot / audit log / RL backfill /
  portfolio merge work — partial refresh pending.

The active tech-debt log lives in the latest audit at the top of
[`AUDIT_REPORT.md`](AUDIT_REPORT.md).

---

## License / contributing

Not yet specified. Private project for now. If you've been granted access and want
to contribute: open a PR with a clear description of the change, run the smoke
checks above, and tag the maintainer.
