# Runbook — Filling the Database

How to populate the market data this engine runs on. Quick reference for a fresh
machine or a stale `data/` dir.

## What "the database" actually is

| Store | Path | Holds | Created by |
|---|---|---|---|
| **OHLCV parquet shards** (primary) | `data/ohlcv_<TICKER>.parquet` | per-ticker daily OHLCV — the real data source the whole system reads | `--task crawl_hose` |
| DuckDB | `data/quant_v6_core.duckdb` | signal/exit ledger + portfolio state (NOT OHLCV) | auto-created on first use by `src/data/db_engine.py` |

`data/` is git-ignored (regenerable). There is no manual schema step — fetching
the shards *is* filling the database. Ticker universe is pulled automatically
from the HOSE listing via `vnstock` (free source, **no API key needed** to crawl).

## Prerequisites

```powershell
# Python 3.11
pip install -r requirements.txt
```

`.env` is **only** needed for `full_pipeline` / the bot (Telegram + Gemini keys).
Plain OHLCV crawling needs none. See `.env.example` for the keys.

## 1. First fill — full history

Crawls HOSE from the 2016 start date into `data/ohlcv_*.parquet`:

```powershell
python main.py --task crawl_hose --force-crawl
```

- `--force-crawl` bypasses the 15:00 ICT market-hour guard (required for any
  manual / off-hours run).
- Takes a while (~350 tickers, ~45s hard cap each). Per-ticker failures are
  logged to `logs/crawler_errors.txt` and skipped — re-run to backfill them.

## 2. Daily refresh — incremental 

Fetch only the last N calendar days (small overlap absorbs late corrections):

```powershell
python main.py --task crawl_hose --force-crawl --days-back 5
```

## 3. Full EOD pipeline (crawl + sentiment + inference + exit alerts)

Needs `.env` (Telegram + Gemini). This is what cron runs daily at 15:30 ICT:

```powershell
python main.py --task full_pipeline --force-crawl
```

## 4. Verify the fill

```powershell
# shard count
(Get-ChildItem data/ohlcv_*.parquet).Count
# spot-check one ticker's latest close
python -c "from src.data import price_lookup; print(price_lookup.latest_close('FPT'))"
```

## Gotchas

- **Market-hour guard:** `crawl_hose` / `full_pipeline` no-op during live hours
  unless `--force-crawl` is passed.
- **Price scale:** parquet OHLCV is in *thousands* of VND. Downstream code scales
  to absolute VND via `WalkForwardConfig.price_unit_vnd=1000` — never feed raw
  shard prices into `VNCostModel` math.
- **Errors:** check `logs/crawler_errors.txt`; re-running the crawl backfills any
  skipped tickers.

## After the data is in

```powershell
python train_models.py --tb-horizon 20          # train (writes models/saved/*.joblib)
python run_backtest.py --mode tranche --hold-days 30   # walk-forward eval
python run_bot.py                                # Telegram bot (polling)
```
