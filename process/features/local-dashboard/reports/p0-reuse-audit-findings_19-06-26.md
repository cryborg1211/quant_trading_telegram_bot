# P0 — Reuse Audit Findings (callable map)

Date: 2026-06-19. Source: research-agent headless-callability audit. Feeds P2 (logic linking).

## Callable map — dashboard calls these direct (no PTB)

| # | Call | Returns | Verdict |
|---|------|---------|---------|
| 1 | `main.daily_inference(window_rows=120, max_candidates=6, broadcast=False, horizon=H)` (`main.py:934`) | `(html, list[dict])` | HEADLESS-OK for sends — **but see GOTCHA 1** |
| 2 | `main.verify_single_ticker(ticker, window_rows=120)` (`main.py:1560`) | `html: str` | HEADLESS-OK (only side-effect: optional paperlog write, try/except) |
| 3 | `main.inference_for_holdings(holding_tickers: list[str], window_rows=120)` (`main.py:1469`) | `html: str` | HEADLESS-OK, no DB write, no telegram. SELL/HOLD verdict via `evaluate_trades_batch` + `_build_sell_hold_report` |
| 4 | portfolio: raw DuckDB on `portfolio` table (`user_id,ticker,volume,price,added_at`) | — | No add/remove/list API on `PortfolioManager`. Bot does raw `INSERT`/`DELETE` (`telegram_bot.py:778,863`). Dashboard mirrors raw SQL via `DuckDBEngine()` |
| 5 | `signal_ledger.list_open()` / `check_exits_due()` (`src/trading/signal_ledger.py`) | `list[dict]` (incl. `sessions_remaining`) | HEADLESS-OK, pure read |
| 6 | `audit_evaluator.run_post_mortem(user_id, days)` (`src/utils/audit_evaluator.py:70`) | `html: str` | HEADLESS-OK — **see GOTCHA 2 (user_id)**. Needs `GEMINI_API_KEY` |
| 7 | `telegram_alerter.TelegramBot().send_text_alert(html, label)` / `send_signal_alert(dict)` (`telegram_alerter.py:96`) | `None` | HEADLESS-OK, sync `requests.post`, send-only, no poll |
| 8 | config: `.env` via `load_dotenv(override=True)`; `Config.from_json()` reads `config/settings.json` | — | **see GOTCHA 3 (hot-reload)** |

`dispatched_signals` dict keys (MUA cards): action, ticker, price, horizon_label, suggested_weight, status, ly_do, market_regime, regime_label, prob_up/side/down, conclusion, sentiment_score, sentiment_status, gemini_summary, article_urls, confidence, hold_label, exit_rule.

## GOTCHAS (resolve in P2)

1. **`daily_inference(broadcast=False)` still mutates DuckDB every call** — `run_trade_execution` always runs `PortfolioManager().update_live_performance` + `process_daily_trades` + RL log + paperlog, NOT gated on `broadcast`. Calling it for a dashboard "preview" writes to the cron `portfolio` book. → P2 needs a preview-safe path: a read-only inference that returns signals WITHOUT `run_trade_execution` side-effects, OR a new `persist=False` gate. **Do not call `daily_inference` for mere display until this is solved.**
2. **`user_id` consistency** — `audit_log` + paperlog are per-user. Bot wrote rows under the Telegram numeric id. Dashboard must pass a FIXED local `user_id`; if different from bot's, audit returns empty. Decide one stable value (P2/Settings).
3. **Config hot-reload** — `.env` reloads via `load_dotenv(override=True)`. `config/settings.json` does NOT auto-reload the `CONFIG` singleton (built once at import). Settings tab → either re-construct `CONFIG` or restart-to-apply. Dashboard must call `load_dotenv()` at startup before importing serve modules.

## PTB-coupled blockers
All interactive commands (`_suggest_buy_dispatch`, `verify_command`, `suggest_sell_command`, `_run_audit_command`, `rebalance_command`, add/remove) are thin PTB wrappers around the headless fns above — **no logic extraction needed**, dashboard calls the underlying fn. `build_application()` is the only poller — dashboard must NEVER call it. `_split_html_report` + `_build_exits_report` are pure (reusable).

## Verdict
P0 GREEN. Core chain headless. Two real P2 design items: preview-safe inference (GOTCHA 1) + fixed user_id (GOTCHA 2).
