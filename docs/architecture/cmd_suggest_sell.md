# `/suggest_sell` — Holdings SELL/HOLD Recommendation

**Entry point:** `telegram_bot.suggest_sell_command` → `main.inference_for_holdings`
**Storage:** `portfolio` table, **filtered by `user_id`** (multi-user safe)
**Heavy work runs in:** `asyncio.to_thread`

## Summary

Reads **this user's** holdings from the `portfolio` table, runs dual-horizon
Stacking GBDT + the LLM arbitrator on those tickers only, and returns a
BÁN (SELL) / GIỮ (HOLD) recommendation per holding. Unlike `/suggest_buy`,
it **skips the liquidity gate and the Top-6/Top-3 funnel** — every holding
gets a verdict. Tickers absent from the live feature universe (delisted /
not crawled) are reported as warnings.

## Sequence

```mermaid
sequenceDiagram
    actor User
    participant Bot as telegram_bot.suggest_sell_command
    participant DB as DuckDBEngine (portfolio)
    participant IFH as main.inference_for_holdings
    participant A360 as Alpha360Generator.build_live_features
    participant ML as predict_stacking_horizon (5d & 20d)
    participant ARB as evaluate_trades_batch
    participant GEM as Gemini (batch sentiment)
    participant RPT as _build_sell_hold_report

    User->>Bot: /suggest_sell
    Bot->>Bot: _extract_user_id(update)
    alt user_id is None (group chat)
        Bot-->>User: ❌ yêu cầu chat 1-1
    else
        Bot->>DB: SELECT DISTINCT ticker FROM portfolio WHERE user_id=?
        DB-->>Bot: holding_tickers
        Bot->>Bot: _audit_log_async(suggest_sell, portfolio_size)
        alt no holdings
            Bot-->>User: EMPTY_PORTFOLIO_MESSAGE
        else
            Bot-->>User: ⏳ Đang phân tích N cổ phiếu...
            Bot->>IFH: await to_thread(inference_for_holdings, tickers)
            IFH->>A360: build_live_features(window_rows=120)
            A360-->>IFH: latest_df (whole universe tail)
            IFH->>IFH: filter to held tickers present in universe
            IFH->>ML: predict 5d, predict 20d (held only)
            ML-->>IFH: probability vectors
            IFH->>ARB: evaluate_trades_batch(horizons, present)
            ARB->>GEM: news scrape + batch sentiment
            GEM-->>ARB: scores + reasoning_vi + urls
            ARB-->>IFH: final_decisions, all_sentiments
            IFH->>IFH: _get_live_exec_prices (VN price scaling)
            IFH->>RPT: build HTML (escaped) incl. missing tickers
            RPT-->>IFH: report_html
            IFH-->>Bot: report_html
            Bot->>Bot: _split_html_report
            Bot-->>User: SELL/HOLD per holding (parse_mode=HTML)
        end
    end
```

## Decision mapping

The same `make_final_decision` dual-horizon veto runs as in `/suggest_buy`
(see [`cmd_suggest_buy.md`](cmd_suggest_buy.md)). For holdings the integer
verdict is rendered as:

| `decision` | Holdings meaning |
|------------|------------------|
| `0` | 🔴 BÁN (SELL) — full exit |
| `1` | 🟡 GIỮ (HOLD) — veto / neutral |
| `2` | 🟢 GIỮ / MUA THÊM (strong hold) |

## Notes / risks

- **Decoupled from entry price.** `/suggest_sell` is a *signal* view; it
  does **not** read the user's `price` (cost basis) and therefore ignores
  realized/unrealized PnL, stop-loss, and take-profit. Risk exits live only
  in `PortfolioManager.update_live_performance` (cron path), not here.
- Holdings missing from the live universe are surfaced as a warning block —
  they are silently skipped by the model, not failed.
- No liquidity gate: an illiquid holding still gets a (low-confidence)
  model verdict.
