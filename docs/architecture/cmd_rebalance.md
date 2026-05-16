# `/rebalance` — AI Portfolio Rebalancing Advisor

**Entry point:** `telegram_bot.rebalance_command` → `main.rebalance_portfolio(user_id)`
**Storage:** `portfolio` table, filtered by `user_id`
**LLM:** `quant_agent_arbitrator.get_rebalance_advice` (Gemini, ≤4 Vietnamese sentences)
**Heavy work runs in:** `asyncio.to_thread`; protected by the 30s rate limiter

## Summary

The only command that combines **cost basis (PnL)** + **model forecast** +
**news** into a single LLM-authored portfolio action. Reads `(ticker, price)`
for the user's holdings, computes per-ticker unrealized PnL against the live
price, runs the **5d** Stacking GBDT, scrapes news for the held names, and
asks Gemini for a concrete rebalance recommendation (hold / take-profit /
cut-loss / rotate capital).

## Sequence

```mermaid
sequenceDiagram
    actor User
    participant Bot as telegram_bot.rebalance_command
    participant RL as _check_and_record_rate_limit
    participant RP as main.rebalance_portfolio
    participant DB as DuckDBEngine (portfolio)
    participant A360 as build_live_features(tickers=held)
    participant ML as predict_stacking_horizon (5d)
    participant PX as _get_live_exec_prices
    participant NEWS as scrape_centralized_news + map_tickers_to_news
    participant GEM as get_rebalance_advice (Gemini)
    participant RPT as _build_rebalance_report

    User->>Bot: /rebalance
    Bot->>RL: user_id + "rebalance"
    alt within 30s cooldown
        RL-->>Bot: cooldown_left
        Bot-->>User: ⏳ Quá nhanh!
    else allowed
        Bot->>Bot: _audit_log_async(rebalance)
        Bot-->>User: ⏳ wait message
        Bot->>RP: await to_thread(rebalance_portfolio, user_id)
        RP->>DB: SELECT ticker, price FROM portfolio WHERE user_id=?
        DB-->>RP: rows
        alt no rows
            RP-->>Bot: "" (empty)
            Bot-->>User: EMPTY_PORTFOLIO_MESSAGE
        else
            RP->>RP: dedup → {ticker: entry_price}
            RP->>A360: build_live_features(tickers=held, 120)
            A360-->>RP: latest_df
            RP->>ML: predict_stacking_horizon(latest_df, 5)
            ML-->>RP: {ticker:[p_dn,p_sd,p_up]}
            RP->>PX: live close (VN scaling)
            PX-->>RP: {ticker: current_price}
            RP->>RP: pnl_pct = (cur-entry)/entry × 100
            RP->>NEWS: scrape + map news for held tickers
            NEWS-->>RP: {ticker: [headlines]}
            RP->>GEM: get_rebalance_advice(holdings_ctx, news)
            GEM-->>RP: ≤4-sentence VN advice
            RP->>RPT: _build_rebalance_report (HTML-escaped)
            RPT-->>RP: report_html
            RP-->>Bot: report_html
            Bot-->>User: ⚖️ TƯ VẤN CƠ CẤU DANH MỤC
        end
    end
```

## `holdings_context` payload to the LLM

Per held ticker, the prompt receives:

```text
ticker      : str
pnl_pct     : (current_price - entry_price) / entry_price * 100   # 0 if entry<=0
pred_label  : 🔴 Giảm | 🟡 Đi ngang | 🟢 Tăng   (argmax of 5d probs)
p_up        : float (5d UP probability)
```

`REBALANCE_SYSTEM_PROMPT` instructs Gemini to act as a Quant Portfolio
Manager and return ≤4 Vietnamese sentences, no markdown,
`temperature=0.2`. Missing API key → graceful fallback string.

## Notes / risks

- **5d only.** Unlike `/suggest_buy`/`/suggest_sell`, `/rebalance` runs a
  single horizon (5d). It does not pass a 20d view to the LLM, so the
  advice has no medium-term anchor.
- PnL uses the **stored entry `price`** vs. **live close** — it ignores
  fees, partial fills, and multiple lots (dedup keeps the last `price`).
- The LLM output is **free-text advice, not an executable order** — no
  `portfolio` write happens. `/rebalance` is advisory only.
