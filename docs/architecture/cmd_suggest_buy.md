# `/suggest_buy` — Daily BUY Signal Generation

**Entry point:** `telegram_bot.suggest_buy_command` → `main.daily_inference(broadcast=False)`
**Heavy work runs in:** `asyncio.to_thread` (event loop stays responsive)

## Summary

On-demand run of the full daily inference funnel. Builds live Alpha360
features for the **entire HOSE universe**, runs dual-horizon Stacking GBDT
inference, applies a liquidity gate, narrows to a Top-6 pool, sends that
pool through the LLM news arbitrator, and replies with the final **Top-3
BUY** signals. `broadcast=False` suppresses the cron push-alert path so the
interactive reply is the only delivery channel (no duplicate spam).

## Funnel

| Stage | Logic | Cardinality |
|-------|-------|-------------|
| Universe | `build_live_features(window_rows=120)` | ~360 tickers |
| Quant | `predict_stacking_horizon` 5d + 20d | ~360 |
| Liquidity gate | 20-day ADDV ≥ 15,000,000,000 VND | N liquid |
| Arbitrator pool | Top-6 by 5d `p_up`, liquid only | ≤ 6 |
| LLM arbitration | `evaluate_trades_batch` (news + Gemini + veto) | ≤ 6 |
| Final filter | `decision == 2`, sorted by sentiment ↓ then `p_up` ↓ | Top-3 |

## Sequence

```mermaid
sequenceDiagram
    actor User
    participant Bot as telegram_bot.suggest_buy_command
    participant RL as _check_and_record_rate_limit
    participant DI as main.daily_inference
    participant A360 as Alpha360Generator.build_live_features
    participant ML as predict_stacking_horizon (5d & 20d)
    participant LQ as Liquidity Filter (ADDV ≥ 15B VND)
    participant ARB as evaluate_trades_batch
    participant GN as AsyncNewsScraper (GNews)
    participant GEM as Gemini (batch sentiment)
    participant DEC as make_final_decision (dual-horizon veto)
    participant EXE as run_trade_execution

    User->>Bot: /suggest_buy
    Bot->>RL: user_id + "suggest_buy"
    alt within 30s cooldown
        RL-->>Bot: cooldown_left
        Bot-->>User: ⏳ Quá nhanh! đợi Ns
    else allowed
        RL-->>Bot: None
        Bot->>Bot: _audit_log_async(suggest_buy)
        Bot-->>User: WAIT_MESSAGE_BUY
        Bot->>DI: await to_thread(daily_inference, broadcast=False)
        DI->>A360: build_live_features(window_rows=120)
        A360-->>DI: latest_df (1 row/ticker, 360 feats)
        DI->>ML: predict 5d, predict 20d
        ML-->>DI: {ticker:[p_dn,p_sd,p_up]} × 2 horizons
        DI->>LQ: 20d SMA(close×volume) ≥ 15B VND
        LQ-->>DI: liquid_tickers set
        DI->>DI: Top-6 liquid by p_up → candidate pool
        DI->>ARB: evaluate_trades_batch(horizons, pool)
        ARB->>GN: scrape_centralized_news(Top-25 by p_up)
        GN-->>ARB: raw articles
        ARB->>GEM: batch (size 5) sentiment scoring
        GEM-->>ARB: {ticker: score, reasoning_vi, urls}
        ARB->>DEC: per-ticker dual-horizon veto
        DEC-->>ARB: decision ∈ {0,1,2}
        ARB-->>DI: final_decisions, all_sentiments
        DI->>DI: keep decision==2; sort sentiment↓, p_up↓; take Top-3
        DI->>EXE: run_trade_execution(top_buy_signals, …)
        EXE-->>DI: combined HTML report
        DI-->>Bot: report_html
        Bot->>Bot: _split_html_report (≤4096 char chunks)
        Bot-->>User: Top-3 BUY signals (parse_mode=HTML)
    end
```

## Arbitrator decision logic (`make_final_decision`)

Evaluated in order; first match wins. `pred_5d`/`pred_20d` = `argmax` of the
class-probability vector.

1. `pred_5d == UP` **and** `sentiment < -0.5` → **1 (HOLD)** — safety override
2. `pred_5d == UP` → **2 (BUY)**
3. `pred_5d == SIDE` **and** `pred_20d == UP` → **2 (trend-active HOLD/BUY)**
4. `pred_5d == DOWN` **and** `pred_20d == DOWN`:
   - `sentiment > 0.5` → **1 (sentiment veto)**
   - else → **0 (full exit)**
5. `pred_5d == DOWN` **and** `sentiment > 0.5` → **1 (partial veto)**
6. else → `pred_5d`

Tickers with **no news coverage** get `p_up *= 0.95` (activity penalty)
before arbitration.

## Notes / risks

- The model recall on the UP class is the binding constraint — see
  [`../../docs/ARCHITECTURE_V6.md`](../ARCHITECTURE_V6.md) and the model
  critique.
- The arbitrator only scrapes news for the **Top-25** by `p_up`, but the
  decision loop iterates **all** predictions; non-Top-25 tickers fall back
  to `sentiment = 0.0`.
- `broadcast=False` is essential here — `True` is the cron-only path.
