# Portfolio CRUD — `/add`, `/remove`, `/verify`

**Table:** `portfolio (user_id VARCHAR, ticker VARCHAR, volume INTEGER, price DOUBLE, added_at TIMESTAMP)`
**Isolation:** every row is tagged with the Telegram `user_id`; all reads/writes filter on it.
**SQL:** 100% parameterized (`?` placeholders) — injection-safe.

## Table schema & ownership

```mermaid
erDiagram
    portfolio {
        VARCHAR   user_id    "Telegram user id, or 'cron' / 'legacy'"
        VARCHAR   ticker     "VN equity symbol (upper)"
        INTEGER   volume     "shares"
        DOUBLE    price      "entry / cost-basis price"
        TIMESTAMP added_at   "insert time"
    }
    audit_log {
        VARCHAR   user_id
        VARCHAR   command
        VARCHAR   ticker
        VARCHAR   details
        TIMESTAMP timestamp
    }
    trade_history {
        INTEGER   id PK
        VARCHAR   telegram_id
        VARCHAR   ticker
        VARCHAR   action
        DOUBLE    price
        DATE      date
        DOUBLE    pnl_percent
    }
    portfolio ||..o{ audit_log : "every mutation logged"
```

> **Migration note:** `_init_portfolio_table` detects a legacy single-user
> schema and `ALTER TABLE … ADD COLUMN user_id`, tagging old rows
> `user_id='legacy'` so they never surface in a real user's queries.
> Automated cron positions use `user_id='cron'` (`PortfolioManager`).

## `/add <ticker> <volume:int> <price:float>`

Pure INSERT — **no dedup, no upsert.** Repeated `/add VNM …` creates
multiple rows; downstream consumers (`/rebalance`, `PortfolioManager`)
dedup on read.

```mermaid
sequenceDiagram
    actor User
    participant Bot as add_portfolio_command
    participant V as Validation
    participant DB as DuckDBEngine

    User->>Bot: /add VNE 1000 32.5
    Bot->>Bot: _extract_user_id (None → reject, needs 1-1 chat)
    Bot->>V: argc==3? _TICKER_RE? int(vol)? float(price)? >0?
    alt invalid
        V-->>User: ❌ syntax / range error
    else valid
        Bot->>DB: INSERT INTO portfolio (user_id,ticker,volume,price,added_at) VALUES (?,?,?,?,?)
        DB-->>Bot: ok
        Bot->>DB: _audit_log_async(add, ticker, "vol:…,price:…")
        Bot-->>User: ✅ Đã thêm vào danh mục
    end
```

## `/remove <ticker>`

Count-then-delete so the user gets a meaningful "not found" vs.
"deleted N rows" reply. Deletes **all** lots of that ticker for the user.

```mermaid
sequenceDiagram
    actor User
    participant Bot as remove_portfolio_command
    participant DB as DuckDBEngine

    User->>Bot: /remove VNE
    Bot->>Bot: _extract_user_id + _TICKER_RE
    Bot->>DB: SELECT COUNT(*) FROM portfolio WHERE user_id=? AND ticker=?
    DB-->>Bot: existing_count
    alt count == 0
        Bot-->>User: ⚠️ Không tìm thấy trong danh mục
    else count > 0
        Bot->>DB: DELETE FROM portfolio WHERE user_id=? AND ticker=?
        Bot->>DB: _audit_log_async(remove, ticker, "rows_deleted:N")
        Bot-->>User: ✅ Đã xóa (N dòng)
    end
```

## `/verify <ticker>` — ad-hoc single-ticker analysis

`/verify` is **read-only** w.r.t. `portfolio` — it does **not** touch the
table. It is a rumor/news fact-check: run 5d + 20d Stacking GBDT plus the
LLM arbitrator on one symbol and return a combined verdict.

```mermaid
sequenceDiagram
    actor User
    participant Bot as verify_command
    participant VS as main.verify_single_ticker
    participant A360 as build_live_features(tickers=[T])
    participant ML as predict_stacking_horizon (5d & 20d)
    participant ARB as evaluate_trades_batch([T])
    participant RPT as _build_verify_report

    User->>Bot: /verify HPG
    Bot->>Bot: validate ticker
    Bot-->>User: ⏳ wait message
    Bot->>VS: await to_thread(verify_single_ticker, "HPG")
    VS->>A360: single-parquet read for HPG
    alt no parquet / empty
        A360-->>VS: FileNotFoundError
        VS-->>User: ⚠️ không đủ thanh khoản / chưa crawl
    else ok
        A360-->>VS: latest_df (1 row)
        VS->>ML: predict 5d, predict 20d
        ML-->>VS: prob vectors
        VS->>ARB: news scrape + Gemini + decision (scoped to HPG)
        ARB-->>VS: decision, sentiment, source_urls
        VS->>RPT: 5d dist + 20d + sentiment + verdict + sources
        RPT-->>VS: report_html
        VS-->>Bot: report_html
        Bot-->>User: 🔍 [KIỂM ĐỊNH] HPG
    end
```

## Cross-cutting

- **Audit:** every mutating command writes to `audit_log` via
  `_audit_log_async` (best-effort, never blocks the user reply).
- **Group chats:** `_extract_user_id` returns `None` → `/add` / `/remove`
  refuse (require 1-1 DM) so portfolios can't be polluted by a shared id.
- **`PortfolioManager` (cron)** shares the same table with
  `user_id='cron'`; it additionally enforces stop-loss
  (`-7%`) / take-profit (`+15%`) — logic that the bot CRUD path does
  **not** apply.
