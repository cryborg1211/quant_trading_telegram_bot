# PLAN — Audit the engine's actual recommendations (issue #3)

Feature: local-dashboard
Created: 2026-06-26
Status: ACTIVE (awaiting EXECUTE approval)
Shape: SIMPLE (read-only addition; one source file + tests)

## Problem

The post-mortem Audit tab (`run_post_mortem`) only grades tickers found in
`audit_log` — i.e. commands the **user** typed (`/verify`, `/add`). The
**engine's own buy recommendations** are never graded. Tab caption says
"Tỷ lệ đúng/sai của khuyến nghị" (win/loss of the recommendations), but the
recommendations themselves are invisible.

## Key finding (changes the approach)

The engine's dispatched picks ARE already persisted — in the
`dispatched_signals` ledger (`src/trading/signal_ledger.py`), written by the
cron broadcast dispatch (`main.py:1520`, `record_dispatch`). The ledger is:

- **Global** — no `user_id` column (system picks, namespace-free).
- Schema: `ticker, dispatch_date, horizon, hold_days, weight, status,
  closed_date, dispatched_at`.
- Written only on `broadcast=True` dispatch (cron/bot). Dashboard MUA preview
  is `persist=False` by design (P0 gotcha) → ephemeral, correctly NOT graded.

→ Therefore #3 needs **NO serve-path / dispatch change**. The picks exist; we
only need to READ + grade them. Risk drops from "touches degree-84 hub" to
"read-only report addition".

## Scope (touchpoints)

| File | Change |
|---|---|
| `src/utils/audit_evaluator.py` | + `_fetch_dispatched_signals`, + `_evaluate_dispatched_signal`, + engine-picks report section; append section in `run_post_mortem` |
| `tests/test_dashboard_audit_logging.py` | + ledger fetch / maturity / return-calc tests |

NOT touched: `main.py`, `_dispatch_signals`, `signal_ledger.py`, bot, dispatch
path. No new DB table. No config flag.

## Design

### Entry / exit pricing (per ledger row)
- **T0 (entry)** = `price_lookup.close_on_or_before(ticker, dispatch_date)`.
- **Exit**:
  - Matured (≥ `hold_days` trading sessions elapsed since `dispatch_date`,
    counted via `price_lookup.trading_dates_after`): exit = close on the
    `hold_days`-th trading session after dispatch. Mark `matured=True`.
  - Still open (< `hold_days` sessions): exit = `latest_close`, mark
    `matured=False` (provisional, labelled "đang giữ").
- **Return**: NET of `_VN_ROUND_TRIP_COST_PCT` (0.30%) — a tranche pick is a
  real buy+sell intent, so grade net (consistent with `_NET_PNL_COMMANDS`).
- Reuse session counting from `signal_ledger.list_open` semantics (do NOT
  import list_open — it only returns OPEN rows; we need CLOSED too). Compute
  elapsed inline with `trading_dates_after(dispatch_date)`.

### Report
- New `run_post_mortem` flow: keep existing user-command section, then append
  an **engine-picks** section from the ledger (window = same `days`).
- Section: header "🤖 TÍN HIỆU HỆ THỐNG ({days} NGÀY)", reuse
  `_summarize_hit_rate` for the win/loss line, then per-ticker:
  `Mã / dispatch_date / T+horizon / move_label (matured|đang giữ)`.
- Engine section is **always shown** when the ledger has rows in-window (the
  more meaningful "did the bot's calls work"). No Gemini call here (cost-free;
  the user-command section already carries the LLM explanations).
- Cap engine rows at `_MAX_TICKERS_PER_REPORT` (10), newest dispatch first.
- Empty ledger → omit the section (don't show an empty block).

### Defaults chosen (not asking — conventional)
- NET pricing for engine picks (real entry intent).
- Include still-open picks as provisional (labelled), not matured-only — gives
  the user signal sooner; matured flag disambiguates.
- Ledger read uses the existing `db.conn` (same DuckDB file the ledger writes).

## Verification

- `python -m pytest tests/test_dashboard_audit_logging.py -q` — new tests pass.
- `python -m pytest -q` — full suite green (currently 412).
- Manual: seed a `dispatched_signals` row (matured + open), call
  `run_post_mortem("local", 30)`, assert the engine section + hit-rate render.

## Out of scope (noted, not done)
- Bot `/suggest_buy` logs command with NULL ticker (`telegram_bot.py:638`).
  The ledger supersedes it for grading → leave as-is. (If per-command audit is
  ever wanted, that's a separate serve-path change.)
- Horizon-aligned grading of the USER-command section (#4b) — still T_now=today.
- Streamlit AppTest for the verify-run log path.

## Rollback
Single file (`audit_evaluator.py`) + test file. `git revert` of the EXECUTE
commit fully restores. No data migration, no schema change.
