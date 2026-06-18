# Sentiment-Entry Forward Paper-Log — Implementation Plan

**Date:** 2026-06-16
**Status:** CODE-COMPLETE — KEEP IN ACTIVE (pending production `source='daily'` confirmation)
**Classification:** SIMPLE (single-phase, single session, multi-file)
**Goal:** Forward-log every arbitrated candidate's prediction + sentiment + price each pipeline run so the "DOWN model / high positive sentiment" hypothesis can be analysed retrospectively over T+3 and T+20 without any backtest or live Gemini scrape expansion.

---

## Closeout Note (2026-06-18)

**Shipped:** commits 442cb09 / 8de1acc / f9bb05e — suite 238/238 green.

**What is implemented and verified:**
- New `sentiment_entry_paperlog` DuckDB table + `seq_sentiment_entry_id` sequence (DDL in `src/data/db_engine.py`).
- `_log_sentiment_entry_paperlog()` capture helper + `_backfill_paperlog_outcomes()` backfill helper in `main.py`.
- Daily pipeline wiring inside `run_trade_execution` (source='daily').
- `/verify` bot command wiring inside `verify_single_ticker` (source='verify') — confirmed end-to-end.
- Config knobs: `sentiment_entry_enabled` (default True) / `sentiment_entry_threshold` (default 0.7) in `TradingConfig`.
- Standalone analysis script: `scripts/analyze_sentiment_paperlog.py`.
- 10-test suite: `tests/test_sentiment_paperlog.py`.

**One deviation during EXECUTE:** `DuckDBEngine` is lazy-imported inside the `/verify` try-block (NameError fix during implementation). Intent from the plan is preserved; only the import location changed.

**Unverified / blocking archival:**
- A `source='daily'` row has NOT yet been observed in production. This requires one live `python main.py` run (cron at 15:30 ICT Mon–Fri). `source='verify'` was proven end-to-end.
- **Archive to `completed/` ONLY AFTER** a `source='daily'` row is confirmed in `SELECT COUNT(*) FROM sentiment_entry_paperlog WHERE source='daily'`.

**Known cleanup debt (do NOT fix here — tracked in Telegram-work files):**
- `src/reports/builders.py:326` hardcodes literal `5 ngày tới` instead of `{SHORT_HORIZON_DAYS}`. Needs a one-line fix when the Telegram-work effort is live.

---

## 1. Overview

### Problem

The hypothesis "names where the price model predicts DOWN but news sentiment is very positive (>0.7) — do they outperform?" cannot be tested against historical data because point-in-time sentiment archives do not exist.

### Solution

Log the full candidate cross-section on every daily pipeline run and on every `/verify` bot command invocation. Realized returns are backfilled after the T+3 / T+20 windows mature using the existing `price_lookup` module. The analysis filter (`DOWN & sentiment>0.7`) is applied **at analysis time**, not capture time — so the control group is always present.

### Key Invariants (locked, do not re-litigate)

1. Log the FULL candidate cross-section (not only treatment names).
2. Two capture sources: `'daily'` (pipeline) and `'verify'` (bot command). Rows are tagged via a `source` column.
3. No change to `make_final_decision`, trading decisions, or Gemini call count.
4. Zero look-ahead risk. Zero extra Gemini cost.

---

## 2. Touchpoints Table

Every file to change, with exact function / line-range:

| File | Change | Target Function / Line |
|---|---|---|
| `config/settings.py` | Add two fields to `TradingConfig` | `TradingConfig` dataclass, lines 61-74 (after `regime_sizing_enabled`) |
| `config/settings.json` | Add two matching keys under `"trading"` | `"trading"` object, line 45 area |
| `src/data/db_engine.py` | Add DDL for `sentiment_entry_paperlog` table + `seq_sentiment_entry_id` sequence | `_init_tables()`, lines 100-163 (add as item 10) |
| `main.py` | Add `_log_sentiment_entry_paperlog()` helper function (pure-ish: takes db + dicts, returns int) | New function, near `_log_rl_predictions` (~line 535) |
| `main.py` | Add `_backfill_paperlog_outcomes()` helper function | New function, near `_backfill_rl_outcomes` (~line 577) |
| `main.py` | Call `_log_sentiment_entry_paperlog` inside `run_trade_execution` | Inside `run_trade_execution`, after `_backfill_rl_outcomes` call (~line 1206), using already-resolved `db = manager.db` and `stacking_predictions` dict |
| `main.py` | Call `_backfill_paperlog_outcomes` inside `run_trade_execution` | Same block as above, one line after the new log call |
| `main.py` | Call `_log_sentiment_entry_paperlog` inside `verify_single_ticker` | Inside `verify_single_ticker`, after `evaluate_trades_batch` returns (Step 3, ~line 1455), before `_build_verify_report` |
| `scripts/analyze_sentiment_paperlog.py` | NEW standalone analysis script | New file, follows `cleanup_legacy_rl_stubs.py` structure |
| `tests/test_sentiment_paperlog.py` | NEW test file | New file, covers capture helper + backfill logic |

---

## 3. Blast Radius Assessment

### Files changed

- `config/settings.py` — additive only; `Config.from_json()` auto-maps via `**raw.get("trading", {})` (settings.py:124). No existing field modified.
- `config/settings.json` — additive only; same auto-mapping path.
- `src/data/db_engine.py` — `_init_tables()` is `CREATE ... IF NOT EXISTS`; idempotent by design. The new `CREATE SEQUENCE IF NOT EXISTS` pattern mirrors line 129. The new `CREATE TABLE IF NOT EXISTS` mirrors the block at lines 130-141. Safe to run against existing DB.
- `main.py` — two new pure helper functions + two new call sites inside existing timed blocks. No existing logic modified. The daily capture runs **inside `run_trade_execution`** where `db = manager.db` is already established, so no new DB connection is opened.
- `scripts/` — new file; no risk to existing scripts.
- `tests/` — new file; additive.

### What is NOT touched

- `src/backtest/` — backtest untouched.
- `src/bot/sizing.py` — no change.
- `src/trading/regime_policy.py` — no change.
- Feature pipeline (`src/backtest/pipeline.py`) — no change.
- `FEATURE_RECIPE_VERSION` — no change (no feature engineering).
- Any model artifact — no change.

### Potential failure modes

| Risk | Mitigation |
|---|---|
| `stacking_predictions_20d` is `{}` (secondary horizon missing) | Capture helper uses `.get(ticker)` on the 20d dict; stores NULL for 20d probs — explicit in schema |
| `sentiment_score` missing from `all_sentiments[ticker]` | `.get("sentiment_score")` → None → NULL in DB |
| `entry_close` missing (parquet shard absent for ticker) | `price_lookup.close_on_or_before` returns None → NULL in DB — does not crash |
| Double-log: daily_inference run twice same day | `UNIQUE(log_date, ticker, source)` constraint + `INSERT OR IGNORE` pattern |
| `/verify` path has no `manager.db` reference | The `/verify` path calls `verify_single_ticker` which does NOT call `run_trade_execution`. Use `DuckDBEngine()` singleton directly (safe — same process, same connection via singleton pattern). |
| Backfill runs before parquet shards are available | Same as existing `_backfill_rl_outcomes` — rows with missing shards stay NULL, retried next run |
| DuckDB config mismatch | Never open a side `duckdb.connect()` — always use `DuckDBEngine().conn` (db_engine.py:15-36 constraint) |

---

## 4. Proposed Schema

```sql
CREATE SEQUENCE IF NOT EXISTS seq_sentiment_entry_id START 1;

CREATE TABLE IF NOT EXISTS sentiment_entry_paperlog (
    id           INTEGER  DEFAULT nextval('seq_sentiment_entry_id'),
    log_date     DATE     NOT NULL,
    ticker       VARCHAR  NOT NULL,
    source       VARCHAR  NOT NULL,          -- 'daily' | 'verify'
    p_down_5d    DOUBLE,
    p_side_5d    DOUBLE,
    p_up_5d      DOUBLE,
    decision_5d  INTEGER,                    -- argmax(p_down, p_side, p_up): 0=DOWN,1=SIDE,2=UP
    p_down_20d   DOUBLE,                     -- NULL when 20d artifact missing
    p_side_20d   DOUBLE,
    p_up_20d     DOUBLE,
    final_decision INTEGER,                  -- evaluate_trades_batch output: 0|1|2
    sentiment_score DOUBLE,                  -- None when arbitrator unavailable
    entry_close  DOUBLE,                     -- price_lookup.close_on_or_before(ticker, log_date)
    ret_3d       DOUBLE,                     -- NULL until matured; filled by backfill
    ret_20d      DOUBLE,                     -- NULL until matured; filled by backfill
    outcome_filled BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (id),
    UNIQUE (log_date, ticker, source)        -- idempotency guard
);
```

**Notes:**
- `log_date` is a `DATE` (not `TIMESTAMP`) to match the `predicted_date DATE` convention in `rl_mistake_logs`.
- `decision_5d` = `int(argmax([p_down_5d, p_side_5d, p_up_5d]))` — 0=DOWN, 1=SIDE, 2=UP. Precomputed at capture time to avoid re-deriving at analysis time.
- `outcome_filled` is flipped to `TRUE` by the backfill function when both `ret_3d` and `ret_20d` are written.
- The `UNIQUE` constraint on `(log_date, ticker, source)` makes the INSERT idempotent via `INSERT OR IGNORE`.

---

## 5. Step-by-Step Implementation (ordered for independent testability)

### Step 1 — Config knobs

**File:** `config/settings.py`, `TradingConfig` dataclass (line 74, after `regime_sizing_enabled`)

Add:
```
sentiment_entry_enabled: bool = True
sentiment_entry_threshold: float = 0.7   # analysis-time reference only; not a capture gate
```

**File:** `config/settings.json`, `"trading"` object (after `"regime_sizing_enabled": true`)

Add:
```
"sentiment_entry_enabled": true,
"sentiment_entry_threshold": 0.7
```

**Verification:** `python -c "from config.settings import CONFIG; print(CONFIG.trading.sentiment_entry_enabled, CONFIG.trading.sentiment_entry_threshold)"`  → `True 0.7`

---

### Step 2 — DDL: table + sequence

**File:** `src/data/db_engine.py`, `_init_tables()` method

After the `_init_audit_log_table()` call at line 163, add a new call: `self._init_sentiment_paperlog_table()`.

Add a new private method `_init_sentiment_paperlog_table(self) -> None` on the `DuckDBEngine` class that executes:
1. `CREATE SEQUENCE IF NOT EXISTS seq_sentiment_entry_id START 1` (separate `.execute()` call — mirror line 129 pattern)
2. `CREATE TABLE IF NOT EXISTS sentiment_entry_paperlog (...)` (separate `.execute()` call — full schema from Section 4)

Both execute calls must be separate (DuckDB multi-statement `execute()` is version-sensitive — see db_engine.py:126-128 comment).

**Verification:** Start Python REPL, `from src.data.db_engine import DuckDBEngine; db = DuckDBEngine(); print(db.conn.execute("SELECT COUNT(*) FROM sentiment_entry_paperlog").fetchone())` → `(0,)`

---

### Step 3 — Capture helper function (daily + verify shared)

**File:** `main.py`

Add `_log_sentiment_entry_paperlog()` as a new module-level function immediately after `_log_rl_predictions` (~line 574).

**Signature:**
```python
def _log_sentiment_entry_paperlog(
    db: Any,
    candidate_tickers: list[str],
    stacking_5d: dict[str, list[float]],
    stacking_20d: dict[str, list[float]],
    final_decisions: dict[str, int],
    all_sentiments: dict[str, dict],
    source: str,
) -> int:
```

**Logic (per ticker in `candidate_tickers`):**
1. Resolve `probs_5d = stacking_5d.get(ticker)` — skip ticker if `None` or `len < 3`.
2. Resolve `probs_20d = stacking_20d.get(ticker)` — may be `None`; store NULL columns.
3. `p_down_5d, p_side_5d, p_up_5d = probs_5d[0], probs_5d[1], probs_5d[2]`
4. `decision_5d = int(probs_5d.index(max(probs_5d)))` — argmax.
5. `sent = all_sentiments.get(ticker, {})`, `sentiment_score = sent.get("sentiment_score")`.
6. `final_decision = final_decisions.get(ticker)`.
7. `log_date = datetime.now().strftime("%Y-%m-%d")`.
8. `entry_close = None` — left NULL at capture time; backfill resolves it. (Rationale: `price_lookup` in the verify path would need a shard read per-ticker at call time, adding latency. The backfill already reads T0 close for `ret_3d`/`ret_20d` anyway — we can resolve `entry_close` in the same pass.)
9. Execute `INSERT OR IGNORE INTO sentiment_entry_paperlog (...) VALUES (?, ...)` under `with db._audit_lock:` using `db.conn.execute(...)`. Mirror the lock pattern from `_log_rl_predictions` (main.py:564-572).
10. Return count of rows actually inserted (pre-query `SELECT COUNT(*)` WHERE `log_date=... AND ticker=... AND source=...` before INSERT, or just track insertions; simpler: just return `len(candidate_tickers)` and let the UNIQUE constraint silently suppress duplicates).

**Return value:** number of successful INSERT calls (ignoring suppressed duplicates via `OR IGNORE`).

**Note on `entry_close` at capture time vs. backfill time:** Deferring to backfill keeps the capture path free of per-ticker parquet I/O and avoids VN price-scale complexity at two call sites. The backfill step will fill `entry_close` in the same pass that computes `ret_3d` / `ret_20d`.

---

### Step 4 — Backfill helper function

**File:** `main.py`

Add `_backfill_paperlog_outcomes()` as a new module-level function immediately after `_backfill_rl_outcomes` (~line 634).

**Signature:**
```python
def _backfill_paperlog_outcomes(db: Any) -> int:
```

**Logic:**
1. Query pending rows: `SELECT id, ticker, log_date FROM sentiment_entry_paperlog WHERE outcome_filled = FALSE AND log_date <= CURRENT_DATE - INTERVAL 21 DAY`.
   - 21 calendar days guarantees ≥20 trading days have elapsed for all VN market calendars.
2. For each `(id, ticker, log_date)`:
   - `t0_close = price_lookup.close_on_or_before(ticker, log_date, conn=db.conn)` — also used as `entry_close`.
   - `t3_close = price_lookup.close_on_or_after(ticker, log_date + timedelta(days=3), conn=db.conn)`.
   - `t20_close = price_lookup.close_on_or_after(ticker, log_date + timedelta(days=20), conn=db.conn)`.
   - Skip row (leave NULL) if `t0_close is None or t0_close <= 0`.
   - `ret_3d = (t3_close - t0_close) / t0_close if t3_close is not None else None`
   - `ret_20d = (t20_close - t0_close) / t0_close if t20_close is not None else None`
   - **VN price-scale note:** `price_lookup.close_on_or_before/after` returns the raw parquet value (thousands of VND). Ret computations are pure ratios — the scale cancels out as long as T0 and TN are from the same parquet shard. Do NOT multiply by 1000 here. (Only `_get_live_exec_prices` needs the ×1000 scale for absolute VND display.)
   - Execute `UPDATE sentiment_entry_paperlog SET entry_close=?, ret_3d=?, ret_20d=?, outcome_filled=TRUE WHERE id=?` under `with db._audit_lock:`.
3. Increment `backfilled` counter per completed UPDATE.
4. Return `backfilled`.

---

### Step 5 — Daily pipeline wiring

**File:** `main.py`, `run_trade_execution()` function

Insertion point: directly after the `_backfill_rl_outcomes(db)` call on line ~1206, still inside the `with timed_step("RL prediction logging...")` block.

Add:
```python
if CONFIG.trading.sentiment_entry_enabled:
    from config.settings import CONFIG as _CFG  # already imported at module top
    _paperlog_count = _log_sentiment_entry_paperlog(
        db=db,
        candidate_tickers=list(stacking_predictions.get("5d", {}).keys()),
        stacking_5d=stacking_predictions.get("5d", {}),
        stacking_20d=stacking_predictions.get("20d", {}),
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        source="daily",
    )
    _paperlog_backfilled = _backfill_paperlog_outcomes(db)
    LOGGER.info(
        "Paperlog: logged=%s new rows, backfilled=%s matured rows",
        _paperlog_count, _paperlog_backfilled,
    )
```

**Why `stacking_predictions.get("5d", {}).keys()`:** Inside `run_trade_execution`, the full `stacking_predictions` dict (both horizons) is passed in from `daily_inference`. The 5d keys are the full cross-section of arbitrated candidates — exactly the control group we need. `top_buy_signals` would give only the Top-3, which is far too narrow.

**Important:** `CONFIG` is already imported at module top in `main.py`. Do not re-import. The `sentiment_entry_enabled` gate means a single `settings.json` key disables all paperlog I/O without a restart if needed.

---

### Step 6 — `/verify` path wiring

**File:** `main.py`, `verify_single_ticker()` function

Insertion point: after `final_decisions, all_sentiments = evaluate_trades_batch(...)` returns (Step 3 in the function, ~line 1455), before the call to `_build_verify_report` (~line 1468).

Add:
```python
if CONFIG.trading.sentiment_entry_enabled:
    try:
        _vdb = DuckDBEngine()   # singleton — safe, same process
        _log_sentiment_entry_paperlog(
            db=_vdb,
            candidate_tickers=[ticker],
            stacking_5d=stacking_5d,
            stacking_20d=stacking_20d,
            final_decisions=final_decisions,
            all_sentiments=all_sentiments,
            source="verify",
        )
        _backfill_paperlog_outcomes(_vdb)
    except Exception:   # noqa: BLE001
        LOGGER.warning("[/verify] Paperlog write failed for %s — non-fatal.", ticker)
```

**Why wrap in try/except:** `/verify` is a user-facing bot command. A paperlog write failure must never surface to the user. The `_build_verify_report` call proceeds regardless.

**Why `DuckDBEngine()` singleton here:** `verify_single_ticker` has no `db` reference (it does not call `PortfolioManager`). The DuckDB singleton is safe — it is the same connection the daily pipeline uses. Mirrors the `audit_log` write path used by `log_user_action`.

**`stacking_5d` / `stacking_20d` variable names in scope:** In `verify_single_ticker`, the local variables at lines 1425-1439 are named `stacking_5d` (from `predict_v3_horizon(latest_df, SHORT_HORIZON)`) and `stacking_20d` (from `predict_v3_horizon(latest_df, 20)`), both as plain dicts `{ticker: [p_down, p_side, p_up]}`. Pass these directly to the helper.

---

### Step 7 — Standalone analysis script

**File:** `scripts/analyze_sentiment_paperlog.py` (NEW)

Follow `scripts/cleanup_legacy_rl_stubs.py` structure exactly:
- Module docstring at top: explains purpose + `Run: python -X utf8 scripts/analyze_sentiment_paperlog.py`
- `_PROJECT_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(...)` pattern.
- Lazy-imports under `if __name__ == "__main__":` body.
- `def analyze() -> int:` function body:
  1. `DuckDBEngine()` to get `db`.
  2. Query: `SELECT * FROM sentiment_entry_paperlog WHERE outcome_filled = TRUE ORDER BY log_date`.
  3. Load into Pandas DataFrame via `db.conn.execute(...).df()`.
  4. Print total rows, filled rows, unfilled rows.
  5. Filter to treatment slice: `df[(df["decision_5d"] == 0) & (df["sentiment_score"] > threshold)]` where `threshold = CONFIG.trading.sentiment_entry_threshold`.
  6. Print mean / median `ret_3d` and `ret_20d` for treatment vs. control slices.
  7. Print source breakdown (`'daily'` vs `'verify'`) so self-selection bias in /verify is visible.
  8. Return exit code 0.
- Exit codes: 0 success, 1 DB error.

---

### Step 8 — Tests

**File:** `tests/test_sentiment_paperlog.py` (NEW)

#### Test Group A: capture helper (`_log_sentiment_entry_paperlog`)

Setup: in-memory `DuckDBEngine` stub (use `monkeypatch` to override `DuckDBEngine._instance` with a fresh in-memory connection, same pattern as existing tests that touch DuckDB). Apply the paperlog DDL to the in-memory DB before each test.

Tests:
1. `test_log_writes_full_crosssection` — pass 3 tickers, assert `SELECT COUNT(*)` = 3.
2. `test_log_idempotent_same_day` — log same 3 tickers twice, assert COUNT still = 3 (UNIQUE + OR IGNORE).
3. `test_log_20d_none_stores_null` — pass empty `stacking_20d = {}`, assert `p_up_20d IS NULL` for all rows.
4. `test_log_sentiment_score_none` — pass `all_sentiments = {}`, assert `sentiment_score IS NULL`.
5. `test_log_source_tagged_verify` — call with `source="verify"`, assert `source = 'verify'` in DB.
6. `test_log_skips_ticker_with_no_5d` — ticker absent from `stacking_5d`, assert it is not written.

#### Test Group B: backfill helper (`_backfill_paperlog_outcomes`)

Setup: insert synthetic rows with `log_date = (today - 25 days)`, `outcome_filled = FALSE`. Monkeypatch `price_lookup.close_on_or_before` and `price_lookup.close_on_or_after` to return deterministic float values.

Tests:
7. `test_backfill_computes_ret_3d_and_ret_20d` — insert 1 row, run backfill, assert `ret_3d ≈ (t3 - t0) / t0`, `ret_20d ≈ (t20 - t0) / t0`, `outcome_filled = TRUE`, `entry_close = t0`.
8. `test_backfill_skips_immature_rows` — insert 1 row with `log_date = today - 5 days`, run backfill, assert `outcome_filled` still `FALSE`.
9. `test_backfill_handles_missing_parquet` — monkeypatch `close_on_or_before` to return `None`, assert row stays `outcome_filled = FALSE` (no crash).
10. `test_backfill_returns_count` — insert 2 mature rows, run backfill, assert return value = 2.

---

## 6. Verification Evidence Checklist

Evidence that each step is complete:

| Step | Observable proof |
|---|---|
| Step 1 (config) | `python -c "from config.settings import CONFIG; print(CONFIG.trading.sentiment_entry_enabled, CONFIG.trading.sentiment_entry_threshold)"` → `True 0.7` |
| Step 2 (DDL) | `python -c "from src.data.db_engine import DuckDBEngine; db = DuckDBEngine(); print(db.conn.execute('SELECT table_name FROM information_schema.tables WHERE table_name=\'sentiment_entry_paperlog\'').fetchone())"` → non-None |
| Step 3+4 (helpers) | `pytest -q tests/test_sentiment_paperlog.py` → all 10 tests GREEN |
| Step 5 (daily wiring) | After one manual `python main.py` run (or a dry-run mock): `python -c "from src.data.db_engine import DuckDBEngine; db = DuckDBEngine(); print(db.conn.execute('SELECT COUNT(*) FROM sentiment_entry_paperlog WHERE source=\'daily\'').fetchone())"` → count > 0 |
| Step 6 (/verify wiring) | Invoke `/verify HPG` in bot; check `SELECT * FROM sentiment_entry_paperlog WHERE source='verify' AND ticker='HPG'` — row present |
| Step 7 (analysis script) | `python -X utf8 scripts/analyze_sentiment_paperlog.py` exits 0 (may report 0 filled rows until data matures) |
| Step 8 (tests) | `pytest -q tests/test_sentiment_paperlog.py` → 10/10 pass; `pytest -q` → ≥228 pass (no regressions) |

---

## 7. Test Commands (from all-tests.md convention)

```bash
# New tests only (fast, surgical)
pytest -q tests/test_sentiment_paperlog.py

# Full suite regression check
pytest -q

# Single test (surgical debug)
pytest -q tests/test_sentiment_paperlog.py::test_backfill_computes_ret_3d_and_ret_20d

# Verbose with stdout (setup debug)
pytest -s -v tests/test_sentiment_paperlog.py
```

---

## 8. Out of Scope

- No change to `make_final_decision` or any trading decision.
- No backtest modifications.
- No historical backfill of sentiment data (none exists).
- No expansion of Gemini call frequency or GNews query volume.
- No new model artifact, no `FEATURE_RECIPE_VERSION` change.
- The `sentiment_entry_threshold = 0.7` is stored in config as an analysis-time **reference constant** — it does NOT gate captures. All candidate rows are logged regardless.

---

## 9. Dependencies and Pre-conditions

- `price_lookup` module (`src/data/price_lookup.py`) must be importable in `main.py` scope. It already is (imported at module top in main.py — grep `price_lookup` confirms usage at ~line 614).
- `DuckDBEngine` singleton must be initialized before any paperlog write. In the daily path it is initialized by `PortfolioManager()` inside `run_trade_execution`. In the `/verify` path, `DuckDBEngine()` call inside the try-block initializes it if not already done.
- The `datetime` import in `main.py` is already present (used by `_log_rl_predictions`).
- The `timedelta` import in `main.py` is already present (used by `_backfill_rl_outcomes`).

---

## 10. Resume and Execution Handoff

**Plan file:** `process/general-plans/active/sentiment-entry-paperlog_PLAN_16-06-26.md`

**Execution order:** Steps 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each step is independently testable. Steps 3 and 4 can be written together since the backfill test requires the same in-memory DB setup as the capture test.

**Mid-implementation resume:** Check which functions exist in `main.py` (grep `_log_sentiment_entry_paperlog`, `_backfill_paperlog_outcomes`) and which tables exist in DuckDB (`information_schema.tables WHERE table_name='sentiment_entry_paperlog'`). Any step not yet confirmed by its verification evidence above should be the resume point.

**Gotcha reminders:**
- The `UNIQUE(log_date, ticker, source)` constraint requires `INSERT OR IGNORE`, not plain `INSERT`.
- In `run_trade_execution`, `final_decisions` is the outer-scope `final_decisions` param, NOT the `stacking_predictions` dict. Do not confuse the two.
- In `verify_single_ticker`, the 5d and 20d prob dicts are keyed by a single ticker only. The capture helper already handles a single-element list.
- VN price-scale: `ret_3d`/`ret_20d` are ratios; the ×1000 scale cancels — do NOT apply the scale factor inside `_backfill_paperlog_outcomes`.
- The `entry_close` column stores the parquet raw value (thousands VND) for reference; downstream analysis should be aware.
