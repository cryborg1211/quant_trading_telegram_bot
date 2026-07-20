# EOD Dual-Horizon Position/Verdict Report — Implementation Plan

**Date**: 16-07-26
**Complexity**: COMPLEX (standard complex — one execution stream, ~6 files, achievable in one EXECUTE session; NOT a phase program)
**Status**: 🔨 CODE DONE (all 6 phases implemented, full suite 703/703 green 16-07-26; stays ACTIVE — NOT ✅ VERIFIED — until the live 15:30 ICT cron gate in item 26 is observed)

## Overview

Add a daily EOD Telegram report that lists **every signal the user actually received** — both the
T+20 primary tranche BUY dispatch and a newly-introduced, independently-tracked T+5 paper position
for the same tickers — showing live NET PnL% while open ("còn N ngày kết thúc dự đoán") and a
correct/wrong verdict with final NET PnL% once each position's hold horizon elapses. Reuses the
existing `dispatched_signals` ledger (`src/trading/signal_ledger.py`), which is schema-ready for a
second horizon but has two latent horizon-blind bugs that MUST be fixed first (Architecture Decision
1). T+5 becomes its own independently tracked position (per INNOVATE decision), not a
display-only probability — confirmed user decision: "do it for T5 too" → "go".

**Context consulted**: process/context/all-context.md (repository routing/architecture) and process/context/tests/all-tests.md (pytest runner conventions — this plan follows the repo's existing in-memory DuckDB stub / monkeypatch test style throughout, see Test Matrix below).

---

## Quick Links

- [Phase Completion Rules](#phase-completion-rules)
- [Non-Goals and Constraints](#non-goals-and-constraints)
- [Architecture Decisions](#architecture-decisions-final)
- [Data Flow](#data-flow)
- [Touchpoints](#touchpoints)
- [Public Contracts](#public-contracts)
- [Blast Radius](#blast-radius)
- [Implementation Checklist](#implementation-checklist)
- [Test Matrix](#test-matrix)
- [Verification Evidence](#verification-evidence)
- [Open Questions](#open-questions-resolved-with-defaults---confirm-before-or-during-execute)
- [Acceptance Criteria](#acceptance-criteria)
- [Resume and Execution Handoff](#resume-and-execution-handoff)

---

## Phase Completion Rules

A phase is NOT complete until:

1. **Integration Test** — Works with other system pieces (ledger write → EOD read → report send)
2. **Manual Test** — Operator can trace a dispatched ticker through open → closed in the ledger
3. **Data Verification** — DuckDB `dispatched_signals` rows confirmed via direct query
4. **Error Handling** — Failure cases (missing price history, disabled config, ledger exception) degrade gracefully, never raise
5. **User Confirmation** — First live EOD cron run (15:30 ICT) confirms real dispatch + report send

Status meanings:
- ⏳ PLANNED — Not started
- 🔨 CODE DONE — Written but not E2E tested
- 🧪 TESTING — Currently being tested
- ✅ VERIFIED — Tested AND confirmed working (requires a live cron run — see Verification Evidence)
- 🚧 BLOCKED — Has issues

---

## Execution Brief

### Phase 1: `signal_ledger.py` correctness fixes + new read/eval functions
**What happens:** Fix two horizon-blind bugs (`record_dispatch` dedup, `mark_closed` WHERE clause)
that would silently corrupt data the moment a second horizon is written to the same table. Add
`evaluate_signal_pnl()` (NET PnL calculator) and `list_closed_on()` (today's closures).
**Test:** `pytest tests/test_signal_ledger.py -q` — new regression tests for the two bugs + unit
tests for the two new functions.

### Phase 2: `main.py` — T+5 tracking dispatch wiring
**What happens:** After the existing T+20 `record_dispatch` call in `run_trade_execution`, add a
second call that books an independent, fixed `hold_days=5` tracking row for the same dispatched
tickers, using a synthetic strategy dict (NOT the T+5 artifact's own `.strategy`, which reflects the
30-session tranche book, not the 5-day label horizon — see Decision 3).
**Test:** New `tests/test_short_horizon_tracking.py` asserting exactly 2 `record_dispatch` calls per
broadcast dispatch (horizon=20, horizon=5) and 0 when `broadcast=False`.

### Phase 3: `src/reports/builders.py` — pure report formatter
**What happens:** `build_position_report(open_rows, closed_rows, today)` — pure HTML string builder
(no DB/price I/O, matching the module's existing contract), rendering the user's open/closed line
formats with char-budget truncation.
**Test:** New `tests/test_position_report_builder.py` — exact string-format assertions, verdict
tie-break, section omission, truncation safety.

### Phase 4: `main.py` orchestration + config + EOD cron wiring
**What happens:** `notify_position_report()` never-raise wrapper (mirrors `notify_portfolio_guard`'s
shape exactly), gated by new `CONFIG.trading.eod_position_report_enabled` (default True), called from
`full_pipeline()` and `inference_only()` immediately AFTER `notify_tranche_exits()`.
**Test:** New `tests/test_position_report_notify.py` mirroring `tests/test_portfolio_guard.py`'s
"Part E" orchestration-test pattern.

### Phase 5: Hygiene — horizon label on the two pre-existing horizon-blind lines
**What happens:** `/exits` (`_build_exits_report`) and `notify_tranche_exits`' per-row exit-due line
both currently render one line per OPEN/DUE row with no horizon tag. Once T5 rows share the table,
both will silently interleave T5 "paper tracking" rows with T20 "real tranche position" rows using
wording that implies a real weighted NAV position. Add a `[T5]`/`[T20]` tag to both (data already
present in the row dicts — zero new plumbing).
**Test:** Extend the existing `tests/test_signal_ledger.py::TestListOpen::test_exits_report_formatting`
assertion.

### Phase 6: Full-suite verification
**What happens:** Run the complete test suite; confirm zero regressions.
**Test:** `pytest -q` — all green (679+ baseline, growing by ~15-20 new tests).

### Expected Outcome
- Every T+20 BUY signal dispatched to the user now has a paired, independently-tracked T+5 paper
  position in the same ledger.
- One new EOD broadcast message per day (both ADMIN + USER chats, via `send_text_alert`) listing
  every open position's live NET PnL% + sessions remaining, and every position closed that day with
  a đúng/sai verdict + final NET PnL%.
- Zero changes to real trading/sizing logic — T5 tracking rows never carry real NAV weight and never
  feed `PortfolioManager`.
- `/exits` and the tranche exit-due alert gain a horizon tag so mixed T5/T20 rows are unambiguous.

---

## Non-Goals and Constraints

**Out of scope for this plan:**
- No on-demand `/positions` command (the new `notify_position_report()`/`build_position_report()`
  functions are reusable for one later, but wiring a slash command is future work).
- No change to `src/utils/audit_evaluator.py` (its `_evaluate_dispatched_signal` duplicate PnL logic
  is intentionally left alone — see Decision 4). It will automatically start showing T5 rows in
  `/audit_weekly`'s engine-picks section with zero code change, as an accepted side effect.
- No retroactive backfill — T5 tracking rows only start accruing from the ship date forward; no
  synthetic history is created for past T20-only dispatches.
- No dashboard (`dashboard/`) changes.
- No change to `PortfolioManager.process_daily_trades` / real position sizing — T5 rows are
  paper-tracking only, weight is always 0.0.
- No settings.json edit required (Python dataclass default is sufficient; settings.json remains an
  optional override surface per existing convention).

**Constraints:**
- Must never break the existing single-horizon (T20-only) behavior of `record_dispatch`, `list_open`,
  `check_exits_due`, `mark_closed` — all existing `tests/test_signal_ledger.py` tests must stay green
  unmodified except the one line noted in Phase 5.
- Must follow the established never-raise EOD-alert-only pattern (`notify_tranche_exits`,
  `notify_portfolio_guard`): zero writes beyond the ledger's own `record_dispatch`/`mark_closed`
  (already existing), alert-only, event-gated (skip send when nothing to report), Telegram
  4096-char-safe by construction.

---

## Architecture Decisions (Final)

### Decision 1 — Fix two horizon-blind bugs in `signal_ledger.py` BEFORE adding a second horizon (REQUIRED, not optional)

**Finding (confirmed by reading the file):**
- `record_dispatch`'s idempotency dedup (`src/trading/signal_ledger.py` ~lines 88-97) checks
  `existing = {ticker for row in SELECT ticker FROM dispatched_signals WHERE dispatch_date = ?}` —
  **ticker-only**, no horizon. Once a T20 row exists for `(VNM, today)`, a later `record_dispatch`
  call for a T5 row on the same `(VNM, today)` would be silently skipped (dedup false-positive) —
  the T5 tracking row would simply never be written.
- `mark_closed` (~lines 174-179) does `UPDATE ... WHERE ticker = ? AND dispatch_date = ? AND status =
  'OPEN'` — **no horizon filter**. If `(VNM, today)` has both a T5 row (due, hold_days=5) and a T20
  row (not due, hold_days=30) open simultaneously, closing the due T5 row would **also incorrectly
  close the still-open T20 row** — a real correctness bug that would silently truncate a live tranche
  position's tracked hold period.

**Decision:** Fix both before any dual-horizon write happens.
- `record_dispatch` dedup key → `(ticker, horizon)`.
- `mark_closed` WHERE clause → add `AND horizon IS NOT DISTINCT FROM ?` (DuckDB/Postgres NULL-safe
  equality; `list_open()`'s row dicts already carry `"horizon"`, so `due` entries have it available).

**Implication:** This is a genuine regression-risk bug fix to ship-independently-worthy code, not
speculative — it is currently *latent* (never triggered because only one horizon has ever been
written) and becomes *live* the moment Phase 2 ships. Both must land in the same PR/commit sequence,
Phase 1 before Phase 2.

### Decision 2 — Reuse the existing `dispatched_signals` table; do NOT create a second table

**Rationale:** The table is already horizon-agnostic by design (`horizon INTEGER`, `hold_days
INTEGER` columns, generic session-elapsed math in `list_open`/`check_exits_due`, already
horizon-aware display in `audit_evaluator._build_engine_section`). Creating a parallel table would
duplicate the entire session-elapsed/maturity logic for no benefit (YAGNI/DRY).

**Rejected alternative:** A separate `signal_tracking` table decoupled from the "real" tranche book,
to avoid conflating "real weighted positions" with "paper-tracked prediction grading." Rejected
because (a) the whole system is still paper/pre-DSR per project memory — `notify_tranche_exits`'
"exit ATC" language is itself simulated, not literal broker execution, so the semantic gap is smaller
than it first appears; (b) it would duplicate ~40 lines of proven, tested session-math logic; (c) the
one real UX cost (mixed wording in `/exits` and the exit-due alert) is cheaply fixed by Decision 6 /
Phase 5 instead.

### Decision 3 — T5 tracking row uses a SYNTHETIC `{"mode": "tranche", "hold_days": SHORT_HORIZON}` strategy dict, NOT `_load_v3_bot(SHORT_HORIZON).strategy`

**Finding:** Per project memory ("T+5 retrain... resume: `train_models.py --tb-horizon 5` →
`run_backtest.py --mode tranche --hold-days 30`"), the T+5 GOLDEN artifact is *also* backtested with
the tranche book's `--hold-days 30` convention for cost-model realism. Its `.strategy["hold_days"]`
therefore reflects the **30-session portfolio-construction window**, NOT the model's own 5-trading-day
label horizon. Using `_load_v3_bot(5).strategy` for the tracking row's `hold_days` would silently
track "T+5 model" predictions for 30 sessions — defeating the entire point of an independent
5-session verification window.

**Decision:** At the dispatch call site in `run_trade_execution`, construct a synthetic dict
`{"mode": "tranche", "hold_days": SHORT_HORIZON}` (SHORT_HORIZON = 5, module-level constant already
in `main.py`) purely to satisfy `record_dispatch`'s `mode == "tranche"` gate and supply the correct
`hold_days`. This does not touch or depend on any artifact file. Document this rationale as an inline
code comment (see Implementation Checklist item 7) so a future reader doesn't "fix" it to use the
loaded artifact's real strategy.

### Decision 4 — Do NOT refactor `src/utils/audit_evaluator.py`; accept intentional logic duplication with the new `signal_ledger.evaluate_signal_pnl`

**Finding:** `audit_evaluator._evaluate_dispatched_signal` (lines 352-390) already computes
NET-of-round-trip-cost PnL% for one ledger row using the exact same open/matured logic this plan
needs, and `_build_engine_section` already renders "T+{horizon}" labels per row — it is *already*
horizon-agnostic and will automatically start showing T5 rows in `/audit_weekly` with **zero code
change** once Phase 2 ships.

**Decision:** Add a fresh `evaluate_signal_pnl()` to `signal_ledger.py` for the new report builder to
use, rather than importing `audit_evaluator`'s private function (which is coupled to a `DuckDBEngine`
wrapper's `.conn`, a different calling convention than `signal_ledger.py`'s own `price_lookup`-default-
connection style) or refactoring `audit_evaluator.py` to delegate to it.

**Rejected alternative:** Refactor `_evaluate_dispatched_signal` to delegate to the new
`signal_ledger.evaluate_signal_pnl`, eliminating duplication. Rejected for THIS plan to minimize blast
radius on an already-shipped, working `/audit_weekly` code path with no dedicated test file found for
that function (`tests/test_dashboard_audit_logging.py` does not cover it) — the risk/reward doesn't
justify it in the same change as the ledger bug fixes + new dispatch wiring. **Flag as a candidate
backlog cleanup item**, not part of this plan's Acceptance Criteria.

### Decision 5 — T5 tracking mirrors the EXACT T20-dispatched ticker list; no independent T5 candidate selection

**Rationale:** The report's stated goal is "every signal the user actually received." The user only
ever receives BUY cards for the T20-selected Top-3 (`top_buy_signals`); the T+5 model's own
probability for those same names is already computed at zero extra inference cost
(`horizon_predictions`/`stacking_predictions_20d` inside `daily_inference` — see the naming caveat in
Data Flow below) but is not currently surfaced as an independent trackable call. Tracking the T+5
model's verdict on the SAME 3 tickers the user actually saw is the most direct, lowest-risk
interpretation of "track T5 too," and requires zero new candidate-selection/threshold logic.

**Explicitly flagged as a decision point** (see Open Questions #3) in case the intended behavior was
instead "T5 has its OWN independent Top-3 selection, potentially different tickers than T20." Default
in this plan: mirrored ticker list.

### Decision 6 — Fix `_build_exits_report` and `notify_tranche_exits`' per-row line to show a horizon tag (Phase 5)

**Rationale:** Both currently render ticker + NAV weight% + sessions with tranche-position wording
("vị thế tranche", "thoát ATC hôm nay"). Once T5 tracking rows share the table, both will silently
interleave paper-tracking rows using wording that implies a real weighted position, with no way to
tell which is which. Both row dicts already carry `"horizon"` (`list_open()`'s output) — this is a
1-line format-string addition per site, zero new plumbing, directly caused by this feature's blast
radius (not scope creep). Included as an in-scope, decided (not optional) Phase 5 step.

### Decision 7 — T5 tracking rows always write `weight = 0.0` (never reuse the T20 book's `suggested_weight`)

**Rationale:** `record_dispatch` reads `weight` from each signal dict's `suggested_weight` key. The
T20-dispatched `dispatched_signals` list carries the REAL tranche book weight — reusing those same
dicts for the T5 call would store T20's real NAV weight against a T5 tracking row, which is
misleading (T5 never drives sizing, per existing code comment: "the short model never drives
sizing"). The T5 call therefore builds fresh minimal dicts `[{"ticker": s["ticker"]} for s in
dispatched_signals]` (no `suggested_weight` key → `record_dispatch` defaults it to `0.0`).

### Decision 8 — `evaluate_signal_pnl` / `build_position_report` split: I/O vs pure formatting

**Rationale:** `src/reports/builders.py`'s own module docstring states its functions are "pure
HTML-string builders with no orchestration logic." `signal_ledger.evaluate_signal_pnl` does the
price-lookup I/O (matching `signal_ledger.py`'s existing convention of self-contained
`price_lookup.*` calls, no `db_path`/conn threading needed since price data lives in parquet, not the
ledger's own DuckDB file). `main.notify_position_report()` (the orchestration layer) calls
`list_open()` + `list_closed_on()` + `evaluate_signal_pnl()` per row to build fully-enriched plain
dicts, THEN passes them to `build_position_report()`, which does zero I/O and is trivially unit-
testable with plain fixture dicts (no monkeypatching required for builder tests).

---

## Data Flow

```
run_trade_execution (main.py)
  └─ _dispatch_signals(...) → dispatched_signals: list[dict]  (T20 BUY cards, real weight)
  └─ if broadcast and dispatched_signals:
        signal_ledger.record_dispatch(dispatched_signals, _strategy, horizon=20)     # existing
        signal_ledger.record_dispatch(short_signals,                                 # NEW
            {"mode":"tranche","hold_days":SHORT_HORIZON}, horizon=SHORT_HORIZON)
                    │
                    ▼
        dispatched_signals table (DuckDB) — 2 rows/ticker/day going forward
                    │
   full_pipeline() / inference_only()
      1. daily_inference()
      2. notify_tranche_exits()      → check_exits_due() → mark_closed() [FIXED: horizon-scoped]
      3. notify_position_report()    → NEW
                    │
                    ├─ signal_ledger.list_open(today)         → OPEN rows (both horizons)
                    ├─ signal_ledger.list_closed_on(today)     → rows closed THIS run
                    ├─ signal_ledger.evaluate_signal_pnl(...)  → NET PnL% + matured flag, per row
                    ▼
        build_position_report(open_rows, closed_rows, today)  → HTML string (pure)
                    ▼
        TelegramBot().send_text_alert(msg, label="position_report")  → broadcast to all chat IDs
```

**Naming landmine (context only, no code dependency for this feature):** Inside `daily_inference`,
when called with its default `horizon=20` (the only way it is ever invoked in the cron paths),
`horizon_predictions = {"5d": <T+20 model output>, "20d": <T+5 model output>}` — the dict keys are
inverted relative to their actual numeric horizons (see the existing code comment at main.py
~line 1137-1141: "Variable names `_5d`/`_20d` are kept for back-compat... they now mean
'primary'/'secondary'"). This plan's ledger writes need **zero probability data** at write time (see
Decision 5), so this landmine does not affect any code this plan touches — noted here purely so a
future maintainer extending this feature (e.g. to show P(UP) in the report) doesn't get bitten.

---

## Touchpoints

| File | Change type | Why |
|---|---|---|
| `src/trading/signal_ledger.py` | Fix 2 bugs + add 2 functions | Dedup/close correctness prerequisite; new PnL/read functions |
| `main.py` | Add T5 dispatch call + `notify_position_report()` + cron wiring | Orchestration |
| `src/reports/builders.py` | Add `build_position_report()` | Pure formatter |
| `config/settings.py` | Add `eod_position_report_enabled: bool = True` | Kill-switch, repo convention |
| `src/utils/telegram_bot.py` | Add horizon tag to `_build_exits_report` | Hygiene (Decision 6) |
| `tests/test_signal_ledger.py` | Extend | Regression + unit coverage |
| `tests/test_short_horizon_tracking.py` | New | Phase 2 dispatch wiring |
| `tests/test_position_report_builder.py` | New | Phase 3 pure formatter |
| `tests/test_position_report_notify.py` | New | Phase 4 orchestration wrapper |

---

## Public Contracts

**Must remain compatible:**
- `dispatched_signals` table schema — unchanged (no `ALTER TABLE`); only SQL predicate logic inside
  `record_dispatch`/`mark_closed` is corrected.
- `signal_ledger.record_dispatch(signals, strategy, horizon, db_path=None, today=None)` — signature
  unchanged; behavior for a single-horizon caller (e.g. any existing T20-only test) is unchanged
  except that the dedup key is now `(ticker, horizon)` instead of `ticker` — for single-horizon
  callers this is behaviorally identical (horizon is constant across calls for a given caller).
- `signal_ledger.list_open()` / `check_exits_due()` — unchanged signatures and return shape.
- `/exits` command and `notify_tranche_exits()` broadcast — same trigger conditions, same overall
  shape; per-row line text gains a `[T5]`/`[T20]` suffix (Decision 6) — a minor, intentional textual
  change to an existing live Telegram message.
- `/audit_weekly`'s engine-picks section (`_build_engine_section`) — code unchanged; row COUNT will
  grow (T5 rows now populate the same query) — accepted, not a contract break.

**New public surface introduced:**
- `signal_ledger.evaluate_signal_pnl(ticker, dispatch_date, hold_days, today=None) -> dict`
- `signal_ledger.list_closed_on(today, db_path=None) -> list[dict]`
- `src/reports/builders.build_position_report(open_rows, closed_rows, today) -> str`
- `main.notify_position_report() -> int`
- `config.settings.TradingConfig.eod_position_report_enabled: bool` (default `True`)

---

## Blast Radius

**Directly touched:** `src/trading/signal_ledger.py`, `main.py`, `src/reports/builders.py`,
`config/settings.py`, `src/utils/telegram_bot.py`, plus the 4 test files listed in Touchpoints.

**Indirectly affected (no code change, behavior grows):**
- `/exits` command output — now may show T5 rows alongside T20 rows (with the new tag).
- `notify_tranche_exits()` EOD broadcast — same.
- `/audit_weekly`'s "TÍN HIỆU HỆ THỐNG" engine-picks section — row count roughly doubles going
  forward.
- `dispatched_signals` table row volume — roughly doubles per dispatch day (1 T20 row + 1 T5 row per
  ticker instead of 1).

**Explicitly NOT touched:**
- `src/utils/audit_evaluator.py` (Decision 4).
- `PortfolioManager` / `process_daily_trades` / any real sizing or NAV logic.
- `dashboard/` (no dashboard surface reads `dispatched_signals` per current context; unaffected).
- Any bot slash-command registration (`build_application` in `telegram_bot.py`) — no new command.

---

## Implementation Checklist

### Phase 1 — `src/trading/signal_ledger.py` fixes + new functions

1. Fix `record_dispatch`'s idempotency dedup (~lines 88-97) to key on `(ticker, horizon)`:
   `SELECT ticker, horizon FROM dispatched_signals WHERE dispatch_date = ?`, build a
   `{(ticker, horizon), ...}` set, filter `rows` by `(r[0], r[2]) not in existing`.
2. Fix `mark_closed`'s UPDATE (~lines 174-179) to add `AND horizon IS NOT DISTINCT FROM ?`, passing
   `d["horizon"]` as the new bound parameter (already present on every `due` dict from `list_open()`).
3. Add module constant `_VN_ROUND_TRIP_COST_PCT: float = 0.30` (same value as
   `audit_evaluator._VN_ROUND_TRIP_COST_PCT`; intentionally duplicated per Decision 4 — add a comment
   noting the sibling constant).
4. Add `evaluate_signal_pnl(ticker: str, dispatch_date: date, hold_days: int, today: date | None =
   None) -> dict`: mirrors `audit_evaluator._evaluate_dispatched_signal`'s open/matured branching
   (t0 = `price_lookup.close_on_or_before(ticker, dispatch_date)`; if `hold_days` trading sessions
   have elapsed per `price_lookup.trading_dates_after(dispatch_date)`, exit price = close on the
   `hold_days`-th session (matured=True), else exit price = `price_lookup.latest_close(ticker)`
   (matured=False, provisional)); `pct = (t_exit - t0) / t0 * 100.0 - _VN_ROUND_TRIP_COST_PCT`. Return
   `{"ticker", "pct", "matured", "t0", "t_exit"}` on success, `{"ticker", "error": <msg>}` on missing
   price history / `t0 <= 0`. Never raises (wrap in try/except, log + return error dict). No
   `db_path`/conn parameter — price lookups use their own default connection, matching this module's
   existing convention.
5. Add `list_closed_on(today: date, db_path: str | None = None) -> list[dict]`: query
   `SELECT ticker, dispatch_date, horizon, hold_days, weight FROM dispatched_signals WHERE status =
   'CLOSED' AND closed_date = ? ORDER BY dispatch_date, ticker`, return the same dict shape as
   `list_open()` rows (minus `sessions_elapsed`/`sessions_remaining`, which are meaningless for a
   closed row).
6. Write tests in `tests/test_signal_ledger.py`:
   - `TestDualHorizonDispatch.test_t5_and_t20_rows_same_ticker_day_both_persist` — two
     `record_dispatch` calls (horizon=20/hold=30, horizon=5/hold=5) for the same ticker+day both
     insert (2 total rows).
   - `TestDualHorizonDispatch.test_second_call_same_horizon_still_idempotent` — calling
     `record_dispatch` twice with the SAME horizon is still a no-op on the second call (regression
     guard for the dedup-key change).
   - `TestDualHorizonDispatch.test_mark_closed_only_closes_matching_horizon` — **critical regression
     test**: T20 row (hold=30) + T5 row (hold=5) same ticker/day; after enough sessions elapse for T5
     only, `check_exits_due` returns only the T5 row, `mark_closed` on it leaves the T20 row `status
     == 'OPEN'` (direct DB assertion).
   - `TestEvaluateSignalPnl` (monkeypatch `signal_ledger.price_lookup`): `test_open_provisional`
     (elapsed < hold_days → matured=False, uses latest_close), `test_matured_exit_price` (elapsed >=
     hold_days → matured=True, uses the hold_days-th session's close), `test_missing_price_returns_
     error`, `test_t0_zero_or_negative_returns_error`.
   - `TestListClosedOn.test_returns_only_todays_closures`, `test_excludes_open_rows`,
     `test_excludes_other_dates`.
7. Run `pytest tests/test_signal_ledger.py -q` — all green before proceeding to Phase 2.

### Phase 2 — `main.py` T+5 tracking dispatch wiring

8. In `run_trade_execution` (~main.py line 1723-1724), immediately after the existing
   `signal_ledger.record_dispatch(dispatched_signals, _strategy, int(horizon))` call, add:
   - Guard `if int(horizon) != SHORT_HORIZON:` (defensive — see Decision 3's rationale on why this
     guard exists even though it is not exercised by any current call path).
   - `_short_signals = [{"ticker": s["ticker"]} for s in dispatched_signals]` (Decision 7 — no
     `suggested_weight` key, defaults to 0.0).
   - `signal_ledger.record_dispatch(_short_signals, {"mode": "tranche", "hold_days": SHORT_HORIZON},
     horizon=SHORT_HORIZON)`.
   - Inline comment explaining WHY a synthetic strategy dict is used instead of
     `_load_v3_bot(SHORT_HORIZON).strategy` (Decision 3, verbatim rationale).
9. Create `tests/test_short_horizon_tracking.py` (new file, mirrors
   `tests/test_dispatch_regime_sizing.py`'s monkeypatch style): monkeypatch `signal_ledger.
   record_dispatch` to a capturing stub; call `main.run_trade_execution(...)` with `broadcast=True` and
   assert exactly 2 calls captured — one with `horizon=20` and the T20 book's real `dispatched_signals`
   dicts (weight present), one with `horizon=SHORT_HORIZON` and ticker-only dicts (no weight key).
   Assert 0 additional ledger calls when `broadcast=False`.
10. Run `pytest tests/test_short_horizon_tracking.py tests/test_dispatch_regime_sizing.py -q` — all
    green (confirms no regression to the existing regime-sizing dispatch tests).

### Phase 3 — `src/reports/builders.py` pure report formatter

11. Add constants: `_HORIZON_LABEL: dict[int, str] = {SHORT_HORIZON_DAYS: "T5", 20: "T20"}`,
    `_POSITION_REPORT_CHAR_BUDGET: int = 3800` (matches `notify_tranche_exits`' existing safe budget).
12. Add `build_position_report(open_rows: list[dict], closed_rows: list[dict], today: date) -> str`.
    Row dicts are pre-enriched by the caller with `evaluate_signal_pnl`'s output merged in (`pct`,
    `matured`, optional `error`) plus the ledger row's own fields (`ticker`, `dispatch_date`,
    `horizon`, `hold_days`, and for open rows `sessions_remaining`). Behavior:
    - Skip any row with `"error"` present (log responsibility stays with the caller;
      `build_position_report` itself does no logging — pure function).
    - **Open-row line** (sorted ascending by `sessions_remaining` — soonest-to-close first):
      `f"Dự báo ngày {dispatch_date:%d/%m/%Y}: Mô hình {hz_label} mã {ticker} hiện đang {verb}
      {abs(pct):.1f}% (còn {sessions_remaining} ngày kết thúc dự đoán)"` where
      `verb = "lãi" if pct >= 0 else "lỗ"`.
    - **Closed-row line** (order = input order, i.e. `list_closed_on`'s
      `ORDER BY dispatch_date, ticker`): `f"Mô hình {hz_label} — mã {ticker}: đã dự báo {verdict} và
      {verb} {abs(pct):.1f}%"` where `verdict = "đúng" if pct >= 0 else "sai"`,
      `verb = "lãi" if pct >= 0 else "lỗ"` (see Open Question #1 — this line intentionally includes
      the ticker, a deviation from the user's literal shorthand template, for usability when >1
      signal closes the same day).
    - Header + two labeled sections ("ĐANG MỞ" / "ĐÃ ĐÓNG HÔM NAY"); omit a section header entirely
      when its row list is empty; return `""` when BOTH lists are empty (post error-row filtering) so
      the caller knows to skip sending.
    - Truncation: build the full ordered line list (open lines first, then closed lines), apply the
      same char-budget accumulate-and-cut loop pattern as `notify_tranche_exits` (`main.py` ~lines
      2170-2186) against `_POSITION_REPORT_CHAR_BUDGET`, append one combined overflow notice
      (`"... và N tín hiệu khác (rút gọn)."`) when truncated.
13. Create `tests/test_position_report_builder.py`: exact open-line string match against a fixture
    row (incl. `.1f` rounding), exact closed-line string match, `pct == 0.0` tie-break → "đúng"/"lãi"
    branch, section omitted when its list is empty, `""` returned when both empty, rows with `"error"`
    silently excluded, truncation test with a synthetic 50-row fixture asserting total output length
    `<= 4096` and an overflow-notice substring present.
14. Run `pytest tests/test_position_report_builder.py -q` — all green.

### Phase 4 — `main.py` orchestration wrapper + config + EOD cron wiring

15. Add to `TradingConfig` in `config/settings.py` (near `portfolio_guard_enabled`):
    `eod_position_report_enabled: bool = True` with a kill-switch comment matching the existing style
    (`# Kill-switch: set "eod_position_report_enabled": false in settings.json + restart to disable.`).
16. Add `notify_position_report() -> int` to `main.py` (place near `notify_tranche_exits`/
    `notify_portfolio_guard`), mirroring `notify_portfolio_guard`'s exact never-raise shape:
    - Short-circuit `if not CONFIG.trading.eod_position_report_enabled: return 0` (zero DB reads).
    - `today = datetime.now().date()`.
    - `open_raw = signal_ledger.list_open(today=today)`; `closed_raw =
      signal_ledger.list_closed_on(today)`.
    - If both empty → log + `return 0` (no send).
    - Enrich each row via `signal_ledger.evaluate_signal_pnl(r["ticker"], r["dispatch_date"],
      r["hold_days"], today=today)`, merging into the row dict; log (not raise) any per-row error.
    - `msg = build_position_report(open_rows, closed_rows, today)`; if falsy, `return 0`.
    - `TelegramBot().send_text_alert(msg, label="position_report")`.
    - Wrap the whole body in try/except → log + `return 0` on any exception.
17. Wire `notify_position_report()` into `full_pipeline()` and `inference_only()`, called
    **immediately after** `notify_tranche_exits()` (so `list_closed_on(today)` observes the same-run
    closures), wrapped in `timed_step("EOD position report (signal ledger)")` matching the existing
    step style used for `notify_tranche_exits`/`notify_portfolio_guard`.
18. Update `full_pipeline`'s docstring numbered step list (currently 1-5) to append step 6: "EOD
    position report — dual-horizon PnL/verdict broadcast (signal ledger)."
19. Create `tests/test_position_report_notify.py` mirroring `tests/test_portfolio_guard.py`'s "Part
    E" pattern: `test_notify_disabled_no_db_read` (config False → `signal_ledger.list_open` never
    called), `test_notify_nothing_to_report_returns_zero` (empty open+closed → `TelegramBot` never
    constructed), `test_notify_sends_combined_report` (fake `TelegramBot` capturing
    `send_text_alert(msg, label="position_report")`), `test_notify_never_raises_on_ledger_exception`
    (monkeypatch `signal_ledger.list_open` to raise → `notify_position_report()` returns `0`, no
    exception propagates).
20. Run `pytest tests/test_position_report_notify.py -q` — all green.

### Phase 5 — Hygiene: horizon tag on the two pre-existing horizon-blind lines

21. `src/utils/telegram_bot.py::_build_exits_report` (~line 1131-1160): append a horizon tag to each
    per-row line, e.g. `f"[{_HORIZON_LABEL.get(r.get('horizon'), '?')}]"` inserted after the ticker
    (reuse or re-import the `_HORIZON_LABEL` mapping from `src/reports/builders.py`, or a local
    equivalent — avoid a new circular import; `telegram_bot.py` may import from `src/reports/builders`
    if it does not already create a cycle — verify at implementation time).
22. `main.py::notify_tranche_exits` (~line 2158-2169): add the same horizon tag to its per-row
    exit-due line.
23. Update `tests/test_signal_ledger.py::TestListOpen::test_exits_report_formatting` to assert the
    horizon tag (`"[T20]"`) appears in both the non-due and due message variants.
24. Run `pytest tests/test_signal_ledger.py -q` again — all green.

### Phase 6 — Full-suite verification

25. Run `pytest -q` (full suite) — zero regressions, all new tests green.
26. Manual/live confirmation gate (documented, not automated — mirrors the Portfolio Guard / Intraday
    Scanner precedent): first production EOD cron run (`full_pipeline`, 15:30 ICT) must be observed to
    (a) write both a T20 and a T5 ledger row per dispatched ticker, and (b) send exactly one
    `position_report`-labeled broadcast with correct content. **This plan stays ACTIVE (not archived)
    until that confirmation lands**, consistent with the repo's established pattern for EOD-broadcast
    features.

---

## Test Matrix

| Area | File | Type | Key scenarios |
|---|---|---|---|
| Ledger dedup/close fix | `tests/test_signal_ledger.py` | Unit | dual-horizon insert, idempotency, cross-horizon close isolation (critical) |
| `evaluate_signal_pnl` | `tests/test_signal_ledger.py` | Unit | open/provisional, matured/exit, missing price, invalid t0 |
| `list_closed_on` | `tests/test_signal_ledger.py` | Unit | today-only filter, excludes OPEN, excludes other dates |
| Dispatch wiring | `tests/test_short_horizon_tracking.py` | Unit (monkeypatch) | 2 calls on broadcast=True, 0 on broadcast=False, correct horizon/hold_days/weight per call |
| Report formatter | `tests/test_position_report_builder.py` | Unit (pure) | exact line format, verdict tie-break, section omission, truncation |
| Orchestration wrapper | `tests/test_position_report_notify.py` | Unit (monkeypatch) | disabled short-circuit, nothing-to-report, send path, never-raises |
| Hygiene | `tests/test_signal_ledger.py::test_exits_report_formatting` | Unit | horizon tag present |
| Regression | full `pytest -q` | Full suite | zero breakage across 679+ existing tests |
| Manual | live cron | E2E | first production run — required before plan archival |

---

## Verification Evidence

- All Phase 1-5 pytest files pass locally (`pytest -q` on the touched files individually, then the
  full suite in Phase 6) — this is the automated evidence bar.
- No live EOD cron run has occurred as of plan authoring — production confirmation is a **manual,
  required gate** (item 26). Do not mark this plan's phases ✅ VERIFIED (only 🔨 CODE DONE / 🧪
  TESTING) until that run is observed and its ledger + Telegram output manually inspected.
- Direct DB query to confirm dual-horizon rows after the first live run:
  `SELECT ticker, dispatch_date, horizon, hold_days, status FROM dispatched_signals ORDER BY
  dispatch_date DESC, ticker, horizon LIMIT 20;` — expect 2 rows per dispatched ticker (horizon 5 and
  20) for the ship date forward.

---

## Open Questions (resolved with defaults — confirm before or during EXECUTE)

1. **Closed-line format omits the ticker in the user's literal template** ("Mô hình T5/T20 đã dự báo
   đúng và lãi X%"). Default in this plan: ticker is included
   (`"Mô hình {hz} — mã {ticker}: đã dự báo {verdict} và {verb} {pct}%"`) for usability when more
   than one signal closes on the same day. Confirm this deviation is acceptable, or provide the
   intended literal wording if the ticker was meant to be omitted (e.g. because each closed signal is
   sent as its own separate message rather than one combined report — NOT this plan's design).
2. **"còn N ngày kết thúc dự đoán"** — no existing calendar-day countdown infra exists; this plan
   reuses `sessions_remaining` (TRADING sessions, the system-wide convention for every hold/exit
   calculation). Default: keep the word "ngày" in the Vietnamese text but source the number from
   trading sessions, not calendar days. Confirm this colloquial-but-inaccurate-by-strict-definition
   choice is acceptable, or the plan would need new calendar-day math with no existing precedent.
3. **T5 candidate selection** — Decision 5 mirrors the exact T20-dispatched ticker list. Confirm this
   matches intent, versus T5 running its own independent Top-3 selection (potentially different
   tickers than T20) — the latter would require new selection/threshold logic not currently scoped.
4. **Decision 6 (horizon tag on `/exits` + exit-due alert)** — included as a mandatory Phase 5 step in
   this plan. Confirm acceptable, or explicitly defer to backlog if strict minimal-diff scope is
   preferred (in which case Phase 5 + its test update are dropped, and `/exits`/`notify_tranche_exits`
   accept the documented wording ambiguity).

---

## Acceptance Criteria

1. `record_dispatch` called with the same ticker+day but different horizons produces 2 independent
   rows; calling it twice with the SAME horizon+day remains idempotent (0 rows on the repeat call).
2. `mark_closed` on a due row for one horizon never flips a still-open row of a different horizon for
   the same ticker/day (regression-tested, see Phase 1 item 6).
3. Every T+20 BUY dispatch produces a paired T+5 tracking row with `hold_days=5`, `horizon=5`,
   `weight=0.0`, written only when `broadcast=True`.
4. `notify_position_report()` sends exactly one combined broadcast (both configured chat IDs, via
   `send_text_alert`) per EOD run containing every currently-OPEN ledger row's live NET PnL% +
   sessions-remaining, and every row closed that same run with a đúng/sai verdict + final NET PnL%.
5. The report is skipped (no send, `TelegramBot` never constructed) when both open and closed-today
   lists are empty, and when `CONFIG.trading.eod_position_report_enabled` is `False`.
6. Report output never exceeds 4096 characters regardless of ledger size (truncation verified with a
   synthetic 50-row test fixture).
7. `/exits` and the tranche exit-due alert display a horizon tag per row.
8. Full test suite (`pytest -q`) green after implementation, with net new tests covering every item
   above.

---

## Resume and Execution Handoff

- **Read this entire plan file before resuming** — every Architecture Decision above is load-bearing,
  especially Decision 1 (bug-fix prerequisite), Decision 3 (why NOT to use the T5 artifact's own
  `.strategy`), and Decision 5 (mirrored ticker list, not independent T5 selection).
- **Phase order is a hard dependency**: Phase 1 (ledger bug fixes) MUST land and pass tests before
  Phase 2 (dual dispatch) is written — Phase 2's correctness depends on Phase 1's dedup/close fixes.
  Phase 4 (orchestration) depends on both Phase 1 (`list_closed_on`/`evaluate_signal_pnl`) and Phase 3
  (`build_position_report`).
- **No other active plan touches `signal_ledger.py`** — confirmed via a scan of
  `process/general-plans/active/` at plan-authoring time (only `intraday-attack-scanner_PLAN_06-07-26.md`
  and `portfolio-guard_PLAN_13-07-26.md` are active; neither writes to `dispatched_signals`).
- If EXECUTE is interrupted mid-phase, `git status`/`git diff` against the Implementation Checklist's
  numbered items to determine the last completed step; do not skip ahead past an incomplete Phase 1.
- This plan **stays in `process/general-plans/active/`** (not archived) until the manual live-cron
  confirmation gate (Implementation Checklist item 26 / Verification Evidence) is observed and
  reported back — same precedent as `portfolio-guard_PLAN_13-07-26.md` and
  `intraday-attack-scanner_PLAN_06-07-26.md`.

---

## Cursor + RIPER-5 Guidance

- **Cursor Plan mode**: Import the "Implementation Checklist" items directly as TODOs, execute Phase
  by Phase, run each phase's `pytest` command before moving to the next phase's checklist items.
- **RIPER-5 mode**: This file is the artifact from the PLAN phase. Say `ENTER EXECUTE MODE` to begin
  Phase 1. Mid-implementation check-in expected around the end of Phase 3 (roughly the checklist
  midpoint). After EXECUTE completes all 6 phases and the full suite is green, this plan is 🔨 CODE
  DONE, not ✅ VERIFIED, until the live-cron gate (item 26) is confirmed — do not archive via UPDATE
  PROCESS before that confirmation.
