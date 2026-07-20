# Cancelled-Signal Regret Tracking + Corporate-Action PnL-Gap Retrofit — Implementation Plan

**Date**: 16-07-26
**Complexity**: COMPLEX (standard complex — one execution stream, ~7 files touched, achievable in one
EXECUTE session; NOT a phase program)
**Status**: 🔨 CODE DONE (code shipped uncommitted in this worktree; stale ⏳ PLANNED field corrected
17-07-26 during the auto-ca-price-adjustment follow-up. Retains its own pending live-cron confirmation
gate — not yet ✅ VERIFIED. NOTE: the corporate-action "detect-and-hide" half of this plan has since been
SUPERSEDED by auto-ca-price-adjustment_PLAN_17-07-26.md, which replaced the warning-line behavior with
automatic PnL adjustment.)

## Overview

Two related pieces of work, shipped together because the second is a prerequisite building block the
first reuses:

1. **Cancelled-signal regret tracking (the user's explicit choice — "do 2 pls").** Every time the model
   screens and REJECTS a ticker ("❌ HỦY BỎ TÍN HIỆU" — below the P(up) safety threshold, or arbitrator/
   meta-gate rejected), capture it into a new `cancelled_signals` DuckDB table, then surface a daily EOD
   Telegram report showing what would have happened "if bought anyway" — explicitly framed as a
   **hypothetical, non-recommendation** hindsight/regret report, never as a trade suggestion.

2. **Corporate-action PnL-gap retrofit.** `signal_ledger.evaluate_signal_pnl` (shipped today, part of
   `process/general-plans/active/eod-position-report_PLAN_16-07-26.md`) silently misreports PnL across a
   corporate-action price-reference reset (verified live: PVD 33.3→19.47 VND overnight 07/09→07/10/2026,
   a 66.9% stock-dividend ex-rights event, computed by the shipped code as a fake −41.3% "loss"). This
   plan retrofits the ALREADY-SHIPPED, ALREADY-TESTED corporate-action gap guard
   (`price_lookup.closes_between` + `price_lookup.has_ca_gap`, already wired into
   `main._backfill_paperlog_outcomes` and `portfolio_guard.evaluate_position`) into `evaluate_signal_pnl`,
   and reuses the exact same guard for the new regret evaluator.

**Context consulted**: `process/context/all-context.md` (routing/architecture),
`process/context/planning/all-planning.md` (plan-shape calibration),
`process/context/tests/all-tests.md` (pytest runner conventions), and
`process/general-plans/active/eod-position-report_PLAN_16-07-26.md` in full (this plan's direct
predecessor — same-day, same files, same conventions; read for structure/tone calibration per user
instruction).

**Critical research finding that changed this plan's shape** (see Architecture Decision 1): the
originating research summary for this plan assumed a brand-new `signal_ledger._has_price_gap` helper was
needed. Reading the actual code found that `src/data/price_lookup.py` **already ships** a general-purpose,
already-tested corporate-action gap guard (`closes_between` + `has_ca_gap`), already wired into two other
call sites. This plan reuses it rather than inventing a parallel helper — a materially smaller Phase 1 than
originally scoped.

---

## Quick Links

- [Phase Completion Rules](#phase-completion-rules)
- [Execution Brief](#execution-brief)
- [Non-Goals and Constraints](#non-goals-and-constraints)
- [Architecture Decisions](#architecture-decisions-final)
- [Data Flow](#data-flow)
- [Schema](#schema)
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

1. **Integration Test** — Works with other system pieces (capture write → EOD read → regret report send)
2. **Manual Test** — Operator can trace a rejected ticker through the ledger to the regret report line
3. **Data Verification** — DuckDB `cancelled_signals` rows confirmed via direct query
4. **Error Handling** — Failure cases (missing price history, disabled config, ledger exception, CA gap)
   degrade gracefully, never raise
5. **User Confirmation** — First live EOD cron run that hits a fallback/no-buy day confirms real capture +
   report send (this may not be the very next cron run — see Verification Evidence)

Status meanings:
- ⏳ PLANNED — Not started
- 🔨 CODE DONE — Written but not E2E tested
- 🧪 TESTING — Currently being tested
- ✅ VERIFIED — Tested AND confirmed working (requires a live cron run on a fallback/no-buy day)
- 🚧 BLOCKED — Has issues

---

## Execution Brief

### Phase 1: `signal_ledger.py` — corporate-action gap retrofit (shared helper)
**What happens:** Extract a private `_evaluate_pnl(ticker, entry_date, hold_days, today, cost_pct)` helper
from `evaluate_signal_pnl`'s current body, adding a corporate-action gap check that reuses the
already-shipped `price_lookup.closes_between` / `price_lookup.has_ca_gap` pair (no new detection code).
`evaluate_signal_pnl` becomes a thin NET wrapper over the shared helper. Retrofit the 4 existing
`TestEvaluateSignalPnl` tests to monkeypatch `closes_between` (mandatory — without it they would silently
perform real parquet I/O against `data/ohlcv_HPG.parquet`, which exists on disk).
**Test:** `pytest tests/test_signal_ledger.py -q`.

### Phase 2: `signal_ledger.py` — new `cancelled_signals` table + read/eval functions
**What happens:** `ensure_cancelled_table`, `record_cancelled` (never-raise, idempotent per
`(ticker, screen_date, horizon)`), `list_cancelled_since` (lookback-window read + session-count math,
mirrors `list_open`/`list_closed_since`), `evaluate_regret_pnl` (thin GROSS wrapper over Phase 1's shared
helper, `cost_pct=0.0`).
**Test:** `pytest tests/test_signal_ledger.py -q` (new `TestCancelledSignalsLedger` /
`TestListCancelledSince` / `TestEvaluateRegretPnl` classes).

### Phase 3: `main.py` — two capture-point wiring (inside `daily_inference`)
**What happens:** New config kill-switch `cancelled_signal_tracking_enabled`. Two capture sites inside
`daily_inference` — the `fallback_mode` branch and the post-arbitration "NoBuyMonitor" branch — each call
`signal_ledger.record_cancelled(...)` with the rejected candidates' probabilities + Vietnamese rejection
reason, gated on the new flag. Fires on EVERY `daily_inference` call that reaches either branch, regardless
of `broadcast`/`persist` (so manual `/suggest_buy5`/`/suggest_buy20` taps are captured too — the source of
the user's own pasted evidence this session).
**Test:** New `tests/test_cancelled_signal_capture.py`, mirroring `tests/test_daily_inference_integration.py`'s
existing fallback-path fixture pattern.

### Phase 4: `src/reports/builders.py` — gap-flag retrofit + new regret report builder
**What happens:** Retrofit `_position_open_line`/`_position_closed_line` (already-shipped
`build_position_report` internals) to render an "unreliable — suspected corporate action" line instead of
the raw lãi/lỗ line when `gap_flag` is set. Add `build_regret_report` — pure HTML builder, same
char-budget/truncation/section-omit conventions as `build_position_report`, with a hard-guaranteed,
never-truncated hypothetical/non-recommendation disclaimer header.
**Test:** New `tests/test_regret_report_builder.py` + 2 new tests in the existing
`tests/test_position_report_builder.py`.

### Phase 5: `main.py` — orchestration wrapper + EOD cron wiring
**What happens:** `notify_regret_report()` — never-raise, event-gated, mirrors `notify_position_report`'s
exact shape. New `regret_report_lookback_days` config knob. Wired into `full_pipeline()` and
`inference_only()` immediately after `notify_position_report()`.
**Test:** New `tests/test_regret_report_notify.py`, mirroring `tests/test_position_report_notify.py`.

### Phase 6: Full-suite verification
**What happens:** Run the complete test suite; confirm zero regressions against the pre-plan 725-test
green baseline (measured live this session, `pytest -q`, 0 failures/errors).
**Test:** `pytest -q` — all green.

### Expected Outcome
- Every daily-inference call that lands in the weak-market fallback report OR the post-arbitration
  no-buy monitor now logs its rejected Top-3 (probabilities + Vietnamese reason) into `cancelled_signals`.
- A new EOD broadcast (`regret_report`) shows, for every logged cancellation in the trailing lookback
  window, what would have happened "if bought anyway" — clearly, unmissably labeled as hypothetical, not a
  recommendation, even under Telegram truncation.
- The PVD-class corporate-action bug in the already-shipped `evaluate_signal_pnl` is fixed by reusing
  existing, tested infrastructure — zero new gap-detection code, one shared private helper.
- Zero changes to real trading/sizing/dispatch logic anywhere in this plan.

---

## Non-Goals and Constraints

**Out of scope for this plan:**
- No new corporate-action detection algorithm — this plan strictly reuses the already-shipped
  `price_lookup.closes_between` / `price_lookup.has_ca_gap` pair, same default 10% threshold as its two
  existing call sites. A true price-adjustment system (computing the REAL adjusted return across a CA
  event) remains explicitly out of scope, same as `portfolio_guard`'s precedent — this is "detect and flag
  as unreliable," not "adjust and compute correctly."
- No `status`/`closed_date` state machine on `cancelled_signals` — every read re-evaluates "matured" fresh
  from the current date vs `hold_days` (see Architecture Decision 7). No `mark_closed`-equivalent function.
- No independent T5-vs-T20 candidate selection for cancelled signals — captured tickers are exactly
  whichever Top-3 the corresponding `daily_inference(horizon=...)` call rejected, mirroring the parent
  EOD-position-report plan's Decision 5 (mirrored ticker list, not independent selection).
- No retroactive backfill of past cancellations — `cancelled_signals` only starts accruing from ship date
  forward. The PVD gap-flag fix DOES retroactively correct that ticker's row the next time
  `notify_position_report` re-evaluates it (PnL is computed live at read time, never stored), so no
  separate backfill script is needed for the retrofit half of this plan.
- No dashboard (`dashboard/`) changes.
- No settings.json edit required (dataclass defaults are sufficient; settings.json remains an optional
  override surface per existing convention).
- No change to `audit_evaluator.py` or `portfolio_guard.py` beyond reusing their already-shipped
  `price_lookup` gap-guard functions (read-only reuse, zero edits to either file).

**Constraints:**
- Must never break existing single-purpose behavior of `evaluate_signal_pnl` (NET, dispatch-tracking) —
  its public signature is unchanged; the only behavioral addition is the new always-present `gap_flag` key.
- Must follow the established never-raise EOD-alert-only pattern (`notify_tranche_exits`,
  `notify_portfolio_guard`, `notify_position_report`): alert-only, event-gated, Telegram 4096-char-safe by
  construction.
- Must NOT perform real file I/O against `data/*.parquet` in any unit test — every test touching
  `_evaluate_pnl`/`evaluate_signal_pnl`/`evaluate_regret_pnl`/`record_cancelled`/`list_cancelled_since`
  monkeypatches `signal_ledger.price_lookup.*`, matching the repo's in-memory/monkeypatch-only test
  convention (`process/context/tests/all-tests.md`).

---

## Architecture Decisions (Final)

### Decision 1 — Reuse the EXISTING `price_lookup.closes_between` / `price_lookup.has_ca_gap` guard; do NOT write a new `_has_price_gap` helper

**Finding (confirmed by reading the actual code, contradicting this plan's originating research
summary):** `src/data/price_lookup.py` already ships:
- `closes_between(ticker, start_date, end_date, conn=None) -> list[float]` — ordered daily closes across
  an inclusive date window.
- `has_ca_gap(closes: list[float], max_session_move: float = 0.10) -> bool` — True when any consecutive
  close-to-close move exceeds the threshold (HOSE caps a session at ±7%, so a >10% single-session jump
  means a corporate action, not price action).

Both are already wired into TWO existing call sites: `main._backfill_paperlog_outcomes` (paperlog
return-window CA guard, shipped 2026-07-12, real incident: KLB 16.55→12.78, a ~30% stock dividend) and
`portfolio_guard.evaluate_position` (downgrades hard-stop/trailing-stop wording, NOT take-profit — an
already-approved, already-tested precedent for exactly this "detect and flag as unreliable, don't try to
adjust" scope). Both are unit-tested (`tests/test_sentiment_paperlog.py`'s Group C,
`tests/test_portfolio_guard.py`'s CA-gap tests).

**Decision:** This plan's retrofit and the new `evaluate_regret_pnl` both call these EXACT existing
functions — zero new gap-detection code anywhere. Default threshold (`max_session_move=0.10`, i.e. 10%)
is reused UNCHANGED, not overridden to the 15% figure floated in this plan's originating research prompt
— the PVD case (33.3→19.47, a −41.5% move) is caught either way, so reusing the default costs nothing and
buys consistency with the two already-shipped call sites (same threshold behaves identically everywhere
in the codebase — a future reader tracing "why did this flag" never has to remember a second number).

### Decision 2 — Extract a private shared `_evaluate_pnl` helper inside `signal_ledger.py`; do NOT duplicate the open/matured branching between `evaluate_signal_pnl` and `evaluate_regret_pnl`

**Rationale:** Unlike the parent EOD-position-report plan's Decision 4 (which accepted duplication
between `signal_ledger.py` and `audit_evaluator.py` because they are different modules with different
connection conventions, and the duplicated code was already-shipped/low-churn) — `evaluate_signal_pnl`
and `evaluate_regret_pnl` live in the SAME module, share the SAME `price_lookup` calling convention, and
are authored in the SAME change. There is no blast-radius or coupling excuse for duplication here.

**Decision:** `_evaluate_pnl(ticker: str, entry_date: date, hold_days: int, today: date, cost_pct: float)
-> dict` holds the full open/matured branching + gap check (Decision 1). `evaluate_signal_pnl` and
`evaluate_regret_pnl` become 3-line public wrappers that normalize their inputs and forward to the shared
helper with different `cost_pct` (Decision 3).

### Decision 3 — `evaluate_regret_pnl` is GROSS (`cost_pct=0.0`), a deliberate deviation from `evaluate_signal_pnl`'s NET convention

**Rationale:** Per the user's explicit instruction — nothing was actually traded for a cancelled signal,
so deducting a round-trip transaction cost (`_VN_ROUND_TRIP_COST_PCT = 0.30`) that was never paid would
understate the model's real "what if I had bought" accuracy and mislead the reader into thinking a real
cost was incurred. `build_regret_report`'s header (Implementation Checklist item 22) states "gộp, chưa trừ
phí" (gross, no cost deducted) explicitly so no reader mistakes this for a literal executable net return.

### Decision 4 — `cancelled_signal_tracking_enabled` is a SINGLE flag gating both the write (capture) and the read (report send); no separate report-only kill-switch

**Rationale:** A report with nothing captured is meaningless — splitting into two independent flags adds
config surface with no realistic operating mode that benefits from the split (YAGNI). See Open Question 1
for the explicit confirm-or-override framing in case the user wants "keep capturing silently, stop
broadcasting" as an operating mode later.

### Decision 5 — Capture fires from `daily_inference` itself, regardless of `broadcast`/`persist` — matches the ALREADY-SHIPPED `suggest_buy_ledger_tracking_enabled` precedent

**Finding:** `src/utils/telegram_bot.py::_suggest_buy_dispatch` (shipped today,
`tests/test_suggest_buy_ledger_tracking.py`) already proves this repo accepts "a manual preview tap writes
to an ISOLATED, purpose-built ledger table despite `persist=False`" as a safe pattern, because
`persist=False`'s ORIGINAL intent (protect `PortfolioManager`/paperlog from manual-preview pollution) has
no bearing on a brand-new table nothing else reads.

**Confirmed empirically (reading the actual code) that this existing mechanism CANNOT be reused for
cancelled-signal capture:** `_suggest_buy_dispatch`'s own ledger-tracking call site tracks
`daily_inference`'s SECOND return value (`dispatched_signals`), which is the EMPTY list `[]` on the
fallback path (`main.py`, `return report_html, []`). The existing suggest_buy tracking wiring structurally
cannot see rejected candidates. **This plan's two new capture points inside `daily_inference` itself are
therefore the only viable capture mechanism** — they cannot be piggybacked onto the existing wiring, only
modeled after its gating philosophy.

### Decision 6 — Day-level dedup on `(ticker, screen_date, horizon)`, matching `dispatched_signals`'s own convention

**Rationale:** Same reasoning as the parent plan's dedup design — a same-day repeat tap (e.g. two
`/suggest_buy5` calls) against deterministic same-day model output produces identical rows, so the
second call's insert becomes a no-op via the dedup key, with zero special-casing needed. See Open Question
2 for the explicit confirm-or-override framing.

### Decision 7 — No `status`/`closed_date` column on `cancelled_signals`; "matured" is computed fresh on every read, never persisted

**Rationale:** Unlike `dispatched_signals`, a cancelled signal is a screening EVENT, not a position with a
real exit action — nothing is ever "closed" by an alert-and-mark-closed step. Adding a state machine here
would be pure YAGNI. `list_cancelled_since` always returns every row in the lookback window;
`build_regret_report` partitions into "ĐANG THEO DÕI" (not yet matured) / "ĐÃ KẾT THÚC" (matured) purely
from `evaluate_regret_pnl`'s freshly-computed `matured` flag — deliberately DIFFERENT Vietnamese section
labels than `build_position_report`'s "ĐANG MỞ"/"ĐÃ ĐÓNG", so a reader never confuses a hypothetical
screening record with a real tracked position.

### Decision 8 — `regret_report_lookback_days` is a DEDICATED new config knob, not a reuse of `eod_position_report_lookback_days`

**Rationale:** Default matches (`7`) for day-one consistency, but the two report cadences are
independently tunable by design — regret-report review cadence (how far back to keep re-showing "what
would have happened") is a genuinely different operating concern than the live position report's lookback,
and coupling them would silently change one report's behavior when tuning the other.

---

## Data Flow

```
daily_inference(horizon)                              [main.py]
  │
  ├─ _select_candidates(...) → fallback_mode, fallback_reasons, candidate_tickers
  │
  ├─ IF fallback_mode:                                  ── Capture point 1 ──
  │     if not candidate_tickers: return (report, [])   (nothing to capture)
  │     ├─ IF cancelled_signal_tracking_enabled:
  │     │     signal_ledger.record_cancelled(
  │     │         rows-from(candidate_tickers, stacking_predictions_5d, fallback_reasons),
  │     │         horizon=int(horizon), hold_days=_cancelled_hold_days(horizon))
  │     ├─ mr_scores / fb_prices / report_html (display-only, unaffected)
  │     └─ return report_html, []                        ── daily_inference EXITS here ──
  │
  ├─ (fallback_mode False path continues: arbitrator, sentiment filter, rescue loop,
  │   run_trade_execution)
  │
  └─ IF not dispatched_signals and not report_html.strip():  ── Capture point 2 ──
        monitor = top-3 post-arbitration rejects
        reasons = {ticker: Vietnamese reason}
        ├─ IF cancelled_signal_tracking_enabled:
        │     signal_ledger.record_cancelled(
        │         rows-from(monitor, stacking_predictions_5d, reasons),
        │         horizon=int(horizon), hold_days=_cancelled_hold_days(horizon))
        └─ mr_scores / fb_prices / report_html (display-only, unaffected)

Capture points 1 and 2 are MUTUALLY EXCLUSIVE within one daily_inference call — point 1's branch always
`return`s before point 2's branch can be reached.

Fires from BOTH the cron path (full_pipeline/inference_only, horizon=20 default, broadcast=True) AND
manual /suggest_buy5 /suggest_buy20 taps (broadcast=False, persist=False) — see Decision 5.

                    ▼
        cancelled_signals table (DuckDB) — 0-3 rows per daily_inference call that hits either branch

   full_pipeline() / inference_only()
      1-6. (existing steps, unchanged — crawl, sentiment, daily_inference, notify_tranche_exits,
            notify_portfolio_guard, notify_position_report)
      7. notify_regret_report()                                              → NEW
                    │
                    ├─ signal_ledger.list_cancelled_since(lookback_days, today)  → cancelled rows
                    ├─ signal_ledger.evaluate_regret_pnl(...)  → GROSS pct + matured + gap_flag, per row
                    │     (reuses Phase 1's _evaluate_pnl(cost_pct=0.0), same CA-gap guard as
                    │      evaluate_signal_pnl's NET path)
                    ▼
        build_regret_report(cancelled_rows, today, lookback_days)  → HTML string (pure)
                    ▼
        TelegramBot().send_text_alert(msg, label="regret_report")  → broadcast to all chat IDs


Separately (retrofit, Phase 1, no new capture — read-time fix only):
        notify_position_report()  [already shipped]
                    │
                    ├─ signal_ledger.evaluate_signal_pnl(...)  → NOW also returns gap_flag
                    ▼
        build_position_report(...)  → NOW renders the "unreliable — suspected CA" line when gap_flag
```

**Naming landmine (context only, inherited from the parent plan, no new code dependency):** Inside
`daily_inference`, `stacking_predictions_5d` holds whichever horizon THIS call is running (T+20's own
probabilities when `horizon=20`, despite the "5d"-suffixed variable name) — this is the correct,
already-used source for both capture points' `p_down`/`p_side`/`p_up` fields (the SAME dict
`_build_fallback_observability_report_vi` already reads for its own display).

---

## Schema

```sql
CREATE TABLE IF NOT EXISTS cancelled_signals (
    screen_date  DATE      NOT NULL,
    ticker       VARCHAR   NOT NULL,
    horizon      INTEGER   NOT NULL,
    hold_days    INTEGER   NOT NULL,
    p_down       DOUBLE,
    p_side       DOUBLE,
    p_up         DOUBLE,
    reason       VARCHAR,
    screened_at  TIMESTAMP DEFAULT current_timestamp
)
```

Dedup key: `(ticker, screen_date, horizon)` — same pattern as `dispatched_signals`'s post-retrofit
`(ticker, horizon)` key scoped to a single `dispatch_date`/`screen_date`. `horizon` is `NOT NULL` here
(unlike `dispatched_signals.horizon`, which stayed nullable for legacy single-horizon rows written before
today's dual-horizon retrofit) — this table is brand new, so no legacy NULL-horizon rows will ever exist.

No `status`/`closed_date` columns (Decision 7) — "matured" is always computed fresh at read time.

---

## Touchpoints

| File | Change type | Why |
|---|---|---|
| `src/trading/signal_ledger.py` | Retrofit `evaluate_signal_pnl` (gap_flag) + add `cancelled_signals` table + 3 new functions | CA-gap fix prerequisite; new capture/read/eval surface |
| `main.py` | Add 2 capture-point calls + `_cancelled_hold_days` helper + `notify_regret_report()` + cron wiring + import update | Orchestration |
| `src/reports/builders.py` | Retrofit `_position_open_line`/`_position_closed_line` (gap_flag) + add `build_regret_report` | Pure formatter, both retrofit and new |
| `config/settings.py` | Add `cancelled_signal_tracking_enabled: bool = True`, `regret_report_lookback_days: int = 7` | Kill-switch + lookback knob, repo convention |
| `tests/test_signal_ledger.py` | Retrofit 4 tests + extend | Regression + unit coverage (gap retrofit, new table) |
| `tests/test_position_report_builder.py` | Extend | 2 new gap_flag-rendering tests |
| `tests/test_cancelled_signal_capture.py` | New | Phase 3 capture-point wiring |
| `tests/test_regret_report_builder.py` | New | Phase 4 pure formatter |
| `tests/test_regret_report_notify.py` | New | Phase 5 orchestration wrapper |

---

## Public Contracts

**Must remain compatible:**
- `signal_ledger.evaluate_signal_pnl(ticker, dispatch_date, hold_days, today=None) -> dict` — signature
  UNCHANGED. Return dict gains one new always-present key, `"gap_flag": bool`, on the success path (error
  path `{"ticker", "error"}` unchanged). Existing callers (`main.notify_position_report`'s `_enrich`, which
  merges the whole dict via `**pnl`) need zero code changes — the new key flows through automatically.
- `dispatched_signals` table schema and every function that reads/writes it — completely untouched by this
  plan (only `evaluate_signal_pnl`'s in-memory return dict changes).
- `src/reports/builders.build_position_report(open_rows, closed_rows, today, lookback_days) -> str` —
  signature unchanged. Rows that never carry a `gap_flag` key render EXACTLY as before (`.get("gap_flag")`
  → `None` → falsy → unchanged branch) — zero regression risk to existing callers/tests.

**New public surface introduced:**
- `signal_ledger._evaluate_pnl(ticker, entry_date, hold_days, today, cost_pct) -> dict` (private, shared)
- `signal_ledger.evaluate_regret_pnl(ticker, screen_date, hold_days, today=None) -> dict`
- `signal_ledger.record_cancelled(candidates, horizon, hold_days, db_path=None, today=None) -> int`
- `signal_ledger.list_cancelled_since(lookback_days, today=None, db_path=None) -> list[dict]`
- `signal_ledger.ensure_cancelled_table(conn) -> None`
- `src/reports/builders.build_regret_report(cancelled_rows, today, lookback_days) -> str`
- `main.notify_regret_report() -> int`
- `main._cancelled_hold_days(horizon: int) -> int` (private)
- `config.settings.TradingConfig.cancelled_signal_tracking_enabled: bool` (default `True`)
- `config.settings.TradingConfig.regret_report_lookback_days: int` (default `7`)

---

## Blast Radius

**Directly touched:** `src/trading/signal_ledger.py`, `main.py`, `src/reports/builders.py`,
`config/settings.py`, plus the 5 test files listed in Touchpoints.

**Indirectly affected (no code change, behavior grows or corrects):**
- `notify_position_report()`'s EOD broadcast — rows involved in a corporate-action price gap now render a
  clearly-flagged "unreliable" line instead of a fake lãi/lỗ number (this is a bug FIX, user-visible text
  change to an existing live Telegram message, on the specific rows affected).
- Every future dispatched-signal PnL evaluation gets the same protection going forward (not just PVD).

**Explicitly NOT touched:**
- `src/data/price_lookup.py` — read-only reuse of already-shipped functions, zero edits.
- `src/trading/portfolio_guard.py` — read-only precedent reference, zero edits.
- `src/utils/audit_evaluator.py` — not touched (same rationale as the parent plan's Decision 4; flagged
  again here as a standing candidate backlog cleanup item, not part of this plan).
- `src/utils/telegram_bot.py::_suggest_buy_dispatch` / `suggest_buy_ledger_tracking_enabled` — read-only
  precedent reference (Decision 5), zero edits.
- `PortfolioManager` / real sizing / NAV logic / `dispatched_signals` table — untouched.
- `dashboard/` — no changes.

---

## Implementation Checklist

### Phase 1 — `src/trading/signal_ledger.py`: corporate-action gap retrofit

1. Add a private helper `_evaluate_pnl(ticker: str, entry_date: date, hold_days: int, today: date,
   cost_pct: float) -> dict` — extract `evaluate_signal_pnl`'s current open/matured branching body
   verbatim (t0 = `close_on_or_before(ticker, entry_date)`; matured once `hold_days` trading sessions
   elapse per `trading_dates_after(entry_date)`, exit = the `hold_days`-th session's close, else exit =
   `latest_close(ticker)`, provisional). AFTER the existing `t0 is None`/`t0 <= 0` error checks, add the
   corporate-action gap check: `window_end = exit_date if matured else today`;
   `closes = price_lookup.closes_between(ticker, entry_date, window_end)`;
   `gap_flag = price_lookup.has_ca_gap(closes)` (default threshold, per Decision 1 — no override). Compute
   `pct = (t_exit - t0) / t0 * 100.0 - cost_pct` regardless of `gap_flag` (per the user's explicit
   contract: the number is still returned, just flagged as unreliable — the caller/report decides how to
   render it, the evaluator never silently drops it). Return
   `{"ticker", "pct", "matured", "t0", "t_exit", "gap_flag"}` on success,
   `{"ticker", "error"}` unchanged on failure (gap check skipped on the error path — nothing to flag).
2. Rewrite `evaluate_signal_pnl(ticker, dispatch_date, hold_days, today=None) -> dict` as: normalize
   `ticker`/`today`/`hold_days` (unchanged from current code), then
   `return _evaluate_pnl(ticker, dispatch_date, hold_days, today, _VN_ROUND_TRIP_COST_PCT)`. Update the
   docstring: cite the PVD 2026-07-09→07-10 incident (33.3→19.47, 66.9% stock-dividend ex-rights) as the
   motivating bug, and document the new `gap_flag` key.
3. Retrofit the 4 existing `TestEvaluateSignalPnl` tests in `tests/test_signal_ledger.py`
   (`test_open_provisional`, `test_matured_exit_price`, `test_missing_price_returns_error`,
   `test_t0_zero_or_negative_returns_error`) to ALSO
   `monkeypatch.setattr(signal_ledger.price_lookup, "closes_between", lambda t, s, e, conn=None: [...])`
   with a short flat/no-gap series consistent with each test's own fixture prices. **This is mandatory,
   not optional** — without it, the retrofit silently makes these tests perform real file I/O against
   `data/ohlcv_HPG.parquet` (confirmed present on disk: `data/ohlcv_HPG.parquet`, `data/ohlcv_FPT.parquet`,
   `data/ohlcv_PVD.parquet` all exist in this repo), violating the in-memory/monkeypatch-only test
   convention and making these tests slow/non-hermetic. Add `assert out["gap_flag"] is False` to each of
   the 4 retrofitted tests as an explicit regression guard (previously this key didn't exist at all).
4. Add 2 new tests to `TestEvaluateSignalPnl`:
   - `test_gap_flag_true_reproduces_pvd_case` — monkeypatch `close_on_or_before` → `33.3` for the entry
     date, exit close → `19.47` (matured path), `closes_between` → `[33.3, 19.47]`; assert
     `out["gap_flag"] is True` AND `out["pct"]` is still present and numerically computed (per the
     evaluator's explicit "never silently drop the number" contract from item 1).
   - `test_gap_flag_false_normal_move` — a mild ±5% move fixture, `closes_between` returns a realistic
     no-gap series; assert `out["gap_flag"] is False`.
5. Run `pytest tests/test_signal_ledger.py -q` — all green; zero regressions to `TestRecordDispatch`,
   `TestExitsDue`, `TestListOpen`, `TestDualHorizonDispatch`, `TestListClosedSince` (untouched by this
   phase).

### Phase 2 — `src/trading/signal_ledger.py`: new `cancelled_signals` table + read/eval functions

6. Add module constant `CANCELLED_TABLE = "cancelled_signals"` and `ensure_cancelled_table(conn) -> None`
   (DDL per the [Schema](#schema) section above).
7. Add `record_cancelled(candidates: list[dict], horizon: int, hold_days: int, db_path: str | None = None,
   today: date | None = None) -> int`: never-raise (wrap the whole body in try/except → log + return `0`,
   mirroring `record_dispatch`'s exact shape). No `mode == tranche` gate (unlike `record_dispatch` —
   cancelled signals have no artifact-strategy concept; `hold_days` is passed directly by the caller).
   Idempotent dedup on `(ticker, horizon)` scoped to `screen_date = today` — query existing
   `(ticker, horizon)` pairs for today, filter `candidates` before insert (identical pattern to
   `record_dispatch`'s already-shipped dedup). Each `candidates[i]` dict shape:
   `{"ticker", "p_down", "p_side", "p_up", "reason"}`.
8. Add `list_cancelled_since(lookback_days: int, today: date | None = None, db_path: str | None = None) ->
   list[dict]`: same inclusive-window read pattern as `list_closed_since`
   (`screen_date >= today - lookback_days AND screen_date <= today`, `ORDER BY screen_date, ticker`), PLUS
   `sessions_elapsed`/`sessions_remaining` computed the same way as `list_open` (via
   `price_lookup.trading_dates_after(min(screen_date))`, count sessions `d0 < s <= today`,
   `sessions_remaining = max(0, hold_days - elapsed)`). Never raises — degrades to `[]`.
9. Add `evaluate_regret_pnl(ticker: str, screen_date: date, hold_days: int, today: date | None = None) ->
   dict`: normalize inputs, `return _evaluate_pnl(ticker, screen_date, hold_days, today, cost_pct=0.0)` —
   **GROSS, not NET** (Decision 3). Docstring must state explicitly: "nothing was actually traded — no
   round-trip transaction cost is deducted; the caller/report must render this as a gross, hypothetical
   figure."
10. Add tests to `tests/test_signal_ledger.py`:
    - New `TestCancelledSignalsLedger` class: `test_record_cancelled_inserts_rows`,
      `test_record_cancelled_dedup_same_day_ticker_horizon`,
      `test_record_cancelled_different_horizons_same_day_both_persist` (a ticker rejected by BOTH the T5
      and T20 model on the same day produces 2 independent rows — mirrors
      `TestDualHorizonDispatch.test_t5_and_t20_rows_same_ticker_day_both_persist`),
      `test_record_cancelled_never_raises` (force an exception, assert `0` returned, no propagation).
    - New `TestListCancelledSince` class mirroring `TestListClosedSince`'s window-filter pattern:
      today-only, inside-window, outside-window-excluded, future-date-excluded (no "excludes OPEN" case —
      this table has no status column).
    - New `TestEvaluateRegretPnl` class: `test_open_provisional`, `test_matured_exit_price`,
      `test_gross_no_cost_deduction` (**critical regression-proof test** — same fixture t0/t_exit values as
      `TestEvaluateSignalPnl.test_matured_exit_price`; assert `evaluate_regret_pnl`'s `pct` is EXACTLY
      `_VN_ROUND_TRIP_COST_PCT` percentage-points higher than `evaluate_signal_pnl`'s `pct` for the
      identical price path — proves the GROSS/NET split is real, not accidentally identical),
      `test_gap_flag_propagates` (same PVD-style fixture as Phase 1 item 4, confirms the shared
      `_evaluate_pnl` helper's gap detection also fires through the regret path).
11. Run `pytest tests/test_signal_ledger.py -q` again — all green.

### Phase 3 — `main.py`: two capture-point wiring inside `daily_inference`

12. Add `TradingConfig.cancelled_signal_tracking_enabled: bool = True` to `config/settings.py`, appended
    after `suggest_buy_ledger_tracking_enabled` (matching that field's kill-switch comment style: "set
    `cancelled_signal_tracking_enabled: false` in settings.json + restart to disable — gates both the
    capture writes AND the regret report send, see plan Decision 4").
13. Add `TradingConfig.regret_report_lookback_days: int = 7` immediately after item 12's field — a
    dedicated knob (Decision 8), not a reuse of `eod_position_report_lookback_days`.
14. Add a small private helper in `main.py`, placed immediately before `daily_inference` alongside
    `_select_candidates`/`_rescue_loop`: `_cancelled_hold_days(horizon: int) -> int` returning
    `SHORT_HORIZON if int(horizon) == SHORT_HORIZON else 30` — the same T5=5/T20=30 real-tranche-window
    convention already established by the paired-tracking dispatch (parent plan Decision 3) and the
    suggest_buy ledger tracking feature.
15. **Capture point 1** — inside `daily_inference`'s `if fallback_mode:` branch, immediately AFTER the
    `if not candidate_tickers: return (...)` early-exit guard (guaranteeing a non-empty list) and BEFORE
    `mr_scores = mr_score_tickers(candidate_tickers)`: when `CONFIG.trading.cancelled_signal_tracking_enabled`,
    build one `{"ticker", "p_down", "p_side", "p_up", "reason"}` dict per `t in candidate_tickers`
    (probabilities from `stacking_predictions_5d.get(t, [0.0, 0.0, 0.0])` — see the naming-landmine note in
    Data Flow; reason from `fallback_reasons.get(t, "")`), then call
    `signal_ledger.record_cancelled(rows, int(horizon), _cancelled_hold_days(horizon))` bare — no wrapping
    try/except at the call site (`record_cancelled` is itself never-raise, mirroring how `record_dispatch`
    is already called bare at its two existing call sites).
16. **Capture point 2** — inside `daily_inference`'s post-arbitration
    `if not dispatched_signals and not (report_html or "").strip():` "NoBuyMonitor" branch, immediately
    AFTER the `reasons = {}` for-loop finishes populating (before `mr_scores = mr_score_tickers(monitor)`):
    same shape as item 15, using `monitor` as the ticker list and `reasons` as the reason-lookup dict.
17. Create `tests/test_cancelled_signal_capture.py`, mirroring
    `tests/test_daily_inference_integration.py`'s `@patch("main....")` stack style (that file already
    covers the fallback path and is the established precedent for driving `daily_inference` through these
    exact branches):
    - `test_fallback_path_records_cancelled_signals` — extends
      `test_daily_inference_fallback_path`'s exact monkeypatch stack (`main.predict_v3_horizon`,
      `main.evaluate_trades_batch`, `main.run_trade_execution`, `main.mr_score_tickers`,
      `main._get_live_exec_prices`, `main._build_fallback_observability_report_vi`), adds
      `@patch("main.signal_ledger.record_cancelled")`; asserts it is called once with the right horizon,
      `hold_days=30` (or `5`, per the test's chosen horizon), and a candidate-ticker set matching
      `predict_side_effect`'s fallback P(up) values.
    - `test_nobuy_monitor_path_records_cancelled_signals` — new fixture: `_select_candidates` returns
      `fallback_mode=False` (predictions clear the technical gate) but `evaluate_trades_batch` returns
      non-BUY (`0`/`1`) decisions for every candidate, and `run_trade_execution` returns `("", [])`; assert
      `record_cancelled` called once with the `monitor` ticker set.
    - `test_capture_points_are_mutually_exclusive` — confirms `record_cancelled` is called at most once per
      `daily_inference` invocation.
    - `test_disabled_config_skips_both_capture_points` —
      `CONFIG.trading.cancelled_signal_tracking_enabled = False`; drive both scenarios; assert zero
      `record_cancelled` calls in either.
18. Run `pytest tests/test_cancelled_signal_capture.py tests/test_daily_inference_integration.py
    tests/test_select_candidates.py tests/test_rescue_loop.py -q` — all green.

### Phase 4 — `src/reports/builders.py`: gap-flag retrofit + new regret report builder

19. Retrofit `_position_open_line(r)` and `_position_closed_line(r)`: when `r.get("gap_flag")` is truthy,
    render `"⚠️ giá bất thường (nghi sự kiện DN — cổ tức/chia tách) — số liệu không đáng tin"` in place of
    the `{verb} {abs(pct):.1f}%` clause (open-row line keeps its `(còn {rem} ngày kết thúc dự đoán)` suffix;
    closed-row line keeps its `Mô hình {hz} — mã {ticker}:` prefix, drops `đã dự báo {verdict} và`).
    Existing rows that never carry a `gap_flag` key render EXACTLY as before (`.get("gap_flag")` → `None` →
    falsy → unchanged branch) — zero regression to `tests/test_position_report_builder.py`'s existing
    assertions.
20. Add 2 new tests to `tests/test_position_report_builder.py`: `test_gap_flag_open_line_renders_warning`,
    `test_gap_flag_closed_line_renders_warning` — fixture rows with `"gap_flag": True`; assert the warning
    substring appears and the raw `{pct}%` number does NOT appear in the rendered line.
21. Add `_REGRET_REPORT_CHAR_BUDGET: int = 3800`, `_regret_open_line(r)` / `_regret_closed_line(r)`
    helpers, and `build_regret_report(cancelled_rows: list[dict], today: date, lookback_days: int) -> str`
    to `src/reports/builders.py` — pure function, same char-budget accumulate-and-cut /
    section-omit-when-empty / `""`-on-both-empty / error-row-skip contract as `build_position_report`.
    Line templates (gap-flag variant substitutes the outcome clause exactly as in item 19):
    - Still-tracking: `f"Đã huỷ ngày {screen_date:%d/%m/%Y}: mã {ticker} (P(tăng)={p_up*100:.0f}%, lý do:
      {reason}) — {outcome} (còn {sessions_remaining} ngày)"` where
      `outcome = f"nếu mua thì hiện {verb} {abs(pct):.1f}%"` or the gap-flag warning text.
    - Matured: `f"Đã huỷ ngày {screen_date:%d/%m/%Y}: mã {ticker} — {outcome}"` where
      `outcome = f"nếu mua thì đã {verb} {abs(pct):.1f}%"` or the gap-flag warning text.
    - Header (**hard requirement — see Non-Goals/Decision framing**): includes the literal title
      `"🔍 KIỂM CHỨNG TÍN HIỆU BỊ HUỶ (giả định — KHÔNG phải khuyến nghị)"` plus a separate italic
      disclaimer sentence stating this is a retroactive simulation for rejected BUY candidates, not a
      trading suggestion, and that PnL is gross (no cost deducted). The header is NEVER part of the
      truncatable line list — it is always fully present in the output, matching `build_position_report`'s
      existing safety property that its own header can never be truncated away.
    - Two labeled sections, "⏳ ĐANG THEO DÕI" / "🏁 ĐÃ KẾT THÚC ({lookback_days} NGÀY QUA)" (Decision 7 —
      deliberately different wording from `build_position_report`'s "ĐANG MỞ"/"ĐÃ ĐÓNG"); section header
      omitted when its row list is empty; returns `""` when both lists are empty post error-filtering.
22. Create `tests/test_regret_report_builder.py` (mirrors `tests/test_position_report_builder.py`'s
    structure): `test_open_line_exact_format`, `test_closed_line_exact_format`,
    `test_gap_flag_line_variant_open`, `test_gap_flag_line_variant_closed`,
    `test_disclaimer_present_and_survives_truncation` (**hard requirement test** — synthetic 50-row
    fixture forcing truncation; assert the header's "KHÔNG phải khuyến nghị" substring is STILL present in
    the output), `test_both_empty_returns_empty_string`, `test_error_rows_excluded`,
    `test_open_rows_sorted_by_sessions_remaining`, `test_truncation_under_telegram_limit`
    (`len(output) <= 4096`).
23. Run `pytest tests/test_position_report_builder.py tests/test_regret_report_builder.py -q` — all green.

### Phase 5 — `main.py`: orchestration wrapper + EOD cron wiring

24. Add `build_regret_report` to the `from src.reports.builders import (...)` block in `main.py` (same
    block that already imports `build_position_report`).
25. Add `notify_regret_report() -> int` to `main.py`, placed near `notify_position_report`, mirroring its
    EXACT never-raise/event-gated shape: short-circuit
    `if not CONFIG.trading.cancelled_signal_tracking_enabled: return 0` (zero DB reads);
    `today = datetime.now().date()`;
    `cancelled_raw = signal_ledger.list_cancelled_since(CONFIG.trading.regret_report_lookback_days, today)`;
    if empty → log + `return 0`; enrich each row via
    `signal_ledger.evaluate_regret_pnl(r["ticker"], r["screen_date"], r["hold_days"], today=today)` merged
    `{**r, **pnl}` (log, don't raise, on any per-row `error`); `msg = build_regret_report(rows, today,
    lookback_days)`; if falsy → `return 0`; `TelegramBot().send_text_alert(msg, label="regret_report")`;
    whole body wrapped in try/except → log + `return 0`.
26. Wire `notify_regret_report()` into `full_pipeline()` and `inference_only()`, called immediately AFTER
    `notify_position_report()`, as a new step 7, wrapped in
    `timed_step("EOD regret report (cancelled signals)")` matching the existing step style.
27. Update `full_pipeline`'s docstring numbered step list (currently 1-6) to append step 7: "EOD regret
    report — hypothetical PnL on rejected signals (signal ledger)." Update `inference_only`'s docstring
    similarly ("+ EOD regret report").
28. Create `tests/test_regret_report_notify.py` mirroring `tests/test_position_report_notify.py`'s exact
    4-test pattern: `test_notify_disabled_no_db_read`, `test_notify_nothing_to_report_returns_zero`,
    `test_notify_sends_combined_report` (assert `label="regret_report"`),
    `test_notify_never_raises_on_ledger_exception`.
29. Run `pytest tests/test_regret_report_notify.py -q` — all green.

### Phase 6 — Full-suite verification

30. Run `pytest -q` (full suite) — zero regressions against the pre-plan 725-test green baseline (measured
    live this session: `pytest -q`, 725 passed, 0 failed, 0 errors), all new/retrofitted tests green.
31. Manual/live confirmation gate (mirrors the parent plan's item 26 and the Portfolio Guard / Intraday
    Scanner precedent): first production EOD cron run (`full_pipeline`, 15:30 ICT) on a day that actually
    triggers `fallback_mode` or the NoBuyMonitor branch must be observed to (a) write `cancelled_signals`
    rows, (b) not double-book on a same-day rerun, and (c) send exactly one `regret_report`-labeled
    broadcast once at least one row exists in-window. **Because this feature's write side only fires on a
    weak-market/no-buy day — unlike the parent plan's position report, which writes on every dispatch day —
    this gate may take several calendar days to naturally occur.** State this explicitly to the user rather
    than silently blocking on it. This plan stays ACTIVE (not archived) until that confirmation lands, same
    precedent as the sibling EOD-broadcast plans.

---

## Test Matrix

| Area | File | Type | Key scenarios |
|---|---|---|---|
| CA-gap retrofit | `tests/test_signal_ledger.py::TestEvaluateSignalPnl` | Unit | PVD-shape gap reproduction, normal-move no-gap, 4 retrofitted existing tests (mandatory `closes_between` monkeypatch) |
| Cancelled-signal ledger | `tests/test_signal_ledger.py::TestCancelledSignalsLedger` | Unit | insert, dedup, dual-horizon same-day, never-raises |
| Cancelled-signal read | `tests/test_signal_ledger.py::TestListCancelledSince` | Unit | today/inside/outside/future-date window filters |
| Regret PnL evaluator | `tests/test_signal_ledger.py::TestEvaluateRegretPnl` | Unit | open/matured, GROSS-vs-NET differential (critical), gap-flag propagation |
| Capture wiring | `tests/test_cancelled_signal_capture.py` | Unit (monkeypatch, mirrors `test_daily_inference_integration.py`) | fallback path, NoBuyMonitor path, mutual exclusivity, disabled config |
| Position report retrofit | `tests/test_position_report_builder.py` | Unit (pure) | gap-flag open/closed line rendering, existing tests unmodified/unbroken |
| Regret report formatter | `tests/test_regret_report_builder.py` | Unit (pure) | line formats, gap-flag variants, disclaimer presence + truncation survival (hard requirement), section omission, char-budget |
| Regret orchestration | `tests/test_regret_report_notify.py` | Unit (monkeypatch, mirrors `test_position_report_notify.py`) | disabled short-circuit, nothing-to-report, send path, never-raises |
| Regression | full `pytest -q` | Full suite | zero breakage across the 725-test baseline |
| Manual | live cron (fallback/no-buy day) | E2E | first production run that exercises either capture branch — required before plan archival |

---

## Verification Evidence

- Pre-plan baseline measured live this session: `pytest -q` → 725 passed, 0 failed, 0 errors (repo root,
  `pytest.ini` addopts `-q --tb=short`).
- All Phase 1-5 pytest files pass locally (targeted runs per phase, then the full suite in Phase 6) — this
  is the automated evidence bar.
- No live EOD cron run on a fallback/no-buy day has occurred as of plan authoring — production confirmation
  is a **manual, required gate** (Implementation Checklist item 31) that may not land on the very next cron
  run, since the write side is conditional on market conditions, not guaranteed daily. Do not mark this
  plan's phases ✅ VERIFIED (only 🔨 CODE DONE / 🧪 TESTING) until that run is observed and its ledger +
  Telegram output manually inspected.
- Direct DB query to confirm capture after the first qualifying live run:
  `SELECT screen_date, ticker, horizon, hold_days, p_up, reason FROM cancelled_signals ORDER BY screen_date
  DESC, ticker, horizon LIMIT 20;`
- Direct DB query / manual read to confirm the CA-gap retrofit on the next `notify_position_report` run
  that re-evaluates the still-open PVD dispatch row (if still within its hold window) — expect the position
  report to render the "⚠️ giá bất thường" line for PVD instead of the fake −41.3% loss line.

---

## Open Questions (resolved with defaults — confirm before or during EXECUTE)

1. **Single flag (write + report) vs a separate report-only kill-switch (Decision 4).** Default: one
   `cancelled_signal_tracking_enabled` flag gates both the two `daily_inference` capture writes AND
   `notify_regret_report`'s send. Confirm this is acceptable, or specify the operating mode that would need
   a split (e.g. "keep capturing silently, stop broadcasting temporarily").
2. **Day-level dedup on repeated same-day taps (Decision 6).** Default: `(ticker, screen_date, horizon)`
   dedup, matching `dispatched_signals`'s own convention — a second same-day `/suggest_buy5` tap against
   the same rejected ticker is a no-op, not a fresh log entry. Confirm this default, or specify that
   repeated same-day screenings of the same rejected ticker should log a NEW row each time (e.g. to track
   how a rejection reason evolves intraday, which the current parquet-refresh-once-a-day serve path
   wouldn't actually exercise differently anyway, but is worth confirming explicitly per the user's own
   flagged uncertainty).

---

## Acceptance Criteria

1. `evaluate_signal_pnl`'s return dict always carries a `gap_flag: bool` key on the success path; a
   synthetic PVD-shape fixture (33.3→19.47) sets `gap_flag=True` while still returning a numeric `pct`.
2. The 4 pre-existing `TestEvaluateSignalPnl` tests pass unchanged in outcome (plus a new `gap_flag is
   False` assertion each) and no longer implicitly depend on real `data/*.parquet` files being present or
   absent.
3. `record_cancelled` writes one row per rejected candidate per `daily_inference` call that reaches either
   capture branch, deduped per `(ticker, screen_date, horizon)`, and never raises on a DB failure.
4. Every fallback-mode or NoBuyMonitor `daily_inference` call — cron OR manual `/suggest_buy5`/
   `/suggest_buy20` tap — captures its rejected Top-3 when `cancelled_signal_tracking_enabled` is `True`,
   and captures nothing when `False` (zero DB reads/writes on that path).
5. `evaluate_regret_pnl`'s `pct` is GROSS — for an identical price path, its value differs from
   `evaluate_signal_pnl`'s NET value by exactly `_VN_ROUND_TRIP_COST_PCT` percentage points (regression
   test, not just documentation).
6. `notify_regret_report()` sends exactly one `regret_report`-labeled broadcast per EOD run containing
   every in-lookback-window cancelled signal's hypothetical PnL, is skipped (no send, `TelegramBot` never
   constructed) when there is nothing to report or when the config flag is `False`, and its rendered
   header unambiguously states this is a hypothetical simulation, NOT a trading recommendation — this
   disclaimer text survives truncation under a synthetic 50-row fixture (hard requirement, explicitly
   tested).
7. `build_position_report`'s existing rows continue to render identically to before this plan when they
   carry no `gap_flag` key; rows that do carry `gap_flag=True` render the "unreliable — suspected
   corporate action" line instead of a numeric lãi/lỗ figure.
8. Report output (`build_regret_report`) never exceeds 4096 characters regardless of ledger size.
9. Full test suite (`pytest -q`) green after implementation, net new/retrofitted tests covering every item
   above, zero regressions against the 725-test pre-plan baseline.

---

## Resume and Execution Handoff

- **Read this entire plan file before resuming** — every Architecture Decision above is load-bearing,
  especially Decision 1 (reuse the existing gap guard, do NOT write a new one), Decision 2 (shared private
  `_evaluate_pnl` helper, DRY), Decision 3 (GROSS vs NET), and Decision 5 (why the existing suggest_buy
  ledger-tracking mechanism cannot be reused for this capture).
- **This plan is a direct, same-day follow-up to
  `process/general-plans/active/eod-position-report_PLAN_16-07-26.md`**, which is still ACTIVE (status
  🔨 CODE DONE, not yet ✅ VERIFIED — pending its own live-cron confirmation gate). This plan's Phase 1
  retrofits code that plan shipped (`evaluate_signal_pnl`, `build_position_report`'s line renderers) — read
  that plan's Architecture Decisions 1-8 for the surrounding context on why `dispatched_signals` is
  horizon-scoped the way it is, before touching `signal_ledger.py`.
- **Phase order is a hard dependency**: Phase 1 (gap retrofit + shared helper) MUST land and pass tests
  before Phase 2 (new table + `evaluate_regret_pnl`, which calls Phase 1's `_evaluate_pnl`). Phase 3
  (capture wiring) depends on Phase 2's `record_cancelled`. Phase 5 (orchestration) depends on Phase 2
  (`list_cancelled_since`/`evaluate_regret_pnl`) and Phase 4 (`build_regret_report`).
- **Active-plan scan at plan-authoring time** (`process/general-plans/active/`): three plans present —
  `eod-position-report_PLAN_16-07-26.md` (this plan's direct predecessor, same files, see above),
  `intraday-attack-scanner_PLAN_06-07-26.md` and `portfolio-guard_PLAN_13-07-26.md` (neither touches
  `signal_ledger.py` or the two `daily_inference` capture points this plan instruments — no conflict).
- If EXECUTE is interrupted mid-phase, `git status`/`git diff` against the Implementation Checklist's
  numbered items to determine the last completed step; do not skip ahead past an incomplete Phase 1 or 2.
- This plan **stays in `process/general-plans/active/`** (not archived) until the manual live-cron
  confirmation gate (Implementation Checklist item 31 / Verification Evidence) is observed and reported
  back. Because the write side is conditional on a weak-market/no-buy day occurring, this confirmation may
  reasonably take longer to land than the parent plan's — do not treat a delay here as a sign of a bug.

---

## Cursor + RIPER-5 Guidance

- **Cursor Plan mode**: Import the "Implementation Checklist" items directly as TODOs, execute Phase by
  Phase, run each phase's `pytest` command before moving to the next phase's checklist items.
- **RIPER-5 mode**: This file is the artifact from the PLAN phase. Say `ENTER EXECUTE MODE` to begin
  Phase 1. Mid-implementation check-in expected around the end of Phase 3 (roughly the checklist midpoint).
  After EXECUTE completes all 6 phases and the full suite is green, this plan is 🔨 CODE DONE, not
  ✅ VERIFIED, until the live-cron gate (item 31) is confirmed — do not archive via UPDATE PROCESS before
  that confirmation.
