# Auto Corporate-Action Price Adjustment — Implementation Plan

**Date**: 17-07-26
**Complexity**: COMPLEX (standard complex — one execution stream, 7 files touched, achievable in one
EXECUTE session; NOT a phase program)
**Status**: ✅ VERIFIED (all 4 phases executed 17-07-26; full suite 767 passed / 0 failed, +9 net-new
tests over the 758 baseline; PVD before/after reproduced: raw NET −41.8% → corrected NET −0.30% with
adjustment_factor 0.585. No live-cron gate blocks archival for this plan — ready for UPDATE PROCESS.)

## Overview

Upgrade the already-shipped corporate-action gap guard (`signal_ledger._evaluate_pnl`, shared by
`evaluate_signal_pnl` NET and `evaluate_regret_pnl` GROSS) from **"detect a gap and hide the number
behind a warning"** to **"detect a gap and automatically compute the correct, adjusted PnL."** No
external corporate-action data source and no manual per-event data entry — both explicitly rejected by
the user this session. Instead, this plan reuses the same self-referential technique already implicit
in the existing `price_lookup.has_ca_gap` heuristic: VN exchanges cap a single session's move at
roughly ±7% (HOSE) / ±10% (HNX), so any single-session move that clears the existing 10% gap threshold
is mechanically a corporate-action price reset, and the ratio between the two adjacent closes **is**
the back-adjustment factor — the same technique used when a data vendor doesn't publish a clean
adjusted-close series.

**Live regression case this plan fixes:** PVD dispatched 2026-07-09 at close 33.3 (thousand VND),
closed 2026-07-10 at 19.47 — a 66.9% stock-dividend ex-rights reset. Pre-fix: the shipped
"detect-and-hide" code correctly detects the gap (`gap_flag=True`) but still computes and would-have-
displayed a fake −41.3%/−41.8% "loss" behind a "⚠️ giá bất thường — không đáng tin" warning line
(this exact warning-line behavior is what the prior same-day plan,
`cancelled-signal-regret-tracking_PLAN_16-07-26.md`, shipped uncommitted in this worktree — see
Resume and Execution Handoff). Post-fix: the same fixture yields a corrected NET pct of **≈ −0.30%**
(the round-trip cost only — the underlying price move is economically flat once the stock-dividend
shares are accounted for), rendered as a **trusted** number with a transparent
"⚙️ đã tự động điều chỉnh sự kiện DN, hệ số 0.585" annotation instead of a warning.

**Context consulted**: `process/context/all-context.md` (routing/architecture),
`process/context/planning/all-planning.md` (plan-shape calibration),
`process/context/tests/all-tests.md` (pytest runner conventions), and both
`process/general-plans/active/eod-position-report_PLAN_16-07-26.md` and
`process/general-plans/active/cancelled-signal-regret-tracking_PLAN_16-07-26.md` in full — this plan is
a direct, same-day, same-worktree follow-up that retrofits code those two plans shipped (uncommitted)
earlier today.

**Verified against the ACTUAL current file contents** (not the sibling plans' own text, which predates
today's implementation) — `src/trading/signal_ledger.py`, `src/data/price_lookup.py`,
`src/reports/builders.py`, and their test files were all read in full before this plan was written. Live
baseline measured this session: `pytest -q` → **758 passed, 0 failed, 0 errors** (confirmed via a fresh
full run, 17-07-26 — matches the orchestrator's "last measured 758" note).

---

## Quick Links

- [Phase Completion Rules](#phase-completion-rules)
- [Execution Brief](#execution-brief)
- [Non-Goals and Constraints](#non-goals-and-constraints)
- [Architecture Decisions](#architecture-decisions-final)
- [Data Flow](#data-flow)
- [Touchpoints](#touchpoints)
- [Public Contracts](#public-contracts)
- [Blast Radius](#blast-radius)
- [Implementation Checklist](#implementation-checklist)
- [Test Matrix](#test-matrix)
- [Verification Evidence](#verification-evidence)
- [Open Questions](#open-questions-confirm-before-or-during-execute)
- [Acceptance Criteria](#acceptance-criteria)
- [Resume and Execution Handoff](#resume-and-execution-handoff)

---

## Phase Completion Rules

A phase is NOT complete until:

1. **Integration Test** — Works with other system pieces (gap detection → adjustment → corrected pct →
   report line)
2. **Manual Test** — A developer can trace the PVD fixture through `_evaluate_pnl` to the rendered
   report line and see the corrected number + annotation
3. **Data Verification** — Not applicable to this plan (zero schema/table changes; PnL is computed live
   at read time, never persisted — see Non-Goals)
4. **Error Handling** — The evaluator's existing never-raise contract is preserved; a malformed/missing
   `adjustment_factor` degrades the report annotation to `""`, never a crash
5. **User Confirmation** — Automated tests are the primary evidence bar for this plan (see Verification
   Evidence) — the live-broadcast confirmation of the corrected number is already owned by the two
   sibling plans' own pending live-cron gates, not duplicated here

Status meanings:
- ⏳ PLANNED — Not started
- 🔨 CODE DONE — Written but not E2E tested
- 🧪 TESTING — Currently being tested
- ✅ VERIFIED — Tested AND confirmed working
- 🚧 BLOCKED — Has issues

---

## Execution Brief

### Phase 1: `src/data/price_lookup.py` — new pure adjustment-factor function
**What happens:** Add `derive_ca_adjustment_factor(closes, max_session_move=0.10) -> float` — a pure
sibling of the already-shipped `has_ca_gap`, operating on the SAME ordered `closes` list and the SAME
default threshold. Walks consecutive pairs; for every pair whose move exceeds the threshold, multiplies
a running cumulative factor by `cur/prev`. Returns `1.0` (identity) when no gap is found. Zero changes
to `has_ca_gap` or `closes_between` — purely additive.
**Test:** New `tests/test_price_lookup_ca_adjustment.py` — no-gap, single-gap, multi-gap composition,
degenerate inputs, custom threshold.

### Phase 2: `src/trading/signal_ledger.py` — `_evaluate_pnl` auto-adjustment retrofit
**What happens:** On `gap_flag=True`, compute `adjustment_factor` from the same `closes` list already
fetched for gap detection, rebase `t0` by that factor (`t0_eff = t0 * adjustment_factor`), and compute
`pct` from the REBASED t0 instead of the raw one. `gap_flag`'s meaning flips from "unreliable, skip
this row" to "was auto-adjusted, this number is now trusted." New `adjustment_factor: float | None` key
on the success-path return dict. Docstrings for `_evaluate_pnl`, `evaluate_signal_pnl`, and
`evaluate_regret_pnl` rewritten to describe the new contract. Zero signature changes.
**Test:** Retrofit + extend `tests/test_signal_ledger.py::TestEvaluateSignalPnl` /
`TestEvaluateRegretPnl` — PVD reproduction now asserts the CORRECTED pct, new multi-gap composition
test, new window-boundary tests.

### Phase 3: `src/reports/builders.py` — trusted-number annotation retrofit
**What happens:** Remove the `_GAP_WARNING_LINE` constant and its "replace the pct clause with a
warning" branches in all 4 line renderers (`_position_open_line`, `_position_closed_line`,
`_regret_open_line`, `_regret_closed_line`). Add one shared `_gap_adjustment_suffix(r) -> str` helper
that APPENDS `" (⚙️ đã tự động điều chỉnh sự kiện DN, hệ số {factor:.3f})"` to the existing lãi/lỗ
clause instead of replacing it — the number is now shown, not hidden.
**Test:** Rewrite the 2 existing gap-flag tests in `tests/test_position_report_builder.py` and the 2 in
`tests/test_regret_report_builder.py` to assert the new annotated-and-trusted format.

### Phase 4: Full-suite verification
**What happens:** Run the complete test suite; confirm zero regressions against the 758-test baseline.
**Test:** `pytest -q` — all green.

### Expected Outcome
- Every dispatched-signal or cancelled-signal PnL evaluation that crosses a corporate-action price reset
  (stock dividend, split, rights issue) now reports the ECONOMICALLY CORRECT return, not a phantom
  −40%-class number hidden behind a warning.
- The PVD case, re-run with this fix, yields NET pct ≈ −0.30% (was a hidden fake ≈ −41.3%).
- `main.py` requires ZERO code changes — the existing `{**r, **pnl}` enrichment merge in
  `notify_position_report`/`notify_regret_report` already forwards the new `adjustment_factor` key
  automatically (verified by reading the current code — see Data Flow).
- `has_ca_gap`, `closes_between`, `portfolio_guard.py`, and `main._backfill_paperlog_outcomes` are
  completely untouched — this plan is purely additive at the `price_lookup.py` level.

---

## Non-Goals and Constraints

**Out of scope for this plan:**
- No external corporate-action data source / API integration (explicitly rejected by the user — no
  clean free VN data source exists; SSI/VietStock/CafeF only publish free-text disclosures).
- No manual per-event data entry command (a `/adjust TICKER DATE RATIO` command was proposed and
  explicitly rejected — "i dont have time to track whether it adjust or not").
- No new corporate-action DETECTION algorithm — this plan reuses the exact same
  `price_lookup.has_ca_gap` threshold heuristic (default 10%) already shipped and tested; only the
  MAGNITUDE-of-correction step is new.
- No new config kill-switch for this feature. It extends an already-unconditional, already-shipped gap
  guard (`_evaluate_pnl` calls `has_ca_gap` unconditionally today, no flag gates it) — adding a toggle
  would let the fake-PnL bug this plan fixes silently come back, which serves no operating mode anyone
  asked for (YAGNI; see Decision 6).
- No exchange-classification data source is added. Confirmed via a full-repo search: the only
  ticker-level classification anywhere in this codebase is `main._VN30_UNIVERSE` (a curated ~30-name
  large-cap INDEX membership list, refreshed quarterly, NOT an exchange listing map) and the dynamic
  ADV-top-N serve universe — neither tells you whether a given ticker trades on HOSE, HNX, or UPCOM. See
  Decision 7 / Open Question 1.
- No retroactive backfill of any stored data. `pct`/`gap_flag`/`adjustment_factor` are ALL computed live
  at read time by `evaluate_signal_pnl`/`evaluate_regret_pnl` — nothing is persisted to
  `dispatched_signals` or `cancelled_signals`. The still-open PVD row (if any) is automatically corrected
  the very next time `notify_position_report` re-evaluates it — zero backfill script needed, same
  zero-backfill property the prior plan already established for the "detect and flag" half of this fix.
- No change to `has_ca_gap` or `closes_between`'s signatures, defaults, or behavior (Decision 1).
- No change to `main.py`, `config/settings.py`, `src/utils/telegram_bot.py`,
  `src/trading/portfolio_guard.py`, or `src/utils/audit_evaluator.py`.
- No dashboard (`dashboard/`) changes.

**Constraints:**
- `evaluate_signal_pnl(ticker, dispatch_date, hold_days, today=None)` and
  `evaluate_regret_pnl(ticker, screen_date, hold_days, today=None)` signatures MUST stay unchanged — only
  their success-path return dict gains one new key.
- `t0` and `t_exit` in the returned dict stay the RAW, as-quoted prices (unadjusted) — for audit/debug
  transparency. The caller reconstructs the effective rebased entry price as `t0 * adjustment_factor`
  if it ever needs to (no current consumer does — see Decision 4).
- `gap_flag` and `adjustment_factor` must NEVER disagree about which sessions counted as a gap — both
  are derived from the exact same `closes` list and the exact same default `max_session_move=0.10`, with
  no per-call threshold override at either call site.
- Every test touching `_evaluate_pnl`/`evaluate_signal_pnl`/`evaluate_regret_pnl` must monkeypatch
  `signal_ledger.price_lookup.*` — zero real parquet I/O (repo convention, already established by the
  prior plan's own retrofit).
- Must preserve the whole call chain's never-raise contract.

---

## Architecture Decisions (Final)

### Decision 1 — New pure sibling function `price_lookup.derive_ca_adjustment_factor`; do NOT change `has_ca_gap`/`closes_between`'s signatures, and do NOT add a signal_ledger-local helper

**Options considered:**
1. Extend `has_ca_gap`'s return shape (bool → tuple/dict including a factor). **Rejected** —
   `has_ca_gap` is called independently by `portfolio_guard.evaluate_position` and
   `main._backfill_paperlog_outcomes` (confirmed by reading both call sites), both of which use it as a
   plain boolean gate with zero interest in an adjustment factor. Changing its return shape would force
   both callers to update, for no benefit to either.
2. Add an optional param to `closes_between` that also returns adjustment ratios. **Rejected** —
   `closes_between` is a raw price-fetch function; conflating it with adjustment-factor math violates
   single-responsibility and, same as option 1, forces two unrelated callers to adapt to a widened
   return shape they never asked for.
3. **Chosen: add `price_lookup.derive_ca_adjustment_factor(closes: list[float], max_session_move: float
   = 0.10) -> float`** — a pure sibling of `has_ca_gap`, taking the exact SAME `closes: list[float]`
   input (no ticker/date/conn params needed — `_evaluate_pnl` already fetches `closes` once via
   `closes_between` for the existing gap check; this function reuses that SAME list with **zero new I/O**).
   Placed immediately after `has_ca_gap` in `price_lookup.py`. `has_ca_gap`/`closes_between` are touched
   **zero lines** — 100% additive.

**Why not a `signal_ledger`-local private helper instead of a `price_lookup` public function?** The
function is a pure numeric transform on a price-close list with no ledger/DB concept — it belongs beside
its sibling `has_ca_gap` in `price_lookup.py` for discoverability (a future caller wanting "the CA
adjustment factor for this closes list" should find it next to "does this closes list have a CA gap,"
not buried in a different module).

**Why return a single cumulative `float` instead of a list of per-event `(date, ratio)` tuples?**
`_evaluate_pnl` only ever needs the CUMULATIVE product to rebase one `t0` value — no requirement in this
plan surfaces individual per-event ratios anywhere (the report renders ONE combined factor per row, not
a breakdown). A single float is simpler, trivially composes via one multiplication at the call site, and
avoids inventing an unused data structure (YAGNI). If a future need arises to show individual events,
that is a clean, additive future extension.

### Decision 2 — Rebase T0 (`t0_eff = t0_raw * cumulative_factor`), NOT rescale-the-return-formula

**The two mathematically distinct options:**
- **(a) Rebase T0**: `t0_eff = t0 * factor`; `pct = (t_exit - t0_eff) / t0_eff * 100 - cost_pct`.
- **(b) Rescale the raw return**: `pct = ((t_exit - t0) / t0 * 100) / factor - cost_pct` (or some other
  post-hoc scaling of the already-computed raw percentage).

**Chosen: (a) rebase T0.** This is the standard "back-adjustment" technique (the same method a data
vendor uses to build an "adjusted close" series when no declared dividend/split ratio is published):
every price BEFORE the corporate-action reset is scaled onto the SAME price basis as prices AFTER it, so
the return between any two points on the adjusted series reflects the shareholder's true economic
outcome (extra shares received exactly offset the ex-rights price drop, assuming CA-neutral economics —
the standard assumption for this technique). Option (b) is NOT mathematically equivalent to (a) in the
general case (they only coincide in the degenerate 2-point case where `t0` and `t_exit` are the exact
two endpoints of the gap itself) and has no standard financial interpretation once there is real organic
price movement on either side of the gap — rejected.

**Worked numeric example (multi-session, NOT the trivial 2-point PVD case — proves the formula handles
real organic movement correctly, not just "always returns ~0%"):**

Ticker XYZ, `t0 = 100.0` (day 0), a stock-dividend reset drops the close from 100.0 → 60.0 the very next
session (factor `0.6`), price drifts organically 60.0 → 63.0 (+5%, no gap), a second, unrelated
stock-dividend reset drops 63.0 → 37.8 (factor `0.6` again), matured `t_exit = 37.8`.

- Raw (broken) calculation: `pct_raw = (37.8 - 100.0) / 100.0 * 100 = -62.2%` — looks like a
  catastrophic loss.
- `closes = [100.0, 60.0, 63.0, 37.8]`; `derive_ca_adjustment_factor(closes)`:
  - pair `(100.0, 60.0)`: `|60/100 - 1| = 0.40 > 0.10` → gap, running factor `*= 0.6` → `0.6`
  - pair `(60.0, 63.0)`: `|63/60 - 1| = 0.05` → not a gap, unchanged
  - pair `(63.0, 37.8)`: `|37.8/63 - 1| = 0.40 > 0.10` → gap, running factor `*= 0.6` → `0.36`
  - `adjustment_factor = 0.36`
- `t0_eff = 100.0 * 0.36 = 36.0`
- `adjusted_pct (gross) = (37.8 - 36.0) / 36.0 * 100 = 5.0%`
- `NET pct = 5.0 - 0.30 = 4.70%`

**Sanity check (shares-multiplier cross-check, independent derivation):** Each CA event with factor `f`
means a holder's SHARE COUNT multiplies by `1/f` (the per-share price drop is exactly offset by receiving
more shares — no wealth created or destroyed by the CA itself). After both events, 1 original share has
become `1/0.6 * 1/0.6 = 2.7778` shares. Portfolio value at `t_exit`: `2.7778 * 37.8 = 105.0`. True return
on the original `100.0` investment: `(105.0 - 100.0) / 100.0 * 100 = 5.0%` — **matches the rebase-T0
formula exactly.** This confirms rebase-T0 is not just convenient but mathematically the economically
correct answer, and demonstrably NOT "always ≈ 0%" (this example shows a real +5% swing where the raw
number showed a fake −62.2%).

**PVD regression case (the trivial 2-point special case, included because item 6 of this plan's scope
requires it explicitly):** `t0 = 33.3`, `t_exit = 19.47` (exit falls immediately after the single gap).
`closes = [33.3, 19.47]`; `adjustment_factor = 19.47 / 33.3 = 0.584684...`; `t0_eff = 33.3 * 0.584684... =
19.47` (exact, by construction, since the factor was derived from these exact two points).
`adjusted_pct (gross) = (19.47 - 19.47) / 19.47 * 100 = 0.0%`. `NET pct = 0.0 - 0.30 = -0.30%` (the
round-trip transaction cost is the only remaining component — the underlying price move nets to flat once
adjusted, matching the intuition that a stock-dividend ex-rights reset by itself creates no real gain or
loss).

**No double-counting against the round-trip cost:** `cost_pct` is a flat percentage-point subtraction
applied ONCE, at the very end, identically regardless of whether a gap fired — `pct = adjusted_pct -
cost_pct`. The cost model has no price-scale dependency, so rebasing `t0` cannot interact with or
double-deduct it.

**Defensive guard:** if `adjustment_factor` is somehow `0` or produces a non-positive `t0_eff` (only
possible with corrupted parquet data — a literal `0.0` or negative close), fall back to the RAW `t0`
rather than dividing by zero / producing a nonsensical negative-basis return. This never fires on real
data (closes are always positive for a listed, trading ticker) but is included as a defensive guard
consistent with the module's existing never-raise philosophy.

### Decision 3 — Cumulative-product composition for multi-gap hold windows

**Rationale:** A T+20 hold window (30 sessions) can span more than one corporate action.
`derive_ca_adjustment_factor` composes ALL detected gaps inside the given `closes` window as a running
product, in date order (the list is already ordered ascending by date via `closes_between`'s own `ORDER
BY date ASC`). This is a natural consequence of the pairwise-scan loop (Decision 1/2's worked example
above demonstrates 2 composed gaps) — no special-casing is needed for "more than one gap," it falls out
of the same loop that handles zero or one.

**Window scoping (why a gap outside `(t0_date, t_exit_date]` cannot be picked up):** `closes` is always
exactly `price_lookup.closes_between(ticker, entry_date, window_end)` where `window_end = exit_date` when
matured, else `today`. Since `derive_ca_adjustment_factor` only ever sees the elements of THIS list, a
corporate action dated before `entry_date` or after `window_end` structurally cannot appear in `closes`
and therefore cannot contribute to the factor — no additional filtering code is required. This plan adds
regression tests (Phase 2, Implementation Checklist) that capture the exact `(start, end)` arguments
passed to `closes_between` to lock this boundary in place.

### Decision 4 — `gap_flag`'s semantic redefinition; `t0`/`t_exit` stay RAW in the return dict

`gap_flag` now means **"this row's `pct` was auto-adjusted for a detected corporate-action reset"** —
NOT "this row is unreliable, do not trust it" (the pre-this-plan meaning). The returned `pct` is always
the TRUSTED, final number (adjusted when `gap_flag=True`, raw when `False`) — the evaluator never again
returns a known-phantom number for the caller/report to hide.

`t0` and `t_exit` in the return dict remain the RAW, as-observed close prices (not the rebased `t0_eff`)
— useful for audit/debugging ("what were the actual quoted prices"), and no current consumer
(`build_position_report`, `build_regret_report`) reads `t0`/`t_exit` at all today (confirmed by reading
both — they only read `pct`, `gap_flag`, `matured`, `sessions_remaining`, `ticker`, `horizon`,
`dispatch_date`/`screen_date`). A caller that ever needs the effective rebased entry price can compute
`t0 * adjustment_factor` itself. See Open Question 2 for the explicit confirm-or-override framing on
this API-shape choice.

### Decision 5 — Report layer: APPEND the annotation, never REPLACE the clause; one shared helper, not 4 duplicated branches

The pre-this-plan code in `src/reports/builders.py` has 4 near-identical `if r.get("gap_flag"): <replace
the pct clause with _GAP_WARNING_LINE>` branches across `_position_open_line`, `_position_closed_line`,
`_regret_open_line`, `_regret_closed_line`. This plan removes the `_GAP_WARNING_LINE` constant entirely
and replaces those 4 branches with ONE shared `_gap_adjustment_suffix(r: dict) -> str` helper, called
identically from all 4 render functions. This is both the correct behavior change (per the user's
explicit instruction: "append... instead of replacing it — the number should now be TRUSTED and shown")
and a DRY cleanup (4 duplicated warning-branches → 1 shared suffix helper + 4 one-line clause changes).

**Exact annotation text:** `f" (⚙️ đã tự động điều chỉnh sự kiện DN, hệ số {factor:.3f})"` — a single
leading space so it concatenates directly onto the existing `{verb} {abs(pct):.1f}%` clause with no
double-space and no separate sentence.

### Decision 6 — No new config kill-switch

This plan extends an already-unconditional, already-shipped guard (`_evaluate_pnl` calls `has_ca_gap`
today with no config gate anywhere in the call chain). Adding a toggle to let a user disable the
correction would mean choosing to keep showing a KNOWN-fake number — no operating mode benefits from
that, and no part of this session's requirements asked for one (YAGNI, consistent with the "Architecture
style: pure functions + procedural orchestration" and existing minimal-config-surface convention for
`has_ca_gap`'s own threshold, which is also not config-exposed).

### Decision 7 — Exchange-scoping Open Question: recommend UNIVERSAL application (no HOSE/HNX-only scoping)

**Research finding (confirmed via a full-repo search for `HOSE|HNX|UPCOM|exchange|universe_filter|VN30`
across all `.py` files):** No ticker-to-exchange classification data source exists anywhere in this
codebase. The only ticker-level classification is `main._VN30_UNIVERSE` — a curated ~30-name **large-cap
INDEX membership** frozenset (all HOSE, by definition, but covers only the top-30 names, not the full
serve universe which can include ~50-350+ tickers via the dynamic ADV-top-N universe or VN30-fallback
mode), and it says nothing about HNX/UPCOM listing status for the tickers it excludes. `config/settings.py`'s
dead `universe_filter` JSON block (already noted elsewhere as dead/unused) also carries no
exchange-membership data. Building a real HOSE/HNX/UPCOM mapping would require a NEW data source — the
same category of thing (an external structured feed) the user explicitly rejected acquiring for this
feature.

**Recommendation: apply the auto-adjustment UNIVERSALLY (no exchange scoping), same as `has_ca_gap`'s own
existing precedent** (it already applies its 10% threshold to every ticker regardless of listing venue,
with zero exchange-awareness, and has been running that way in production since 2026-07-12 with no
reported false positives). Consistency argument: introducing exchange-scoping HERE while `has_ca_gap`
itself stays universal would create two different rules for what "counts as a gap" depending on which
of the two call sites you're looking at — a worse outcome than accepting the documented caveat.

**Accepted, explicitly documented caveat (NOT silently hidden):** UPCOM-listed tickers have a wider
±15% daily limit. A genuine (non-CA) organic UPCOM move between 10% and 15% would be misclassified as a
corporate action and incorrectly auto-adjusted. This is documented as a code comment attached to the new
`derive_ca_adjustment_factor` function (Implementation Checklist item 3) — a comment, not a runtime
check, per the explicit instruction that a real fix would require a new data source this plan does not
introduce. **This is Open Question 1 below — the user should explicitly confirm this default before
EXECUTE**, per the standard PLAN-review checkpoint protocol.

---

## Data Flow

```
signal_ledger._evaluate_pnl(ticker, entry_date, hold_days, today, cost_pct)
  │
  ├─ (unchanged) t0 = close_on_or_before(entry_date); matured branching → t_exit, matured
  ├─ (unchanged) t0 is None / t0 <= 0 → early "error" return (gap/adjustment logic never runs)
  │
  ├─ window_end = exit_date if matured else today                          [unchanged]
  ├─ closes = price_lookup.closes_between(ticker, entry_date, window_end)  [unchanged call]
  ├─ gap_flag = price_lookup.has_ca_gap(closes)                            [unchanged call]
  │
  ├─ NEW: adjustment_factor = price_lookup.derive_ca_adjustment_factor(closes) if gap_flag else None
  ├─ NEW: t0_eff = t0 * adjustment_factor  (guarded: falls back to raw t0 if the product is <= 0)
  ├─ CHANGED: pct = (t_exit - t0_eff) / t0_eff * 100.0 - cost_pct   (was: always used raw t0)
  │
  └─ return {"ticker", "pct", "matured", "t0", "t_exit", "gap_flag", "adjustment_factor"}  [+1 key]
        │
        ├─ evaluate_signal_pnl(...)  → cost_pct = _VN_ROUND_TRIP_COST_PCT (NET)   [signature unchanged]
        └─ evaluate_regret_pnl(...)  → cost_pct = 0.0 (GROSS)                     [signature unchanged]
              │
              ▼
   main.notify_position_report() / main.notify_regret_report()   ── ZERO CODE CHANGES ──
        │  (both already do `rows.append({**r, **pnl})` — the new `adjustment_factor` key
        │   flows through this existing merge automatically; confirmed by reading the
        │   current main.py, lines ~2459-2468 and ~2511-2518)
        ▼
   src/reports/builders.py
        ├─ _position_open_line(r) / _position_closed_line(r)   [retrofit: append, don't replace]
        └─ _regret_open_line(r) / _regret_closed_line(r)         [retrofit: append, don't replace]
              │  via shared _gap_adjustment_suffix(r) helper (NEW)
              ▼
   "... hiện đang lỗ 0.3% (⚙️ đã tự động điều chỉnh sự kiện DN, hệ số 0.585) (còn N ngày...)"
```

**No new I/O anywhere in this plan.** `closes_between` is already called once per `_evaluate_pnl`
invocation (pre-existing); `derive_ca_adjustment_factor` reuses that SAME `closes` list — this plan adds
zero new database/parquet reads.

---

## Touchpoints

| File | Change type | Why |
|---|---|---|
| `src/data/price_lookup.py` | Add `derive_ca_adjustment_factor` (new pure function, additive only) | Adjustment-magnitude computation |
| `src/trading/signal_ledger.py` | Retrofit `_evaluate_pnl` (adjustment logic + new key) + docstring rewrites on `_evaluate_pnl`/`evaluate_signal_pnl`/`evaluate_regret_pnl` | Core correction |
| `src/reports/builders.py` | Remove `_GAP_WARNING_LINE`, add `_gap_adjustment_suffix`, retrofit 4 line renderers | Transparent trusted-number rendering |
| `tests/test_price_lookup_ca_adjustment.py` | New | Phase 1 pure-function coverage |
| `tests/test_signal_ledger.py` | Retrofit `TestEvaluateSignalPnl` + `TestEvaluateRegretPnl` | Phase 2 regression + new coverage |
| `tests/test_position_report_builder.py` | Rewrite 2 existing gap-flag tests | Phase 3 |
| `tests/test_regret_report_builder.py` | Rewrite 2 existing gap-flag tests | Phase 3 |

**Explicitly NOT touched:** `main.py`, `config/settings.py`, `src/utils/telegram_bot.py`,
`src/trading/portfolio_guard.py`, `src/utils/audit_evaluator.py`, `dashboard/`.

---

## Public Contracts

**Must remain compatible:**
- `signal_ledger.evaluate_signal_pnl(ticker, dispatch_date, hold_days, today=None) -> dict` — signature
  UNCHANGED. Success-path dict gains ONE new key, `"adjustment_factor": float | None`. The MEANING of
  `pct` changes on gap rows (was a known-phantom raw move, is now the corrected/trusted value) — this is
  the plan's whole intentional purpose, documented in the function's own rewritten docstring, not a
  silent breaking change. Error-path dict (`{"ticker", "error"}`) is completely unchanged.
- `signal_ledger.evaluate_regret_pnl(ticker, screen_date, hold_days, today=None) -> dict` — same
  treatment.
- `price_lookup.has_ca_gap(closes, max_session_move=0.10) -> bool` and
  `price_lookup.closes_between(ticker, start_date, end_date, conn=None) -> list[float]` — COMPLETELY
  UNTOUCHED (zero lines changed). Safe for `portfolio_guard.evaluate_position` and
  `main._backfill_paperlog_outcomes`, both of which call these functions independently for unrelated
  purposes.
- `src/reports/builders.build_position_report(open_rows, closed_rows, today, lookback_days) -> str` and
  `build_regret_report(cancelled_rows, today, lookback_days) -> str` — signatures UNCHANGED; only the
  internal rendering of `gap_flag=True` rows changes (append vs replace).
- `main.py` — requires literally zero code changes (verified: the existing `{**r, **pnl}` enrichment
  pattern in both `notify_position_report._enrich` and `notify_regret_report` already forwards any new
  key in the `pnl` dict with no caller-side update needed).

**New public surface introduced:**
- `price_lookup.derive_ca_adjustment_factor(closes: list[float], max_session_move: float = 0.10) ->
  float`

---

## Blast Radius

**Directly touched:** `src/data/price_lookup.py`, `src/trading/signal_ledger.py`,
`src/reports/builders.py`, plus the 4 test files listed in Touchpoints.

**Indirectly affected (no code change, behavior corrects — a further user-visible text/number change to
an already-shipped-today Telegram message):**
- `notify_position_report()`'s EOD broadcast — any currently-open or closed-today row that crossed a
  corporate-action gap now renders the CORRECTED trusted number with a visible annotation, instead of
  today's shipped "⚠️ giá bất thường" warning line.
- `notify_regret_report()`'s EOD broadcast — same, for cancelled/hypothetical signals.
- `/audit_weekly`'s engine-picks section is unaffected (it uses `audit_evaluator`'s own independent
  `_evaluate_dispatched_signal`, not this module's `_evaluate_pnl` — per the parent plan's Decision 4,
  the two remain intentionally un-refactored/duplicated).

**Explicitly NOT touched:**
- `src/data/price_lookup.py::has_ca_gap` / `closes_between` — zero lines changed.
- `src/trading/portfolio_guard.py` — read-only precedent reference only, zero edits.
- `main._backfill_paperlog_outcomes` — zero edits, uses `has_ca_gap`/`closes_between` for an unrelated
  purpose (paperlog return-window exclusion, not PnL correction).
- `src/utils/audit_evaluator.py` — zero edits (standing, previously-flagged backlog item, not touched
  again here).
- `PortfolioManager` / real sizing / NAV logic / `dispatched_signals` and `cancelled_signals` table
  schemas — untouched (no new columns, no new tables).
- `dashboard/` — no changes.

---

## Implementation Checklist

### Phase 1 — `src/data/price_lookup.py`: new pure adjustment-factor function

1. Immediately after `has_ca_gap`'s definition (before `latest_close`), add:
   ```python
   # ---------------------------------------------------------------------------
   # Corporate-action auto-adjustment (sibling of has_ca_gap — same threshold,
   # same input list, zero new I/O). UPCOM tickers have a wider ±15% daily limit
   # than HOSE/HNX (~±7%/±10%), so a genuine (non-CA) 10-15% UPCOM move can be
   # misclassified as a corporate action and incorrectly auto-adjusted. No
   # exchange-classification data source exists anywhere in this codebase
   # (confirmed by repo search) to scope this to HOSE/HNX only without adding a
   # new external data source — explicitly out of scope (see the plan's
   # Decision 7). Applied universally, same precedent as has_ca_gap itself.
   # ---------------------------------------------------------------------------
   def derive_ca_adjustment_factor(closes: list[float], max_session_move: float = 0.10) -> float:
       """Cumulative back-adjustment factor across every corporate-action gap in
       ``closes``.

       For every consecutive close-to-close move whose magnitude exceeds
       ``max_session_move`` (the SAME VN daily-limit heuristic behind
       ``has_ca_gap`` — a split / stock dividend / rights issue, not price
       action), multiplies a running factor by ``cur / prev``. Two or more gaps
       inside the window compose as a cumulative product, in date order (a hold
       window can span more than one corporate action). Mirrors ``has_ca_gap``'s
       ``prev > 0`` guard so a corrupted zero/negative close is skipped rather
       than raising a ZeroDivisionError.

       Returns ``1.0`` (identity — "no adjustment needed") when no gap is found
       or ``closes`` has fewer than 2 points. Callers determine "is there a gap
       at all" via ``has_ca_gap(closes)`` first; this function only supplies the
       MAGNITUDE of the correction once a gap is already known to exist.
       """
       factor = 1.0
       for prev, cur in zip(closes, closes[1:]):
           if prev > 0 and abs(cur / prev - 1.0) > max_session_move:
               factor *= cur / prev
       return factor
   ```
2. Create `tests/test_price_lookup_ca_adjustment.py`:
   - `test_no_gap_returns_one` — `[100.0, 105.0, 101.0, 108.0]` (all moves < 10%) → `1.0`.
   - `test_single_gap_returns_ratio` — `[100.0, 60.0, 63.0]` (first pair gaps at −40%, second pair is a
     +5% organic move) → `pytest.approx(0.6)`.
   - `test_multi_gap_composes_product_in_date_order` — `[100.0, 60.0, 63.0, 37.8]` (two gaps, one
     organic move between them) → `pytest.approx(0.36)` (matches the plan's Decision 2 worked example
     exactly).
   - `test_empty_and_single_element_returns_one` — `[]` → `1.0`; `[100.0]` → `1.0`.
   - `test_zero_or_negative_prev_skipped` — `[0.0, 50.0]` → `1.0`; `[-10.0, 50.0]` → `1.0` (mirrors
     `has_ca_gap`'s own `prev > 0` guard, no `ZeroDivisionError`).
   - `test_custom_threshold_param` — `[100.0, 94.0]` (a −6% move): default `max_session_move=0.10` →
     `1.0` (no gap); `max_session_move=0.05` → `pytest.approx(0.94)` (now a gap).
3. Run `pytest tests/test_price_lookup_ca_adjustment.py -q` — all green.

### Phase 2 — `src/trading/signal_ledger.py`: `_evaluate_pnl` auto-adjustment retrofit

4. In `_evaluate_pnl`, replace the existing tail (currently, verbatim, after the `t0 <= 0` error-return
   guard):
   ```python
   # Corporate-action gap guard — reuse existing infra, default 10% threshold.
   window_end = exit_date if matured else today
   closes = price_lookup.closes_between(ticker, entry_date, window_end)
   gap_flag = price_lookup.has_ca_gap(closes)

   pct = (t_exit - t0) / t0 * 100.0 - cost_pct
   return {"ticker": ticker, "pct": pct, "matured": matured,
           "t0": t0, "t_exit": t_exit, "gap_flag": gap_flag}
   ```
   with:
   ```python
   # Corporate-action gap guard + auto-adjustment — reuse existing gap-detection
   # infra (has_ca_gap), plus derive_ca_adjustment_factor for the correction
   # magnitude. Both are called against the SAME `closes` list and the SAME
   # default 10% threshold so gap_flag and adjustment_factor can never disagree
   # about which sessions counted as a corporate-action reset (plan Decision 2/3).
   window_end = exit_date if matured else today
   closes = price_lookup.closes_between(ticker, entry_date, window_end)
   gap_flag = price_lookup.has_ca_gap(closes)
   adjustment_factor = (
       price_lookup.derive_ca_adjustment_factor(closes) if gap_flag else None
   )

   # Rebase T0 onto the post-event price scale (standard back-adjustment
   # technique — see the plan's Architecture Decision 2 for the full derivation
   # + worked example). Defensive fallback to the raw t0 if the rebased value is
   # non-positive (only reachable with corrupted parquet data).
   t0_eff = t0
   if gap_flag and adjustment_factor and t0 * adjustment_factor > 0:
       t0_eff = t0 * adjustment_factor
   pct = (t_exit - t0_eff) / t0_eff * 100.0 - cost_pct
   return {"ticker": ticker, "pct": pct, "matured": matured,
           "t0": t0, "t_exit": t_exit, "gap_flag": gap_flag,
           "adjustment_factor": adjustment_factor}
   ```
5. Rewrite `_evaluate_pnl`'s docstring paragraph describing the corporate-action guard (currently states
   "The number is STILL computed and returned — the guard flags it as unreliable... the caller/report
   owns how a gap-flagged row is rendered") to instead state: the guard now AUTO-ADJUSTS `pct` by
   rebasing T0 via `derive_ca_adjustment_factor` whenever a gap fires; `gap_flag=True` now means "this
   row's `pct` was auto-adjusted," not "unreliable, do not show"; the returned dict gains
   `"adjustment_factor"` (the applied factor, or `None` when no gap fired).
6. Rewrite `evaluate_signal_pnl`'s docstring paragraph (currently ends with "...`gap_flag` now marks
   such a row unreliable while STILL returning the (phantom) numeric `pct` so the caller, not the
   evaluator, owns the render decision") — replace with the corrected framing: the PVD 2026-07-09→07-10
   incident (33.3→19.47) is now AUTOMATICALLY CORRECTED (NET pct ≈ −0.30%, was a hidden fake ≈ −41.3%)
   by rebasing T0 via the observed gap ratio; document the new `"adjustment_factor"` key.
7. Rewrite `evaluate_regret_pnl`'s docstring sentence ("...a cancelled ticker that later had a split /
   stock dividend is flagged unreliable the same way a tracked dispatch is") to match the new "auto-
   adjusted, trusted" framing.
8. Retrofit `tests/test_signal_ledger.py::TestEvaluateSignalPnl`:
   - `test_open_provisional` — add `assert out["adjustment_factor"] is None`.
   - `test_matured_exit_price` — add `assert out["adjustment_factor"] is None`.
   - `test_missing_price_returns_error` — add `assert "adjustment_factor" not in out` (error path
     returns before the gap/adjustment computation, unchanged early-return contract).
   - `test_t0_zero_or_negative_returns_error` — same addition.
   - `test_gap_flag_true_reproduces_pvd_case` — REWRITE the final assertions block. Keep the exact
     existing fixture (`d0=2026-07-09`, `prices={d0: 33.3, exit_date: 19.47}`,
     `closes_between` mocked to `lambda t, s, e, conn=None: [33.3, 19.47]`, `hold_days=3`,
     `today=sessions[4]`). Replace:
     ```python
     assert out["gap_flag"] is True
     assert out["matured"] is True
     # The (phantom) number is still present and numerically computed.
     assert isinstance(out["pct"], float)
     assert out["pct"] == pytest.approx((19.47 - 33.3) / 33.3 * 100.0 - 0.30)
     ```
     with:
     ```python
     assert out["gap_flag"] is True
     assert out["matured"] is True
     assert out["adjustment_factor"] == pytest.approx(19.47 / 33.3)
     # Corrected — near zero (the round-trip cost only), NOT the pre-fix fake
     # -41.3%/-41.8% "loss". t0 is rebased to 33.3 * (19.47/33.3) == 19.47, so
     # the underlying move nets to flat once adjusted.
     assert out["pct"] == pytest.approx(-0.30)
     ```
   - `test_gap_flag_false_normal_move` — add `assert out.get("adjustment_factor") is None`.
   - Add `test_multi_gap_composition` — reuse the `_calendar`/monkeypatch pattern; `d0 = date(2026, 6,
     1)`, `sessions = _calendar(d0, 5)`, `exit_date = sessions[2]`, `hold_days=3`,
     `close_on_or_before` → `{d0: 100.0, exit_date: 37.8}[d]`, `latest_close` → `999.0` (must not be
     used), `closes_between` → `lambda t, s, e, conn=None: [100.0, 60.0, 63.0, 37.8]`. Assert
     `out["gap_flag"] is True`, `out["adjustment_factor"] == pytest.approx(0.36)`,
     `out["pct"] == pytest.approx(4.70)` (matches the plan's Decision 2 worked example exactly:
     `t0_eff=36.0`, gross `5.0%`, NET `4.70%`).
   - Add `test_window_bounds_scoped_to_matured_exit_date` — capture the args `closes_between` is called
     with; `d0 = date(2026, 6, 1)`, `sessions = _calendar(d0, 5)`, `exit_date = sessions[2]`,
     `hold_days=3`, `today=sessions[4]`; a capturing stub
     `def _cb(t, s, e, conn=None): calls.append((s, e)); return [100.0, 101.0, 102.0, 103.0]`
     (flat, no gap). Assert `calls == [(d0, exit_date)]` (NOT `(d0, today)`) and
     `out["gap_flag"] is False` / `out["adjustment_factor"] is None`. Proves a gap dated AFTER the true
     exit (but before `today`) is structurally excluded from the matured window.
   - Add `test_window_bounds_scoped_to_today_when_open` — same capture pattern, but
     `today=sessions[1]` (< hold_days=5, provisional/open path). Assert `calls == [(d0, sessions[1])]`.
9. Retrofit `tests/test_signal_ledger.py::TestEvaluateRegretPnl`:
   - `test_open_provisional` — add `assert out["adjustment_factor"] is None`.
   - `test_matured_exit_price` — add `assert out["adjustment_factor"] is None`.
   - `test_gap_flag_propagates` — currently only asserts `out["gap_flag"] is True` and
     `isinstance(out["pct"], float)` (loose). Tighten to the same rigor as the NET-side PVD test: add
     `assert out["adjustment_factor"] == pytest.approx(19.47 / 33.3)` and
     `assert out["pct"] == pytest.approx(0.0)` (GROSS — no cost deduction, so the corrected pct lands
     exactly at 0.0, not −0.30 like the NET side).
10. Run `pytest tests/test_signal_ledger.py -q` — all green; zero regressions to `TestRecordDispatch`,
    `TestExitsDue`, `TestListOpen`, `TestDualHorizonDispatch`, `TestListClosedSince`,
    `TestCancelledSignalsLedger`, `TestListCancelledSince` (all untouched by this phase).

### Phase 3 — `src/reports/builders.py`: trusted-number annotation retrofit

11. Remove the `_GAP_WARNING_LINE` constant (lines ~142-149: the
    `_GAP_WARNING_LINE = ("⚠️ giá bất thường (nghi sự kiện DN — cổ tức/chia tách) — số liệu không đáng
    tin")` block and its preceding comment). Replace with a new shared helper in the same location:
    ```python
    def _gap_adjustment_suffix(r: dict) -> str:
        """Transparent auto-adjustment annotation appended (never replacing) the
        trusted, corrected lãi/lỗ clause on any row whose evaluator's shared
        corporate-action gap guard fired (``gap_flag=True`` —
        ``signal_ledger._evaluate_pnl``). ``pct`` is ALREADY the corrected,
        trusted number (T0 rebased by ``adjustment_factor``) by the time it
        reaches this module; this note only discloses WHY. Returns ``""`` when
        ``gap_flag`` is falsy or ``adjustment_factor`` is missing (defensive —
        production rows always carry both together, computed live at read
        time)."""
        if not r.get("gap_flag"):
            return ""
        factor = r.get("adjustment_factor")
        if factor is None:
            return ""
        return f" (⚙️ đã tự động điều chỉnh sự kiện DN, hệ số {factor:.3f})"
    ```
12. Retrofit `_position_open_line`: replace
    `clause = _GAP_WARNING_LINE if r.get("gap_flag") else f"{verb} {abs(pct):.1f}%"`
    with
    `clause = f"{verb} {abs(pct):.1f}%{_gap_adjustment_suffix(r)}"`
    (the rest of the function — the `hz`, `disp_str`, `rem` computation and the final f-string — is
    unchanged).
13. Retrofit `_position_closed_line`: remove the
    `if r.get("gap_flag"): return f"Mô hình {hz} — mã {ticker}: {_GAP_WARNING_LINE}"` early-return
    branch entirely. The function now always falls through to the existing
    `pct = float(r.get("pct") or 0.0)`; `verdict = ...`; `verb = ...`; return statement, with the return
    f-string changed from
    `f"Mô hình {hz} — mã {ticker}: đã dự báo {verdict} và {verb} {abs(pct):.1f}%"`
    to
    `f"Mô hình {hz} — mã {ticker}: đã dự báo {verdict} và {verb} {abs(pct):.1f}%{_gap_adjustment_suffix(r)}"`.
14. Retrofit `_regret_open_line`: replace the `if r.get("gap_flag"): outcome = _GAP_WARNING_LINE else:
    ...` block with unconditional:
    ```python
    pct = float(r.get("pct") or 0.0)
    verb = "lãi" if pct >= 0 else "lỗ"
    outcome = f"nếu mua thì hiện {verb} {abs(pct):.1f}%{_gap_adjustment_suffix(r)}"
    ```
15. Retrofit `_regret_closed_line`: same pattern:
    ```python
    pct = float(r.get("pct") or 0.0)
    verb = "lãi" if pct >= 0 else "lỗ"
    outcome = f"nếu mua thì đã {verb} {abs(pct):.1f}%{_gap_adjustment_suffix(r)}"
    ```
16. In `tests/test_position_report_builder.py`, rewrite the 2 gap-flag tests:
    ```python
    def test_gap_flag_open_line_renders_adjusted_annotation() -> None:
        out = build_position_report(
            [_open_row(gap_flag=True, pct=-0.30, adjustment_factor=19.47 / 33.3)],
            [], TODAY, LOOKBACK)
        assert "lỗ 0.3%" in out                                    # corrected pct IS shown (trusted)
        assert "đã tự động điều chỉnh sự kiện DN" in out
        assert "hệ số 0.585" in out
        assert "giá bất thường" not in out                         # old warning line gone
        assert "(còn 3 ngày kết thúc dự đoán)" in out               # suffix retained


    def test_gap_flag_closed_line_renders_adjusted_annotation() -> None:
        out = build_position_report(
            [], [_closed_row(gap_flag=True, pct=4.70, adjustment_factor=0.36)],
            TODAY, LOOKBACK)
        assert "đã dự báo đúng và lãi 4.7%" in out
        assert "đã tự động điều chỉnh sự kiện DN" in out
        assert "hệ số 0.360" in out
        assert "giá bất thường" not in out
        assert "Mô hình T5 — mã FPT:" in out
    ```
17. In `tests/test_regret_report_builder.py`, rewrite the 2 gap-flag tests:
    ```python
    def test_gap_flag_line_variant_open() -> None:
        out = build_regret_report(
            [_open_row(gap_flag=True, pct=-0.30, adjustment_factor=19.47 / 33.3)],
            TODAY, LOOKBACK)
        assert "lỗ 0.3%" in out
        assert "đã tự động điều chỉnh sự kiện DN" in out
        assert "hệ số 0.585" in out
        assert "giá bất thường" not in out
        assert "(còn 4 ngày)" in out                # suffix retained
        assert "P(tăng)=30%" in out                 # screening metadata still shown


    def test_gap_flag_line_variant_closed() -> None:
        out = build_regret_report(
            [_closed_row(gap_flag=True, pct=4.70, adjustment_factor=0.36)],
            TODAY, LOOKBACK)
        assert "nếu mua thì đã lãi 4.7%" in out
        assert "đã tự động điều chỉnh sự kiện DN" in out
        assert "hệ số 0.360" in out
        assert "giá bất thường" not in out
    ```
18. Run `pytest tests/test_position_report_builder.py tests/test_regret_report_builder.py -q` — all
    green.

### Phase 4 — Full-suite verification

19. Run `pytest -q` (full suite) — zero regressions against the 758-test baseline (measured live this
    session: `pytest -q`, 758 passed, 0 failed, 0 errors), all new/retrofitted tests green.
20. No live-cron confirmation gate is opened by THIS plan specifically — the underlying broadcasts
    (`notify_position_report`, `notify_regret_report`) are already governed by the two sibling plans'
    own pending live-cron gates (`eod-position-report_PLAN_16-07-26.md` item 26,
    `cancelled-signal-regret-tracking_PLAN_16-07-26.md` item 31). Recommended lightweight manual check
    (not a hard blocking gate): the next time `notify_position_report` re-evaluates PVD (if its
    dispatched row is still within its hold window at execution time), confirm the corrected number +
    "⚙️ đã tự động điều chỉnh..." annotation renders instead of the old warning line.

---

## Test Matrix

| Area | File | Type | Key scenarios |
|---|---|---|---|
| Adjustment-factor pure function | `tests/test_price_lookup_ca_adjustment.py` | Unit | no-gap, single-gap, multi-gap composition, degenerate inputs, custom threshold |
| `_evaluate_pnl` auto-adjust retrofit | `tests/test_signal_ledger.py::TestEvaluateSignalPnl` | Unit | PVD reproduction (corrected, not fake), multi-gap composition, window-boundary scoping (matured + open), `adjustment_factor is None` regression guard on all no-gap paths |
| GROSS-side propagation | `tests/test_signal_ledger.py::TestEvaluateRegretPnl` | Unit | gap-flag propagation with corrected value + factor, `adjustment_factor is None` regression guard |
| Position-report rendering | `tests/test_position_report_builder.py` | Unit (pure) | gap row shows corrected pct + annotation, old warning text absent |
| Regret-report rendering | `tests/test_regret_report_builder.py` | Unit (pure) | same, for the hypothetical/regret line variants |
| Regression | full `pytest -q` | Full suite | zero breakage across the 758-test baseline |
| Manual (non-blocking) | live cron re-evaluation of an in-window CA-affected row | E2E | recommended spot-check, not a hard gate — see item 20 |

---

## Verification Evidence

- Pre-plan baseline measured live this session: `pytest -q` → **758 passed, 0 failed, 0 errors** (repo
  root, `pytest.ini` addopts `-q --tb=short`; independently confirmed via
  `pytest --collect-only -q` summed per-file counts = 758).
- All Phase 1-3 pytest files pass locally (targeted runs per phase), then the full suite in Phase 4 —
  this is the primary automated evidence bar for this plan.
- Every new/retrofitted numeric assertion in this plan is traceable to a worked-by-hand example in
  Architecture Decision 2 (the multi-gap `0.36`/`4.70%` example and the PVD `0.585`/`-0.30%` example) —
  EXECUTE should not need to derive any expected value independently.
- No live cron confirmation is required to close THIS plan (see Implementation Checklist item 20) —
  automated test coverage is sufficient given the change is a pure-math correction with zero new I/O,
  zero new tables, and zero new config surface.

---

## Open Questions (confirm before or during EXECUTE)

1. **Universal application vs HOSE/HNX-only scoping (Decision 7).** Default in this plan: apply the
   auto-adjustment universally to every ticker, with the UPCOM ±15%-daily-limit caveat documented as a
   code comment (not a runtime check) — because no exchange-classification data source exists anywhere
   in this codebase to cheaply scope it otherwise, and universal application is consistent with
   `has_ca_gap`'s own existing, already-shipped, already-in-production precedent. **Confirm this default
   is acceptable**, or direct that a new exchange-classification data source be built first (a
   materially larger, out-of-band scope not covered by this plan).
2. **Return-dict shape: `t0`/`t_exit` stay RAW, only `adjustment_factor` is new (Decision 4).** Default:
   do NOT add a separate `t0_adjusted` key to the return dict — no current consumer needs it, and a
   caller can trivially compute `t0 * adjustment_factor` itself if a future need arises. Confirm this
   minimal API surface is acceptable, or specify that an explicit `t0_adjusted` field should also be
   returned for direct display/audit convenience.

---

## Acceptance Criteria

1. `price_lookup.derive_ca_adjustment_factor` returns `1.0` for no-gap/degenerate inputs, the correct
   single ratio for one gap, and the correct cumulative product (in date order) for 2+ gaps inside one
   window — all regression tested.
2. `_evaluate_pnl` (and both public wrappers `evaluate_signal_pnl`/`evaluate_regret_pnl`) rebase T0 by
   the cumulative adjustment factor whenever `gap_flag=True`; the returned `pct` is the corrected,
   trusted value, never the raw phantom move. `adjustment_factor` is exactly `None` on every non-gap
   success-path row and absent on every error-path row.
3. The PVD synthetic reproduction (33.3→19.47 across the single gap) yields a NET pct of
   `pytest.approx(-0.30)` (the round-trip cost only) instead of the pre-fix fake ≈ −41.3%/−41.8%, and a
   GROSS pct of `pytest.approx(0.0)`.
4. A synthetic multi-gap fixture (2 corporate-action events inside one hold window, with organic
   movement between them) proves cumulative composition — `derive_ca_adjustment_factor` and
   `_evaluate_pnl`'s resulting `pct` both match the plan's worked-by-hand example (`0.36` factor, `4.70%`
   NET).
5. `closes_between`'s window is verified (via call-argument capture) to be scoped EXACTLY to
   `(entry_date, exit_date]` when matured and `(entry_date, today]` when open/provisional — a gap dated
   outside that window structurally cannot be picked up.
6. `build_position_report` and `build_regret_report` render the corrected `pct` WITH a visible
   `"⚙️ đã tự động điều chỉnh sự kiện DN, hệ số X.XXX"` annotation on every `gap_flag=True` row; the old
   `"⚠️ giá bất thường — không đáng tin"` replacement text no longer appears anywhere in either builder's
   output.
7. Zero signature changes to `evaluate_signal_pnl`, `evaluate_regret_pnl`, `build_position_report`,
   `build_regret_report`, `has_ca_gap`, `closes_between`. `main.py` requires zero code changes (verified
   both by static reading of the existing `{**r, **pnl}` merge pattern and by the full-suite run passing
   with no `main.py` edits).
8. Full test suite (`pytest -q`) green after implementation — 758 baseline + net-new/retrofitted tests,
   zero regressions.

---

## Resume and Execution Handoff

- **Read this entire plan file before resuming** — every Architecture Decision above is load-bearing,
  especially Decision 1 (new `price_lookup` sibling function, zero changes to `has_ca_gap`/
  `closes_between`), Decision 2 (rebase-T0 formula + both worked examples — the multi-gap example is the
  one that proves correctness, the PVD example is the one this plan's scope explicitly requires), and
  Decision 7 (why exchange-scoping is not cheaply available, and the recommended universal-application
  default — **confirm Open Question 1 with the user before or during EXECUTE**).
- **This plan builds directly on top of the CURRENT, UNCOMMITTED state of `signal_ledger.py` and
  `src/reports/builders.py`**, which is itself the deliverable of
  `process/general-plans/active/cancelled-signal-regret-tracking_PLAN_16-07-26.md` (confirmed by reading
  the actual current file contents before writing this plan — `_evaluate_pnl`, `gap_flag`,
  `_GAP_WARNING_LINE`, and both report builders' warning-line branches are all already live in this
  worktree as of 17-07-26, even though that sibling plan's own status field still reads `⏳ PLANNED` — a
  stale status field, not a sign the code is missing; reconciling that status field is out of scope for
  THIS plan). **If that sibling plan's changes are ever reset/reverted before this plan's EXECUTE runs,
  the exact line references in this plan's Implementation Checklist will be stale — re-read the current
  file contents before applying any Phase 2/3 edit.**
- **Active-plan scan at plan-authoring time** (`process/general-plans/active/`): three plans present —
  `eod-position-report_PLAN_16-07-26.md` (🔨 CODE DONE, pending its own live-cron gate, unrelated code
  path to this plan's edits), `cancelled-signal-regret-tracking_PLAN_16-07-26.md` (direct predecessor,
  see above), `portfolio-guard_PLAN_13-07-26.md` (unrelated — does not touch `signal_ledger.py` or
  `src/reports/builders.py`'s position/regret report functions).
- **Phase order is a hard dependency**: Phase 1 (`derive_ca_adjustment_factor`) MUST land and pass tests
  before Phase 2 (`_evaluate_pnl` retrofit, which calls it). Phase 3 (report rendering) depends on Phase
  2's new `adjustment_factor` key existing on the enriched row dicts the tests construct.
- If EXECUTE is interrupted mid-phase, `git status`/`git diff` against the Implementation Checklist's
  numbered items to determine the last completed step; do not skip ahead past an incomplete Phase 1 or 2.
- This plan can be archived via UPDATE PROCESS once Phase 4's full-suite green run is confirmed — unlike
  its two sibling plans, it does NOT require a separate live-cron confirmation gate (see Implementation
  Checklist item 20 and Verification Evidence).

---

## Cursor + RIPER-5 Guidance

- **Cursor Plan mode**: Import the "Implementation Checklist" items directly as TODOs, execute Phase by
  Phase, run each phase's `pytest` command before moving to the next phase's checklist items.
- **RIPER-5 mode**: This file is the artifact from the PLAN phase. Say `ENTER EXECUTE MODE` to begin
  Phase 1. Before EXECUTE starts, the user should explicitly confirm Open Question 1 (universal vs
  HOSE/HNX-only scoping) — the recommended default (universal) is safe to proceed with unless the user
  overrides it. After EXECUTE completes all 4 phases and the full suite is green, this plan can move
  straight to ✅ VERIFIED / UPDATE PROCESS — no live-cron gate blocks archival for this plan specifically.
