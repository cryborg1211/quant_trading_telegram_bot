# P0 — Reuse Audit Plan

**Date:** 2026-06-19
**Status:** ACTIVE — PLANNED
**Phase:** P0 of local-dashboard program
**Complexity:** SIMPLE (research/audit task; no code written)
**Umbrella plan:** `process/features/local-dashboard/active/dashboard-umbrella_PLAN_19-06-26.md`
**Report target:** `process/features/local-dashboard/reports/p0-reuse-audit_REPORT_19-06-26.md`

---

## Objective

Produce a callable map of every reuse function the dashboard will call (listed below). For each
function, verify it can be invoked headless — without triggering a Telegram push, without starting
PTB polling, and without any runtime side-effect that would be unsafe from a GUI process. Flag any
Telegram coupling and specify the exact decoupling change needed.

This plan is fulfilled by a **research-agent** reading source files and writing findings directly to
the report target. No code is implemented in P0.

---

## Scope

### Functions to audit (mandatory)

| ID | Function | Source file | Headless concern |
|---|---|---|---|
| R1 | `daily_inference(broadcast=False, horizon=H)` | `main.py:934` | `broadcast` param gates alerter — verify gating is complete; check for any unconditional alerter call |
| R2 | `inference_for_holdings(holding_tickers, window_rows=120)` | `main.py:1469` | No `broadcast` param — check if alerter is called inside the fn body |
| R3 | `verify_single_ticker(ticker)` | `main.py:1560` | Returns HTML string — confirm no push inside body |
| R4 | `audit_evaluator.run_post_mortem(user_id, days)` | `src/utils/audit_evaluator.py:70` | Pure analytical fn expected — confirm |
| R5 | `PortfolioManager(user_id=...)` add/remove/query | `src/trading/portfolio_manager.py` | DB-only; no Telegram expected — confirm |
| R6 | `signal_ledger.list_open()` | `src/trading/signal_ledger.py:113` | DB-only — confirm |
| R7 | `signal_ledger.check_exits_due()` | `src/trading/signal_ledger.py:152` | DB-only — confirm |
| R8 | `telegram_alerter.send_text_alert` (send-only) | `src/utils/telegram_alerter.py` | This IS the send call; confirm it does NOT start a bot application or polling loop |

### Secondary audit items

- Check whether `main.py` module-level code (imports, globals, `CONFIG` init) triggers any network
  call, PTB `Application` construction, or polling setup on import. The dashboard will `import main`
  and call its functions; module-level side effects would be a problem.
- Check `src/utils/telegram_bot.py` to confirm the dashboard MUST NOT import it (contains
  `build_application` which starts polling). Confirm the only Telegram import needed is
  `telegram_alerter`.
- Confirm `PortfolioManager` user_id semantics: what user_id should the dashboard use for a
  single-user local install? (Likely a fixed string like `"dashboard"` or the user's chat_id from
  Settings.)

---

## Audit Checklist (research-agent fulfills these)

Each item below maps to a numbered finding in the report.

1. Read `main.py:934–1100` (daily_inference body). Confirm every alerter call is inside a
   `if broadcast:` guard. Record the exact line numbers.
2. Read `main.py:1469–1560` (inference_for_holdings body). Find every call to
   `telegram_alerter`, `TelegramBot`, `AlerterBot`, or `send_text_alert`. If found, classify as
   COUPLED and propose a `broadcast=False` guard or thin wrapper.
3. Read `main.py:1560–1650` (verify_single_ticker body). Same search. Classify as CLEAN or COUPLED.
4. Read `src/utils/audit_evaluator.py:70` function body. Same search. Classify.
5. Read `src/trading/portfolio_manager.py` public methods. Confirm no Telegram import or call.
6. Read `src/trading/signal_ledger.py:113–200`. Confirm no Telegram import or call.
7. Read `src/utils/telegram_alerter.py` top-level. Confirm `send_text_alert` is a plain HTTP/async
   function with no `ApplicationBuilder`, no `run_polling`, no `run_webhook`. Record its exact
   signature and any async requirements.
8. Read `main.py:1–100` (module-level imports and globals). List any module-level side effects
   (network calls, DB writes, PTB construction). Any module-level `Application` or polling setup
   is a HIGH blocker.
9. Confirm `src/utils/telegram_bot.py` contains `build_application` or `ApplicationBuilder` and
   MUST NOT be imported by the dashboard.
10. Determine recommended user_id for dashboard PortfolioManager: options are a fixed constant
    (e.g. `"dashboard"`), or the user-configured TELEGRAM_CHAT_ID from .env. Record the tradeoff.

---

## Report Format

The research-agent writes findings to:
`process/features/local-dashboard/reports/p0-reuse-audit_REPORT_19-06-26.md`

Required report sections:

1. **Callable Map** — table: fn ID, module, signature, headless status (CLEAN / COUPLED / BLOCKER),
   evidence (line numbers).
2. **Coupling Findings** — for each COUPLED item: exact line, what it calls, proposed fix
   (additive wrapper or guard), estimated effort.
3. **Module-level Side Effects** — list of any found; severity.
4. **user_id Recommendation** — chosen approach + rationale.
5. **P1 Unblocked / Blocked** — explicit statement: "P1 is unblocked" OR list of blockers that
   must be resolved before P1 can start safely.
6. **Open questions for P2** — any coupling issues that only matter at wiring time (P2), not at
   static UI time (P1).

---

## Acceptance Criteria

- [ ] All 10 checklist items have a recorded finding (CLEAN, COUPLED, or BLOCKER).
- [ ] Every COUPLED item has a specific proposed fix (file, function, change type).
- [ ] Module-level side effects section is present and complete.
- [ ] "P1 unblocked / blocked" verdict is explicit.
- [ ] Report file exists at the target path.

---

## Dependencies

- None. P0 is the first phase; no prior phase outputs needed.
- Read-only. No code changes.

---

## Rollback

Not applicable — P0 produces only a report artifact. No source changes.

---

## Verification Evidence

Report file exists at target path. All 10 checklist items answered. Acceptance criteria checklist
complete. Research-agent status DONE or DONE_WITH_CONCERNS.

---

## Resume and Execution Handoff

Pass this file to a **research-agent** (vc-research-agent). The agent reads source files, fills in
the audit checklist, and writes the report to:
`process/features/local-dashboard/reports/p0-reuse-audit_REPORT_19-06-26.md`

After the report is written and reviewed, advance to P1:
`process/features/local-dashboard/active/p1-ui-build_PLAN_19-06-26.md`
