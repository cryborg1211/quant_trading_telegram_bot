# Portfolio Guard — EOD Protective Alert Layer

**Date**: 13-07-26
**Complexity**: Simple
**Status**: 🧪 TESTING — CODE DONE + VERIFIED (14-07-26, orchestrator-run evidence). Full suite: 653/653 green (independently re-confirmed via a full `pytest -q` run during UPDATE PROCESS, exit code 0, zero failures; 613 pre-existing + 40 new `tests/test_portfolio_guard.py`). `pytest --collect-only -q` clean (all `import main` files collect, zero circular-import breakage). Live-DB read-only dry run completed against `data/quant_v6_core.duckdb` (no write paths invoked by construction — only `load_guard_positions`/`closes_between`/`evaluate_position` were exercised; empirical before/after row-count DB diff was not run); price-normalization confirmed correct on real rows (entries `27.5`/`152.5` → `27,500`/`152,500` VND). Per this repo's plan-artifact convention (CODE DONE ≠ VERIFIED, see `intraday-attack-scanner_PLAN_06-07-26.md`'s Gate 5), this plan stays **ACTIVE** — archival is gated on confirming the first real production EOD run at the 15:30 ICT cron (same Gate-5-style precedent as the intraday scanner), not yet observed.

**Manual verification (20-07-26, 00:06 ICT):** a live `python main.py --task full_pipeline` run (operator-triggered, not the scheduled cron) fired the guard for real — `[TelegramBot] Alert sent → 1818282405 for portfolio_guard` in `logs/quant_v6.log`. Confirms the wiring end-to-end against the real DB/holdings on a real trading day. Does **not** satisfy the plan's own stated gate (the specific 15:30 ICT scheduled cron invocation) — stays ACTIVE pending that.

## Execution Progress (13-07-26)

- [x] 1. `src/trading/portfolio_guard.py` created — HARD CONTRACT + PURITY LAYERING docstring, "MUST NEVER import main" stated, exact imports per plan.
- [x] 2. `normalize_entry_price_vnd` — `_ENTRY_PRICE_SCALE_THRESHOLD_VND = 1_000.0`, precedent-citing docstring (main.py:89 + headless.py:65-79), `None`/`<=0` → `0.0`.
- [x] 3. `evaluate_position` — 5 triggers in fixed order, exact Vietnamese templates, CA-gap downgrade scoped to hard-stop + trailing only, model-flip OR-across-horizons, `[]` on empty series / non-positive entry.
- [x] 4. `build_guard_alert_card` — exact header/footer/trigger copy, `_MR_SELL_VETO` per rule (SELL-leaning only, once per lot), 3800-char block-truncate with overflow notice.
- [x] 5. `load_guard_positions` — READ-ONLY SELECT, cron-excluded, optional user filter, `[]` on any exception (mirrors `signal_ledger.list_open`).
- [x] 6. Tests Part A (normalize + evaluate_position) — green.
- [x] 7. Tests Part B (build_guard_alert_card) — green.
- [x] 8. Tests Part C (load_guard_positions, temp-file DuckDB, cron exclusion) — green.
- [x] 9. `telegram_alerter.py` — `_send_to_one` extracted from `_dispatch` (byte-for-byte), new `send_text_to_chat` reuses it (single recipient).
- [x] 10. Tests Part D (Telegram single-send + mock-mode) — green; `test_cards.py`/`test_telegram_split.py` still green.
- [x] 11. Three `TradingConfig` fields added after `serve_adv_window` with house-style comments.
- [x] 12. Matching keys added to `config/settings.json` `"trading"` object.
- [x] 13. `main._run_guard_for_users` implemented + `from src.trading import portfolio_guard` import added.
- [x] 14. `main.notify_portfolio_guard` implemented (config gate + never-raise, mirrors `notify_tranche_exits`).
- [x] 15. Wired into `full_pipeline` and `inference_only` after each `notify_tranche_exits`; docstrings updated.
- [x] 16. Tests Part E (orchestration, monkeypatched serve stack) — green; `pytest --collect-only -q` clean.
- [x] 17. Optional `/guard` command added to `telegram_bot.py` (handler + `build_application` registration + `_BOT_COMMANDS` + `HELP_TEXT`; NOT edit-only); `import src.utils.telegram_bot` clean.

**Verified**: `python -c` import check prints `True 0.08 True` for the three new config knobs; `/guard` present in `_BOT_COMMANDS`; full `pytest -q` suite (653/653 green, exit code 0); `pytest --collect-only -q` clean; live-DB read-only dry run executed (no write paths invoked by construction; empirical DB diff not run), price-normalization confirmed correct on real rows.
**Not yet verified**: an empirical before/after row-count DB diff (`portfolio`/`trade_history`/`signal_ledger`/`sentiment_entry_paperlog`) for the dry run; the first live production EOD run at the 15:30 ICT cron — this cron run is the sole remaining archival gate (mirrors the intraday scanner's Gate 5 precedent: CODE DONE + suite-green is not the same as a confirmed live run).
**Deviations**: see the "Deviations from plan (EXECUTE, 13-07-26)" note under Functional Requirements.

## Overview

Today nothing watches a human user's `/add`-tracked holdings between the moment they buy and the moment they manually run `/suggest_sell`. `PortfolioManager.update_live_performance` only auto-sells the automated **cron** virtual book (`user_id='cron'`); real users get zero proactive protection — the triggering incident was a user riding VHM −7% with no alert. This plan adds `notify_portfolio_guard()`, an EOD, alert-only, two-stage protective scan: **Stage 1** is five deterministic, zero-LLM triggers (hard stop-loss, take-profit, trailing-stop, model-flip-to-SELL, NO_TRADE regime warning) that decide *whether* an alert fires — the LLM can never suppress delivery. **Stage 2** is a single config-gated arbitrator + sentiment + mean-reversion enrichment pass, run only on tickers that already tripped a Stage-1 trigger. Delivery is event-only (silent when nothing fires) and per-user (Telegram DM to `chat_id == user_id`). The feature is alert-only: it writes nothing to parquet/DuckDB/`signal_ledger`/`sentiment_entry_paperlog` and never auto-sells a user position.

This plan also resolves a **must-investigate price-scale question** (see the dedicated section below) with a confirmed, evidence-backed finding: the codebase already has two *conflicting* assumptions about what unit `portfolio.price` is stored in for human users, and this plan adopts a third, safer, precedented resolution.

## Quick Links

- [Critical Investigation Finding: Price-Scale Ambiguity](#critical-investigation-finding-price-scale-ambiguity)
- [Goals and Success Metrics](#goals-and-success-metrics)
- [Phase Completion Rules](#phase-completion-rules)
- [Execution Brief](#execution-brief)
- [Scope](#scope)
- [Assumptions and Constraints](#assumptions-and-constraints)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [Acceptance Criteria](#acceptance-criteria)
- [Implementation Checklist](#implementation-checklist)
- [Risks and Mitigations](#risks-and-mitigations)
- [Integration Notes](#integration-notes)
- [Touchpoints](#touchpoints)
- [Public Contracts](#public-contracts)
- [Blast Radius](#blast-radius)
- [Verification Evidence](#verification-evidence)
- [Resume and Execution Handoff](#resume-and-execution-handoff)
- [Cursor + RIPER-5 Guidance](#cursor--riper-5-guidance)

---

## Critical Investigation Finding: Price-Scale Ambiguity

**Question posed by the approved design:** does `portfolio.price` (a human user's `/add`-inserted entry price) live in the parquet "thousands of VND" convention, or in absolute VND? A mismatch would fire false extreme stop alerts — the single highest-risk defect for this feature.

**What was actually found (three files, three different assumptions, confirmed by reading each one):**

1. `src/utils/telegram_bot.py::add_portfolio_command` (the live `/add` handler) does a bare `float(raw_price)` cast — **no scale normalization at all**. Its own help text / `_BOT_COMMANDS` example is `/add VNE 1000 32.5`. A value of "32.5" is only economically sane as **thousands-of-VND** (32,500 VND) — no HOSE stock trades at an absolute 32.5 VND. So the bot's own documented convention implies *thousands*.
2. `dashboard/utils/headless.py::_pnl_ratio` (added later, for the Streamlit GIỮ tab) explicitly asserts the **opposite** in its own docstring: *"The `portfolio` table stores absolute-VND entry prices (the bot's /add takes whole-VND prices)"* — and its code only rescales the *current-close* side (`if current_close < 1000: current_close *= 1000`), trusting `entry_price` unconditionally as already-absolute. If a user actually followed the bot's own `/add` example (32.5), this dashboard code computes a wildly wrong ratio: `(32500 − 32.5) / 32.5 ≈ +99,900%`. This looks like a **pre-existing, unverified assumption baked into shipped dashboard code**, not a settled convention.
3. `dashboard/utils/headless.py::_parse_price` documents a third historical variant: some `portfolio` rows once stored price as **display TEXT** (e.g. `'47,800 VND'`). The current DDL (`src/data/db_engine.py:287-293`) declares `portfolio.price DOUBLE`, and both known insertion paths (`telegram_bot.py::add_portfolio_command`, `dashboard/utils/headless.py::portfolio_add`) do a Python `float(...)` cast before binding — so a literal comma/letter string cannot be inserted via either live path today. This is defensive code for old/migrated data, not a live write path; it does not change the core ambiguity.
4. `src/trading/portfolio_manager.py::update_live_performance` (the **cron** path, `user_id='cron'`, which this feature explicitly skips) documents its own inputs as absolute VND (docstring example `{"FPT": 136000}`), consistent with `src/reports/builders.py`'s house-wide display convention of formatting prices as `f"{price:,.0f} VND"`.

**The resolving precedent — this is the key discovery:** `main.py` already has a **named, canonical fix for exactly this class of ambiguity**, already in production use by `daily_inference`, `verify_single_ticker`, `rebalance_portfolio`, and `_dispatch_signals`:

```
main.py:89   _VN_PRICE_SCALE_THRESHOLD = 1_000.0  # VN stocks quoted in thousands; raw < 1000 → multiply by 1000
main.py:92-116  _get_live_exec_prices(...)  "VN market convention: prices are stored in thousands
                 (e.g. 10.5 = 10,500 VND). If extracted price < 1,000 we multiply by 1,000..."
```

This is the **same magnitude heuristic and the same `1000` threshold** the dashboard's `_pnl_ratio` independently reinvented (inline, undocumented-as-shared) for the *current-close* side only. `main._VN_PRICE_SCALE_THRESHOLD` always fires (parquet closes are always < 1000 in this unit) because it is applied to a value that is *unconditionally* thousands-scale — it does no real disambiguation there. Applied to `portfolio.price`, however, this exact rule *does* real disambiguation work, because that field's scale is genuinely unverified.

**Decision (binding for this plan):** `src/trading/portfolio_guard.py` will define its own pure `normalize_entry_price_vnd(raw_price)` using the **identical `1_000.0` threshold and identical "< 1000 ⇒ thousands ⇒ ×1000" rule**, applied directly to `portfolio.price` (the genuinely ambiguous field) rather than only to the current-close side (the dashboard's incomplete fix). The function's docstring must cite both `main._VN_PRICE_SCALE_THRESHOLD` (main.py:89) and `dashboard/utils/headless.py::_pnl_ratio` (headless.py:65-79) as precedent, so a future reader understands this is a deliberate, cross-referenced convention, not a new arbitrary number. `price_lookup.closes_between(...)` output is **unconditionally** thousands-of-VND by documented contract (no ambiguity), so guard code multiplies it by `1000.0` unconditionally to reach absolute VND — no conditional needed on that side.

**Open question flagged for the user (not fixed by this plan — explicitly out of scope):** `dashboard/utils/headless.py::_pnl_ratio`'s one-sided assumption ("entry_price is always absolute") appears to already be a **latent bug** in the shipped Streamlit GIỮ tab if any user ever followed the bot's own `/add` example convention literally. This plan does **not** touch `dashboard/utils/headless.py` (scope discipline — the approved design is the portfolio guard feature only). Recommend a small, separate follow-up ticket to reconcile `/add`'s stored-price convention repo-wide (e.g., enforce/validate thousands-VND at input time, or store an explicit unit tag) so the bot and dashboard stop disagreeing. Flagging this clearly rather than silently fixing it or silently ignoring it.

**Residual limitation accepted (shared with the existing `_get_live_exec_prices`/`_pnl_ratio` heuristic, not a new regression):** a genuine absolute-VND penny price under 1,000 VND (rare, but possible for some warrants) would misclassify as thousands-scale. This is the same edge case the rest of the codebase already accepts under the identical threshold; not hardened further here (YAGNI — no evidence any currently-held ticker is in this range).

---

## Goals and Success Metrics

**Goals:**
- Give every human user (any `portfolio.user_id != "cron"`) a same-day, evidence-based warning when their position crosses a hard-stop, trailing-stop, take-profit, model-flip, or NO_TRADE-regime condition.
- Never let an LLM/arbitrator call suppress an alert that Stage 1 already decided should fire.
- Never write to any persistent store and never auto-sell a position — this is advisory only.
- Keep Gemini spend bounded to at most one batched call per EOD run, only when at least one trigger fired anywhere.
- Correctly handle the confirmed `portfolio.price` scale ambiguity so a unit mismatch cannot produce a false extreme alert.

**Success Metrics:**
- `tests/test_portfolio_guard.py` green, covering every trigger, the CA-gap downgrade, the price-normalization boundary, event-only gating, and the Gemini-failure/`llm_enabled=False` fallback.
- Full existing suite (currently 591 tests across 47 files) stays green after this change — zero regressions, in particular for the 12 test files that `import main` directly.
- A manual dry run (see [Verification Evidence](#verification-evidence)) produces a correctly-scaled, correctly-worded Vietnamese alert card for a synthetic underwater position, and produces **no message at all** for a synthetic healthy position.

---

## Phase Completion Rules

A phase is NOT complete until:

1. **Integration Test** - Works with other system pieces (main.py orchestration, Telegram delivery, config gating)
2. **Manual Test** - Operator can run the dry-run command and observe correct behavior
3. **Data Verification** - No unintended writes occur (confirmed by inspecting `portfolio`/`sentiment_entry_paperlog`/`signal_ledger` row counts before/after a dry run)
4. **Error Handling** - Failure cases (missing data, Gemini exception, bad chat_id, empty portfolio) degrade gracefully, never raise
5. **User Confirmation** - User says "it works" after reviewing the pytest output and (optionally) a manual dry run

Status meanings:
- ⏳ PLANNED - Not started
- 🔨 CODE DONE - Written but not E2E tested
- 🧪 TESTING - Currently being tested
- ✅ VERIFIED - Tested AND confirmed working
- 🚧 BLOCKED - Has issues

After each phase, document:
- [x] What was tested manually — import-surface check (`import main`, `import src.utils.telegram_bot`, config prints `True 0.08 True`, `/guard` in `_BOT_COMMANDS`) passed. Full mock-mode dry run (`python -c "import main; print(main.notify_portfolio_guard())"`) was executed against the live `data/quant_v6_core.duckdb` (read-only) — price-normalization confirmed correct on real rows (entries `27.5`/`152.5` → `27,500`/`152,500` VND).
- [ ] Data verified in DB (show query + result) — the dry run itself was executed against the live DB (14-07-26), but an empirical before/after row-count diff was NOT captured. By construction the feature issues only one READ-ONLY `SELECT` (`load_guard_positions`) and no `INSERT`/`UPDATE`/`DELETE` anywhere in its code paths (verified by reading the diff: no write SQL, no `signal_ledger`/`paperlog` calls) — this remains construction-level evidence, not an empirical row-count confirmation. Still PENDING for a literal before/after diff if stricter proof is wanted.
- [x] Errors encountered and fixed — none in code; only shell-quoting friction invoking PowerShell through the git-bash-backed tool (worked around, no code impact).
- [ ] User confirmation received — reframed per this plan's own closeout gate: the remaining confirmation is the first live production EOD run at the 15:30 ICT cron, not a chat "it works" (mirrors the intraday-scanner Gate 5 precedent). PENDING.

---

## Execution Brief

**This is a SIMPLE (one-session) plan** — implement continuously without approval gates between phases below. The phases are logical groupings for understanding flow, not stop points. Steps within each phase are enumerated precisely in [Implementation Checklist](#implementation-checklist).

### Phase 1: Pure Trigger Engine (`src/trading/portfolio_guard.py`)
**What happens:** Create the new pure module: price normalization, the five-trigger evaluation function, the CA-gap downgrade, and the per-user HTML card builder. No DB, no network, no `main` import anywhere in this file (hard constraint — see [Risks](#risks-and-mitigations)).
**Test:** `pytest -q tests/test_portfolio_guard.py -k "normalize_entry_price or evaluate_position or build_guard_alert_card"` — all green, including CA-gap and boundary cases.
**Verify:** Inspect that `evaluate_position` never imports `duckdb`, `requests`, or `main`; confirm via `python -c "import ast"`-free manual read or simply `grep -n "^import\|^from" src/trading/portfolio_guard.py` shows only pure/data imports.
**Done when:** All pure-function tests pass and the module has zero I/O imports at the top of the file (I/O helpers are isolated in their own clearly-marked section, matching `src/trading/intraday_scanner.py`'s "PURITY LAYERING" convention).

### Phase 2: Position Loading + Config Knobs
**What happens:** Add `load_guard_positions(db_path=None, user_id=None)` (the module's one I/O function) to `portfolio_guard.py`, and add the three new `TradingConfig` fields to `config/settings.py` + matching keys to `config/settings.json`.
**Test:** `pytest -q tests/test_portfolio_guard.py -k load_guard_positions` against an in-memory DuckDB fixture seeded with a `cron` row + two real-user rows.
**Verify:** Query result excludes the `cron` row in both the all-users mode and the single-user-filtered mode; `python -c "from config.settings import CONFIG; print(CONFIG.trading.portfolio_guard_enabled, CONFIG.trading.portfolio_guard_trailing_pct, CONFIG.trading.portfolio_guard_llm_enabled)"` prints `True 0.08 True`.
**Done when:** Loader tests pass and `Config.from_json()` round-trips the three new `settings.json` keys without error.

### Phase 3: Telegram Single-Chat Delivery
**What happens:** Refactor `TelegramBot._dispatch` in `src/utils/telegram_alerter.py` to extract a `_send_to_one` helper, then add the public `send_text_to_chat(chat_id, html_text, label)` method that reuses it — single-recipient send, not the broadcast `chat_id_list` loop.
**Test:** `pytest -q tests/test_portfolio_guard.py -k send_text_to_chat` — one test asserts a single `requests.post` call targeting exactly the given `chat_id` (not looped over `chat_id_list`); one test asserts mock-mode (`bot_token == "YOUR_BOT_TOKEN"`) logs instead of posting.
**Verify:** `pytest -q tests/test_cards.py tests/test_telegram_split.py` still green (these exercise `_build_message`/split helpers that must be untouched by this refactor).
**Done when:** New method + refactor are in place and every existing Telegram-related test still passes.

### Phase 4: Main.py Orchestration Wiring
**What happens:** Add `_run_guard_for_users(...)` (the shared evaluation core: build live features once, dual-horizon predict, per-lot trigger evaluation, one batched Stage-2 enrichment call, per-user card building) and `notify_portfolio_guard() -> int` (config gate, load positions, send per user, never raise) to `main.py`. Wire `notify_portfolio_guard()` into `full_pipeline` and `inference_only`, immediately after each one's existing `notify_tranche_exits()` call.
**Test:** `pytest -q tests/test_portfolio_guard.py -k "run_guard_for_users or notify_portfolio_guard"` with `predict_v3_horizon`, `evaluate_trades_batch`, `mr_score_tickers`, and `Alpha360Generator.load_live_ohlcv_window` monkeypatched (mirrors `tests/test_intraday_scanner.py`'s `test_run_scan_*` monkeypatch style).
**Verify:** Event-only gating test confirms zero `TelegramBot` calls when no trigger fires anywhere; Gemini-failure test confirms a card still builds when `evaluate_trades_batch` raises; config-disabled test confirms `notify_portfolio_guard()` returns `0` and performs zero DB reads when `portfolio_guard_enabled=False`.
**Done when:** All orchestration tests pass and `pytest --collect-only -q` still collects all 12 `import main` test files without a collection error.

### Phase 5: Optional On-Demand `/guard` Command
**What happens:** Add `guard_command` to `src/utils/telegram_bot.py` (mirrors `suggest_sell_command`'s structure exactly), register it in `build_application`, and add it to `_BOT_COMMANDS` + `HELP_TEXT`. Not edit-only (both Admin and User roles may call it for their own holdings, matching `/suggest_sell`'s classification).
**Test:** Manual smoke test only is required at minimum (handler logic is thin glue over the already-tested `_run_guard_for_users`); add 1-2 focused tests if time permits for the "empty portfolio" and "no triggers" reply branches.
**Verify:** `python -c "import src.utils.telegram_bot"` still imports cleanly (no syntax/registration errors).
**Done when:** Command is registered and does not break `build_application()`.

### Phase 6: Full-Suite Verification
**What happens:** Run the complete pytest suite and the manual dry run described in [Verification Evidence](#verification-evidence).
**Test:** `pytest -q` — full suite.
**Verify:** Test count increases by the new file's test count with zero failures; manual dry run (mock-mode Telegram) shows a correctly-worded, correctly-scaled card for a synthetic triggered position and silence for a healthy one.
**Done when:** Full suite green, manual dry run behaves as specified, user confirms.

### Expected Outcome
- `src/trading/portfolio_guard.py` exists as a new pure-plus-thin-IO module, fully unit-tested.
- `main.notify_portfolio_guard()` runs automatically at the end of every `full_pipeline`/`inference_only` EOD pass, alerting only users with at least one fired trigger, to their own chat only.
- `cron` rows are never evaluated.
- No new writes anywhere; the pipeline cannot be broken by a guard failure (always degrades to a log line).
- An optional `/guard` command lets any user pull their own check on demand.
- Full pytest suite (591 existing + new) green.

---

## Scope

**In-Scope:**
- New pure module `src/trading/portfolio_guard.py`: price normalization, five Stage-1 triggers, CA-gap downgrade, per-user card builder, position loader.
- New `main.py` orchestration: `_run_guard_for_users`, `notify_portfolio_guard`, wiring into `full_pipeline` and `inference_only`.
- New `TelegramBot.send_text_to_chat` (+ `_send_to_one` extraction) in `src/utils/telegram_alerter.py`.
- Three new `TradingConfig` fields + matching `config/settings.json` keys.
- New `tests/test_portfolio_guard.py`.
- Optional `/guard` on-demand bot command.

**Out-of-Scope:**
- Any change to `dashboard/utils/headless.py` (including the flagged `_pnl_ratio` latent-bug candidate) — noted, not fixed, here.
- Any change to `PortfolioManager`/the cron automated book.
- Any change to `src/backtest/pipeline.py`, `FEATURE_RECIPE_VERSION`, or any model retrain.
- Auto-selling, auto-adjusting position size, or any write to `portfolio`, `trade_history`, `signal_ledger`, or `sentiment_entry_paperlog`.
- A new DB table for guard history/audit trail (not requested; alert-only, stateless per run).
- Multi-channel delivery (email, SMS) — Telegram DM only, per approved design.

## Assumptions and Constraints

**Assumptions:**
- `chat_id == user_id` holds for real Telegram users in a 1:1 private chat (confirmed: `src/utils/telegram_bot.py:505-506` comment states this explicitly and it underlies the existing split-ID access-control system).
- Dashboard-inserted `portfolio` rows (`user_id` typically `"local"` or a custom `dashboard_user_id` string from `config/settings.json`) are **not** valid Telegram chat targets. The guard will still evaluate their triggers (uniform, simple logic) but delivery will silently no-op for that pseudo-user via the existing per-chat try/except degrade in `_send_to_one` (same pattern that already tolerates one bad `chat_id` in the broadcast loop) — this is an accepted, logged-warning degrade, not a crash risk. Confirmed via `dashboard/utils/headless.py:30-32` (`_DEFAULT_USER_ID = "local"`) and `_EDIT_ONLY_COMMANDS = {"add", "remove"}` (`telegram_bot.py:511`, meaning only Admin ID1 can `/add` through the bot — real Telegram-sourced rows are therefore realistically few, but the design must not hardcode a cardinality assumption).
- `price_lookup.closes_between(ticker, entry_date, today)` returning `[]` (delisted ticker, brand-new position with no bar yet, or a shard read failure) means "insufficient data" — the position's triggers are skipped entirely for that run, not treated as an error.
- Multiple `/add` rows for the same `(user_id, ticker)` (no DB-level uniqueness constraint — confirmed via `dashboard/utils/headless.py:183-185`'s own comment) are evaluated **per lot**, not collapsed. This deliberately diverges from `/rebalance`'s existing dict-overwrite dedup (`main.py:1985-1990`, "last row wins") because collapsing could mask a real stop-loss on one lot behind a healthy average — correctness for a protective feature outweighs matching that specific precedent.

**Constraints (verbatim from approved design — binding):**
- Alert-only. ZERO writes: no parquet, no DuckDB tables, no `signal_ledger`, no `sentiment_entry_paperlog`, no auto-sell of user positions.
- No feature-recipe change, no model retrain (pure serve-path wiring; `FEATURE_RECIPE_VERSION` untouched).
- Pipeline must never die on guard failure — every public entrypoint (`notify_portfolio_guard`) is wrapped exactly like `notify_tranche_exits` (broad `except Exception`, log, return `0`).
- Gemini spend bounded: LLM only when triggers fired, single batch call across the union of all triggered tickers (not per-user, not per-trigger).
- `src/trading/portfolio_guard.py` must **never** import `main` (even lazily) — this is a stricter constraint than `intraday_scanner.py`'s precedent (which does lazily import `main`), chosen deliberately here because `portfolio_guard.py` does not need any `main`-anchored call directly: all ML/arbitrator/Telegram orchestration is performed by `main.py`'s own new functions, which import `portfolio_guard`, not the reverse. This keeps the dependency graph one-directional and eliminates circular-import risk for the 12 test files that `import main`.

**Step-count note:** the Implementation Checklist below runs to 17 items, above the template's usual 8-15 guidance for a SIMPLE plan. This is intentional — the user explicitly required an exhaustive, zero-ambiguity investigation of the price-scale risk and full test coverage per trigger; the extra steps are test/verification granularity, not hidden scope creep.

---

## Functional Requirements

**Stage 1 — deterministic triggers (zero LLM, cannot be suppressed by Stage 2):**

Evaluated per non-cron `user_id`, per portfolio row/lot (`ticker`, `volume`, `entry_price_raw`, `entry_date`):

1. **Hard stop-loss** — fires when `pnl_pct <= CONFIG.trading.stop_loss_pct` (existing knob, `-0.07`). `pnl_pct = (today_close_abs - entry_price_abs) / entry_price_abs`.
2. **Take-profit (info)** — fires when `pnl_pct >= CONFIG.trading.take_profit_pct` (existing knob, `+0.15`).
3. **Trailing stop** — fires when `(peak_abs - today_close_abs) / peak_abs >= CONFIG.trading.portfolio_guard_trailing_pct` (new knob, default `0.08`), where `peak_abs = max(closes_since_entry_abs)` over `price_lookup.closes_between(ticker, entry_date, today)`, scaled to absolute VND.
4. **Model flip** — fires when the argmax of **either** the T+5 or the T+20 `predict_v3_horizon` output for the held ticker is class `0` (SELL/DOWN). See [Risks](#risks-and-mitigations) for the OR-vs-AND rationale. No arbitrator call in this stage.
5. **Regime warning** — fires when the ticker's latest cached `market_regime` (from `main._LATEST_REGIME_BY_TICKER`, refreshed as a documented side effect of this feature's own `predict_v3_horizon` call — see [Integration Notes](#integration-notes) for the full verification of this mechanism) is in `NO_TRADE_REGIMES` (`{0, 7}`, `src/trading/regime_policy.py`).
6. **Corporate-action shield** — before finalizing triggers 1 and 3, compute `price_lookup.has_ca_gap(closes_since_entry_abs)` (default `max_session_move=0.10`, unit-agnostic). If `True`, replace triggers 1/3's message with the downgraded wording (exact copy below) instead of a confident SELL-protect framing. The trigger still counts as "fired" for event-only gating and Stage-2 eligibility — only the wording softens, per the approved design ("downgrade the wording", not "suppress").

**Stage 2 — enrichment, triggered tickers only, config-gated:**

- Compute the **union** of every triggered ticker across every user for this run.
- If `CONFIG.trading.portfolio_guard_llm_enabled` is `True`: call `evaluate_trades_batch({"5d": stacking_5d, "20d": stacking_20d}, sorted(union))` exactly **once** (mirrors `verify_single_ticker`'s step 3, `main.py:1906-1916`). On any exception, fall through with `({}, {})` — same fallback pattern.
- Always call `mr_score_tickers(sorted(union))` (free, no LLM, already never-raises internally per its own contract in `main.py:238-281`).
- Each user's card is built from only the enrichment data relevant to *their own* triggered tickers — no per-user LLM calls.

**Exact Vietnamese content contract** (must match verbatim — this removes all wording ambiguity from EXECUTE):

| Trigger | Vietnamese message template |
|---|---|
| Hard stop-loss (no CA gap) | `🔴 <b>CẮT LỖ</b>: PnL hiện tại {pnl_pct:+.1f}% (đã vượt ngưỡng cắt lỗ {stop_loss_pct:.0f}%)` |
| Hard stop-loss (CA gap detected) | `⚠️ PnL {pnl_pct:+.1f}% — có thể do hành động doanh nghiệp (chia cổ tức/cổ phiếu thưởng/phát hành thêm), KHÔNG chắc là lỗ thật. Vui lòng tự kiểm tra giá điều chỉnh — đây KHÔNG phải tín hiệu cắt lỗ chắc chắn.` |
| Take-profit | `🟢 <b>CHỐT LỜI</b>: PnL hiện tại {pnl_pct:+.1f}% (đã vượt ngưỡng chốt lời {take_profit_pct:.0f}%)` |
| Trailing stop (no CA gap) | `🟠 <b>TRAILING STOP</b>: giá đã giảm {drawdown_pct:.1f}% từ đỉnh {peak:,.0f} VND kể từ khi mua (ngưỡng {trailing_pct:.0f}%)` |
| Trailing stop (CA gap detected) | `⚠️ Giảm {drawdown_pct:.1f}% từ đỉnh — có thể do hành động doanh nghiệp, KHÔNG chắc là lỗ thật. Vui lòng tự kiểm tra — đây KHÔNG phải tín hiệu cắt lỗ chắc chắn.` |
| Model flip | `🔻 <b>MÔ HÌNH ĐỔI TÍN HIỆU</b>: {horizon_label} → BÁN (P(Tăng)={p_up:.0f}%)` (one line per flipped horizon) |
| Regime warning | `🌐 <b>CẢNH BÁO PHA THỊ TRƯỜNG</b>: đang ở pha {regime_label_vi} (Regime {regime}) — rủi ro hệ thống cao, hạn chế mở/giữ vị thế mới.` |
| MR knife-catch caution (appended only to hard-stop / trailing-stop / model-flip lines, only when `mr_state.get("fired")` is True) | Reuse `_MR_SELL_VETO` **verbatim** from `src/reports/builders.py:120-124` (`"⚠️ <b>[CẢNH BÁO BÁN ĐÚNG ĐÁY: ... Hạn chế bán tháo lúc này!]</b>"`) — do not paraphrase. |
| Card header | `🛡️ <b>CẢNH BÁO DANH MỤC</b>\n{dd/mm/yyyy}\n══════════════════════════════` (separator style matches `main.py::_build_exits_report`) |
| Card footer | `<i>Đây là cảnh báo tự động dựa trên quy tắc + mô hình, KHÔNG phải lệnh giao dịch. Hệ thống KHÔNG tự động bán bất kỳ vị thế nào — quyết định cuối cùng luôn thuộc về bạn.</i>` |
| `/guard` on-demand, no triggers | `"✅ Danh mục ổn định — không có cảnh báo nào hôm nay."` |
| `/guard` on-demand, empty portfolio | Reuse `EMPTY_PORTFOLIO_MESSAGE` verbatim (`telegram_bot.py:117-119`) |

MR caution is **never** appended to take-profit-only or regime-warning-only lines (nonsensical pairing — you are not "selling the bottom" on a profitable exit).

### Deviations from plan (EXECUTE, 13-07-26)

All MINOR — none alter a binding decision (price-scale rule, trigger math, alert-only/no-write, never-raise, no retrain, model-flip OR-across-horizons, CA-gap scope are all implemented exactly as written). Recorded per EXECUTE discipline:

1. **Enrichment-line copy (card).** The card contract table defines exact copy for every trigger, the header, and the footer, but NOT for the arbitrator `(final_decisions, all_sentiments)` enrichment, while the FR still states the card is "built from … the enrichment data." Filled the gap with a minimal, clearly-labeled line reusing the EXISTING house label from `telegram_alerter.py`: `🧠 <b>Trọng tài tin tức:</b> {BÁN/THOÁT|GIỮ|MUA} (tâm lý {score:+.2f})` + an optional 300-char-truncated `reasoning_vi` on the next line. Rendered only when Stage 2 produced data for that ticker (absent ⇒ the "quant-only" card of Acceptance Criterion 10). No paraphrase of any defined-contract copy.
2. **`build_guard_alert_card` header date reads the wall clock.** The Public Contracts signature `(ticker_lots, enrichment, mr_scores) -> str` has no date param, yet the header contract requires `{dd/mm/yyyy}`. The function calls `date.today()` for the header only (same approach as `notify_tranche_exits`); otherwise it is pure/deterministic on its inputs.
3. **Model-flip representation.** Emitted as ONE `model_flip` trigger dict whose `message_vi` is the newline-joined per-horizon lines (so the MR veto appends once per lot), rather than one dict per horizon. The rendered card still shows one line per flipped horizon exactly as the contract requires.
4. **Stdlib imports beyond the enumerated list.** Added `from __future__ import annotations`, `html` (card HTML-escaping), and `logging` (loader degrade path) — the plan's step-1 import list enumerated the domain imports, not stdlib. No new third-party dependency.
5. **`_run_guard_for_users(db_path=…)` accepted but unused by the live path.** The live evaluation sources features from parquet via `Alpha360Generator` (not DuckDB), so `db_path` is kept in the signature for contract parity/testing but is not read; documented in the function docstring.

## Non-Functional Requirements

- **Never-raise contract:** `notify_portfolio_guard()` wraps its entire body in `try/except Exception`, logs, and returns `0` — identical shape to `main.py::notify_tranche_exits` (`main.py:2123-2172`).
- **No persistence:** zero `INSERT`/`UPDATE`/`DELETE` statements anywhere in this feature's code paths.
- **Bounded external calls:** at most one `evaluate_trades_batch` call and one `mr_score_tickers` call per EOD run, regardless of user count.
- **Config-reversible:** `portfolio_guard_enabled=False` fully short-circuits (zero DB reads, zero compute); `portfolio_guard_llm_enabled=False` skips only the arbitrator call, keeping quant-only alerts live.
- **Accepted redundant compute:** the guard's own `predict_v3_horizon` calls and its own `Alpha360Generator().load_live_ohlcv_window(...)` build are independent of (and duplicate some work already done inside) `daily_inference()`'s own internal calls earlier in the same pipeline run. This is a deliberate tradeoff (self-sufficiency, no `daily_inference` signature change, matches how `/verify`/`/rebalance` already independently rebuild their own live frame) accepted for an EOD-cadence, non-latency-sensitive job.

## Acceptance Criteria

1. `normalize_entry_price_vnd(32.5)` returns `32500.0`; `normalize_entry_price_vnd(47800.0)` returns `47800.0` unchanged; boundary `normalize_entry_price_vnd(1000.0)` returns `1000.0` unchanged (rule is `< 1000`, not `<=`).
2. A synthetic position with `pnl_pct <= -0.07` fires a hard-stop trigger with the exact template wording above.
3. A synthetic position with `pnl_pct >= 0.15` fires a take-profit trigger.
4. A synthetic position whose post-entry close series drops `>= 0.08` from its peak fires a trailing-stop trigger.
5. A synthetic position with a T+5 or T+20 argmax of `0` fires a model-flip trigger naming the correct horizon(s).
6. A synthetic position whose cached regime is `0` or `7` fires a regime-warning trigger with the correct `regime_label_vi` text.
7. Injecting a `>10%` single-session jump into the closes series causes hard-stop/trailing-stop alerts to render the downgraded CA wording instead of the confident wording, while take-profit/regime wording is unaffected.
8. A user with zero fired triggers receives no `TelegramBot` call at all from the EOD sweep (asserted via a monkeypatched call counter).
9. Rows with `user_id == "cron"` never appear in `load_guard_positions()`'s output, in either all-users or single-user-filtered mode.
10. `evaluate_trades_batch` raising an exception, and `portfolio_guard_llm_enabled=False`, both still produce a fully-rendered quant-only card (asserted by the same test shape, two config paths).
11. `TelegramBot.send_text_to_chat("123", html, "portfolio_guard")` results in exactly one `requests.post` call with `payload["chat_id"] == "123"`, never iterating `self.chat_id_list`.
12. Full `pytest -q` suite passes with zero regressions after all changes land.

---

## Implementation Checklist

1. **Create `src/trading/portfolio_guard.py`** — module docstring (mirror `intraday_scanner.py`'s HARD CONTRACT + PURITY LAYERING doc-comment style), imports: `from src.data import price_lookup` (for the pure `has_ca_gap`), `from src.trading.portfolio_manager import CRON_USER_ID`, `from src.trading.regime_policy import NO_TRADE_REGIMES`, `from src.features.market_regime import regime_label_vi`, `from src.reports.builders import _MR_SELL_VETO`, `duckdb`, `from config.settings import CONFIG`, `from datetime import date`. Explicit module-docstring line stating **"this module MUST NEVER import `main`"**.
2. **Implement `normalize_entry_price_vnd(raw_price: float) -> float`** (pure) with the `_ENTRY_PRICE_SCALE_THRESHOLD_VND = 1_000.0` module constant and the precedent-citing docstring (cites `main.py:89` and `dashboard/utils/headless.py:65-79`). Handles `None`/`<= 0` → `0.0`.
3. **Implement `evaluate_position(position, closes_since_entry_abs, prediction_5d, prediction_20d, regime, *, stop_loss_pct, take_profit_pct, trailing_pct) -> list[dict]`** (pure) — implements Functional Requirements triggers 1-6 in the fixed evaluation order `hard_stop, take_profit, trailing_stop, model_flip, regime_warning`, using the exact Vietnamese templates from the Functional Requirements table. Each returned dict: `{"kind": str, "message_vi": str, "ca_gap_downgraded": bool}`. Degrades to `[]` immediately when `closes_since_entry_abs` is empty or `entry_price_abs <= 0`.
4. **Implement `build_guard_alert_card(ticker_lots, enrichment, mr_scores) -> str`** (pure) — `ticker_lots: list[dict]` (one user's fired-trigger lots, each `{"ticker", "entry_date", "entry_price_norm", "current_price", "pnl_pct", "volume", "triggers": [...]}`), `enrichment: tuple[dict, dict] | None` (the `(final_decisions, all_sentiments)` pair, or `None`/empty when Stage 2 didn't run), `mr_scores: dict`. Renders the exact header/footer/per-trigger copy from the Functional Requirements table, appends `_MR_SELL_VETO` per the stated rule, and applies the same 3800-char soft-truncate-with-overflow-notice pattern as `main.py::notify_tranche_exits` (`main.py:2149-2165`).
5. **Implement `load_guard_positions(db_path: str | None = None, user_id: str | None = None) -> list[dict]`** (I/O) — `SELECT user_id, ticker, volume, price, CAST(added_at AS DATE) AS entry_date FROM portfolio WHERE user_id != ?` plus an optional additional `AND user_id = ?` clause when `user_id` is given, parameterized, ordered by `user_id, ticker, added_at`. Returns `[]` on any exception (log + degrade, mirrors `signal_ledger.list_open`'s try/except shape).
6. **Write `tests/test_portfolio_guard.py` Part A** — `normalize_entry_price_vnd` (4 tests: thousands-scale, already-absolute, exact-boundary, zero/negative) + `evaluate_position` (covering each of the 5 triggers firing and not-firing, both CA-gap branches, empty-series degrade, and the OR-across-horizons model-flip rule). Run `pytest -q tests/test_portfolio_guard.py -k "normalize_entry_price or evaluate_position"` green before continuing.
7. **Write `tests/test_portfolio_guard.py` Part B** — `build_guard_alert_card` (MR-veto inclusion for SELL-leaning triggers, MR-veto omission for take-profit-only, 4096-char safety with many positions, multi-lot-same-ticker rendering distinctly by entry date). Run green.
8. **Write `tests/test_portfolio_guard.py` Part C** — `load_guard_positions` against an in-memory DuckDB fixture seeded with one `cron` row and two real-user rows (cron-exclusion in both all-users and single-user-filtered modes). Run green.
9. **Refactor `src/utils/telegram_alerter.py`** — extract `_send_to_one(self, chat_id, msg, ticker="N/A") -> None` from `TelegramBot._dispatch`'s loop body (identical mock-check / `requests.post` / status-code-logging / `time.sleep(0.5)` / exception-logging behavior, byte-for-byte preserved); have `_dispatch` call `_send_to_one` per `chat_id` in its existing loop; add the new public `send_text_to_chat(self, chat_id: str, html_text: str, label: str = "alert") -> None` that calls `_send_to_one(chat_id, html_text, ticker=label)` directly (single recipient, not the broadcast list). Add matching docstring cross-referencing `send_text_alert`.
10. **Write `tests/test_portfolio_guard.py` Part D (Telegram)** — one test asserting `send_text_to_chat` posts to exactly the given `chat_id` (monkeypatched `requests.post`, one call, correct payload); one test asserting mock-mode (`bot_token == "YOUR_BOT_TOKEN"`) never calls `requests.post`. Then run `pytest -q tests/test_cards.py tests/test_telegram_split.py` to confirm the refactor didn't disturb existing Telegram tests.
11. **Add three fields to `TradingConfig` in `config/settings.py`**, inserted after the existing `serve_adv_window: int = 20` line: `portfolio_guard_enabled: bool = True`, `portfolio_guard_trailing_pct: float = 0.08`, `portfolio_guard_llm_enabled: bool = True` — each with a house-style comment block (mechanism + kill-switch instruction, matching the `garch_brake_enabled`/`intraday_scanner_enabled` comment style already in the file).
12. **Add matching keys to `config/settings.json`'s `"trading"` object**: `"portfolio_guard_enabled": true`, `"portfolio_guard_trailing_pct": 0.08`, `"portfolio_guard_llm_enabled": true`.
13. **Implement `_run_guard_for_users(positions_by_user: dict[str, list[dict]], db_path: str | None = None, today: date | None = None) -> dict[str, str]`** in `main.py` (placed near `notify_tranche_exits`, after line ~2172). Add `from src.trading import portfolio_guard` to the import block. Algorithm (numbered, no branching logic to reimplement beyond what's specified):
    1. Return `{}` immediately if `positions_by_user` is empty.
    2. Build `latest_df` once via `Alpha360Generator().load_live_ohlcv_window(window_rows=120)`; on `FileNotFoundError` or an empty frame, log a warning and return `{}`.
    3. Call `predict_v3_horizon(latest_df, SHORT_HORIZON)` for `stacking_5d` and `predict_v3_horizon(latest_df, 20)` for `stacking_20d`, each independently try/excepted (`FileNotFoundError`, `RuntimeError`) with an empty-dict fallback — mirrors `daily_inference`'s existing secondary-horizon degrade (`main.py:1140-1149`). This call refreshes `main._LATEST_REGIME_BY_TICKER` as a documented side effect (see [Integration Notes](#integration-notes)).
    4. For each `user_id, rows` and each `row` (lot) within: derive `entry_date` from `row["entry_date"]`; call `price_lookup.closes_between(row["ticker"], entry_date, today or datetime.now().date())`; skip the lot if empty; scale to absolute VND (`* 1000.0` per element); look up `stacking_5d.get(ticker)`, `stacking_20d.get(ticker)`, `main._LATEST_REGIME_BY_TICKER.get(ticker)`; call `portfolio_guard.evaluate_position(...)` with `CONFIG.trading.stop_loss_pct`, `CONFIG.trading.take_profit_pct`, `CONFIG.trading.portfolio_guard_trailing_pct`; if any trigger fired, record the lot (with its rendered display fields) under that user.
    5. Collect the union of every triggered ticker across every user.
    6. Return `{}` if the union is empty (event-only gating).
    7. If `CONFIG.trading.portfolio_guard_llm_enabled`, call `evaluate_trades_batch({"5d": stacking_5d, "20d": stacking_20d}, sorted(union))` in a try/except with `({}, {})` fallback on any exception; else use `({}, {})` directly without calling it.
    8. Call `mr_score_tickers(sorted(union))`.
    9. For each user with >=1 triggered lot, call `portfolio_guard.build_guard_alert_card(...)` and collect into the result dict.
    10. Return the result dict.
14. **Implement `notify_portfolio_guard() -> int`** in `main.py`, directly after `_run_guard_for_users`: return `0` immediately (with a log line) if `not CONFIG.trading.portfolio_guard_enabled`; else call `portfolio_guard.load_guard_positions()`, return `0` if empty; group into `positions_by_user`; call `_run_guard_for_users`; for each `(user_id, card_html)` call `TelegramBot().send_text_to_chat(user_id, card_html, label="portfolio_guard")`; log a summary (`N alerted / M total users`); return `len(cards)`. Wrap the entire body (from the positions load onward) in `try/except Exception` → log + return `0`, matching `notify_tranche_exits`'s exact shape.
15. **Wire into `full_pipeline` and `inference_only`** — in `full_pipeline` (`main.py:2039-2101`), add `with timed_step("Portfolio guard EOD check"): notify_portfolio_guard()` immediately after the existing `notify_tranche_exits()` block; add the identical block to `inference_only` (`main.py:2104-2120`) immediately after its own `notify_tranche_exits()` call. Update both functions' docstrings to list the new step. Do **not** wire into `daily_inference()` itself, `/suggest_buy20`, `/suggest_buy5`, or the dashboard's `daily_inference_headless` — guard is EOD-sweep-only by design, to avoid redundant Gemini spend on interactive on-demand calls.
16. **Write `tests/test_portfolio_guard.py` Part E (orchestration)** — `_run_guard_for_users` and `notify_portfolio_guard`, all with `predict_v3_horizon`, `evaluate_trades_batch`, `mr_score_tickers`, `Alpha360Generator.load_live_ohlcv_window`, and `TelegramBot` monkeypatched: event-only no-send test, Gemini-exception-fallback test, `portfolio_guard_llm_enabled=False` skip-call test (assert `evaluate_trades_batch` never invoked), `portfolio_guard_enabled=False` no-op test (assert zero DB calls), cron-row-skipped end-to-end test, per-user `chat_id == user_id` routing test. Run green, then run `pytest --collect-only -q` to confirm all 12 `import main` test files still collect cleanly.
17. **(Optional) Add `guard_command` to `src/utils/telegram_bot.py`** — mirrors `suggest_sell_command` (`telegram_bot.py:899-983`) exactly: extract `user_id`, log request, reply `EMPTY_PORTFOLIO_MESSAGE` if `load_guard_positions(user_id=user_id)` is empty, post a wait message, run `_run_guard_for_users({user_id: positions}, ...)` via `asyncio.to_thread`, reply with the card via `_send_or_reply_chunks`/`_split_html_report` if present else the "danh mục ổn định" message. Register in `build_application` (near the `/suggest_sell` handler), add to `_BOT_COMMANDS` (between `suggest_sell` and `rebalance`) and `HELP_TEXT`. Do **not** add to `_EDIT_ONLY_COMMANDS` (read-only, matches `/suggest_sell`). Then run the **full** `pytest -q` suite and perform the manual dry run in [Verification Evidence](#verification-evidence).

---

## Risks and Mitigations

**Risk 1 (HIGHEST) — price-scale misclassification produces a false extreme alert.**
Mitigation: dedicated, precedent-cited `normalize_entry_price_vnd` (see [Critical Investigation Finding](#critical-investigation-finding-price-scale-ambiguity)), exhaustive boundary unit tests (thousands / absolute / exact-threshold / zero-negative), explicit non-goal of touching the dashboard's separate, differently-assumed code path.

**Risk 2 — circular import between `portfolio_guard.py` and `main.py`.**
Mitigation: hard constraint, stated in the module docstring and enforced by code review at EXECUTE time — `portfolio_guard.py` never imports `main`, even lazily; every `main`-anchored call is orchestrated from `main.py`'s own new functions.

**Risk 3 — Gemini cost overrun.**
Mitigation: exactly one `evaluate_trades_batch` call per EOD run across the union of all triggered tickers (never per-user, never per-trigger); `portfolio_guard_llm_enabled` kill-switch; the other four triggers plus MR scoring are LLM-free.

**Risk 4 — Telegram send to a non-chat `user_id` (e.g., the dashboard's `"local"` pseudo-user).**
Mitigation: `_send_to_one`'s existing per-chat try/except (preserved unchanged by the Phase 3 refactor) degrades to a logged warning and continues to the next user — never blocks or crashes the sweep.

**Risk 5 — `main.py` import-surface breakage for the 12 test files that `import main` directly** (`test_sentiment_paperlog.py`, `test_dashboard_persist_gate.py`, `test_universe_resolution.py`, `test_event_overrides.py`, `test_daily_inference_integration.py`, `test_dispatch_regime_sizing.py`, `test_strategy_serve_path.py`, `test_serve_resilience.py`, `test_rescue_loop.py`, `test_select_candidates.py`, `test_main_logic.py`, `test_feature_serve.py`).
Mitigation: purely additive changes to `main.py` (no existing public function signature changes), one lightweight new top-level import (`from src.trading import portfolio_guard`, itself import-light per Risk 2's constraint), full suite run as the final checklist step before closeout.

**Risk 6 — multi-lot same-ticker collisions in trigger evaluation / card rendering.**
Mitigation: explicit per-lot (not per-ticker) evaluation, dedicated test coverage, entry-date-qualified display so two lots of the same ticker never silently merge.

**Risk 7 — `TradingConfig` field additions break a strict-shape test.**
Verified: no test in the repo asserts `TradingConfig`'s exact field count/list (`grep` for `fields(TradingConfig)`/`dataclasses.fields`/`__dataclass_fields__` returned zero hits). Low risk, no mitigation needed beyond the full-suite run.

**Risk 8 — CA-gap detection is scale-invariant but relies on a dense-enough daily close series.**
Mitigation: reuses `price_lookup.has_ca_gap` exactly as-is (no new logic); a sparse/gapped series is the same limitation already accepted elsewhere in the codebase, not hardened further here.

**Open design decision surfaced, not silently resolved — model-flip horizon logic:** the approved design says "dual-horizon predict ... argmax decision == SELL (0)" without specifying whether one or both horizons must agree. This plan chooses **OR-across-horizons** (either T+5 or T+20 argmax == SELL fires the trigger) because the guard's stated purpose is early protective warning (catching a drawdown *before* it fully develops), and Stage 1 must never suppress information — a false positive here costs the user one extra line in a message they can ignore; a false negative silently reproduces the exact VHM-style miss this feature exists to prevent. The rejected alternative (AND-across-both-horizons, mirroring `quant_agent_arbitrator.py`'s own conservative "both DOWN → full exit" rule) would be more conservative/fewer alerts but risks the same silent-miss failure mode. Flagged explicitly for the user to override if this reasoning is unwanted.

**Open design decision surfaced — CA-gap downgrade scope:** the approved design scopes the CA-gap downgrade to stop-loss and trailing-stop only. Take-profit shares the exact same `pnl_pct` computation and is therefore equally exposed to CA-gap corruption (a stock dividend could produce a phantom "+150% take-profit"). This plan implements **exactly as approved** (stop-loss + trailing-stop only) and flags take-profit's shared exposure as a candidate for a future iteration rather than silently expanding scope.

## Integration Notes

**Dependencies (all pre-existing, no new packages):**
- `main.predict_v3_horizon`, `main._LATEST_REGIME_BY_TICKER`, `main.SHORT_HORIZON`, `main.evaluate_trades_batch`, `main.mr_score_tickers`, `main.Alpha360Generator` — all already defined/imported in `main.py`.
- `src.data.price_lookup.closes_between`, `.has_ca_gap` — already defined, pure/near-pure.
- `src.trading.regime_policy.NO_TRADE_REGIMES` — already defined.
- `src.features.market_regime.regime_label_vi` — already defined, already imported in `main.py`.
- `src.trading.portfolio_manager.CRON_USER_ID` — already defined.
- `src.reports.builders._MR_SELL_VETO` — already defined, already imported in `main.py`.
- `src.utils.telegram_alerter.TelegramBot` — extended, not replaced.

**`_LATEST_REGIME_BY_TICKER` cache mechanism — investigated and confirmed (per the approved design's explicit "verify and specify" instruction):** `main._compute_v3_features` (called internally by `predict_v3_horizon`) stashes `market_regime` for **every ticker present in the panel it is given** (`main.py:419-431`), not just tickers inside the VN30/ADV-liquid candidate-universe gate (that gate is applied separately, later, only for BUY-dispatch selection). Because this feature builds its **own** full-universe `latest_df` (same pattern as `/verify`/`/rebalance`) and calls `predict_v3_horizon` on it for the model-flip trigger anyway, the regime cache is refreshed for every held ticker that has a parquet shard and survives feature-panel construction — **the guard is self-sufficient and does not depend on `daily_inference()` having already run earlier in the same process**, even though in the actual `full_pipeline`/`inference_only` wiring it always will have. A held ticker with no parquet shard at all (delisted) will be absent from every guard signal (predictions, regime, and price-based triggers), degrading to "no triggers for this ticker" — not a crash.

**Environment:** no new environment variables. Reuses `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID_1`/`_2` (indirectly, via `TelegramBot`), and `GEMINI_API_KEY` (indirectly, via `evaluate_trades_batch`, only when `portfolio_guard_llm_enabled=True` and at least one trigger fired).

**Data model:** no new tables, no schema changes. Read-only access to the existing `portfolio` table (`user_id`, `ticker`, `volume`, `price`, `added_at` — `src/data/db_engine.py:287-293`).

**Simplified data flow:**

```
portfolio (DuckDB, user_id != "cron")
   -> load_guard_positions()                                [I/O, portfolio_guard.py]
   -> grouped by user_id                                     [main.py orchestration]
   -> Alpha360Generator.load_live_ohlcv_window()              [main.py, existing]
   -> predict_v3_horizon(5d) + predict_v3_horizon(20d)        [main.py, existing; refreshes regime cache]
   -> per lot: price_lookup.closes_between(entry_date..today) [thousands VND]
   -> normalize_entry_price_vnd() + x1000 on closes           [-> absolute VND, portfolio_guard.py]
   -> evaluate_position() per lot                             [pure, portfolio_guard.py]
   -> union of triggered tickers across all users
        -> (if llm_enabled) evaluate_trades_batch()  [main.py, existing, ONE call]
        -> mr_score_tickers()                        [main.py, existing, ONE call]
   -> build_guard_alert_card() per user                       [pure, portfolio_guard.py]
   -> TelegramBot().send_text_to_chat(user_id, card)           [chat_id == user_id]
```

---

## Touchpoints

| File | Change | Approx. anchor (as of 13-07-26, may drift) |
|---|---|---|
| `src/trading/portfolio_guard.py` | **NEW** — pure trigger engine + card builder + position loader | n/a |
| `main.py` | New `_run_guard_for_users`, `notify_portfolio_guard`; new `from src.trading import portfolio_guard` import; two new call sites | imports ~line 42-50; new functions after `notify_tranche_exits` (~line 2172); `full_pipeline` call site ~line 2101; `inference_only` call site ~line 2120 |
| `config/settings.py` | 3 new `TradingConfig` fields | after `serve_adv_window: int = 20` (~line 118) |
| `config/settings.json` | 3 new keys in `"trading"` object | end of `"trading"` block (~line 53) |
| `src/utils/telegram_alerter.py` | Extract `_send_to_one`; add `send_text_to_chat` | `_dispatch` (~line 301-325) |
| `src/utils/telegram_bot.py` (optional, Phase 5) | New `guard_command`; registration in `build_application`; `_BOT_COMMANDS` + `HELP_TEXT` entries | mirrors `suggest_sell_command` ~line 899-983; `_BOT_COMMANDS` ~line 1528-1543; `HELP_TEXT` ~line 124-140; `build_application` ~line 1718-1727 |
| `tests/test_portfolio_guard.py` | **NEW** — full test suite for this feature | n/a |

## Public Contracts

| Symbol | Signature | Notes |
|---|---|---|
| `portfolio_guard.normalize_entry_price_vnd` | `(raw_price: float) -> float` | pure |
| `portfolio_guard.evaluate_position` | `(position: dict, closes_since_entry_abs: list[float], prediction_5d: list[float] \| None, prediction_20d: list[float] \| None, regime: int \| None, *, stop_loss_pct: float, take_profit_pct: float, trailing_pct: float) -> list[dict]` | pure |
| `portfolio_guard.build_guard_alert_card` | `(ticker_lots: list[dict], enrichment: tuple[dict, dict] \| None, mr_scores: dict) -> str` | pure |
| `portfolio_guard.load_guard_positions` | `(db_path: str \| None = None, user_id: str \| None = None) -> list[dict]` | I/O, never raises |
| `main._run_guard_for_users` | `(positions_by_user: dict[str, list[dict]], db_path: str \| None = None, today: date \| None = None) -> dict[str, str]` | orchestration |
| `main.notify_portfolio_guard` | `() -> int` | orchestration, never raises, returns count of users alerted |
| `TelegramBot.send_text_to_chat` | `(self, chat_id: str, html_text: str, label: str = "alert") -> None` | new public method |
| `TelegramBot._send_to_one` | `(self, chat_id: str, msg: str, ticker: str = "N/A") -> None` | new private helper, extracted from `_dispatch` |
| `CONFIG.trading.portfolio_guard_enabled` | `bool`, default `True` | settings.json kill-switch |
| `CONFIG.trading.portfolio_guard_trailing_pct` | `float`, default `0.08` | |
| `CONFIG.trading.portfolio_guard_llm_enabled` | `bool`, default `True` | settings.json kill-switch |
| `guard_command` (optional) | `(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None` | new bot handler, not edit-only |

**External behavior that must remain compatible:** every existing `TelegramBot` method (`send_signal_alert`, `send_text_alert`, `_build_message`) keeps its current signature and byte-for-byte output; every existing `main.py` public function (`daily_inference`, `verify_single_ticker`, `rebalance_portfolio`, `full_pipeline`, `inference_only`, `notify_tranche_exits`) keeps its current signature; `Config.from_json()` continues to construct successfully from a `settings.json` that lacks the three new keys (dataclass defaults cover it) as well as from one that has them.

## Blast Radius

- **`main.py`** is a hub-adjacent file (per `process/context/all-context.md`'s hub list, `daily_inference` alone is degree 84) but this plan does **not** modify `daily_inference`, `_dispatch_signals`, `_select_candidates`, or any existing predict/report function — it only adds two new functions and two small additive call-site insertions in `full_pipeline`/`inference_only`, neither of which has any existing direct test (confirmed: zero hits searching `tests/` for `full_pipeline(` or `inference_only(` or `def test.*full_pipeline`/`inference_only`). Residual risk is therefore concentrated in **import-time collection** for the 12 files that `import main` (Risk 5), not in behavioral regression of existing tested code paths.
- **`config/settings.py` / `settings.json`** — additive only; confirmed no test asserts `TradingConfig`'s exact field shape (Risk 7).
- **`src/utils/telegram_alerter.py`** — confirmed no existing test touches `TelegramBot._dispatch`'s internals or `requests.post` directly (only `_build_message`'s pure HTML output is tested, in `test_cards.py`); the `_send_to_one` extraction is a behavior-preserving refactor.
- **`src/utils/telegram_bot.py`** (Phase 5, optional) — confirmed no existing test calls `build_application()` or references `guard_command`/`EMPTY_PORTFOLIO_MESSAGE` from `tests/`; low risk, additive only.
- **No blast radius** into `src/backtest/`, `src/models/`, `src/labels/`, `train_models.py`, `run_backtest.py`, or any parquet/DuckDB schema — this feature is entirely serve-path, read-only against `portfolio`.

## Verification Evidence

**Automated:**
1. `pytest -q tests/test_portfolio_guard.py` — new file, all tests green (see per-phase test scoping in [Execution Brief](#execution-brief) and the full test list embedded in [Implementation Checklist](#implementation-checklist) steps 6-10, 16).
2. `pytest -q` — full suite, currently 591 tests across 47 files; must stay 100% green with the new file's tests added on top.
3. `pytest --collect-only -q` — sanity check that all 12 `import main` test files still collect without error (guards against Risk 2/Risk 5's circular-import failure mode, which would otherwise surface as a collection-time error across many unrelated test files simultaneously).

**Manual dry run (safe, mock-mode by default):**
- Before running, confirm the current live `data/quant_v6_core.duckdb` has no unexpected non-cron `portfolio` rows you don't want alerted-on: inspect via a read-only query such as `SELECT DISTINCT user_id FROM portfolio WHERE user_id != 'cron'`.
- With `TELEGRAM_BOT_TOKEN` unset or left at the `.env.example` placeholder, `TelegramBot`'s existing mock-mode short-circuit (`self.bot_token == "YOUR_BOT_TOKEN"`) makes any send **log-only** — this is the existing, already-relied-upon safety mechanism for manual testing, not new behavior introduced by this plan.
- Run `python -c "import main; print(main.notify_portfolio_guard())"` and inspect the log output for a correctly-scaled, correctly-worded card (or a clean "0 users alerted" line if the current DB has no triggered positions).
- **Caution:** if `.env` has real `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID_1`/`_2` configured, this command **will** push a real Telegram message to whichever non-cron `portfolio` rows currently exist and currently trigger. Prefer running against a scratch copy of the DuckDB file, or verifying the live table is empty/non-triggering first, before running with real credentials.
- Confirm zero new rows were written to `portfolio`, `trade_history`, `signal_ledger`, or `sentiment_entry_paperlog` after the dry run (row-count diff before/after).

**Test context reference:** `process/context/tests/all-tests.md` (pytest runner, in-memory DuckDB stub conventions, `conftest.py`'s prefer-real-module-else-stub pattern) governs how `tests/test_portfolio_guard.py` should be structured — pure-function tests need no DB/stubs at all; the `load_guard_positions` and orchestration tests use an in-memory DuckDB connection seeded directly (mirrors `signal_ledger`/`sentiment_paperlog` test conventions) rather than the module-mocking fallback in `tests/conftest.py` (which only activates when the real ML stack is absent).

## Resume and Execution Handoff

Repo-wide context: `process/context/all-context.md` (architecture, conventions, current-state routing).

If EXECUTE resumes this plan after a context reset or compaction, read in this order:
1. This plan file in full, especially [Critical Investigation Finding](#critical-investigation-finding-price-scale-ambiguity) (the price-scale decision is binding and must not be re-litigated without returning to PLAN) and [Functional Requirements](#functional-requirements) (the exact Vietnamese copy contract).
2. `process/context/all-context.md` for repo-wide conventions (Polars-native, strict typing, config pattern, price-scale convention).
3. `src/trading/intraday_scanner.py` as the structural template for `portfolio_guard.py` (pure/IO layering, hard-contract docstring style).
4. `src/utils/telegram_alerter.py` (current state) before touching Phase 3, to confirm `_dispatch`'s exact current body hasn't drifted from what this plan documents.

**Dependency order is strict** — later phases depend on earlier ones' public contracts:
`Phase 1 (pure engine) -> Phase 2 (loader + config) -> Phase 3 (Telegram single-send) -> Phase 4 (main.py orchestration, depends on 1+2+3) -> Phase 5 (optional /guard command, depends on 4) -> Phase 6 (full verification)`.

A partially-complete resume should check: does `src/trading/portfolio_guard.py` exist and does `pytest -q tests/test_portfolio_guard.py` currently pass for whatever subset exists? That single command tells a resuming executor exactly how far Phases 1-3 (and partially 4) have progressed, since each phase's tests are additive to the same file.

If scope changes mid-flight (e.g., the model-flip OR-vs-AND decision or the CA-gap take-profit-scope decision gets overridden), stop, update this plan's [Risks and Mitigations](#risks-and-mitigations) "Open design decision" entries and the [Functional Requirements](#functional-requirements) table, then continue — do not silently diverge from what's written here.

## Cursor + RIPER-5 Guidance

**Cursor Plan mode:**
- Import the [Implementation Checklist](#implementation-checklist) (17 items) directly.
- Execute continuously in one session (SIMPLE plan) — no approval gate between checklist items, but run each phase's stated test command before moving to the next phase.
- Check off items as completed; after Phase 6, run the full [Verification Evidence](#verification-evidence) block before declaring done.

**RIPER-5 mode:**
- **RESEARCH:** Complete — this plan embeds the full investigation (price-scale conflict, `_LATEST_REGIME_BY_TICKER` mechanism, existing test coverage, blast-radius confirmation via direct file reads and targeted greps of `tests/`).
- **INNOVATE:** Complete — approach was pre-approved by the user before this PLAN session; open sub-decisions (model-flip horizon logic, CA-gap scope) are resolved and flagged in [Risks and Mitigations](#risks-and-mitigations), not left ambiguous.
- **PLAN:** Current — this document.
- **EXECUTE:** Next — requires the user to explicitly say `ENTER EXECUTE MODE`. Do not auto-transition.
- **REVIEW:** After execution — validate implementation matches this plan's exact Vietnamese copy contract and trigger math; flag any deviation explicitly rather than silently reconciling it.

**Next Step:** Review this plan. Say `ENTER EXECUTE MODE` when ready to implement Phase 1.
