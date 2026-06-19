# Local Dashboard — Program Goal Charter + Umbrella Orchestration Plan

**Date:** 2026-06-19
**Status:** ACTIVE
**Complexity:** Phase Program (P0–P5)
**Feature folder:** `process/features/local-dashboard/`

---

## Program Goal Charter

### North star

Deliver a fully self-contained Streamlit dashboard that replaces the Telegram interactive bot for a
non-developer user on their own Windows laptop — local compute, their own Gemini key, send-only
Telegram push notifications, no polling conflict, no ML-DLL installer nightmare.

### Definition of done

An unattended executor can:

1. Drop the Inno Setup `setup.exe` on a clean Windows laptop, run it, and have a working venv.
2. Launch `run_dashboard.bat` (or the Start-menu shortcut) and see the Streamlit UI open in a
   browser within 30 seconds with no further configuration beyond the first-run Settings tab.
3. Open MUA, see live buy signals scored by T+5 and T+20 models (toggle), click a ticker card to
   quick-add it to GIU.
4. Open GIU, add / remove positions, see live PnL and per-name sell verdict inline, see exit
   countdown from signal_ledger.
5. Open BAN, see sell recommendations for all holdings and a Gemini rebalance suggestion.
6. Open Verify, type a ticker, get a dual-horizon card, push a send-only Telegram notification.
7. Open Audit, toggle Tuan / Thang, see hit-rate post-mortem sourced from audit_evaluator.
8. Open Settings, save Gemini key and telegram chat_id to .env + config/settings.json — persists
   across restarts.

### What "verified" means (program level)

- P0: callable map document exists; every listed fn runs headless in a unit harness with no
  Telegram side-effect; any coupling is listed with a mitigation plan.
- P1: `streamlit run dashboard/app.py` renders all 6 tabs with stub data; no import error;
  Settings form writes a temp .env without crashing.
- P2: each tab calls real fns; inference runs in a background thread without freezing the UI;
  send-only push works; quick-add pre-fills GIU form.
- P3: `run_dashboard.bat` creates venv on first run, activates it, opens browser — tested on a
  fresh directory without pre-installed packages.
- P4: Inno Setup builds `setup.exe`; install on a clean Windows test environment produces a working
  dashboard with bundled offline wheels + models + data.
- P5: missing key, Gemini-down, stale data, duplicate-launch edge cases handled gracefully with
  visible banners or friendly error states.

### Scope tiers → phase mapping

- Tier 1 Foundation (reuse + static UI) → Phases P0, P1
- Tier 2 Logic wiring + launch → Phases P2, P3
- Tier 3 Packaging + hardening → Phases P4, P5
- This program retires Tiers 1–3.

### Explicitly out of scope (deferred tier)

- Tier 4 (not addressed by this program):
  - Live OHLCV crawling from within the dashboard (user data assumed pre-populated or daily cron
    populates it on the VPS; dashboard is read-only on the data layer)
  - Multi-user / multi-laptop sync (single-user local-only)
  - CI pipeline for the installer build
  - Mobile / tablet layout
  - Dark-mode theming beyond Streamlit defaults

### Hard safety constraints (non-negotiable, per phase)

- Never modify existing serve-path, bot, or backtest source files during P0–P2. Dashboard is
  additive only; reuse fns are called as-is, not patched.
- Never run live Telegram polling inside the dashboard process (send-only alerter only; no
  `Application.run_polling()` call anywhere in `dashboard/`).
- Never write real API keys or secrets to plan files, reports, or references.
- Never bundle or commit `.env` with real values into the installer package or git.
- Keep process/plan/context commits separate from execution commits; commit each phase before
  moving to the next.

---

## Phase Sequencing

| Phase | ID | Plan file | Goal | Depends on | Status |
|---|---|---|---|---|---|
| Reuse audit | P0 | `p0-reuse-audit_PLAN_19-06-26.md` | Verify callable map, decouple Telegram | — | PLANNED |
| UI skeleton | P1 | `p1-ui-build_PLAN_19-06-26.md` | Static Streamlit 6-tab shell | P0 findings | PLANNED |
| Logic linking | P2 | *(create before executing P2)* | Wire tabs to real fns, threading | P1 green | PLANNED |
| Launcher | P3 | *(create before executing P3)* | venv bootstrap + browser open | P2 green | PLANNED |
| Installer | P4 | *(create before executing P4)* | Inno Setup setup.exe | P3 green | PLANNED |
| Hardening | P5 | *(create before executing P5)* | Error states, guards, banners | P4 green | PLANNED |

P2–P5 plans are intentionally deferred; they must be created (with a fresh research pass) before
execution of each respective phase, because P0 and P1 findings will change the exact implementation
details.

### Order-dependency note (keystone)

GIU, BAN, and Audit all READ the user portfolio. The `/add` position form in GIU is the
keystone: an empty portfolio produces empty BAN recommendations, no PnL summary, and thin Audit
results. This must be prominently documented in P2 wiring order: implement GIU add/remove first,
then BAN, then Audit.

---

## New Package Location

All new dashboard code lives under `dashboard/` at the repo root:

```
dashboard/
  app.py              -- Streamlit entry: st.set_page_config, tab router
  tabs/
    mua.py            -- MUA tab (buy signals)
    giu.py            -- GIU tab (portfolio)
    ban.py            -- BAN tab (sell / rebalance)
    verify.py         -- Verify tab (single ticker)
    audit.py          -- Audit tab (post-mortem)
    settings.py       -- Settings tab (.env + settings.json writer)
  components/
    ticker_card.py    -- Reusable per-ticker card component
    signal_bar.py     -- 3-segment up/side/down probability bar
  utils/
    headless.py       -- Thin wrappers that call main.py fns with broadcast=False
    thread_runner.py  -- st.cache_data + background thread helpers
run_dashboard.bat     -- Windows launcher (venv bootstrap + streamlit run)
requirements_dashboard.txt  -- Streamlit + dashboard-only additions (if any)
installer/
  setup.iss           -- Inno Setup script (P4)
  offline_wheels/     -- Pre-downloaded .whl files for offline install (P4)
```

No modifications to `src/`, `main.py`, `config/`, or `tests/` until P2 at the earliest, and then
only additive (new headless wrappers, never mutating existing fn signatures).

---

## Reuse Points (confirmed from source)

| Dashboard tab | Reuse fn / module | Headless call signature |
|---|---|---|
| MUA (buy signals) | `main.daily_inference(broadcast=False, horizon=H)` | Returns `(html, signal_data_list)` |
| BAN (sell) | `main.inference_for_holdings(holding_tickers, window_rows=120)` | Returns HTML string |
| BAN (rebalance) | `main.inference_for_holdings` + Gemini arbitrator path | Same fn, different filter |
| GIU (holdings) | `PortfolioManager(user_id=...)` methods | add/remove/query `portfolio` table |
| GIU (exit countdown) | `signal_ledger.list_open()`, `signal_ledger.check_exits_due()` | `src/trading/signal_ledger.py` |
| Verify | `main.verify_single_ticker(ticker)` | Returns HTML string |
| Audit | `audit_evaluator.run_post_mortem(user_id, days)` | `src/utils/audit_evaluator.py:70` |
| Push notification | `telegram_alerter.send_text_alert` (send-only) | `src/utils/telegram_alerter.py` |

Telegram coupling audit (P0 must confirm or refute for each):
- `daily_inference(broadcast=False)` — `broadcast` param exists; alerter call is gated on it.
  Low coupling risk.
- `inference_for_holdings` — no broadcast param in current signature. P0 must check if it calls
  alerter internally.
- `verify_single_ticker` — returns HTML string; no known broadcast. P0 must confirm.
- `run_post_mortem` — pure analytical fn; no Telegram expected. P0 must confirm.

---

## Risks

| Risk | Severity | Phase | Mitigation |
|---|---|---|---|
| Bundle size (models/ + data/ + offline wheels) | HIGH | P4 | Size-audit in P3; consider crawl-on-first-run for data/ instead of bundling; document tradeoff |
| `data/` staleness — user may not run daily cron | MEDIUM | P2, P5 | Data freshness banner showing last parquet mtime; P5 hardens error state |
| Offline wheels availability | HIGH | P4 | pip download to `installer/offline_wheels/` during P4 build step; audit package list in P3 |
| Streamlit UI freeze on heavy inference | MEDIUM | P2 | All inference calls in `st.cache_data` + background thread; spinner overlay |
| `inference_for_holdings` Telegram coupling | MEDIUM | P0 | P0 audit is the gate; if coupling found, headless wrapper added in P2 (additive, no existing code change) |
| send-only token extractable from .env | LOW (accepted) | P1 | Documented as accepted risk; .env is user-local, not committed |
| Inno Setup python-embeddable DLL conflicts | MEDIUM | P4 | No PyInstaller; python-embeddable avoids this; still must test on clean machine |
| Multiple Telegram pollers (409 Conflict) | n/a — by design | — | Dashboard never polls; send-only alerter only; no ApplicationBuilder anywhere in dashboard/ |

---

## Reports and References Paths

- Phase reports: `process/features/local-dashboard/reports/`
- Research references: `process/features/local-dashboard/references/`

---

## Resume and Execution Handoff

Entry agent: run the **per-phase loop** from `process/development-protocols/phase-programs.md` for
each phase in order.

Current next action: P0 (research-agent task). Pass
`process/features/local-dashboard/active/p0-reuse-audit_PLAN_19-06-26.md` to the research agent.

After P0 completes and its findings are durable-captured, run P1 using
`process/features/local-dashboard/active/p1-ui-build_PLAN_19-06-26.md`.

Do NOT execute P2–P5 without first creating their dedicated plan files (re-research required at
each phase entry).
