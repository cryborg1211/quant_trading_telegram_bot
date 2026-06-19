# P1 — UI Skeleton Build Plan

**Date:** 2026-06-19
**Status:** ACTIVE — PLANNED (unblocked after P0 report confirms no module-level blockers)
**Phase:** P1 of local-dashboard program
**Complexity:** COMPLEX (multi-file new package; 6 tabs; form wiring; Settings writes .env)
**Umbrella plan:** `process/features/local-dashboard/active/dashboard-umbrella_PLAN_19-06-26.md`
**Report target:** `process/features/local-dashboard/reports/p1-ui-build_REPORT_19-06-26.md`

---

## Objective

Build a fully navigable static Streamlit skeleton: 6 tabs render without error, all UI surfaces
(cards, tables, toggles, forms) are visible with stub data, and the Settings form writes `.env` +
`config/settings.json` for real. No real inference is wired — all data surfaces use hardcoded
stub dicts/lists. The goal is a runnable, reviewable shell that proves the layout before P2 wires
real logic.

**Gate: `streamlit run dashboard/app.py` starts cleanly, renders all 6 tabs, Settings persists
fields to disk, no ImportError, no unhandled exception on first load.**

---

## Touchpoints (new files only — no existing files modified in P1)

```
dashboard/
  __init__.py                    -- empty, makes dashboard a package
  app.py                         -- entry: st.set_page_config + 6-tab router
  tabs/
    __init__.py
    mua.py                       -- MUA tab: stub buy-signal card list
    giu.py                       -- GIU tab: stub holdings table + add/remove form
    ban.py                       -- BAN tab: stub sell verdicts + stub rebalance text
    verify.py                    -- Verify tab: ticker input + stub dual-horizon card
    audit.py                     -- Audit tab: tuần/tháng toggle + stub post-mortem table
    settings.py                  -- Settings tab: form + .env/.json writer (REAL, not stub)
  components/
    __init__.py
    ticker_card.py               -- Reusable card: action badge, price, signal bar, weight
    signal_bar.py                -- 3-segment horizontal bar (up %, side %, down %)
  utils/
    __init__.py
    headless.py                  -- Empty stubs for P1; real wrappers added in P2
    thread_runner.py             -- Empty stubs for P1; real threading added in P2
run_dashboard.bat                -- Windows launcher: venv bootstrap + streamlit run (P3 scope,
                                    but placeholder .bat created in P1 for completeness)
requirements_dashboard.txt       -- streamlit>=1.35, plus any dashboard-only deps
```

---

## Public Contracts

### `dashboard/app.py`

Entry point called by `streamlit run dashboard/app.py`. Sets page config (wide layout, title
"Quant V4 Dashboard", favicon optional). Creates 6 tab objects via `st.tabs(...)`. Calls each
tab module's `render()` function. No logic here.

Tab label order (Vietnamese, matching approved design):
`["MUA", "GIU", "BAN", "Verify", "Audit", "Settings"]`

### Per-tab module contract

Each tab module in `dashboard/tabs/` exposes exactly one public function:

```
def render() -> None
```

Called by `app.py`. No return value. All Streamlit calls are inside `render()`. No module-level
`st.*` calls that would execute on import.

### `dashboard/components/ticker_card.py`

```
def render_ticker_card(
    ticker: str,
    action: str,          # "MUA" | "GIU" | "BAN"
    price: float,
    prob_up: float,
    prob_side: float,
    prob_down: float,
    sentiment: float,
    weight_pct: float,
    hold_days: int,
    on_add_click: bool = False,  # if True, render "da mua -> them" button
) -> bool                         # returns True if add button was clicked
```

### `dashboard/components/signal_bar.py`

```
def render_signal_bar(prob_up: float, prob_side: float, prob_down: float) -> None
```

Renders a horizontal 3-segment bar using `st.progress` or `st.columns` with colored markdown.
Proportions must sum to 1.0; clamp if not.

### `dashboard/tabs/settings.py`

Real (not stub) in P1. Reads current `.env` and `config/settings.json` on load. Displays masked
fields for:
- `GEMINI_API_KEY` (password field, masked)
- `TELEGRAM_BOT_TOKEN` (password field, masked)
- `TELEGRAM_CHAT_ID` (text field)
- Horizon default (selectbox: T+5 / T+20)
- Sentiment threshold (slider 0.5–0.95, step 0.05)

On "Save" button click:
1. Write `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` to `.env` (append-or-update,
   do not clobber other lines).
2. Write horizon default and sentiment threshold to `config/settings.json` (merge, not overwrite).
3. Show `st.success("Da luu cai dat.")`.

`.env` write must use python-dotenv's `set_key()` to avoid clobbering unrelated vars.
`settings.json` write must read-parse-merge-write (not overwrite the whole file).

---

## Implementation Checklist

### Setup

1. Create `requirements_dashboard.txt` at repo root with `streamlit>=1.35` (and any other
   dashboard-only pip deps identified; do NOT duplicate packages already in `requirements.txt`).
2. Create `dashboard/__init__.py` (empty).
3. Create `dashboard/tabs/__init__.py` (empty).
4. Create `dashboard/components/__init__.py` (empty).
5. Create `dashboard/utils/__init__.py` (empty).

### Components (no Streamlit state dependencies — build first)

6. Create `dashboard/components/signal_bar.py`: implement `render_signal_bar(...)` using
   `st.columns` with colored `st.markdown` blocks. Clamp probabilities so they sum to 1.0.
7. Create `dashboard/components/ticker_card.py`: implement `render_ticker_card(...)`.
   Use `st.container()` with an `st.expander` or plain `st.columns` layout. Action badge via
   `st.markdown` with inline HTML span for color (MUA=green, GIU=yellow, BAN=red). Call
   `render_signal_bar` inside. Return `st.button("Da mua -> them")` result if `on_add_click=True`,
   else return `False`.

### Tab modules — stub data

8. Create `dashboard/tabs/mua.py` with `render()`. Defines a local `STUB_SIGNALS` list of 3 dicts
   (one per fake ticker: HPG, VHM, TCB) with all `render_ticker_card` fields. Renders a
   T+5 / T+20 toggle (`st.radio`) and calls `render_ticker_card` for each stub signal. The
   quick-add button is rendered (returns True/False) but in P1 does nothing (no session_state
   wiring yet).
9. Create `dashboard/tabs/giu.py` with `render()`. Top section: add-position form with
   `st.form("add_position")` containing ticker text input, quantity number input, entry price
   number input, and submit button — in P1 the submit writes to `st.session_state["pending_add"]`
   but does NOT call PortfolioManager. Holdings table: `st.dataframe` with a stub 3-row DataFrame
   (columns: ma, KL, gia_vao, PnL, lenh, thoat_countdown). Remove button per row is a stub
   `st.button` that does nothing in P1. Summary cards: 4 `st.metric` widgets (von_vao, PnL_today,
   PnL_total, lenh_mo) all showing stub values. "Chay lai" (re-run) button renders but does
   nothing in P1.
10. Create `dashboard/tabs/ban.py` with `render()`. Renders a stub holdings sell-verdict section
    (3 rows, same ticker_card component with action="BAN"). Below it, renders a stub rebalance
    text block in `st.info(...)`.
11. Create `dashboard/tabs/verify.py` with `render()`. Text input for ticker symbol. T+5 / T+20
    result rendered as two `st.metric` columns (prob_up, prob_down, sentiment) using stub values
    whenever the input is non-empty. "Gui Telegram" (push) button renders but does nothing in P1.
12. Create `dashboard/tabs/audit.py` with `render()`. Tuan / Thang toggle via `st.radio`. Stub
    post-mortem table (`st.dataframe`, columns: ma, lenh, gia_vao, net_return, dung_sai). Below
    table: two `st.metric` widgets (hit_rate, vs_VNINDEX) with stub values.
13. Create `dashboard/tabs/settings.py` with `render()`. Full real implementation per Public
    Contracts section above. Use `python-dotenv`'s `dotenv_values()` to read, `set_key()` to
    write. Use `json.loads` / `json.dumps` for `config/settings.json`. Wrap writes in try/except
    and show `st.error(...)` on failure.

### Stub wrappers

14. Create `dashboard/utils/headless.py` with module-level docstring explaining it will contain
    real wrappers in P2. Add stub functions (all `raise NotImplementedError`) for:
    `daily_inference_headless(horizon: int)`, `inference_for_holdings_headless(tickers: list[str])`,
    `verify_single_ticker_headless(ticker: str)`.
15. Create `dashboard/utils/thread_runner.py` with module-level docstring. Add stub
    `run_in_thread(fn, *args, **kwargs)` function body `raise NotImplementedError`.

### Entry point

16. Create `dashboard/app.py`. Set page config: `st.set_page_config(page_title="Quant V4 Dashboard",
    layout="wide")`. Import all 6 tab render functions. Create tabs:
    `tab_mua, tab_giu, tab_ban, tab_verify, tab_audit, tab_settings = st.tabs([...])`.
    Inside each `with tab_X:` block call the corresponding `render()`. Wrap each `render()` call
    in `try/except Exception as e: st.error(f"Tab error: {e}")` to prevent one broken tab from
    crashing the whole app.

### Placeholder launcher

17. Create `run_dashboard.bat` (placeholder content: `@echo off` + `echo P3: launcher not yet
    implemented` + `pause`). Real implementation is P3 scope.

### Smoke test

18. After all files created, verify locally: `streamlit run dashboard/app.py` starts without
    ImportError. Each tab is navigable. Settings Save button writes to `.env` and
    `config/settings.json`. Confirm with `cat .env` and `cat config/settings.json` that the test
    values appear.

---

## Blast Radius

- **New files only.** `dashboard/` is a new package; zero existing source files are touched.
- `requirements_dashboard.txt` is a new file; `requirements.txt` is not modified.
- `run_dashboard.bat` is a new file.
- `config/settings.json` IS written to by Settings tab (step 13) — this is intentional and
  expected; the merge logic must not clobber unrelated keys.
- `.env` IS written to by Settings tab — python-dotenv `set_key()` preserves unrelated lines.
- No test files modified. No `src/`, `main.py`, `config/settings.py` changes.

---

## Test Approach

P1 does not add pytest tests (stub-only UI; no logic to unit test yet).

Manual smoke test (checklist item 18) is the P1 gate:

- `streamlit run dashboard/app.py` runs without error.
- All 6 tabs render visually.
- Settings Save writes to `.env` and `config/settings.json` correctly.
- No tab crashes the app (error boundary in app.py catches and shows st.error).

P2 will add `tests/test_dashboard_headless.py` once real logic is wired.

---

## Failure Modes

| Failure | Root cause | Recovery |
|---|---|---|
| ImportError on `streamlit run` | Missing `__init__.py` or bad relative import | Check each `dashboard/*/` has `__init__.py`; use absolute imports (`from dashboard.components...`) |
| `set_key()` corrupts `.env` | python-dotenv not in requirements_dashboard.txt | Add `python-dotenv` to requirements_dashboard.txt (already in requirements.txt; just needs to be in the venv) |
| `settings.json` overwritten | json.dumps without merge | Implement read-parse-merge-write pattern; test with a pre-existing key |
| Tab crashes entire app | Unguarded exception in render() | `try/except` wrapper in app.py (checklist item 16) |
| signal_bar proportions > 1.0 | Stub data not normalized | Clamp in render_signal_bar; test with stub data |

---

## Dependencies

- P0 report must be complete and must not list a "P1 BLOCKED" verdict.
- If P0 finds module-level side effects on `import main`, the decoupling fix must be applied
  (additive) before P1 execution begins (even though P1 does not import main — it removes
  a risk for P2).
- `streamlit` must be installable in the dev environment (not in requirements.txt yet).

---

## Rollback

P1 is entirely new files. Rollback = `git rm -r dashboard/ run_dashboard.bat requirements_dashboard.txt`. No existing behaviour is changed.

---

## Verification Evidence

- `streamlit run dashboard/app.py` exits cleanly with no traceback (stdout evidence in report).
- Screenshot or session log showing all 6 tabs rendered.
- `.env` diff confirming Settings write.
- `config/settings.json` content confirming Settings write.
- `pytest` (existing 238 tests) still green — confirm no regression.

---

## Resume and Execution Handoff

Pass this file to **vc-execute-agent**. The agent implements checklist steps 1–18 in order.
All new files go under `dashboard/` (new package root) and `run_dashboard.bat` /
`requirements_dashboard.txt` at repo root.

After P1 gate passes, record findings in the report target and advance to P2 (plan file to be
created before execution).
