# P2 — Logic Linking Plan

**Date:** 2026-06-19
**Status:** ACTIVE — ready to execute (P1 gate green)
**Phase:** P2 of local-dashboard program
**Complexity:** COMPLEX (multi-file wiring; one approved existing-source change; threading primitives; cross-tab state)
**Umbrella plan:** `process/features/local-dashboard/active/dashboard-umbrella_PLAN_19-06-26.md`
**Report target:** `process/features/local-dashboard/reports/p2-logic-linking_REPORT_19-06-26.md`

---

## Objective

Replace every P1 stub in `dashboard/` with calls to real serve-path functions. After P2 the
dashboard renders live buy signals, live portfolio holdings with PnL, live sell verdicts, live
verify results, send-only Telegram push, and a Gemini-backed audit post-mortem. Heavy inference
calls run on a background thread so the Streamlit UI does not freeze.

**Gate: `streamlit run dashboard/app.py` renders all 6 tabs with live data; persist=False
inference produces no new rows in any DuckDB table; existing 246 tests remain green; at least
one new pytest test (`tests/test_dashboard_persist_gate.py`) covers the persist gate.**

---

## Settled Decisions (user-approved — do not re-litigate)

| # | Decision | Detail |
|---|---|---|
| D1 | persist gate | `run_trade_execution` gains `persist: bool = True`; writes in `if persist:` block; `daily_inference` passes it through; default `True` = cron unchanged |
| D2 | user_id | `LOCAL_USER_ID = "local"` constant in `dashboard/utils/headless.py`; optionally overridable via `dashboard_user_id` key in `config/settings.json` |
| D3 | HTML vs structured per tab | MUA structured, GIỮ raw SQL, BÁN/Verify/Audit HTML→markdown |
| D4 | threading | `st.status` (available >=1.28, pin is >=1.35) + `concurrent.futures.ThreadPoolExecutor`; `st.cache_data(ttl=...)` for cheap reads |

---

## Touchpoints

### Existing source files modified (one file, one approved change)

| File | Change | Lines affected |
|---|---|---|
| `main.py` | Add `persist: bool = True` param to `run_trade_execution`; wrap write block in `if persist:`; add passthrough to `daily_inference` | `run_trade_execution:1318`, `daily_inference:934` |

### New / modified dashboard files

| File | P2 action |
|---|---|
| `dashboard/utils/headless.py` | Replace NotImplementedError stubs; add `LOCAL_USER_ID`; add `_read_user_id()`; add `_pnl_ratio()` |
| `dashboard/utils/thread_runner.py` | Implement `run_in_thread` + `ThreadPoolExecutor` + spinner using `st.status` |
| `dashboard/tabs/mua.py` | Wire `daily_inference_headless`; replace STUB_SIGNALS; wire quick-add → session_state → GIỮ pre-fill |
| `dashboard/tabs/giu.py` | Wire portfolio raw SQL (INSERT/DELETE/SELECT); wire PnL via `price_lookup.latest_close`; wire exit countdown via `signal_ledger.list_open()`; wire "Chạy lại" re-run |
| `dashboard/tabs/ban.py` | Wire `inference_for_holdings_headless`; render HTML via `st.markdown(unsafe_allow_html=True)` |
| `dashboard/tabs/verify.py` | Wire `verify_single_ticker_headless`; render HTML via `st.markdown`; wire push button → `TelegramBot().send_text_alert` + `log_user_action` |
| `dashboard/tabs/audit.py` | Wire `run_post_mortem(LOCAL_USER_ID, days)` with `st.cache_data(ttl=300)` |
| `dashboard/app.py` | Update sidebar status dots to reflect real env-key presence (GEMINI_API_KEY, TELEGRAM_BOT_TOKEN) |
| `tests/test_dashboard_persist_gate.py` | New test file for persist gate |

---

## Public Contracts

### `main.run_trade_execution` (modified signature)

```
def run_trade_execution(
    top_buy_signals: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    latest_df: Any,
    xgb_model_5d: Any,
    selected_features_5d: list[str],
    horizon: int = 20,
    broadcast: bool = True,
    event_overrides: dict | None = None,
    persist: bool = True,      # NEW — default True keeps cron byte-for-byte identical
) -> tuple[str, list[dict]]:
```

`persist=False` skips the `manager.update_live_performance`, `manager.process_daily_trades`,
`_log_rl_predictions`, `_backfill_rl_outcomes`, `_log_sentiment_entry_paperlog`, and
`_backfill_paperlog_outcomes` blocks. Signal dispatch (`_dispatch_signals`) still runs — the
caller still receives the full `dispatched_signals` list. `signal_ledger.record_dispatch` is
already `broadcast`-gated and is left unchanged.

### `main.daily_inference` (modified signature)

```
def daily_inference(
    window_rows: int = 120,
    max_candidates: int = 6,
    broadcast: bool = True,
    horizon: int = 20,
    persist: bool = True,      # NEW passthrough to run_trade_execution
) -> tuple[str, list[dict]]:
```

### `dashboard/utils/headless.py` (new public surface)

```python
LOCAL_USER_ID: str  # = "local" or settings.json override

def _read_user_id() -> str
    # reads dashboard_user_id from config/settings.json; falls back to "local"

def _pnl_ratio(entry_price: float, current_close: float | None) -> float | None
    # (current - entry) / entry; returns None if current_close is None

def daily_inference_headless(horizon: int) -> tuple[str, list[dict]]
    # calls main.daily_inference(broadcast=False, persist=False, horizon=horizon)

def inference_for_holdings_headless(tickers: list[str]) -> str
    # calls main.inference_for_holdings(tickers, window_rows=120)

def verify_single_ticker_headless(ticker: str) -> str
    # calls main.verify_single_ticker(ticker, window_rows=120)

def portfolio_list(user_id: str) -> list[dict]
    # SELECT ticker, volume, price, added_at FROM portfolio WHERE user_id=?
    # via DuckDBEngine().conn; returns list of row dicts

def portfolio_add(user_id: str, ticker: str, volume: int, price: float) -> None
    # INSERT INTO portfolio (user_id, ticker, volume, price, added_at) VALUES (?,?,?,?,now())
    # under _audit_lock; raises on duplicate ticker for same user_id

def portfolio_remove(user_id: str, ticker: str) -> None
    # DELETE FROM portfolio WHERE user_id=? AND ticker=?
    # under _audit_lock
```

`_audit_lock` is a module-level `threading.Lock()` that serialises all portfolio writes.

### `dashboard/utils/thread_runner.py` (new public surface)

```python
def run_in_thread(
    fn: Callable[..., Any],
    *args: Any,
    label: str = "Đang xử lý...",
    ttl: int | None = None,
    **kwargs: Any,
) -> Any:
```

Implementation: `ThreadPoolExecutor(max_workers=1).submit(fn, *args, **kwargs)` stored in
`st.session_state` keyed by `(fn.__name__, args_key)`. Wraps execution in `st.status(label)`.
`future.result()` called once done (raises if the fn raised). When `ttl` is set, a second
call within `ttl` seconds returns the cached result from session_state without re-submitting.

---

## Blast Radius

### Existing source change (main.py — single, default-safe)

- `run_trade_execution` signature gains one kwarg with `True` default. All existing callers
  (`daily_inference`, `run_full_pipeline_with_backfill`, any test that calls it directly) pass
  `persist=True` by default — ZERO behavior change unless `persist=False` is explicitly passed.
- `daily_inference` gains the same passthrough kwarg.
- No trading-decision, sizing, regime, or feature-engineering logic is touched.
- No test file is modified except adding the new `test_dashboard_persist_gate.py`.

### Dashboard wiring (all additive, no existing dashboard logic removed)

- Stub bodies in `headless.py` and `thread_runner.py` are replaced; the module-level docstrings
  and function signatures from P1 are preserved (signatures are extended, not changed).
- Tab modules replace `STUB_*` data reads with real calls; the `render()` signatures stay `-> None`.
- `app.py` receives a minor sidebar update (status dot logic); tab dispatch loop unchanged.

### DuckDB writes scoped to dashboard namespace

- Portfolio writes use `user_id=LOCAL_USER_ID` ("local") — separate from cron ("cron") and bot
  (Telegram numeric id) rows. No existing rows touched.
- `log_user_action` appends new audit rows under `user_id="local"` — existing bot rows untouched.

### Hub node blast radius (from code-review-graph)

- `daily_inference` (degree 84): changing its signature is the highest-risk delta in this plan.
  Mitigation: default `persist=True` keeps all existing callers correct without any argument change.
- `run_trade_execution` (degree ~40): same mitigation.
- All other changes are in the new `dashboard/` package (degree 0 from existing graph).

---

## Implementation Checklist

Steps are ordered so each is independently testable before the next. Steps 1–2 touch
existing source; steps 3–9 are dashboard-only additive; step 10 verifies the whole.

### Step 1 — Persist gate in `main.py` (existing source, approved change)

1. Open `main.py`. At line 1327 (end of `run_trade_execution` params), add
   `persist: bool = True` as the last keyword argument.
2. Locate the `with timed_step("Portfolio update/process_daily_trades"):` block (lines 1360–1366).
   Wrap the entire block (both `manager.update_live_performance(...)` and
   `manager.process_daily_trades(...)` calls) in `if persist:`.
3. Locate the `with timed_step("RL prediction logging ..."):` block (lines 1376–1407).
   Wrap the entire block (both `_log_rl_predictions`/`_backfill_rl_outcomes` calls AND the
   `if CONFIG.trading.sentiment_entry_enabled:` sub-block for `_log_sentiment_entry_paperlog`/
   `_backfill_paperlog_outcomes`) in `if persist:`.
4. At `daily_inference` definition (line 934), add `persist: bool = True` as the last keyword
   argument.
5. Locate the call to `run_trade_execution(...)` inside `daily_inference`. Add
   `persist=persist` to that call's argument list.
6. Run `pytest tests/ -x -q` — all 246 existing tests must stay green.

### Step 2 — Persist gate test

7. Create `tests/test_dashboard_persist_gate.py`.
   - Fixture: in-memory DuckDB stub (reuse conftest pattern from existing tests).
   - Test A (`test_persist_false_no_db_writes`): call `run_trade_execution(... persist=False ...)`
     with a mocked `PortfolioManager` and mocked `_log_rl_predictions`/`_backfill_rl_outcomes`/
     `_log_sentiment_entry_paperlog`/`_backfill_paperlog_outcomes`. Assert each mocked function was
     called zero times.
   - Test B (`test_persist_true_default_writes`): call with `persist=True` (or omit kwarg).
     Assert each mocked write function was called at least once.
   - Test C (`test_daily_inference_persist_passthrough`): mock `run_trade_execution` and assert it
     is called with `persist=False` when `daily_inference(..., persist=False)` is invoked.
8. Run `pytest tests/test_dashboard_persist_gate.py -v` — all 3 new tests must pass.
9. Run `pytest tests/ -x -q` — confirm total green count is now 249 (246+3).

### Step 3 — `headless.py` wrappers + `LOCAL_USER_ID`

10. Replace the NotImplementedError stubs in `dashboard/utils/headless.py` with real implementations.
    Add at module top (before any serve imports):
    - `LOCAL_USER_ID: str = _read_user_id()` — reads `config/settings.json` key
      `dashboard_user_id`; falls back to `"local"`.
    - `_audit_lock = threading.Lock()` module-level.
    Import `main` lazily inside each wrapper function body (not at module top) to avoid circular
    import and to keep startup cost deferred.
11. Implement `daily_inference_headless(horizon: int)`:
    - `from main import daily_inference` (inside function body)
    - return `daily_inference(broadcast=False, persist=False, horizon=horizon)`
12. Implement `inference_for_holdings_headless(tickers: list[str])`:
    - `from main import inference_for_holdings` (inside function body)
    - return `inference_for_holdings(tickers, window_rows=120)`
13. Implement `verify_single_ticker_headless(ticker: str)`:
    - `from main import verify_single_ticker` (inside function body)
    - return `verify_single_ticker(ticker, window_rows=120)`
14. Implement `portfolio_list(user_id: str) -> list[dict]`:
    - `from src.data.db_engine import DuckDBEngine`
    - `rows = DuckDBEngine().conn.execute("SELECT ticker, volume, price, added_at FROM portfolio WHERE user_id=?", [user_id]).fetchall()`
    - return list of `{"ticker": r[0], "volume": r[1], "price": r[2], "added_at": r[3]}` dicts.
    - Wrap in try/except; log warning and return `[]` on failure.
15. Implement `portfolio_add(user_id: str, ticker: str, volume: int, price: float) -> None`:
    - Acquire `_audit_lock`.
    - `DuckDBEngine().conn.execute("INSERT INTO portfolio (user_id, ticker, volume, price, added_at) VALUES (?, ?, ?, ?, now())", [user_id, ticker.upper(), volume, price])`
    - On `duckdb.ConstraintException` (duplicate), raise `ValueError(f"{ticker} already in portfolio")`; re-raise nothing else.
16. Implement `portfolio_remove(user_id: str, ticker: str) -> None`:
    - Acquire `_audit_lock`.
    - `DuckDBEngine().conn.execute("DELETE FROM portfolio WHERE user_id=? AND ticker=?", [user_id, ticker.upper()])`
17. Implement `_pnl_ratio(entry_price: float, current_close: float | None) -> float | None`:
    - If `current_close is None` return `None`.
    - Apply VN price-scale: if `current_close < 1000` then `current_close *= 1000` (parquet in thousands-VND).
    - Return `(current_close - entry_price) / entry_price`.

### Step 4 — `thread_runner.py` implementation

18. Replace the NotImplementedError stub in `dashboard/utils/thread_runner.py`.
    - Import `concurrent.futures.ThreadPoolExecutor`, `streamlit as st`, `time`, `hashlib`.
    - Module-level `_executor = ThreadPoolExecutor(max_workers=2)`.
    - `run_in_thread(fn, *args, label="Đang xử lý...", ttl=None, **kwargs)`:
      - Build `cache_key = f"_thread_{fn.__name__}_{hashlib.md5(str(args).encode()).hexdigest()[:8]}"`.
      - If `ttl` is set and `st.session_state.get(cache_key + "_ts")` is within `ttl` seconds of
        `time.time()`, return `st.session_state[cache_key]`.
      - If a future is stored at `st.session_state.get(cache_key + "_fut")` and not done, render
        `st.status(label)` spinner and `st.stop()` (re-run on next tick).
      - If no future or future is done and result not yet cached: submit `_executor.submit(fn, *args, **kwargs)`,
        store future in session_state, render spinner, `st.stop()`.
      - Once `future.done()`: call `result = future.result()` (raises on exception, propagates to
        tab error boundary); store in `st.session_state[cache_key]` and `cache_key + "_ts"`;
        return `result`.

### Step 5 — MUA tab wire

19. In `dashboard/tabs/mua.py`:
    - Remove `STUB_SIGNALS`.
    - Import `run_in_thread` from `dashboard.utils.thread_runner`.
    - Import `daily_inference_headless` from `dashboard.utils.headless`.
    - On horizon toggle change (detect via `st.session_state["mua_horizon"]`), submit
      `run_in_thread(daily_inference_headless, horizon, label="Tính tín hiệu MUA...", ttl=300)`.
    - The function returns `(html, signal_list)`. Use `signal_list` to populate `render_ticker_card`
      calls. Map `dispatched_signals` dict keys (from P0 report: `action`, `ticker`, `price`,
      `prob_up`, `prob_side`, `prob_down`, `sentiment_score`, `suggested_weight`,
      `hold_label`) to the component args. `sentiment_score` → `sentiment`, `suggested_weight * 100`
      → `weight_pct` (convert fraction to pct), `hold_label` splits as `hold_days` numeric.
    - If signal_list is empty, show `st.info("Không có tín hiệu mua cho khung thời gian này.")`.
    - Quick-add click: on `render_ticker_card(..., on_add_click=True)` returning `True`,
      write `st.session_state["giu_prefill"] = {"ticker": sig["ticker"], "price": sig["price"]}`
      and call `st.rerun()` so GIỮ tab picks it up on next render.
    - Remove P1 caption suffix "(dữ liệu mẫu — P1)".

### Step 6 — GIỮ tab wire

20. In `dashboard/tabs/giu.py`:
    - Remove `STUB_HOLDINGS` and `STUB_SUMMARY`.
    - Import `portfolio_list`, `portfolio_add`, `portfolio_remove`, `LOCAL_USER_ID`, `_pnl_ratio`
      from `dashboard.utils.headless`.
    - Import `signal_ledger` from `src.trading.signal_ledger` (read-only `list_open()`).
    - Import `latest_close` from `src.data.price_lookup`.
    - Import `st.cache_data` usage: wrap `portfolio_list` read in `st.cache_data(ttl=30)` with a
      thin cached wrapper `_cached_holdings(user_id)`.
    - Pre-fill: at top of `render()`, check `st.session_state.get("giu_prefill")`. If present, set
      `st.session_state["add_ticker"]` and `st.session_state["add_price"]` from it, then clear
      `st.session_state["giu_prefill"]`.
    - Add form `on_submit`: call `portfolio_add(LOCAL_USER_ID, ticker, volume, price)`. On
      `ValueError` show `st.warning(...)`. On success show `st.success(...)` and `st.cache_data.clear()`.
    - Remove button per row: call `portfolio_remove(LOCAL_USER_ID, ticker)` on click; `st.cache_data.clear()`.
    - Build holdings DataFrame from `portfolio_list(LOCAL_USER_ID)`. For each row compute PnL:
      `current = latest_close(row["ticker"])`, `ratio = _pnl_ratio(row["price"], current)`.
      Append `pnl_str = f"{ratio:+.1%}"` if ratio is not None, else `"N/A"`.
    - Exit countdown: `open_positions = {p["ticker"]: p for p in signal_ledger.list_open()}`.
      For each holding, look up `open_positions.get(ticker, {}).get("sessions_remaining", "-")`.
    - Summary cards: `von_vao` = sum of `volume * price` for all holdings; `lenh_mo` = count.
      PnL today / total: compute from the ratio list above (average or sum per weight); show N/A
      if any close is missing.
    - "Chạy lại" button: call `st.cache_data.clear()` and `st.rerun()`.
    - Remove P1 caption suffix.

### Step 7 — BÁN tab wire

21. In `dashboard/tabs/ban.py`:
    - Remove `STUB_VERDICTS` and `STUB_REBALANCE`.
    - Import `portfolio_list`, `LOCAL_USER_ID` from `dashboard.utils.headless`.
    - Import `inference_for_holdings_headless` from `dashboard.utils.headless`.
    - Import `run_in_thread` from `dashboard.utils.thread_runner`.
    - At top of `render()`, fetch `holdings = portfolio_list(LOCAL_USER_ID)`.
    - If empty, show `st.info("Chưa có vị thế nào trong danh mục — thêm từ tab GIỮ.")`.
    - Else: `tickers = [h["ticker"] for h in holdings]`.
    - Submit `run_in_thread(inference_for_holdings_headless, tickers, label="Phân tích bán/giữ...", ttl=300)`.
    - The function returns HTML. Render via `st.markdown(html, unsafe_allow_html=True)`.
    - Remove stub ticker_card calls (BÁN tab now renders HTML directly, not structured cards).
    - Remove P1 caption suffix.

### Step 8 — Verify tab wire

22. In `dashboard/tabs/verify.py`:
    - Import `verify_single_ticker_headless` from `dashboard.utils.headless`.
    - Import `run_in_thread` from `dashboard.utils.thread_runner`.
    - Import `TelegramBot` from `src.utils.telegram_alerter`.
    - Import `DuckDBEngine` from `src.data.db_engine`.
    - Import `LOCAL_USER_ID` from `dashboard.utils.headless`.
    - Remove `_stub_horizon_result`.
    - When ticker is non-empty and user clicks "Kiểm tra" button (add this button before
      the existing push button): submit `run_in_thread(verify_single_ticker_headless, ticker, label=f"Kiểm tra {ticker}...", ttl=120)`.
    - Render result HTML via `st.markdown(html, unsafe_allow_html=True)`. Store in
      `st.session_state[f"verify_result_{ticker}"]` so the push button has access.
    - "Gửi Telegram" button (existing): retrieve `html` from session_state. If present:
      `TelegramBot().send_text_alert(html, label=ticker)`. Show `st.success("Đã gửi.")`.
      Then: `DuckDBEngine().log_user_action(LOCAL_USER_ID, "verify", ticker)`.
      If no result yet, show `st.warning("Chạy kiểm tra trước khi gửi.")`.
    - Remove stub dual-horizon metric columns.
    - Remove P1 caption suffix.

### Step 9 — Audit tab wire

23. In `dashboard/tabs/audit.py`:
    - Remove `STUB_POSTMORTEM` and `STUB_SUMMARY`.
    - Import `run_post_mortem` from `src.utils.audit_evaluator`.
    - Import `LOCAL_USER_ID` from `dashboard.utils.headless`.
    - Create a module-level cached wrapper:
      `@st.cache_data(ttl=300)` on `_cached_postmortem(user_id: str, days: int) -> str`
      that calls `run_post_mortem(user_id, days=days)`.
    - In `render()`: map window toggle to days: `"Tuần" -> 7, "Tháng" -> 30`.
    - Call `html = _cached_postmortem(LOCAL_USER_ID, days)`.
    - Render via `st.markdown(html, unsafe_allow_html=True)`.
    - Remove stub DataFrame and metric widgets.
    - Remove P1 caption suffix.
    - Add `st.info` note: "Audit chỉ hiển thị dữ liệu từ phiên giao dịch dashboard — các lệnh
      từ bot Telegram sử dụng user_id khác." — onboarding note per D2.

### Step 10 — app.py sidebar status update

24. In `dashboard/app.py`, update `_render_sidebar()` to compute real status dots:
    - GEMINI_API_KEY: `os.environ.get("GEMINI_API_KEY")` — green if present, red if missing.
    - TELEGRAM_BOT_TOKEN: `os.environ.get("TELEGRAM_BOT_TOKEN")` — green if present, red if missing.
    - Data freshness: `max(p.stat().st_mtime for p in Path("data").glob("ohlcv_*.parquet"))`
      vs `time.time() - 86400*3` — green if a parquet is <3 days old, yellow otherwise.
    - Wrap in try/except; fall back to yellow dot on any error.

### Step 11 — Settings tab: user_id field

25. In `dashboard/tabs/settings.py` `render()`:
    - Add a new text input field "Dashboard user_id" defaulting to current `_read_user_id()` value.
    - On Save: write `dashboard_user_id` key to `config/settings.json` via the existing
      read-parse-merge-write pattern (already implemented in P1).
    - Add inline caption: "User_id separates dashboard portfolio from Telegram bot history."

### Step 12 — Smoke verification

26. Run `streamlit run dashboard/app.py` (dev environment). Manual checklist:
    - MUA tab: horizon toggle triggers background inference; spinner appears; cards render with real
      data (or empty state if no signals).
    - GIỮ tab: add a test position (e.g. HPG / 100 / 27000); confirm row appears; PnL shows
      (may be N/A if parquet not current); remove the row; confirm it disappears.
    - BÁN tab: with a holding present, inference runs and HTML renders.
    - Verify tab: enter HPG; click Kiểm tra; HTML renders; click Gửi Telegram (only if
      TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID present in .env).
    - Audit tab: Tuần/Tháng toggle renders HTML (may be short if no audit rows yet — the onboarding
      note explains this).
    - Settings tab: user_id field present; save updates settings.json.
27. Run `pytest tests/ -x -q` — confirm 249 green (246 existing + 3 persist-gate tests).
28. Confirm zero new rows in `portfolio` table for `user_id = "cron"` by querying DuckDB:
    `SELECT count(*) FROM portfolio WHERE user_id = 'cron'` must equal pre-test count
    (persist=False prevents phantom cron writes from dashboard preview).

---

## Data Flow

### MUA tab

```
user selects horizon (T+5 or T+20)
    → run_in_thread(daily_inference_headless, horizon, ttl=300)
        → main.daily_inference(broadcast=False, persist=False, horizon=H)
            → run_trade_execution(..., broadcast=False, persist=False)
                → _dispatch_signals(broadcast=False)   # no Telegram send
                # portfolio / RL / paperlog writes SKIPPED (persist=False)
            → returns (html, dispatched_signals: list[dict])
    → thread_runner caches result for 300s in session_state
→ dispatched_signals rendered via render_ticker_card (structured)
→ quick-add click → st.session_state["giu_prefill"] → st.rerun() → GIỮ pre-fill
```

### GIỮ tab

```
render():
    portfolio_list(LOCAL_USER_ID) [st.cache_data ttl=30]
        → DuckDBEngine().conn SELECT portfolio WHERE user_id='local'
    for each holding:
        latest_close(ticker) → ohlcv_<TICKER>.parquet
        _pnl_ratio(entry_price, current_close)   # VN thousands-VND scale applied
    signal_ledger.list_open() → sessions_remaining per ticker
→ DataFrame + PnL + countdown rendered

add form submit:
    portfolio_add(LOCAL_USER_ID, ticker, volume, price) [_audit_lock]
        → INSERT INTO portfolio
    st.cache_data.clear() + st.rerun()

remove button:
    portfolio_remove(LOCAL_USER_ID, ticker) [_audit_lock]
        → DELETE FROM portfolio WHERE user_id='local' AND ticker=?
    st.cache_data.clear() + st.rerun()
```

### BÁN tab

```
portfolio_list(LOCAL_USER_ID) → tickers
run_in_thread(inference_for_holdings_headless, tickers, ttl=300)
    → main.inference_for_holdings(tickers, window_rows=120) → html
→ st.markdown(html, unsafe_allow_html=True)
```

### Verify tab

```
user enters ticker + clicks Kiểm tra
    → run_in_thread(verify_single_ticker_headless, ticker, ttl=120)
        → main.verify_single_ticker(ticker) → html
    → st.markdown(html, unsafe_allow_html=True)
    → stored in st.session_state[f"verify_result_{ticker}"]

user clicks Gửi Telegram:
    → TelegramBot().send_text_alert(html, label=ticker)   # send-only, no poll
    → DuckDBEngine().log_user_action("local", "verify", ticker)
```

### Audit tab

```
user selects Tuần/Tháng → days=7 or 30
_cached_postmortem("local", days)  [st.cache_data ttl=300]
    → audit_evaluator.run_post_mortem("local", days=days)  → html
→ st.markdown(html, unsafe_allow_html=True)
```

---

## Failure Modes

| Mode | Root cause | Recovery |
|---|---|---|
| MUA inference returns empty list | No signals today / model gate | `st.info("Không có tín hiệu mua...")` |
| `daily_inference_headless` raises | Feature parquet stale / missing | tab error boundary catches; `st.error(str(exc))` |
| `portfolio_add` raises `ValueError` (duplicate) | Ticker already in portfolio | `st.warning(str(e))` — no crash |
| `latest_close` returns `None` | Parquet shard missing for ticker | Show "N/A" in PnL column |
| `verify_single_ticker` raises | Missing features / model artifact | tab error boundary; user-visible error |
| `run_post_mortem` raises | `GEMINI_API_KEY` missing or quota | tab error boundary; show Settings reminder |
| `TelegramBot().send_text_alert` raises | Bad token or no connectivity | `st.error("Gửi thất bại: ...")` — no retry loop |
| Thread future.result() raises | Any exception inside the thread | Exception propagates through thread_runner; caught by tab error boundary |
| `DuckDBEngine` connection error | DB file locked by cron | log warning; degrade to empty list; show stale-data banner |
| `config/settings.json` missing `dashboard_user_id` | New install | `_read_user_id()` falls back to `"local"` silently |

---

## Test Plan

### Automated (pytest)

| Test file | Tests | Coverage target |
|---|---|---|
| `tests/test_dashboard_persist_gate.py` | 3 (A, B, C above) | `run_trade_execution persist=False` writes nothing; `daily_inference` passthrough |

Note: Streamlit tab render functions cannot be easily unit-tested headless (they call `st.*`
which requires a running Streamlit runtime). The following dashboard-internal pure functions
CAN and SHOULD be unit-tested:

| Function | Test approach |
|---|---|
| `_pnl_ratio(entry, current)` | Pure math; parametrize happy path + None current + thousands-VND scale boundary |
| `_read_user_id()` | Mock `json.loads` / file read; test fallback to `"local"` |
| `portfolio_list` | In-memory DuckDB stub (same pattern as existing `conftest.py`) |
| `portfolio_add` / `portfolio_remove` | In-memory DuckDB stub; verify INSERT/DELETE rows |

Add these as additional test classes inside `tests/test_dashboard_persist_gate.py` (or a
separate `tests/test_dashboard_headless.py` — either is acceptable; keep total under 30 tests).

### Manual smoke (P2 gate)

Checklist for the executor to verify before marking P2 complete:

- [ ] `streamlit run dashboard/app.py` starts without ImportError.
- [ ] MUA tab: spinner appears on horizon toggle; real signal cards render OR empty state message.
- [ ] GIỮ tab: add HPG position, confirm row in table with PnL; remove it, confirm gone.
- [ ] BÁN tab: with one holding added, HTML inference renders without crash.
- [ ] Verify tab: enter ticker, run inference, HTML renders; push button sends Telegram (if token set).
- [ ] Audit tab: Tuần/Tháng toggle renders HTML; onboarding note visible.
- [ ] Settings tab: user_id field visible; save updates settings.json.
- [ ] `pytest tests/ -q` passes 249+.
- [ ] DuckDB check: `SELECT count(*) FROM portfolio WHERE user_id='cron'` unchanged from baseline.

---

## Dependencies

- P1 gate must be green before P2 execution: `streamlit run dashboard/app.py` with no ImportError,
  all 6 tabs render with stub data, Settings writes .env/.json.
- `streamlit>=1.35` is already in `requirements_dashboard.txt` — `st.status` available.
- `python-dotenv` and `duckdb` are already in `requirements.txt`.
- `GEMINI_API_KEY` required for Audit and Verify (Gemini arbitrator). Audit degrades gracefully
  if key missing; Settings tab instructs user to add it.
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` required for Verify push. Verify tab shows warning
  if push attempted without token.
- No new pip packages required beyond what P1 already added (`streamlit>=1.35`).

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `daily_inference` imports `main` at module level → slow cold start | MEDIUM | Lazy import inside each headless wrapper function body; first call in MUA tab is background-threaded anyway |
| ThreadPoolExecutor workers holding open DuckDB connections across Streamlit reruns | LOW | `DuckDBEngine()` creates a per-call connection; no shared connection held in thread pool |
| `_audit_lock` does not protect concurrent DuckDB writers from outside process (cron) | LOW | DuckDB WAL mode handles this; lock only serialises within the dashboard process |
| Streamlit `st.cache_data` decorator on a function that returns mutable dicts | LOW | `st.cache_data` deep-copies by default; safe |
| `st.session_state` key collision between tabs | LOW | All session_state keys are prefixed with tab name (`mua_`, `giu_`, `verify_`) |
| `run_in_thread` submits a new future on every rerun before ttl expires | LOW | Cache key check at top of `run_in_thread` prevents redundant submits within ttl |

---

## Backwards Compatibility

- `main.run_trade_execution` and `main.daily_inference` gain one kwarg each with `True` default.
  All existing callers (cron `__main__`, `run_full_pipeline_with_backfill`, bot commands,
  all existing tests) call without the new arg → behavior identical to pre-P2.
- `dashboard/utils/headless.py` module signature is extended, not replaced — the P1 function names
  are preserved; bodies change from `raise NotImplementedError` to real implementations.

---

## Rollback

- If the persist gate causes regression: revert `main.py` lines 1318–1407 (remove `persist` kwarg
  and `if persist:` wrapping). All other P2 changes are dashboard-only — revert independently with
  `git checkout -- dashboard/ tests/test_dashboard_persist_gate.py`.
- The persist gate is the ONLY production-path change. The dashboard package has no callers outside
  `streamlit run`.

---

## Out of Scope

- P3: `run_dashboard.bat` real implementation (venv bootstrap + browser open).
- P4: Inno Setup installer.
- P5: edge-case hardening (missing key banners, stale-data guardrails beyond the sidebar status).
- Structured-data wrappers for `verify_single_ticker` or `inference_for_holdings` (HTML→markdown
  is the approved approach per D3; no structured refactor of these fns).
- Any new feature engineering, model changes, or ML logic.

---

## Verification Evidence (required before marking DONE)

1. `pytest tests/ -q` output showing 249+ green, 0 failures (executor pastes stdout in report).
2. `pytest tests/test_dashboard_persist_gate.py -v` output showing all 3 tests PASSED.
3. Terminal output of `streamlit run dashboard/app.py` showing no traceback on startup.
4. DuckDB pre/post query: `SELECT count(*) FROM portfolio WHERE user_id='cron'` — counts must match.
5. Screenshot or session log from executor confirming all 6 tabs render with live data.

---

## Resume and Execution Handoff

Pass this file path to **vc-execute-agent**:
`process/features/local-dashboard/active/p2-logic-linking_PLAN_19-06-26.md`

Execute the implementation checklist in order:

1. Step 1 (persist gate, main.py) → verify with `pytest tests/ -x -q` → must stay at 246 green.
2. Step 2 (persist gate test) → run new test file → 249 green total.
3. Step 3 (headless.py) → import-check `python -c "from dashboard.utils.headless import daily_inference_headless"`.
4. Step 4 (thread_runner.py) → same import check.
5. Steps 5–9 (tab wiring) → after each tab, run `streamlit run dashboard/app.py` briefly to confirm no new ImportError.
6. Step 10 (sidebar) → visual confirm in browser.
7. Step 11 (Settings user_id) → save, inspect settings.json.
8. Step 12 (smoke) → full manual checklist + final pytest run.

After P2 gate passes, record findings in `process/features/local-dashboard/reports/p2-logic-linking_REPORT_19-06-26.md` and advance to P3 plan creation.
