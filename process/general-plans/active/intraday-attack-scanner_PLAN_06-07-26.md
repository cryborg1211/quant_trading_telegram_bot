# Intraday Attack Scanner (Monitoring-Only) — Implementation Plan

**Date**: 06-07-26
**Complexity**: COMPLEX-lite (new pure module + config knobs + bot job wiring + tests; no schema/retrain change)
**Status**: 🔨 CODE DONE (stale header corrected 08-08-26 — `src/trading/intraday_scanner.py` exists, 542 lines, all listed public functions present; `tests/test_intraday_scanner.py` 40/40 green; the checklist below was simply never checked off as work landed). Stays ACTIVE — NOT ✅ VERIFIED — pending Gate 5 (live 09:15 ICT market-open smoke test), same convention as `portfolio-guard_PLAN_13-07-26.md` / `eod-position-report_PLAN_16-07-26.md`. Per `all-context.md`'s own note, Gate 5 was still unobserved as of the last content sync.

---

## Overview

Add a repeating background job to the Telegram bot process that rescores the model
on **provisional intraday bars** every N minutes during HOSE trading hours, and
sends a compact "top movers" card when something notable changes mid-session
(e.g. a name moving 42%→45% P(UP) before the 15:30 EOD crawl runs). This is
**monitoring-only**: it does not place trades, does not touch the paperlog, does
not invoke the sentiment arbitrator, and does not change any EOD serve behavior.

**Non-Goals (explicit)**:
- No new trading signal, no BUY dispatch, no `signal_ledger` writes.
- No sentiment/arbitrator calls (no Gemini in the loop).
- No writes to `sentiment_entry_paperlog` or any other DuckDB table.
- No writes to parquet or any on-disk store (splice is in-memory only, per scan).
- No change to `FEATURE_RECIPE_VERSION`, no retrain.
- No change to the EOD cron pipeline (`main.daily_inference` broadcast path) or `/suggest_buy*` semantics.

**Goal**: give the operator earlier visibility into intraday model movement so they
can *manually* act before the close, without altering the automated system's
trading behavior in any way.

Read before implementing: `process/context/all-context.md` (repo router — architecture,
conventions, stack) and `process/context/tests/all-tests.md` (pytest runner, test file
map, debugging quick-ref).

---

## Verified Resources (do not re-verify — confirmed this session, 06-07-26)

- **SSI iBoard bulk snapshot** (`src/data/foreign_flow_crawler.py::_fetch_ssi_hose_snapshot`,
  `_SSI_EXCHANGE_URL = "https://iboard-query.ssi.com.vn/stock/exchange/{exchange}"`,
  `_SSI_HEADERS`). One GET returns ~407 HOSE tickers. Per-ticker fields confirmed live:
  `stockSymbol`, `openPrice`, `highest`, `lowest`, `matchedPrice`, `nmTotalTradedQty`,
  `refPrice`, `ceiling`, `floor`, `tradingDate` (YYYYMMDD string). Prices are **absolute VND**;
  repo's on-disk parquet convention is **thousands of VND** — the existing crawler already
  divides by 1000.0 for its value fields (`foreign_flow_crawler.py:206-207`); the scanner
  MUST do the same for `openPrice`/`highest`/`lowest`/`matchedPrice`.
- **PTB JobQueue is NOT currently available in this environment.** Verified directly:
  ```
  ApplicationBuilder().token(...).build() → app.job_queue is None
  PTBUserWarning: No `JobQueue` set up. To use `JobQueue`, you must install PTB via
  pip install "python-telegram-bot[job-queue]".
  ```
  `python -c "import apscheduler"` → `ModuleNotFoundError`. `requirements.txt:27` pins
  `python-telegram-bot==22.7` with **no** `[job-queue]` extra, and no `APScheduler` is
  otherwise installed. **This is a new dependency this plan must add** — see Touchpoints
  section 1. This directly affects the "no new process, run inside the bot" architecture
  constraint: without this extra, `run_repeating` is unavailable and the described
  design cannot work as stated.
- **Serve inference path**: `main._compute_v3_features(latest_df, feature_list, frac_diff_d)`
  → `build_v3_feature_panel` → `V3BotInference.predict_proba_3class` / `.predict_proba`.
  `_load_v3_bot(horizon)` lazy-loads + caches per horizon in module-level `_V3_BOT_CACHE`
  (dict keyed by horizon int) — already a "load once, reuse" cache the scanner should
  piggyback on rather than re-implementing its own model cache.
  `_resolve_candidate_universe(ohlcv_pl)` → `src/trading/serve_universe.liquid_universe`
  (ADV top-50, 20-day window) is the SAME gate `daily_inference` uses.
- **T+20 τ = `SAFE_BUY_THRESHOLD = 0.45`** (`main.py:1280`, module-level constant).
  **T+5 τ is NOT a hardcoded constant** — it is `bot.up_threshold` read from the loaded
  `V3BotInference` bundle (`models/saved/v3_ensemble_5d.joblib`, currently ~0.40 per the
  brief but must be read live from the artifact, never hardcoded, so a future retrain
  can't silently desync the scanner's alert threshold from the model's actual gate).
- **Live OHLCV window loader**: `Alpha360Generator.load_live_ohlcv_window(window_rows=120)`
  → `src/features/alpha360_generator.py:54-73` → globs `data/ohlcv_*.parquet`, reads each
  shard's tail via `pl.scan_parquet(...).tail(window_rows).collect()`, concatenates
  `diagonal`, sorted `[ticker, date]`. Returns exactly `ticker, date, open, high, low,
  close, volume` — this is the frame the scanner must splice a provisional row onto,
  **in memory**, never touching the underlying parquet files.
- **Baseline test suite**: `pytest -q` → 481 passed (confirmed this session, matches
  `project_quant_engine` memory). This is the pre-change baseline the plan's exit
  criteria compares against.

---

## Frozen Design Decisions (from user debate — do not relitigate)

1. **Cadence**: repeating job, default 15 min (config knob `intraday_scan_interval_min`,
   valid range 10–30, validated at config load / job-scheduling time — see Failure Modes
   section). Runs ONLY during HOSE trading hours: **09:15–11:30** and **13:00–14:45 ICT**,
   Mon–Fri. Outside the window (or on an empty snapshot indicating a holiday), the scan is
   a silent no-op (log at DEBUG/INFO, no Telegram message, no error).
2. **Provisional bar splice** (pure, in-memory only):
   - Map one SSI snapshot row → one provisional OHLCV row:
     `open=openPrice/1000, high=highest/1000, low=lowest/1000, close=matchedPrice/1000,
     volume=nmTotalTradedQty`.
   - `date` = parsed from the snapshot's own `tradingDate` (YYYYMMDD), never wall-clock —
     mirrors `foreign_flow_crawler`'s convention of trusting the exchange's own session date.
   - Splice onto the in-memory 120-row tail per ticker: **replace** today's row if the
     EOD crawl for that date already landed in the tail (i.e. `tail["date"].max() ==
     provisional_date`), **append** otherwise (i.e. today's session hasn't settled into
     the parquet yet).
   - **IN-MEMORY ONLY.** The scanner never calls `write_parquet`, never opens a DuckDB
     write connection, never touches `data/ohlcv_*.parquet` on disk.
3. **Rescore**: both horizons (T+20 primary via `_load_v3_bot(20)`, T+5 via
   `_load_v3_bot(5)`), full cross-section (all tickers with a valid spliced tail), then
   apply the ADV-top-50 gate (`_resolve_candidate_universe` / `serve_universe.liquid_universe`)
   to restrict the alert-worthy set. Models are loaded once and cached (reuse
   `main._load_v3_bot`'s existing `_V3_BOT_CACHE` — do NOT build a second cache).
4. **Alert policy — event-only, no spam.** Telegram message ONLY when:
   - (a) a new ticker enters the in-universe **top-3 by P(UP)** for a given horizon vs the
     previous scan, OR
   - (b) any in-universe ticker's P(UP) crosses its horizon's τ (0.45 for T+20,
     `bot.up_threshold` for T+5) in either direction since the previous scan, OR
   - (c) `|ΔP(UP)| >= 0.02` (2 percentage points) vs the previous scan for a current
     top-3 name.
   First scan of the trading day (no prior in-memory state) = baseline: always sends
   one compact "session opened" card, never treated as a false "new entrant" flood.
   Last-scan state is an in-memory dict, **reset daily** (cleared at the start of the
   morning session window, or lazily when a new calendar date is detected).
5. **Card content**: compact HTML, respecting the existing 4096-char Telegram
   convention (target well under, e.g. < 1500 chars — this is a short card, not a
   full report). Per horizon, list top-3 in-universe by P(UP): ticker, P(UP) now,
   Δ vs previous scan, Δ vs yesterday's EOD score (if available), last price, %
   change vs `refPrice`. Must carry a visible **"⚠️ TẠM THỜI TRONG PHIÊN — KHÔNG PHẢI
   TÍN HIỆU GIAO DỊCH"** (provisional intraday — NOT a trade signal) tag. No BUY
   wording anywhere in the card. No arbitrator/sentiment invocation in this path at all.
6. **Paperlog discipline (CRITICAL, hard constraint)**: the scanner writes **nothing**
   to `sentiment_entry_paperlog`, `signal_ledger`, `portfolio`, `trade_history`, or any
   other DuckDB table. This is a read-model-only, alert-only feature. Any code path
   that would call `evaluate_trades_batch`, `PortfolioManager.process_daily_trades`,
   or persist a paperlog row is out of scope and MUST NOT be touched by this module.
7. **Config** (new knobs on `TradingConfig`, `config/settings.py` + `settings.json`):
   - `intraday_scanner_enabled: bool = False` (kill-switch precedent — OFF by default,
     matching the "default False" instruction in the brief; this is DIFFERENT from
     `regime_sizing_enabled`/`sentiment_entry_enabled` which default True — the brief
     explicitly asked for default False here because this is a brand-new always-on
     background job with a new runtime dependency).
   - `intraday_scan_interval_min: int = 15` (valid 10–30 inclusive).
   - `intraday_alert_delta_pp: float = 0.02` (the 2pp Δ threshold from decision 4c,
     exposed as a knob rather than a hardcoded literal, following the repo's
     dataclass-config convention).
8. **Degrade-not-crash**: SSI fetch failure / empty snapshot / any per-scan exception
   → log a warning, skip that scan cycle, **never** kill the bot process or the PTB
   event loop. Reuse the retry style already proven in
   `foreign_flow_crawler._fetch_ssi_hose_snapshot` (tenacity, retry only on
   `_TRANSIENT_EXC` + 5xx via `_is_retryable_5xx`), imported/reused rather than
   duplicated (see Touchpoints section 2).

---

## Touchpoints

### 1. New dependency (BLOCKING discovery — must land before job wiring works)

- **File**: `requirements.txt`
- **Change**: add `APScheduler` (the PTB `[job-queue]` extra dependency) as a pinned
  line, OR re-pin `python-telegram-bot==22.7` with the `[job-queue]` extra
  (`python-telegram-bot[job-queue]==22.7`). Prefer the extra-pin form so PTB's own
  compatibility matrix governs the APScheduler version, matching how the repo already
  pins `tenacity==9.1.4` (comment-annotated, single source of truth).
- **Verification**: after the dependency change, `ApplicationBuilder().token(...).build()`
  must NOT emit `PTBUserWarning: No JobQueue set up` and `app.job_queue` must not be
  `None`. This is a manual smoke check (see Verification Evidence section), not something
  pytest can assert without a live Application build (heavy — token/network). A
  lightweight unit assertion can check `Application.job_queue` is non-`None` after
  `build_application()` is called with a dummy token via monkeypatched env var, IF that
  does not attempt network I/O at construction time (verify during EXECUTE;
  `ApplicationBuilder().build()` does not itself make a network call — polling only
  starts at `run_polling()`).

### 2. New pure module — `src/trading/intraday_scanner.py`

Pure functions only (no I/O, no CONFIG import inside pure functions — mirrors
`serve_universe.py`'s purity contract). Proposed functions:

- `snapshot_row_to_provisional_bar(item: dict, price_unit_divisor: float = 1000.0) -> dict | None`
  Maps one SSI snapshot dict to `{ticker, date, open, high, low, close, volume}`
  (scaled). Returns `None` for malformed rows (missing `stockSymbol`, malformed
  `tradingDate`) — mirrors `foreign_flow_crawler.crawl_today`'s per-row skip-not-fail
  pattern.
- `splice_provisional_bar(tail_pl: pl.DataFrame, provisional: dict) -> pl.DataFrame`
  Pure Polars transform: replace-if-same-date-else-append for ONE ticker's tail. Never
  mutates the input frame (returns a new one). Must preserve the `ticker, date, open,
  high, low, close, volume` schema/dtypes `load_live_ohlcv_window` produces.
- `splice_all(tails_pl: pl.DataFrame, snapshot_items: list[dict]) -> pl.DataFrame`
  Cross-sectional wrapper: groups `tails_pl` by ticker, applies `splice_provisional_bar`
  per ticker that has a matching snapshot row, leaves tickers without a snapshot match
  untouched (pass-through, not dropped).
- `is_trading_window(now: datetime, tz: str = "Asia/Ho_Chi_Minh") -> bool`
  Pure time-window gate: Mon–Fri, 09:15–11:30 OR 13:00–14:45 ICT. Takes `now` as a
  parameter (never calls `datetime.now()` internally) so it is deterministically
  unit-testable across boundary times.
- `detect_events(current: dict[str, dict[int, float]], previous: dict[str, dict[int, float]] | None, thresholds: dict[int, float], delta_pp: float, top3_by_horizon: dict[int, list[str]], prev_top3_by_horizon: dict[int, list[str]] | None) -> list[dict]`
  Pure event-detection: returns a list of event dicts (`{"ticker", "horizon", "kind":
  "new_entrant"|"threshold_cross"|"delta_move"|"baseline", "p_up", "delta"}`).
  `previous is None` → returns a single `baseline` event marker (caller sends the
  opening card) rather than flooding "new_entrant" for every top-3 name.
- `build_scan_card(events: list[dict], scores_now: dict, scores_prev_eod: dict | None, prices: dict) -> str`
  Pure HTML formatter. Enforces the "provisional — not a trade signal" tag, no BUY
  wording, respects the char-budget convention (assert/trim if a defensive cap is
  exceeded — mirror `_safe_split_block` philosophy but this card should never need
  splitting given only top-3×2-horizon content).

I/O + orchestration (still in this module, but exercised only via monkeypatched
tests, never real HTTP in tests):

- `fetch_snapshot() -> list[dict]` — thin wrapper delegating to
  `src.data.foreign_flow_crawler._fetch_ssi_hose_snapshot("HOSE")` (import and reuse,
  do NOT copy/duplicate the tenacity retry decorator — one retry policy, one place).
  If import-reuse of a "private" `_`-prefixed function across modules is judged too
  fragile during EXECUTE, promote `_fetch_ssi_hose_snapshot` to a public
  `fetch_ssi_hose_snapshot` in `foreign_flow_crawler.py` (backwards-compatible rename
  with the old name kept as a thin deprecated alias) rather than forking the retry logic.
- `run_scan(state: ScannerState) -> ScannerState` (or equivalent explicit
  state-threading function) — the actual per-tick orchestration: fetch snapshot →
  splice → rescore both horizons → resolve ADV universe → detect events → send card
  if warranted → return updated state. This is the function the PTB job callback calls;
  it is NOT itself async (PTB job callbacks can call sync code via a thin async wrapper
  in the bot module — keep this module free of `asyncio`/`telegram` imports so it stays
  independently testable, matching `serve_universe.py`'s "no CONFIG import, no I/O"
  purity precedent as closely as the orchestration function's nature allows; the
  orchestration function IS I/O-bearing by necessity but should not import `telegram`).

### 3. Config — `config/settings.py` (`TradingConfig` dataclass)

Add three fields with inline comments following the existing kill-switch comment
style (see `regime_sizing_enabled`, `sentiment_entry_enabled` precedent at
`config/settings.py:70-93`):

```
intraday_scanner_enabled: bool = False
intraday_scan_interval_min: int = 15
intraday_alert_delta_pp: float = 0.02
```

### 4. Config — `config/settings.json`

Add matching keys under `"trading"`:

```json
"intraday_scanner_enabled": false,
"intraday_scan_interval_min": 15,
"intraday_alert_delta_pp": 0.02
```

### 5. Bot wiring — `src/utils/telegram_bot.py`

- In `build_application()` (`src/utils/telegram_bot.py:1601`), after
  `app = ApplicationBuilder()...build()` and before `return app`, add:
  - Read `CONFIG.trading.intraday_scanner_enabled`. If `False`, skip job registration
    entirely (log at INFO that the scanner is disabled) — this is the primary
    kill-switch and rollback lever.
  - If `True` AND `app.job_queue is not None`: register a repeating job via
    `app.job_queue.run_repeating(callback=_intraday_scan_job, interval=<seconds>,
    first=<small delay>)`, where `<seconds>` = `CONFIG.trading.intraday_scan_interval_min
    * 60`, clamped/validated to the 10–30 min range (log + clamp rather than crash on
    an out-of-range settings.json value).
  - If `True` AND `app.job_queue is None` (job-queue extra not installed despite the
    flag being on): log an ERROR explaining the missing `[job-queue]` extra and
    **do not crash the bot** — the scanner silently stays off, everything else in the
    bot continues to work. This is a required degrade path given the dependency gap
    found in the Verified Resources section.
- New async callback `_intraday_scan_job(context: ContextTypes.DEFAULT_TYPE) -> None`
  (thin adapter): calls `src.trading.intraday_scanner.run_scan(...)` off the event loop
  via `asyncio.to_thread` (matching the existing pattern used for `daily_inference` /
  `verify_single_ticker` — see `_suggest_buy_dispatch`, `verify_command`), then, if the
  returned state carries a non-empty card, sends it via `context.bot.send_message` to
  the same admin/oversight destination(s) already used elsewhere in this file
  (`ADMIN_CHAT_ID`; reuse, do not invent a third audience). Wrap the entire callback
  body in a broad `try/except Exception` that logs and swallows — a scan-cycle failure
  must never propagate into PTB's job-queue error surface in a way that stops future
  runs (JobQueue by default keeps re-scheduling `run_repeating` jobs even after an
  exception, but explicit swallow-and-log is still required so partial failures are
  diagnosable and consistent with this module's degrade-not-crash contract).
- Import placement: lazy-import `src.trading.intraday_scanner` inside the callback
  (matching this file's existing lazy-import convention for heavy modules, e.g.
  `from main import daily_inference` inside `_suggest_buy_dispatch`) rather than a
  top-level import, so bot startup stays fast when the scanner is disabled.

### 6. Tests — new file `tests/test_intraday_scanner.py`

Sibling in style to `tests/test_foreign_flow_crawler.py` (pure monkeypatch-based, no
live HTTP, no live PTB Application). Add to `process/context/tests/all-tests.md`'s
Test File Map row set as part of this plan's cleanup step (not a blocker to EXECUTE,
but must happen before UPDATE PROCESS closeout — see Blast Radius section).

---

## Public Contracts

What this plan exposes / depends on.

**New public surface** (must remain stable for the bot-wiring call site):

- `src.trading.intraday_scanner.is_trading_window(now, tz="Asia/Ho_Chi_Minh") -> bool`
- `src.trading.intraday_scanner.snapshot_row_to_provisional_bar(item, price_unit_divisor=1000.0) -> dict | None`
- `src.trading.intraday_scanner.splice_provisional_bar(tail_pl, provisional) -> pl.DataFrame`
- `src.trading.intraday_scanner.splice_all(tails_pl, snapshot_items) -> pl.DataFrame`
- `src.trading.intraday_scanner.detect_events(current, previous, thresholds, delta_pp, top3_by_horizon, prev_top3_by_horizon) -> list[dict]`
- `src.trading.intraday_scanner.build_scan_card(events, scores_now, scores_prev_eod, prices) -> str`
- `src.trading.intraday_scanner.fetch_snapshot() -> list[dict]`
- `src.trading.intraday_scanner.run_scan(state) -> state` (exact signature/state shape
  to be finalized during EXECUTE — must be explicit, typed, and documented in the
  module docstring; this plan intentionally does not over-specify the state container
  shape since that is an implementation-detail decision appropriate for EXECUTE, but
  EXECUTE must NOT introduce any hidden global mutable state beyond what the Frozen
  Design Decisions section explicitly allows (the in-memory last-scan dict) — no new
  DuckDB tables, no new parquet writes).

**Depended-on existing contracts** (must not be modified by this plan):

- `main._load_v3_bot(horizon) -> V3BotInference` (reused, not modified)
- `main._resolve_candidate_universe(ohlcv_pl) -> frozenset[str]` (reused, not modified)
- `main._compute_v3_features(latest_df, feature_list, frac_diff_d) -> pd.DataFrame` (reused, not modified)
- `src.features.alpha360_generator.Alpha360Generator.load_live_ohlcv_window(window_rows=120) -> pl.DataFrame` (reused, not modified)
- `src.data.foreign_flow_crawler._fetch_ssi_hose_snapshot(exchange="HOSE") -> list[dict]`
  (reused; MAY be renamed to a public name with a deprecated alias per Touchpoints
  section 2 — this is the only existing-file signature change this plan permits, and
  it must remain backwards-compatible)

---

## Data Flow

```
PTB JobQueue tick (every intraday_scan_interval_min, gated to trading-hour window)
  -> _intraday_scan_job (telegram_bot.py, async, asyncio.to_thread)
       -> intraday_scanner.run_scan(state)
            1. is_trading_window(now)? no -> return state unchanged (no-op)
            2. fetch_snapshot() -> list[dict]  (SSI iBoard bulk GET, tenacity-retried)
               empty/failed -> log + return state unchanged (degrade, no alert)
            3. Alpha360Generator().load_live_ohlcv_window(window_rows=120) -> pl.DataFrame
               (existing 120-row tails per ticker; READ-ONLY, no mutation of source parquet)
            4. splice_all(tails, snapshot_items) -> spliced pl.DataFrame (in-memory only)
            5. latest_df = spliced.to_pandas()
            6. for horizon in (5, 20):
                 bot = main._load_v3_bot(horizon)   [existing cache — no reload]
                 feats = main._compute_v3_features(latest_df, bot.tabular_features, frac_diff_d)
                 p_up = bot.predict_proba(feats)    -> dict[ticker, float]
            7. universe = main._resolve_candidate_universe(spliced) -> frozenset[str]
            8. restrict both horizons' p_up to `universe`
            9. compute top-3 per horizon by p_up
            10. detect_events(current, state.previous, thresholds, delta_pp, top3, state.prev_top3)
            11. events non-empty? -> build_scan_card(...) -> send via context.bot.send_message(ADMIN_CHAT_ID, ...)
                events empty?     -> no Telegram send this cycle
            12. state.previous = current scores; state.prev_top3 = top3; (persisted IN-MEMORY only,
                cleared at first scan of a new calendar date)
       <- updated state returned, held in a module-level or Application.bot_data slot
            (decide during EXECUTE: Application.bot_data is the PTB-idiomatic place for
            job-persistent state, avoiding a bespoke module-level global if PTB's own
            context object can carry it cleanly — but either choice must guarantee
            single-process, single-instance state, since this bot runs as one process)
```

**No writes anywhere in this flow** to: `data/ohlcv_*.parquet`, `data/quant_v6_core.duckdb`
(`sentiment_entry_paperlog`, `portfolio`, `trade_history`, `signal_ledger`, audit tables),
`data/foreign_flow_daily.parquet`. The ONLY network call is the SSI snapshot GET
(reused, already-audited call site). The ONLY Telegram send is the compact scan card
to the existing admin/oversight destination.

---

## Failure Modes and Handling

| Failure | Handling |
|---|---|
| SSI snapshot fetch exhausts retries (5xx) or times out | `fetch_snapshot()` catches, logs WARNING, `run_scan` returns state unchanged, no alert this cycle |
| SSI snapshot returns empty `data` (holiday, off-hours quirk) | Treated identically to "not an error" per `foreign_flow_crawler` precedent — log INFO, no-op |
| Snapshot row malformed (`tradingDate` not 8 chars, no `stockSymbol`) | `snapshot_row_to_provisional_bar` returns `None` for that row only; other tickers unaffected |
| `load_live_ohlcv_window` raises (`FileNotFoundError`/`ValueError`, e.g. no parquet shards at all) | `run_scan` catches broadly, logs ERROR, skips the cycle — this should be rare/only-at-fresh-install |
| Model bundle missing (`_load_v3_bot` raises `FileNotFoundError`) | `run_scan` catches, logs ERROR (actionable: "run train_models.py..."), skips the cycle — mirrors `_suggest_buy_dispatch`'s existing `FileNotFoundError` handling philosophy but degrades silently here (no user-facing command to reply to) |
| Feature-recipe mismatch (`_load_v3_bot` raises `RuntimeError`) | Same as above — caught, logged, cycle skipped; this is a hard "something is badly wrong" signal so the log line must be actionable (retrain command) |
| `app.job_queue is None` (job-queue extra missing) | Job never registered; ERROR logged once at bot startup; bot continues normally with the scanner permanently off until the dependency is installed |
| `intraday_scan_interval_min` outside 10–30 in settings.json | Clamp to nearest bound (10 or 30) + WARNING log at job-registration time, never crash bot startup |
| Telegram send fails (`BadRequest`, network) | Caught in the job callback, logged WARNING, next cycle proceeds normally (no retry-storm) |
| Bot restarts mid-session | In-memory `previous` state is lost — next scan after restart is treated as a fresh baseline (one "session opened" card), which is an acceptable, already-specified behavior (decision 4, "first scan of the day") |

---

## Implementation Checklist

1. **Dependency**: Update `requirements.txt` to add the PTB job-queue extra
   (`python-telegram-bot[job-queue]==22.7`) or a pinned `APScheduler` line with a
   comment explaining why (mirrors `tenacity==9.1.4` comment-pin precedent). Run
   `pip install -r requirements.txt` (or targeted `pip install "python-telegram-bot[job-queue]==22.7"`)
   in the project's environment.
2. **Verify JobQueue availability**: confirm `ApplicationBuilder().token(...).build().job_queue`
   is non-`None` and the `PTBUserWarning` no longer fires (manual smoke check).
3. **Config dataclass**: add `intraday_scanner_enabled`, `intraday_scan_interval_min`,
   `intraday_alert_delta_pp` fields to `TradingConfig` in `config/settings.py`, with
   inline comments following the existing kill-switch comment convention.
4. **Config JSON**: add matching keys under `"trading"` in `config/settings.json`
   (`intraday_scanner_enabled: false`, `intraday_scan_interval_min: 15`,
   `intraday_alert_delta_pp: 0.02`).
5. **New module**: create `src/trading/intraday_scanner.py` with the pure functions
   listed in Touchpoints section 2 (`is_trading_window`, `snapshot_row_to_provisional_bar`,
   `splice_provisional_bar`, `splice_all`, `detect_events`, `build_scan_card`) plus the
   I/O functions (`fetch_snapshot`, `run_scan`). Strict Python 3.10+ type hints on every
   signature. Module docstring must state the monitoring-only / no-persistence contract
   explicitly (mirroring `foreign_flow_crawler.py`'s "SOURCE HONESTY" docstring style).
6. **Reuse retry logic**: decide and implement the `fetch_snapshot` reuse strategy from
   Touchpoints section 2 — either import `foreign_flow_crawler._fetch_ssi_hose_snapshot`
   directly, or promote it to a public `fetch_ssi_hose_snapshot` with a
   backwards-compatible deprecated alias for the old private name. Do not duplicate the
   tenacity decorator.
7. **Bot wiring**: in `src/utils/telegram_bot.py::build_application()`, add the
   config-gated `job_queue.run_repeating(...)` registration (with the `job_queue is None`
   degrade-log) and the new `_intraday_scan_job` async callback per Touchpoints section 5.
   Import `intraday_scanner` lazily inside the callback.
8. **State container**: implement the "previous scan" state per Touchpoints section 5 /
   Data Flow section (decide `Application.bot_data` vs. module-level global during
   EXECUTE; document the choice in the module/function docstring), including the "reset
   when calendar date changes" rule from decision 4.
9. **Tests — pure functions**: create `tests/test_intraday_scanner.py` covering (at
   minimum):
   - `is_trading_window`: true inside both windows, false before 09:15, false in the
     11:30–13:00 lunch gap, false after 14:45, false on a Saturday/Sunday `now`.
   - `snapshot_row_to_provisional_bar`: correct ÷1000 scaling on open/high/low/close,
     volume passed through unscaled, malformed `tradingDate` → `None`, missing
     `stockSymbol` → `None`.
   - `splice_provisional_bar`: replace-when-same-date, append-when-new-date, dtype/
     schema preserved, input frame not mutated (assert original frame unchanged after
     the call).
   - `splice_all`: tickers without a snapshot match pass through untouched; multiple
     tickers spliced correctly in one call.
   - `detect_events`: first-scan (`previous=None`) → single baseline event only, no
     false "new_entrant" flood; a name crossing τ upward triggers `threshold_cross`;
     `|Δ| >= delta_pp` on a top-3 name triggers `delta_move`; a name newly entering
     top-3 triggers `new_entrant`; no event when nothing crosses any threshold.
   - `build_scan_card`: output contains the "provisional — not a trade signal" tag,
     contains no BUY-wording, stays under a sane char budget for a 3-name×2-horizon card,
     valid/closed HTML tags.
10. **Tests — degrade paths**: cover `fetch_snapshot`/`run_scan` failure branches via
    monkeypatch (empty snapshot, exception during fetch, malformed rows mixed with
    valid rows) asserting the function degrades to a no-op/unchanged-state result and
    never raises past `run_scan`. No live HTTP calls in any test.
11. **Full regression**: run `pytest -q` and confirm the pre-existing 481 tests plus
    the new `tests/test_intraday_scanner.py` tests all pass (see Verification Evidence
    section exit criteria).
12. **Context update**: add a one-line entry to `process/context/all-context.md`'s
    "Current Features" or inline convention section describing the intraday scanner
    (kill-switch default OFF, monitoring-only, no paperlog/no persistence), and add
    `tests/test_intraday_scanner.py` to `process/context/tests/all-tests.md`'s Test
    File Map. This step happens as part of UPDATE PROCESS closeout, not blocking EXECUTE
    completion, but must not be skipped.

---

## Acceptance Criteria

- [ ] `requirements.txt` pins the PTB `[job-queue]` extra (or `APScheduler` directly),
      and `ApplicationBuilder().token(...).build().job_queue` is confirmed non-`None`
      with no `PTBUserWarning` after install.
- [ ] `TradingConfig` in `config/settings.py` exposes `intraday_scanner_enabled` (default
      `False`), `intraday_scan_interval_min` (default `15`), `intraday_alert_delta_pp`
      (default `0.02`); `config/settings.json` carries matching keys under `"trading"`.
- [ ] `src/trading/intraday_scanner.py` exists with all functions listed in Public
      Contracts, fully type-hinted, with a module docstring stating the monitoring-only /
      no-persistence contract.
- [ ] `is_trading_window` correctly gates both HOSE sessions (09:15–11:30, 13:00–14:45
      ICT) and rejects weekends and outside-window times, verified by unit tests.
- [ ] `snapshot_row_to_provisional_bar` correctly divides SSI absolute-VND price fields
      by 1000 to match the parquet convention, passes volume through unscaled, and
      returns `None` (never raises) on malformed rows.
- [ ] `splice_provisional_bar` / `splice_all` never mutate their input frames and never
      write to disk (`data/ohlcv_*.parquet` untouched — verified by test assertion on
      file mtimes/content around the call).
- [ ] `detect_events` implements all four event kinds (`baseline`, `new_entrant`,
      `threshold_cross`, `delta_move`) exactly per the Frozen Design Decisions alert
      policy, with the first-scan case producing exactly one `baseline` event and no
      false `new_entrant` flood.
- [ ] `build_scan_card` output always contains the "provisional — not a trade signal"
      tag and never contains BUY-wording.
- [ ] `build_application()` in `src/utils/telegram_bot.py` registers the repeating job
      only when `CONFIG.trading.intraday_scanner_enabled` is `True` AND `app.job_queue`
      is not `None`; when the flag is `True` but `job_queue` is `None`, it logs an ERROR
      and continues without crashing.
- [ ] With `intraday_scanner_enabled=false` (the shipped default), the bot's behavior
      is unchanged from pre-plan baseline (no job registered, no new network calls).
- [ ] The scanner writes nothing to any DuckDB table or parquet file, verified by test
      assertions and by the market-open manual smoke test's `data/` mtime check.
- [ ] `pytest -q` reports the pre-existing 481 tests plus all new
      `tests/test_intraday_scanner.py` tests passing, zero regressions.
- [ ] `tests/test_intraday_scanner.py` contains zero live HTTP calls (all SSI fetch
      paths monkeypatched).

---

## Phase Completion Rules

This is a single-plan (non-phased) COMPLEX-lite effort with one execute-anchor file —
there are no sibling phase files to sequence. Nonetheless, the checklist has an internal
dependency order that functions as an implicit phase gate:

- **Gate 1 (Dependency, checklist items 1-2)**: must complete and be verified (
  `app.job_queue` non-`None`) before item 7 (bot wiring) is attempted — wiring a job
  against a `None` job_queue is a no-op that would silently mask a dependency-install
  failure during manual testing.
- **Gate 2 (Config, items 3-4)**: must land before item 7, since the bot-wiring code
  reads `CONFIG.trading.intraday_scanner_enabled` at `build_application()` time.
- **Gate 3 (Pure module, items 5-6)**: must be complete and independently unit-tested
  (item 9) before item 7 wires it into the bot — the bot-wiring code is a thin adapter
  and should not be the first place scanner logic is exercised.
- **Gate 4 (Bot wiring + state, items 7-8)**: CODE DONE once the job registers correctly
  under both the enabled and disabled config states and the degrade path (`job_queue is
  None`) is exercised manually or via a monkeypatched unit check.
- **Gate 5 (Full regression, items 9-11)**: this plan is CODE DONE only when `pytest -q`
  is green (481 pre-existing + new scanner tests). It is **VERIFIED** only after the
  manual market-open smoke test (Verification Evidence section) has been run at least
  once during live HOSE trading hours by the user — CODE DONE and VERIFIED must not be
  conflated in the EXECUTE completion report, per this repo's plan-artifact convention.
- **Gate 6 (Docs, item 12)**: closeout-only, appropriate for UPDATE PROCESS, does not
  block EXECUTE's own "implementation complete" report but must not be silently dropped.

**What's Functional Now** (to be updated by EXECUTE as gates close): PLANNED — no gates
closed yet.

---

## Verification Evidence

**Automated (must pass before considering this CODE DONE)**:
- `pytest -q` — full 481 pre-existing tests unaffected + all new
  `tests/test_intraday_scanner.py` tests green. Zero live network calls in the new
  test file (verified by code review during EXECUTE — no `requests.get`/real HTTP
  client instantiation outside of `monkeypatch`-substituted call sites).
- Targeted: `pytest -q tests/test_intraday_scanner.py -v` for a readable pass/fail list
  during development.

**Manual (cannot be automated without a live market/live bot — must be documented as
still-required post-EXECUTE verification, not skipped)**:
- **Market-closed smoke test**: with `intraday_scanner_enabled=true` and the bot running
  outside trading hours, confirm `is_trading_window` correctly no-ops every tick (check
  logs for the no-op message, confirm zero Telegram sends).
- **Market-open smoke test** (requires a live trading-hour window, Mon–Fri
  09:15–11:30 or 13:00–14:45 ICT): run the bot with the scanner enabled, confirm (a)
  one baseline "session opened" card is sent on the first in-window scan, (b) the SSI
  snapshot fetch succeeds and provisional bars are spliced without error, (c) no
  parquet/DuckDB files are modified during the session (`git status` / file mtimes
  on `data/` should show no scanner-caused changes), (d) at least one subsequent scan
  produces either "no new events" (silent) or a correctly-tagged event card.
- **Dependency verification**: confirm `pip show python-telegram-bot` reflects the
  `[job-queue]` extra (or `pip show apscheduler` succeeds) in the actual deployment
  environment (VPS via systemd), not just the dev machine — `deploy/quant-v6-bot.service`
  may need a `pip install -r requirements.txt` re-run on deploy; this plan does not
  change the systemd unit file itself but flags the dependency-install step as a
  deploy-time prerequisite.

**Explicitly NOT required for this plan** (per frozen decisions): no backtest, no
retrain, no `FEATURE_RECIPE_VERSION` bump, no paperlog analysis.

---

## Blast Radius

**Direct new files**: `src/trading/intraday_scanner.py`,
`tests/test_intraday_scanner.py`.

**Modified files**:
- `requirements.txt` (new dependency pin)
- `config/settings.py` (`TradingConfig` — additive fields, default-safe, no existing
  field changed)
- `config/settings.json` (additive keys under `"trading"`, no existing key changed)
- `src/utils/telegram_bot.py` (`build_application()` gains a config-gated job
  registration block + one new async callback function; no existing command handler
  logic is touched)
- Possibly `src/data/foreign_flow_crawler.py` (only if `_fetch_ssi_hose_snapshot` is
  promoted to a public name — additive alias, zero behavior change to existing callers)
- `process/context/all-context.md` + `process/context/tests/all-tests.md` (documentation
  only, at UPDATE PROCESS time)

**Not touched, by design**: `main.py` (daily_inference, dispatch, arbitrator, paperlog),
`src/bot/sizing.py`, `src/trading/portfolio_manager.py`, `src/trading/signal_ledger.py`
(if it exists as a module — referenced by `telegram_bot.py`'s `/exits`), any DuckDB
schema/table, any parquet writer, `FEATURE_RECIPE_VERSION` / `src/backtest/pipeline.py`,
any trained model artifact.

**Runtime blast radius when `intraday_scanner_enabled=false` (default)**: zero — the
job is never registered, `build_application()` behaves identically to today except for
one additional `if` check and one unreachable-when-disabled function definition. This
is the primary safety property: shipping this plan with the default config is a no-op
for the running bot.

**Runtime blast radius when enabled**: one new periodic network call (SSI snapshot,
~407-ticker payload) every 10–30 min during ~4h15m of trading hours per day (roughly
17–25 calls/day at the 15-min default), one new in-process dual-horizon rescore per
tick (measured ~4s for the full 357-ticker rescore per the existing serve-path
benchmark in `all-context.md` — acceptable for a background job on a 10–30 min cadence,
does not block command handlers since it runs via `asyncio.to_thread`), and
intermittent Telegram sends gated by the event policy in the Frozen Design Decisions
section.

---

## Rollback

- **Primary rollback lever**: flip `intraday_scanner_enabled` to `false` in
  `config/settings.json` and restart the bot process (`run_bot.py` / systemd unit
  restart). This immediately and completely disables the feature with zero code
  changes, matching the repo's established kill-switch pattern.
- **Full revert**: `git revert` the commit(s) introducing this plan's changes. Since
  no schema, no parquet, no model artifact, and no paperlog row is ever written by
  this feature, a full revert has zero data-migration concerns — it is a pure
  code/config rollback.
- **Partial rollback**: if only the new dependency proves problematic in production
  (e.g. APScheduler version conflict on the VPS), the kill-switch in
  `config/settings.json` fully neutralizes the feature without requiring the
  dependency to be uninstalled (the job simply never gets registered when the flag
  is `false`, regardless of whether the extra is installed).

---

## Resume and Execution Handoff

- **Plan file**: `process/general-plans/active/intraday-attack-scanner_PLAN_06-07-26.md`
  (this file) — the single execute-anchor for this feature. No sibling phase files.
- **Execution order**: follow the Implementation Checklist items 1→11 in sequence;
  item 12 (context docs) is a closeout step, appropriate for UPDATE PROCESS rather
  than blocking EXECUTE's own completion report.
- **If EXECUTE is interrupted and resumed later**: re-read this plan file in full
  before continuing (no separate state file). Check `git status` / `git diff` to see
  which checklist items already have code; check `pytest -q tests/test_intraday_scanner.py`
  existence/pass state to see which test items are done. The dependency change
  (item 1–2) is the highest-priority item to verify first on resume, since every
  downstream item assumes `app.job_queue` is available.
- **What must NOT change without returning to PLAN**: the paperlog-write prohibition,
  the in-memory-only splice contract, the default-OFF kill-switch, and the
  "no arbitrator/no Gemini calls" constraint (all in the Frozen Design Decisions
  section) are hard constraints from the user's frozen decisions — any EXECUTE-time
  discovery that seems to require relaxing one of these must pause and return to
  PLAN/INNOVATE rather than being decided unilaterally during EXECUTE.
- **Validator command** (run before presenting EXECUTE completion):
  ```
  node .claude/skills/vc-generate-plan/scripts/validate-plan-artifact.mjs "process/general-plans/active/intraday-attack-scanner_PLAN_06-07-26.md"
  ```

---

## Open Questions for the User

Surface before/during EXECUTE — not blocking PLAN completion.

1. **Job-queue dependency approval**: this plan requires adding `APScheduler` (via the
   PTB `[job-queue]` extra) to `requirements.txt` — a new runtime dependency not
   previously present. Confirm this is acceptable before EXECUTE installs it.
2. **Alert destination**: Touchpoints section 5 assumes scan cards go to `ADMIN_CHAT_ID`
   (the existing oversight destination). Confirm this is correct, or specify a different
   destination if the operator wants intraday alerts routed elsewhere (e.g. only to
   `USER_CHAT_ID`, or a dedicated third channel).
3. **T+5 vs T+20 priority in the card**: the brief says "surface top-3 tickers by
   P(UP)" generally and later says "per horizon" for card content — confirming both
   horizons get their own top-3 block in one card (not just T+20 primary) matches
   decision 3/5 in Frozen Design Decisions; flag if the intent was T+20-primary-only
   with T+5 as a secondary confirmation line.

---

## Next Step

Review this plan carefully. Say **"ENTER EXECUTE MODE"** when ready to implement.
This is a critical safety checkpoint — EXECUTE mode will follow this plan with 100%
fidelity, including the three Open Questions above being resolved (or explicitly
deferred to EXECUTE's judgment within the stated constraints) before Touchpoints
sections 1 and 5 are implemented.
