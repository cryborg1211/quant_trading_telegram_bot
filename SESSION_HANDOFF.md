# SESSION HANDOFF — 2026-06-29 (Quant Engine V4)

## NEXT TASK = AUDIT THIS SESSION'S WORK
All work below is **UNCOMMITTED** in the working tree (HEAD = `96fdeb2`, branch
`main`). Next session: review the diff for correctness before committing. Code
graph rebuilt 2026-06-29 (2351 nodes / 23154 edges / 261 files) — use
`detect_changes` + `get_review_context` against the working tree.

## STATE NOW
- Branch **`main`** @ `96fdeb2`. Nothing from this session committed yet.
- Test suite: **473 passed** (was 417 at session start; +56 net new).
- Python: `C:\Users\caokh\AppData\Local\Programs\Python\Python311\python.exe`
  (polars/ML/pytest/streamlit). conda has NO polars. PowerShell only.
- `streamlit` app is RUNNING (holds a lock on `data/quant_v6_core.duckdb`) →
  any direct DuckDB read from a 2nd process fails until it's closed.

## CHANGED FILES (the audit surface)

### Dashboard fixes (3 bugs, user-reported live)
- `dashboard/app.py` — **module-not-found fix.** `streamlit run` puts the script
  dir (`dashboard/`) on `sys.path`, not repo root → `import dashboard.*` failed.
  Added `sys.path.insert(0, _REPO_ROOT)` guard before the dashboard imports.
- `dashboard/tabs/mua.py` — **"could not convert '22,600 VND'" crash.** Serve
  path (`main.py:1350`) stores `signal["price"]` as Telegram-display text; MUA's
  raw `float()` choked. Routed both price reads through existing
  `headless._parse_price`. Serve format untouched (bot still gets its string).

### Fan-chart redesign — `dashboard/utils/fan_chart.py` (TradingView-style)
- Replaced history line with `go.Candlestick` (#22c55e / #ef4444); removed shaded
  fan bands; added **12 Monte Carlo GBM paths** + neon median (#00F2FE);
  rangeslider + 1M/3M/6M/ALL rangeselector; crosshair spikes
  (`spikemode="toaxis+across"` — `"both"` is NOT valid plotly); transparent bg;
  enterprise strings + `st.columns(3)` metric strip (Volatility / Expected T+5
  Median / current price) in `dashboard/tabs/tam_nhin.py`.
- **Per-ticker MC seed** (`_ticker_seed` = `crc32(ticker)`, not `hash()` which is
  process-salted). FIXED a real bug: a single shared seed `_MC_SEED=7` made every
  ticker draw the IDENTICAL standard-normal sequence → all fans were one cloned
  shape scaled by price. Now DCM ≠ VHM, still stable across reruns.
- `src/data/price_lookup.py` — new `ohlc_history(ticker, n)` (candles need OHLC;
  `close_history` only returned closes). Mirrors the single-shard read contract.

### Model accuracy auditor (NEW — paperlog-integrated, READ-ONLY)
- `src/utils/accuracy_audit.py` — confusion-matrix analytics over the EXISTING
  `sentiment_entry_paperlog` (no new table, no serve-path write). `classify_outcome`
  (BUY+R>0=TP, BUY+R≤0=FP, SELL/HOLD+R≤0=TN, SELL/HOLD+R>0=FN), `summarize_accuracy`
  (BUY Precision, Defensive Recall), `build_accuracy_report` (15-row 🟢🔴🔵 table).
  Realized return = `ret_20d`; settled = `outcome_filled=TRUE`.
- `src/utils/telegram_bot.py` — new `/audit_accuracy` command (system-wide; paperlog
  is global, no user_id). **`/audit_weekly` left untouched** (keeps last session's
  post-mortem + engine-picks grading).
- ⚠️ Was a consolidated user spec to build a NEW `model_predictions_audit` table +
  patch capture in 3 serve routes + `settle_matured_predictions`. REJECTED as
  redundant (paperlog already captures+settles) and bug-prone (spec's calendar-add
  settlement reintroduces the fixed temporal flaw). User approved the lean path.

### Gemini 503 resilience + backfill
- `src/crawlers/sentiment_crawler.py` — extracted the Gemini call into
  `_generate_content`, `@retry`-wrapped (tenacity: `wait_exponential(1..60)`,
  `stop_after_attempt(5)`, retry ONLY on transient via `_is_transient_gemini_error`
  = 503/502/504/UNAVAILABLE/OVERLOADED). New reusable `score_payload`; `_score_item`
  now delegates to it. 4xx / JSON-parse errors are NOT retried.
- `scripts/backfill_sentiment_503.py` — re-scores `reason LIKE 'Gemini fallback%'`
  rows, UPDATEs by `url`. Dry-run default; `--commit` to write. ⚠️ NOTE: user added
  a `time.sleep(10)` at the end of `backfill_503()` (rate-limit pacing?) — this makes
  the 3 backfill unit tests sleep 10s each. Harmless but slow; flag if it bites CI.
- `requirements.txt` — pinned `tenacity==9.1.4` (was installed, unpinned).

### Tests (+56)
- `tests/test_dashboard_fan_chart.py` — rewritten for candlestick + MC paths +
  per-ticker-seed divergence + crc32 stability.
- `tests/test_accuracy_audit.py` (16) — matrix, precision/recall, query contract.
- `tests/test_sentiment_resilience.py` (15) — retry/give-up/no-retry-on-4xx,
  clamp, fallback, backfill recover/skip/dry-run.

## NOT DONE (deferred, explicit user calls)
- **Arbitration risk-gate** (cross-check classifier vs path-median vs sentiment).
  User reviewed; decided NOT a new file — harden existing
  `make_final_decision` (quant_agent_arbitrator.py:923): tighten sentiment veto
  −0.5→−0.2 + add `expected_return` param. Then said "leave it aside." UNSTARTED.
  Needs kill-switch + A/B before default-on (touches degree-84 serve hub).
- Wiring `accuracy_audit` into the dashboard Audit tab (currently bot-only).
- Committing ANY of this session's work.

## AUDIT TARGETS / RISK NOTES for next session
1. **Serve-path touch:** only `sentiment_crawler.py` (refactor) + `telegram_bot.py`
   (new command) are serve-side. fan_chart/mua/tam_nhin/accuracy_audit are
   read-only/dashboard. Confirm the crawler refactor preserves `_score_item`
   output exactly (parity test: full suite green).
2. **Empty-data caveats:** `dispatched_signals` and the new accuracy report are
   EMPTY until a live cron broadcast + T+20 maturity. Code unit-tested; needs real
   data to show non-empty.
3. **Transparent fan-chart bg** was the cause of an earlier "unreadable font"
   report; re-introduced per explicit redesign spec. All text colors now explicit
   light (#A3AED0) so it reads on the dark container. If it goes unreadable again,
   that's why.
4. **`backfill_sentiment_503.py` `time.sleep(10)`** — user-added; slows the 3
   backfill tests. Not a bug, but verify intent during audit.

## ENV GOTCHAS (CRITICAL)
- git-bash BROKEN → **PowerShell only** for git/python/pytest.
- PowerShell native-arg quirk: embedded `"` in a `@'...'@` here-string to
  `git commit -m` re-splits into pathspecs → commit fails. Keep messages quote-free.
- python = explicit `C:\Users\caokh\AppData\Local\Programs\Python\Python311\python.exe`.
- Prefix heavy runs: `$env:PYTHONIOENCODING="utf-8"`.
- code-review-graph post-commit hook → cp1252 `UnicodeEncodeError` = COSMETIC,
  commit succeeds.
- DuckDB locked while streamlit runs — close it before any 2nd-process DB read.

## KEY CMDS
- tests: `python -m pytest -q` (473 pass)
- new-work tests only: `python -m pytest tests/test_accuracy_audit.py tests/test_sentiment_resilience.py tests/test_dashboard_fan_chart.py -q`
- launch dashboard: `streamlit run dashboard/app.py`
- 503 backfill (preview): `python scripts/backfill_sentiment_503.py`  (add `--commit` to write)
- accuracy report smoke: `python -c "from src.utils.accuracy_audit import build_accuracy_report; print(build_accuracy_report())"`
- rebuild graph: code-review-graph `build_or_update_graph_tool(full_rebuild=True)`
