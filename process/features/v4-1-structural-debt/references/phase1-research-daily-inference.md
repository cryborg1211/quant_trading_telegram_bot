# Phase 1 Research: daily_inference & main.py Structural Map

**Date:** 2026-06-09  
**Status:** Complete  
**Source:** Research agent deep-scan of main.py (2090 lines, 32 functions)

---

## 1. main.py Function Inventory (32 functions)

| Name | Lines | Count | Category | Extract? |
|---|---|---|---|---|
| `_humanize_feature` | 153–195 | 43 | Report formatter | YES → src/reports/ |
| `_build_feature_explanation` | 198–221 | 24 | Report builder | YES → src/reports/ |
| `_get_live_exec_prices` | 227–251 | 25 | Data utility | No |
| `_format_sentiment_status` | 254–271 | 18 | Report formatter | YES → src/reports/ |
| `setup_logging` | 274–281 | 8 | Infrastructure | No |
| `timed_step` | 288–294 | 7 | Utility | No |
| `is_crawl_allowed` | 297–323 | 27 | Orchestration gate | No |
| `crawl_hose` | 326–365 | 40 | Data ingestion | No |
| `_load_mr` | 382–389 | 8 | Model loader (globals) | No |
| `mr_score_tickers` | 392–435 | 44 | MR inference | No |
| `_load_v3_bot` | 459–517 | 59 | Model loader (cache) | No |
| `_compute_v3_features` | 532–591 | 60 | Feature compute | No |
| `predict_v3_horizon` | 594–652 | 59 | Model dispatch | No |
| `_build_combined_report` | 668–680 | 13 | Report builder | YES → src/reports/ |
| `_log_rl_predictions` | 704–743 | 40 | DB write | No |
| `_backfill_rl_outcomes` | 746–803 | 58 | DB read+write | No |
| **`daily_inference`** | **815–1085** | **271** | **GOD-FUNCTION** | **DECOMPOSE** |
| `_smart_truncate` | 1088–1102 | 15 | Text utility | YES → src/reports/ |
| `_build_fallback_observability_report_vi` | 1105–1173 | 69 | Report builder | YES → src/reports/ |
| `build_event_overrides` | 1188–1244 | 57 | Pure logic | DONE (Phase 1) |
| `run_trade_execution` | 1247–1402 | 156 | Orchestration + dispatch | Decompose internally |
| `_build_sell_hold_report` | 1419–1514 | 96 | Report builder | YES → src/reports/ |
| `inference_for_holdings` | 1517–1601 | 85 | Orchestration | No |
| `_mr_state_line` | 1626–1639 | 14 | Report formatter | YES → src/reports/ |
| `_build_verify_report` | 1642–1701 | 60 | Report builder | YES → src/reports/ |
| `verify_single_ticker` | 1704–1817 | 114 | Orchestration | No |
| `_build_rebalance_report` | 1827–1848 | 22 | Report builder | YES → src/reports/ |
| `rebalance_portfolio` | 1851–1924 | 74 | Orchestration | No |
| `full_pipeline` | 1927–1957 | 31 | Orchestration | No |
| `parse_args` | 1960–1987 | 28 | CLI | No |
| `_send_crash_alert` | 1990–2036 | 47 | Serving/alerting | No |
| `main` | 2039–2090 | 52 | Entry point | No |

**Report builders to extract: 10 functions (~374 lines)**

---

## 2. daily_inference Section-by-Section Breakdown

| Section | Lines | Name | Purpose | Extraction Target? |
|---|---|---|---|---|
| A | 839–852 | Feature Loading | Alpha360 OHLCV window → latest_df | No (entry point) |
| B | 854–874 | Dual-Horizon Inference | predict_v3_horizon × 2 horizons | No (core inference) |
| C | 882–888 | Observability Logging | Log top/bottom P(UP) | No (7 lines) |
| D | 893–906 | **VN30 Gate** | Filter to _VN30_UNIVERSE | **YES → `_gate_vn30()`** |
| E | 908–934 | Meta-gate + Candidate | meta_gate_5d filter, top-6 sort | Part of gate logic |
| F | 936–1011 | Fallback Mode | No candidates → observability report → EARLY RETURN | Stays in daily_inference |
| G | 980–982 | Arbitrator Call | evaluate_trades_batch | Part of main flow |
| H | 1013–1045 | Sentiment Ranking | BUY filter, sort, top-3 → top_buy_signals | Part of main flow |
| I | 1047–1070 | **Event/Rescue Loop** | Rescue pool + build_event_overrides | **YES → `_rescue_loop()`** |
| J | 1072–1085 | **Dispatch** | run_trade_execution call + return | **YES → simplify** |

### Data Flow

```
latest_df → predict_v3_horizon() → stacking_predictions_{5d,20d}
    → _gate_vn30() → universe_tickers
    → meta_gate filter → candidate_tickers
    → [fallback early return if empty]
    → evaluate_trades_batch() → final_decisions, all_sentiments
    → sentiment ranking → top_buy_signals (top 3)
    → _rescue_loop() → extended top_buy_signals + event_overrides
    → run_trade_execution() → report_html
```

---

## 3. Key Risks & Gotchas

### Global mutable state
- `_LATEST_REGIME_BY_TICKER` (line 529): mutated by `_compute_v3_features`, read by `run_trade_execution` at line 1351. Must carry this coupling explicitly if extracting.
- `_V3_BOT_CACHE`, `_MR_MODEL`, `_MR_TAU`: lazy-load caches, not problematic for extraction.

### `_rescue_loop()` is impure
- The pure core (`build_event_overrides`) is extracted. The wrapper fetches sentiment for missing rescue candidates via `evaluate_trades_batch` (Gemini API I/O). Extraction as `_rescue_loop()` will still be impure.
- Must receive: `fallback_mode`, `stacking_predictions_5d`, `universe_tickers`, `top_buy_signals`, `all_sentiments`, `horizon_predictions`
- Must return: modified `top_buy_signals`, `event_overrides`

### `_dispatch()` is already extracted
- `run_trade_execution` (156 lines) IS the dispatch. `daily_inference` calls it at line 1072 and returns its result. Adding a `_dispatch()` wrapper adds no value. The real decomposition target inside `run_trade_execution` is the inner Telegram send loop (lines 1335–1395).

### `fallback_mode` early return
- Line 1011 returns early, bypassing `run_trade_execution` entirely. The fallback flag (line 943) is re-checked at line 1051. Any extraction must preserve this dual check.

### `top_buy_signals` mutation
- Line 1067: rebinds `top_buy_signals = list(top_buy_signals) + _rescued`. `_rescue_loop()` must return the extended list, not mutate a caller-owned list.

### Event-layer constants
- `SAFE_BUY_THRESHOLD`, `EVENT_MIN_P_UP`, `EVENT_BULL_SENTIMENT`, `EVENT_BEAR_SENTIMENT`, `_EVENT_CAP` (lines 1178–1182) — used by `build_event_overrides`. Tests import directly from `main`. These must co-move with the function.

### Report builder datetime impurity
- `_build_verify_report`, `_build_sell_hold_report`, `run_trade_execution` call `datetime.now()` inline. Tests would need to patch or accept dynamic dates.

### `_build_combined_report` → TelegramBot dependency
- Calls `TelegramBot._build_message` from `src/utils/telegram_alerter.py`. Moving to `src/reports/` creates a back-dependency on `src/utils/`.

---

## 4. Test Coverage

### Covered
- `build_event_overrides` — 7 tests in `test_event_overrides.py` (Phase 1 pattern)
- `is_crawl_allowed`, `_humanize_feature`, `_format_sentiment_status`, `_get_live_exec_prices`, `_build_feature_explanation`, `_build_combined_report`, `_build_rebalance_report` — ~40 tests in `test_main_logic.py`

### NOT covered (Phase 2/3 targets)
- `daily_inference` (no integration test)
- `run_trade_execution`
- `_build_fallback_observability_report_vi`
- `_build_sell_hold_report`
- `_build_verify_report`
- `_smart_truncate`
- `inference_for_holdings`, `verify_single_ticker`, `rebalance_portfolio`
- `mr_score_tickers`

---

## 5. Unresolved Design Questions

1. **`_gate_vn30()` scope**: Just the VN30 filter (14 lines) or VN30 + meta-gate + candidate selection (42 lines)?
2. **`_dispatch()` value**: `run_trade_execution` already exists. A thin wrapper adds nothing. Should we decompose `run_trade_execution` internally instead?
3. **Event constants location**: Where do `SAFE_BUY_THRESHOLD`, `EVENT_MIN_P_UP` etc. live after `build_event_overrides` moves to `src/`?
4. **Integration test strategy**: `daily_inference` has 4 I/O dependencies. Heavy mock fixture or stub pipeline?
