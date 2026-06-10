# Phase 1 — Decompose `daily_inference` + Extract Report Builders

**Plan type:** COMPLEX  
**Date:** 2026-06-09  
**Feature folder:** `process/features/v4-1-structural-debt/`  
**Status:** READY FOR EXECUTE  
**Design decisions:** ALL RESOLVED — do not re-litigate in EXECUTE

---

## Overview

`main.py` is a 2090-line god-module. Phase 1 has two orthogonal work-streams:

1. **Report-builder extraction** — move 10 pure HTML-string functions (plus their associated constants) from `main.py` into a new `src/reports/` package. These functions have no orchestration logic, so the move is mechanical and low-risk.

2. **`daily_inference` decomposition** — break the 271-line god-function at lines 815–1085 into three named sub-functions: `_select_candidates()`, `_rescue_loop()`, and `_dispatch_signals()` (internal decomposition of `run_trade_execution`). Write a happy-path integration test for `daily_inference`.

The two work-streams are independent. The report-builder extraction must be completed first because several report functions are imported from `main` by existing tests and must keep working from their new home.

### Goals

- Reduce `daily_inference` from 271 to ~145 lines by naming its three phases.
- Reduce `run_trade_execution` from 156 to ~100 lines by extracting the inner Telegram send loop.
- Create `src/reports/` as a proper package with 10 extracted report builders.
- Add an integration test for `daily_inference`'s happy path and fallback path.
- Maintain 100% green status on the existing 137 tests throughout every step.

### Out of scope for Phase 1

- Moving report builders that would require back-dependencies on new modules (that check is done per-function in Step 1 below).
- Any changes to model inference code (`predict_v3_horizon`, `TabularEnsemble`, `_compute_v3_features`).
- Changing `run_trade_execution`'s public signature or behavior.
- Phase 2 extractions (`inference_for_holdings`, `verify_single_ticker`, etc.).

---

## Touchpoints

### Files read (no change permitted)

| File | Purpose |
|---|---|
| `main.py` | Source of all functions being moved/extracted |
| `tests/test_main_logic.py` | Imports directly from `main`; must keep working |
| `tests/test_event_overrides.py` | Imports `build_event_overrides` and constants from `main`; must keep working |
| `src/utils/telegram_alerter.py` | `TelegramBot._build_message` — called by `_build_combined_report`; creates import back-dep if moved naively |
| `config/settings.py` | `CONFIG.trading.*` read by `_build_sell_hold_report` |

### Files modified

| File | Nature of change |
|---|---|
| `main.py` | Delete 10 report-builder function bodies + constants (kept as re-exports); add 3 private helper functions; shorten `daily_inference` and `run_trade_execution` |
| `src/reports/__init__.py` | NEW — package init, re-exports all public names |
| `src/reports/builders.py` | NEW — all 10 report-builder functions + their constants |

### Files created (tests)

| File | Purpose |
|---|---|
| `tests/test_select_candidates.py` | Unit tests for `_select_candidates()` |
| `tests/test_rescue_loop.py` | Unit tests for `_rescue_loop()` |
| `tests/test_daily_inference_integration.py` | Integration test (happy path + fallback path) |

---

## Public Contracts

### `_select_candidates()` (new private function in `main.py`)

```
Signature:
  _select_candidates(
      predictions: dict[str, list[float]],
      meta_gate: dict[str, bool],
      universe_tickers: set[str],
      max_candidates: int,
  ) -> tuple[list[str], bool, dict[str, str]]

Returns:
  (candidate_tickers, fallback_mode, fallback_reasons)

Purity: PURE (no I/O, no mutations)
```

This combines the current Section D (VN30 gate, lines 893–906), Section E (meta-gate + top-N sort, lines 908–934), and Section F (fallback mode population, lines 936–978) into one function. The `fallback_mode` bool and `fallback_reasons` dict are bundled with the candidates to avoid three separate return values from inline code.

### `_rescue_loop()` (new private function in `main.py`)

```
Signature:
  _rescue_loop(
      fallback_mode: bool,
      stacking_predictions_5d: dict[str, list[float]],
      universe_tickers: set[str],
      top_buy_signals: list[str],
      all_sentiments: dict[str, dict],
      horizon_predictions: dict[str, dict],
  ) -> tuple[list[str], dict[str, dict]]

Returns:
  (extended_top_buy_signals, event_overrides)

Purity: IMPURE — may call evaluate_trades_batch for missing rescue candidates
```

Extracts Section I (lines 1047–1070). Returns the potentially extended `top_buy_signals` list (never mutates input list) and `event_overrides` dict.

### `_dispatch_signals()` (new private function in `main.py`)

```
Signature:
  _dispatch_signals(
      top_buy_signals: list[str],
      all_sentiments: dict[str, dict],
      stacking_predictions: dict[str, dict],
      live_exec_prices: dict[str, float],
      event_overrides: dict[str, dict],
      top_pos_features: str,
      top_neg_features: str,
      horizon: int,
      broadcast: bool,
      bot: TelegramBot,
  ) -> list[dict]

Returns:
  dispatched_signals list (used to build the combined report)

Purity: IMPURE — calls bot.send_signal_alert (I/O)
```

Extracts the inner Telegram send loop from `run_trade_execution` (lines 1335–1395). Reads `_LATEST_REGIME_BY_TICKER` module-level global — this coupling must be documented in the function docstring.

### `src/reports/builders.py` exports

All 10 functions below become importable from `src.reports`:

```
_humanize_feature
_build_feature_explanation
_format_sentiment_status
_build_combined_report
_smart_truncate
_build_fallback_observability_report_vi
_build_sell_hold_report
_mr_state_line
_build_verify_report
_build_rebalance_report
```

Constants co-moved to `src/reports/builders.py`:

```
FEATURE_HUMAN_NAMES          (dict, lines 58–118)
_MACRO_INNER_NAMES           (dict, lines 122–143)
_NUMERIC_SUFFIX_RE           (compiled regex, line 148)
_MACRO_PREFIX_RE             (compiled regex, line 150)
_REPORT_SEPARATOR            (str, line 665)
_VERIFY_5D_PRED_LABELS       (dict, lines 1609–1613)
_VERIFY_20D_PRED_LABELS      (dict, lines 1614–1618)
_VERIFY_VERDICT_LABELS       (dict, lines 1619–1623)
_REBALANCE_PRED_LABELS       (dict, line 1824)
_SELL_DECISION               (int, line 1409)
_MR_SELL_VETO                (str, lines 1412–1416)
```

IMPORTANT: `_build_combined_report` at line 668 calls `TelegramBot._build_message` from `src/utils/telegram_alerter.py`. Moving this function to `src/reports/builders.py` creates a `src/reports/ → src/utils/` dependency. This is acceptable — `src/utils/` already exists and is a legitimate dependency target. No circular dependency is introduced.

---

## Blast Radius

`daily_inference` has blast-radius degree 84 (third largest hub). The three callers of `daily_inference` are:

- `full_pipeline` (line 1927) — calls it directly; unchanged
- `main` entry point (line 2039) — calls it via `full_pipeline`; unchanged
- `build_application` in `src/utils/telegram_bot.py` — calls it via the `/suggest_buy` handler; unchanged

`run_trade_execution` is only called from `daily_inference` (line 1072). Internal decomposition does not change its signature.

Report builders are called from within `main.py` and from `inference_for_holdings`, `verify_single_ticker`, `rebalance_portfolio`. These all call the functions by name from `main` namespace. After the move, `main.py` re-exports the names as `from src.reports.builders import ...` so all existing call sites continue to work without change.

`tests/test_main_logic.py` imports `_build_combined_report`, `_build_feature_explanation`, `_build_rebalance_report`, `_format_sentiment_status`, `_get_live_exec_prices`, `_humanize_feature`, `is_crawl_allowed` directly from `main`. The re-export pattern in `main.py` preserves all of these. `_get_live_exec_prices` and `is_crawl_allowed` are NOT moved (they are data utility / infrastructure, not report builders), so they remain naturally in `main.py`.

`tests/test_event_overrides.py` imports `build_event_overrides`, `SAFE_BUY_THRESHOLD`, `EVENT_MIN_P_UP`, `EVENT_BULL_SENTIMENT`, `EVENT_BEAR_SENTIMENT`, `_EVENT_CAP` from `main`. These constants and `build_event_overrides` are NOT moved in Phase 1 (they co-move only if `build_event_overrides` is later extracted to `src/`). They remain in `main.py` unchanged.

---

## Implementation Checklist

Each step is atomic, independently verifiable with `pytest -q`, and ordered bottom-up (least risky first).

---

### STEP 1 — Create `src/reports/__init__.py` (empty init)

**File:** `src/reports/__init__.py`  
**Change:** Create as empty file (just the package marker).  
**Verification:** `python -c "import src.reports"` succeeds without error.  
**What could go wrong:** Nothing — creating an empty `__init__.py` has zero blast radius.

---

### STEP 2 — Create `src/reports/builders.py` with all 10 report builder functions

**File:** `src/reports/builders.py`  
**Change:** Create new file containing, in this exact order:

1. Module docstring explaining the package.
2. Standard library imports: `html`, `re`, `datetime` (from datetime).
3. Third-party imports: none needed here (functions use only stdlib + config).
4. Local imports:
   - `from config.settings import CONFIG` — needed by `_build_sell_hold_report` for `CONFIG.trading.take_profit_pct` / `stop_loss_pct`.
   - `from src.utils.telegram_alerter import TelegramBot, format_source_links` — needed by `_build_combined_report` and `_build_fallback_observability_report_vi`.
5. Constants block (in source order):
   - `FEATURE_HUMAN_NAMES` (copy from main.py lines 58–118)
   - `_MACRO_INNER_NAMES` (copy from main.py lines 122–143)
   - `_NUMERIC_SUFFIX_RE` (copy from main.py line 148)
   - `_MACRO_PREFIX_RE` (copy from main.py line 150)
   - `_REPORT_SEPARATOR` (copy from main.py line 665)
   - `_SELL_DECISION` (copy from main.py line 1409)
   - `_MR_SELL_VETO` (copy from main.py lines 1412–1416)
   - `_VERIFY_5D_PRED_LABELS` (copy from main.py lines 1609–1613)
   - `_VERIFY_20D_PRED_LABELS` (copy from main.py lines 1614–1618)
   - `_VERIFY_VERDICT_LABELS` (copy from main.py lines 1619–1623)
   - `_REBALANCE_PRED_LABELS` (copy from main.py line 1824)
6. Functions (in source order, exact bodies copy-pasted, no logic changes):
   - `_humanize_feature` (lines 153–195)
   - `_build_feature_explanation` (lines 198–221)
   - `_format_sentiment_status` (lines 254–271)
   - `_build_combined_report` (lines 668–680)
   - `_smart_truncate` (lines 1088–1102)
   - `_build_fallback_observability_report_vi` (lines 1105–1173)
   - `_build_sell_hold_report` (lines 1419–1514)
   - `_mr_state_line` (lines 1626–1639)
   - `_build_verify_report` (lines 1642–1701)
   - `_build_rebalance_report` (lines 1827–1848)

**Type annotation imports:** `from typing import Any` — needed by `_build_feature_explanation` (`model: Any`), `_build_verify_report`.

**numpy import:** `import numpy as np` — needed by `_build_feature_explanation` (uses `np.asarray`, `np.argsort`).

**What could go wrong:**
- Missing import in `builders.py` causes `ImportError`. Mitigation: verify each function's dependencies before pasting. `_build_feature_explanation` needs `np`. `_build_sell_hold_report` needs `CONFIG`, `_format_sentiment_status`, `_smart_truncate`, `_MR_SELL_VETO`, `_SELL_DECISION`, `_REPORT_SEPARATOR`. `_build_combined_report` needs `TelegramBot`. `_build_fallback_observability_report_vi` needs `html`, `_smart_truncate`, `format_source_links`.
- `_build_verify_report` calls `_mr_state_line`, `_format_sentiment_status`, `_smart_truncate` — all must be in scope within the same file. Since they are all in `builders.py`, this is fine.
- `_build_sell_hold_report` calls `_format_sentiment_status` and `_smart_truncate` — both in same file, fine.

**Verification:** At this step, run `python -c "from src.reports.builders import _smart_truncate; print(_smart_truncate('hello world test', 8))"` — should print `hello…`.

---

### STEP 3 — Update `src/reports/__init__.py` to re-export all public names

**File:** `src/reports/__init__.py`  
**Change:** Add `from src.reports.builders import (...)` re-exporting all 10 functions and any constants callers outside the module may need. The list:

```python
from src.reports.builders import (
    FEATURE_HUMAN_NAMES,
    _MACRO_INNER_NAMES,
    _REPORT_SEPARATOR,
    _SELL_DECISION,
    _MR_SELL_VETO,
    _VERIFY_5D_PRED_LABELS,
    _VERIFY_20D_PRED_LABELS,
    _VERIFY_VERDICT_LABELS,
    _REBALANCE_PRED_LABELS,
    _humanize_feature,
    _build_feature_explanation,
    _format_sentiment_status,
    _build_combined_report,
    _smart_truncate,
    _build_fallback_observability_report_vi,
    _build_sell_hold_report,
    _mr_state_line,
    _build_verify_report,
    _build_rebalance_report,
)

__all__ = [
    "FEATURE_HUMAN_NAMES",
    "_MACRO_INNER_NAMES",
    "_REPORT_SEPARATOR",
    "_SELL_DECISION",
    "_MR_SELL_VETO",
    "_humanize_feature",
    "_build_feature_explanation",
    "_format_sentiment_status",
    "_build_combined_report",
    "_smart_truncate",
    "_build_fallback_observability_report_vi",
    "_build_sell_hold_report",
    "_mr_state_line",
    "_build_verify_report",
    "_build_rebalance_report",
]
```

**Verification:** `python -c "from src.reports import _smart_truncate; print('OK')"` should print `OK`.

---

### STEP 4 — Update `main.py`: replace constant/function bodies with imports from `src.reports`

**File:** `main.py`  
**Change:** This is the riskiest step in the report-builder stream. Do it in this exact sequence to avoid leaving `main.py` in a broken intermediate state:

4a. Add at the top of `main.py` (after existing imports, before any constants) the import block:
```python
from src.reports.builders import (
    FEATURE_HUMAN_NAMES,
    _MACRO_INNER_NAMES,
    _NUMERIC_SUFFIX_RE,
    _MACRO_PREFIX_RE,
    _REPORT_SEPARATOR,
    _SELL_DECISION,
    _MR_SELL_VETO,
    _VERIFY_5D_PRED_LABELS,
    _VERIFY_20D_PRED_LABELS,
    _VERIFY_VERDICT_LABELS,
    _REBALANCE_PRED_LABELS,
    _humanize_feature,
    _build_feature_explanation,
    _format_sentiment_status,
    _build_combined_report,
    _smart_truncate,
    _build_fallback_observability_report_vi,
    _build_sell_hold_report,
    _mr_state_line,
    _build_verify_report,
    _build_rebalance_report,
)
```

4b. Delete the constant blocks in `main.py` (they now live in `builders.py`):
- Lines 58–118: `FEATURE_HUMAN_NAMES` dict
- Lines 122–143: `_MACRO_INNER_NAMES` dict
- Lines 147–150: the two compiled regexes
- Line 665: `_REPORT_SEPARATOR`
- Lines 1409–1416: `_SELL_DECISION`, `_MR_SELL_VETO`
- Lines 1609–1623: the three verify label dicts
- Line 1824: `_REBALANCE_PRED_LABELS`

4c. Delete the function bodies for all 10 functions in `main.py` (the import from Step 4a already binds their names in the `main` namespace). Delete:
- `_humanize_feature` (lines 153–195)
- `_build_feature_explanation` (lines 198–221)
- `_format_sentiment_status` (lines 254–271)
- `_build_combined_report` (lines 668–680)
- `_smart_truncate` (lines 1088–1102)
- `_build_fallback_observability_report_vi` (lines 1105–1173)
- `_build_sell_hold_report` (lines 1419–1514)
- `_mr_state_line` (lines 1626–1639)
- `_build_verify_report` (lines 1642–1701)
- `_build_rebalance_report` (lines 1827–1848)

**CRITICAL NOTE:** Do NOT delete the `numpy as np` import at line 17 from `main.py` — it is still used elsewhere (e.g., `_compute_v3_features`, `mr_score_tickers`). Only the function body of `_build_feature_explanation` used `np` in `main.py`; after deletion, `main.py` still uses `np` in other functions.

**CRITICAL NOTE:** After deleting `_NUMERIC_SUFFIX_RE` and `_MACRO_PREFIX_RE` from `main.py`, verify no other function in `main.py` outside `_humanize_feature` references them directly. (They are only used by `_humanize_feature` which is now in `builders.py`.)

**What could go wrong:**
- Deleting a constant/function that is still used inline elsewhere in `main.py` before deleting all uses. Mitigation: Run `pytest -q` immediately after Step 4a (imports added, originals still present) to confirm no import collision. Then delete in 4b/4c.
- Name-shadowing: importing `FEATURE_HUMAN_NAMES` from `src.reports` while the dict literal still exists at line 58. Python will use the dict (later binding) over the import (earlier binding). Fix: delete the local dict before running tests.

**Verification after Step 4:** Run `pytest -q`. All 137 tests must pass. If any test fails, the import chain is broken — check the specific import error before proceeding.

---

### STEP 5 — Run full test suite (checkpoint)

**Command:** `pytest -q`  
**Expected:** 137 passed, 0 failed.  
**If failing:** Do not proceed to Step 6. Debug the import error in `src/reports/builders.py` or `main.py` re-export. Common causes: missing `numpy` import in `builders.py`, `CONFIG` not importable in the `builders.py` context, or a `_build_sell_hold_report` internal call to `_format_sentiment_status` that is not yet in scope.

---

### STEP 6 — Extract `_select_candidates()` from `daily_inference`

**File:** `main.py`  
**Change:** Before `daily_inference` definition (line 815), add the new private function `_select_candidates`. Then replace Sections D, E, and F inside `daily_inference` with a single call.

**Function specification:**

`_select_candidates` receives:
- `predictions: dict[str, list[float]]` — the full `stacking_predictions_5d` dict
- `meta_gate: dict[str, bool]` — `meta_gate_5d` from `predict_v3_horizon`
- `universe_tickers: set[str]` — the VN30-gated universe (already computed in Section D)
- `max_candidates: int` — forwarded from `daily_inference`'s parameter

Wait — the current Section D also computes `universe_tickers` from `_VN30_UNIVERSE`. Per the design decision, `_select_candidates()` combines VN30 gate + meta-gate + top-N sort. So the full set of inputs is:

- `predictions: dict[str, list[float]]` — `stacking_predictions_5d`
- `meta_gate: dict[str, bool]` — `meta_gate_5d`
- `vn30_universe: frozenset[str]` — `_VN30_UNIVERSE` constant
- `max_candidates: int`

And the function internally computes `liquid_tickers`, `universe_tickers`, `candidate_tickers`, `fallback_mode`, and `fallback_reasons`.

Returns: `tuple[list[str], set[str], bool, dict[str, str]]`
→ `(candidate_tickers, universe_tickers, fallback_mode, fallback_reasons)`

The `universe_tickers` must be returned because `_rescue_loop` needs it. Returning it avoids re-computing.

**Body of `_select_candidates`:** Copy Sections D (lines 893–906), E (lines 908–934), and F (lines 936–978) exactly. Replace the `liquid_tickers`/`universe_tickers`/`candidate_tickers`/`fallback_mode`/`fallback_reasons` variable assignments with the same logic, ending with `return candidate_tickers, universe_tickers, fallback_mode, fallback_reasons`.

**Replacement in `daily_inference`:** Lines 893–978 become:
```python
candidate_tickers, universe_tickers, fallback_mode, fallback_reasons = _select_candidates(
    stacking_predictions_5d, meta_gate_5d, _VN30_UNIVERSE, max_candidates
)
```

**What could go wrong:**
- `_gated_out` log (lines 916–920) is only used for a `LOGGER.info` call. If placed inside `_select_candidates`, it logs from the helper function — acceptable. The log content is unchanged.
- `_ARBITRATOR_POOL = 6` is a local constant inside `daily_inference`. It is used only by the `candidate_tickers` computation. It should be moved inside `_select_candidates` as a local constant or made a parameter with default `6`. Decision: keep it as an internal constant `_ARBITRATOR_POOL = 6` inside `_select_candidates` (not a parameter, since the value 6 is a system constant, not a per-call knob).
- The fallback early-return block at lines 986–1011 stays inside `daily_inference` — it calls `mr_score_tickers`, `_get_live_exec_prices`, and `_build_fallback_observability_report_vi`, all of which are impure or have side effects. It is NOT extracted. After calling `_select_candidates`, `daily_inference` checks `if fallback_mode:` and runs the early return exactly as before.

**Verification:** `pytest -q` → 137 passed.

---

### STEP 7 — Extract `_rescue_loop()` from `daily_inference`

**File:** `main.py`  
**Change:** Before `daily_inference`, add `_rescue_loop`. Then replace Section I (lines 1047–1070) with a single call.

**Function specification (per design decision):**

```python
def _rescue_loop(
    fallback_mode: bool,
    stacking_predictions_5d: dict[str, list[float]],
    universe_tickers: set[str],
    top_buy_signals: list[str],
    all_sentiments: dict[str, Any],
    horizon_predictions: dict[str, dict],
) -> tuple[list[str], dict[str, dict]]:
```

Returns: `(extended_top_buy_signals, event_overrides)`

**Body:** Copy Section I (lines 1047–1070) exactly. The `if not fallback_mode:` guard is kept — it is the entry condition. When `fallback_mode` is True, immediately `return top_buy_signals, {}` (no rescue needed in fallback). Inside the `if not fallback_mode:` block, compute `_rescue_pool`, handle missing sentiments via `evaluate_trades_batch`, call `build_event_overrides`, compute `_rescued`, build the extended list, and return `(list(top_buy_signals) + _rescued, overrides)` or `(top_buy_signals, overrides)` depending on whether rescues occurred.

**IMPORTANT — do not mutate input:** The current code at line 1067 rebinds `top_buy_signals = list(top_buy_signals) + _rescued`. Inside `_rescue_loop`, never mutate the `top_buy_signals` parameter. Always return a new list.

**Replacement in `daily_inference`:** Lines 1047–1070 become:
```python
top_buy_signals, event_overrides = _rescue_loop(
    fallback_mode,
    stacking_predictions_5d,
    universe_tickers,
    top_buy_signals,
    all_sentiments,
    horizon_predictions,
)
```

**`event_overrides` initialization:** The `event_overrides: dict[str, dict] = {}` initialization at line 1050 is removed from `daily_inference` because `_rescue_loop` now produces it. The call site above replaces lines 1050–1070 entirely.

**What could go wrong:**
- `all_sentiments` is mutated by `all_sentiments.update(_resc_sent)` inside the rescue block. In the extracted version, this mutation happens inside `_rescue_loop` on the dict reference passed by the caller. Since Python dicts are passed by reference, `all_sentiments` in `daily_inference` will be mutated as before. This is acceptable — the existing behavior is preserved. Document this in the function docstring: "Side effect: may extend `all_sentiments` in-place with fetched rescue candidate data."
- The `LOGGER.warning` call at lines 1068–1070 referencing `event_overrides` and `_rescued` must be inside `_rescue_loop` since those are local to it.

**Verification:** `pytest -q` → 137 passed.

---

### STEP 8 — Extract `_dispatch_signals()` from `run_trade_execution`

**File:** `main.py`  
**Change:** Before `run_trade_execution`, add `_dispatch_signals`. Then replace the inner for-loop (lines 1335–1395) with a call to `_dispatch_signals`.

**Function specification (per design decision):**

```python
def _dispatch_signals(
    top_buy_signals: list[str],
    all_sentiments: dict[str, Any],
    stacking_predictions: dict[str, dict],
    live_exec_prices: dict[str, float],
    event_overrides: dict[str, dict] | None,
    top_pos_features: str,
    top_neg_features: str,
    horizon: int,
    broadcast: bool,
    bot: TelegramBot,
) -> list[dict]:
```

Returns: `dispatched_signals` list (list of signal dicts for `_build_combined_report`).

**Body:** Copy lines 1335–1395 from `run_trade_execution`. Note: `_LATEST_REGIME_BY_TICKER` is a module-level global (defined at line 529). The extracted function accesses it as a free variable (module global) — this is the explicit global-coupling that must be documented in the docstring: "Reads module-level global `_LATEST_REGIME_BY_TICKER` set by `_compute_v3_features`. This coupling is intentional and documented."

**Replacement in `run_trade_execution`:** Lines 1335–1395 become:
```python
dispatched_signals = _dispatch_signals(
    top_buy_signals=top_buy_signals,
    all_sentiments=all_sentiments,
    stacking_predictions=stacking_predictions,
    live_exec_prices=live_exec_prices,
    event_overrides=event_overrides,
    top_pos_features=top_pos_features,
    top_neg_features=top_neg_features,
    horizon=horizon,
    broadcast=broadcast,
    bot=bot,
)
```

`bot` is the `TelegramBot()` instance created at line 1316. It is now passed as a parameter rather than created inside the loop body.

**The market-summary header block (lines 1327–1333)** stays inside `run_trade_execution` because it is a single `bot.send_text_alert` call with simple logic, not part of the per-ticker loop being extracted. `_dispatch_signals` handles only the per-ticker loop.

**What could go wrong:**
- `_LATEST_REGIME_BY_TICKER` being accessed as a global inside a new function. Python will find it at module scope via normal LEGB lookup — no code change needed, but it must be documented.
- `sent` counter (local to `run_trade_execution`) is updated by the old loop. After extraction, `sent = len(dispatched_signals)` replaces the incremental counter. Verify the `LOGGER.info("Telegram alerts dispatched: %s...")` call uses `sent = len(dispatched_signals)`.

**Verification:** `pytest -q` → 137 passed.

---

### STEP 9 — Write unit tests for `_select_candidates()`

**File:** `tests/test_select_candidates.py`  
**Framework:** pytest, no fixtures needed (pure function)

**Test cases:**

| Test name | Input state | Expected output |
|---|---|---|
| `test_vn30_filter_keeps_only_universe_members` | predictions for VN30 + non-VN30 tickers, meta_gate all True | candidate_tickers only contains VN30 members |
| `test_meta_gate_rejects_unprofitable_tickers` | 3 VN30 tickers, meta_gate says 1 is False | that ticker absent from candidates |
| `test_top_n_cap_limits_candidates` | 10 VN30 tickers all meta-gated True, max_candidates=3 | len(candidates) == 3 |
| `test_candidates_sorted_by_p_up_desc` | 3 VN30 tickers with P(UP) 0.7, 0.5, 0.6 | order is [0.7, 0.6, 0.5] |
| `test_fallback_mode_triggered_when_no_candidates` | 0 tickers pass gate | fallback_mode == True |
| `test_fallback_mode_selects_top3_by_p_up` | 5 VN30 tickers, none pass meta_gate | candidates == top-3 by P(UP) |
| `test_fallback_reasons_include_low_p_up_reason` | fallback scenario, ticker with P(UP) < tau | fallback_reasons[ticker] contains "ngưỡng an toàn" |
| `test_fallback_reasons_include_meta_gate_reason` | fallback scenario, ticker passes P(UP) but fails meta_gate | fallback_reasons[ticker] contains "bộ lọc" |
| `test_no_fallback_when_candidates_exist` | normal gate pass | fallback_mode == False, fallback_reasons == {} |
| `test_vn30_universe_empty_fallback_uses_all_predictions` | no VN30 tickers in predictions | falls back to all predictions as universe |

**Import:** `from main import _select_candidates, _VN30_UNIVERSE`

**What could go wrong:** The test imports `_select_candidates` which is a private function added in Step 6. If Step 6 was not completed, this import fails. Run Step 6 first.

**Verification:** `pytest -q tests/test_select_candidates.py` → 10 passed.

---

### STEP 10 — Write unit tests for `_rescue_loop()`

**File:** `tests/test_rescue_loop.py`  
**Framework:** pytest with `unittest.mock.patch` for `evaluate_trades_batch`

**Test cases:**

| Test name | Setup | Expected |
|---|---|---|
| `test_fallback_mode_returns_unchanged_signals_and_empty_overrides` | fallback_mode=True | returns (top_buy_signals, {}) immediately |
| `test_no_rescue_candidates_returns_unchanged` | all universe tickers either held or outside rescue range | event_overrides == {}, top_buy_signals unchanged |
| `test_rescue_candidate_fetches_missing_sentiment` | rescue candidate missing from all_sentiments | evaluate_trades_batch called once for missing ticker |
| `test_rescue_candidate_added_when_sentiment_strong` | P(UP)=0.43, sentiment=0.8 in rescue range | ticker appended to returned list |
| `test_all_sentiments_mutated_with_rescue_data` | rescue candidate sentiment fetched | all_sentiments dict extended in-place |
| `test_sentiment_fetch_failure_is_swallowed` | evaluate_trades_batch raises Exception | no exception propagated, continues gracefully |
| `test_bear_veto_applied_via_build_event_overrides` | top_buy_signal has sentiment <= -0.5 | returned event_overrides includes veto entry |
| `test_output_list_is_new_object_not_mutated_input` | rescue adds ticker | returned list is not the same object as input |

**Mocking approach:** Patch `main.evaluate_trades_batch` using `unittest.mock.patch("main.evaluate_trades_batch", ...)` to return controlled `(final_decisions, sentiments)` tuple. Do not hit the real Gemini API.

**Import:** `from main import _rescue_loop`

**Verification:** `pytest -q tests/test_rescue_loop.py` → 8 passed.

---

### STEP 11 — Write integration test for `daily_inference` (happy path + fallback path)

**File:** `tests/test_daily_inference_integration.py`  
**Framework:** pytest with `unittest.mock.patch`

**Strategy:** Boundary-patch the three I/O-boundary functions:
1. `main.predict_v3_horizon` — patch to return a controlled `(predictions_dict, tau, xgb_stub, features_list, meta_gate_dict)` tuple.
2. `main.evaluate_trades_batch` — patch to return controlled `(final_decisions, all_sentiments)`.
3. `main.run_trade_execution` — patch to return a fixed HTML string `"<b>test report</b>"`.

Also patch:
- `main.Alpha360Generator` — so `load_live_ohlcv_window` returns a small Polars DataFrame (3 VN30 tickers, minimal columns).
- `main.mr_score_tickers` — returns empty dict (used only in fallback path).
- `main._get_live_exec_prices` — returns price dict (used in fallback path).
- `main._build_fallback_observability_report_vi` — returns fixed string (fallback path test).

**Happy path test (`test_daily_inference_happy_path`):**

Setup:
- `predict_v3_horizon` returns 3 VN30 tickers (`VCB`, `BID`, `VHM`) with P(UP) = [0.70, 0.65, 0.60], all meta_gate=True.
- `evaluate_trades_batch` returns final_decisions with all 3 tickers = 2 (BUY), sentiments with score=0.3 each.
- `run_trade_execution` returns `"<b>test report</b>"`.

Assertions:
- `daily_inference(broadcast=False)` returns `"<b>test report</b>"`.
- `run_trade_execution` was called exactly once.
- `run_trade_execution` was called with `top_buy_signals` containing the 3 tickers (order by sentiment+quant).
- `evaluate_trades_batch` was called exactly once with `candidate_tickers` = 3 VN30 tickers.

**Fallback path test (`test_daily_inference_fallback_path`):**

Setup:
- `predict_v3_horizon` returns 3 VN30 tickers with P(UP) = [0.30, 0.28, 0.25], meta_gate all False (triggers fallback).
- `evaluate_trades_batch` returns empty decisions, no sentiment scores.
- `_build_fallback_observability_report_vi` returns `"<b>fallback report</b>"`.

Assertions:
- `daily_inference(broadcast=False)` returns `"<b>fallback report</b>"`.
- `run_trade_execution` was NOT called.
- `_build_fallback_observability_report_vi` was called once.

**Rescue loop test (`test_daily_inference_rescue_loop_invoked`):**

Setup:
- 3 tickers pass gate → 2 are BUY decisions, 1 is a rescue candidate (P(UP)=0.43, sentiment=0.8).
- `_rescue_loop` is NOT patched — let it run with the patched `evaluate_trades_batch`.
- `run_trade_execution` is patched.

Assertions:
- `run_trade_execution` called with `top_buy_signals` containing 3 tickers (2 standard + 1 rescued).
- `event_overrides` passed to `run_trade_execution` contains the rescued ticker with `weight == 0.05`.

**What could go wrong:**
- `Alpha360Generator.load_live_ohlcv_window` returns a Polars DataFrame. Test must construct a minimal valid Polars DataFrame with required columns (`ticker`, `raw_close`). Use `polars.DataFrame({"ticker": ["VCB", "BID", "VHM"], "raw_close": [50000.0, 45000.0, 40000.0]})`. This might need additional columns if `_compute_v3_features` is called — but since `predict_v3_horizon` is patched, `latest_df` is only consumed by the patch boundary, not the real model. Confirm by checking what `daily_inference` does with `latest_df` before calling `predict_v3_horizon`.
  - `latest_df` is used at line 850: `if latest_df.empty: raise ValueError(...)` — need to call `.to_pandas()` on the Polars frame before this check. The test must patch at the boundary so `latest_df.empty` is False.
  - Actually `latest_df = live_pl.to_pandas()` is called on line 848. So we patch `Alpha360Generator` such that `generator.load_live_ohlcv_window(...)` returns a Polars DataFrame, which `.to_pandas()` converts to a Pandas DataFrame for the `latest_df.empty` check.
- `daily_inference` at line 851 calls `len(latest_df)` and `len(latest_df.columns)` for logging. This is fine with a minimal DataFrame.
- Secondary horizon `predict_v3_horizon` call (line 867) for `_secondary_h=20` will also hit the patch (it uses the same mock), which must distinguish by the `horizon` argument. Use `side_effect` on the mock to return a different dict for horizon=20 (can return empty dict `{}` safely — that's the graceful degradation path).

**Verification:** `pytest -q tests/test_daily_inference_integration.py` → 3 passed.

---

### STEP 12 — Full test suite final verification

**Command:** `pytest -q`  
**Expected:** 137 + (new tests from Steps 9, 10, 11) passed, 0 failed.  
**Count:** 137 existing + 10 (select_candidates) + 8 (rescue_loop) + 3 (integration) = minimum 158 tests.

---

### STEP 13 — Manual smoke test (optional, strongly recommended before merge)

**Not automated — requires live environment.**  
Run the pipeline in dry-run mode to confirm no import errors or runtime regressions:

```bash
python main.py --daily-inference --broadcast false
```

Or if a `--dry-run` flag exists, use that. Check:
- No import errors on startup.
- Log output shows `_select_candidates`, `_rescue_loop`, `_dispatch_signals` being called (add a single `LOGGER.debug` to each extracted function, or verify by log output structure).
- No behavioral change in the report output compared to a baseline run.

---

## Data Flow After Phase 1

```
daily_inference(window_rows, max_candidates, broadcast, horizon)
  │
  ├─ Alpha360Generator.load_live_ohlcv_window()
  │    → live_pl (Polars DataFrame)
  │    → latest_df = live_pl.to_pandas()
  │
  ├─ predict_v3_horizon(latest_df, horizon)
  │    → stacking_predictions_5d, thr_5d, xgb_model_5d, selected_features_5d, meta_gate_5d
  │
  ├─ predict_v3_horizon(latest_df, _secondary_h)  [may fail gracefully → {}]
  │    → stacking_predictions_20d
  │
  ├─ _select_candidates(stacking_predictions_5d, meta_gate_5d, _VN30_UNIVERSE, max_candidates)
  │    → (candidate_tickers, universe_tickers, fallback_mode, fallback_reasons)
  │    [PURE — no I/O]
  │
  ├─ evaluate_trades_batch(horizon_predictions, candidate_tickers)
  │    → final_decisions, all_sentiments
  │
  ├─ if fallback_mode → _build_fallback_observability_report_vi(...) → return
  │
  ├─ sentiment_ranking → top_buy_signals (top 3)
  │
  ├─ _rescue_loop(fallback_mode, stacking_predictions_5d, universe_tickers,
  │               top_buy_signals, all_sentiments, horizon_predictions)
  │    → (top_buy_signals [extended], event_overrides)
  │    [IMPURE — may call evaluate_trades_batch for rescue sentiment]
  │
  └─ run_trade_execution(top_buy_signals, final_decisions, all_sentiments,
                          horizon_predictions, latest_df, xgb_model_5d,
                          selected_features_5d, horizon, broadcast, event_overrides)
       │
       ├─ portfolio updates (PortfolioManager)
       ├─ RL logging (_log_rl_predictions, _backfill_rl_outcomes)
       ├─ header message (bot.send_text_alert)
       └─ _dispatch_signals(...)
            │    [reads _LATEST_REGIME_BY_TICKER global]
            └─ per-ticker signal build + bot.send_signal_alert → dispatched_signals
       └─ _build_combined_report(dispatched_signals) → report_html
```

---

## Failure Modes and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Import cycle: `src/reports/builders.py` → `src/utils/telegram_alerter.py` → back to `src/reports` | Low — `telegram_alerter.py` doesn't import from `src/reports` | Verify with `python -c "from src.reports.builders import _build_combined_report"` after Step 2 |
| `_build_feature_explanation` uses `np.asarray` but `builders.py` missing `import numpy as np` | Medium | Add `import numpy as np` at top of `builders.py` as Step 2 requirement |
| `_build_sell_hold_report` references `CONFIG.trading.*` but `config.settings` not importable in test env | Low — conftest stubs only a limited list, `config.settings` is a pure dataclass | Verify `pytest -q` after Step 4 |
| `_select_candidates` extracts `_ARBITRATOR_POOL = 6` local but existing code also has it inline | Low — it is only in `daily_inference`, not a module-level constant | Delete the inline version when replacing the code block |
| `_rescue_loop` mutates `all_sentiments` in-place via `update()` — caller may not expect this | Medium — existing behavior preserved but now implicit | Document explicitly in function docstring |
| `_dispatch_signals` reads `_LATEST_REGIME_BY_TICKER` as a free variable | Low — Python LEGB resolves module globals correctly | Document in docstring; no code change needed |
| Existing tests import specific functions from `main` by name; those names not in `main` namespace after deletion | High if re-exports are forgotten | Steps 4a runs BEFORE 4b/4c — imports are added before deletions |
| Secondary `predict_v3_horizon` call (horizon=20) raises `FileNotFoundError` in test env | Medium — mock must handle both calls | Use `side_effect` list or `wraps` to differentiate horizon args |
| `daily_inference` line count after refactor exceeds 145 (fallback block is ~35 lines alone) | Low — target is ~145, not a hard limit | Recount after Steps 6–7; acceptable if fallback block explanation comments are preserved |

---

## Verification Evidence

The plan is DONE only when ALL of the following are observable:

1. `pytest -q` exits 0 with at least 158 tests passing (137 existing + minimum 21 new).
2. `python -c "from src.reports import _smart_truncate, _build_combined_report, _build_sell_hold_report"` — exits without error.
3. `python -c "from main import _humanize_feature, _build_feature_explanation, _smart_truncate"` — exits without error (re-exports work).
4. `python -c "from main import _select_candidates, _rescue_loop, _dispatch_signals"` — exits without error.
5. `python -c "from main import build_event_overrides, SAFE_BUY_THRESHOLD, EVENT_MIN_P_UP"` — exits without error (event constants NOT moved, remain in `main`).
6. Line count of `daily_inference` after refactor: `grep -n "^def daily_inference" main.py` → start line; count to next `^def` → should be under 170 lines (was 271).
7. Line count of `run_trade_execution` after refactor: should be under 110 lines (was 156).
8. No new `import main` statements needed in any existing test file.

---

## Dependencies

| Dependency | Type | Impact if violated |
|---|---|---|
| Step 2 must complete before Step 3 | Hard | `__init__.py` re-exports a non-existent `builders.py` |
| Step 3 must complete before Step 4 | Hard | `main.py` imports from `src.reports` before package exists |
| Step 4 must complete before Step 5 | Hard | `pytest -q` checkpoint requires import chain to be live |
| Step 5 must be green before Step 6 | Hard | Decomposing `daily_inference` while imports are broken hides bugs |
| Step 6 must complete before Step 9 | Hard | Test imports `_select_candidates` which doesn't exist yet |
| Step 7 must complete before Step 10 | Hard | Test imports `_rescue_loop` which doesn't exist yet |
| Steps 6 and 7 may be done in parallel | Soft | Both modify `daily_inference` — do sequentially to avoid conflicts |
| Steps 8, 9, 10, 11 are independent of each other | Soft | Can be reordered if needed |

---

## Rollback Notes

Phase 1 makes additive + mechanical changes. Rollback approach:

- **Before Step 4:** The new `src/reports/` package does not affect `main.py` yet. Rollback is `rm -rf src/reports/`.
- **After Step 4 (imports + deletion):** `git diff main.py` shows exactly what was deleted. Rollback is `git checkout -- main.py` + `rm -rf src/reports/`.
- **No schema migrations, no model changes, no DuckDB writes.** This refactor is pure code reorganization. Production behavior is unchanged.

---

## Resume and Execution Handoff

**Plan file:** `process/features/v4-1-structural-debt/active/phase1-decompose-daily-inference_PLAN_09-06-26.md`

**Executor:** Start with Step 1. Steps are numbered and each has an explicit verification command. Do NOT skip verification steps.

**Current step if resuming mid-plan:** Check which `src/reports/` files exist and whether `pytest -q` passes to determine resume point:

| State | Resume at |
|---|---|
| `src/reports/` does not exist | Step 1 |
| `src/reports/__init__.py` exists (empty) | Step 2 |
| `src/reports/builders.py` exists but `__init__.py` empty | Step 3 |
| `src/reports/__init__.py` has re-exports but `main.py` still has full function bodies | Step 4 |
| `main.py` imports from `src.reports`, `pytest -q` passing | Step 6 |
| `_select_candidates` exists in `main.py`, `pytest -q` passing | Step 7 |
| `_rescue_loop` exists in `main.py`, `pytest -q` passing | Step 8 |
| All functions extracted, tests not yet written | Step 9 |

**Single execute command to confirm full completion:**
```bash
pytest -q && python -c "from src.reports import _smart_truncate; from main import _select_candidates, _rescue_loop, _dispatch_signals; print('PHASE 1 COMPLETE')"
```

---

## Acceptance Criteria

- [ ] `src/reports/__init__.py` and `src/reports/builders.py` exist with all 10 functions.
- [ ] `main.py` imports all 10 report builder functions from `src.reports.builders` (no duplicate definitions).
- [ ] `main.py` contains `_select_candidates()` with type-annotated signature.
- [ ] `main.py` contains `_rescue_loop()` with type-annotated signature and global-coupling note in docstring.
- [ ] `main.py` contains `_dispatch_signals()` with type-annotated signature and `_LATEST_REGIME_BY_TICKER` coupling note in docstring.
- [ ] `daily_inference` body is under 170 lines.
- [ ] `run_trade_execution` body is under 110 lines.
- [ ] `tests/test_select_candidates.py` contains at least 10 test cases, all passing.
- [ ] `tests/test_rescue_loop.py` contains at least 8 test cases, all passing.
- [ ] `tests/test_daily_inference_integration.py` contains happy path, fallback path, and rescue loop tests, all passing.
- [ ] `pytest -q` exits 0 with at least 158 tests passing.
- [ ] No existing test file requires any change.
- [ ] Event constants (`SAFE_BUY_THRESHOLD`, `EVENT_MIN_P_UP`, `EVENT_BULL_SENTIMENT`, `EVENT_BEAR_SENTIMENT`, `_EVENT_CAP`) remain in `main.py` (untouched).
- [ ] `build_event_overrides` remains in `main.py` (untouched).
