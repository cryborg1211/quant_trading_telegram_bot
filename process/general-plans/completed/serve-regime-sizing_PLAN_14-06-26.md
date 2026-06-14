# Serve-Path Regime-Conditional Sizing — Implementation Plan

**Classification:** SIMPLE (single-session, 3-file surgical change, 1 new test file)
**Date:** 14-06-26
**Author:** vc-plan-agent
**Status:** SHIPPED 2026-06-14 (commit b3eda3e). `_dispatch_signals` applies regime sizing in the else-branch (NO_TRADE skip / PENALTY 0.5×), gated by `CONFIG.trading.regime_sizing_enabled` (default ON, settings.json kill-switch). 228 tests green. Execute fix: NO_TRADE skip moved into the else-branch to preserve event-override precedence.

---

## 1. Overview and Goal

The backtest engine gained regime-conditional tranche sizing in commit 77c1412 (A/B validated:
MaxDD ~23% → ~17%, Sharpe +0.73 → +0.88). The live serve path (`_dispatch_signals` in `main.py`)
does NOT yet apply the same logic: a NO_TRADE regime name is dispatched with its full cohort weight,
and a PENALTY regime name is not penalised. This is a train/serve parity gap.

**Goal:** Wire the same regime policy into `_dispatch_signals` — guarded by a `TradingConfig`
kill-switch (default ON) — so paper-trading uses the identical DD-controlling sizing the backtest
validated, without changing any backtest engine code, model artefacts, or training pipelines.

**A/B evidence:** `process/general-plans/active/regime-conditional-tranche-sizing_PLAN_13-06-26.md`
and memory key `regime-sizing-ab-result`.

---

## 2. Scope

### In scope

- `config/settings.py` — add `regime_sizing_enabled: bool = True` to `TradingConfig`
- `config/settings.json` — add `"regime_sizing_enabled": true` to the `"trading"` block
- `main.py` — add import from `src.trading.regime_policy`; add regime skip/penalty logic inside
  `_dispatch_signals` (non-event-override branch only); no signature changes
- `tests/test_dispatch_regime_sizing.py` — new test file (5 test cases)

### Out of scope

- `src/backtest/walk_forward.py` — already shipped, no changes
- `src/bot/sizing.py` — not changed; `suggested_weight` continues to apply regime logic for the
  legacy half-Kelly path; tranche path is handled directly in `_dispatch_signals`
- `train_models.py`, `FEATURE_RECIPE_VERSION`, any model artefacts — no retrain
- `_select_candidates`, candidate-selection funnel — regime filtering stays at DISPATCH only
- `/verify`, `/suggest_sell`, `/rebalance`, `/exits` commands — no sizing involved; out of scope
- Telegram card format — a skipped name produces no card at all (fewer messages, not more);
  no 4096-char risk introduced

---

## 3. Precise Serve Sizing Semantics and Parity Statement

### 3a. Event-override precedence (unchanged)

Lines 1074-1078 in `main.py`: if `_ov` is truthy (event-override dict present for ticker), the
weight, status, and `ly_do` come entirely from the override. Regime logic MUST NOT touch this
branch. The new code lives exclusively in the `else` block (lines 1079-1086).

### 3b. Regime dispatch rules (mirrors `_tranche_day` at walk_forward.py:822-828)

For each ticker in the `for ticker in top_buy_signals` loop (line 1058):

| Condition | Action | Ledger | Cash |
|---|---|---|---|
| `regime_sizing_enabled` is False | No-op — existing path byte-for-byte | unchanged | unchanged |
| `_regime` is None | No-op — pre-regime artefact, no action | dispatched | unchanged |
| `_regime in NO_TRADE_REGIMES {0,7}` | `continue` — skip ticker entirely | NOT recorded | stays cash |
| `_regime in PENALTY_REGIMES {1,6}` | Multiply resolved `_w` by `REGIME_PENALTY_FACTOR (0.5)` | dispatched at 0.5× weight | half stays cash |
| Any other regime (2,3,4,5) | No-op | dispatched at full weight | unchanged |

### 3c. Fixed-denominator invariant (cash-preserving parity)

`_tranche_signal_fields` (line 1006-1031) computes `weight = 1.0 / (hold_days * max(1, n_picks))`
where `n_picks = len(top_buy_signals)` is frozen at the call site (line 1056, before the dispatch
loop). Skipping a NO_TRADE name inside the loop MUST NOT recompute `n_picks`. The weight for
surviving names stays at `1 / (hold_days * original_n_picks)`. This is the cash-preserving
Design-2 behaviour: freed budget stays as cash, NOT redistributed to survivors. Exactly mirrors
`_tranche_day` which uses a pre-computed `per_name` that never changes as names are skipped.

### 3d. Ledger behaviour for skipped names

`signal_ledger.record_dispatch` (line 1236) is called with `dispatched_signals` after the loop.
A name skipped via `continue` is never appended to `dispatched_signals`. The ledger sees only
the actually dispatched names — correct and consistent with the backtest no-fill semantics.
No schema change to the ledger is required.

### 3e. Legacy half-Kelly path (non-tranche artefacts)

When `tranche_fields` is empty (no `strategy` dict or mode != "tranche"), `_w` resolves via
`suggested_weight(float(_p5[2]), market_regime=_regime)` (line 1082-1084). That function in
`src/bot/sizing.py` already applies `NO_TRADE_REGIMES` (returns 0.0) and `PENALTY_REGIMES`
(shrinks cap). With the new dispatch-level guard active, the NO_TRADE skip (`continue`) would
fire before we build `signal_data` at all, which is consistent. The PENALTY case would double-
count if `suggested_weight` already applied the penalty and we then multiply `_w` by 0.5 again.

**Resolution:** The PENALTY 0.5× multiplication at dispatch level MUST be applied only when in
tranche mode (i.e., `tranche_fields` is non-empty). For the legacy half-Kelly path, `suggested_weight`
already applies the penalty via `REGIME_PENALTY_CAP`, so dispatch must not re-penalise.
Only the NO_TRADE `continue` skip applies to both modes.

Concretely:
- NO_TRADE skip (`continue`) — applies regardless of tranche or legacy mode
- PENALTY 0.5× multiply — applies ONLY when `tranche_fields` is non-empty (tranche mode)

---

## 4. Exact Touchpoints (file:line)

### 4a. `config/settings.py` — TradingConfig (line 61-70 current)

**Change:** Add one field at the end of the `TradingConfig` dataclass:

```
regime_sizing_enabled: bool = True
```

Location: after `default_telegram_id: str = "default_user"` (currently line 70). No other
changes to `Config`, `from_json`, or any other dataclass.

### 4b. `config/settings.json` — trading block (lines 36-45 current)

**Change:** Add one key to the `"trading"` JSON object:

```
"regime_sizing_enabled": true
```

Location: append after `"default_telegram_id": "default_user"` (currently line 44). No other
JSON changes.

### 4c. `main.py` — import block (approximately line 43, after `from src.trading import signal_ledger`)

**Change:** Add import of the two regime sets and the penalty factor:

```
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_FACTOR,
)
```

Note: `NO_TRADE_REGIMES`, `PENALTY_REGIMES` are already in scope via `src.bot.sizing` re-exports
that exist in module globals, but they are not currently imported directly at the `main.py` top
level. Adding the explicit import from `regime_policy` is safer, avoids magic re-export reliance,
and is consistent with how `walk_forward.py` imports them. Verify no name collision with the
`suggested_weight` import from `src.bot.sizing` (which already re-exports the same objects but
does not make them available at module scope in `main.py` — only `suggested_weight` and
`regime_label_vi` are currently imported).

### 4d. `main.py` — `_dispatch_signals` function body (lines 1058-1086 current)

The `for ticker in top_buy_signals:` loop currently has this structure (simplified):

```
for ticker in top_buy_signals:
    exec_price = live_exec_prices.get(ticker)
    if exec_price is None:
        ...continue...
    # [sentiment, _p5, confidence_5d computed here]
    _regime = _LATEST_REGIME_BY_TICKER.get(ticker)          # line 1071
    _ov = (event_overrides or {}).get(ticker)               # line 1074
    if _ov:
        _w = float(_ov["weight"])                           # event-override branch
        ...
    else:
        _w = tranche_fields.get("suggested_weight", ...)    # current line 1082-1084
        _status = "MUA"
        _ly_do = ""
    signal_data = {...}
    ...
    dispatched_signals.append(signal_data)
```

**Change:** Both new guards go INSIDE the `else` block (lines 1079-1086), so event-override
names (the `if _ov:` branch) keep precedence and are never affected. The exact insertion sequence:

STEP 1 — Immediately after `else:` (line 1079), as the FIRST statements in the else block (before
`_w = tranche_fields.get(...)`), insert the NO_TRADE skip:

```
        # ── Regime-conditional sizing (serve/backtest parity) ──────────────
        # In the else (non-event-override) branch ONLY, so event overrides win.
        if (CONFIG.trading.regime_sizing_enabled
                and _regime is not None
                and _regime in NO_TRADE_REGIMES):
            LOGGER.info("[Regime] %s skipped — NO_TRADE regime %s", ticker, _regime)
            continue  # skip this ticker entirely; freed cohort weight stays cash
```

**CRITICAL:** This MUST be inside the `else` block, NOT before `_ov = ...`. Placing it before the
`if _ov:` check would skip a NO_TRADE name even when it has an event override — breaking the locked
event-override-precedence decision and test TC-4.

STEP 2 — Inside the same `else` block, after `_w = tranche_fields.get("suggested_weight", ...)` is
resolved (after line 1084), insert PENALTY multiplication — but only in tranche mode:

```
        # PENALTY 0.5× applies only in tranche mode (legacy half-Kelly already
        # applies REGIME_PENALTY_CAP inside suggested_weight).
        if (CONFIG.trading.regime_sizing_enabled
                and tranche_fields  # non-empty ↔ tranche mode
                and _regime is not None
                and _regime in PENALTY_REGIMES):
            _w = _w * REGIME_PENALTY_FACTOR
            LOGGER.info(
                "[Regime] %s PENALTY regime %s → weight %.4f (0.5×)",
                ticker, _regime, _w,
            )
```

No other lines in the function body change. The `signal_data["market_regime"] = _regime`
(line 1096) remains unchanged — the regime label on the Telegram card is informational and
is unaffected by the sizing decision.

### 4e. How `CONFIG` is reached inside `_dispatch_signals`

`CONFIG` is the module-level singleton already imported at line 21 (`from config.settings import CONFIG`).
`_dispatch_signals` has no local shadowing. `CONFIG.trading.regime_sizing_enabled` is accessible
directly — no parameter threading needed.

---

## 5. Config Flag Wiring

### Default state (ON)

`TradingConfig.regime_sizing_enabled = True` means every live dispatch after deploy applies the
regime policy. This is the intended post-ship default — the A/B validation already showed this
is net-positive.

### Kill-switch (disable via settings.json)

To revert to the current dispatch behaviour without a code deploy:

```json
{
  "trading": {
    "regime_sizing_enabled": false
  }
}
```

Because `Config.from_json()` deserialises `trading` block via `TradingConfig(**raw.get("trading", {}))`,
the `regime_sizing_enabled` key is picked up automatically. The `CONFIG` singleton is built once
at import time (line 125 of settings.py: `CONFIG = Config.from_json()`), so a settings.json change
requires a process restart (already the norm for the systemd daily-cron bot).

### Flag-OFF semantic guarantee

When `regime_sizing_enabled` is False, NEITHER the `continue` skip NOR the `_w *= REGIME_PENALTY_FACTOR`
line executes. The dispatch loop is byte-for-byte the current behaviour. This must be verified
in tests (see Section 6).

---

## 6. Implementation Checklist

1. Open `config/settings.py` and add `regime_sizing_enabled: bool = True` as the last field in
   `TradingConfig` (after `default_telegram_id`, line 70). Save.

2. Open `config/settings.json` and add `"regime_sizing_enabled": true` to the `"trading"` object
   (after `"default_telegram_id"` key, line 44). Save.

3. Open `main.py`. Locate the import block around line 43-44. Add the explicit import:
   ```
   from src.trading.regime_policy import (
       NO_TRADE_REGIMES,
       PENALTY_REGIMES,
       REGIME_PENALTY_FACTOR,
   )
   ```
   Place it after `from src.trading import signal_ledger` for logical grouping.

4. Open `main.py`. Navigate to `_dispatch_signals` body. Insert the NO_TRADE `continue` guard as
   the FIRST statements INSIDE the `else` block (right after `else:` on line 1079, before
   `_w = tranche_fields.get(...)`), NOT before `_ov = ...` (see Section 4d Step 1 — placement is
   critical for event-override precedence / TC-4).

5. Open `main.py`. Inside `_dispatch_signals` `else` block, after `_w = tranche_fields.get(...)`
   resolves (after line 1084), insert the PENALTY 0.5× block guarded by `tranche_fields` check
   (see Section 4d Step 2).

6. Create `tests/test_dispatch_regime_sizing.py` with the 5 test cases defined in Section 7.

7. Run the full test suite:
   ```
   pytest --tb=short -q
   ```
   Expect: all 223 existing tests green plus 5 new tests green (total 228). Confirm the actual
   pre-change count first (the all-context "158" figure is stale; real baseline is 223).

8. Verify `FEATURE_RECIPE_VERSION` is unchanged (no accidental edit to `src/backtest/pipeline.py`
   or `src/utils/schema_hash.py`). Check: `grep -r FEATURE_RECIPE_VERSION src/`.

---

## 7. Test Plan

**New file:** `tests/test_dispatch_regime_sizing.py`

**Import strategy for `_dispatch_signals`:** The function is a module-level private in `main.py`.
It reads `_LATEST_REGIME_BY_TICKER` (module global) and `CONFIG.trading.regime_sizing_enabled`.
Direct unit-testing requires:
- Import `_dispatch_signals` from `main`
- Populate `main._LATEST_REGIME_BY_TICKER` directly (dict mutation)
- Patch `CONFIG.trading.regime_sizing_enabled` via `unittest.mock.patch.object` or direct
  attribute assignment (the dataclass is not frozen)
- Provide minimal stubs: `top_buy_signals` list, `stacking_predictions` dict with `"5d"` key,
  `live_exec_prices` dict (one price per ticker), `all_sentiments` empty dict, `event_overrides`
  None, `top_pos_features`/`top_neg_features` empty strings, `horizon` int, `broadcast=False`,
  `bot` stub (object with no-op `send_signal_alert`), `strategy` tranche dict

The function returns `list[dict]` when `broadcast=False` without calling `bot.send_signal_alert`.
The list length and the `suggested_weight` values on each dict are the assertion targets.

**Test cases:**

### TC-1: NO_TRADE regime — ticker not dispatched

Setup:
- `top_buy_signals = ["AAA", "BBB"]`
- `_LATEST_REGIME_BY_TICKER = {"AAA": 0, "BBB": 2}`  (AAA = Freeze)
- `regime_sizing_enabled = True`
- `strategy = {"mode": "tranche", "hold_days": 30}`
- `live_exec_prices = {"AAA": 25000.0, "BBB": 30000.0}`

Assert:
- `len(result) == 1`
- `result[0]["ticker"] == "BBB"`

### TC-2: PENALTY regime — weight halved in tranche mode

Setup:
- `top_buy_signals = ["AAA", "BBB"]`
- `_LATEST_REGIME_BY_TICKER = {"AAA": 1, "BBB": 2}`  (AAA = Squeeze)
- `regime_sizing_enabled = True`
- `strategy = {"mode": "tranche", "hold_days": 30}`
- Expected tranche weight = `1 / (30 * 2) = 0.01667`

Assert:
- `len(result) == 2`
- `result[0]["suggested_weight"] == pytest.approx(0.01667 * 0.5, rel=1e-4)`  (AAA halved)
- `result[1]["suggested_weight"] == pytest.approx(0.01667, rel=1e-4)`         (BBB unchanged)

### TC-3: Flag OFF — NO_TRADE name is dispatched (regime ignored)

Setup: same as TC-1 but `regime_sizing_enabled = False`

Assert:
- `len(result) == 2`
- Both "AAA" and "BBB" in `{r["ticker"] for r in result}`

### TC-4: Event override takes precedence over regime

Setup:
- `top_buy_signals = ["AAA"]`
- `_LATEST_REGIME_BY_TICKER = {"AAA": 0}`  (NO_TRADE)
- `regime_sizing_enabled = True`
- `event_overrides = {"AAA": {"weight": 0.05, "status": "OVERRIDE", "ly_do": "test"}}`

Assert:
- `len(result) == 1`  (name IS dispatched because event override fires before regime skip)
- `result[0]["suggested_weight"] == pytest.approx(0.05)`

### TC-5: None regime (pre-regime artefact) — no-op

Setup:
- `top_buy_signals = ["AAA"]`
- `_LATEST_REGIME_BY_TICKER = {}`  (AAA absent → `.get` returns None)
- `regime_sizing_enabled = True`
- `strategy = {"mode": "tranche", "hold_days": 30}`
- `live_exec_prices = {"AAA": 25000.0}`
- Expected tranche weight = `1 / (30 * 1) = 0.03333`

Assert:
- `len(result) == 1`
- `result[0]["suggested_weight"] == pytest.approx(0.03333, rel=1e-4)`  (no regime modification)

**Existing tests that remain GREEN without modification:**
- `tests/test_strategy_serve_path.py` — `_tranche_signal_fields` tests; no dispatch involved
- `tests/test_market_regime.py` — `suggested_weight` regime overrides (no change to sizing.py)
- `tests/test_regime_tranche_sizing.py` — backtest engine (no change to walk_forward.py)
- `tests/test_sizing.py` — pure Kelly math (no change to sizing.py)
- `tests/test_signal_ledger.py` — ledger schema (no change)
- `tests/test_select_candidates.py` — candidate funnel (no change)

---

## 8. Data Flow

```
daily_inference (main.py)
  └─> _compute_v3_features()
        └─> populates _LATEST_REGIME_BY_TICKER {ticker: int}   [already live, no change]
  └─> run_trade_execution()
        └─> _tranche_signal_fields(strategy, n_picks=len(top_buy_signals))
              └─> tranche_fields = {"suggested_weight": 1/(hold_days*n_picks), ...}
        └─> _dispatch_signals(top_buy_signals, ..., strategy=_strategy)
              for ticker in top_buy_signals:
                _regime = _LATEST_REGIME_BY_TICKER.get(ticker)         [int | None]
                if _ov: _w = override weight                            [event-override wins, unchanged]
                else:
                  [NEW] if enabled and regime in NO_TRADE: continue     [cash-preserving skip]
                  _w = tranche_fields.get("suggested_weight", half-Kelly)
                  [NEW] if enabled and tranche and regime in PENALTY:
                        _w *= REGIME_PENALTY_FACTOR (0.5)
                build signal_data; append to dispatched_signals
        └─> signal_ledger.record_dispatch(dispatched_signals, ...)      [only dispatched names]
```

---

## 9. Failure Modes

| Failure | Guard | Recovery |
|---|---|---|
| `_LATEST_REGIME_BY_TICKER` missing a ticker (pre-regime artefact) | `.get(ticker)` returns None; both guards check `_regime is not None` → no-op | None |
| `CONFIG.trading.regime_sizing_enabled` not in settings.json (old file) | Dataclass default `True` applies | No action needed |
| New TradingConfig field breaks `Config.from_json` with old settings.json | Field has a default — JSON key absent → default applies, no error | None |
| Regime sizing too aggressive (overfits to backtest OOS) | Kill-switch: set `"regime_sizing_enabled": false` in settings.json + restart | Instant no-op |
| PENALTY multiplier applied to half-Kelly path (double-count) | Guard `and tranche_fields` ensures the 0.5× only fires in tranche mode | Verified by TC-3 |

---

## 10. Blast Radius and Rollback

**Blast radius:**
- `main.py` is hub node degree 84 — the change is surgical (3 guard blocks, ~12 new lines inside
  one function). No new function, no signature change, no new module-level state.
- `config/settings.py` and `config/settings.json` gain one field each. The dataclass default
  ensures backward compat for any older settings.json without the key.
- No other files touched.

**Rollback:**
- Fastest: set `"regime_sizing_enabled": false` in `config/settings.json` and restart the bot
  (systemd `systemctl restart quant-v6-bot.service`). Zero code change. Takes effect on next
  daily inference run.
- Full revert: `git revert HEAD` if the commit is clean (only the 3 files above).

---

## 11. Verification Evidence Checklist

- [ ] `pytest --tb=short -q` → all green, count shows 163+ passed (158 existing + 5 new)
- [ ] `grep FEATURE_RECIPE_VERSION src/backtest/pipeline.py src/utils/schema_hash.py` → unchanged
- [ ] Manual smoke: import `main._dispatch_signals` in a Python REPL, populate
  `main._LATEST_REGIME_BY_TICKER = {"HPG": 0}`, call with a minimal tranche strategy dict and
  `broadcast=False` → verify "HPG" absent from returned list
- [ ] Flag-off parity: set `CONFIG.trading.regime_sizing_enabled = False` in the REPL smoke and
  repeat → verify "HPG" IS in returned list
- [ ] `config/settings.json` round-trip: `Config.from_json()` in REPL → `CONFIG.trading.regime_sizing_enabled == True`
- [ ] No new Telegram 4096-char risk: regime-skipped names produce 0 cards (net reduction);
  no new fields added to `signal_data` dict

---

## 12. Acceptance Criteria

1. `pytest` green: 158 existing tests pass unchanged; 5 new dispatch-regime tests pass.
2. NO_TRADE skip: a ticker with regime in {0,7} is absent from `dispatched_signals` and absent
   from the signal ledger when `regime_sizing_enabled=True`.
3. PENALTY halving (tranche mode): a ticker with regime in {1,6} has `suggested_weight` equal to
   `REGIME_PENALTY_FACTOR × (1 / (hold_days × n_picks))` when `regime_sizing_enabled=True`.
4. Flag OFF: with `regime_sizing_enabled=False`, dispatch is byte-for-byte unchanged (NO_TRADE name
   dispatched at full weight; PENALTY name not penalised).
5. Event-override precedence: a ticker with a non-None event override and a NO_TRADE regime is
   still dispatched (override wins).
6. Legacy half-Kelly path (tranche_fields empty): NO double-penalisation; PENALTY 0.5× NOT applied.
7. `FEATURE_RECIPE_VERSION` is unchanged.
8. `config/settings.json` deserialises without error (backward compat, no key required).

---

## 13. Dependencies and Ordering

All dependencies are already satisfied:
- `src/trading/regime_policy.py` — shipped in 77c1412 (NO_TRADE_REGIMES, PENALTY_REGIMES, REGIME_PENALTY_FACTOR defined)
- `_LATEST_REGIME_BY_TICKER` — already populated in `_compute_v3_features` (line 420)
- `CONFIG` singleton — already imported at main.py line 21
- No new packages required

Checklist order is dependency-safe: config changes (steps 1-2) → import addition (step 3) →
dispatch logic (steps 4-5) → tests (step 6) → full verification (steps 7-8).

---

## 14. Resume and Execution Handoff

**Plan file:** `process/general-plans/active/serve-regime-sizing_PLAN_14-06-26.md`

**Pass to vc-execute-agent exactly:**
> "Implement the serve-path regime-conditional sizing per plan at
> `process/general-plans/active/serve-regime-sizing_PLAN_14-06-26.md`.
> Follow the Implementation Checklist (Section 6) step by step.
> Do not touch `src/backtest/walk_forward.py`, `train_models.py`, `FEATURE_RECIPE_VERSION`,
> or any model artefacts. Do not modify any file outside the 4 specified files."

**Files to modify (exhaustive list):**
1. `config/settings.py` — add `regime_sizing_enabled: bool = True` to `TradingConfig`
2. `config/settings.json` — add `"regime_sizing_enabled": true` to `"trading"` block
3. `main.py` — add import + NO_TRADE skip + PENALTY multiply inside `_dispatch_signals`
4. `tests/test_dispatch_regime_sizing.py` — CREATE (new file, 5 test cases)

**No other files.**

**Session resume pointer:** if the session is interrupted, resume at the first unchecked step in
Section 6. Steps 1-2 are idempotent. Steps 3-5 are idempotent if the insertions have not been made.
Step 6 (test file) can be written fresh from Section 7 specification. Step 7 is the gate — do not
mark DONE until pytest is green.
