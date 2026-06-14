# Regime-Conditional Tranche Sizing — Implementation Plan

**Classification:** COMPLEX (multi-file, parity enforcement, A/B eval harness)
**Date:** 13-06-26
**Author:** vc-plan-agent
**Status:** SHIPPED 2026-06-14 (commit 77c1412). A/B GOLDEN: MaxDD −23.32%→−16.88%, Sharpe +0.734→+0.876, Net +46.21%→+42.17%, DSR 0.345→0.447 (still <0.95). Default-OFF `--regime-sizing` flag. Serve parity follow-up: serve-regime-sizing_PLAN_14-06-26 (SHIPPED).

---

## 1. Overview and Goal

The backtested tranche engine currently applies only a scalar `p_bull` multiplier for macro-regime
risk. The serve path (`src/bot/sizing.py`) already applies finer per-ticker regime logic:

- **NO_TRADE_REGIMES `{0, 7}`** — freeze or liquidity sweep, stand aside completely.
- **PENALTY_REGIMES `{1, 6}`** — squeeze or choppy, shrink allocation ceiling to `REGIME_PENALTY_CAP = 0.10`.

This gap means the backtest does NOT validate the exact live behaviour (train/serve parity
violation). The observed MaxDD ≈ −27.4% defeats DSR (p≈0.31 vs target 0.95) and the root cause
is portfolio-level DD, not lack of per-stock edge. Per-name PT/SL barriers are FALSIFIED by
research and remain OFF.

**Goal:** Mirror the serve-path regime constants into the tranche engine so backtest accurately
models live sizing. Extract shared constants into a single canonical module so the two consumers
can never drift. Gate all new behaviour behind a config flag (default OFF) for safe A/B comparison.

---

## 2. Scope

### In Scope

- New shared module `src/trading/regime_policy.py` with `NO_TRADE_REGIMES`, `PENALTY_REGIMES`,
  `REGIME_PENALTY_CAP`, `STRONG_TREND_REGIME`.
- `src/bot/sizing.py` — re-import those constants from the shared module (zero behaviour change).
- `src/backtest/walk_forward.py` — add `use_regime_sizing: bool = False` to `WalkForwardConfig`;
  inject `market_regime` into `_day_index`; apply regime logic inside `_tranche_day`.
- `run_backtest.py` — add `--regime-sizing` CLI flag; wire to `_build_wf_config`; add
  `use_regime_sizing` to `eval_fields` so it is tunable without retrain.
- New test file `tests/test_regime_tranche_sizing.py` + a parity-constants test.
- Existing test `tests/test_walk_forward_tranche.py` — add a `no_trade_regime_excludes_name` test.
- Existing test `tests/test_run_backtest_config.py` — add a `regime_sizing_propagates` test.

### Explicitly Out of Scope

- **NO serve-path behaviour change.** `sizing.py`'s public functions and outputs are unchanged;
  we only move four constants to a shared module and re-import them.
- **NO feature-recipe bump / NO retrain.** Regime logic is a sizing/eval knob; it does NOT
  touch `FEATURE_SCHEMA`, `FEATURE_RECIPE_VERSION`, `frac_diff_d`, or any model weight.
  The `market_regime` column is already in the feature pool (pipeline.py line 78) and in the
  panel passed to the engine. No new features are added.
- **NO vol-scaling layer** (e.g. per-tranche σ target).
- **NO gross-exposure cap** (e.g. circuit-breaker on total invested NAV).
- **NO grid-mode changes.** Grid path is structurally unfit and untouched.
- VN price-scale is irrelevant here — `market_regime` is a categorical integer; no conversion needed.

---

## 3. Critical Architecture Decision — Resolved Open Question

### market_regime: per-ticker, not market-wide

`build_regime_features` (src/features/market_regime.py line 98) groups ALL rolling windows
`.over("ticker")` — every indicator is computed per-ticker over that ticker's own OHLCV history.
A "Freeze" for ticker AAA does NOT imply "Freeze" for ticker BBB on the same date. Each ticker
independently enters regime 0–7 based on its own ATR, Bollinger bandwidth, efficiency ratio, RSI,
etc.

**Consequence:** The correct injection mechanism is the `_day_index` dict
`{date → {ticker → row_dict}}`, NOT the per-day `_p_bull` scalar approach. We add
`market_regime` as a field in each ticker's row dict during `_prepare`, then look it up per-name
in `_tranche_day`.

---

## 4. Shared Constants Module — Parity Design

### 4.1 New file: `src/trading/regime_policy.py`

Contains only the four constants currently defined inline in `src/bot/sizing.py`:

```
NO_TRADE_REGIMES: frozenset[int] = frozenset({0, 7})
PENALTY_REGIMES:  frozenset[int] = frozenset({1, 6})
REGIME_PENALTY_CAP: float = 0.10
STRONG_TREND_REGIME: int = 3
# Scale-invariant penalty multiplier. In serve (sizing.py) a PENALTY regime
# shrinks the per-name NAV ceiling from DEFAULT_NAV_CAP (0.20) to
# REGIME_PENALTY_CAP (0.10) — a 0.5x reduction of the effective size. The
# tranche engine's per_name notional (~nav/(H*picks) ≈ 0.7% NAV) is far below
# the 0.10 absolute cap, so an absolute cap is inert there. To mirror the serve
# INTENT regardless of sizing base, the tranche engine multiplies per_name by
# this ratio. Hardcoded 0.5 with an anti-drift test asserting
# REGIME_PENALTY_FACTOR == REGIME_PENALTY_CAP / sizing.DEFAULT_NAV_CAP.
REGIME_PENALTY_FACTOR: float = 0.5
```

Also exports `__all__` with those five names.

No functions, no imports beyond `__future__`. Pure data module. `DEFAULT_NAV_CAP`
stays in `sizing.py` (it is a Kelly default, not a regime constant); the
anti-drift test cross-checks the ratio so the two can never silently diverge.

### 4.2 `src/bot/sizing.py` changes

Remove the four constant definitions (lines 30–33). Replace with:

```python
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_CAP,
    STRONG_TREND_REGIME,
)
```

`__all__` in `sizing.py` already exports these names — no change needed there. Public behaviour
of every function is byte-for-byte identical.

---

## 5. Regime Sizing Semantics in the Tranche Engine

The tranche engine uses equal-weight notional allocation (not NAV-weight Kelly). The mapping from
`suggested_weight` semantics to notional semantics:

| Regime class | `sizing.py` action | Tranche engine action |
|---|---|---|
| NO_TRADE `{0, 7}` | return 0.0 (zero weight) | exclude ticker from `picks` before allocation |
| PENALTY `{1, 6}` | shrink cap `0.20 → 0.10` (0.5× of the ceiling) | multiply ticker's `per_name` by `REGIME_PENALTY_FACTOR` (0.5×); freed half stays cash |
| STRONG_TREND `{3}` | full half-Kelly up to cap | no additional action (already gets full `per_name`) |
| NEUTRAL `{2, 4, 5}` or `None` | unmodified half-Kelly to cap | no additional action |

> **Why a multiplier, not an absolute cap.** In serve, sizing is half-Kelly that
> can reach 12.5–16.25% NAV for the calibrated band p∈[0.50,0.55], so a 0.10
> absolute cap binds and cuts size. In the tranche engine `per_name ≈ nav/(H·picks)
> ≈ 0.7% NAV` — already ~15× below 0.10·nav — so `min(per_name, 0.10·nav)` would
> NEVER bind and the penalty would be a silent no-op. Mirroring the serve *intent*
> (penalty halves the effective ceiling) as a 0.5× multiplier on `per_name` makes
> the penalty meaningful and scale-invariant.

**Decision on freed capital (NO_TRADE exclusion):** When a name is excluded from `picks` due to a
NO_TRADE regime, the remaining names receive their normal equal share of the budget. The excluded
name's share stays as cash — this is the DD-reducing behaviour (smaller tranche deployed on bad
days). We do NOT redistribute the excluded budget among the survivors.

**Rationale:** Redistributing would inflate the per-surviving-name notional on regime-bad days,
concentrating the portfolio at the worst time. Keeping freed capital as cash is the conservative
and DD-reducing path.

**Interaction with `p_bull` multiplier and cash guard:**

Execution order inside `_tranche_day` step 3 (buy side):

```
1. Run inference + liquidity filter → candidate picks (unchanged)
2. Rank by p_up, apply signal_threshold, apply max_positions → raw_picks list
3. [NEW] If use_regime_sizing: remove any name from raw_picks whose market_regime ∈ NO_TRADE_REGIMES
   Result: picks (may be shorter than raw_picks)
4. If picks is empty → return (no tranche)
5. p_bull = self._p_bull.get(D, 1.0)  [unchanged]
6. nav = self._compute_nav(D)  [unchanged]
7. budget = (nav / cfg.tranche_hold_days) * p_bull  [unchanged — p_bull first]
8. budget = min(budget, max(self.cash, 0.0) / (1.0 + cfg.fee_buffer))  [unchanged]
9. If budget <= 0 → return  [unchanged]
10. per_name = budget / len(picks)  [uses filtered picks length]
11. [NEW] For each pick: if use_regime_sizing and market_regime ∈ PENALTY_REGIMES:
        effective_notional = per_name * REGIME_PENALTY_FACTOR   # 0.5×
    Else:
        effective_notional = per_name
12. qty = round_down_to_lot(int(effective_notional / row["close"]), 100)  [unchanged otherwise]
```

**Important:** The penalty is a `REGIME_PENALTY_FACTOR` (0.5×) multiplier on the individual
ticker's `per_name` notional, NOT an absolute `REGIME_PENALTY_CAP × nav` cap (which would never
bind at tranche scale — see the "Why a multiplier" note above). The freed half of the notional
stays as cash — same conservative, DD-reducing logic as NO_TRADE (no redistribution to survivors).

**`market_regime` lookup:** `day.get(tk, {}).get("market_regime")` from `_day_index`. If the
column is absent or None (panel built without the regime column), the lookup returns `None` and no
regime adjustment is applied — this is the safe fallback for the flag-OFF path.

---

## 6. Config Flag, CLI, and eval_fields Wiring

### 6.1 `WalkForwardConfig` (`src/backtest/walk_forward.py`)

Add one field after the existing tranche barrier fields (~line 157):

```python
# ── REGIME-CONDITIONAL SIZING ─────────────────────────────────────────────
# When True, per-name allocations inside each tranche are modulated by the
# ticker's `market_regime` column (0–7, built by build_regime_features).
# NO_TRADE regimes {0,7} exclude the ticker from that day's cohort;
# PENALTY regimes {1,6} cap its notional at REGIME_PENALTY_CAP × NAV.
# Default False → original equal-weight behaviour, no allocation change.
use_regime_sizing: bool = False
```

### 6.2 `_build_wf_config` (`run_backtest.py`)

Add parameter `use_regime_sizing: bool = False` to signature. Pass to `WalkForwardConfig(...)`.

### 6.3 `run_oos` (`run_backtest.py`)

Add parameter `use_regime_sizing: bool = False` to signature. Pass through to `_build_wf_config`.

### 6.4 `main` (`run_backtest.py`)

Add parameter `use_regime_sizing: bool = False`. Pass through to `run_oos` at each call site.

### 6.5 `_apply_eval_overrides` (`run_backtest.py`) — NO CHANGE

**Do NOT add `use_regime_sizing` to `eval_fields`/`overrides`.** That mechanism only accepts
**`RunConfig`** attribute keys — `_apply_eval_overrides` does `getattr(RunConfig(), key)` and raises
`ValueError` for any key not in `eval_fields`. `use_regime_sizing` is a **`WalkForwardConfig`-only**
field, and section 2 forbids adding it to `RunConfig` (no retrain). Routing it through
`overrides` would crash every run. Instead it is threaded as an explicit keyword param —
**identically to the existing `WalkForwardConfig`-only knobs `pt_sigma`/`sl_sigma`** (verified:
`_cli` return tuple → `main(...)` → `run_oos(...)` → `_build_wf_config(...)`, none of which are in
`eval_fields`). This matches section 9 Public Contracts. No retrain needed — it is a CLI flag, not
a swept `RunConfig` value.

### 6.6 `_cli` + `__main__` (`run_backtest.py`)

Add argument:

```python
p.add_argument("--regime-sizing", action="store_true", default=False,
               help="enable per-ticker regime-conditional sizing in the tranche engine "
                    "(mirrors src/bot/sizing.py NO_TRADE/PENALTY logic; default off)")
```

Extend the `_cli` **return tuple** with `a.regime_sizing` (mirror how `pt_sigma`/`sl_sigma` are
returned). In the `if __name__ == "__main__":` block, unpack it and pass
`use_regime_sizing=<value>` into `main(...)`. Do NOT add it to any `overrides` dict.

---

## 7. `_prepare` — Injecting `market_regime` into `_day_index`

In `_prepare` (`src/backtest/walk_forward.py` ~lines 387–396), within the loop that builds
`_day_index`, add `market_regime` to each ticker's row dict **if the column is present in the
panel**:

```python
# Inside the loop: for tk, g in self.ticker_frames.items(): for row in g.itertuples():
row_dict = {
    "open": row.open, "high": row.high, "low": row.low,
    "close": row.close, "volume": row.volume,
    "ref_price": row.ref_price, "vol": row.vol,
    "adv20": row.adv20,
    "exchange": row.exchange,
}
# NEW: include market_regime when present (per-ticker per-day)
if hasattr(row, "market_regime"):
    row_dict["market_regime"] = int(row.market_regime)
self._day_index.setdefault(row.date, {})[tk] = row_dict
```

**Why conditional (`hasattr`):** The column is only present when the engine receives a panel
built with `build_features`. The existing test fixtures supply minimal panels without it. Using
`hasattr` means all existing tests continue to pass unchanged — the flag-OFF code path at step 11
above safely falls through to `effective_notional = per_name` when the dict key is missing.

---

## 8. Touchpoints (Complete File:Line Reference)

| File | Change | Approx. Lines |
|---|---|---|
| `src/trading/regime_policy.py` | NEW file — 5 constants (incl. `REGIME_PENALTY_FACTOR`) + `__all__` | ~24 lines total |
| `src/trading/__init__.py` | confirm exists or create empty; no content required | 0–2 lines |
| `src/bot/sizing.py` | remove 4 inline constants; add import from `regime_policy` | lines 30–33 → import |
| `src/backtest/walk_forward.py` | Add `use_regime_sizing: bool = False` to `WalkForwardConfig` | ~line 157 |
| `src/backtest/walk_forward.py` | Extend `_prepare` to include `market_regime` in `_day_index` row dicts | ~lines 388–396 |
| `src/backtest/walk_forward.py` | Add NO_TRADE pick filter + 0.5× PENALTY multiplier in `_tranche_day` step 3 | ~lines 769–788 |
| `run_backtest.py` | Add `use_regime_sizing` param to `_build_wf_config` + pass to `WalkForwardConfig` | ~lines 77–104 |
| `run_backtest.py` | Add `use_regime_sizing` param to `run_oos` + pass through | ~lines 107–143 |
| `run_backtest.py` | Add `use_regime_sizing` param to `main` + pass through to each `run_oos` call | ~lines 289–297, 382–386 |
| `run_backtest.py` | Add `--regime-sizing` CLI arg; extend `_cli` return tuple; unpack in `__main__` and pass `use_regime_sizing=` into `main()` (mirror `pt_sigma`/`sl_sigma`; NOT via `eval_fields`/`overrides`) | ~lines 781–798 |
| `tests/test_regime_tranche_sizing.py` | NEW — regime sizing unit + integration tests | ~120 lines |
| `tests/test_walk_forward_tranche.py` | Add `test_no_trade_regime_excludes_from_picks` | ~20 lines |
| `tests/test_run_backtest_config.py` | Add `test_regime_sizing_propagates_to_wf_config` | ~10 lines |

---

## 9. Public Contracts

| Contract | Status after this plan |
|---|---|
| `WalkForwardConfig` fields | `use_regime_sizing: bool = False` added; all existing fields unchanged |
| `WalkForwardEngine.run()` signature | unchanged |
| `_build_wf_config(...)` signature | `use_regime_sizing: bool = False` added as keyword param |
| `run_oos(...)` signature | `use_regime_sizing: bool = False` added as keyword param |
| `main(...)` signature | `use_regime_sizing: bool = False` added as keyword param |
| `src/bot/sizing.py` public API | ALL unchanged — same functions, same return types, same constants re-exported from same names |
| `src/trading/regime_policy.py` | NEW — exports `NO_TRADE_REGIMES`, `PENALTY_REGIMES`, `REGIME_PENALTY_CAP`, `STRONG_TREND_REGIME`, `REGIME_PENALTY_FACTOR` |
| `FEATURE_RECIPE_VERSION` | UNCHANGED — no feature engineering change |
| `_day_index` row dict schema | backward-compatible extension: `market_regime` key added only when column present |

---

## 10. Blast Radius

### Hub node adjacency

`build_regime_features` (market_regime.py) has degree 142 — the highest hub in the codebase.
**We only READ its output** (the `market_regime` column already in the panel). We do NOT call
`build_regime_features`, do NOT modify it, do NOT add parameters to it. Zero blast radius on
that hub.

`daily_inference` (main.py) has degree 84 — the serve path. We DO NOT modify `main.py` or
`daily_inference`. Zero blast radius.

`TabularEnsemble.fit` (degree 75) — untouched. Zero blast radius.

`triple_barrier_pipeline` (degree 82) — untouched. Zero blast radius.

### Affected paths

| Path | Impact |
|---|---|
| Tranche mode backtest (flag ON) | Modified behaviour — tranche budget filtering and per-name caps |
| Tranche mode backtest (flag OFF, default) | Zero change — all new branches are `if use_regime_sizing:` gated |
| Grid mode backtest | Zero change — `_tranche_day` not called |
| Serve path / bot | Zero change — `sizing.py` re-exports unchanged constants |
| Model training / feature pipeline | Zero change |
| Tests (existing 210) | All must pass unchanged — covered by flag-OFF default + `hasattr` guard |

---

## 11. A/B Evaluation Harness

### Commands

**Baseline (flag OFF — current behaviour):**

```bash
python run_backtest.py --mode tranche --hold-days 30 --no-save 2>&1 | tee logs/backtest_baseline.log
```

**Treatment (flag ON — regime-conditional sizing):**

```bash
python run_backtest.py --mode tranche --hold-days 30 --regime-sizing --no-save 2>&1 | tee logs/backtest_regime.log
```

Both runs use the identical frozen checkpoint and feature set. Use `--no-save` to avoid
overwriting the live bot payload during experimentation.

### Metrics Table to Fill

| Metric | Baseline (OFF) | Treatment (ON) | Delta | Pass? |
|---|---|---|---|---|
| MaxDD | −27.4% (known) | ? | ? | Target: materially lower (< −20%) |
| Net Sharpe (annualized) | ? | ? | ? | Must not gut (< −0.10 delta acceptable) |
| Net PnL (VND) | ? | ? | ? | Must stay positive |
| Total Return | ? | ? | ? | Directional sanity |
| DSR p-value | ~0.31 (known) | ? | ? | Target: > 0.50 (ideally approach 0.95) |
| Avg gross exposure | ? | ? | ? | Expect reduction on NO_TRADE days |
| # Trades (buys) | ? | ? | ? | Expect modest reduction |
| UP-precision (unchanged) | ? | ? | ? | Must be identical (signal unchanged) |

### Success Criteria

- **Primary:** MaxDD materially lower than −27.4% baseline (target < −20%).
- **Secondary:** Net Sharpe not gutted (delta > −0.10 across seeds).
- **DSR improvement:** p-value meaningfully higher than 0.31 baseline.
- **Parity:** UP-precision identical between runs (regime logic does not touch the oracle).
- **Smoke:** All 210+ tests green.

### Kill Criteria (do not ship to serve path)

- MaxDD is NOT improved (within 1% of baseline after rounding).
- Net PnL turns negative.
- Net Sharpe drops by more than 0.20 (regime filtering too aggressive).
- Any existing test fails.

---

## 12. Test Plan

### 12.1 New file: `tests/test_regime_tranche_sizing.py`

**Setup:** Reuse/extend the `_panel()` and `_oracle()` helpers from `test_walk_forward_tranche.py`.
Add `market_regime` column to synthetic panels for targeted tests.

| Test | Assertion |
|---|---|
| `test_no_trade_regime_ticker_not_bought` | Ticker with `market_regime=0` on a given day is NOT in that day's fills (buy side). |
| `test_no_trade_regime_does_not_affect_others` | Remaining eligible tickers ARE bought; budget splits among them. |
| `test_penalty_regime_halves_notional` | Ticker with `market_regime=1` on a given day: buy notional ≈ `0.5 × per_name` (i.e. half what the same ticker gets in a NEUTRAL regime, within one lot rounding). Assert it is materially below `per_name`, NOT merely below the inert `0.10 × nav` cap. |
| `test_penalty_factor_matches_serve_ratio` | Anti-drift: `REGIME_PENALTY_FACTOR == REGIME_PENALTY_CAP / sizing.DEFAULT_NAV_CAP` (0.10/0.20 = 0.5). Catches drift if the serve NAV cap ever changes. |
| `test_normal_regime_unaffected` | Ticker with `market_regime=3` (Strong Trend) gets the same `per_name` as with flag OFF. |
| `test_flag_off_no_regime_effect` | With `use_regime_sizing=False` (default), even a ticker explicitly set to `market_regime=0` is still bought. |
| `test_flag_off_panel_without_regime_column` | Panel with no `market_regime` column runs without error in flag-OFF mode. |
| `test_no_trade_freed_capital_stays_cash` | NAV of a flag-ON run with a NO_TRADE day is >= NAV of same run if NO_TRADE ticker were force-included (i.e., cash is conserved, not redistributed). |
| `test_parity_constants_identical` | Import `NO_TRADE_REGIMES`/`PENALTY_REGIMES`/`REGIME_PENALTY_CAP` from both `src.trading.regime_policy` and `src.bot.sizing`; assert identical values. This is the anti-drift test. |

### 12.2 Addition to `tests/test_walk_forward_tranche.py`

| Test | Assertion |
|---|---|
| `test_no_trade_regime_excludes_from_picks` | Build a panel where AAA has `market_regime=0` on day 2. Run with `use_regime_sizing=True`. Assert AAA is not bought on day 2 but IS bought on other days (its regime is 0 only on day 2). |

### 12.3 Addition to `tests/test_run_backtest_config.py`

| Test | Assertion |
|---|---|
| `test_regime_sizing_propagates_to_wf_config` | `_build_wf_config(..., use_regime_sizing=True)` → `wf.use_regime_sizing is True`. |
| `test_regime_sizing_default_off` | `_build_wf_config(FEATURES, CUTOFF, _cfg())` → `wf.use_regime_sizing is False`. |

### 12.4 Addition to `tests/test_sizing.py`

| Test | Assertion |
|---|---|
| `test_sizing_constants_from_shared_module` | Import the four constants from `src.trading.regime_policy`; assert they match the values asserted by existing `test_config_locked` (cross-check, not duplicate). |

### 12.5 Existing test preservation

All 210 existing tests must pass unchanged. The `hasattr` guard in `_prepare` and the
`if use_regime_sizing:` gate in `_tranche_day` guarantee that the flag-OFF path is byte-for-byte
identical to the current code.

---

## 13. Implementation Checklist

1. **[SETUP]** Verify `src/trading/__init__.py` exists (or is an empty placeholder). If not,
   confirm `src/trading/` directory exists and create the init file.

2. **[NEW MODULE]** Create `src/trading/regime_policy.py`. Define the five constants
   (`NO_TRADE_REGIMES`, `PENALTY_REGIMES`, `REGIME_PENALTY_CAP`, `STRONG_TREND_REGIME`,
   `REGIME_PENALTY_FACTOR = 0.5`) and `__all__`. Add module docstring referencing the
   serve-path consumer (`sizing.py`) and backtest consumer (`walk_forward.py`), and noting that
   `REGIME_PENALTY_FACTOR = REGIME_PENALTY_CAP / sizing.DEFAULT_NAV_CAP` (enforced by test).

3. **[SIZING.PY]** In `src/bot/sizing.py` lines 30–33: remove the four inline constant
   definitions; replace with `from src.trading.regime_policy import (NO_TRADE_REGIMES,
   PENALTY_REGIMES, REGIME_PENALTY_CAP, STRONG_TREND_REGIME)`. Verify `__all__` still exports
   all four names (it already does via the existing list).

4. **[WALK_FORWARD CONFIG]** In `src/backtest/walk_forward.py`: add
   `use_regime_sizing: bool = False` field to `WalkForwardConfig` dataclass after
   `tranche_sl_sigma` (~line 157). Add import `from src.trading.regime_policy import
   NO_TRADE_REGIMES, PENALTY_REGIMES, REGIME_PENALTY_FACTOR` at the top of the file
   (the engine uses the 0.5× factor, not the absolute cap).

5. **[PREPARE]** In `src/backtest/walk_forward.py` `_prepare` method (~lines 388–396): extend
   the `_day_index` row dict construction to include `"market_regime": int(row.market_regime)`
   when `hasattr(row, "market_regime")`, else omit the key.

6. **[TRANCHE_DAY]** In `src/backtest/walk_forward.py` `_tranche_day` method:
   - After forming `picks` list (~line 771): add NO_TRADE filter guarded by
     `if cfg.use_regime_sizing:` — drop any name whose `market_regime ∈ NO_TRADE_REGIMES`.
   - After computing `per_name` (~line 781): inside the per-ticker buy loop, apply the PENALTY
     multiplier guarded by `if cfg.use_regime_sizing:` — `effective_notional = per_name *
     REGIME_PENALTY_FACTOR` when `market_regime ∈ PENALTY_REGIMES`, else `per_name`.
   - The `market_regime` for each ticker is read from `day.get(tk, {}).get("market_regime")`.
   - When the key is absent, default to `None` (no regime adjustment).
   - Import `NO_TRADE_REGIMES`, `PENALTY_REGIMES`, `REGIME_PENALTY_FACTOR` from
     `src.trading.regime_policy` (no `DEFAULT_NAV_CAP` import → no backtest→bot dependency).

7. **[RUN_BACKTEST _BUILD_WF_CONFIG]** Add `use_regime_sizing: bool = False` parameter to
   `_build_wf_config` signature; pass it to `WalkForwardConfig(...)`.

8. **[RUN_BACKTEST RUN_OOS]** Add `use_regime_sizing: bool = False` parameter to `run_oos`;
   pass through to `_build_wf_config(...)`.

9. **[RUN_BACKTEST MAIN]** Add `use_regime_sizing: bool = False` parameter to `main`; pass
   through to each `run_oos(...)` call site.

10. **[RUN_BACKTEST EVAL_FIELDS — NO-OP]** Do NOT touch `eval_fields`/`overrides`. See section
    6.5: those keys must be `RunConfig` attrs; `use_regime_sizing` is `WalkForwardConfig`-only and
    routing it there would crash. It is threaded as a keyword param instead (steps 7–9 + 11),
    mirroring `pt_sigma`/`sl_sigma`.

11. **[RUN_BACKTEST CLI]** In `_cli`: add `--regime-sizing` argparse argument (store_true,
    default False). Extend the `_cli` **return tuple** with `a.regime_sizing` (mirror
    `pt_sigma`/`sl_sigma`). Update the `_cli` return type annotation if present. In the
    `if __name__ == "__main__":` block, unpack the new value and pass `use_regime_sizing=<value>`
    into `main(...)`. Do NOT add it to any `overrides` dict.

12. **[TESTS — NEW FILE]** Create `tests/test_regime_tranche_sizing.py` with all 8 tests from
    section 12.1.

13. **[TESTS — TRANCHE]** Add `test_no_trade_regime_excludes_from_picks` to
    `tests/test_walk_forward_tranche.py`.

14. **[TESTS — CONFIG]** Add `test_regime_sizing_propagates_to_wf_config` and
    `test_regime_sizing_default_off` to `tests/test_run_backtest_config.py`.

15. **[TESTS — SIZING]** Add `test_sizing_constants_from_shared_module` to
    `tests/test_sizing.py`.

16. **[VERIFY RECIPE VERSION]** In `src/backtest/pipeline.py`: confirm
    `FEATURE_RECIPE_VERSION` string is unchanged. No recipe bump needed — document in commit
    message.

17. **[RUN TESTS]** Execute `pytest tests/ -x -q` and confirm 210 + new tests all green.

18. **[A/B BACKTEST]** Run baseline command, then treatment command, record metrics in a log
    file. Compare MaxDD, Sharpe, Net PnL, DSR per section 11. Confirm success criteria met
    before tagging for serve.

---

## 14. Verification Evidence (Checklist)

- [ ] `pytest tests/ -x -q` — all 210 original + N new tests pass (zero regressions).
- [ ] `FEATURE_RECIPE_VERSION` grep confirms string is unchanged from `"v2-sha8:53b5bd85"` (or
      current computed value).
- [ ] `from src.trading.regime_policy import NO_TRADE_REGIMES; from src.bot.sizing import
      NO_TRADE_REGIMES as NTR; assert NO_TRADE_REGIMES == NTR` passes in a Python session.
- [ ] Baseline backtest (flag OFF) produces results identical to pre-plan baseline within
      float tolerance (same engine, same flag-off code path).
- [ ] Treatment backtest (flag ON) shows MaxDD materially improved vs baseline.
- [ ] No import of `build_regime_features` added to `walk_forward.py` (we read the already-built
      column; we do NOT re-run the classifier inside the engine).
- [ ] `git diff src/features/market_regime.py` is empty (hub node untouched).
- [ ] `git diff src/bot/sizing.py` shows ONLY constant removal + import replacement, no logic
      change.

---

## 15. Rollback

**Flag default is OFF.** Before any serve-path integration, existing users and cron runs see
zero change. To roll back:

1. Remove the `--regime-sizing` flag from any cron/systemd invocations.
2. If `sizing.py` import change causes issues (unlikely — pure re-import), revert
   `src/bot/sizing.py` and `src/trading/regime_policy.py` creation in one commit.

No schema migrations. No model artifacts affected. No database changes.

---

## 16. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `market_regime` column absent from panel in some environments | LOW — pipeline always adds it; flag-OFF safe regardless | `hasattr` guard in `_prepare`; `dict.get` in `_tranche_day` |
| `src/trading/__init__.py` missing causing import error | LOW — `src/trading/` already has `vn_cost_model.py` | Step 1 checks for the init file |
| A/B backtest shows no DD improvement (kill criterion) | MEDIUM — if regime flags too rare on OOS dates | Report regime frequency distribution in backtest log |
| Treatment path changes existing test expectations | LOW — all new logic gated behind `use_regime_sizing=True` | Existing tests run flag-OFF path; no assertions change |
| Drift re-introduced if someone edits only one constants source | LOW — test_sizing_constants_from_shared_module catches it | CI enforces the anti-drift test |

---

## 17. Resume and Execution Handoff

**Plan file:** `process/general-plans/active/regime-conditional-tranche-sizing_PLAN_13-06-26.md`

**Execute with:** Pass this exact plan file path to vc-execute-agent.

**Execution order:** Steps 1–11 (code changes) must precede steps 12–15 (tests). Steps 17–18
(verify + A/B) are the final gates.

**Partial-state resume:** If execution is interrupted after step 6 (`_tranche_day` logic) but
before step 11 (CLI), the engine has the behaviour but the CLI cannot enable it. Resume from
step 7. All steps are idempotent.

**Test runner:** `pytest tests/ -x -q` from repo root. **210** tests green before this plan
(per the 13-06 handoff; the all-context "158" figure is stale); expect 210 + ~13 new tests ≈ 223
total. Execute should confirm the actual pre-change count first, then the post-change count.

**Relevant code anchors for EXECUTE:**
- `src/bot/sizing.py` lines 30–33: the four constants to move.
- `src/backtest/walk_forward.py` line ~157: `WalkForwardConfig` insertion point.
- `src/backtest/walk_forward.py` lines ~387–396: `_prepare` `_day_index` build loop.
- `src/backtest/walk_forward.py` lines ~769–788: `_tranche_day` picks + buy loop.
- `run_backtest.py` line ~264: `eval_fields` set in `_apply_eval_overrides`.
- `run_backtest.py` lines ~77–104: `_build_wf_config` signature.
- `run_backtest.py` lines ~786–798: `_cli` overrides dict.

**Confirm before starting EXECUTE:**
- `FEATURE_RECIPE_VERSION` is `"v2-sha8:53b5bd85"` (grep `pipeline.py`).
- `src/trading/vn_cost_model.py` exists (confirm `src/trading/` directory is present).
- Checkpoint at `models/saved/v3_training_checkpoint.joblib` exists for A/B run.
