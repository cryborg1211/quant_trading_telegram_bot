# Phase 3: Hub-Node Test Coverage

**Plan type:** COMPLEX (4 ordered steps, one new test file per step)
**Feature folder:** `process/features/v4-1-structural-debt/`
**Plan file:** `process/features/v4-1-structural-debt/completed/phase3-hub-coverage_PLAN_21-06-26.md`
**Created:** 2026-06-21
**Status:** COMPLETE (2026-06-21) — 95 tests added (orphaned commit `194525f`; tests on disk). See `_GUIDE.md`.

---

## Overview

Phase 1 and Phase 2 of V4.1 Structural Debt are complete. Phase 3 adds direct
unit / characterization tests for four hub nodes that currently have zero direct
test coverage. The goal is NOT to test ML quality — it is to lock wiring,
math correctness, and public contracts so future refactors fail loudly instead
of silently.

**Context fix included in Step 4:** `all-context.md` erroneously lists the
Phase 3 hub node as `run_backtest.main` (degree 97). The real public nodes are
`run_oos` and `_build_wf_config` — there is no standalone `main` function with
that name in `run_backtest.py`'s module-level. Step 4 includes a sub-task to
fix that stale reference.

---

## Goals

1. Add `tests/test_vn_cost_model.py` — pure cost-model math and rejection paths.
2. Add `tests/test_triple_barrier.py` — labeling correctness (PT/SL/vertical hits, no look-ahead leakage).
3. Add `tests/test_tabular_ensemble.py` — stacking wiring and CalibratedClassifierCV meta path (mocked boosters).
4. Add `tests/test_run_backtest_wiring.py` — `_build_wf_config` and `run_oos` engine wiring with a tiny synthetic walk-forward.
5. Fix stale hub-node name in `process/context/all-context.md`.

After Phase 3, the baseline 251-test suite must still be entirely green.

---

## Scope Boundaries

**In scope:**
- New test files only (no changes to production source files except the context fix)
- Testing wiring, math, contracts, and rejection reasons
- Mocking GPU-backed boosters (LightGBM/XGBoost/CatBoost) in Step 3 to avoid CUDA dependency
- Mocking `WalkForwardEngine.run` output in Step 4 to avoid full dataset materialization

**Out of scope:**
- ML model quality / accuracy metrics
- Run-time performance benchmarks
- `build_application` (telegram_bot.py) — hub node but skipped here per program scope
- Any source file change other than the all-context.md stale reference fix

---

## Execution Environment

- **Python runtime:** bare `python -m pytest` (Python 3.11 on PATH with ML stack installed)
- **Shell for verification:** PowerShell only (git-bash is broken on this machine)
- **Subagent role:** EXECUTE agent writes test files. ORCHESTRATOR runs `python -m pytest` and commits.
- **No test execution inside subagent:** subagents have Bash but cannot reliably run pytest in this environment.

---

## Touchpoints

| File (new) | Imports from | Risk |
|---|---|---|
| `tests/test_vn_cost_model.py` | `src.execution.vn_cost_model` | LOW — pure module, no DuckDB, no GPU |
| `tests/test_triple_barrier.py` | `src.labels.triple_barrier` | LOW — pure pandas/numpy/polars, no DuckDB |
| `tests/test_tabular_ensemble.py` | `src.models.tabular_ensemble`, `src.models.stacking_model.purged_kfold` | MEDIUM — needs mock for LGB/XGB/CAT |
| `tests/test_run_backtest_wiring.py` | `run_backtest`, `src.backtest.walk_forward`, `src.execution.vn_cost_model` | MEDIUM — heaviest imports, mock engine.run |
| `process/context/all-context.md` | (context update) | LOW — doc-only change |

---

## Public Contracts

These contracts are being pinned by the new tests:

- `VNCostModel.simulate(order)` returns a `Fill` with `is_filled=True/False` and exact `brokerage_paid`, `tax_paid`, `vat_paid` values consistent with the fee schedule fractions. Prices in ABSOLUTE VND (not thousands-VND).
- `triple_barrier_pipeline(panel_df, cfg)` returns a Polars DataFrame with columns `ticker, t0, t1, trgt, ret, bin, num_co_events, uniqueness, w`. `bin` values are in `{0, 1, 2}`. No look-ahead: `t1 >= t0` always holds.
- `TabularEnsemble.fit(X, y, start_times, end_times, sample_weight=w)` calls `_manual_oof` for each available base learner, assembles a 9-column OOF meta matrix, fits a `LogisticRegression` meta (optionally wrapped in `CalibratedClassifierCV`), and then refits each base on the full dataset. After fit, `predict_proba(X)` returns shape `(n,)` with values in `[0, 1]`.
- `_build_wf_config(tabular_features, cutoff, cfg, mode, hold_days)` returns a `WalkForwardConfig` with the passed `mode`, `hold_days`, `start_trading_date=cutoff`, and a fresh `ExecutionConfig()`.
- `run_oos(panel, tabular_features, ensemble, corporate_actions, cutoff, cfg)` builds a `WalkForwardEngine`, calls `eng.run`, and returns an equity-curve DataFrame trimmed to `date >= cutoff`.

---

## Blast Radius

Changes are additive (new test files only). No production source file is touched
except the context doc fix in Step 4. The new tests run in the same pytest
session as the existing 251 tests. Any import-time failure in the new test files
would surface as a collection error, not a silent pass — which is the desired safety behavior.

---

## Step Ordering Rationale

| Step | Target | Why this order |
|---|---|---|
| 1 | `VNCostModel.simulate` | Pure + deterministic; zero external deps; easiest to isolate; highest ROI for test lines vs complexity |
| 2 | `triple_barrier_pipeline` | Pure pandas/numpy; no ML framework; second easiest; correctness of labeling is foundational |
| 3 | `TabularEnsemble.fit` | Needs mocking for 3 GPU boosters; more setup but structure is clear |
| 4 | `run_oos` + `_build_wf_config` | Heaviest imports; integration-ish; last, builds on all prior context |

Each step ends with an orchestrator-run pytest verification gate before the next step begins.

---

## Implementation Checklist

### Step 1 — `tests/test_vn_cost_model.py`

**Target:** `src/execution/vn_cost_model.py` — `VNCostModel.simulate`, `VNCostModel.simulate_batch`

**Context note:** Parquet prices are in thousands of VND. Tests must use ABSOLUTE VND prices (e.g., `13_450.0`, not `13.45`) — the `_prepare` scaling in `WalkForwardEngine` converts at ingest time; `VNCostModel` itself never sees raw parquet values. Test both the correct (absolute VND) path and document what happens at thousand-scale (PRICE_AT_CEILING_BUY rejection) for contract clarity.

**New file:** `tests/test_vn_cost_model.py`

**No mocks required** — the module is pure Python with no DuckDB/ML deps.

**Imports to use:**
```
from src.execution.vn_cost_model import (
    VNCostModel, ExecutionConfig, Order, OrderSide, Exchange,
    RejectionReason, FeeSchedule, SlippageModel, ParticipationPolicy,
    Fill, tick_size_vnd, round_to_tick, price_band_bounds,
)
from datetime import date, datetime, time as dtime
import math
```

**Test classes and function names (exact):**

- [ ] `1.1` Create `tests/test_vn_cost_model.py`

- [ ] `1.2` **Class `TestFeeScheduleMath`**
  - `test_buy_fee_pct_formula` — assert `FeeSchedule().buy_fee_pct() == pytest.approx(0.0015 * 1.10)`
  - `test_sell_fee_pct_formula` — assert `FeeSchedule().sell_fee_pct() == pytest.approx(0.0015 * 1.10 + 0.0010)`
  - `test_round_trip_pct_equals_buy_plus_sell` — assert round_trip == buy + sell
  - `test_custom_fees_propagate` — custom brokerage=0.002 propagates into buy_fee_pct

- [ ] `1.3` **Class `TestTickSizeVnd`** (pure function `tick_size_vnd`)
  - `test_hose_tier1_below_10k` — price 9_999.0 → tick 10
  - `test_hose_tier1_boundary_10k` — price 10_000.0 → tick 50
  - `test_hose_tier2_below_50k` — price 49_999.0 → tick 50
  - `test_hose_tier2_boundary_50k` — price 50_000.0 → tick 100
  - `test_hose_tier3_above_50k` — price 70_000.0 → tick 100
  - `test_hnx_flat_tick` — any price, HNX → 100
  - `test_upcom_flat_tick` — any price, UPCOM → 100

- [ ] `1.4` **Class `TestPriceBandBounds`** (pure function `price_band_bounds`)
  - `test_hose_band_7pct` — ref 20_000 → floor 18_600, ceiling 21_400 (approx)
  - `test_hnx_band_10pct` — ref 20_000 → floor 18_000, ceiling 22_000 (approx)
  - `test_upcom_band_15pct` — ref 20_000 → floor 17_000, ceiling 23_000 (approx)

- [ ] `1.5` **Helper function `_make_order`** (module-level fixture factory)
  Signature: `_make_order(price=20_000.0, ref=20_000.0, qty=1000, side=OrderSide.BUY, exchange=Exchange.HOSE, vol=1_000_000, volatility=0.02, is_atc=False, atc_volume=None, ticker="VCB", timestamp=None) -> Order`

- [ ] `1.6` **Class `TestVNCostModelSimulateHappyPath`**
  - `test_buy_fill_absolute_vnd_fees` — buy 1000 shares @ 20_000 HOSE, assert `fill.is_filled`, `fill.tax_paid == 0.0` (no tax on buy), `fill.brokerage_paid == pytest.approx(20_000 * 1000 * 0.0015)`, `fill.vat_paid == pytest.approx(fill.brokerage_paid * 0.10)`
  - `test_sell_fill_absolute_vnd_tax` — sell 1000 shares @ 20_000 HOSE, assert `fill.tax_paid == pytest.approx(20_000 * 1000 * 0.0010)` (sell transfer tax)
  - `test_lot_rounding_down` — qty=150, lot=100 → `fill.filled_quantity == 100` (rounds down to 100)
  - `test_gross_notional_equals_price_times_qty` — assert `fill.gross_notional == pytest.approx(fill.filled_price * fill.filled_quantity)`
  - `test_atc_fill_no_slippage` — `is_atc=True`, assert `fill.slippage_cost == 0.0`, `fill.filled_price == pytest.approx(target_price)`
  - `test_simulate_batch_is_loop_over_simulate` — two orders, assert `simulate_batch(orders) == [simulate(o) for o in orders]` (element-by-element `is_filled` and `filled_price`)

- [ ] `1.7` **Class `TestVNCostModelRejectPaths`**
  - `test_reject_nan_price` — `target_price=float('nan')` → `fill.is_filled == False`, `fill.rejection_reason == RejectionReason.INVALID_INPUT`
  - `test_reject_zero_volume` — `daily_volume=0` → `rejection_reason == RejectionReason.ZERO_VOLUME`
  - `test_reject_below_lot_size` — `qty=50` → `rejection_reason == RejectionReason.BELOW_LOT_SIZE`
  - `test_reject_price_outside_band` — `target_price = ref * 1.10` on HOSE (7% band) → `rejection_reason == RejectionReason.PRICE_OUTSIDE_BAND`
  - `test_reject_buy_at_ceiling` — `target_price == ceiling` on HOSE → `rejection_reason == RejectionReason.PRICE_AT_CEILING_BUY`
  - `test_reject_sell_at_floor` — `target_price == floor`, side=SELL → `rejection_reason == RejectionReason.PRICE_AT_FLOOR_SELL`
  - `test_reject_participation_exceed_policy_reject` — qty=200_000, adv=1_000_000 (20% > 10% max), policy=REJECT → `rejection_reason == RejectionReason.PARTICIPATION_REJECT`

- [ ] `1.8` **Class `TestVnPriceScaleConvention`** (documents the critical contract)
  - `test_absolute_vnd_fills_correctly` — price 13_450.0 (absolute) → `fill.is_filled == True`
  - `test_thousand_scale_rejected_as_ceiling_buy` — price 13.45 ("thousand-VND" passed raw) → `fill.is_filled == False`, `fill.rejection_reason == RejectionReason.PRICE_AT_CEILING_BUY` (documents the bug that `_prepare` prevents)

- [ ] `1.9` **Class `TestTickRoundingOnFill`**
  - `test_buy_tick_rounds_up_on_hose` — BUY impact produces non-grid price; assert `fill.filled_price % tick_size_vnd(fill.filled_price, Exchange.HOSE) == 0` and price is on the correct grid
  - `test_sell_tick_rounds_down_on_hose` — SELL; assert filled_price is on grid and rounded down (conservative)
  - `test_no_tick_rounding_when_disabled` — `ExecutionConfig(enforce_tick=False)` → filled price not necessarily on grid

**Verification gate for Step 1:**
```powershell
python -m pytest tests/test_vn_cost_model.py -v
```
Expected: all new tests pass, existing 251 unchanged.

---

### Step 2 — `tests/test_triple_barrier.py`

**Target:** `src/labels/triple_barrier.py` — `triple_barrier_pipeline` (plus building-block functions it calls)

**No mocks required** — pure pandas/numpy/polars, no ML framework, no DuckDB.

**New file:** `tests/test_triple_barrier.py`

**Imports to use:**
```
from src.labels.triple_barrier import (
    TripleBarrierConfig, triple_barrier_pipeline,
    get_daily_vol, apply_pt_sl_on_t1, get_vertical_barriers,
    get_events, get_bins, get_num_co_events, get_sample_tw,
    get_sample_weights,
)
import pandas as pd
import numpy as np
import polars as pl
import pytest
from datetime import date
```

**Helper factories (module-level, not pytest fixtures):**

- `_close_series(n=60, start='2023-01-02', values=None) -> pd.Series` — business-day-indexed close prices. Default: flat 20_000 with small random perturbation seeded deterministically (np.random.default_rng(42)). `values` overrides the array.
- `_panel_df(tickers=['AAA'], n=60, close=20_000.0) -> pl.DataFrame` — synthetic Polars panel with `ticker, date, open, high, low, close, volume` columns. `high = close * 1.02`, `low = close * 0.98`. Dates are business days starting 2023-01-02.

**Test classes and function names (exact):**

- [ ] `2.1` Create `tests/test_triple_barrier.py`

- [ ] `2.2` **Class `TestGetDailyVol`**
  - `test_returns_pandas_series` — output is `pd.Series`
  - `test_length_matches_input` — output length equals input length
  - `test_flat_price_gives_near_zero_vol` — all-constant close → all σ_t ≈ 0 (after warm-up)
  - `test_raises_on_non_series` — passing a list raises `TypeError`
  - `test_non_positive_prices_masked` — zero/negative prices in input → no inf in output

- [ ] `2.3` **Class `TestGetVerticalBarriers`**
  - `test_t1_is_within_close_index` — every t1 date is a valid date in the close index
  - `test_horizon_5_gives_correct_offset` — event at position 0, horizon=5 → t1 is close.index[5]
  - `test_censored_at_end_of_history` — event near end of series, horizon=20, data only has 5 bars left → t1 is last date

- [ ] `2.4` **Class `TestApplyPtSlOnT1`**
  - `test_pt_hit_before_vertical_barrier` — construct scenario where price rises by >1.5σ within 5 bars; assert `result['pt'].notna().any()`
  - `test_sl_hit_before_vertical_barrier` — price drops by >1.5σ within 5 bars; assert `result['sl'].notna().any()`
  - `test_vertical_barrier_when_no_touch` — price stays flat (near-zero σ); assert `result['pt'].isna().all()` and `result['sl'].isna().all()` and `result['t1'] == events['t1']`
  - `test_conservative_tiebreak_sl_wins` — same bar triggers both PT and SL (wide-range bar); assert `result['pt'].isna()` and `result['sl'].notna()` (SL wins)
  - `test_t1_is_never_before_t0` — for all rows, `result['t1'] >= events.index` (no look-ahead)
  - `test_empty_events_returns_empty_dataframe` — empty events → empty DataFrame with correct columns

- [ ] `2.5` **Class `TestGetBins`**
  - `test_bin_values_in_012_scheme` — scheme "012": all `bin` values in `{0.0, 1.0, 2.0, nan}`
  - `test_bin_values_in_raw_scheme` — scheme "raw": all `bin` values in `{-1.0, 0.0, 1.0, nan}`
  - `test_invalid_scheme_raises` — unknown scheme → `ValueError`
  - `test_pt_hit_gives_up_bin` — synthetic event where `events['pt']` is set and `events['sl']` is NaT → bin == 2.0 (scheme "012")
  - `test_sl_hit_gives_down_bin` — `events['sl']` is set and `events['pt']` is NaT → bin == 0.0 (scheme "012")
  - `test_invalid_price_gives_nan_bin` — close price at entry is 0 → bin is NaN (ruthless excision)

- [ ] `2.6` **Class `TestTripleBarrierPipeline`**
  - `test_returns_polars_dataframe` — output is `pl.DataFrame`
  - `test_output_columns_complete` — columns include `{ticker, t0, t1, trgt, ret, bin, num_co_events, uniqueness, w}`
  - `test_bin_dtype_is_int64` — after pipeline, `bin` column dtype is Int64 (not float)
  - `test_no_look_ahead_t1_ge_t0` — for every row, `t1 >= t0` (no backward time travel)
  - `test_no_nan_bin_in_output` — output contains no NaN bin values (unlabelable rows are excised)
  - `test_normalize_weights_gives_mean_1` — `normalize_weights=True` (default) → `result['w'].mean() ≈ 1.0`
  - `test_two_tickers_both_present` — panel with 2 tickers → both appear in output
  - `test_raises_on_missing_close_column` — `close_col='nonexistent'` → `ValueError`
  - `test_raises_when_all_tickers_skipped` — tiny frame (3 rows per ticker, fewer than `vol_span+horizon+5=35`) → `ValueError` with "zero ticker had usable history"
  - `test_t5_vs_t20_horizons_produce_different_outputs` — same data, `cfg.horizon=5` vs `cfg.horizon=20` → different `t1` dates (assert not equal)

**Verification gate for Step 2:**
```powershell
python -m pytest tests/test_triple_barrier.py -v
```
Expected: all new tests pass, existing 251 unchanged.

---

### Step 3 — `tests/test_tabular_ensemble.py`

**Target:** `src/models/tabular_ensemble.py` — `TabularEnsemble.fit`, `TabularEnsemble.predict_proba`, `TabularEnsemble.predict_proba_3class`

**Strategy:** Mock the three GPU-backed base learners (LightGBM, XGBoost, CatBoost) using `unittest.mock.MagicMock` returned from `_build_base_models`. The mocks must satisfy the sklearn interface (`fit`, `predict_proba`, `classes_`, `feature_names_in_`). This avoids CUDA dependency and slow training while still testing the stacking wiring, OOF assembly, LogisticRegression meta fit, and CalibratedClassifierCV wrapper.

**Mocking approach:** Patch `src.models.tabular_ensemble._build_base_models` with a factory that returns one or two `MagicMock` classifiers whose `predict_proba` returns a deterministic `(n, 3)` float32 array (e.g., `np.tile([0.3, 0.4, 0.3], (n, 1)).astype(np.float32)`). Patch `_HAS_LGB`, `_HAS_XGB`, `_HAS_CAT` as needed.

**Real dependencies used (not mocked):**
- `sklearn.linear_model.LogisticRegression` (fast, no GPU)
- `sklearn.calibration.CalibratedClassifierCV` (fast, no GPU)
- `src.models.stacking_model.purged_kfold.PurgedKFold` (pure sklearn)
- `numpy`, `pandas`

**New file:** `tests/test_tabular_ensemble.py`

**Imports to use:**
```
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from src.models.tabular_ensemble import TabularEnsemble, UP_CLASS, CLASSES
```

**Helper factories (module-level):**

- `_make_xy(n=200, n_features=9, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  Returns `(X, y, start_times, end_times, sample_weight)` where:
  - `X` is `(n, n_features)` float32, values from rng
  - `y` is `(n,)` int64 with all three classes present (≈ 33% each, seeded)
  - `start_times` is `(n,)` datetime64 array, one day apart starting 2022-01-03
  - `end_times` is `start_times + 5 business days` (approximated as +7 calendar days)
  - `sample_weight` is `np.ones(n, dtype=np.float32)`

- `_mock_base_model(n_classes=3) -> MagicMock`
  Returns a MagicMock with:
  - `fit` returning `self`
  - `predict_proba(X)` returning `np.tile([1/3, 1/3, 1/3], (len(X), 1)).astype(np.float32)`
  - `classes_` = `np.array([0, 1, 2])`
  - `feature_names_in_` = None (triggers `_as_named_df` fallback)

- `_patched_base_models(learner_names=['mock_a', 'mock_b']) -> dict`
  Returns `{name: _mock_base_model() for name in learner_names}` — used as the return value when patching `_build_base_models`.

**Test classes and function names (exact):**

- [ ] `3.1` Create `tests/test_tabular_ensemble.py`

- [ ] `3.2` **Class `TestTabularEnsembleUnfitted`**
  - `test_predict_proba_raises_before_fit` — `ens.predict_proba(X)` before fit → `RuntimeError("not fitted")`

- [ ] `3.3` **Class `TestTabularEnsembleFitWiring`** (uses `patch('src.models.tabular_ensemble._build_base_models', return_value=_patched_base_models())`)
  - `test_fit_returns_self` — `ens.fit(...)` returns the ensemble instance
  - `test_base_models_populated_after_fit` — `len(ens.base_models) == 2` (one per patched learner)
  - `test_meta_is_set_after_fit` — `ens.meta is not None`
  - `test_learner_names_match_patch` — `ens.learner_names` matches keys returned by the patch
  - `test_oof_meta_feature_names_count` — `len(ens.oof_meta_feature_names) == 2 * 3 == 6` (2 learners × 3 class probs)
  - `test_contribution_sums_to_1` — `sum(ens.contribution.values()) == pytest.approx(1.0)`

- [ ] `3.4` **Class `TestTabularEnsemblePredictProba`** (fit once in setup with patched models, reuse)
  - `test_predict_proba_shape` — output shape == `(n,)` where n = number of input rows
  - `test_predict_proba_values_in_unit_interval` — all values in `[0.0, 1.0]`
  - `test_predict_proba_3class_shape` — output shape == `(n, 3)`
  - `test_predict_proba_3class_rows_sum_to_1` — `np.allclose(result.sum(axis=1), 1.0)`
  - `test_predict_proba_accepts_numpy_array` — X as numpy array, no exception
  - `test_predict_proba_accepts_dataframe` — X as named DataFrame, no exception

- [ ] `3.5` **Class `TestTabularEnsembleCalibration`**
  - `test_calibrate_true_wraps_in_calibrated_cv` — `ens.calibrate=True` → `type(ens.meta).__name__ == 'CalibratedClassifierCV'` (default path)
  - `test_calibrate_false_uses_raw_logreg` — `ens.calibrate=False` → `type(ens.meta).__name__ == 'LogisticRegression'`
  - `test_calibration_skipped_when_class_count_too_small` — patch y so one class has only 1 sample → `ens.meta` is `LogisticRegression` (fallback, with a warning logged) — assert `type(ens.meta).__name__ == 'LogisticRegression'`

- [ ] `3.6` **Class `TestTabularEnsembleAugmentMissingClasses`**
  - `test_missing_flat_class_gets_synth_injected` — `y` has only classes 0 and 2 (no FLAT); after `_augment_for_missing_classes`, all three classes present in `y_aug`
  - `test_synth_weight_is_tiny` — injected rows have `w < 1e-3`
  - `test_no_augmentation_when_all_classes_present` — no missing class → inputs returned unchanged

- [ ] `3.7` **Class `TestTabularEnsembleFeatureImportances`**
  - `test_feature_importances_shape` — `ens.feature_importances_` shape == `(n_features,)` after fit (mock models return `feature_importances_` from `MagicMock` attribute — set it explicitly on the mock to a valid numpy array)
  - `test_feature_names_in_shape` — `ens.feature_names_in_` is ndarray of strings matching `feature_names`

**Verification gate for Step 3:**
```powershell
python -m pytest tests/test_tabular_ensemble.py -v
```
Expected: all new tests pass, existing 251 unchanged.

---

### Step 4 — `tests/test_run_backtest_wiring.py` + context fix

**Targets:**
- `run_backtest._build_wf_config` — pure WalkForwardConfig builder (no external deps)
- `run_backtest.run_oos` — calls `WalkForwardEngine.run`; mock the engine's `run` output
- Context fix: `process/context/all-context.md` line `run_backtest.main — degree 97` → corrected to `run_backtest.run_oos / _build_wf_config — degree 97`

**Mocking strategy for `run_oos`:**
- Mock `src.backtest.walk_forward.WalkForwardEngine.run` to return a minimal `EngineResult` (or a `MagicMock` with `.equity_curve` attribute set to a minimal DataFrame with columns `date, nav, daily_return`).
- Do NOT call `main()` in any test (avoids checkpoint file, dataset materialization, full sweep).
- Do NOT mock `_build_wf_config` itself — it is a pure function and should be tested real.

**Price-scale critical assertion in `run_oos` test:** After engine construction, verify via the `wf_cfg.price_unit_vnd` field that the config carries `1000.0` (the default in `WalkForwardConfig`), proving the scale conversion is wired.

**Checkpoint avoidance:** Do not call `main()` or `_load_checkpoint()`. Only test the two extracted pure/semi-pure functions.

**New file:** `tests/test_run_backtest_wiring.py`

**Imports to use:**
```
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import polars as pl
import pytest
from datetime import date
from run_backtest import _build_wf_config, run_oos, equity_metrics, monthly_net_sharpe
from src.backtest.pipeline import RunConfig
from src.backtest.walk_forward import WalkForwardConfig
from src.execution.vn_cost_model import ExecutionConfig
```

**Helper factories (module-level):**

- `_run_config() -> RunConfig` — returns `RunConfig()` with defaults
- `_tiny_panel(n_days=10) -> pl.DataFrame` — polars panel: 2 tickers ("AAA", "BBB"), 10 business days, `date, ticker, open, high, low, close, volume` columns. Prices in thousands-VND (e.g., close=15.0). Includes one feature column `feat=0.5`.
- `_mock_equity_curve(n_days=5) -> pd.DataFrame` — DataFrame with columns `date (date objects), nav, daily_return` for `n_days` rows, nav=1_000_000 + linear trend, daily_return=0.001.
- `_mock_engine_result(n_days=5) -> MagicMock` — MagicMock with `.equity_curve = _mock_equity_curve(n_days)`.

**Test classes and function names (exact):**

- [ ] `4.1` Create `tests/test_run_backtest_wiring.py`

- [ ] `4.2` **Sub-task: Fix stale context reference** — edit `process/context/all-context.md`
  - Find: `run_backtest.main — degree 97` in the Hub nodes section
  - Replace with: `run_backtest.run_oos / _build_wf_config — degree 97` (no `main` node at this hub degree exists; real hub nodes are the two extracted functions)
  - This is a documentation fix, not a source file change.

- [ ] `4.3` **Class `TestBuildWfConfig`** (no mocks — pure function)
  - `test_returns_walk_forward_config_instance` — return value is `WalkForwardConfig`
  - `test_mode_tranche_propagates` — `mode='tranche'` → `cfg.rebalance_mode == 'tranche'`
  - `test_mode_grid_propagates` — `mode='grid'` → `cfg.rebalance_mode == 'grid'`
  - `test_hold_days_propagates` — `hold_days=20` → `cfg.tranche_hold_days == 20`
  - `test_cutoff_becomes_start_trading_date` — `cutoff=date(2024, 1, 2)` → `cfg.start_trading_date == date(2024, 1, 2)`
  - `test_exec_config_is_execution_config` — `cfg.exec_config` is instance of `ExecutionConfig`
  - `test_feature_cols_propagates` — `tabular_features=['feat_a', 'feat_b']` → `cfg.feature_cols == ['feat_a', 'feat_b']`
  - `test_regime_sizing_propagates` — `use_regime_sizing=True` → `cfg.use_regime_sizing == True`
  - `test_price_unit_vnd_default_is_1000` — without explicit override, `cfg.price_unit_vnd == 1000.0` (documents VN price-scale convention wiring)

- [ ] `4.4` **Class `TestRunOosWiring`** (patches `WalkForwardEngine.run`)
  - Setup: `make_ensemble_oracle` must also be patched to return a lambda → patch `run_backtest.make_ensemble_oracle` returning `lambda X: np.full((X.shape[0],), 0.5)`. Patch `src.backtest.walk_forward.WalkForwardEngine.run` to return `_mock_engine_result(n_days=5)`.

  - `test_run_oos_returns_dataframe` — return value is `pd.DataFrame`
  - `test_run_oos_columns_include_date_nav_return` — output has `date, nav, daily_return` columns
  - `test_run_oos_equity_curve_trimmed_to_cutoff` — all `date` values in output are `>= cutoff`
  - `test_run_oos_calls_engine_run_once` — `WalkForwardEngine.run` was called exactly once
  - `test_run_oos_mode_tranche_passes_to_config` — spy on `_build_wf_config` (patch it to call-through but record args): `mode='tranche'` is passed through correctly
  - `test_run_oos_mode_grid_passes_to_config` — same for `mode='grid'`

- [ ] `4.5` **Class `TestEquityMetrics`** (pure function, no mocks)
  - `test_net_pnl_correct` — `eq = DataFrame(nav=[1_000_000, 1_100_000], daily_return=[0.0, 0.1])` → `metrics['net_pnl'] == 100_000.0`
  - `test_total_return_correct` — `nav[-1]/nav[0] - 1 == 0.1` → `metrics['total_return'] == pytest.approx(0.1)`
  - `test_max_drawdown_negative` — nav dips below start then recovers → `metrics['max_drawdown'] < 0`
  - `test_empty_nav_returns_initial_capital` — `eq = DataFrame(nav=[], daily_return=[])` → `metrics['final_nav'] == initial_capital`

**Verification gate for Step 4:**
```powershell
python -m pytest tests/test_run_backtest_wiring.py -v
```
Expected: all new tests pass, existing 251 unchanged.

---

## Full-Suite Final Gate

After all four steps:
```powershell
python -m pytest -q
```
Expected: 251 (baseline) + new tests, zero failures, zero regressions.

---

## Dependencies

| Dependency | Available | Notes |
|---|---|---|
| `src.execution.vn_cost_model` | Yes — pure Python | No mock needed |
| `src.labels.triple_barrier` | Yes — pandas/numpy/polars | No mock needed |
| `src.models.tabular_ensemble` | Yes — imports LGB/XGB/CAT | Mock `_build_base_models` |
| `src.models.stacking_model.purged_kfold` | Yes — pure sklearn | Use real |
| `sklearn.linear_model.LogisticRegression` | Yes | Use real |
| `sklearn.calibration.CalibratedClassifierCV` | Yes | Use real |
| `run_backtest._build_wf_config`, `run_oos` | Yes | Mock `WalkForwardEngine.run` and `make_ensemble_oracle` |
| `src.backtest.walk_forward.WalkForwardConfig` | Yes | Use real |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CatBoost/LightGBM imports fail in bare env | LOW (ML stack installed on Windows box) | Step 3 mocks `_build_base_models` entirely so actual GPU libs are never called |
| `PurgedKFold` purges all folds on tiny synthetic n=200 data | LOW | Use n=200 with n_splits=3 and embargo_bars=5; verify coverage in test |
| `triple_barrier_pipeline` raises "zero usable history" on short panel | ANTICIPATED | `TestTripleBarrierPipeline.test_raises_when_all_tickers_skipped` explicitly tests this; happy-path tests use n=60 bars (above vol_span+horizon+5=35) |
| `run_oos` import triggers checkpoint loading at module level | NO — `_load_checkpoint` is only called inside `main()` which tests never call | Confirmed by reading `run_backtest.py` — no module-level side effects |
| `WalkForwardEngine.run` requires materialized Polars panel | MITIGATED | Mock `WalkForwardEngine.run` return value; tiny panel only flows through `_build_wf_config` path |
| stale all-context.md "run_backtest.main" entry misleads future agents | CONFIRMED | Sub-task 4.2 fixes this in the same step |

---

## Verification Evidence

Each step's tests produce these observable artifacts:

**Step 1 (VNCostModel):**
- `fill.brokerage_paid == gross_notional * 0.0015` (exact arithmetic, not approximation)
- `fill.tax_paid == 0` on BUY (explicit zero assertion)
- `fill.tax_paid == gross_notional * 0.0010` on SELL
- Rejection reason enum value is `"price_at_ceiling_buy"` for thousand-scale price input

**Step 2 (TripleBarrier):**
- All `bin` values in `{0, 1, 2}` with `dtype Int64`
- `t1 >= t0` for every row (no look-ahead)
- `result['w'].mean() ≈ 1.0` under `normalize_weights=True`
- `ValueError` raised on panels too short to label

**Step 3 (TabularEnsemble):**
- `type(ens.meta).__name__` is either `'CalibratedClassifierCV'` or `'LogisticRegression'` depending on `calibrate` flag
- `ens.meta is not None` after fit
- `predict_proba` output shape is `(n,)` with values in `[0, 1]`
- `oof_meta_feature_names` has `n_learners * 3` entries

**Step 4 (run_backtest wiring):**
- `cfg.rebalance_mode == 'tranche'` (mode propagated)
- `cfg.price_unit_vnd == 1000.0` (price-scale convention wired)
- `WalkForwardEngine.run` called exactly once per `run_oos` call
- all equity-curve rows have `date >= cutoff`

---

## Backwards Compatibility

All changes are additive. No production source files are touched. The only non-test file edited is `process/context/all-context.md` (documentation fix). No schema changes. No config changes.

---

## Rollback

If any step breaks the baseline suite:
1. Run `python -m pytest -q` to confirm which tests regress.
2. Delete the offending new test file (e.g., `tests/test_triple_barrier.py`).
3. The baseline suite is unchanged — rollback is instant.
4. Investigate the root cause before rewriting the test file.

---

## Resume and Execution Handoff

**For EXECUTE agent:**

Exact plan file path:
`C:\Users\caokh\Desktop\vscode\stock_price_v3\process\features\v4-1-structural-debt\active\phase3-hub-coverage_PLAN_21-06-26.md`

Execute steps in order: Step 1 → Step 2 → Step 3 → Step 4.

**After each step, the ORCHESTRATOR (not the EXECUTE subagent) runs:**
```powershell
python -m pytest tests/test_<new_file>.py -v
```
and must see a clean pass before spawning the next subagent for the next step.

**Environment reminders:**
- Use PowerShell for all shell commands (git-bash is broken).
- pytest runner: `python -m pytest` (bare `pytest` may not resolve on this machine).
- Python 3.11 with ML stack is on PATH; conda `stock` env has no pytest — do not activate it.
- All test files go in `tests/` (repo root level).
- All imports use the same pattern as `test_walk_forward_price_scale.py`: absolute imports from `src.*` or top-level `run_backtest`.

**Sub-task 4.2 (context fix) should be done before or alongside writing the test file for Step 4.**

**After all steps complete:** archive this plan to `process/features/v4-1-structural-debt/completed/` and update the `_GUIDE.md` to mark Phase 3 as completed.

---

## Acceptance Criteria

- [ ] `tests/test_vn_cost_model.py` exists with all classes and test functions named above.
- [ ] `tests/test_triple_barrier.py` exists with all classes and test functions named above.
- [ ] `tests/test_tabular_ensemble.py` exists with all classes and test functions named above.
- [ ] `tests/test_run_backtest_wiring.py` exists with all classes and test functions named above.
- [ ] `process/context/all-context.md` Hub nodes section no longer says `run_backtest.main`; says `run_backtest.run_oos / _build_wf_config` instead.
- [ ] `python -m pytest -q` passes with zero failures (baseline 251 + new tests).
- [ ] No source file under `src/` or `run_backtest.py` was modified.

---

## Plan Validator

```
node .claude/skills/vc-generate-plan/scripts/validate-plan-artifact.mjs process/features/v4-1-structural-debt/active/phase3-hub-coverage_PLAN_21-06-26.md
```
