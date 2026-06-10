# Phase 2 — Automated Feature-Schema Hashing
**Plan type:** COMPLEX (multi-file, system-wide parity enforcement)
**Feature:** v4-1-structural-debt
**Created:** 2026-06-10
**Status:** READY FOR EXECUTE

---

## Overview

Replace the manually maintained `FEATURE_RECIPE_VERSION = "v1.1"` string with a
deterministic SHA-256 hash computed from the actual feature schema at module-import
time. The hash encodes what matters — column names, column order, dtype strings, and
the `frac_diff_d` hyperparameter — so any drift is automatically detected without
requiring a developer to remember to bump a version string.

A parallel schema hash is added to the MR-LGBM pipeline, which currently has NO
version guard at all.

---

## Goals

1. `FEATURE_RECIPE_VERSION` can never drift from the actual pipeline shape silently.
2. Any column add/remove/reorder or dtype change instantly produces a new hash, which
   triggers a retrain demand on the next serve-path load.
3. The MR sub-model gains its own parity guard for the first time.
4. `train_models.py` checkpoint metadata stamps the hash so post-training audits are
   traceable.
5. No breaking changes to the artifact-loading tripwire logic in `main._load_v3_bot`.
6. All existing tests remain green; new unit tests cover the hash utility and
   pipeline integration.

---

## Scope

**In scope:**
- New utility: `src/utils/schema_hash.py`
- Modified: `src/backtest/pipeline.py` (schema constant + computed hash)
- Modified: `src/features/mr_features.py` (MR schema hash constant)
- Modified: `src/models/train_mr_lgbm.py` (stamp hash into `mr_threshold.json`)
- Modified: `train_models.py` (stamp hash into training checkpoint metadata)
- Modified: `run_backtest.py` (no logic change; already uses the imported constant)
- New tests: `tests/test_schema_hash.py`
- Updated tests: `tests/test_serve_resilience.py` (update `"v1.0"` fake metadata)

**Out of scope:**
- Changing the live tripwire logic in `main._load_v3_bot` (behaviour preserved)
- Migration of existing artifacts (pre-hash artifacts accepted with warning, same as
  current behavior for missing stamps)
- MR serve-path guard in `main._load_mr` (hash is stamped into `mr_threshold.json`
  but the optional feature-list check there is not changed)

---

## Design Decisions (LOCKED)

| Decision | Choice |
|---|---|
| What to hash | Column names (ordered) + dtype strings (Polars canonical) + `frac_diff_d` |
| Hash format | `"v2-sha8:a3f7c921"` — SHA-256 truncated to 8 hex chars, version-prefixed |
| New utility | `src/utils/schema_hash.py` — pure Python, no Polars dependency |
| Schema constant location | `src/backtest/pipeline.py` as `FEATURE_SCHEMA: list[tuple[str, str]]` |
| `FEATURE_RECIPE_VERSION` | Replaced by `compute_feature_schema_hash(FEATURE_SCHEMA, frac_diff_d)` at module level |
| MR constant location | `MR_SCHEMA_HASH` in `src/features/mr_features.py` |
| MR stamp target | `mr_threshold.json` gets `"schema_hash"` field |
| Training checkpoint | `train_models.py` adds `"feature_schema_hash"` to `checkpoint["metadata"]` |
| Backward compat | Pre-hash artifacts (no stamp) → WARN + continue (existing behavior) |

---

## Touchpoints

| File | Symbol | Change type |
|---|---|---|
| `src/backtest/pipeline.py` | `FEATURE_RECIPE_VERSION` (line 65) | Replace with computed hash |
| `src/backtest/pipeline.py` | `FEATURE_SCHEMA` | New constant (ordered name+dtype pairs) |
| `src/backtest/pipeline.py` | `build_features()` | Add runtime assertion that `all_features` names match schema |
| `src/backtest/pipeline.py` | `__all__` | Add `FEATURE_SCHEMA`, replace `FEATURE_RECIPE_VERSION` import surface |
| `src/features/mr_features.py` | `MR_SCHEMA_HASH` | New constant computed from `MR_FEATURE_COLUMNS` |
| `src/models/train_mr_lgbm.py` | `threshold_doc` dict (line 341) | Add `"schema_hash"` key |
| `train_models.py` | `checkpoint["metadata"]` (line 168) | Add `"feature_schema_hash"` key |
| `run_backtest.py` | `bundle["metadata"]["feature_recipe_version"]` (line 491) | No change — already uses the imported constant |
| `main.py` | `_load_v3_bot()` (lines 332–353) | No logic change — tripwire uses the same string comparison |
| `tests/test_serve_resilience.py` | `_FakeBot.metadata` (line 54) | Update `"v1.0"` → a valid hash format string |
| `src/utils/schema_hash.py` | `compute_feature_schema_hash()` | New file + function |
| `tests/test_schema_hash.py` | all | New file |

---

## Public Contracts

- `FEATURE_RECIPE_VERSION` (exported from `src/backtest/pipeline`) remains the same
  public name and string type. Its value changes from `"v1.1"` to `"v2-sha8:XXXXXXXX"`.
  No import sites change.
- `compute_feature_schema_hash(schema, frac_diff_d)` is a pure function:
  `(list[tuple[str, str]], float | None) -> str`. No side effects. No Polars import.
- `FEATURE_SCHEMA` is a new module-level constant of type `list[tuple[str, str]]`
  exported from `src/backtest/pipeline`.
- `MR_SCHEMA_HASH` is a new module-level `str` constant exported from
  `src/features/mr_features.py`.

---

## Blast Radius

**High:**
- `src/backtest/pipeline.py` — hub node (degree 142 per code-review-graph). Changing
  `FEATURE_RECIPE_VERSION` value means ALL existing artifacts on disk will appear
  "unstamped" (no stamp) on next load — they get the warning-and-continue path, not
  RuntimeError. This is safe but forces a retrain to get clean stamps.

**Medium:**
- `main.py` — imports `FEATURE_RECIPE_VERSION` (line 34). Value changes but logic
  is untouched. Any artifact stamped with the old `"v1.1"` string will mismatch →
  `RuntimeError` at serve time. Operator must run retrain before deploying Phase 2.
- `run_backtest.py` — imports `FEATURE_RECIPE_VERSION` (line 52) and stamps it at
  line 491. No logic change needed; stamping continues to work.
- `tests/test_serve_resilience.py` — uses `"v1.0"` in fake metadata (line 54). This
  will NOT mismatch the live constant (it's just a fake artifact) but it should be
  updated to use the hash format for test clarity and to avoid confusion.

**Low:**
- `train_models.py` — metadata key addition only; checkpoint schema version
  (`CHECKPOINT_SCHEMA = "v4-train-ckpt-1.0"`) is not changed.
- `src/features/mr_features.py` — new constant appended; no existing logic touched.
- `src/models/train_mr_lgbm.py` — new key in `threshold_doc` dict; JSON structure is
  additive, no existing consumer breaks.

**No blast radius:**
- `src/bot/bot_inference.py` — reads `metadata["feature_recipe_version"]` but that
  key and its comparison live in `main._load_v3_bot`, not in `V3BotInference` itself.

---

## Feature Schema Specification

The 15-column schema (ordered, as per `pipeline.py` line 417):

```
FEATURE_SCHEMA = [
    # 9 original / always-survive (continuous Float32)
    ("close_fd_xsz",           "Float32"),
    ("volume_fd_xsz",          "Float32"),
    ("mom20_xsz",              "Float32"),
    ("overext_5_xsz",          "Float32"),
    ("overext_20_xsz",         "Float32"),
    ("rs_10_xsz",              "Float32"),
    ("rs_20_xsz",              "Float32"),
    ("smart_money_20_xsz",     "Float32"),
    ("vol_squeeze_xsz",        "Float32"),
    # 5 candidates (continuous Float32)
    ("amihud_liquidity_xsz",   "Float32"),
    ("realized_skewness_20d_xsz", "Float32"),
    ("vol_of_vol_20d_xsz",     "Float32"),
    ("hl_range_ratio_xsz",     "Float32"),
    ("gap_risk_xsz",           "Float32"),
    # 1 categorical (Int8)
    ("market_regime",          "Int8"),
]
```

Dtype strings are the **Polars canonical** type names (verified: `tensor_builder.py`
uses `pl.Float32` for all `_xsz` columns; `market_regime.py` line 176 uses
`.cast(pl.Int8)`).

The `frac_diff_d` default is `0.4` (from `RunConfig`). It is part of the hash input
because it changes the numerical values (not just names) of `close_fd` and
`volume_fd`. When `frac_diff_d=None` is passed the hash serializes as `"None"`.

**MR schema** (11 columns, all treated as `float64` / untyped for the hash — the MR
pipeline is pandas-native with no Polars dtype contract):

```
MR_SCHEMA_COLS = [
    "mr_dma_sma10", "mr_dma_sma20", "mr_dma_sma50",
    "mr_bb_pctb", "mr_bb_below_lower",
    "mr_rsi_9", "mr_rsi_14", "mr_williams_r_14",
    "mr_atr_norm_14", "mr_gap_pct", "mr_gap_down",
]
```

For MR, all dtypes are serialized as `"float64"` (the numpy output dtype from
`build_mr_features`). `frac_diff_d=None` (MR has no frac-diff step).

---

## Implementation Checklist

### Step 1 — Create `src/utils/schema_hash.py`

**File:** `src/utils/schema_hash.py` (new file)

Create a pure-Python module with a single exported function. No Polars import. No
dependency on any project module.

Implementation contract:
- Function signature: `compute_feature_schema_hash(schema: list[tuple[str, str]], frac_diff_d: float | None) -> str`
- Serialization: build a deterministic string from `schema` items: each item as
  `"name:dtype"`, joined with `"|"`, then append `"|frac_diff_d:{frac_diff_d}"`.
- Hashing: `hashlib.sha256(serialized.encode()).hexdigest()[:8]`
- Return: `f"v2-sha8:{hex8}"`
- Module-level docstring explaining the contract (what is hashed and why).
- `__all__ = ["compute_feature_schema_hash"]`

Verification: the function is deterministic — same inputs always produce the same
output. Different column order or dtype change produces a different output.

---

### Step 2 — Add `FEATURE_SCHEMA` constant to `src/backtest/pipeline.py`

**File:** `src/backtest/pipeline.py`

After line 65 (the `FEATURE_RECIPE_VERSION` line), add:

1. Import `compute_feature_schema_hash` from `src.utils.schema_hash` at the top of
   the file (add to imports block, after the existing `from src.features.market_regime
   import build_regime_features` line).
2. Add `FEATURE_SCHEMA: list[tuple[str, str]]` constant — the 15-tuple list from the
   specification above — as a new module-level constant with a doc comment explaining
   it is the authoritative schema for the feature pool in `build_features`.
3. Replace `FEATURE_RECIPE_VERSION = "v1.1"` with:
   ```
   FEATURE_RECIPE_VERSION: str = compute_feature_schema_hash(
       FEATURE_SCHEMA, RunConfig().frac_diff_d
   )
   ```
   Keep the existing comment block above it, but update the comment to say the value
   is now computed automatically from `FEATURE_SCHEMA` instead of manually maintained.

Note: `RunConfig().frac_diff_d` defaults to `0.4`. This makes the hash stable across
imports unless `RunConfig` defaults change — which is the desired behavior (a default
change IS a recipe change).

Verification: `python -c "from src.backtest.pipeline import FEATURE_RECIPE_VERSION; print(FEATURE_RECIPE_VERSION)"` prints `v2-sha8:XXXXXXXX` (8 hex chars, `v2-sha8:` prefix).

---

### Step 3 — Add runtime assertion in `build_features()`

**File:** `src/backtest/pipeline.py`, inside `build_features()` function

After line 417 (`all_features = original_features + candidate_features + CATEGORICAL_FEATURES`), add a runtime assertion that checks the names in `all_features` match the names in `FEATURE_SCHEMA` in the same order:

```
assert [name for name, _ in FEATURE_SCHEMA] == all_features, (
    f"FEATURE_SCHEMA names do not match all_features order. "
    f"Schema: {[n for n,_ in FEATURE_SCHEMA]} | built: {all_features}. "
    f"Update FEATURE_SCHEMA to match the hardcoded pool."
)
```

This assertion fires at pipeline runtime (not import time) if someone changes the pool
order in `build_features` without updating `FEATURE_SCHEMA`, making the drift
immediately visible.

Verification: existing pipeline tests continue to pass. If `all_features` order
changes, the assert fires with a clear message.

---

### Step 4 — Update `__all__` in `src/backtest/pipeline.py`

**File:** `src/backtest/pipeline.py`, `__all__` list (line 691)

Add `"FEATURE_SCHEMA"` to the `__all__` list alongside the existing
`"FEATURE_RECIPE_VERSION"`. The string `"FEATURE_RECIPE_VERSION"` stays (it is the
public name; its value is now hash-computed).

Verification: `from src.backtest.pipeline import FEATURE_SCHEMA, FEATURE_RECIPE_VERSION` works without error.

---

### Step 5 — Add `MR_SCHEMA_HASH` to `src/features/mr_features.py`

**File:** `src/features/mr_features.py`

After the `MR_FEATURE_COLUMNS` list definition (line 61), add:

1. Import `compute_feature_schema_hash` from `src.utils.schema_hash`.
2. Build the MR schema list: `_MR_SCHEMA = [(col, "float64") for col in MR_FEATURE_COLUMNS]`
3. Compute: `MR_SCHEMA_HASH: str = compute_feature_schema_hash(_MR_SCHEMA, None)`
4. Add `MR_SCHEMA_HASH` to `__all__` if one exists, or leave at module level (no
   `__all__` currently exists in this file — leave as-is).

Note: `_MR_SCHEMA` is a private helper list, not exported.

Verification: `python -c "from src.features.mr_features import MR_SCHEMA_HASH; print(MR_SCHEMA_HASH)"` prints `v2-sha8:XXXXXXXX`.

---

### Step 6 — Stamp MR schema hash into `mr_threshold.json` in `train_mr_lgbm.py`

**File:** `src/models/train_mr_lgbm.py`

1. Add import: `from src.features.mr_features import MR_SCHEMA_HASH` (alongside the
   existing `from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features`
   at line 52).
2. In the `threshold_doc` dict (around line 341), add `"schema_hash": MR_SCHEMA_HASH`
   as a new key at the top of the dict, before `"model"`.

Resulting `threshold_doc` shape (additive change only):
```python
threshold_doc = {
    "schema_hash": MR_SCHEMA_HASH,   # NEW — automated parity guard
    "model": "mr_lgbm_single",
    "tau": tau,
    ...
}
```

Verification: after running `python src/models/train_mr_lgbm.py`, the file
`models/mr/mr_threshold.json` contains a `"schema_hash"` key with a `"v2-sha8:"`
prefixed value.

---

### Step 7 — Stamp feature schema hash into training checkpoint in `train_models.py`

**File:** `train_models.py`

1. Add import: `from src.backtest.pipeline import FEATURE_RECIPE_VERSION` is already
   imported indirectly through other pipeline symbols. Confirm `FEATURE_RECIPE_VERSION`
   is accessible. If not, add explicitly.
   
   Actually `train_models.py` does NOT currently import `FEATURE_RECIPE_VERSION`. The
   hash value is available via the already-imported pipeline module. Add:
   `from src.backtest.pipeline import FEATURE_RECIPE_VERSION` to the imports block.

2. In the `checkpoint["metadata"]` dict (around line 168–180), add:
   `"feature_schema_hash": FEATURE_RECIPE_VERSION,` as a new entry.

The key name `"feature_schema_hash"` is deliberately distinct from
`"feature_recipe_version"` (which lives in the bundle persisted by `run_backtest.py`)
to make the checkpoint and the bundle independently queryable.

Verification: after running `python train_models.py`, the loaded checkpoint dict has
`checkpoint["metadata"]["feature_schema_hash"]` set to the current hash string.

---

### Step 8 — Update `tests/test_serve_resilience.py`

**File:** `tests/test_serve_resilience.py`, line 54

The `_FakeBot.metadata` dict uses `"feature_recipe_version": "v1.0"`. After this
change, the live `FEATURE_RECIPE_VERSION` is `"v2-sha8:XXXXXXXX"`, so this fake
value will NOT match — but the test monkeypatches `_load_v3_bot` entirely, so the
tripwire never runs against the live constant. However, updating the fake value to
a hash-format string reduces future confusion.

Change:
```python
metadata = {"feature_recipe_version": "v1.0", "tb_horizon": 5}
```
to:
```python
metadata = {"feature_recipe_version": "v2-sha8:00000000", "tb_horizon": 5}
```

This is a cosmetic update — it makes the fake artifact look like a real post-Phase-2
artifact without affecting test logic.

Verification: `pytest tests/test_serve_resilience.py` still passes with 2 tests
green.

---

### Step 9 — Create `tests/test_schema_hash.py`

**File:** `tests/test_schema_hash.py` (new file)

Write the following test cases (all importable without the ML stack — no Polars, no
pipeline imports required for the hash utility itself):

**Test 1: `test_hash_format`**
- Call `compute_feature_schema_hash([("close_fd_xsz", "Float32")], 0.4)`
- Assert result starts with `"v2-sha8:"`
- Assert `len(result) == len("v2-sha8:") + 8`

**Test 2: `test_hash_deterministic`**
- Call twice with same args, assert results are equal.

**Test 3: `test_hash_column_order_matters`**
- Call with `[("a", "Float32"), ("b", "Float32")]` vs `[("b", "Float32"), ("a", "Float32")]`
- Assert results differ.

**Test 4: `test_hash_dtype_matters`**
- Call with `[("col", "Float32")]` vs `[("col", "Float64")]`
- Assert results differ.

**Test 5: `test_hash_frac_diff_matters`**
- Call with `frac_diff_d=0.4` vs `frac_diff_d=0.5`
- Assert results differ.

**Test 6: `test_hash_none_frac_diff`**
- Call with `frac_diff_d=None`
- Assert result starts with `"v2-sha8:"` — no TypeError.

**Test 7: `test_feature_recipe_version_format`**
- Import `FEATURE_RECIPE_VERSION` from `src.backtest.pipeline`
- Assert starts with `"v2-sha8:"`
- Assert length is 16 (`"v2-sha8:"` + 8 hex chars)

**Test 8: `test_mr_schema_hash_format`**
- Import `MR_SCHEMA_HASH` from `src.features.mr_features`
- Assert starts with `"v2-sha8:"`
- Assert length is 16

**Test 9: `test_feature_schema_names_match_pipeline`**
- Import `FEATURE_SCHEMA`, `FEATURE_RECIPE_VERSION` from `src.backtest.pipeline`
- Import `CATEGORICAL_FEATURES` from `src.backtest.pipeline`
- Verify `[name for name, _ in FEATURE_SCHEMA]` contains all of `CATEGORICAL_FEATURES`
- Verify length is 15 (9 originals + 5 candidates + 1 categorical)

**Test 10: `test_hash_different_from_old_manual_version`**
- Import `FEATURE_RECIPE_VERSION` from `src.backtest.pipeline`
- Assert `FEATURE_RECIPE_VERSION != "v1.1"` (the old manual value)
- This documents that Phase 2 successfully replaced the old string.

Verification: `pytest tests/test_schema_hash.py -v` passes all 10 tests.

---

## Test Plan

### Existing tests to run after each step

```bash
pytest tests/ -x -q
```

Focus regression tests at each step:

| After step | Run | Expected |
|---|---|---|
| Step 1 | `pytest tests/test_schema_hash.py` | Tests 1–6 pass (utility only) |
| Steps 2–4 | `pytest tests/test_schema_hash.py tests/test_serve_resilience.py` | All pass |
| Step 5 | `pytest tests/test_schema_hash.py` tests 7–9 | Pass |
| Step 6 | Manual: `python src/models/train_mr_lgbm.py` (skippable in CI) | JSON has hash key |
| Step 7 | Manual: `python train_models.py` (skippable in CI) | Checkpoint has hash key |
| Step 8 | `pytest tests/test_serve_resilience.py` | 2 tests green |
| Step 9 | `pytest tests/test_schema_hash.py -v` | All 10 green |
| Final | `pytest tests/ -x -q` | Full suite (158 + 10 new = 168 tests) green |

### Key test files

- `tests/test_schema_hash.py` — 10 new unit tests (pure-Python, fast, no ML stack)
- `tests/test_serve_resilience.py` — 2 existing tests (updated fake metadata)
- `tests/test_daily_inference_integration.py` — should be unaffected (no recipe
  version assertions)

### Manual smoke tests (not automated)

After steps 6 and 7, manually verify:
1. `python -c "from src.backtest.pipeline import FEATURE_RECIPE_VERSION; print(FEATURE_RECIPE_VERSION)"`
   — must print `v2-sha8:XXXXXXXX`
2. `python -c "import json; d=json.load(open('models/mr/mr_threshold.json')); print(d.get('schema_hash', 'MISSING'))"`
   — must print `v2-sha8:XXXXXXXX` (run after step 6 if MR artifacts exist)

---

## Rollback Strategy

Phase 2 is fully reversible until the first artifact retrain:

1. The tripwire in `main._load_v3_bot` handles the pre-hash case (no stamp → warn +
   continue). Existing artifacts are unaffected on disk.
2. If Phase 2 must be reverted:
   - Revert `src/backtest/pipeline.py`: restore `FEATURE_RECIPE_VERSION = "v1.1"`,
     remove `FEATURE_SCHEMA` and the import.
   - Revert `src/features/mr_features.py`: remove `MR_SCHEMA_HASH`.
   - Revert `src/models/train_mr_lgbm.py` and `train_models.py`: remove the new
     metadata keys.
   - Delete `src/utils/schema_hash.py`.
   - Revert `tests/test_serve_resilience.py` line 54.
   - Delete `tests/test_schema_hash.py`.
3. Any artifacts retrained AFTER Phase 2 is merged will stamp the hash. If you
   revert Phase 2 after retraining, those artifacts will mismatch `"v1.1"` and
   trigger RuntimeError. In that case, delete the stamped artifacts and retrain from
   the reverted codebase.

**Decision point:** EXECUTE should run steps 1–9 and confirm `pytest tests/ -x -q`
passes before any artifact retrain. The code change alone is safe; the operational
impact is the mismatch on existing artifacts.

---

## Migration Impact

Existing on-disk artifacts (`v3_ensemble_5d.joblib`, `v3_ensemble_20d.joblib`,
`v3_training_checkpoint.joblib`, `models/mr/mr_threshold.json`) were written before
Phase 2. Their `feature_recipe_version` stamps (if any) will be `"v1.1"` or absent.

Behavior after Phase 2 code merge:
- Artifacts with `"v1.1"` → `RuntimeError` at serve time (mismatch with new hash).
- Artifacts with no stamp → `WARN + continue` (existing backward-compat path).
- Artifacts with the new hash → clean serve.

**Required operator action:** After deploying Phase 2, run:
```
python train_models.py
python run_backtest.py
```
to produce new artifacts stamped with the Phase 2 hash. Until then, the bot will
either warn (unstamped artifacts) or error (old `"v1.1"`-stamped artifacts).

---

## Dependencies

| Dependency | Status |
|---|---|
| Phase 1 of V4.1 Structural Debt | COMPLETE (`_select_candidates`, `_rescue_loop`, `_dispatch_signals` extracted) |
| `src/utils/` directory | EXISTS (`audit_evaluator.py`, `logging_utils.py`, etc. are present) |
| `hashlib` stdlib module | Always available (Python stdlib) |
| `src/backtest/pipeline.py` import of `RunConfig` | Already at module level — `RunConfig().frac_diff_d` works at import time |
| `tests/` pytest infrastructure | EXISTS, 158 tests passing |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `RunConfig()` instantiation at module import raises | Low | High | `RunConfig.__post_init__` is trivial (path coercion only); no I/O. Add a try/except in step 2 if needed. |
| dtype strings are wrong (Float64 vs Float32 for `_xsz` columns) | Medium | Low | Verified: `tensor_builder.py` line 359 uses `pl.Float32` for all `_xsz` suffix columns. The hash still works correctly with any consistent dtype string. |
| Assertion in `build_features` fails in CI (column order drift) | Low | High — test failure | CI will catch it immediately with a clear error message. This is the intended behavior. |
| `test_serve_resilience.py` fake metadata update breaks the test | Low | Low | The fake bot is monkeypatched entirely; the tripwire never runs against the real constant in that test. |
| MR trainer not runnable in CI (no parquet files) | High | Low | Step 6 is a manual smoke test only. No CI test covers the MR trainer end-to-end. |

---

## Verification Evidence

The plan is complete when ALL of these are true:

- [ ] `python -c "from src.backtest.pipeline import FEATURE_RECIPE_VERSION; assert FEATURE_RECIPE_VERSION.startswith('v2-sha8:'); print(FEATURE_RECIPE_VERSION)"` exits 0
- [ ] `python -c "from src.features.mr_features import MR_SCHEMA_HASH; assert MR_SCHEMA_HASH.startswith('v2-sha8:'); print(MR_SCHEMA_HASH)"` exits 0
- [ ] `pytest tests/test_schema_hash.py -v` — all 10 tests green
- [ ] `pytest tests/test_serve_resilience.py -v` — both tests green
- [ ] `pytest tests/ -x -q` — full suite (158 existing + 10 new = ~168 tests) green with no regressions
- [ ] `FEATURE_RECIPE_VERSION != "v1.1"` (verified by test 10 in `test_schema_hash.py`)

---

## Acceptance Criteria

1. `src/utils/schema_hash.py` exists and exports `compute_feature_schema_hash`.
2. `FEATURE_RECIPE_VERSION` in `src/backtest/pipeline.py` is computed, not a string
   literal, and its value matches `"v2-sha8:" + 8_hex_chars`.
3. `FEATURE_SCHEMA` is defined in `src/backtest/pipeline.py` with exactly 15 entries
   matching the documented specification.
4. `build_features()` contains an assertion that fires if `all_features` names drift
   from `FEATURE_SCHEMA`.
5. `MR_SCHEMA_HASH` is defined in `src/features/mr_features.py`.
6. `mr_threshold.json` (post-retrain) contains `"schema_hash"` key.
7. `checkpoint["metadata"]` (post-retrain) contains `"feature_schema_hash"` key.
8. `tests/test_schema_hash.py` exists with 10 passing tests.
9. Full pytest suite is green with no regressions.
10. No modification to `main._load_v3_bot()` tripwire logic.

---

## Resume and Execution Handoff

**Plan path:** `process/features/v4-1-structural-debt/active/phase2-feature-schema-hashing_PLAN_10-06-26.md`

**Phase context:** This is Phase 2 of the V4.1 Structural Debt program. Phase 1
(daily_inference decomposition) is COMPLETE in `process/features/v4-1-structural-debt/completed/`.

**Execution order:** Steps 1–9 must be executed in order. Steps 1–5 and 9 are
code-only (fast, CI-testable). Steps 6–7 require live ML artifacts and are manual
smoke tests. Step 8 is a test-file update.

**Recommended execution sequence for EXECUTE:**
1. Steps 1–5 first (pure code — hash utility, schema constant, assertion, MR constant)
2. Step 9 (new test file — now runnable against steps 1–5 completions)
3. Step 8 (update existing test — fast)
4. Run `pytest tests/ -x -q` to confirm green
5. Steps 6–7 (training artifact stamps — manual, requires live data)

**After EXECUTE completes:**
- Archive this plan to `process/features/v4-1-structural-debt/completed/`
- Update `process/context/all-context.md` Phase 2 status to COMPLETE
- Advance Phase 3 (Hub-node test coverage) by checking backlog

**Validator command:**
```
pytest tests/test_schema_hash.py -v && pytest tests/ -x -q
```
