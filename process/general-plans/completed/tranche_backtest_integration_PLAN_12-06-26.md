# Tranche Mode → run_backtest.py Integration - Plan

**Date:** 12-06-26
**Complexity:** Simple
**Status:** ✅ COMPLETE — implemented in `67e3e1f`; full-sweep validation recorded in `process/general-plans/reports/tranche_sweep_validation_12-06-26.md`

## Overview

Wire the validated tranche rebalance mode (`WalkForwardEngine.rebalance_mode="tranche"`, commit `f6b1c4d`) into the V4.0 Fast Evaluator (`run_backtest.py`) so the staggered-tranche strategy gets the full multi-seed threshold sweep, Deflated Sharpe, and PBO (CSCV) treatment that currently only the legacy grid book receives. Persist the winning tranche parameters in the bot payload (additive, serve-path consumption is explicitly Phase 2).

**Motivating evidence (single seed 42, full VN cost model, 985 OOS days):**

| Config | Net PnL | Gross/trip | Turnover |
|---|---|---|---|
| Grid (legacy) | −36.7% | −0.12% | 7.4x |
| Tranche sig_thr 0.43 / H=30 | **+45.5%** | +2.84% | 5.7x |

Per-day net edge peaks at H=30 (≈0.098%/day, scratch/hold_horizon_check.py). The +45.5% figure is seed-42-only — multi-seed validation with DSR/PBO gates is exactly what this integration provides.

## Goals and Success Metrics

**Goals:**
- `run_backtest.py` runs the threshold sweep in tranche mode by default (grid stays as `--mode grid` escape hatch)
- GOLDEN selection, signal evaluation, DSR, and PBO operate on tranche equity curves unchanged
- Bot payload carries the tranche strategy parameters additively
- Zero behavior change for `--mode grid`

**Success Metrics:**
- Full sweep (4 seeds × 4 thresholds) completes and prints a GOLDEN tranche config with mode + hold days in the banner
- `--mode grid` output matches current behavior (same code path, params untouched)
- All existing tests green + new unit tests for config propagation

---

## Execution Brief

**IMPORTANT:** This is a SIMPLE (one-session) plan — implement continuously without approval gates.

### Phase 1: run_oos parameter plumbing
`run_oos()` gains `mode: str` and `hold_days: int` parameters, forwarded into `WalkForwardConfig(rebalance_mode=..., tranche_hold_days=...)`. Extract the `WalkForwardConfig` construction into a pure helper `_build_wf_config(...)` so it is unit-testable without running the engine.

### Phase 2: CLI + sweep loop
`--mode {tranche,grid}` (default `tranche`), `--hold-days` (default 30). Sweep loop passes both through. Per-seed inference caches already make the threshold sweep cheap after the first threshold (oracle scoring is threshold-independent; tranche mode scores daily so the FIRST threshold per seed pays ~905 scorings).

### Phase 3: Reporting + payload
Teardown report and GOLDEN banner print `mode` and `hold_days`. `_persist_bot_payload` adds an additive `"strategy": {"mode", "hold_days", "signal_threshold"}` dict — verify `main._load_v3_bot` tolerates unknown payload keys before relying on it (it should: payload is a plain dict; the loader reads specific keys).

### Phase 4: Tests + context
`tests/test_run_backtest_config.py`: `_build_wf_config` propagates tranche fields; grid default regression; payload strategy dict shape. Update `process/context/all-context.md` current-state notes (two rebalance modes; tranche is the evaluator default).

### Post-Implementation Testing

1. **Unit:** `python -m pytest tests/ -q` — all green (182 existing + new)
2. **Smoke (fast):** `python run_backtest.py --no-save --sweep-thresholds 0.48 --max-positions 5` — single-threshold tranche run completes, banner shows mode/hold
3. **Full validation (long, ~1.5–2h):** `python run_backtest.py --no-save` — 4 seeds × 4 thresholds; record mean NetPnL per config, DSR p-value, PBO
4. **Legacy regression:** `python run_backtest.py --no-save --mode grid --sweep-thresholds 0.50` — grid path unchanged

### Expected Outcome
- Tranche GOLDEN config with multi-seed mean NetPnL > 0 (single-seed evidence: +45.5% at 0.43/H=30)
- DSR/PBO verdict on the tranche strategy — the honest production gate
- If multi-seed results collapse vs seed 42, that is a finding, not a failure of this integration

---

## Scope

**In-Scope:**
- `run_backtest.py`: CLI flags, `run_oos` plumbing, `_build_wf_config` extraction, banner/report labels, additive payload `strategy` dict
- New unit tests for config propagation
- `process/context/all-context.md` current-state note

**Out-of-Scope (Phase 2 follow-up):**
- Serve-path changes: `main.py` dispatch consuming the strategy dict (NAV/H/5 sizing, exit-after-H alerts, tranche ledger)
- Retraining, label-convention changes, new sweep axes (hold-days sweep)
- `vc_cost_model` / `walk_forward.py` changes (engine is done, commit `f6b1c4d`)

## Assumptions and Constraints

- Checkpoint schema untouched — mode/hold-days are CLI-only, no `RunConfig` field added (old checkpoints keep loading)
- Sweep threshold → sig_thr mapping (`sig_thr = thr − 0.05`) unchanged so UP-precision reporting stays comparable
- CSCV/PBO math is config-count-based (N=4) — unchanged
- Full tranche sweep wall-clock ~1.5–2h (daily inference × 4 seeds, then cache hits); document in module docstring

## Acceptance Criteria

1. ✅ `run_backtest.py --no-save` defaults to tranche mode, H=30, completes the sweep
2. ✅ GOLDEN banner + teardown show `mode=tranche hold=30`
3. ✅ DSR + PBO computed on the tranche GOLDEN equity curve
4. ✅ `--mode grid` reproduces legacy behavior byte-for-byte in config construction
5. ✅ Payload contains `strategy` dict; existing loader unaffected
6. ✅ Full pytest suite green

## Implementation Checklist

1. Extract `_build_wf_config(cfg, features, cutoff, sig_thr, mode, hold_days)` from `run_oos`; add `mode`/`hold_days` params to `run_oos`
2. Add `--mode` / `--hold-days` argparse flags; thread through `main()` → sweep loop → `run_oos`
3. Label sweep phase logs, GOLDEN banner, and teardown report with mode + hold days
4. Add `strategy` dict to `_persist_bot_payload`; confirm `main._load_v3_bot` ignores unknown keys
5. Write `tests/test_run_backtest_config.py` (tranche propagation, grid regression, payload shape)
6. Update `process/context/all-context.md` (two rebalance modes, evaluator default)
7. Run smoke + full validation; record sweep report in `process/general-plans/reports/`

## Risks and Mitigations

**Risk 1:** Full sweep wall-clock (~2h) blocks iteration
- **Mitigation:** `--sweep-thresholds` single-value runs for smoke; document expected runtime

**Risk 2:** Payload `strategy` key breaks the live loader
- **Mitigation:** additive-only; grep `_load_v3_bot` required-key validation before merging; covered by payload-shape test

**Risk 3:** Multi-seed results materially below seed 42
- **Mitigation:** that is the point of the run — report honestly; per-row evidence says edge is seed-independent (scores are ensemble-averaged), but selection variance is real

## Resume Handoff

- Engine feature complete at `f6b1c4d`; this plan only touches `run_backtest.py`, tests, context
- Validation tooling: `scripts/engine_pnl_attribution.py --tranche --hold-days N` for single-seed deep dives
- Per-row reference numbers: `scratch/hold_horizon_check.py` (H=30 net/day peak), `scripts/within_day_rank_check.py` (top-5 +1.60% net at H=20)
