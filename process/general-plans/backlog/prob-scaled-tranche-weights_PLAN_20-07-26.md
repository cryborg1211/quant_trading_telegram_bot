# Probability-Scaled Tranche Cohort Weights — Backlog Plan

**Date**: 20-07-26
**Status**: 📋 BACKLOG (design ready, NOT started — requires backtest A/B before any serve change)
**Origin**: July-2026 loss post-mortem optimization list, item #5 (user-approved direction, deferred from the 20-07 autopilot session for scope discipline).

## Problem

Tranche cohorts equal-weight every pick: weight = 1/(hold_days × n_picks). A pick at
P(UP)=0.62 gets the same capital as one at 0.46. Calibrated probabilities carry edge
information the sizing throws away.

## Proposal

Scale within-cohort weights by normalized calibrated edge:

```
edge_i   = max(0, p_up_i − up_threshold)
w_i      = base_cohort_weight × n_picks × edge_i / Σ edge_j   (fallback: equal weight when Σ=0)
```

Cap per-name at 2× equal-weight to bound concentration (the July lesson — do NOT let
probability scaling re-concentrate the book).

## Touchpoints (parity is mandatory)

1. `src/backtest/walk_forward.py` — tranche pick weighting (opt-in flag `WalkForwardConfig.use_prob_weights`, default False).
2. `run_backtest.py` — CLI `--prob-weights` for the A/B; teardown report must print which mode ran.
3. `main.py` `_tranche_signal_fields` / `_dispatch_signals` — serve-side same formula, gated by `CONFIG.trading.prob_weighted_cohorts_enabled` (default False until A/B passes).
4. `src/bot/sizing.py` — respect the cap interaction with Kelly/NAV caps.
5. Tests: pure weight-function unit tests + parity test (backtest formula == serve formula on same inputs).

## Acceptance gate

A/B on the current T+20 GOLDEN checkpoint (4 seeds): ship-eligible only if Sharpe improves
AND MaxDD does not worsen by >1pp AND PBO does not worsen. Same standard as the
regime-sizing A/B (14-06-26).

## Notes

- Extract ONE pure function `prob_scaled_weights(p_ups, up_threshold, cap_mult=2.0)` shared by both paths — no formula duplication (regime_policy.py precedent).
- Interaction with the 19-07 drift brake + GARCH brake: scalars multiply AFTER cohort weights — orthogonal, no change needed.
