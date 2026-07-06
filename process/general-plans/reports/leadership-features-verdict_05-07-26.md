# Leadership-Features Recipe — Verdict: FAIL (PBO gate)

- **Date:** 05-07-26 (retrain 23:14–23:49, backtest 23:49–00:27 ICT; verdict applied 06-07-26 by the orchestrator after the execute-agent hit its session limit at the verdict-writing step)
- **Plan:** `process/general-plans/active/leadership-features-recipe_PLAN_05-07-26.md`
- **Run:** `python train_models.py --tb-horizon 20` → `python run_backtest.py --mode tranche --hold-days 30` (T+20 only per plan Section 9; T+5 untouched)
- **Feature set (frozen Section 4):** `updown_ratio_20_xsz`, `efficiency_ratio_20_xsz`, `dist_from_252high_xsz`, `rs_rank_stability_10_xsz` — separate candidate sub-pool, `top_k` 3→4
- **Logs:** `logs/retrain_leadership_20260705.log`, `logs/backtest_leadership_20260705.log`
- **Code state at evaluation:** full pytest suite green (552 passed, incl. 6 new leak/range tests)

## Threshold sweep (new recipe, 4-seed means)

| up_thr | sig_thr | mean NetPnL (VND) | mean Sharpe | mean DD | total predUP | mean UP prec |
|---|---|---|---|---|---|---|
| 0.50 | 0.45 | +1,239,031,317 | +0.443 | −9.97% | 310 | 0.5967 |
| 0.45 | 0.40 | +3,062,754,066 | +0.666 | −13.11% | 34,380 | 0.4910 |
| **0.40** | **0.35** | **+3,107,039,613** | **+0.673** | **−13.11%** | 645,648 | 0.4313 — **GOLDEN** |
| 0.35 | 0.30 | +3,107,039,613 | +0.673 | −13.11% | 1,111,575 | 0.4136 |

Note: GOLDEN moved from the old recipe's 0.45/0.40 to 0.40/0.35. Sweep is
saturated at the bottom two rows (identical numbers) — same saturation shape
as the 02-07 baseline sweep.

## Teardown (GOLDEN, seed=45) vs 02-07 baseline

| Metric | New recipe | 02-07 baseline | Gate (frozen §7) | Result |
|---|---|---|---|---|
| Net PnL | **+34.80%** | +32.55% | — | better |
| Sharpe (teardown seed-45) | **+0.732** | +0.689 | > +0.689 | ✓ (see note) |
| Sharpe (4-seed mean @GOLDEN) | +0.673 | +0.573 | — | better |
| Max Drawdown | −13.22% | −13.16% | ≥ −14.16% | ✓ |
| DSR p-value | 0.3448 (FAIL <0.95) | 0.3146 (FAIL) | reported, not gated | — |
| **PBO (CSCV)** | **18.9%** | **3.0%** | **≤ 10%** | **✗ FAIL** |

**Criterion-1 baseline-type note (honesty):** the plan's frozen text compared
"4-seed mean Sharpe > +0.689", but +0.689 is the baseline's *seed-45 teardown*
number (the baseline 4-seed mean was +0.573). Under either apples-to-apples
reading — teardown vs teardown (0.732 > 0.689) or mean vs mean (0.673 > 0.573)
— criterion 1 passes. Recorded because the letter of the frozen text (mean
0.673 vs 0.689) reads as a miss; the mismatch is a baseline-type labeling
error in the plan, not a renegotiation. **Immaterial to the verdict: PBO
fails regardless.**

## Verdict — FAIL, mechanically applied

Section 7: "Any single miss → FAIL." **PBO 18.9% > 10%.** The new features
raise raw performance (+2.25pp Net, +0.10 mean Sharpe, DD flat) but config
selection became six times less robust across CSCV resamples than the
incumbent (3.0% → 18.9%). The Sharpe gain is not trustworthy if the config
choice itself doesn't survive resampling. Both sweeps share the same
saturated-grid shape, so the PBO deterioration is relative and real, not a
mechanical artifact of a different grid.

**Actions taken per the FAIL path (plan §7/§12.11):**
- Working-tree code changes reverted (pipeline.py, train_models.py, main.py
  4-tuple unpack, test fixture updates; `leadership_features.py` +
  `test_leadership_features.py` removed — every formula is preserved verbatim
  in the plan file for future re-registration).
- `models/saved/v3_ensemble_20d.joblib`, `v3_training_checkpoint.joblib`,
  `prob_distribution.png` restored to the 02-07 GOLDEN versions (git-tracked;
  the run's own pre-overwrite backup also exists:
  `models/saved/backups/v3_ensemble_20d_20260705T172738Z.joblib`).
- Serve never at risk: recipe-hash gate + no serve deploy during the trial.

## Diagnostics not captured

June-2026 monthly slice (plan §6 step 6, non-gating): the execute-agent died
at the session limit before extracting it and the backtest log does not
contain the monthly table. Omitted rather than reconstructed.

## What a future attempt needs (NEW pre-registration required — no retry under this plan)

1. **De-saturated sweep grid.** Both this and the baseline sweep saturate in
   the bottom rows; `tranche_sweep_validation_12-06-26.md` already flagged
   that saturated sweeps mechanically inflate PBO. A grid differentiated on a
   non-threshold axis would make the PBO reading cleaner for BOTH arms.
2. **Fewer/cheaper candidates.** 4 new candidates + top_k 3→4 doubles the
   effective selection surface; a 2-feature shot (e.g. `updown_ratio_20` +
   `dist_from_252high` only, top_k unchanged) is a smaller trial with less
   selection freedom to overfit.
3. Item 1 (paperlog rank-sleeve counterfactual, matures from ~07-07) may
   independently show whether leadership-persistence has forward edge before
   more GPU is spent.

Item 2 (breadth-conditional sleeve, sibling backlog) remains gated — its
precondition was "this plan's checkpoint (PASS or FAIL)": verdict is FAIL,
incumbent 02-07 GOLDEN stays the reference checkpoint.
