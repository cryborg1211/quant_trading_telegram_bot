# T+5 τ=0.42 Sandbox Sweep — Exploratory, No Action

- **Date:** 06-07-26
- **Type:** Exploratory sandbox trial (NOT a pre-registered A/B, no code/config change)
- **Command:** `python run_backtest.py --mode tranche --hold-days 30 --no-save` (T+5 checkpoint, threshold sweep including 0.42)
- **Log:** `logs/backtest_t5_tau042_20260706.log` (no artifacts written — `--no-save`)
- **Checkpoint:** T+5 (`v3_ensemble_5d.joblib` / `v3_training_checkpoint.joblib`), 4 seeds [42, 43, 44, 45]

## Motivation

Morning question (06-07-26): does a τ=0.42 admission threshold sit in a
meaningfully different part of the T+5 threshold surface than the existing
sweep grid (which brackets 0.40/0.45 but does not sample 0.42 directly)? This
was a one-off curiosity check, not a proposal to change any deployed
threshold. No pre-registration was filed because this was explicitly scoped
as read-only exploration of an already-frozen artifact (the current T+5
checkpoint), not a new trial that could influence any live decision.

Related, more consequential context already on record: the
`attack-narrow-market-preregistration_05-07-26.md` backlog doc and the
04-07-26 admission A/B (`serve-admission-ab-result_04-07-26.md`) both bear on
whether absolute-threshold admission gates cost real Sharpe/Net during
narrow-leadership months. This sandbox run is a smaller, single-horizon,
single-session echo of that same question, using the sweep mechanism already
built into `run_backtest.py` rather than the dedicated admission-mode A/B
harness from the 04-07-26 plan.

## Result table (4-seed means, tranche hold=30)

| up_thr | sig_thr | mean NetPnL (VND) | mean Sharpe | mean DD | total predUP | mean UP prec |
|---|---|---|---|---|---|---|
| 0.50 | 0.45 | +2,629,294,153 | +0.592 | −13.47% | 15,676 | 0.5582 |
| 0.45 | 0.40 | +2,818,603,884 | +0.629 | −13.00% | 218,361 | 0.4648 |
| **0.42** | **0.37** | **+2,905,539,059** | **+0.644** | **−13.00%** | 518,921 | 0.4538 (tagged GOLDEN by tie-break only) |
| 0.40 | 0.35 | +2,905,539,059 | +0.644 | −13.00% | 718,281 | 0.4468 |
| 0.35 | 0.30 | +2,905,539,059 | +0.644 | −13.00% | 1,039,723 | 0.4346 |

Per-seed rows for 0.40 and 0.35 are BYTE-IDENTICAL to the 0.42 row on
NetPnL/Sharpe/DD (only `predUP`/`mean_UPprec` differ, since those are pure
counting artifacts of how many names clear a lower absolute floor before the
top-5/day cross-sectional cap binds). This confirms **saturation starts at
0.42**: below 0.42, the tranche engine's top-5-per-day admission is the
binding constraint, not the absolute τ floor — so the floor value itself
stops mattering once it is low enough to admit at least 5 eligible names most
days.

## Teardown (seed=45, "GOLDEN" tie-break row = 0.42/0.37)

| Metric | Value | Gate | Result |
|---|---|---|---|
| Net PnL | +30.33% | — | — |
| Sharpe | +0.668 | — | — |
| Max Drawdown | −12.62% | — | — |
| DSR p-value | 0.2686 | ≥0.95 for live | FAIL |
| PBO (CSCV) | 85.6% | ≤10% | FAIL (N=4 seed-configs, T=45 months) |

**85.6% PBO is the SAME reading as the 04-07-26 admission A/B's winning
config** (also 85.6%, also N=4, T≈45 months) — this is chronic T+5
seed/threshold-selection instability already on record, not new damage from
this sandbox run. No new statistical finding here beyond confirming the
existing diagnosis reproduces on a slightly different threshold slice.

## Two findings worth keeping on record

1. **Saturation finding.** τ=0.42 sits inside the already-known saturation
   band (0.35–0.42 are numerically identical on the metrics that matter).
   It catches neither incremental edge nor incremental garbage relative to
   the existing 0.40 GOLDEN — same book, same risk profile, differing only
   in cosmetic prediction-count/precision bookkeeping. There is no reason to
   prefer 0.42 over the existing 0.40 GOLDEN threshold on this evidence.
2. **Argmax-proxy finding (the more interesting one).** The 0.50/0.45 row is
   the first backtest-level proxy evidence AGAINST the serve arbitrator's
   argmax-based BUY gate as currently framed: raising the absolute floor to
   0.50 does raise precision (0.558 vs 0.454 at 0.42), but Sharpe, Net PnL,
   and DD are ALL worse, and participation drops by roughly 46x (15,676 vs
   718,281+ predicted-UP rows across the sweep). A stricter, "more confident"
   gate looks superficially safer (higher precision) while being worse on
   every risk-adjusted metric that actually matters for capital allocation —
   consistent with the "good armor, bad architecture" framing from this
   session's graduated-attack direction debate (see orchestrator memory).
   This is a proxy result on a T+5 sandbox sweep, not a direct test of the
   live arbitrator's argmax mechanism — treat as suggestive, not conclusive.

## Verdict: NO ACTION

- No config change, no code change, no artifact written to `models/saved/`.
- τ stays at the existing GOLDEN values (T+5 unaffected by this run; T+20 is
  the primary horizon and was not touched at all in this sandbox).
- This trial counts toward the repo's general trial-count discipline as ONE
  exploratory, non-pre-registered look — it should not be treated as
  grounds for a future PASS/FAIL claim on any threshold near 0.42, since no
  criteria were frozen in advance of running it.

## Links

- `process/general-plans/backlog/attack-narrow-market-preregistration_05-07-26.md`
  — the broader narrow-leadership attack plan this sits adjacent to.
- `process/general-plans/reports/serve-admission-ab-result_04-07-26.md` —
  the 04-07-26 admission A/B this sandbox run's PBO/DSR readings echo.
- `process/general-plans/completed/completed_serve-admission-tranche-ab_PLAN_04-07-26.md`
  — the plan that produced the above A/B (archived this session).

## Deferred / follow-ups

- None. This was a closed-loop curiosity check; no open threads spawned.
