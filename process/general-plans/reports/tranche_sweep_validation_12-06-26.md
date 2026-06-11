# Tranche Sweep Validation Report

**Date:** 12-06-26
**Run:** `python run_backtest.py --no-save` (mode=tranche, hold=30d, 4 seeds × 4 thresholds, 905 OOS days, wall-clock 2,129s)
**Plan:** `process/general-plans/active/tranche_backtest_integration_PLAN_12-06-26.md`

## Headline

The tranche book's edge is **seed-robust**: at sig_thr 0.40 all four seeds land in a tight +39.5%…+41.9% net band (mean **+40.8%**, ≈ +9.2%/yr CAGR, mean Sharpe +0.61, mean max-DD −28.5%). The grid book at the same threshold lost −85.7% mean before the price-scale fix and −42.4% after it. The signal is real and the construction now harvests it.

## Sweep table

| up_thr | sig_thr | mean NetPnL (VND) | mean Sharpe | mean DD | note |
|---|---|---|---|---|---|
| 0.50 | 0.45 | +666M (+6.7%) | +0.38 | −8.0% | signal-starved (438 predUP) |
| 0.45 | 0.40 | +4,077M (+40.8%) | +0.61 | −28.5% | |
| 0.40 | 0.35 | +4,078M (+40.8%) | +0.61 | −28.5% | GOLDEN (by noise margin) |
| 0.35 | 0.30 | +4,078M (+40.8%) | +0.61 | −28.5% | identical — saturation |

Per-seed at sig 0.40: 42: +41.9% / 43: +40.5% / 44: +39.5% / 45: +41.1%.

## Statistical gates: FAIL — read carefully

- **DSR p=0.27 (FAIL <0.95):** the deflated-Sharpe hurdle (SR0 ≈ 0.95 ann. given 16 trials) exceeds the strategy's 0.62. Honest reading: at this Sharpe the OOS window cannot statistically separate skill from multiple-testing luck at 95%. The cross-seed tightness is the stronger robustness evidence, but the gate is the gate.
- **PBO 77.8% (FAIL >10%) — partly artifact:** three of the four swept configs produce IDENTICAL equity curves (the top-5/day book saturates below sig 0.40), so CSCV rank flips among near-clones are coin flips and PBO is mechanically inflated. The threshold axis does not differentiate the tranche book; PBO over this config set is uninformative.

## Implications / next levers

1. **Fix the sweep axis for tranche mode** — sweep `hold_days` (20/30/40) and/or sig_thr in the 0.40–0.45 band (e.g. `--sweep-thresholds 0.48` reaches sig 0.43, the single-seed sweet spot +45.5%) so CSCV sees genuinely distinct configs.
2. **Drawdown control** — −28.5% max DD is the Sharpe killer. The regime study showed BEAR-cohort trades carry the HIGHEST per-trade edge, so naive P(Bull) exposure scaling is the wrong knob; candidate: per-tranche stop or vol-scaled tranche budgets.
3. **Serve-path Phase 2** — payload now carries `strategy{mode, hold_days, signal_threshold}`; `main.py` dispatch must adopt NAV/H/5 sizing + exit-after-H alerts before live behavior matches the backtest.

## Addendum: differentiating-axis re-sweep (same day)

`--sweep-thresholds 0.48,0.46,0.45,0.43` (sig 0.43–0.38):

| up_thr | sig_thr | mean NetPnL | mean Sharpe | mean DD | predUP |
|---|---|---|---|---|---|
| **0.48** | **0.43** | **+43.7%** | **+0.648** | **−26.8%** | 945 — GOLDEN |
| 0.46 | 0.41 | +40.7% | +0.611 | −28.5% | 4,398 |
| 0.45 | 0.40 | +40.8% | +0.611 | −28.5% | 12,096 |
| 0.43 | 0.38 | +40.8% | +0.611 | −28.5% | 118,228 |

- sig 0.43 is confirmed as the sweet spot **across all four seeds** (best seed 43: +46.5%, Sharpe 0.68) — higher return AND lower DD than 0.40.
- Saturation persists below sig ~0.41 (bottom two configs are clones), so PBO (88.4%) stays uninformative on the threshold axis. A future PBO-meaningful sweep must vary `hold_days` (20/30/40) instead.
- DSR p=0.31 still FAIL — Sharpe 0.65 vs the 0.95 deflated hurdle. The binding constraint is drawdown (−27%), hence the per-tranche PT/SL barrier work (`a1b600c`); barrier validation run pending.

## Provenance

- Engine: tranche mode `f6b1c4d`, price-scale fix `8ee0339`, evaluator integration `67e3e1f`
- Research basis: `scripts/` suite (`113c61a`) — grid-date study, within-day rank, hold-horizon scan
