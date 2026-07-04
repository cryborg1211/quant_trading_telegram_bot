# T+5 / T+20 Retrain Result (01-07-26 → 02-07-26)

**Date:** 02-07-26
**Run:** `python train_models.py --tb-horizon {20,5}` → `python run_backtest.py --mode tranche --hold-days 30` (per horizon, 4 seeds, full GOLDEN threshold sweep)
**Recipe:** `v2-sha8:53b5bd85` (unchanged — refresh with more recent data, not a feature-engineering change)
**Artifacts updated (uncommitted):** `models/saved/v3_ensemble_20d.joblib`, `models/saved/v3_ensemble_5d.joblib`, `models/saved/v3_training_checkpoint.joblib`, `models/saved/prob_distribution.png`

## Headline

Both horizons land in the same place as every prior retrain: **solid raw returns, statistically ungated.** T+20 passes the overfit check (PBO 3.0%) but still fails the deflated-Sharpe hurdle. T+5 fails BOTH gates — PBO jumped to 43.1%, the worst overfit reading either horizon has produced. Both stay paper-only.

## T+20

### Threshold sweep

| up_thr | sig_thr | seeds | mean NetPnL (VND) | mean Sharpe | mean DD | total predUP | mean UP prec |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.45 | 4 | +44,857,288 | +0.098 | −3.29% | 9 | 0.3500 |
| **0.45** | **0.40** | 4 | **+2,570,309,194** | **+0.573** | **−13.69%** | 6,192 | 0.5348 — **GOLDEN** |
| 0.40 | 0.35 | 4 | +2,568,439,139 | +0.573 | −13.69% | 622,869 | 0.4243 |
| 0.35 | 0.30 | 4 | +2,568,439,139 | +0.573 | −13.69% | 1,173,898 | 0.4073 |

### OOS teardown (GOLDEN, seed=45)

| Metric | Value |
|---|---|
| OOS trading days | 911 |
| Initial capital | 10,000,000,000 VND |
| Final NAV | 13,254,901,121 VND |
| Total Net PnL | **+3,254,901,121 VND (+32.55%)** |
| Net Sharpe (ann.) | **+0.689** |
| Max Drawdown | **−13.16%** |
| Deflated Sharpe | SR=+0.689, SR0=+0.947 |
| **DSR p-value** | **0.3146 — FAIL (<0.95)** |
| **PBO (CSCV)** | **3.0% — PASS (≤10%)** [T=45mo, N=4 configs] |
| **Verdict** | **✗ UNFIT FOR PRODUCTION** |
| Wall-clock | 2088.0s |

## T+5

### Threshold sweep

| up_thr | sig_thr | seeds | mean NetPnL (VND) | mean Sharpe | mean DD | total predUP | mean UP prec |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.45 | 4 | +2,730,776,332 | +0.611 | −13.34% | 15,547 | 0.5573 |
| 0.45 | 0.40 | 4 | +2,891,814,877 | +0.642 | −12.99% | 217,415 | 0.4645 |
| **0.40** | **0.35** | 4 | **+2,973,740,007** | **+0.657** | **−12.99%** | 715,061 | 0.4471 — **GOLDEN** |
| 0.35 | 0.30 | 4 | +2,973,740,007 | +0.657 | −12.99% | 1,035,162 | 0.4350 |

### OOS teardown (GOLDEN, seed=45)

| Metric | Value |
|---|---|
| OOS trading days | 911 |
| Initial capital | 10,000,000,000 VND |
| Final NAV | 13,207,706,532 VND |
| Total Net PnL | **+3,207,706,532 VND (+32.08%)** |
| Net Sharpe (ann.) | **+0.700** |
| Max Drawdown | **−12.62%** |
| Deflated Sharpe | SR=+0.700, SR0=+0.947 |
| **DSR p-value** | **0.3228 — FAIL (<0.95)** |
| **PBO (CSCV)** | **43.1% — FAIL (>10%)** [T=45mo, N=4 configs] |
| **Verdict** | **✗ UNFIT FOR PRODUCTION** |
| Wall-clock | 2007.4s |

## T+20 vs T+5

| | T+20 | T+5 |
|---|---|---|
| Net PnL | +32.55% | +32.08% |
| Sharpe | +0.689 | +0.700 |
| Max DD | −13.16% | −12.62% |
| DSR p | 0.3146 (FAIL) | 0.3228 (FAIL) |
| PBO | **3.0% (PASS)** | **43.1% (FAIL)** |

Return/Sharpe/DD are near-identical between horizons — not surprising, both saturate the top-5/day tranche book at similar thresholds. The real divergence is PBO: T+20's config selection is robust across the CSCV resample, T+5's is not. Read T+5's GOLDEN config with more skepticism than T+20's until a re-sweep on a differentiating axis (see prior finding in `tranche_sweep_validation_12-06-26.md` — saturated threshold sweeps mechanically inflate PBO; same likely applies here, not necessarily "real" 43% overfit).

## Notes

- **GPU confirmed engaged**, not just configured: `nvidia-smi` showed the live training PID at ~25% util with real VRAM allocated on the RTX 3050 4GB. All 3 base learners are GPU-wired (XGBoost `device="cuda"`, LightGBM `device_type="gpu"`, CatBoost `task_type="GPU"` — `tabular_ensemble.py`). 4GB VRAM forces sequential (not parallel) seed×model fits; dataset materialization stays CPU/Polars-only — wall-clock is GPU-fit-time plus a fixed CPU-bound data-prep floor.
- T+5's run was interrupted mid-training on 01-07-26 (laptop shutdown) and cleanly resumed 02-07-26 from scratch (`train_models.py --tb-horizon 5` → `run_backtest.py`) — no partial/corrupt artifact was ever written; the old `v3_ensemble_5d.joblib` stayed in place until this run's clean completion.
- Both `.joblib` artifacts + checkpoint + `prob_distribution.png` are updated on disk but **not committed**.
