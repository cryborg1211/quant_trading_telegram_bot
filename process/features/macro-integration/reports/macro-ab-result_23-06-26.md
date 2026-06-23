# Macro-Integration P4 — A/B Result

**Date:** 2026-06-23
**Verdict:** ❌ **KILL the GBM macro features.** Keep `use_macro_features=False` (default).
**Keep:** the P2 macro-aware regime HMM overlay (the real win).

---

## Setup

Two full 4-seed checkpoints trained on the same panel, identical config except the
GBM-feature flag:

- `models/saved/ab_baseline.joblib` — `use_macro_features=False` (recipe `v2-sha8:53b5bd85`)
- `models/saved/ab_macro.joblib` — `use_macro_features=True` (3 macro cols joined; 16 features selected vs 13)

Both arms share the P2 macro-HMM overlay (regime observation widened to 4 dims:
`market_ret + sp500_ret + dxy_ret + usdvnd_ret`) — so only the GBM features differ.

Backtest: `run_backtest.py --no-save --sweep-thresholds 0.43`, tranche mode, hold 30d.

## Results

| Metric | Baseline (OFF) | Macro-GBM (ON) |
|---|---|---|
| mean Net PnL | +3,145,071,397 | +3,322,323,481 |
| mean Sharpe | +0.682 | +0.703 |
| mean MaxDD | **−13.03%** | **−18.47%** |
| Net Sharpe (ann.) | +0.732 | +0.749 |
| Deflated Sharpe (SR / SR0) | 0.732 / 0.554 | 0.749 / 0.573 |
| DSR p-value | 0.630 (FAIL <0.95) | 0.624 (FAIL <0.95) |
| PBO (CSCV) | 42.7% (FAIL >10%) | **87.0% (FAIL >10%)** |
| total predUP | 178,418 | 159,680 |
| UP-precision | 0.4463 | 0.4574 |

## Analysis

- **Net / Sharpe gain is noise.** +5.6% net, +0.02 Sharpe — inside the per-seed
  spread (baseline seeds alone: Sharpe 0.64–0.73).
- **Drawdown got worse:** −13.0% → −18.5% (+5.4pp).
- **Overfitting roughly doubled:** PBO 42.7% → 87.0%. Adding market-level features
  let the GBMs fit spurious market-wide patterns.
- **DSR did not improve** (both fail; macro marginally worse).

This matches the V4 design rationale (`src/backtest/pipeline.py:415`): macro is
market-level (identical across the cross-section each day), so it carries ~0
*ranking* signal for a cross-sectional GBM. Forcing it in adds overfitting and
drawdown without alpha.

### Caveat
Cutoff drift confounds the comparison: baseline OOS T=45 months (cutoff
2022-10-26) vs macro T=42 months (cutoff 2023-01-09) — macro warm-up trimmed
early rows, moving the `train_frac` split. The windows mostly overlap; the
PBO 87% + worse DD are decisive enough to override the confound. A perfectly
controlled redo would pin an aligned `--start-date` for both arms.

## Decision

- `RunConfig.use_macro_features` stays **default False** (P3's default is correct).
- Recipe `v2-sha8:53b5bd85` unchanged → serve path + existing artifacts untouched.
- **Keep P2** (`use_macro_in_hmm`): the macro-aware regime overlay is the
  program's real value (tighter, more risk-aware exposure scaling).
- A/B checkpoints (`ab_baseline.joblib`, `ab_macro.joblib`) are scratch — safe to delete.
