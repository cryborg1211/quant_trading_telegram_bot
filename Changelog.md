# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## 2026-06-24

### GARCH-HMM macro-regime overlay — new risk layer
- Added `src/models/garch_hmm_regime.py`: GARCH(1,1) conditional volatility
  fused with a multi-dimensional Gaussian HMM into a leak-free market-exposure
  scaler. 5-D emission matrix `[market_ret, sp500_ret, dxy_ret, usdvnd_ret,
  log(σₜ)]` → N-state HMM → `exposure_scaler = clip(P(Bull), 0.2, 1.0)`.
- **Numerical-stability engineering** (the bits that make it converge on real
  data):
  - **Log-vol space** + 99th-pctl winsorize — raw σₜ is right-skewed and
    random-walks under IGARCH, violating the Gaussian-emission assumption.
  - **Persistence guard** — post-fit projection caps α+β ≤ 0.96 (escapes the
    IGARCH trap) by rescaling (α, β) and recomputing ω to preserve the
    unconditional variance.
  - Per-column z-score + degenerate-fit rejection + multi-restart EM.
- Added scripts: `train_macro_regime.py` (fit + serialize weights),
  `scripts/validate_garch_hmm_brake.py` (A/B vs baseline),
  `scripts/sweep_garch_hmm_brake.py` (floor × persistence robustness grid),
  `scripts/walk_forward_macro_pipeline.py` (rolling-window T+5/T+20 diagnostic).
- **OOS (915d, T+5, seed 0):** Sharpe −0.36 → −0.15, MaxDD −55% → −39%,
  Return −38% → −15%. Validated **loss-mitigation** (bear OOS window), not
  standalone alpha. Verdict KEEP; robustness sweep in progress.
- New dependency: `arch==8.0.0`.
- Added `README.md` (GitHub-facing) and `tests/test_garch_hmm_regime.py` (26).
- Test suite: 369 → **395 passed**.

## 2026-06-21

### V4.1 Structural Debt — Phase 3: hub-node test coverage
- Added 95 characterization tests across 4 new files for hub nodes that had
  zero direct coverage:
  - `tests/test_vn_cost_model.py` (32) — `VNCostModel.simulate`: fee math,
    tick tiers, price bands, lot rounding, all rejection reasons, ATC path,
    and the VN price-scale contract (absolute VND vs thousands-VND).
  - `tests/test_triple_barrier.py` (30) — `triple_barrier_pipeline` + blocks:
    PT/SL/vertical first-touch, conservative same-bar tie-break, no look-ahead
    (`t1 >= t0`), 012/raw schemes, unlabelable-row excision, weight normalize.
  - `tests/test_tabular_ensemble.py` (21) — `TabularEnsemble.fit`: OOF assembly,
    meta-learner + CalibratedClassifierCV wiring, missing-class augmentation,
    predict shapes (boosters replaced by a clonable `DecisionTreeClassifier`).
  - `tests/test_run_backtest_wiring.py` (12) — `run_oos` (engine mocked) +
    `equity_metrics`; `_build_wf_config` already covered elsewhere.
- Fixed a stale hub-node reference in `process/context/all-context.md`:
  `run_backtest.main` (no such function) → `run_backtest.run_oos` /
  `_build_wf_config`.
- Test suite: 251 → **346 passed**.

### Local Dashboard — P2 live-render gate closed
- Added `tests/test_dashboard_app_smoke.py`: deterministic Streamlit `AppTest`
  boot smoke for `dashboard/app.py`. Verifies all six tabs
  (MUA / GIỮ / BÁN / Verify / Audit / Settings) render with no uncaught
  exception and no per-tab error boundary, plus a one-holding GIỮ render path.
  Heavy seams (`daily_inference` / `run_post_mortem` / `portfolio_list` /
  `price_lookup`) are stubbed at the tab use-sites, so the smoke needs no
  models, parquet, DuckDB, Gemini, or Telegram.
- Installed `streamlit` (1.58.0) per `requirements_dashboard.txt`; confirmed a
  real `streamlit run --server.headless` boot is clean (health `ok`, no
  traceback).
- Test suite: 249 → **251 passed**.
- Updated `process/features/local-dashboard/HANDOFF.md`: P2 gate marked CLOSED;
  NEXT now points to P3 (launcher).
