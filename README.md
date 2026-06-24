# Quant Engine V4 — Vietnamese Equity Swing-Trading System

A research-grade, end-to-end quantitative trading engine for the **HOSE** (Ho Chi Minh Stock Exchange). It ingests daily OHLCV, engineers leak-safe features, trains a pure-tabular stacking ensemble for T+5 / T+20 swing horizons, sizes a portfolio under realistic Vietnamese-market microstructure costs, and serves signals to a Telegram bot — with a GARCH-HMM macro-regime overlay that dynamically throttles market exposure.

> **Status:** Research / paper-trading. Signals currently fail the deflated-Sharpe deployment gate and are **not** traded with real capital. See [Disclaimer](#disclaimer).

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [GARCH-HMM Regime Overlay](#garch-hmm-regime-overlay)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [Testing](#testing)
- [Disclaimer](#disclaimer)

---

## Highlights

- **Pure-tabular stacking ensemble** — LightGBM + XGBoost + CatBoost → calibrated logistic-regression meta-learner. No deep learning; nothing to overfit on a small market.
- **Leak-safe by construction** — triple-barrier labels with AFML purged K-fold + embargo, frac-diff features, and a feature-schema hash that hard-gates the serve path against train/serve drift.
- **Realistic execution** — a Vietnamese cost model with tick tiers, price bands, lot sizes, ATC volume caps, T+2.5 settlement, and corporate-action handling.
- **Walk-forward backtester** — staggered AFML cohort book (tranche mode), deflated-Sharpe and PBO (CSCV) statistical-rigor gates.
- **GARCH-HMM regime brake** — GARCH(1,1) conditional volatility fused with a multi-dimensional Gaussian HMM into a continuous, leak-free market-exposure scaler.
- **Telegram delivery** — daily signal cards, holdings verification, and post-mortem audits in Vietnamese.

---

## Architecture

```
        ┌────────────┐   ┌──────────────┐   ┌───────────────┐
OHLCV → │  Features  │ → │   Labels     │ → │   Ensemble    │ → P(UP)
shards  │ (frac-diff │   │ (triple      │   │ (LGB+XGB+CAT  │
        │  + regime) │   │  barrier)    │   │  → LogReg)    │
        └────────────┘   └──────────────┘   └───────┬───────┘
                                                     │
   ┌──────────────────────┐   ┌──────────────┐      ▼
   │  GARCH-HMM regime     │ → │  Exposure    │ → ┌───────────────┐
   │  overlay (macro)      │   │  scaler ×w   │   │  Portfolio    │ → weights
   └──────────────────────┘   └──────────────┘   │  (Kelly + MVO)│
                                                  └───────┬───────┘
                                                          ▼
                                          ┌──────────────────────────┐
                                          │  VN cost model + execution│ → fills / NAV
                                          └──────────────────────────┘
                                                          ▼
                                              Telegram bot (serve)
```

**Two decoupled entry points** communicate only through a frozen checkpoint:

- `train_models.py` (heavy lifter) — ingest → features → labels → align → split → feature-select → train ensemble + HMM → checkpoint. Run only when data/features/labels/architecture change.
- `run_backtest.py` (fast evaluator) — loads the checkpoint, replays the identical dataset, runs the walk-forward threshold sweep + statistical gates, and persists the live-bot payload.

---

## GARCH-HMM Regime Overlay

The flagship risk layer. It learns market regimes **unsupervised** and emits a continuous exposure scaler that downsizes the book in bearish/volatile tapes instead of a hard cash-out.

**Pipeline** (`src/models/garch_hmm_regime.py`):

1. **GARCH(1,1)** is fit on the market-breadth return proxy to extract conditional volatility σₜ.
2. A **5-D emission matrix** is formed: `[market_ret, sp500_ret, dxy_ret, usdvnd_ret, log(σₜ)]`.
3. A multi-dimensional **Gaussian HMM** (3–4 states) is fit on the standardized matrix.
4. The **Bull state** is identified (highest mean market return, lowest-variance tiebreak).
5. A leak-free **exposure scaler** is emitted: `clip(P(Bull), 0.2, 1.0)`, multiplied directly into target weights.

**Numerical-stability engineering** (the parts that actually make it work on real data):

- **Log-volatility space** — raw σₜ is always-positive and right-skewed (and random-walks under IGARCH), violating the Gaussian-emission assumption. `log(σₜ)` is approximately symmetric; winsorized at the 99th percentile to tame tails without creating a clipping spike.
- **Persistence guard** — the unconstrained GARCH MLE often lands at α+β ≈ 1.0 (the IGARCH trap → explosive vol tail). A post-fit projection caps α+β ≤ 0.96 by proportionally rescaling (α, β) and recomputing ω to preserve the unconditional variance, forcing mean-reverting volatility.
- **Per-column z-score** + degenerate-fit rejection + multi-restart EM for robust convergence.
- **Leak discipline** — GARCH is causal by construction; HMM inference uses an expanding-window filtered posterior (the smoothed posterior at the last bar equals the forward-only estimate). All normalization params are frozen from the train split.

**Out-of-sample result** (915-day OOS window, T+5, seed 0 — baseline → braked):

| Metric | Baseline | + GARCH-HMM Brake | Δ |
|---|---:|---:|---:|
| Sharpe | −0.36 | −0.15 | +0.21 |
| Max Drawdown | −54.99% | −38.55% | +16.44 pp |
| Total Return | −38.25% | −14.68% | +23.57 pp |

> The OOS window was a bear market. The brake is validated **loss-mitigation** (it roughly halves the drawdown and bleed), **not** standalone alpha — every figure is still negative because the underlying signal lost money in that regime. Robustness across the floor/persistence grid is under active validation.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| ML | LightGBM 4.6, XGBoost 3.2, CatBoost 1.2, scikit-learn 1.8 |
| Regime | `arch` 8.0 (GARCH), `hmmlearn` 0.3 (Gaussian HMM) |
| Data | Polars 1.40 (feature pipeline), DuckDB 1.5 + PyArrow Parquet |
| Numerics | NumPy 2.3, SciPy 1.16 |
| Macro data | yfinance (`^GSPC` / `DX-Y.NYB` / `VND=X`) |
| Sentiment | Google GenAI SDK (Gemini Flash) — soft overlay + hard bear veto |
| Bot | python-telegram-bot 22.7 |
| Deploy | systemd (bot) + cron (daily 15:30 ICT) on a bare-metal VPS |

---

## Quick Start

```bash
# 1. Install (Python 3.11)
pip install -r requirements.txt

# 2. Fill the market data (no API key needed for plain OHLCV crawl)
python main.py --task crawl_hose --force-crawl

# 3. Pull macro series (S&P 500 / DXY / USD-VND)
python main.py --task crawl_macro

# 4. Train the ensemble (writes models/saved/*.joblib)
python train_models.py --tb-horizon 20

# 5. Walk-forward backtest
python run_backtest.py --mode tranche --hold-days 30

# 6. Train the GARCH-HMM regime overlay
python train_macro_regime.py --n-states 3

# 7. (optional) Telegram bot — needs .env (Telegram + Gemini keys)
python run_bot.py
```

`.env` is only required for the full pipeline / bot (Telegram + Gemini). See `.env.example` for the required key names. Plain crawling and backtesting need none. For data-filling details see [`runbook.md`](runbook.md).

---

## Project Structure

```
stock_price_v3/
├── main.py                  # Pipeline orchestrator + serving
├── train_models.py          # Heavy lifter: train ensemble + HMM → checkpoint
├── run_backtest.py          # Fast evaluator: walk-forward + statistical gates
├── train_macro_regime.py    # Train the GARCH-HMM regime overlay
├── run_bot.py               # Telegram bot entry
├── config/                  # Dataclass-based settings + JSON overrides
├── src/
│   ├── backtest/            # Feature pipeline + walk-forward engine
│   ├── data/                # OHLCV / macro crawlers, DuckDB engine
│   ├── features/            # Market-regime classifier, mean-reversion features
│   ├── labels/              # Triple-barrier labeling
│   ├── models/              # Tabular ensemble, macro HMM, garch_hmm_regime
│   ├── execution/           # Vietnamese cost model
│   ├── portfolio/           # Mean-variance + Kelly construction
│   └── reports/             # Telegram report builders
├── scripts/                 # Validation + diagnostic scripts
│   ├── validate_garch_hmm_brake.py   # A/B: signals with vs without brake
│   ├── sweep_garch_hmm_brake.py      # Floor × persistence robustness grid
│   └── walk_forward_macro_pipeline.py# Rolling-window regime diagnostic
└── tests/                   # pytest suite (395 tests)
```

---

## Key Design Decisions

- **Pure functions over OOP** — procedural orchestration, no deep inheritance. Prefer extracting a pure function to adding a class method.
- **Feature-recipe versioning** — `FEATURE_RECIPE_VERSION` is a schema hash; any feature change bumps it and the serve path refuses to load a mismatched model.
- **Tranche over grid** — the backtester defaults to a staggered AFML cohort book; the legacy concentrated delta-rebalance let market beta dominate a thin set of correlated entry dates.
- **VN price-scale contract** — Parquet OHLCV is in *thousands* of VND; everything downstream scales to absolute VND before any cost-model math.
- **Soft regime scaling over hard thresholds** — a continuous exposure scaler avoids the non-differentiable risk cliff of a binary cash-out.

---

## Testing

```bash
python -m pytest -q          # full suite (395 tests)
python -m pytest tests/test_garch_hmm_regime.py -v
```

Tests use in-memory DuckDB stubs and synthetic panels — no live data, models, or network required.

---

## Disclaimer

This is a **personal research project**, provided for educational purposes only. It is **not financial advice**, carries **no warranty**, and must not be used to make investment decisions. The strategies here are paper-traded and currently fail the project's own statistical deployment gates. Markets are risky; past (and especially backtested) performance does not predict future results. Use at your own risk.
