# Unified Signal Card + Discrete Risk-Tier NAV Cap (evidence-gated)

- **Date:** 02-07-26
- **Type:** SIMPLE
- **Status:** CODE-COMPLETE (02-07-26) — full suite green (505 tests, +23 new).
  Remaining: the A/B backtest run itself (blocked on background training finishing).
- **Approved shape:** user approved counter-proposal (Task 1 serve-side now; Task 2 backtest-first, serve promotion only after A/B evidence)

## Goal

1. **Task 1 — unified VN notification card (T+5/T+20)** with 4-section contract:
   Header (horizon, timestamp, ticker) / Model Prediction (raw probs, base class) /
   Overlay (regime, GARCH state, arbitrator intervention) / Deployment (risk tier, sizing).
   Attribution fields threaded through `main._dispatch_signals` signal dicts.
2. **Task 2 — `RiskTier` module (HIGH=20 / MID=60 / LOW=80 % NAV total-deployment cap)**
   wired into the tranche backtest engine behind an OFF-by-default flag.
   **NOT wired into serve** — serve promotion blocked on A/B evidence vs. incumbent
   regime-sizing policy (the only validated risk overlay).

## Touchpoints

| File | Change |
|---|---|
| `src/trading/risk_tier.py` | NEW — `RiskTier` enum (nav_cap_pct ∈ {20,60,80} exact ints), `classify_risk_tier(p_bull, market_regime)`, `apply_nav_tier_cap(...)` pure clamp |
| `src/backtest/walk_forward.py` | `WalkForwardConfig.use_nav_tier_cap: bool = False`; `_tranche_day` step-3 budget clamp via `apply_nav_tier_cap` |
| `run_backtest.py` | `--nav-tier-cap` flag threaded `_cli → main → run_oos → _build_wf_config` (mirrors `--regime-sizing`) |
| `main.py::_dispatch_signals` | attribution fields: `base_decision`, `regime_action_vi`, `garch_scalar`, `arb_note_vi`, `risk_tier`, `risk_tier_pct`, `risk_tier_label_vi` |
| `src/utils/telegram_alerter.py::_build_message` | 4-section restructure; all new lines conditional on field presence (backward compat + "no stray N/A" contract) |
| `tests/test_risk_tier.py` | NEW — enum exactness, boundary classification, clamp math |
| `tests/test_cards.py` | new-section render/omit tests appended |

## Guardrails honored

- `pipeline.py` untouched; no `models/saved/` writes (training running in background).
- Discrete-int contract: `nav_cap_pct` returns exactly 20/60/80 (int), enum-typed.
- Card stays emoji-free/bullet-free (test_cards.py institutional contract).
- Classification thresholds (`p_bull < 0.40 → HIGH`, `≥ 0.75 → LOW`) documented as
  UNVALIDATED starting points — the backtest A/B sweeps them before any serve use.
- Serve exposure_scalar = clip(p_bull, 0.2, 1.0); both thresholds sit above the 0.2
  floor so passing the scalar as p_bull proxy preserves tier boundaries.

## Verification

- Full pytest green (baseline ~482).
- `use_nav_tier_cap=False` ⇒ byte-identical tranche behavior (default-off test).
- Card renders old fixtures unchanged (existing test_cards assertions).

## Deferred / follow-ups

- A/B backtest run (`--nav-tier-cap` + tier-level sweep {20/40/60/80}) — after
  background training completes (GPU/CPU contention).
- `signal_ledger` attribution columns (schema change) — separate pass.
- Serve-side NAV-tier enforcement — blocked on A/B verdict.
