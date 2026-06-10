# V4.1 Structural Debt

## Scope

V4.1 "Structural Debt" program — three-phase engineering hardening of the Quant Engine V4.0 codebase. No new alpha signals or model changes. Pure structural quality.

Driven by the Architecture & Alpha Bottleneck Audit (2026-06-09) which identified:
- 20 untested hub-node hotspots
- 9 single-file monolithic communities
- Manual FEATURE_RECIPE_VERSION gate with no automated drift detection
- `daily_inference` (271 lines, degree 84) and `main.py` (32-node community) as God-function/God-module

## Phases

| Phase | Name | Status |
|---|---|---|
| 1 | `daily_inference` decomposition + report builder extraction | not-started |
| 2 | Automated feature-schema hashing | not-started |
| 3 | Hub-node test coverage | not-started |

## Key Source Files

- `main.py` — God-module (orchestration + serving + 4 report builders)
- `src/backtest/pipeline.py` — FEATURE_RECIPE_VERSION, build_features
- `src/models/tabular_ensemble.py` — TabularEnsemble.fit (untested hub)
- `src/labels/triple_barrier.py` — triple_barrier_pipeline (untested hub)
- `src/utils/telegram_bot.py` — build_application (untested hub)
- `run_backtest.py` — main (untested hub, degree 97)

## Related Context

- `process/context/all-context.md` — architecture and hub node inventory
- `process/context/tests/all-tests.md` — test coverage gaps

## Current Status

Status: not-started
