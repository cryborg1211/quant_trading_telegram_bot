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
| 1 | `daily_inference` decomposition + report builder extraction | completed (2026-06-10) |
| 2 | Automated feature-schema hashing | completed (2026-06-13) |
| 3 | Hub-node test coverage | completed (2026-06-21) |

## Key Source Files

- `main.py` — Pipeline orchestrator + serving (report builders extracted to src/reports/)
- `src/backtest/pipeline.py` — FEATURE_RECIPE_VERSION, build_features
- `src/models/tabular_ensemble.py` — TabularEnsemble.fit (untested hub)
- `src/labels/triple_barrier.py` — triple_barrier_pipeline (untested hub)
- `src/utils/telegram_bot.py` — build_application (untested hub; scoped OUT of Phase 3 — PTB ApplicationBuilder is awkward under the stubbed telegram module)
- `run_backtest.py` — run_oos / _build_wf_config (hub, degree 97; covered in Phase 3 — there is no standalone `main`)

## Related Context

- `process/context/all-context.md` — architecture and hub node inventory
- `process/context/tests/all-tests.md` — test coverage gaps

## Current Status

Status: **COMPLETE (2026-06-21)** — all three phases done.

- Phase 1: `daily_inference` decomposed (271→169 lines) into `_select_candidates` / `_rescue_loop` / `_dispatch_signals`; 10 report builders extracted to `src/reports/builders.py`.
- Phase 2: feature-schema hashing live (`src/utils/schema_hash.py`); `FEATURE_RECIPE_VERSION` = `v2-sha8:53b5bd85`.
- Phase 3: hub-node coverage — `VNCostModel.simulate`, `triple_barrier_pipeline`, `TabularEnsemble.fit`, `run_oos`/`_build_wf_config` (95 tests). Plan archived to `completed/`.

Follow-up candidate (not in this program's scope): direct `build_application` coverage, and the other untested hubs / monolithic communities from the 2026-06-09 audit.
