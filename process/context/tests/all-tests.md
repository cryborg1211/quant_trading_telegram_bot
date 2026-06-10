# Quant Engine V4.0 - All Tests

Last updated: 2026-06-10

Attach this file first when the task involves testing, verification, or test debugging.

This is the fast operator guide for the testing surface:

- which runner to use
- what command to start with
- how to quickly debug common failures
- which deeper file to read next

Do not load the whole `process/context/tests/` folder by default. Start here, then drill down.

---

## How This File Works

This is the `all-tests.md` entrypoint for the `tests/` context group. It follows the `all-*.md` routing convention:

1. Agents read `all-context.md` first and get routed here for testing tasks
2. This file gives quick decision rules and commands
3. For deeper details, agents follow the routing table below to specific docs

---

## What This Covers

- pytest runner configuration
- quick commands
- test file map and coverage areas
- fast debugging procedures
- current testing gaps

## Read This When

Use this file when you need to:

- run tests after implementation
- debug failing tests
- understand what is and isn't covered by tests
- add new tests for a feature

## Quick Routing

(No deeper test docs yet. Add routing entries here as they are created.)

## Quick Decision Guide

### Use `pytest` for everything

- All 158 tests run through pytest
- `pytest -q` for quick output (default via `pytest.ini`)
- No separate test runners, no browser tests, no e2e framework
- Tests use in-memory DuckDB and stubs — no external services needed

## Default Verification Order

Unless the task clearly needs a different path:

1. run `pytest -q` (full suite, ~fast with in-memory stubs)
2. run `pytest -q tests/test_<specific>.py` for targeted verification
3. check FEATURE_RECIPE_VERSION alignment if touching feature engineering

## Commands

| Scope | Command | Notes |
|---|---|---|
| Full suite | `pytest -q` | 158 tests, all in-memory |
| Single file | `pytest -q tests/test_<name>.py` | targeted verification |
| Single test | `pytest -q tests/test_<name>.py::test_<func>` | surgical |
| Verbose | `pytest -v` | full test names + pass/fail |
| Stop on first failure | `pytest -x` | useful for debugging |
| Show print output | `pytest -s` | unhides stdout |

## Test File Map

| File | Area | Tests | Notes |
|---|---|---|---|
| `test_main_logic.py` | daily_inference, pipeline orchestration | core serve path | touches the God-function |
| `test_feature_serve.py` | feature pipeline, recipe version gating | train/serve parity | FEATURE_RECIPE_VERSION checks |
| `test_arbitrator.py` | Gemini sentiment scoring, bear veto | arbitrator logic | stubs Gemini API calls |
| `test_market_regime.py` | HMM regime classification | regime features | hub node (build_regime_features) |
| `test_sizing.py` | half-Kelly sizing, NAV cap | position sizing | pure math tests |
| `test_cards.py` | Telegram alert card formatting | message rendering | 4096-char limit awareness |
| `test_telegram_split.py` | multi-message splitting | long report handling | recently added after crash fix |
| `test_event_overrides.py` | build_event_overrides (rescue/veto) | Phase 1 extraction | newly added with decomposition |
| `test_select_candidates.py` | `_select_candidates()` VN30+meta gate | 10 tests | Phase 1 extraction |
| `test_rescue_loop.py` | `_rescue_loop()` sentiment rescue | 8 tests | Phase 1 extraction |
| `test_daily_inference_integration.py` | `daily_inference` end-to-end | 3 tests | happy path, fallback, rescue |
| `test_serve_resilience.py` | serve-path error handling | graceful degradation | API failure stubs |
| `conftest.py` (tests/) | shared fixtures | fixture definitions | in-memory DuckDB, sample data |

## Debugging Quick Reference

- **In-memory DuckDB:** Tests create throwaway in-memory databases — no cleanup needed, no port conflicts.
- **Gemini stubs:** Arbitrator tests stub the Google GenAI SDK. If a test hits the real API, a fixture is missing.
- **Feature recipe mismatch:** If `test_feature_serve.py` fails after touching `build_features`, check that `FEATURE_RECIPE_VERSION` in `src/backtest/pipeline.py` was bumped.
- **Telegram HTML crashes:** If `test_cards.py` or `test_telegram_split.py` fail, check HTML tag closure and string length. The 4096-char limit is enforced in tests.
- **Import errors:** The project uses relative imports under `src/`. Make sure you run pytest from the repo root.

## Known Gaps

- No tests for `run_backtest.py::main` flow (high hub degree but untested end-to-end) — Phase 3 target
- No tests for `train_models.py::main` flow
- No tests for `triple_barrier_pipeline` (degree 82) — Phase 3 target
- No tests for `TabularEnsemble.fit` (degree 75) — Phase 3 target
- No tests for `build_application` (degree 72) — Phase 3 target
- Report builders in `src/reports/builders.py` have partial coverage via `test_main_logic.py` (original import paths)
- No performance/load tests
- No tests for the systemd deployment path or cron scheduling
