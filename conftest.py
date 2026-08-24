"""Root pytest conftest — makes the repo root importable for the test suite.

pytest adds this file's directory (the repo root) to sys.path, so tests can do
`import main`, `from src.bot.sizing import ...`, etc. without per-file path hacks.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _live_db_guards_off_by_default(monkeypatch):
    """Default OFF for the whole suite — every `_select_candidates` leg that
    reads the LIVE repo DB.

    admission hysteresis (20-07-26) opens a real `duckdb.connect` to the live
    repo DB (read always, write under persist=True) whenever it runs with a
    non-empty survivor list. Pre-existing test files (e.g.
    test_select_candidates.py) predate this knob and have no reason to know
    about it — defaulting it off here protects them (and any future test)
    without needing every call site individually patched. Tests that
    specifically exercise hysteresis re-enable it locally with their own
    `monkeypatch.setattr(CONFIG.trading, "hysteresis_enabled", True)`, which
    simply overrides this default within that test.

    open-cohort dedup (25-08-26) has the SAME exposure and was missed when the
    fixture was written: it calls `signal_ledger.open_tickers()`, which reads the
    live `dispatched_signals` table. That made the suite depend on production
    state — and it duly broke. A `/suggest` tap on 25-08 created real OPEN
    cohorts for BID/GEX/SSI/VPB/VSC, so four tests using BID/VCB/VHM started
    failing with `[DispatchDedup] 1 candidate(s) skipped ... ['BID']` on a machine
    where nothing about the tests had changed. Tests are not allowed to care what
    the live book holds.
    """
    from config.settings import CONFIG

    monkeypatch.setattr(CONFIG.trading, "hysteresis_enabled", False, raising=False)
    monkeypatch.setattr(
        CONFIG.trading, "dispatch_open_cohort_dedup_enabled", False, raising=False)
