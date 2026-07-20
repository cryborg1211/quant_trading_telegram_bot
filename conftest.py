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
def _hysteresis_off_by_default(monkeypatch):
    """Default OFF for the whole suite — `_select_candidates`' admission-
    hysteresis leg (20-07-26) opens a real `duckdb.connect` to the live repo
    DB (read always, write under persist=True) whenever it runs with a
    non-empty survivor list. Pre-existing test files (e.g.
    test_select_candidates.py) predate this knob and have no reason to know
    about it — defaulting it off here protects them (and any future test)
    without needing every call site individually patched. Tests that
    specifically exercise hysteresis re-enable it locally with their own
    `monkeypatch.setattr(CONFIG.trading, "hysteresis_enabled", True)`, which
    simply overrides this default within that test.
    """
    from config.settings import CONFIG

    monkeypatch.setattr(CONFIG.trading, "hysteresis_enabled", False, raising=False)
