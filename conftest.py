"""Root pytest conftest — makes the repo root importable for the test suite.

pytest adds this file's directory (the repo root) to sys.path, so tests can do
`import main`, `from src.bot.sizing import ...`, etc. without per-file path hacks.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
