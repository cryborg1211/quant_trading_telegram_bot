"""Pytest bootstrap: sys.path injection + heavy-import stubs (FALLBACK ONLY).

Each module below is replaced with a MagicMock sentinel ONLY IF the real module
cannot be imported (e.g. a bare CI runner without the ML/Telegram stack).  When
the full stack is installed (see requirements.txt), the REAL modules are used —
so the serve-path integration tests actually exercise production code instead of
mocking the very code that had bugs.

Historically these stubs were UNCONDITIONAL, which is precisely why the suite
never caught the live serve regressions: it tested mocks, not the real path.
"""
import importlib
import pathlib
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Ensure the project root is importable from any working directory.
# ---------------------------------------------------------------------------
_ROOT = str(pathlib.Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 2. Prefer the REAL module; stub only when it genuinely cannot be imported.
# ---------------------------------------------------------------------------
_MAYBE_STUB = [
    # main.py hard ML deps
    "joblib",
    "catboost",
    # main.py local modules that transitively pull in polars/duckdb
    "src.features.alpha360_generator",
    "src.trading.portfolio_manager",
    # telegram_bot.py hard deps
    "dotenv",
    "telegram",
    "telegram.constants",
    "telegram.error",
    "telegram.ext",
    # telegram_alerter.py hard dep
    "requests",
]
for _mod in _MAYBE_STUB:
    if _mod in sys.modules:
        continue
    try:
        importlib.import_module(_mod)          # real module present → use it
    except Exception:                          # noqa: BLE001 — absent/broken → stub for bare-env unit tests
        sys.modules[_mod] = MagicMock()
