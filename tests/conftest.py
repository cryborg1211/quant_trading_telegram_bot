"""Pytest bootstrap: sys.path injection and heavy-import stubs.

Stubs register MagicMock sentinels in sys.modules before any test module is
collected. This lets `import main` and `from src.utils.telegram_bot import ...`
work in environments where the full ML / Telegram dependency stack is absent.

Do NOT stub:
  - src.models.quant_agent_arbitrator  (real module under test)
  - src.utils.telegram_alerter         (real module under test)
Their own heavy transitive deps (aiohttp, gnews, google-genai, requests) are
either guarded by try/except inside those modules or stubbed below.
"""
import pathlib
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Ensure project root is importable from any working directory.
# ---------------------------------------------------------------------------
_ROOT = str(pathlib.Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 2. Register lightweight stubs BEFORE pytest collects test modules.
#    The stubs replace packages that would fail to import in a bare test env.
# ---------------------------------------------------------------------------
_STUBS = [
    # main.py hard ML deps
    "joblib",
    "catboost",
    # main.py local-module deps that transitively pull in polars/duckdb
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
for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
