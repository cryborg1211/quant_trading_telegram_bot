"""Capture the running code version for logs / crash alerts (TD-31).

Resolution order:
    1. `git rev-parse --short HEAD` — works when running from a git checkout.
    2. Contents of `./VERSION` at project root — for built / deployed artifacts
       where `.git/` may have been stripped.
    3. Literal `"unknown"` — when neither source is available.

The result is memoized on first call (the version is constant within a single
process run), so repeated calls from log lines / crash alerts are free.

Usage:
    from src.utils.version import get_version
    LOGGER.info("Quant V6 starting | version=%s", get_version())
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Project root resolved from this file's location:
# src/utils/version.py → parents[2] = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return short git SHA, or VERSION file contents, or 'unknown'.

    Never raises — every failure mode (no git binary, not in a repo, IO error,
    permission denied) falls through to the next strategy and ultimately to
    the literal `"unknown"` so callers can use the result unconditionally.
    """
    # --- Strategy 1: git rev-parse --short HEAD ---
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        # No git binary, repo, or filesystem access — fall through.
        LOGGER.debug("git rev-parse failed: %s", exc)

    # --- Strategy 2: VERSION file at project root ---
    version_file = _PROJECT_ROOT / "VERSION"
    if version_file.exists():
        try:
            content = version_file.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("VERSION file read failed: %s", exc)

    # --- Strategy 3: give up ---
    return "unknown"
