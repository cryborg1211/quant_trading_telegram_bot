"""Consolidated logging utilities.

Two generations live in this module:

    Older (still used by inner modules):
        • setup_logging()    — basicConfig-based, no rotation.
        • get_logger(name)   — shorthand for `logging.getLogger(name)`.
        • timed_step(msg)    — context manager logging elapsed wall-clock.

    Newer (TD-26 — production safety net):
        • setup_rotating_logging(...)
            Configure the ROOT logger with a stdout handler + a
            RotatingFileHandler on logs/quant_v6.log. Call this once
            at process startup from main()/run_bot.py.
        • get_crawler_error_logger()
            Return a dedicated `crawler.errors` logger backed by a
            RotatingFileHandler on logs/crawler_errors.txt.

Order of calls matters: `setup_rotating_logging()` should be invoked
BEFORE `setup_logging()`. Once the root logger has handlers attached,
`basicConfig` becomes a no-op — so the older API gracefully defers
to the rotating handlers we install first.

Rotation policy:
    • Each file capped at 10 MiB.
    • 5 rotated backups retained → effective max disk per stream = 60 MiB.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Generator

_CONFIGURED = False

_DEFAULT_LOG_DIR = Path("logs")
_ROTATING_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_ROTATING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB per file (TD-26 spec)
_BACKUP_COUNT = 5               # 5 rotated backups (TD-26 spec)


def setup_logging(level: int = logging.INFO) -> None:
    """One-time root logger configuration (idempotent).

    Legacy entry point — prefer `setup_rotating_logging()` in production code.
    If `setup_rotating_logging()` has already installed handlers on the root
    logger, this call's `logging.basicConfig` is a no-op (stdlib behavior).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str = __name__) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def timed_step(message: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    """Context-manager that logs elapsed wall-clock time for a block."""
    _log = logger or get_logger()
    start = time.perf_counter()
    _log.info("%s …", message)
    try:
        yield
    finally:
        _log.info("%s done in %.2fs.", message, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# TD-26: Rotating handlers for production
# ---------------------------------------------------------------------------

def _resolve_target_path(directory: Path, filename: str) -> Path:
    """Return an absolute Path for the target file (creates parent dirs)."""
    directory.mkdir(parents=True, exist_ok=True)
    return (directory / filename).resolve()


def setup_rotating_logging(
    log_filename: str = "quant_v6.log",
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> None:
    """Wire up stdout + a 10 MiB × 5-backup RotatingFileHandler on the root logger.

    Safe to call multiple times: handlers are deduped by class + target path,
    so a `main()` that calls this and then an inner module that re-runs
    `logging.basicConfig` will NOT duplicate output.

    Marks the module-level `_CONFIGURED` flag so the legacy `setup_logging()`
    becomes a no-op — preventing it from re-running `basicConfig` after we
    already wired up rotation.
    """
    global _CONFIGURED

    directory = log_dir or _DEFAULT_LOG_DIR
    target_path = _resolve_target_path(directory, log_filename)
    formatter = logging.Formatter(_ROTATING_LOG_FORMAT, datefmt=_ROTATING_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Add a StreamHandler (stdout) only if no plain StreamHandler is attached.
    # Exact-type check so we don't false-match a RotatingFileHandler (subclass).
    has_stream = any(
        type(h) is logging.StreamHandler
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(level)
        root.addHandler(sh)

    # Add the RotatingFileHandler only if one isn't already pointing here.
    target_str = str(target_path)
    has_rotating = any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", "") == target_str
        for h in root.handlers
    )
    if not has_rotating:
        rfh = RotatingFileHandler(
            target_str,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        rfh.setFormatter(formatter)
        rfh.setLevel(level)
        root.addHandler(rfh)

    _CONFIGURED = True  # gate the legacy setup_logging() from re-running basicConfig.


def get_crawler_error_logger() -> logging.Logger:
    """Return the dedicated `crawler.errors` logger.

    Output goes ONLY to `logs/crawler_errors.txt` (rotating: 10 MiB × 5
    backups). Does not propagate to root — these are bulk per-ticker fail
    records meant for offline review, not for the live console stream.

    The format is plain `%(message)s` so callers can keep their existing
    TSV-like layout (timestamp \\t ticker \\t context \\t error).
    """
    logger = logging.getLogger("crawler.errors")
    already_has_file_handler = any(
        isinstance(h, RotatingFileHandler) for h in logger.handlers
    )
    if not already_has_file_handler:
        target_path = _resolve_target_path(_DEFAULT_LOG_DIR, "crawler_errors.txt")
        rfh = RotatingFileHandler(
            str(target_path),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        rfh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rfh)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # don't double-log to root / stdout
    return logger