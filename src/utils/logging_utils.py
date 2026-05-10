"""Consolidated logging utilities – replaces duplicated setup_logging / timed_step."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """One-time root logger configuration (idempotent)."""
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