"""One-off cleanup: purge legacy `actual_t5_outcome = -0.05` stub rows.

Before TD-05 was fixed, every high-confidence UP prediction wrote a row to
`rl_mistake_logs` with `actual_t5_outcome` hardcoded to -0.05 — a stub that
never reflected reality. These rows must be deleted before Phase-3 RL
training starts, otherwise the model trains on synthetic labels.

This script is **idempotent** — running it twice is harmless. It logs a row
count before deleting so you can verify the scope of the purge.

Run:
    python -X utf8 scripts/cleanup_legacy_rl_stubs.py

Exit codes:
    0 — purge ran (possibly zero rows; non-error).
    1 — DB engine failed to initialize or the DELETE raised.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when this script is run directly
# (`python scripts/cleanup_legacy_rl_stubs.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_utils import setup_rotating_logging  # noqa: E402
from src.utils.version import get_version  # noqa: E402

LOGGER = logging.getLogger(__name__)

# The exact value used by the pre-TD-05 stub. Stored as a Python float
# literal, which round-trips through DuckDB DOUBLE without precision loss
# (IEEE-754 double of -0.05 is reproducible byte-for-byte).
_STUB_VALUE: float = -0.05


def purge_stubs() -> int:
    """Delete every row in `rl_mistake_logs` where actual_t5_outcome = -0.05.

    Returns:
        Number of rows deleted (0 if the table was already clean).
    """
    # Lazy-import so the version stamp + logging banner appear before any
    # heavy ML imports get pulled in transitively.
    from src.data.db_engine import DuckDBEngine  # noqa: PLC0415

    db = DuckDBEngine()

    # 1. Count first — gives an unambiguous before-state for the operator.
    before = db.conn.execute(
        "SELECT COUNT(*) FROM rl_mistake_logs WHERE actual_t5_outcome = ?",
        [_STUB_VALUE],
    ).fetchone()
    stub_count = int(before[0]) if before else 0
    LOGGER.info(
        "Found %s row(s) with actual_t5_outcome = %s in rl_mistake_logs.",
        stub_count, _STUB_VALUE,
    )

    if stub_count == 0:
        LOGGER.info("Nothing to purge — table is already clean. ✓")
        return 0

    # 2. Sample a few rows so the operator can sanity-check what's about to die.
    sample = db.conn.execute(
        """
        SELECT ticker, predicted_date, predicted_action
        FROM rl_mistake_logs
        WHERE actual_t5_outcome = ?
        ORDER BY predicted_date DESC
        LIMIT 5
        """,
        [_STUB_VALUE],
    ).fetchall()
    LOGGER.info("Sample of rows to be purged (newest 5):")
    for ticker, pdate, action in sample:
        LOGGER.info("    ticker=%s predicted_date=%s action=%s", ticker, pdate, action)

    # 3. DELETE — single statement, locked the same way as the RL writers
    #    (TD-50) so a concurrent backfill or audit-log write can't interleave.
    with db._audit_lock:
        db.conn.execute(
            "DELETE FROM rl_mistake_logs WHERE actual_t5_outcome = ?",
            [_STUB_VALUE],
        )

    # 4. Verify.
    after = db.conn.execute(
        "SELECT COUNT(*) FROM rl_mistake_logs WHERE actual_t5_outcome = ?",
        [_STUB_VALUE],
    ).fetchone()
    leftover = int(after[0]) if after else 0
    if leftover != 0:
        LOGGER.error(
            "Post-delete count is %s (expected 0). DELETE may have failed.",
            leftover,
        )
        return -1

    deleted = stub_count - leftover
    LOGGER.info("✓ Purged %s stub row(s) from rl_mistake_logs.", deleted)

    # 5. Report remaining table size — context for the operator.
    total = db.conn.execute(
        "SELECT COUNT(*) FROM rl_mistake_logs"
    ).fetchone()
    LOGGER.info("rl_mistake_logs now has %s row(s) total.", int(total[0]) if total else 0)
    real_outcomes = db.conn.execute(
        "SELECT COUNT(*) FROM rl_mistake_logs "
        "WHERE actual_t5_outcome IS NOT NULL AND actual_t5_outcome != ?",
        [_STUB_VALUE],
    ).fetchone()
    pending = db.conn.execute(
        "SELECT COUNT(*) FROM rl_mistake_logs WHERE actual_t5_outcome IS NULL"
    ).fetchone()
    LOGGER.info(
        "  └── real-outcome rows: %s  |  pending (NULL) rows: %s",
        int(real_outcomes[0]) if real_outcomes else 0,
        int(pending[0]) if pending else 0,
    )
    return deleted


def main() -> int:
    setup_rotating_logging()
    LOGGER.info("=" * 70)
    LOGGER.info(
        "TD-44 cleanup: purging legacy actual_t5_outcome=%s stub rows "
        "(version=%s)",
        _STUB_VALUE,
        get_version(),
    )
    LOGGER.info("=" * 70)
    try:
        deleted = purge_stubs()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Cleanup failed: %s", exc)
        return 1
    LOGGER.info("=" * 70)
    LOGGER.info("Done. Rows deleted = %s", deleted)
    LOGGER.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
