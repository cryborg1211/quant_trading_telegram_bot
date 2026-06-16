"""Analyse the sentiment-entry forward paper-log (observability only).

Reads the `sentiment_entry_paperlog` table populated by the daily pipeline +
/verify command (see `_log_sentiment_entry_paperlog` in main.py) and reports
the realized T+3 / T+20 returns for the TREATMENT slice — names where the 5d
price model predicts DOWN but news sentiment is very positive — versus the
control (every other matured row).

The treatment filter (`decision_5d == 0 AND sentiment_score > threshold`) is
applied HERE, at analysis time. Capture is unconditional, so the control group
is always present. `threshold` is read from
`CONFIG.trading.sentiment_entry_threshold` (default 0.7).

This script changes NO data and makes NO trading decision — it is a read-only
report. It may report 0 filled rows until the T+20 windows mature (~21 calendar
days after the first capture).

Run:
    python -X utf8 scripts/analyze_sentiment_paperlog.py

Exit codes:
    0 — analysis ran (possibly 0 filled rows; non-error).
    1 — DB engine failed to initialize or a query raised.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when this script is run directly
# (`python scripts/analyze_sentiment_paperlog.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_utils import setup_rotating_logging  # noqa: E402
from src.utils.version import get_version  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _fmt(value: float | None) -> str:
    """Format an optional return as a signed percentage, or 'n/a'."""
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def analyze() -> int:
    """Load the paper-log, print treatment-vs-control return stats.

    Returns:
        0 always (a 0-row table is a valid, non-error state). DB / query
        failures propagate as exceptions and are turned into exit code 1 by
        `main()`.
    """
    # Lazy-import so the version banner + logging appear before the heavy
    # DuckDB / pandas imports get pulled in transitively.
    from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
    from config.settings import CONFIG  # noqa: PLC0415

    db = DuckDBEngine()

    total = int(
        db.conn.execute("SELECT COUNT(*) FROM sentiment_entry_paperlog").fetchone()[0]
    )
    filled = int(
        db.conn.execute(
            "SELECT COUNT(*) FROM sentiment_entry_paperlog WHERE outcome_filled = TRUE"
        ).fetchone()[0]
    )
    LOGGER.info(
        "Paper-log rows: total=%s | matured (filled)=%s | pending=%s",
        total, filled, total - filled,
    )

    # Source breakdown — self-selection bias in /verify is visible here, since
    # /verify rows are user-requested rather than a systematic cross-section.
    source_rows = db.conn.execute(
        "SELECT source, COUNT(*) FROM sentiment_entry_paperlog GROUP BY source ORDER BY source"
    ).fetchall()
    LOGGER.info("Source breakdown:")
    for src, cnt in source_rows:
        LOGGER.info("    %-7s : %s row(s)", src, int(cnt))

    if filled == 0:
        LOGGER.info(
            "No matured rows yet — returns backfill once the T+20 window elapses "
            "(~21 calendar days after the first capture). Nothing to analyse."
        )
        return 0

    df = db.conn.execute(
        "SELECT * FROM sentiment_entry_paperlog WHERE outcome_filled = TRUE ORDER BY log_date"
    ).df()

    threshold = float(CONFIG.trading.sentiment_entry_threshold)
    LOGGER.info("Treatment filter: decision_5d == 0 (DOWN) AND sentiment_score > %.2f", threshold)

    treatment = df[
        (df["decision_5d"] == 0) & (df["sentiment_score"] > threshold)
    ]
    control = df.drop(treatment.index)

    def _summary(label: str, frame) -> None:
        n = len(frame)
        if n == 0:
            LOGGER.info("    %-10s : 0 rows", label)
            return
        # Pandas .mean()/.median() skip NaN by default → matured-but-missing-shard
        # rows (NULL ret) are excluded from the stat cleanly.
        ret3_mean = frame["ret_3d"].mean()
        ret3_med = frame["ret_3d"].median()
        ret20_mean = frame["ret_20d"].mean()
        ret20_med = frame["ret_20d"].median()
        LOGGER.info(
            "    %-10s : n=%s | ret_3d mean=%s med=%s | ret_20d mean=%s med=%s",
            label, n,
            _fmt(None if ret3_mean != ret3_mean else float(ret3_mean)),
            _fmt(None if ret3_med != ret3_med else float(ret3_med)),
            _fmt(None if ret20_mean != ret20_mean else float(ret20_mean)),
            _fmt(None if ret20_med != ret20_med else float(ret20_med)),
        )

    LOGGER.info("Realized returns (NaN/missing-shard rows skipped):")
    _summary("TREATMENT", treatment)
    _summary("CONTROL", control)

    return 0


def main() -> int:
    setup_rotating_logging()
    LOGGER.info("=" * 70)
    LOGGER.info(
        "Sentiment-entry paper-log analysis (version=%s)", get_version()
    )
    LOGGER.info("=" * 70)
    try:
        code = analyze()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed: %s", exc)
        return 1
    LOGGER.info("=" * 70)
    LOGGER.info("Done.")
    LOGGER.info("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
