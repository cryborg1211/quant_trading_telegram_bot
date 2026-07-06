"""Item-1 pre-registered evaluation: paperlog rank-sleeve counterfactual.

Answers the FROZEN question from
``process/general-plans/backlog/attack-narrow-market-preregistration_05-07-26.md``:

    Does top-3-by-P(UP) (T+20), ignoring the absolute admission gate, make
    money at T+20 in the current regime?

FROZEN criteria (do not edit after outcome data exists — see the backlog doc):
  * Sleeve  = each day's top-3 tickers by ``p_up_20d`` (``source='daily'``
    rows only), equal-weight, T+20 horizon (``ret_20d``).
  * Minimum sample: n >= 60 settled sleeve name-days (``outcome_filled=TRUE``)
    before ANY conclusion is drawn.
  * SUCCESS: sleeve mean ``ret_20d`` > 0 AND sleeve mean > the same-period
    equal-weight cross-section mean (ranking must beat the tape, not ride it).
  * FAILURE: either condition misses -> item 2 (breadth-conditional sleeve
    A/B) is CANCELLED, not retuned.
  * Max 3 evaluations total, one per month-of-data — count them by hand in
    the backlog doc each time this script is run for a verdict.

READ-ONLY on the DuckDB — never writes. Run with the bot stopped (the live
bot holds an exclusive lock).

Usage:
    python scripts/analyze_rank_sleeve.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import setup_logging  # noqa: E402

LOGGER = logging.getLogger("__main__")

_DB_PATH = Path("data/quant_v6_core.duckdb")
_SLEEVE_SIZE = 3
_MIN_SETTLED_N = 60  # frozen minimum sample (sleeve name-days)


def sleeve_verdict(
    rows: pl.DataFrame,
    sleeve_size: int = _SLEEVE_SIZE,
    min_settled_n: int = _MIN_SETTLED_N,
) -> dict:
    """Apply the frozen item-1 evaluation to raw paperlog rows.

    Parameters
    ----------
    rows : pl.DataFrame
        Columns: ``log_date, ticker, p_up_20d, ret_20d, outcome_filled``.
        Caller pre-filters to ``source='daily'``.

    Returns
    -------
    dict with keys:
        ``settled_sleeve_n``   settled sleeve name-days available
        ``sleeve_mean``        mean ret_20d of settled sleeve rows (or None)
        ``control_mean``       same-period settled cross-section mean (or None)
        ``sufficient``         settled_sleeve_n >= min_settled_n
        ``verdict``            "SUCCESS" | "FAILURE" | "INSUFFICIENT_DATA"
    """
    if rows.is_empty():
        return {
            "settled_sleeve_n": 0, "sleeve_mean": None, "control_mean": None,
            "sufficient": False, "verdict": "INSUFFICIENT_DATA",
        }

    # Sleeve membership is decided at LOG time from the full cross-section
    # (top-N by p_up_20d per day), independent of whether the row later
    # settles — settlement only gates which sleeve rows are EVALUATED.
    sleeve = (
        rows.filter(pl.col("p_up_20d").is_not_null())
        .sort(["log_date", "p_up_20d", "ticker"], descending=[False, True, False])
        .group_by("log_date", maintain_order=True)
        .head(sleeve_size)
    )
    settled_sleeve = sleeve.filter(
        pl.col("outcome_filled") & pl.col("ret_20d").is_not_null()
    )
    n = settled_sleeve.height
    if n == 0:
        return {
            "settled_sleeve_n": 0, "sleeve_mean": None, "control_mean": None,
            "sufficient": False, "verdict": "INSUFFICIENT_DATA",
        }

    d0 = settled_sleeve["log_date"].min()
    d1 = settled_sleeve["log_date"].max()
    control = rows.filter(
        pl.col("outcome_filled")
        & pl.col("ret_20d").is_not_null()
        & (pl.col("log_date") >= d0)
        & (pl.col("log_date") <= d1)
    )

    sleeve_mean = float(settled_sleeve["ret_20d"].mean())
    control_mean = float(control["ret_20d"].mean()) if control.height else None
    sufficient = n >= min_settled_n

    if not sufficient:
        verdict = "INSUFFICIENT_DATA"
    elif sleeve_mean > 0 and control_mean is not None and sleeve_mean > control_mean:
        verdict = "SUCCESS"
    else:
        verdict = "FAILURE"

    return {
        "settled_sleeve_n": n,
        "sleeve_mean": sleeve_mean,
        "control_mean": control_mean,
        "sufficient": sufficient,
        "verdict": verdict,
    }


def main() -> int:
    setup_logging()
    LOGGER.info("=" * 70)
    LOGGER.info("Item-1 rank-sleeve counterfactual (frozen criteria, read-only)")
    LOGGER.info("=" * 70)

    try:
        con = duckdb.connect(str(_DB_PATH), read_only=True)
    except Exception as exc:  # noqa: BLE001 -- typically the live bot's lock
        LOGGER.error("Cannot open DuckDB read-only (bot running?): %s", exc)
        return 1

    try:
        raw = con.execute(
            "SELECT log_date, ticker, p_up_20d, ret_20d, outcome_filled "
            "FROM sentiment_entry_paperlog WHERE source = 'daily'"
        ).fetchall()
    finally:
        con.close()

    rows = pl.DataFrame(
        raw,
        schema={"log_date": pl.Date, "ticker": pl.Utf8, "p_up_20d": pl.Float64,
                "ret_20d": pl.Float64, "outcome_filled": pl.Boolean},
        orient="row",
    )
    LOGGER.info("daily paperlog rows: %d (settled: %d)",
                rows.height, rows.filter(pl.col("outcome_filled")).height)

    res = sleeve_verdict(rows)
    LOGGER.info("settled sleeve name-days : %d (minimum %d)",
                res["settled_sleeve_n"], _MIN_SETTLED_N)
    if res["sleeve_mean"] is not None:
        LOGGER.info("sleeve  mean ret_20d     : %+.2f%%", res["sleeve_mean"] * 100)
    if res["control_mean"] is not None:
        LOGGER.info("control mean ret_20d     : %+.2f%%", res["control_mean"] * 100)
    LOGGER.info("VERDICT: %s", res["verdict"])
    if res["verdict"] == "INSUFFICIENT_DATA":
        LOGGER.info("No conclusion drawn — rows mature 21 calendar days after "
                    "log_date (first daily rows: 16-06-26 -> ~07-07-26).")
    else:
        LOGGER.info("Record this evaluation in the backlog doc (max 3 total).")
    LOGGER.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
