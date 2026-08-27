"""One-off repair: re-mark /suggest-created cohorts as paper (27-08-26).

`/suggest` recorded real (is_paper=FALSE) cohorts until 27-08-26. Those rows sit
in `open_tickers()` and therefore veto the same names out of the 15:30 cron and
out of any later /suggest tap — the operator's preview consumed the day's own
candidates.

The code no longer does this. This fixes the rows already written.

HOW A CRON ROW IS TOLD APART FROM A /suggest ROW, without guessing from
timestamps: `main.py`'s dispatch path books a T+20 cohort AND a paired T+5
tracking row for the same ticker+date (`run_trade_execution`, the
`int(horizon) != SHORT_HORIZON` branch). The bot's /suggest path records ONE
horizon only. So a (ticker, date) group holding BOTH horizons is a cron dispatch —
the real book — and is never touched.

Dry-run by default; --commit to write.

    python scripts/repair-suggest-cohorts-to-paper.py --since 2026-08-27
    python scripts/repair-suggest-cohorts-to-paper.py --since 2026-08-27 --commit
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path.cwd()))

import duckdb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", required=True,
                    help="only rows with dispatch_date >= this (YYYY-MM-DD)")
    ap.add_argument("--tickers", default=None,
                    help="comma list to restrict to; default = all matching rows")
    ap.add_argument("--db", default="data/quant_v6_core.duckdb")
    ap.add_argument("--commit", action="store_true", help="write (default: dry run)")
    a = ap.parse_args()

    since = date.fromisoformat(a.since)
    only = ({t.strip().upper() for t in a.tickers.split(",")}
            if a.tickers else None)

    con = duckdb.connect(a.db)
    rows = con.execute(
        """SELECT ticker, dispatch_date, horizon, hold_days, weight, status,
                  dispatched_at, COALESCE(is_paper, FALSE) AS is_paper
             FROM dispatched_signals
            WHERE dispatch_date >= ? AND status = 'OPEN'
              AND COALESCE(is_paper, FALSE) = FALSE
            ORDER BY dispatched_at""",
        [since],
    ).fetchall()
    if only:
        rows = [r for r in rows if r[0].upper() in only]

    # A (ticker, date) group carrying BOTH horizons came from main.py's dispatch
    # path, which books the T+20 cohort plus a paired T+5 tracking row. That is
    # the real book — exclude it. /suggest records a single horizon.
    horizons: dict[tuple, set] = {}
    for r in con.execute(
        """SELECT ticker, dispatch_date, horizon FROM dispatched_signals
            WHERE COALESCE(is_paper, FALSE) = FALSE"""
    ).fetchall():
        horizons.setdefault((r[0].upper(), r[1]), set()).add(r[2])

    cron_keys = {k for k, hs in horizons.items() if {5, 20} <= hs}
    kept = [r for r in rows if (r[0].upper(), r[1]) in cron_keys]
    rows = [r for r in rows if (r[0].upper(), r[1]) not in cron_keys]

    if kept:
        print(f"LEAVING {len(kept)} cron row(s) alone (h20+h5 pair = real book):")
        for r in kept:
            print(f"   {r[0]:<5} {r[1]}  h={r[2]}")
        print()

    if not rows:
        print("nothing to repair.")
        con.close()
        return

    print(f"{len(rows)} row(s) would be re-marked is_paper=TRUE:")
    for r in rows:
        print(f"   {r[0]:<5} {r[1]}  h={r[2]:<3} weight={r[4]:.6f}  at {r[6]}")

    from src.trading import signal_ledger  # noqa: PLC0415

    before = signal_ledger.open_tickers(db_path=a.db)
    print(f"\nopen_tickers() BEFORE: {sorted(before)}")

    if not a.commit:
        print("\nDRY RUN — pass --commit to apply.")
        con.close()
        return

    for r in rows:
        con.execute(
            """UPDATE dispatched_signals SET is_paper = TRUE
                WHERE ticker = ? AND dispatch_date = ? AND horizon = ?""",
            [r[0], r[1], r[2]],
        )
    con.close()

    after = signal_ledger.open_tickers(db_path=a.db)
    print(f"open_tickers() AFTER : {sorted(after)}")
    print(f"\nfreed for the cron: {sorted(before - after)}")


if __name__ == "__main__":
    main()
