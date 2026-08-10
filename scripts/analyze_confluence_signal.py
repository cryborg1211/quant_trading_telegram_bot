"""T+5/T+20 confluence signal analysis (2026-07-22, READ-ONLY).

Idea (from the "smarter system" brainstorm): both horizons are already
scored for every arbitrated candidate and cross-checked by the arbitrator
for veto purposes, but "do T+5 and T+20 AGREE on direction" is never fed
back into ranking/admission as its own signal. Classic multi-timeframe
confirmation â€” cheap to test because `sentiment_entry_paperlog` already
carries both horizons' probabilities AND realized forward returns for
every row logged since 2026-06-16 (no rebuild, no new data).

Decision encoding (shared with the arbitrator, see accuracy_audit.py):
0=DOWN/SELL, 1=SIDE/HOLD, 2=UP/BUY. `decision_primary` is stored directly;
`decision_20d` is derived here via argmax(p_down_secondary, p_side_secondary, p_up_secondary)
since no column stores it.

Zero writes, zero schema changes, zero model changes.

Run: python scripts/analyze_confluence_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402

MIN_RELIABLE_N = 15


def _decision20(p_down, p_side, p_up) -> int:
    probs = [p_down, p_side, p_up]
    return int(np.argmax(probs))


def _bucket_stats(label: str, rets: list[float]) -> None:
    if not rets:
        print(f"{label:<32}{'(empty)':>8}")
        return
    arr = np.array(rets, dtype=np.float64)
    n = len(arr)
    mean_ret = arr.mean()
    hit_rate = float((arr > 0).mean())
    note = "" if n >= MIN_RELIABLE_N else f"<{MIN_RELIABLE_N}, noisy"
    print(f"{label:<32}{n:>6}{mean_ret * 100:>10.2f}%{hit_rate:>10.1%}  {note}")


def main() -> None:
    conn = duckdb.connect(str(REPO / "data" / "quant_v6_core.duckdb"), read_only=True)
    rows = conn.execute("""
        SELECT decision_primary, p_down_secondary, p_side_secondary, p_up_secondary, ret_20d, ret_3d
        FROM sentiment_entry_paperlog
        WHERE p_up_secondary IS NOT NULL AND decision_primary IS NOT NULL
          AND ret_20d IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"Loaded {len(rows)} settled rows with both horizons present.\n")

    buckets: dict[str, list[float]] = {
        "Confluence UP (5d=UP,20d=UP)": [],
        "Confluence DOWN (5d=DN,20d=DN)": [],
        "Divergent (5d=UP,20d=DN)": [],
        "Divergent (5d=DN,20d=UP)": [],
        "5d UP, 20d SIDE": [],
        "5d DOWN, 20d SIDE": [],
        "5d SIDE, any 20d": [],
    }
    all_5d_up: list[float] = []
    conf_up_only: list[float] = []

    for d5, pd20, ps20, pu20, ret20, ret3 in rows:
        d20 = _decision20(pd20, ps20, pu20)
        if d5 == 2:
            all_5d_up.append(ret20)
        if d5 == 1:
            buckets["5d SIDE, any 20d"].append(ret20)
        elif d5 == 2 and d20 == 2:
            buckets["Confluence UP (5d=UP,20d=UP)"].append(ret20)
            conf_up_only.append(ret20)
        elif d5 == 0 and d20 == 0:
            buckets["Confluence DOWN (5d=DN,20d=DN)"].append(ret20)
        elif d5 == 2 and d20 == 0:
            buckets["Divergent (5d=UP,20d=DN)"].append(ret20)
        elif d5 == 0 and d20 == 2:
            buckets["Divergent (5d=DN,20d=UP)"].append(ret20)
        elif d5 == 2 and d20 == 1:
            buckets["5d UP, 20d SIDE"].append(ret20)
        elif d5 == 0 and d20 == 1:
            buckets["5d DOWN, 20d SIDE"].append(ret20)

    print(f"{'bucket':<32}{'n':>6}{'mean ret20d':>11}{'hit rate':>10}  note")
    for label, rets in buckets.items():
        _bucket_stats(label, rets)

    print(f"\n{'=' * 72}\nTHE DIRECT TEST: does requiring 20d confirmation on top of a 5d=UP\n"
          f"pick improve outcomes vs 5d=UP alone (any 20d)?\n{'=' * 72}")
    _bucket_stats("5d=UP, ANY 20d (today's effective pool)", all_5d_up)
    _bucket_stats("5d=UP AND 20d=UP (confluence-gated)", conf_up_only)
    comp = []
    for d5, pd20, ps20, pu20, ret20, ret3 in rows:
        d20 = _decision20(pd20, ps20, pu20)
        if d5 == 2 and d20 != 2:
            comp.append(ret20)
    _bucket_stats("5d=UP but 20d != UP (would be FILTERED OUT)", comp)

    print("\nDone. All buckets use ret_20d (realized 20-session return). "
          "n<15 flagged noisy per this session's reliability floor.")


if __name__ == "__main__":
    main()
