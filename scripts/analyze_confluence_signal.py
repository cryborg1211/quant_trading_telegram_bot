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
        "Confluence UP (T20=UP,T5=UP)": [],
        "Confluence DOWN (T20=DN,T5=DN)": [],
        "Divergent (T20=UP,T5=DN)": [],
        "Divergent (T20=DN,T5=UP)": [],
        "T20 UP, T5 SIDE": [],
        "T20 DOWN, T5 SIDE": [],
        "T20 SIDE, any T5": [],
    }
    all_5d_up: list[float] = []
    conf_up_only: list[float] = []

    for d5, pd20, ps20, pu20, ret20, ret3 in rows:
        d20 = _decision20(pd20, ps20, pu20)
        if d5 == 2:
            all_5d_up.append(ret20)
        if d5 == 1:
            buckets["T20 SIDE, any T5"].append(ret20)
        elif d5 == 2 and d20 == 2:
            buckets["Confluence UP (T20=UP,T5=UP)"].append(ret20)
            conf_up_only.append(ret20)
        elif d5 == 0 and d20 == 0:
            buckets["Confluence DOWN (T20=DN,T5=DN)"].append(ret20)
        elif d5 == 2 and d20 == 0:
            buckets["Divergent (T20=UP,T5=DN)"].append(ret20)
        elif d5 == 0 and d20 == 2:
            buckets["Divergent (T20=DN,T5=UP)"].append(ret20)
        elif d5 == 2 and d20 == 1:
            buckets["T20 UP, T5 SIDE"].append(ret20)
        elif d5 == 0 and d20 == 1:
            buckets["T20 DOWN, T5 SIDE"].append(ret20)

    print(f"{'bucket':<32}{'n':>6}{'mean ret20d':>11}{'hit rate':>10}  note")
    for label, rets in buckets.items():
        _bucket_stats(label, rets)

    print(f"\n{'=' * 72}\nTHE DIRECT TEST: does requiring T5 confirmation on top of a T20=UP\n"
          f"pick improve outcomes vs T20=UP alone (any T5)?\n{'=' * 72}")
    _bucket_stats("T20=UP, ANY T5 (the real gate)", all_5d_up)
    _bucket_stats("T20=UP AND T5=UP (confluence-gated)", conf_up_only)
    comp = []
    for d5, pd20, ps20, pu20, ret20, ret3 in rows:
        d20 = _decision20(pd20, ps20, pu20)
        if d5 == 2 and d20 != 2:
            comp.append(ret20)
    _bucket_stats("T20=UP but T5 != UP (confluence would DROP these)", comp)

    print("\nDone. All buckets use ret_20d (realized 20-session return). "
          "n<15 flagged noisy per this session's reliability floor.")


if __name__ == "__main__":
    main()
