"""Passive Entry Hypothesis Test — Adverse Selection Kill Switch.

Tests whether placing limit orders 0.5–2% below ATO open produces better
expected P&L *per signal* (not per fill) than naive ATO entry.

Usage:
    python scripts/backtest_entry_optimizer.py [path/to/historical_trades.parquet]

Required columns in parquet: open, low, close_t3

Interpretation guide (printed after results):
    - Compare `expected_pnl_per_signal` rows, not mean_pnl_per_fill.
    - If fill_rate drops while mean_pnl_per_fill rises, you're just
      filtering out the winners (adverse selection in reverse — you miss
      the breakouts and keep the dips).
    - The `gap_and_go_winners` cohort count is the kill-switch signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_naive_ato(df: pl.DataFrame) -> dict:
    """Baseline: all signals filled at ATO open price."""
    pnl = ((pl.col("close_t3") - pl.col("open")) / pl.col("open") * 100)
    result = df.with_columns(pnl_t3=pnl)
    return {
        "strategy": "naive_ato",
        "discount_pct": 0.0,
        "fill_count": len(result),
        "fill_rate": 1.0,
        "mean_pnl_per_fill": result["pnl_t3"].mean(),
        "expected_pnl_per_signal": result["pnl_t3"].mean(),  # fill_rate=1
        "win_rate": (result["pnl_t3"] > 0).mean(),
    }


def simulate_hybrid(df: pl.DataFrame, discount_pct: float) -> dict:
    """
    Passive limit entry at `open * (1 - discount_pct)`.

    Fill condition: `low <= buy_price` — bar-touch approximation.
    This is *optimistic*: real queue priority means actual fill rate
    is lower (only prints if price *slices through* your level).
    A positive result here is a necessary but not sufficient condition.
    """
    result = (
        df.with_columns(buy_price=pl.col("open") * (1 - discount_pct))
        .with_columns(is_filled=pl.col("low") <= pl.col("buy_price"))
        .with_columns(
            pnl_t3=pl.when(pl.col("is_filled"))
            .then((pl.col("close_t3") - pl.col("buy_price")) / pl.col("buy_price") * 100)
            .otherwise(None)
        )
    )

    total = len(result)
    filled = result.filter(pl.col("is_filled"))
    fill_count = len(filled)
    fill_rate = fill_count / total if total > 0 else 0.0
    mean_pnl = filled["pnl_t3"].mean() if fill_count > 0 else None
    win_rate = (filled["pnl_t3"] > 0).mean() if fill_count > 0 else None

    return {
        "strategy": f"hybrid_{discount_pct:.1%}",
        "discount_pct": discount_pct,
        "fill_count": fill_count,
        "fill_rate": fill_rate,
        "mean_pnl_per_fill": mean_pnl,
        # KEY METRIC: expected return per *signal issued*, accounting for missed trades.
        # A hybrid that fills 40% of signals at +3% earns +1.2% per signal.
        # Naive at +0.85% per signal beats it only if breakout P&L isn't the bulk.
        "expected_pnl_per_signal": fill_rate * mean_pnl if mean_pnl is not None else None,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Adverse selection cohort decomposition
# ---------------------------------------------------------------------------

def analyze_adverse_selection(df: pl.DataFrame, discount_pct: float) -> dict[str, dict]:
    """
    Decompose fills into four cohorts that reveal the adverse selection
    structure predicted by the Red Team critique.

    Cohort logic uses *naive* T+3 return as ground truth for "was this
    a good signal?" — independent of entry price.

    Cohorts:
        gap_and_go_winners  — NOT filled, naive return > 0 → missed breakouts
        dip_first_winners   — filled, naive return > 0     → dip-and-rip (legit)
        dip_first_losers    — filled, naive return <= 0    → adverse selection trap
        resilient_losers    — NOT filled, naive return <= 0 → bullets dodged
    """
    result = (
        df.with_columns(
            buy_price=pl.col("open") * (1 - discount_pct),
            naive_pnl=((pl.col("close_t3") - pl.col("open")) / pl.col("open") * 100),
        )
        .with_columns(
            is_filled=pl.col("low") <= pl.col("buy_price"),
            is_winner=pl.col("naive_pnl") > 0,
        )
    )

    total = len(result)
    cohorts = {
        "gap_and_go_winners": result.filter(~pl.col("is_filled") & pl.col("is_winner")),
        "dip_first_winners": result.filter(pl.col("is_filled") & pl.col("is_winner")),
        "dip_first_losers": result.filter(pl.col("is_filled") & ~pl.col("is_winner")),
        "resilient_losers": result.filter(~pl.col("is_filled") & ~pl.col("is_winner")),
    }

    out: dict[str, dict] = {}
    for name, cohort in cohorts.items():
        n = len(cohort)
        out[name] = {
            "count": n,
            "pct_of_total": round(n / total * 100, 1) if total > 0 else 0.0,
            "mean_naive_pnl": round(cohort["naive_pnl"].mean(), 3) if n > 0 else None,
        }
    return out


# ---------------------------------------------------------------------------
# Verdict engine
# ---------------------------------------------------------------------------

def render_verdict(
    naive: dict,
    results: list[dict],
    cohort_map: dict[float, dict[str, dict]],
) -> None:
    """Print structured post-mortem and kill-switch verdict."""
    naive_exp = naive["expected_pnl_per_signal"]

    print("\n" + "=" * 72)
    print("POST-MORTEM: PASSIVE ENTRY HYPOTHESIS")
    print("=" * 72)

    # --- Per-discount analysis ---
    for r in results:
        d = r["discount_pct"]
        cohorts = cohort_map[d]
        exp = r["expected_pnl_per_signal"]
        mean_fill = r["mean_pnl_per_fill"]
        fill_rate = r["fill_rate"]

        print(f"\n[discount = {d:.1%}]")
        print(f"  fill_rate              : {fill_rate:.1%}  ({r['fill_count']} fills)")
        print(f"  mean_pnl per fill      : {mean_fill:.3f}%  vs naive {naive['mean_pnl_per_fill']:.3f}%")
        print(f"  expected_pnl/signal    : {exp:.3f}%  vs naive {naive_exp:.3f}%")
        print(f"  win_rate (fills)       : {r['win_rate']:.1%}" if r["win_rate"] else "  win_rate: n/a")

        print(f"  Cohort breakdown (naive-based):")
        for cname, cdata in cohorts.items():
            print(
                f"    {cname:<25} n={cdata['count']:>4} "
                f"({cdata['pct_of_total']:>5.1f}%)  "
                f"mean_naive_pnl={cdata['mean_naive_pnl']}"
            )

        # Adverse selection signal: are dip_first_losers the dominant fill cohort?
        fills_total = cohorts["dip_first_winners"]["count"] + cohorts["dip_first_losers"]["count"]
        loser_fill_pct = (
            cohorts["dip_first_losers"]["count"] / fills_total * 100
            if fills_total > 0 else 0
        )
        missed_winner_pct = cohorts["gap_and_go_winners"]["pct_of_total"]

        print(f"\n  ADVERSE SELECTION CHECK:")
        if loser_fill_pct > 55:
            print(f"  ⚠  {loser_fill_pct:.0f}% of fills are dip_first_losers — "
                  "adverse selection CONFIRMED at this discount.")
        elif loser_fill_pct > 40:
            print(f"  ~  {loser_fill_pct:.0f}% of fills are dip_first_losers — "
                  "marginal adverse selection, needs deeper review.")
        else:
            print(f"  OK {loser_fill_pct:.0f}% of fills are dip_first_losers — "
                  "adverse selection not dominant at this level.")

        if missed_winner_pct > 20:
            print(f"  ⚠  {missed_winner_pct:.1f}% of all signals are missed gap-and-go winners — "
                  "breakout alpha leakage is SIGNIFICANT.")
        else:
            print(f"  OK {missed_winner_pct:.1f}% missed gap-and-go winners — "
                  "breakout leakage is manageable.")

        if exp is not None and exp < naive_exp * 0.90:
            print(f"  ⛔ expected_pnl/signal {exp:.3f}% < 90% of naive {naive_exp:.3f}% — "
                  "hybrid DESTROYS value at this discount.")
        elif exp is not None and exp >= naive_exp:
            print(f"  ✓  expected_pnl/signal BEATS naive — hypothesis survives at {d:.1%}.")
        elif exp is not None:
            print(f"  ~  expected_pnl/signal within 10% of naive — marginal, depends on "
                  "queue priority and slippage reality.")

    # --- Overall verdict ---
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)

    best_exp = max(
        (r["expected_pnl_per_signal"] for r in results if r["expected_pnl_per_signal"] is not None),
        default=None,
    )

    if best_exp is None:
        print("INCONCLUSIVE — no fills across all discount levels. Data issue?")
        return

    gap_pcts = [cohort_map[r["discount_pct"]]["gap_and_go_winners"]["pct_of_total"] for r in results]
    avg_gap_pct = sum(gap_pcts) / len(gap_pcts)

    loser_fill_pcts = []
    for r in results:
        c = cohort_map[r["discount_pct"]]
        fills_total = c["dip_first_winners"]["count"] + c["dip_first_losers"]["count"]
        if fills_total > 0:
            loser_fill_pcts.append(c["dip_first_losers"]["count"] / fills_total * 100)
    avg_loser_fill_pct = sum(loser_fill_pcts) / len(loser_fill_pcts) if loser_fill_pcts else 0

    kill_signals = 0

    if best_exp < naive_exp:
        kill_signals += 1
        print("⛔ KILL SIGNAL 1: Best hybrid expected_pnl/signal BELOW naive ATO.")
        print(f"   Best hybrid: {best_exp:.3f}%  vs  Naive ATO: {naive_exp:.3f}%")

    if avg_gap_pct > 20:
        kill_signals += 1
        print(f"⛔ KILL SIGNAL 2: Average {avg_gap_pct:.1f}% of signals are missed gap-and-go winners.")
        print("   Your best alpha is structurally excluded from hybrid fills.")

    if avg_loser_fill_pct > 50:
        kill_signals += 1
        print(f"⛔ KILL SIGNAL 3: Average {avg_loser_fill_pct:.1f}% of fills are dip_first_losers.")
        print("   Adverse selection is the dominant fill pattern — you are writing")
        print("   a put on your own watchlist and collecting 1% as premium.")

    print()
    if kill_signals >= 2:
        print("VERDICT: KILL TIER-2 EXECUTION LAYER.")
        print(
            "Two or more kill signals fired. The passive entry hypothesis is NOT "
            "supported by this data. Building the WebSocket/limit-order infrastructure "
            "will cost 3 months and destroy signal-level P&L. Do not build it."
        )
        print(
            "\nRecommended path: archive the hypothesis. If you revisit in 6 months,\n"
            "run this test again on a larger sample before writing a single line of\n"
            "broker API code."
        )
    elif kill_signals == 1:
        print("VERDICT: MARGINAL — collect more data before committing to build.")
        print(
            "One kill signal fired. The hypothesis is not clearly positive. Run\n"
            "this test on 6+ months of data, and validate fill rate against real\n"
            "HOSE limit order queue depth before building infrastructure."
        )
    else:
        print("VERDICT: HYPOTHESIS SURVIVES — proceed to fill-rate validation phase.")
        print(
            "No kill signals. BUT: bar-touch fill simulation is optimistic.\n"
            "Next step: paper-trade with actual limit order submissions for 4 weeks\n"
            "to measure real fill rates before building production infrastructure."
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parquet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("historical_trades.parquet")

    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.")
        print("Expected columns: open, low, close_t3")
        sys.exit(1)

    df = pl.read_parquet(parquet_path)

    required_cols = {"open", "low", "close_t3"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    print(f"Loaded {len(df):,} trades from {parquet_path}")
    print(f"Columns: {df.columns}")

    # --- Baseline ---
    naive = simulate_naive_ato(df)

    print("\n" + "-" * 72)
    print("BASELINE — Naive ATO Entry")
    print("-" * 72)
    print(f"  total signals  : {naive['fill_count']:,}")
    print(f"  mean T+3 pnl   : {naive['mean_pnl_per_fill']:.3f}%")
    print(f"  win_rate       : {naive['win_rate']:.1%}")

    # --- Hybrid sweep ---
    discount_levels = [0.005, 0.01, 0.015, 0.02]
    results = [simulate_hybrid(df, d) for d in discount_levels]
    cohort_map = {d: analyze_adverse_selection(df, d) for d in discount_levels}

    print("\n" + "-" * 72)
    print("HYBRID SWEEP RESULTS")
    print("-" * 72)
    header = f"{'discount':>10} {'fills':>7} {'fill_rate':>10} {'mean_pnl/fill':>14} {'exp_pnl/signal':>15} {'win_rate':>9}"
    print(header)
    print("-" * len(header))
    print(
        f"{'naive_ato':>10} {naive['fill_count']:>7} {'100.0%':>10} "
        f"{naive['mean_pnl_per_fill']:>13.3f}% {naive['expected_pnl_per_signal']:>14.3f}% "
        f"{naive['win_rate']:>8.1%}"
    )
    for r in results:
        exp = r["expected_pnl_per_signal"]
        mean_f = r["mean_pnl_per_fill"]
        print(
            f"{r['discount_pct']:>10.1%} {r['fill_count']:>7} {r['fill_rate']:>10.1%} "
            f"{mean_f:>13.3f}%  " if mean_f is not None else
            f"{r['discount_pct']:>10.1%} {r['fill_count']:>7} {r['fill_rate']:>10.1%} {'n/a':>13}  ",
            end=""
        )
        print(
            f"{exp:>13.3f}%  " if exp is not None else f"{'n/a':>13}  ",
            end=""
        )
        print(f"{r['win_rate']:>8.1%}" if r["win_rate"] is not None else f"{'n/a':>8}")

    # --- Post-mortem ---
    render_verdict(naive, results, cohort_map)


if __name__ == "__main__":
    main()
