"""Passive Entry Hypothesis Test — live data edition.

Connects to the local DuckDB instance + OHLCV parquet shards.
No external file path needed.

Data sources (auto-discovered, in priority order):
  1. rl_mistake_logs  — historical model predictions (BUY/SELL/HOLD)
  2. trade_history    — executed live trades (sparse fallback)

close_t3 is computed as LEAD(close, 3) over the sorted trading calendar
per ticker — correct HOSE trading-day arithmetic, no calendar assumptions.

Run from repo root:
    python scripts/passive_entry_backtest.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Resolve repo root regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DB_PATH = "data/quant_v6_core.duckdb"
_OHLCV_GLOB = (_REPO_ROOT / "data" / "ohlcv_*.parquet").as_posix()
_DISCOUNT_LEVELS = [0.005, 0.01, 0.015, 0.02]
_MIN_SAMPLE_WARN = 50
_MIN_SAMPLE_KILL = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_signals(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load BUY signals from rl_mistake_logs; fall back to trade_history.

    Returns a Polars DataFrame with columns: ticker (str), signal_date (date).
    """
    # --- Source 1: rl_mistake_logs ---
    try:
        n = conn.execute("SELECT COUNT(*) FROM rl_mistake_logs").fetchone()[0]
        if n > 0:
            raw = conn.execute("""
                SELECT ticker, predicted_date::DATE AS signal_date, predicted_action
                FROM rl_mistake_logs
            """).df()
            df = pl.from_pandas(raw)
            # Normalise action strings (handles Vietnamese chars / mixed case)
            action_col = df["predicted_action"].cast(pl.Utf8).str.to_uppercase().str.strip_chars()
            df = df.with_columns(action_col.alias("predicted_action"))
            buy_df = df.filter(pl.col("predicted_action") == "BUY")
            if len(buy_df) > 0:
                print(f"[source] rl_mistake_logs — {n} total rows, "
                      f"{len(buy_df)} BUY signals selected.")
                return buy_df.select(["ticker", "signal_date"])

            # All rows present but none are BUY — show distinct actions and warn
            actions = df["predicted_action"].unique().to_list()
            print(f"[warn] rl_mistake_logs has {n} rows but no 'BUY' actions found.")
            print(f"       Distinct predicted_action values: {actions}")
            print("       Falling back to all signals (ignoring action filter).")
            return df.select(["ticker", "signal_date"])
    except Exception as exc:
        print(f"[warn] rl_mistake_logs unavailable: {exc}")

    # --- Source 2: trade_history (fallback) ---
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM trade_history WHERE action = 'BUY'"
        ).fetchone()[0]
        if n > 0:
            raw = conn.execute("""
                SELECT ticker, date::DATE AS signal_date
                FROM trade_history WHERE action = 'BUY'
            """).df()
            print(f"[source] trade_history — {n} BUY rows.")
            return pl.from_pandas(raw)
    except Exception as exc:
        print(f"[warn] trade_history unavailable: {exc}")

    print("[error] No signal source found. Aborting.")
    sys.exit(1)


def _build_analysis_frame(
    signals: pl.DataFrame,
    db_conn: duckdb.DuckDBPyConnection,
) -> pl.DataFrame:
    """Join signals with OHLCV to produce [ticker, signal_date, open, low, close_t3].

    Uses LEAD(close, 3) over trading-calendar order — correct for HOSE holidays.
    Falls back gracefully: rows without T+3 price (end-of-dataset) are dropped.
    """
    # DuckDB can query Polars DataFrames directly via register()
    db_conn.register("signals_view", signals.to_pandas())

    query = f"""
        WITH ohlcv_with_t3 AS (
            SELECT
                ticker,
                date,
                open,
                low,
                LEAD(close, 3) OVER (
                    PARTITION BY ticker ORDER BY date
                ) AS close_t3
            FROM read_parquet('{_OHLCV_GLOB}')
        )
        SELECT
            s.ticker,
            s.signal_date,
            o.open,
            o.low,
            o.close_t3
        FROM signals_view s
        INNER JOIN ohlcv_with_t3 o
            ON s.ticker = o.ticker
            AND s.signal_date = o.date
        WHERE o.close_t3 IS NOT NULL
        ORDER BY s.signal_date, s.ticker
    """
    try:
        raw = db_conn.execute(query).df()
    except Exception as exc:
        print(f"[error] OHLCV join failed: {exc}")
        sys.exit(1)

    df = pl.from_pandas(raw)
    n_dropped = len(signals) - len(df)
    if n_dropped > 0:
        print(f"[info] {n_dropped} signals dropped "
              f"(no OHLCV match or T+3 date beyond dataset).")
    return df


# ---------------------------------------------------------------------------
# Historical proxy fallback
# ---------------------------------------------------------------------------

def _build_proxy_frame(
    tickers: list[str],
    db_conn: duckdb.DuckDBPyConnection,
    date_start: str = "2016-01-01",
    date_end: str = "2025-12-31",
) -> pl.DataFrame:
    """Build a proxy analysis frame from full OHLCV history for `tickers`.

    Used when live signal T+3 outcomes have not yet materialized in the parquet
    shards. Every OHLCV trading day in the window becomes one 'simulated signal',
    testing whether the passive-entry method itself (not the signal quality)
    produces better expected P&L per entry attempt.

    The entry-method question is separable from signal quality: the adverse
    selection structure (gap-and-go winners vs. dip-first losers) is a property
    of HOSE intraday price dynamics, not of any specific model. Testing on
    thousands of historical days gives statistically robust evidence.
    """
    # Build a ticker filter expression for the IN clause
    ticker_list = ", ".join(f"'{t}'" for t in tickers)

    query = f"""
        WITH ohlcv_with_t3 AS (
            SELECT
                ticker,
                date,
                open,
                low,
                close,
                LEAD(open, 1) OVER (
                    PARTITION BY ticker ORDER BY date
                ) AS open_t1,
                LEAD(close, 3) OVER (
                    PARTITION BY ticker ORDER BY date
                ) AS close_t3
            FROM read_parquet('{_OHLCV_GLOB}')
            WHERE ticker IN ({ticker_list})
              AND date >= CAST('{date_start}' AS DATE)
              AND date <= CAST('{date_end}' AS DATE)
        )
        SELECT ticker, date AS signal_date, open, low, close, open_t1, close_t3
        FROM ohlcv_with_t3
        WHERE close_t3 IS NOT NULL
          AND open > 0
          AND close > 0
          AND open_t1 > 0
        ORDER BY date, ticker
    """
    try:
        raw = db_conn.execute(query).df()
    except Exception as exc:
        print(f"[error] Historical proxy query failed: {exc}")
        return pl.DataFrame()

    df = pl.from_pandas(raw)
    print(
        f"  Tickers: {tickers}\n"
        f"  Window:  {date_start} to {date_end}\n"
        f"  Rows:    {len(df)}"
    )
    return df


# ---------------------------------------------------------------------------
# Simulation core (mirrors backtest_entry_optimizer.py)
# ---------------------------------------------------------------------------

def _simulate_naive_ato(df: pl.DataFrame) -> dict:
    result = df.with_columns(
        pnl_t3=((pl.col("close_t3") - pl.col("open")) / pl.col("open") * 100)
    )
    mean_pnl = result["pnl_t3"].mean()
    return {
        "strategy": "naive_ato",
        "discount_pct": 0.0,
        "fill_count": len(result),
        "fill_rate": 1.0,
        "mean_pnl_per_fill": mean_pnl,
        "expected_pnl_per_signal": mean_pnl,
        "win_rate": (result["pnl_t3"] > 0).mean(),
    }


def _simulate_hybrid(df: pl.DataFrame, discount_pct: float) -> dict:
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
        "expected_pnl_per_signal": fill_rate * mean_pnl if mean_pnl is not None else None,
        "win_rate": win_rate,
    }


def _adverse_selection_cohorts(df: pl.DataFrame, discount_pct: float) -> dict[str, dict]:
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
# Verdict
# ---------------------------------------------------------------------------

def _render_verdict(
    naive: dict,
    results: list[dict],
    cohort_map: dict[float, dict[str, dict]],
    sample_n: int,
) -> None:
    naive_exp = naive["expected_pnl_per_signal"]

    print("\n" + "=" * 72)
    print("POST-MORTEM: PASSIVE ENTRY HYPOTHESIS")
    if sample_n < _MIN_SAMPLE_WARN:
        print(f"  ⚠  WARNING: only {sample_n} signals — results are directional only.")
        print("     Statistical significance requires 100+ signals.")
    print("=" * 72)

    for r in results:
        d = r["discount_pct"]
        cohorts = cohort_map[d]
        exp = r["expected_pnl_per_signal"]
        mean_fill = r["mean_pnl_per_fill"]
        fill_rate = r["fill_rate"]

        print(f"\n[discount = {d:.1%}]")
        print(f"  fill_rate              : {fill_rate:.1%}  ({r['fill_count']} fills / {sample_n} signals)")
        pct_diff = ((mean_fill or 0) - naive["mean_pnl_per_fill"])
        print(f"  mean_pnl per fill      : {mean_fill:.3f}%  (naive: {naive['mean_pnl_per_fill']:.3f}%, delta: {pct_diff:+.3f}%)"
              if mean_fill is not None else f"  mean_pnl per fill      : n/a")
        print(f"  expected_pnl/signal    : {exp:.3f}%  vs naive {naive_exp:.3f}%"
              if exp is not None else f"  expected_pnl/signal    : n/a")
        print(f"  win_rate (fills)       : {r['win_rate']:.1%}"
              if r["win_rate"] is not None else "  win_rate               : n/a")

        print("  Cohort breakdown:")
        for cname, cdata in cohorts.items():
            print(
                f"    {cname:<25} n={cdata['count']:>4} "
                f"({cdata['pct_of_total']:>5.1f}%)  "
                f"mean_naive_pnl={cdata['mean_naive_pnl']}"
            )

        fills_total = cohorts["dip_first_winners"]["count"] + cohorts["dip_first_losers"]["count"]
        loser_fill_pct = cohorts["dip_first_losers"]["count"] / fills_total * 100 if fills_total > 0 else 0
        missed_winner_pct = cohorts["gap_and_go_winners"]["pct_of_total"]

        print(f"\n  ADVERSE SELECTION CHECK:")
        tag = "⚠ " if loser_fill_pct > 55 else ("~  " if loser_fill_pct > 40 else "OK ")
        print(f"  {tag} {loser_fill_pct:.0f}% of fills are dip_first_losers.")
        tag2 = "⚠ " if missed_winner_pct > 20 else "OK "
        print(f"  {tag2} {missed_winner_pct:.1f}% of signals are missed gap-and-go winners.")
        if exp is not None and exp < naive_exp * 0.90:
            print(f"  ⛔ expected_pnl/signal {exp:.3f}% < 90% of naive {naive_exp:.3f}% "
                  "→ hybrid DESTROYS value.")
        elif exp is not None and exp >= naive_exp:
            print(f"  ✓  expected_pnl/signal BEATS naive at {d:.1%}.")
        elif exp is not None:
            print(f"  ~  expected_pnl/signal within 10% of naive — marginal.")

    # --- Kill signals ---
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)

    best_exp = max(
        (r["expected_pnl_per_signal"] for r in results if r["expected_pnl_per_signal"] is not None),
        default=None,
    )
    if best_exp is None:
        print("INCONCLUSIVE — no fills at any discount level.")
        return

    gap_pcts = [cohort_map[r["discount_pct"]]["gap_and_go_winners"]["pct_of_total"] for r in results]
    avg_gap_pct = sum(gap_pcts) / len(gap_pcts)

    loser_fill_pcts = []
    for r in results:
        c = cohort_map[r["discount_pct"]]
        ft = c["dip_first_winners"]["count"] + c["dip_first_losers"]["count"]
        if ft > 0:
            loser_fill_pcts.append(c["dip_first_losers"]["count"] / ft * 100)
    avg_loser_fill_pct = sum(loser_fill_pcts) / len(loser_fill_pcts) if loser_fill_pcts else 0

    kill_signals = 0
    if best_exp < naive_exp:
        kill_signals += 1
        print(f"⛔ KILL SIGNAL 1: Best hybrid expected_pnl/signal ({best_exp:.3f}%) "
              f"BELOW naive ({naive_exp:.3f}%).")
    if avg_gap_pct > 20:
        kill_signals += 1
        print(f"⛔ KILL SIGNAL 2: {avg_gap_pct:.1f}% avg missed gap-and-go winners "
              "→ breakout alpha excluded from fills.")
    if avg_loser_fill_pct > 50:
        kill_signals += 1
        print(f"⛔ KILL SIGNAL 3: {avg_loser_fill_pct:.1f}% avg fills are dip_first_losers "
              "→ adverse selection dominant.")

    print()
    if sample_n < _MIN_SAMPLE_WARN:
        print(f"⚠  SAMPLE TOO SMALL ({sample_n} signals) for a definitive verdict.")
        print("   Direction shown below is indicative only.")
        print("   Collect 100+ signals before making the build/kill decision.")
        print()

    if kill_signals >= 2:
        print("VERDICT: KILL — do not build Tier-2 execution infrastructure.")
        print("  Two or more kill signals fired. The hypothesis is structurally "
              "unsupported.")
        print("  Passive entry destroys signal-level P&L by excluding breakout trades.")
    elif kill_signals == 1:
        print("VERDICT: MARGINAL — collect 100+ signals, then re-run.")
        print("  One kill signal. Not enough evidence to build OR kill.")
        print("  Paper-trade for 4 weeks (0.5% discount only) before any infrastructure.")
    else:
        print("VERDICT: HYPOTHESIS SURVIVES (bar-touch simulation — optimistic).")
        print("  No kill signals. Next step: validate with real limit order fill rates")
        print("  before building broker API infrastructure.")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Day-0 strength filter study
# ---------------------------------------------------------------------------

def _simulate_strength_filter(df: pl.DataFrame, discount_pct: float) -> dict:
    """Measure T+3 returns conditional on day-0 price strength.

    "Strength confirmed" = stock never dipped `discount_pct` below its open,
    meaning `low > open * (1 - discount_pct)`.  This is the mirror-image of the
    passive entry cohort.

    Two entry points measured:
    - from_open: theoretical (open is known only in hindsight at signal time)
    - from_close: implementable ATC entry (confirmed at day-0 EOD, tradeable at
      day-0 close or next-day ATO via limit at ATC price)
    """
    result = df.with_columns(
        is_strong=pl.col("low") > pl.col("open") * (1 - discount_pct),
        pnl_from_open=((pl.col("close_t3") - pl.col("open")) / pl.col("open") * 100),
        pnl_from_close=((pl.col("close_t3") - pl.col("close")) / pl.col("close") * 100),
        pnl_from_open_t1=((pl.col("close_t3") - pl.col("open_t1")) / pl.col("open_t1") * 100),
        day0_move=((pl.col("close") - pl.col("open")) / pl.col("open") * 100),
        overnight_gap=((pl.col("open_t1") - pl.col("close")) / pl.col("close") * 100),
    )
    strong = result.filter(pl.col("is_strong"))
    n = len(strong)
    total = len(result)
    return {
        "discount_pct": discount_pct,
        "n_strong": n,
        "strong_rate": n / total if total > 0 else 0.0,
        "mean_pnl_from_open": strong["pnl_from_open"].mean() if n > 0 else None,
        "mean_pnl_from_close": strong["pnl_from_close"].mean() if n > 0 else None,
        "mean_pnl_from_open_t1": strong["pnl_from_open_t1"].mean() if n > 0 else None,
        "win_rate_from_open": (strong["pnl_from_open"] > 0).mean() if n > 0 else None,
        "win_rate_from_close": (strong["pnl_from_close"] > 0).mean() if n > 0 else None,
        "win_rate_from_open_t1": (strong["pnl_from_open_t1"] > 0).mean() if n > 0 else None,
        "mean_day0_move": strong["day0_move"].mean() if n > 0 else None,
        "mean_overnight_gap": strong["overnight_gap"].mean() if n > 0 else None,
    }


def _render_strength_study(
    naive: dict,
    strength_results: list[dict],
) -> None:
    naive_pnl = naive["mean_pnl_per_fill"]
    naive_wr = naive["win_rate"]

    print("\n\n" + "=" * 72)
    print("STRENGTH FILTER STUDY: DAY-0 MOMENTUM CONFIRMATION")
    print("=" * 72)
    print(
        "Conditioning signal: stock price never dipped N% below ATO open.\n"
        "Identifies the gap-and-go cohort that passive entry systematically misses.\n"
        "Two entry points: theoretical (from open) and implementable (from ATC close)."
    )
    print(
        f"\nUnconditional baseline: mean T+3 = {naive_pnl:.3f}%  "
        f"win_rate = {naive_wr:.1%}  (n = {naive['fill_count']:,})"
    )

    print("\n" + "-" * 88)
    header = (
        f"{'threshold':>10} {'n_strong':>9} {'strong%':>7} "
        f"{'pnl/open':>9} {'pnl/close':>10} {'pnl/t1open':>11} "
        f"{'wr/open':>8} {'wr/close':>9} {'wr/t1open':>10} "
        f"{'overnight':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in strength_results:
        vals = [
            r["mean_pnl_from_open"], r["mean_pnl_from_close"], r["mean_pnl_from_open_t1"],
            r["win_rate_from_open"], r["win_rate_from_close"], r["win_rate_from_open_t1"],
            r["mean_overnight_gap"],
        ]
        if any(v is None for v in vals):
            print(f"{r['discount_pct']:>10.1%} {r['n_strong']:>9,} {r['strong_rate']:>7.1%}  n/a")
            continue
        pnl_o, pnl_c, pnl_t1, wr_o, wr_c, wr_t1, ovn = vals
        print(
            f"{r['discount_pct']:>10.1%} {r['n_strong']:>9,} {r['strong_rate']:>7.1%} "
            f"{pnl_o:>+8.3f}% {pnl_c:>+9.3f}% {pnl_t1:>+10.3f}% "
            f"{wr_o:>8.1%} {wr_c:>9.1%} {wr_t1:>10.1%} "
            f"{ovn:>+9.3f}%"
        )

    print()
    print("KEY:")
    print("  threshold  -- min intraday dip to EXCLUDE a stock (tighter = stricter)")
    print("  pnl/open   -- T+3 from day-0 open    [theoretical, unimplementable]")
    print("  pnl/close  -- T+3 from day-0 close   [ATC entry, implementable same day]")
    print("  pnl/t1open -- T+3 from day+1 open    [next-day ATO, implementable overnight]")
    print("  overnight  -- mean gap between day-0 close and day+1 open")

    # --- Viability assessment ---
    print("\n" + "-" * 72)
    print("VIABILITY ASSESSMENT")
    print("-" * 72)
    ROUND_TRIP_COST = 0.5  # 0.2% fee each side + ~0.1% slippage

    for r in strength_results:
        pnl_c = r["mean_pnl_from_close"]
        pnl_t1 = r["mean_pnl_from_open_t1"]
        if pnl_c is None or pnl_t1 is None:
            continue
        net_c = pnl_c - ROUND_TRIP_COST
        net_t1 = pnl_t1 - ROUND_TRIP_COST
        tag_c = "✓ " if net_c > 0.1 else ("~ " if net_c > 0 else "⛔")
        tag_t1 = "✓ " if net_t1 > 0.1 else ("~ " if net_t1 > 0 else "⛔")
        print(
            f"  {r['discount_pct']:.1%} threshold: "
            f"ATC = {pnl_c:+.3f}% (net {net_c:+.3f}%) [{tag_c}]  "
            f"T+1 ATO = {pnl_t1:+.3f}% (net {net_t1:+.3f}%) [{tag_t1}]"
        )

    print()
    best = max(
        (r for r in strength_results if r["mean_pnl_from_open_t1"] is not None),
        key=lambda r: r["mean_pnl_from_open_t1"],
        default=None,
    )
    if best:
        pnl_t1 = best["mean_pnl_from_open_t1"]
        pnl_c = best["mean_pnl_from_close"]
        ovn = best["mean_overnight_gap"]
        net_t1 = pnl_t1 - ROUND_TRIP_COST
        net_c = pnl_c - ROUND_TRIP_COST
        if net_t1 > 0.1:
            print(
                f"FINDING: Strength filter at {best['discount_pct']:.1%} threshold\n"
                f"  T+1 ATO entry: {pnl_t1:+.3f}% gross, {net_t1:+.3f}% net of costs.\n"
                f"  Overnight gap consumes {ovn:+.3f}% on average (partial momentum follow-through).\n"
                f"  This is {pnl_t1 - naive_pnl:+.3f}% conditional lift over the naive baseline.\n"
                f"  Implementation: queue ATO order for confirmed-strength tickers at EOD.\n"
                f"  Cost: zero infrastructure change. One extra filter in daily_inference."
            )
        elif net_c > 0.1:
            print(
                f"FINDING: T+1 ATO entry is not cost-viable ({net_t1:+.3f}% net).\n"
                f"  ATC entry is viable at {net_c:+.3f}% net — enter at day-0 close.\n"
                f"  The overnight gap ({ovn:+.3f}%) erodes returns vs ATC entry."
            )
        else:
            print(
                f"FINDING: Neither ATC ({net_c:+.3f}%) nor T+1 ATO ({net_t1:+.3f}%) clears costs.\n"
                f"  The strength signal exists (+{best['mean_pnl_from_open']:+.3f}% from open) but\n"
                f"  the confirmation lag consumes the gain before any implementable entry.\n"
                f"  The signal may require intraday entry (e.g., first 30-min breakout)\n"
                f"  which is not currently possible with the EOD cron architecture."
            )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import glob as _glob

    print(f"Working directory: {Path.cwd()}")
    print(f"OHLCV glob: {_OHLCV_GLOB}\n")

    # ---- Discover all tickers from parquet shards ----
    shards = _glob.glob(_OHLCV_GLOB)
    if not shards:
        print(f"[error] No OHLCV shards found at {_OHLCV_GLOB}")
        sys.exit(1)
    tickers = sorted(
        Path(s).stem.removeprefix("ohlcv_") for s in shards
    )
    print(f"OHLCV shards discovered: {len(tickers)} tickers")

    # ---- Open DuckDB (read path only — no write lock needed) ----
    try:
        conn = duckdb.connect(_DB_PATH)
    except Exception as exc:
        print(f"[error] Cannot open DuckDB: {exc}")
        print("  Stop the bot service if it holds the write lock.")
        sys.exit(1)

    # ---- Full-market historical proxy ----
    print("Building full-market historical proxy frame (2016-2025)...")
    df = _build_proxy_frame(tickers, conn)
    conn.close()

    if len(df) == 0:
        print("[error] Proxy frame is empty — check OHLCV shard paths.")
        sys.exit(1)

    sample_n = len(df)
    print(f"\nSample rows:")
    print(df.head(5))

    # ---- Baseline ----
    naive = _simulate_naive_ato(df)
    print("\n" + "-" * 72)
    print("BASELINE — Naive ATO Entry (full market, 2016-2025)")
    print("-" * 72)
    print(f"  total observations : {naive['fill_count']:,}")
    print(f"  mean T+3 pnl       : {naive['mean_pnl_per_fill']:.3f}%")
    print(f"  win_rate           : {naive['win_rate']:.1%}")

    # ---- Hybrid sweep ----
    results = [_simulate_hybrid(df, d) for d in _DISCOUNT_LEVELS]
    cohort_map = {d: _adverse_selection_cohorts(df, d) for d in _DISCOUNT_LEVELS}

    print("\n" + "-" * 72)
    print("HYBRID SWEEP RESULTS")
    print("-" * 72)
    header = (
        f"{'discount':>10} {'fills':>8} {'fill_rate':>10} "
        f"{'mean_pnl/fill':>14} {'exp_pnl/sig':>12} {'win_rate':>9}"
    )
    print(header)
    print("-" * len(header))
    base_exp = naive["expected_pnl_per_signal"]
    print(
        f"{'naive_ato':>10} {naive['fill_count']:>8,} {'100.0%':>10} "
        f"{naive['mean_pnl_per_fill']:>13.3f}% {base_exp:>11.3f}% "
        f"{naive['win_rate']:>8.1%}"
    )
    for r in results:
        mf = r["mean_pnl_per_fill"]
        ep = r["expected_pnl_per_signal"]
        mf_str = f"{mf:>13.3f}%" if mf is not None else f"{'n/a':>13} "
        ep_str = f"{ep:>11.3f}%" if ep is not None else f"{'n/a':>11} "
        wr_str = f"{r['win_rate']:>8.1%}" if r["win_rate"] is not None else f"{'n/a':>8}"
        print(
            f"{r['discount_pct']:>10.1%} {r['fill_count']:>8,} {r['fill_rate']:>10.1%} "
            f"{mf_str}  {ep_str}  {wr_str}"
        )

    # ---- Post-mortem (passive entry) ----
    _render_verdict(naive, results, cohort_map, sample_n)

    # ---- Strength filter study ----
    strength_results = [_simulate_strength_filter(df, d) for d in _DISCOUNT_LEVELS]
    _render_strength_study(naive, strength_results)


if __name__ == "__main__":
    main()
