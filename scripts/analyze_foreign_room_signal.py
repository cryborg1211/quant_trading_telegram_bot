"""Foreign ownership ROOM as a signal -- the one FastConnect field never
tested (08-08-26).

WHY THIS IS A DIFFERENT HYPOTHESIS FROM THE REJECTED FLOW TESTS
---------------------------------------------------------------
Everything foreign-flow tested so far (992e7ad, 93368cc, 8199777) measured
FLOW: how much did foreigners buy/sell on day t. All rejected as linear
predictors (r ~0.001-0.002), with only the down-day-conditional divergence
angle surviving (small, d~0.08).

`foreign_remain_room_vol` is not a flow -- it is a CONSTRAINT LEVEL: how
many shares foreigners are still legally allowed to buy. In VN this is a
hard regulatory cap per ticker ("het room" = room exhausted), and it is a
well-known local market structure feature:
  * Room near zero = unmet foreign demand with no legal way to express it
    in the continuous market -- often a persistent bid and a premium in
    the negotiated market.
  * Room newly opening (a foreign holder exits) = buyable capacity that
    was previously blocked.

Note the model already has `smart_money_20_xsz` -- but that is a PROXY
built from price x volume (sum(ret*vol)/sum(vol)), not real foreign data.
Room is genuinely new information the feature set has never seen.

METRICS TESTED
--------------
  1. room_days = remain_room_vol / ADV20  -- scale-free "how many days of
     average volume could foreigners still absorb". Low = near lock-out.
  2. room_pctile -- each ticker's room vs its OWN trailing 1y history
     (captures "unusually tight/loose FOR THIS NAME", ticker-neutral).
  3. room_delta_5d -- is room opening or closing (flow-like, but on the
     constraint rather than the trade).

Same lead-lag methodology as the other flow EDAs + a decile table (the
08-08-26 lesson: a linear Pearson r is blind to the U-shaped/threshold
relationships this kind of constraint variable is most likely to have).

READ-ONLY. Run: python scripts/analyze_foreign_room_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

from src.data.foreign_flow_crawler import _DEFAULT_PARQUET  # noqa: E402

HORIZONS = (5, 20)


def _decile_table(d: pl.DataFrame, value_col: str, ret_col: str, label: str) -> None:
    d = d.select([value_col, ret_col]).drop_nulls()
    if d.height < 1000:
        print(f"  {label}: too few rows ({d.height}) — skipped")
        return
    ranks = d[value_col].rank(method="ordinal") / d.height
    d = d.with_columns(pl.Series("decile", (ranks * 10).clip(0, 9.999).floor().cast(pl.Int32) + 1))
    table = d.group_by("decile").agg(
        pl.len().alias("n"),
        pl.col(ret_col).mean().alias("mean_fwd_ret"),
        pl.col(value_col).mean().alias("mean_value"),
    ).sort("decile")
    print(f"\n  {label} (decile 1 = LOWEST {value_col} .. 10 = HIGHEST):")
    print(table)

    # Tail-vs-middle significance, same shape as analyze_flow_extra_fields.
    lo = d.filter(pl.col("decile") == 1)[ret_col].to_numpy()
    hi = d.filter(pl.col("decile") == 10)[ret_col].to_numpy()
    mid = d.filter(pl.col("decile").is_in([5, 6]))[ret_col].to_numpy()
    for name, arr in (("D1 (lowest)", lo), ("D10 (highest)", hi)):
        t, p = stats.ttest_ind(arr, mid, equal_var=False)
        cohen_d = (arr.mean() - mid.mean()) / ((arr.std() ** 2 + mid.std() ** 2) / 2) ** 0.5
        print(f"    {name:<15} mean={arr.mean():+.3%} vs MID {mid.mean():+.3%}  "
              f"t={t:+.3f} p={p:.6f} d={cohen_d:+.4f}")


def main() -> None:
    flow = pl.read_parquet(_DEFAULT_PARQUET).filter(
        pl.col("source") == "ssi_fastconnect_history")
    price = pl.read_parquet("data/ohlcv_*.parquet").sort(["ticker", "date"])
    for h in HORIZONS:
        price = price.with_columns(
            (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0).alias(f"fwd_ret_{h}d")
        )
    price = price.with_columns(
        pl.col("volume").rolling_mean(20, min_samples=10).over("ticker").alias("adv20")
    )

    df = (
        flow.select(["ticker", "date", "foreign_remain_room_vol"])
        .join(price.select(["ticker", "date", "close", "adv20",
                            *[f"fwd_ret_{h}d" for h in HORIZONS]]),
              on=["ticker", "date"], how="inner")
        .sort(["ticker", "date"])
    )
    print(f"Joined rows: {df.height}  tickers: {df['ticker'].n_unique()}")

    # Sanity: does room actually VARY, or is it a near-constant per ticker?
    var_check = df.group_by("ticker").agg(
        pl.col("foreign_remain_room_vol").std().alias("sd"),
        pl.col("foreign_remain_room_vol").mean().alias("mean"),
    ).with_columns((pl.col("sd") / (pl.col("mean").abs() + 1e-9)).alias("cv"))
    print(f"Room coefficient-of-variation across tickers: "
          f"median={var_check['cv'].median():.4f}  "
          f"pct_near_constant(cv<0.01)={float((var_check['cv'] < 0.01).mean()):.1%}")

    df = df.with_columns([
        (pl.col("foreign_remain_room_vol") / (pl.col("adv20") + 1e-9)).alias("room_days"),
        (pl.col("foreign_remain_room_vol")
         - pl.col("foreign_remain_room_vol").shift(5).over("ticker")).alias("room_delta_5d"),
    ])
    # Room vs the ticker's OWN trailing 1y -- ticker-neutral "tight for this name".
    df = df.with_columns(
        ((pl.col("foreign_remain_room_vol")
          - pl.col("foreign_remain_room_vol").rolling_mean(250, min_samples=60).over("ticker"))
         / (pl.col("foreign_remain_room_vol").rolling_std(250, min_samples=60).over("ticker") + 1e-9)
         ).alias("room_z250")
    )

    print(f"\n{'=' * 78}\nLINEAR lead-lag correlation\n{'=' * 78}")
    for col in ("room_days", "room_z250", "room_delta_5d"):
        for h in HORIZONS:
            pair = df.select([col, f"fwd_ret_{h}d"]).drop_nulls()
            if pair.height < 100:
                print(f"  {col:<15} T+{h:<3} n={pair.height} — too few")
                continue
            r = np.corrcoef(pair[col].to_numpy(), pair[f"fwd_ret_{h}d"].to_numpy())[0, 1]
            print(f"  {col:<15} T+{h:<3} r={r:+.4f}  n={pair.height}")

    print(f"\n{'=' * 78}\nDECILE tables (catches threshold / U-shaped effects a linear r misses)\n{'=' * 78}")
    for col in ("room_days", "room_z250"):
        for h in HORIZONS:
            _decile_table(df, col, f"fwd_ret_{h}d", f"{col} vs T+{h}d")

    # ── The clean, economically-specified test ────────────────────────────────
    # The decile tables above are contaminated at BOTH ends: D1 by the
    # zero/negative-room mass and D10 by room_days exploding when ADV20 is
    # tiny (a denominator artifact of MY metric, NOT bad source data --
    # verified: 0 rows have room > 1e12, and negative room is only 0.14% of
    # rows, a real over-limit edge case after corporate actions). A binary
    # "het room" flag sidesteps both and directly tests the actual VN market
    # phenomenon: 16.2% of all rows sit at EXACTLY zero remaining room.
    print(f"\n{'=' * 78}\nCLEAN TEST — binary 'het room' (room <= 0) vs room available\n{'=' * 78}")
    df = df.with_columns((pl.col("foreign_remain_room_vol") <= 0).alias("room_exhausted"))
    rate = float(df["room_exhausted"].mean())
    print(f"Room-exhausted rate: {rate:.2%} of rows")

    for h in HORIZONS:
        ret_col = f"fwd_ret_{h}d"
        ex = df.filter(pl.col("room_exhausted"))[ret_col].drop_nulls().to_numpy()
        av = df.filter(~pl.col("room_exhausted"))[ret_col].drop_nulls().to_numpy()
        t, p = stats.ttest_ind(ex, av, equal_var=False)
        cohen_d = (ex.mean() - av.mean()) / ((ex.std() ** 2 + av.std() ** 2) / 2) ** 0.5
        print(f"\nT+{h}d:")
        print(f"  HET ROOM (<=0):   n={len(ex):>7}  mean={ex.mean():+.3%}  P(ret>0)={float((ex > 0).mean()):.1%}")
        print(f"  room available:   n={len(av):>7}  mean={av.mean():+.3%}  P(ret>0)={float((av > 0).mean()):.1%}")
        print(f"  Welch t={t:+.3f}  p={p:.6f}  cohens_d={cohen_d:+.4f}")

    # Breadth: is this broad, or a few illiquid names?
    ex_tickers = df.filter(pl.col("room_exhausted")).group_by("ticker").agg(pl.len().alias("n"))
    print(f"\nBreadth: {ex_tickers.height} distinct tickers ever het-room "
          f"({ex_tickers.filter(pl.col('n') >= 50).height} with >=50 such days)")

    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
