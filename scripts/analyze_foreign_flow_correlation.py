"""Re-test foreign flow vs price — is the ~0.001 correlation real, or the test's?

WHY RE-RUN (13-08-26)
─────────────────────
`scripts/eda_flow_features.py` concluded foreign flow has correlation ~0.001 with
returns "everywhere", and the thread was closed. Three things about that test make
the conclusion unsafe, and none of them are about the data:

1. **It only ever measured FORWARD returns (T+5, T+20).** There is no lag-0 arm.
   Same-day foreign net buying is mechanically part of the day's buying pressure,
   so a contemporaneous correlation near zero would mean the JOIN IS BROKEN, not
   that flow is uninformative. Without that arm, the pipeline was never validated,
   and a broken join produces ~0 at every horizon — exactly what was observed.

2. **It correlated RAW `foreign_net_val` (absolute VND) against RETURNS (a
   ratio).** Pooled across 411 tickers, a 14-billion-VND day on VCB and a
   50-thousand-VND day on a microcap enter the same Pearson sum. Flow must be
   normalised by the name's own traded value to be dimensionless — what matters
   is flow INTENSITY (flow / ADV), not flow size.

3. **It pooled every ticker-date with no within-ticker demeaning.** Cross-
   sectional heterogeneity then dominates the covariance, and a per-ticker effect
   of any size gets averaged toward zero.

So this script measures the same question with the instrument fixed: lag-0 first
as a wiring check, flow normalised by trailing traded value, and per-ticker
correlations reported alongside pooled ones. Read-only — computes and prints, and
decides nothing.

PRICE-SCALE NOTE: parquet OHLCV `close` is in THOUSANDS of VND (see
[[vn_price_scale_convention]]), so traded value in VND is `close * 1000 * volume`.
`foreign_net_val` is already absolute VND.

Run:
    python scripts/analyze_foreign_flow_correlation.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

LOGGER = logging.getLogger("quant.flow_corr")

_PRICE_UNIT_VND = 1000.0        # parquet close is in thousands of VND
_ADV_WINDOW = 20                # sessions for the normalising traded-value base
_MIN_OBS_PER_TICKER = 120       # ~6 months before a per-ticker r means anything


def load(flow_path: Path, ohlcv_glob: str) -> pl.DataFrame:
    flow = pl.read_parquet(flow_path).select(
        ["date", "ticker", "foreign_net_val", "foreign_buy_val", "foreign_sell_val"]
    )
    # MIXED SHARD DTYPES: vnstock wrote int64 volume, FastConnect float64, and
    # several int64 shards belong to inactive tickers that will never be
    # rewritten. A post-read `.cast()` is too late — the multi-file scan itself
    # raises SchemaError before any expression runs. Must be a SCAN option.
    px = (pl.scan_parquet(ohlcv_glob,
                          cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
            .select(["date", "ticker", "close", "volume"])
            .with_columns([pl.col("close").cast(pl.Float64),
                           pl.col("volume").cast(pl.Float64)])
            .collect())
    flow = flow.with_columns(pl.col("date").cast(pl.Date))
    px = px.with_columns(pl.col("date").cast(pl.Date))
    return flow.join(px, on=["ticker", "date"], how="inner").sort(["ticker", "date"])


def build(df: pl.DataFrame) -> pl.DataFrame:
    """Returns, traded value, and the normalised flow intensity."""
    df = df.with_columns(
        (pl.col("close") * _PRICE_UNIT_VND * pl.col("volume")).alias("traded_val")
    )
    df = df.with_columns([
        # Same-day return: the lag-0 arm the original EDA never had.
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_0"),
        pl.col("traded_val").rolling_mean(_ADV_WINDOW, min_samples=10)
          .over("ticker").alias("adv_val"),
    ])
    # Flow INTENSITY — dimensionless, comparable across a microcap and VCB.
    # Shift the ADV base by one day so today's own turnover cannot normalise
    # today's flow (that would inject same-day volume information).
    df = df.with_columns(
        (pl.col("foreign_net_val") / pl.col("adv_val").shift(1).over("ticker"))
        .alias("flow_intensity")
    )
    # Forward returns, plus a couple of BACKWARD lags to see the shape.
    for h in (1, 3, 5, 20):
        df = df.with_columns(
            (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0)
            .alias(f"fwd_{h}")
        )
    # NOTE: a 1-day "past" return IS `ret_0` (both are close[t]/close[t-1]-1),
    # so only h>=2 adds anything. Kept explicit to avoid re-deriving it.
    for h in (2, 3, 5):
        df = df.with_columns(
            (pl.col("close").shift(1).over("ticker")
             / pl.col("close").shift(h).over("ticker") - 1.0)
            .alias(f"past_{h}")
        )
    return df


def lead_lag_profile(df: pl.DataFrame, flow_col: str, label: str) -> None:
    """Spearman of flow[t] against the 1-day return at t+k, for k = -5..+5.

    The single most informative view: if flow LEADS price the profile peaks at
    k>0; if flow merely accompanies it, the peak sits exactly at k=0 and there is
    nothing to trade. Spearman because the flow distribution is extremely
    fat-tailed (one +14.2e9 VND day against a median of 0), which is precisely
    what collapses Pearson toward zero.
    """
    print(f"\n{'=' * 78}")
    print(f" LEAD-LAG PROFILE (Spearman, 1-day returns) — {label}")
    print(f"{'=' * 78}")
    print(f" {'k':>4} {'spearman':>10} {'pearson':>10} {'n':>10}   direction")
    print(" " + "-" * 62)
    d = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_r1")
    )
    for k in range(-5, 6):
        d2 = d.with_columns(pl.col("_r1").shift(-k).over("ticker").alias("_target"))
        pair = d2.select([flow_col, "_target"]).drop_nulls()
        pair = pair.filter(pl.all_horizontal(pl.all().is_finite()))
        if pair.height < 30:
            continue
        r, rho, _ = _corr(pair[flow_col].to_numpy(), pair["_target"].to_numpy())
        arrow = ("price BEFORE flow" if k < 0 else
                 "SAME DAY" if k == 0 else "price AFTER flow (tradeable)")
        bar = "#" * int(abs(rho) * 100)
        print(f" {k:>+4} {rho:>+10.4f} {r:>+10.4f} {pair.height:>10,}   "
              f"{arrow:<28} {bar}")


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Pearson r, Spearman rho, and Pearson's two-sided p."""
    if len(x) < 30 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return float(r), float(rho), float(p)


def pooled_table(df: pl.DataFrame, flow_col: str, label: str) -> None:
    print(f"\n{'=' * 78}")
    print(f" POOLED — {label}   (flow column: {flow_col})")
    print(f"{'=' * 78}")
    print(f" {'target':>10} {'pearson':>10} {'spearman':>10} {'p':>12} {'n':>10}")
    print(" " + "-" * 56)
    for target, note in (("past_5", "price BEFORE flow"), ("past_3", ""),
                         ("past_2", ""),
                         ("ret_0", "<-- SANITY CHECK (same day)"), ("fwd_1", ""),
                         ("fwd_3", ""), ("fwd_5", "TRADEABLE"),
                         ("fwd_20", "TRADEABLE")):
        pair = df.select([flow_col, target]).drop_nulls()
        pair = pair.filter(pl.all_horizontal(pl.all().is_finite()))
        if pair.height < 30:
            print(f" {target:>10} {'--':>10} {'--':>10} {'--':>12} {pair.height:>10,}")
            continue
        r, rho, p = _corr(pair[flow_col].to_numpy(), pair[target].to_numpy())
        print(f" {target:>10} {r:>+10.4f} {rho:>+10.4f} {p:>12.2e} "
              f"{pair.height:>10,}  {note}")


def per_ticker_table(df: pl.DataFrame, flow_col: str, label: str) -> None:
    print(f"\n{'=' * 78}")
    print(f" PER-TICKER (median across names) — {label}")
    print(f"{'=' * 78}")
    print(f" {'target':>10} {'median r':>10} {'p25':>8} {'p75':>8} "
          f"{'%r>0':>7} {'tickers':>8}")
    print(" " + "-" * 56)
    for target in ("ret_0", "fwd_1", "fwd_5", "fwd_20"):
        rs: list[float] = []
        for (_tkr,), g in df.group_by(["ticker"], maintain_order=True):
            pair = g.select([flow_col, target]).drop_nulls()
            pair = pair.filter(pl.all_horizontal(pl.all().is_finite()))
            if pair.height < _MIN_OBS_PER_TICKER:
                continue
            r, _, _ = _corr(pair[flow_col].to_numpy(), pair[target].to_numpy())
            if np.isfinite(r):
                rs.append(r)
        if not rs:
            print(f" {target:>10}  (no ticker has >= {_MIN_OBS_PER_TICKER} obs)")
            continue
        a = np.array(rs)
        print(f" {target:>10} {np.median(a):>+10.4f} {np.percentile(a, 25):>+8.3f} "
              f"{np.percentile(a, 75):>+8.3f} {(a > 0).mean():>6.1%} {len(a):>8,}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-parquet", type=Path,
                    default=Path("data/foreign_flow_daily.parquet"))
    ap.add_argument("--ohlcv-glob", type=str, default="data/ohlcv_*.parquet")
    ap.add_argument("--liquid-top-n", type=int, default=50,
                    help="also report the tradeable slice by trailing ADV value")
    args = ap.parse_args()

    raw = load(args.flow_parquet, args.ohlcv_glob)
    LOGGER.info("joined rows=%s  tickers=%s  %s..%s", f"{raw.height:,}",
                raw["ticker"].n_unique(), raw["date"].min(), raw["date"].max())

    df = build(raw)
    nonzero = df.filter(pl.col("foreign_net_val") != 0)
    LOGGER.info("rows with NONZERO foreign_net_val: %s of %s (%.1f%%)",
                f"{nonzero.height:,}", f"{df.height:,}",
                100.0 * nonzero.height / max(1, df.height))

    # 1. Exactly what the original EDA did: raw absolute VND, all rows, pooled.
    pooled_table(df, "foreign_net_val", "RAW absolute VND, all rows (the ORIGINAL test)")

    # 2. Same rows, flow normalised by the name's own trailing traded value.
    pooled_table(df, "flow_intensity", "NORMALISED flow/ADV, all rows")

    # 3. Drop the 36% of rows where flow is exactly zero (no foreign activity at
    #    all). Those carry no information and only shrink |r| toward zero.
    pooled_table(nonzero, "flow_intensity", "NORMALISED, nonzero-flow rows only")

    # 4. Per-ticker, which is where a pooled test can hide a real effect.
    per_ticker_table(df, "flow_intensity", "NORMALISED flow/ADV")

    # 5. The tradeable slice — the only universe a signal could ever be used in.
    liq = (df.filter(pl.col("adv_val").is_not_null())
             .with_columns(pl.col("adv_val").rank("ordinal", descending=True)
                             .over("date").alias("adv_rank"))
             .filter(pl.col("adv_rank") <= args.liquid_top_n))
    LOGGER.info("liquid top-%d slice: %s rows", args.liquid_top_n, f"{liq.height:,}")
    pooled_table(liq, "flow_intensity", f"NORMALISED, ADV top-{args.liquid_top_n} only")
    per_ticker_table(liq, "flow_intensity", f"NORMALISED, ADV top-{args.liquid_top_n}")

    # 6. The shape settles the question: peak at k=0 means impact, not information.
    lead_lag_profile(df, "flow_intensity", "all rows")
    lead_lag_profile(liq, "flow_intensity", f"ADV top-{args.liquid_top_n}")


if __name__ == "__main__":
    main()
