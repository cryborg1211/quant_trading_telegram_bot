"""EDA for foreign/proprietary flow data — run BEFORE any model insertion.

Companion to `src/data/foreign_flow_crawler.py` + `src/features/flow_features.py`.
This script produces evidence, not a decision: it does not gate/retrain
anything by itself. Read the printed report and decide by hand whether the
lead-lag correlations are strong/stable enough to justify a walk-forward
backtest with these features wired into `pipeline.build_features`.

DEPENDENCY NOTE: uses `statsmodels` (ADF test). It is present as a transitive
dependency in this env (via scikit-learn's ecosystem) but is NOT pinned in
requirements.txt — pin it explicitly (`statsmodels==0.14.6` at time of
writing) before relying on this script in a fresh environment.

Run:
    python scripts/eda_flow_features.py --flow-parquet data/foreign_flow_daily.parquet
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

LOGGER = logging.getLogger("quant.eda_flow_features")

# HOSE ATC (at-the-close) session end — the earliest wall-clock moment a
# same-day foreign-flow figure could legitimately be known. Anything fetched
# before this on the SAME calendar date as the flow's `date` is a same-day
# leak candidate (should be impossible if the crawler runs post-close, but
# this check exists precisely to catch a misconfigured cron time).
_ATC_CLOSE_HOUR = 14
_ATC_CLOSE_MINUTE = 45


# ─────────────────────────────────────────────────────────────────────────
# Phase 2.1 — Lag / look-ahead bias check
# ─────────────────────────────────────────────────────────────────────────
def check_no_leakage(flow_df: pl.DataFrame) -> pl.DataFrame:
    """Flag rows whose `fetched_at` is not safely after `date`'s ATC close.

    Two failure modes checked:
      1. `fetched_at` is on an EARLIER calendar date than `date` — impossible
         under a correct post-close crawl; if this fires, the crawler is
         reading same-day intraday data and mislabeling it as `date`'s final
         EOD figure (classic look-ahead leak).
      2. `fetched_at` is on the SAME calendar date as `date` but before ATC
         close — the crawl ran before the session actually closed.
    A `fetched_at` on `date + 1` (or later — e.g. a weekend backfill) is
    always safe and never flagged.
    """
    atc_cutoff = (
        pl.col("date").cast(pl.Datetime)
        + pl.duration(hours=_ATC_CLOSE_HOUR, minutes=_ATC_CLOSE_MINUTE)
    )
    flagged = flow_df.with_columns(
        (pl.col("fetched_at") < atc_cutoff).alias("_leak_flag"),
    ).filter(pl.col("_leak_flag"))

    n_flagged = flagged.height
    if n_flagged:
        LOGGER.warning(
            "[leakage] %d/%d rows have fetched_at BEFORE date's ATC close "
            "(%02d:%02d) — investigate crawler schedule before trusting this data.",
            n_flagged, flow_df.height, _ATC_CLOSE_HOUR, _ATC_CLOSE_MINUTE,
        )
    else:
        LOGGER.info("[leakage] 0/%d rows flagged — fetched_at is safely post-close for all rows.", flow_df.height)
    return flagged


# ─────────────────────────────────────────────────────────────────────────
# Phase 2.2 — Stationarity (ADF) on raw flow vs. rolling differentials
# ─────────────────────────────────────────────────────────────────────────
def adf_report(flow_df: pl.DataFrame, ticker: str) -> dict[str, dict[str, float]]:
    """ADF test on raw `foreign_net_val` vs. its 1st difference, for one ticker.

    Raw cumulative-style flow levels are frequently non-stationary (trending
    with cumulative foreign ownership drift); the differenced series usually
    is. This matters because a model consuming a non-stationary raw feature
    can pick up a spurious trend correlation with ANY other trending series
    (including price) that looks like alpha in-sample and evaporates OOS.
    """
    from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415 — see module docstring

    series = (
        flow_df.filter(pl.col("ticker") == ticker)
        .sort("date")
        .get_column("foreign_net_val")
        .drop_nulls()
        .to_numpy()
    )
    if len(series) < 20:
        LOGGER.warning("[adf] %s has < 20 observations — skipping (result would be unreliable).", ticker)
        return {}

    diffed = np.diff(series)

    def _run(x: np.ndarray, label: str) -> dict[str, float] | None:
        # adfuller raises ValueError on a zero-variance series (e.g. a
        # thinly-traded ticker with literally zero foreign flow its whole
        # local history) -- real data surfaced this on the first run against
        # the actual backfill (26-07-26); every prior run only ever saw a
        # single day, never enough history for a constant series to appear.
        if np.std(x) == 0.0:
            LOGGER.warning("[adf] %s %s series is constant (no variance) -- ADF undefined, skipping.",
                           ticker, label)
            return None
        stat, pvalue, *_ = adfuller(x, autolag="AIC")
        return {"adf_stat": float(stat), "p_value": float(pvalue), "n_obs": float(len(x))}

    raw_result = _run(series, "raw")
    diff_result = _run(diffed, "diff")
    if raw_result is None or diff_result is None:
        return {}
    LOGGER.info(
        "[adf] %s  raw: stat=%.3f p=%.4f  |  diff: stat=%.3f p=%.4f  "
        "(p<0.05 = reject unit root = stationary)",
        ticker, raw_result["adf_stat"], raw_result["p_value"],
        diff_result["adf_stat"], diff_result["p_value"],
    )
    return {"raw": raw_result, "diff": diff_result}


# ─────────────────────────────────────────────────────────────────────────
# Phase 2.3 — Lead-lag correlation: rolling flow vs. future realized return
# ─────────────────────────────────────────────────────────────────────────
def _forward_return(price_df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """`close[t+horizon] / close[t] - 1`, per ticker, via LEAD-style shift(-h).

    Matches this project's existing T+H convention (see triple_barrier.py):
    horizon is in TRADING sessions (row offset within the sorted per-ticker
    frame), not calendar days.
    """
    return price_df.sort(["ticker", "date"]).with_columns(
        (
            pl.col("close").shift(-horizon).over("ticker") / pl.col("close") - 1.0
        ).alias(f"fwd_ret_{horizon}d")
    )


def lead_lag_correlation(
    flow_df: pl.DataFrame, price_df: pl.DataFrame, horizons: list[int] = (5, 20)
) -> pl.DataFrame:
    """Cross-correlation table: rolling flow windows (1/3/5/10d) x forward
    return horizons (T+5/T+20 by default), Pearson r + n, ONE ROW PER
    (flow_window, horizon).

    This is a FILTER, not a proof of alpha: a rolling window/horizon pair
    with |r| indistinguishable from noise across tickers should not be fed
    into `flow_features.py`'s feature set. Compute this per-ticker AND
    pooled — a signal that only "works" pooled (driven by one or two large
    tickers) is much weaker evidence than one that is consistent per-ticker.
    """
    joined = (
        flow_df.select(["ticker", "date", "foreign_net_val"])
        .sort(["ticker", "date"])
        .with_columns([
            pl.col("foreign_net_val").rolling_sum(w, min_samples=max(1, w // 2)).over("ticker").alias(f"flow_{w}d")
            for w in (1, 3, 5, 10)
        ])
    )

    price_with_fwd = price_df.select(["ticker", "date", "close"])
    for h in horizons:
        price_with_fwd = _forward_return(price_with_fwd, h)

    merged = joined.join(price_with_fwd, on=["ticker", "date"], how="inner")

    rows: list[dict[str, float | str | int]] = []
    for w in (1, 3, 5, 10):
        flow_col = f"flow_{w}d"
        for h in horizons:
            ret_col = f"fwd_ret_{h}d"
            pair = merged.select([flow_col, ret_col]).drop_nulls()
            if pair.height < 30:
                rows.append({"flow_window_days": w, "return_horizon_days": h, "pearson_r": None, "n": pair.height})
                continue
            r = np.corrcoef(pair[flow_col].to_numpy(), pair[ret_col].to_numpy())[0, 1]
            rows.append({
                "flow_window_days": w, "return_horizon_days": h,
                "pearson_r": float(r), "n": pair.height,
            })

    result = pl.DataFrame(rows)
    LOGGER.info("[lead-lag] correlation table:\n%s", result)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-parquet", type=Path, default=Path("data/foreign_flow_daily.parquet"))
    parser.add_argument(
        "--ohlcv-glob", type=str, default="data/ohlcv_*.parquet",
        help="Matches the existing per-ticker shard convention (see price_lookup.py).",
    )
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated subset for the ADF report.")
    args = parser.parse_args()

    if not args.flow_parquet.exists():
        raise FileNotFoundError(
            f"{args.flow_parquet} not found — run the crawler "
            "(src/data/foreign_flow_crawler.update_foreign_flow_daily) first."
        )

    flow_df = pl.read_parquet(args.flow_parquet)
    price_df = pl.read_parquet(args.ohlcv_glob)

    LOGGER.info("=== Phase 2.1: leakage check ===")
    check_no_leakage(flow_df)

    LOGGER.info("=== Phase 2.2: ADF stationarity ===")
    tickers = (
        args.tickers.split(",") if args.tickers
        else sorted(flow_df.get_column("ticker").unique().to_list())[:5]
    )
    for t in tickers:
        adf_report(flow_df, t)

    LOGGER.info("=== Phase 2.3: lead-lag correlation ===")
    lead_lag_correlation(flow_df, price_df, horizons=[5, 20])


if __name__ == "__main__":
    main()
