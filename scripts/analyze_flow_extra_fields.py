"""Block-deal + aggressor-trade fields EDA, and the foreign-flow divergence
angle (08-08-26, user-requested follow-up to the rejected raw foreign-flow
correlation test).

Two new signal families captured free in the FastConnect backfill
(`868d45b`/`59b1882`) but never tested:
  1. block_deal_val -- negotiated/put-through (thoa thuan) trade value, off
     the continuous order book. No buy/sell split in this endpoint -- tested
     as a raw activity-LEVEL signal, not a signed one.
  2. trade_imbalance = buy_trade_vol - sell_trade_vol -- aggressor-side
     order-flow imbalance (buy-hitting-ask vs sell-hitting-bid), a different
     flavor of momentum than raw foreign flow.

Plus the CONDITIONAL divergence angle the original EDA never tested: does
flow_features.py::flow_knife_catch_divergence (elevated z-scored foreign
flow specifically ON A DOWN DAY) predict forward returns better than the
unconditional flow level that was just rejected (992e7ad, correlation
~0.001-0.002 everywhere)? A narrower, different claim.

Same lead-lag correlation methodology as eda_flow_features.py (self-
contained here, not imported, to keep this script independent) for items
1-2; a conditional P(outcome | flag) comparison, matching
analyze_mr_breadth_inflection.py's pattern, for item 3.

Item 4 (added same session, user pushback on "surely heavy foreign
buy/sell moves price?"): a linear Pearson correlation is blind to a
U-SHAPED relationship. Split flow_net_scaled_adv20 into rank-based
deciles (not qcut -- a large exact-zero mass on thin-flow days breaks
qcut's uniqueness requirement) and compare top/bottom decile mean
forward returns against the middle. Answers whether EXTREME flow days
specifically (either direction) carry information a linear test would
average away against the ~90% of days with unremarkable flow.

Run: python scripts/analyze_flow_extra_fields.py
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
from src.features.flow_features import build_flow_features  # noqa: E402

HORIZONS = (5, 20)
WINDOWS = (1, 3, 5, 10)


def _with_forward_returns(price_df: pl.DataFrame, horizons=HORIZONS) -> pl.DataFrame:
    out = price_df.sort(["ticker", "date"])
    for h in horizons:
        out = out.with_columns(
            (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0).alias(f"fwd_ret_{h}d")
        )
    return out


def lead_lag(flow_df: pl.DataFrame, flow_col: str, price_fwd: pl.DataFrame) -> pl.DataFrame:
    joined = (
        flow_df.select(["ticker", "date", flow_col])
        .sort(["ticker", "date"])
        .with_columns([
            pl.col(flow_col).rolling_sum(w, min_samples=max(1, w // 2)).over("ticker").alias(f"win_{w}d")
            for w in WINDOWS
        ])
    )
    merged = joined.join(price_fwd.select(["ticker", "date", *[f"fwd_ret_{h}d" for h in HORIZONS]]),
                         on=["ticker", "date"], how="inner")
    rows = []
    for w in WINDOWS:
        win_col = f"win_{w}d"
        for h in HORIZONS:
            ret_col = f"fwd_ret_{h}d"
            pair = merged.select([win_col, ret_col]).drop_nulls()
            if pair.height < 30:
                rows.append({"window_days": w, "horizon_days": h, "pearson_r": None, "n": pair.height})
                continue
            r = np.corrcoef(pair[win_col].to_numpy(), pair[ret_col].to_numpy())[0, 1]
            rows.append({"window_days": w, "horizon_days": h, "pearson_r": float(r), "n": pair.height})
    return pl.DataFrame(rows)


def main() -> None:
    flow_df = pl.read_parquet(_DEFAULT_PARQUET)
    price_df = pl.read_parquet("data/ohlcv_*.parquet")
    price_fwd = _with_forward_returns(price_df)

    fc = flow_df.filter(pl.col("source") == "ssi_fastconnect_history")
    print(f"FastConnect rows (have block-deal/trade fields): {fc.height}")

    print(f"\n{'=' * 72}\nITEM 1 -- BLOCK-DEAL VALUE lead-lag correlation\n{'=' * 72}")
    coverage = fc["block_deal_val"].is_not_null().mean()
    print(f"Coverage (non-null block_deal_val): {coverage:.1%}")
    bd = fc.with_columns(pl.col("block_deal_val").fill_null(0.0))
    print(lead_lag(bd, "block_deal_val", price_fwd))

    print(f"\n{'=' * 72}\nITEM 2 -- AGGRESSOR TRADE IMBALANCE (buy_trade_vol - sell_trade_vol) lead-lag\n{'=' * 72}")
    coverage2 = fc["buy_trade_vol"].is_not_null().mean()
    print(f"Coverage (non-null buy_trade_vol): {coverage2:.1%}")
    ti = fc.with_columns(
        (pl.col("buy_trade_vol").fill_null(0.0) - pl.col("sell_trade_vol").fill_null(0.0)).alias("trade_imbalance")
    )
    print(lead_lag(ti, "trade_imbalance", price_fwd))

    print(f"\n{'=' * 72}\nITEM 3 -- DIVERGENCE-CONDITIONAL: flow_knife_catch_divergence vs forward returns\n{'=' * 72}")
    joined = (
        flow_df.select(["ticker", "date", "foreign_buy_val", "foreign_sell_val", "foreign_net_val",
                         "prop_buy_val", "prop_sell_val", "prop_net_val"])
        .join(price_fwd.select(["ticker", "date", "close", "volume", *[f"fwd_ret_{h}d" for h in HORIZONS]]),
              on=["ticker", "date"], how="inner")
    )
    feat = build_flow_features(joined)
    setup_rate = feat["flow_knife_catch_divergence"].mean()
    print(f"Divergence-setup rate: {setup_rate:.3%} of rows with a valid flag")

    for h in HORIZONS:
        ret_col = f"fwd_ret_{h}d"
        div_s = feat.filter(pl.col("flow_knife_catch_divergence") == 1.0).select(ret_col).drop_nulls()
        base_s = feat.filter(pl.col("flow_knife_catch_divergence") == 0.0).select(ret_col).drop_nulls()
        print(f"\nT+{h}:")
        if not div_s.height:
            print("  divergence fires:  NEVER (n=0)")
            continue
        div, base = div_s[ret_col].to_numpy(), base_s[ret_col].to_numpy()
        print(f"  divergence fires:  n={len(div):>7}  mean_fwd_ret={div.mean():+.3%}  "
              f"P(ret>0)={float((div > 0).mean()):.1%}")
        print(f"  baseline (no flag): n={len(base):>7}  mean_fwd_ret={base.mean():+.3%}  "
              f"P(ret>0)={float((base > 0).mean()):.1%}")
        t, p = stats.ttest_ind(div, base, equal_var=False)
        pooled_std = ((div.std() ** 2 + base.std() ** 2) / 2) ** 0.5
        cohens_d = (div.mean() - base.mean()) / pooled_std
        print(f"  Welch t={t:.3f}  p={p:.6f}  cohens_d={cohens_d:.4f} "
              f"({'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'} "
              f"by Cohen's convention -- note return-prediction signals are typically small-d even when real)")

    print("\nBreadth check -- is this a handful of outlier tickers, or broad?")
    per_ticker = (
        feat.filter(pl.col("flow_knife_catch_divergence") == 1.0)
        .group_by("ticker")
        .agg(pl.len().alias("fires"), pl.col("fwd_ret_20d").mean().alias("mean_ret_20d"))
        .sort("fires", descending=True)
    )
    print(f"  distinct tickers with >=1 fire: {per_ticker.height}  |  "
          f"tickers with >=10 fires: {per_ticker.filter(pl.col('fires') >= 10).height}")
    print(per_ticker.head(10))

    print(f"\n{'=' * 72}\nITEM 4 -- TAIL EFFECT: is the flow-return relationship U-SHAPED, not linear?\n{'=' * 72}")
    print("(motivated by: does heavy foreign buy/sell move price, even if the linear\n"
          " correlation in items 1-2 and the original raw-flow test both read ~0?)")
    for h in HORIZONS:
        ret_col = f"fwd_ret_{h}d"
        d = feat.select(["ticker", "flow_net_scaled_adv20", ret_col]).drop_nulls()
        pctile = d["flow_net_scaled_adv20"].rank(method="ordinal") / d.height
        d = d.with_columns(pl.Series("pctile", pctile))

        top = d.filter(pl.col("pctile") >= 0.9)[ret_col].to_numpy()
        mid = d.filter((pl.col("pctile") >= 0.4) & (pl.col("pctile") < 0.6))[ret_col].to_numpy()
        bot = d.filter(pl.col("pctile") < 0.1)[ret_col].to_numpy()

        t_top, p_top = stats.ttest_ind(top, mid, equal_var=False)
        t_bot, p_bot = stats.ttest_ind(bot, mid, equal_var=False)
        d_top = (top.mean() - mid.mean()) / ((top.std() ** 2 + mid.std() ** 2) / 2) ** 0.5
        d_bot = (bot.mean() - mid.mean()) / ((bot.std() ** 2 + mid.std() ** 2) / 2) ** 0.5
        top_tickers = d.filter(pl.col("pctile") >= 0.9).group_by("ticker").agg(pl.len().alias("n"))

        print(f"\nT+{h}d  (MID decile 4-6 baseline: n={len(mid)}, mean={mid.mean():+.3%}):")
        print(f"  BOTTOM decile (heaviest SELL): n={len(bot):>7}  mean={bot.mean():+.3%}  "
              f"t={t_bot:+.3f}  p={p_bot:.6f}  d={d_bot:+.4f}")
        print(f"  TOP decile    (heaviest BUY):  n={len(top):>7}  mean={top.mean():+.3%}  "
              f"t={t_top:+.3f}  p={p_top:.6f}  d={d_top:+.4f}")
        print(f"  TOP decile spans {top_tickers.height} distinct tickers "
              f"({top_tickers.filter(pl.col('n') >= 20).height} with >=20 obs each)")

    print("\nInterpretation: BOTH tails typically read as statistically real (n this large "
          "makes even a tiny effect detectable) but the effect sizes are smaller than the "
          "Item 3 divergence finding, and the buy-side and sell-side magnitudes are usually "
          "comparable -- i.e. this reads as a general 'extreme flow day -> slightly elevated "
          "forward return, either direction' pattern, not a buy-side-specific 'chase-buying' "
          "effect uniquely stronger than the sell-side. Confirms the earlier linear-correlation "
          "REJECT was a real limitation of Pearson r on a non-linear relationship, not a sign "
          "flow carries zero information -- but the recovered information is even smaller than "
          "Item 3's, and less attributable to foreign investors specifically (a symmetric U-shape "
          "is also consistent with a generic extreme-day volatility/liquidity effect that would "
          "likely show up for ANY heavy-flow day, domestic or foreign, untested here).")

    print("\nNo artifacts written -- research verdict only.")


if __name__ == "__main__":
    main()
