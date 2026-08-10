"""Do fundamentals predict forward returns? Free-tier feasibility test
(08-08-26) -- run BEFORE paying for vnstock's full-history tier.

THE DECISION THIS INFORMS
-------------------------
Full history costs money. The free tier caps at 8 periods/ticker (~2 years
of quarters), which is too short for a production backtest but NOT too
short to answer the one question that matters: is there a cross-sectional
signal here at all? If P/E / P/B / ROE rank shows nothing across ~360
tickers x 8 quarters, the paid tier buys a longer series of nothing.

POINT-IN-TIME SAFETY (the classic fundamentals backtest trap)
-------------------------------------------------------------
vnstock gives period labels (2024-Q3) but NO publish date. VN quarterly
reports land ~30-45 days after quarter end, so joining a quarter's ratios
to that quarter's own dates is look-ahead bias. Here every value becomes
visible only at quarter_end + PUBLISH_LAG_DAYS (45, deliberately
conservative), then applies forward until the next report -- implemented as
a backward as-of join, so no row ever sees a number that was not public.

METRIC: Information Coefficient (IC)
------------------------------------
Per date, Spearman rank correlation between the cross-sectional ratio rank
and the forward 20d return; then the mean IC and its t-stat across dates.
This is the standard factor-evaluation metric and is scale-free. Sign
expectations: value ratios (P/E, P/B, P/S) should be NEGATIVE (cheap
predicts higher returns), quality (ROE, ROA, margins) POSITIVE.

Also reports a decile spread (top-minus-bottom mean forward return), since
this session already learned a linear/monotone read can miss a threshold
effect (see analyze_flow_extra_fields.py item 4).

READ-ONLY. Requires data/fundamentals_freetier.parquet from
scripts/crawl_fundamentals_freetier.py (collected in the `vnfund` env).

Run: python scripts/analyze_fundamentals_freetier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

FUND = REPO / "data" / "fundamentals_freetier.parquet"
HORIZON = 20                 # match the serve model's T+20 primary horizon
PUBLISH_LAG_DAYS = 45        # conservative VN quarterly reporting lag
MIN_TICKERS_PER_DATE = 30    # a cross-sectional rank needs breadth to mean anything

# MAX STALENESS -- the fix for the free tier's shape. It returns mostly 2018
# for ~300 tickers and almost nothing after, so a naive backward as-of join
# carries a 2018Q4 ratio forward through 2026 and silently "tests" whether a
# 2018 P/E predicts 2024 returns. A live system re-reports every ~90 days, so
# with a 45d publish lag nothing is ever more than ~135 days old; 180 is the
# generous version of that bound. Rows staler than this are dropped, which
# confines the test to roughly 2018-05 .. 2019-08 -- an old window, but a
# HONEST one.
MAX_STALENESS_DAYS = 180

# Expected IC sign per factor family -- lets the report flag results that are
# statistically real but economically BACKWARDS (a classic overfit tell).
EXPECTED_SIGN = {
    "P/E": -1, "P/B": -1, "P/S": -1, "Price/Cash Flow": -1, "EV/EBITDA": -1,
    "ROE (%)": +1, "ROA (%)": +1, "Gross Margin (%)": +1, "EBIT Margin (%)": +1,
    "Debt/Equity": -1, "Current Ratio": +1, "Quick Ratio": +1,
    "Dividend Yield (%)": +1, "Market Cap": -1,   # size effect: small > large
}


def _quarter_available_from(year: int, quarter: int) -> str:
    """Quarter end + PUBLISH_LAG_DAYS, as an ISO date string."""
    end_month = quarter * 3
    import datetime as dt
    # Last day of the quarter's final month.
    if end_month == 12:
        qend = dt.date(year, 12, 31)
    else:
        qend = dt.date(year, end_month + 1, 1) - dt.timedelta(days=1)
    return (qend + dt.timedelta(days=PUBLISH_LAG_DAYS)).isoformat()


def main() -> None:
    if not FUND.exists():
        print(f"Missing {FUND}\nRun scripts/crawl_fundamentals_freetier.py in the vnfund env first.")
        return

    fund = pl.read_parquet(FUND)
    print(f"Fundamentals: {fund.height} rows, {fund['ticker'].n_unique()} tickers, "
          f"{fund.group_by(['year', 'quarter']).len().height} distinct periods")
    periods = (fund.select(["year", "quarter"]).unique()
               .sort(["year", "quarter"]).rows())
    print(f"  periods: {periods}")

    fund = fund.with_columns(
        pl.struct(["year", "quarter"]).map_elements(
            lambda s: _quarter_available_from(s["year"], s["quarter"]),
            return_dtype=pl.Utf8,
        ).str.to_date().alias("available_from")
    )

    # FEASIBILITY GATE, before any IC math: a cross-sectional rank test needs
    # MANY tickers observable on the SAME date. The free tier hands each
    # ticker its own (mostly old, sometimes gappy) window, so this can easily
    # be unusable -- check it explicitly rather than discovering it via a
    # meaningless IC.
    per_period = (fund.select(["ticker", "year", "quarter"]).unique()
                  .group_by(["year", "quarter"])
                  .agg(pl.col("ticker").n_unique().alias("n_tickers"))
                  .sort(["year", "quarter"]))
    print("\nTickers per reporting period (need >= "
          f"{MIN_TICKERS_PER_DATE} for a usable cross-section):")
    for yr, qt, n in per_period.rows():
        flag = "OK" if n >= MIN_TICKERS_PER_DATE else "thin"
        print(f"  {yr}Q{qt}: {n:>4} tickers   {flag}")
    usable = per_period.filter(pl.col("n_tickers") >= MIN_TICKERS_PER_DATE)
    print(f"\n  periods with a usable cross-section: {usable.height}/{per_period.height}")
    if usable.height == 0:
        print("\n  STOP: no reporting period has enough tickers for a cross-sectional")
        print("  test. The free tier's per-ticker windows do not line up in time.")
        print("  This says nothing about whether fundamentals WORK -- only that the")
        print("  free sample cannot answer it. Decide on the paid tier some other way.")
        return

    # `volume` dtype varies across shards (vnstock int64 vs FastConnect float64,
    # 10-08-26) and a multi-file read unifies on the first shard's schema.
    price = (pl.read_parquet(
                 "data/ohlcv_*.parquet",
                 cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
             .select(["ticker", "date", "close"])
             .sort(["ticker", "date"]))
    price = price.with_columns(
        (pl.col("close").shift(-HORIZON).over("ticker") / pl.col("close") - 1.0)
        .alias("fwd_ret")
    ).drop_nulls("fwd_ret")
    print(f"Prices: {price.height} rows, {price['ticker'].n_unique()} tickers")

    print(f"\n{'=' * 88}")
    print(f"IC per factor (Spearman rank vs forward {HORIZON}d return, "
          f"publish lag {PUBLISH_LAG_DAYS}d)")
    print(f"{'=' * 88}")
    print(f"{'factor':<22}{'mean IC':>9}{'t(naive)':>8}{'t(indep)':>9}"
          f"{'dates':>7}{'indep':>6}{'D10-D1':>10}{'sign':>10}")

    results = []
    for item in sorted(fund["item_en"].unique().to_list()):
        f_item = (fund.filter(pl.col("item_en") == item)
                  .select(["ticker", "available_from", "value"])
                  .sort(["ticker", "available_from"]))
        if f_item.height == 0:
            continue

        # Backward as-of join: each (ticker, date) sees only the latest report
        # whose available_from <= date.
        joined = price.sort(["ticker", "date"]).join_asof(
            f_item, left_on="date", right_on="available_from",
            by="ticker", strategy="backward",
        ).drop_nulls("value")

        # Drop stale carries (see MAX_STALENESS_DAYS) and non-finite ratios
        # (P/E explodes when earnings ~ 0; inf/nan wreck both the decile
        # spread and any mean-based read).
        joined = joined.filter(
            ((pl.col("date") - pl.col("available_from")).dt.total_days() <= MAX_STALENESS_DAYS)
            & pl.col("value").is_finite()
            & pl.col("fwd_ret").is_finite()
        )
        if joined.height < 1000:
            print(f"{item:<22}{'--':>9}{'--':>8}{joined.height:>9}{'--':>10}"
                  f"{'thin':>7}")
            continue

        ics: list[float] = []
        ic_dates: list = []
        for (_d,), grp in joined.sort("date").group_by(["date"], maintain_order=True):
            if grp.height < MIN_TICKERS_PER_DATE:
                continue
            v = grp["value"].to_numpy()
            r = grp["fwd_ret"].to_numpy()
            if np.std(v) == 0 or np.std(r) == 0:
                continue
            ic, _ = stats.spearmanr(v, r)
            if np.isfinite(ic):
                ics.append(float(ic))
                ic_dates.append(_d)

        if len(ics) < 20:
            print(f"{item:<22}{'--':>9}{'--':>8}{len(ics):>9}{'--':>10}{'thin':>7}")
            continue

        # Window of the dates the IC was ACTUALLY computed on (not the raw
        # joined range -- a handful of tickers carry recent reports and would
        # otherwise stretch this to 2026 while contributing no usable date).
        window = f"{min(ic_dates)} .. {max(ic_dates)}"
        ic_arr = np.array(ics)
        mean_ic = ic_arr.mean()
        t_stat = mean_ic / (ic_arr.std(ddof=1) / np.sqrt(len(ic_arr))) if ic_arr.std(ddof=1) > 0 else 0.0

        # OVERLAP CORRECTION. Consecutive dates share ~19/20 of their forward
        # window, so the naive t-stat above treats ~15 independent
        # observations as 314 and inflates significance by ~sqrt(20). Recompute
        # on every HORIZON-th date only -- crude but honest, and it is the
        # number that should drive a spending decision.
        ic_indep = ic_arr[::HORIZON]
        if len(ic_indep) >= 3 and ic_indep.std(ddof=1) > 0:
            t_indep = ic_indep.mean() / (ic_indep.std(ddof=1) / np.sqrt(len(ic_indep)))
        else:
            t_indep = float("nan")

        # Decile spread on the pooled sample (catches non-monotone effects).
        vals = joined["value"].to_numpy()
        rets = joined["fwd_ret"].to_numpy()
        order = np.argsort(vals)
        n = len(order)
        d1 = rets[order[: n // 10]].mean()
        d10 = rets[order[-(n // 10):]].mean()
        spread = d10 - d1

        exp = EXPECTED_SIGN.get(item)
        sign_ok = "" if exp is None else ("ok" if np.sign(mean_ic) == np.sign(exp) else "BACKWARD")
        results.append((item, mean_ic, t_stat, t_indep, len(ics), len(ic_indep),
                        spread, sign_ok, window))
        print(f"{item:<22}{mean_ic:>+9.4f}{t_stat:>+8.2f}{t_indep:>+9.2f}"
              f"{len(ics):>7}{len(ic_indep):>6}{spread * 100:>+9.2f}%{sign_ok:>10}")

    print(f"\n{'=' * 88}\nVERDICT\n{'=' * 88}")
    if not results:
        print("  No factor had enough coverage to evaluate — the free tier's 8 periods")
        print("  did not overlap the price history usefully. Do NOT pay on this evidence.")
        return

    # Judge on the OVERLAP-CORRECTED t, not the naive one.
    strong = [r for r in results if abs(r[3]) >= 2.0 and r[7] == "ok"]
    backward = [r for r in results if r[7] == "BACKWARD" and abs(r[3]) >= 2.0]
    win_any = results[0][8] if results else "n/a"
    print(f"  IC evaluation window                       : {win_any}")
    print(f"  factors evaluated                          : {len(results)}")
    print(f"  |t_indep|>=2 AND economically right-signed : {len(strong)}")
    if strong:
        for it, ic, t, ti, nd, ni, sp, _, win in sorted(strong, key=lambda r: -abs(r[3])):
            print(f"     {it:<20} IC={ic:+.4f}  t_indep={ti:+.2f} (naive {t:+.2f})  "
                  f"n_indep={ni}  D10-D1={sp * 100:+.2f}%")
    print(f"  |t_indep|>=2 but BACKWARD-signed           : {len(backward)}"
          f"   (noise/regime, not signal)")
    if backward:
        for it, ic, t, ti, nd, ni, sp, _, win in sorted(backward, key=lambda r: -abs(r[3]))[:5]:
            print(f"     {it:<20} IC={ic:+.4f}  t_indep={ti:+.2f}")

    print("\n  Reference: a mean |IC| around 0.02-0.05 with |t|>2 is a normal,")
    print("  usable equity factor. |IC| < 0.01 is noise at this sample size.")
    print("  This is an 8-quarter window -- a positive read here justifies buying")
    print("  full history to test properly; a null read means the paid tier buys")
    print("  a longer series of the same nothing.")


if __name__ == "__main__":
    main()
