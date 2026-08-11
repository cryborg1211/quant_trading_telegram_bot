"""Does the exchange's LIMIT-DOWN count time the MR knife-catch? (READ-ONLY)

VERDICT: UNTESTABLE on the available history — `floors` is not a decade-long
series, it is a ~1.5-year one. SSI reports a literal 0 (not null) for the eras it
does not carry the field, so the backfill looks complete while being structurally
empty. Fraction of VNINDEX sessions with floors>=1, by year:

    2016 0.000   2017 0.000   2018 0.000   2019 0.000   2020 0.008
    2021 0.004   2022 0.000   2023 0.000   2024 0.000
    2025 0.502   2026 0.878

2022 was a ~35% VNINDEX drawdown; it did not have zero limit-down sessions. So
inside a chronological split the flag is a TIME DUMMY, not a capitulation
measure: train (to 2025-08) is 1.9% floors>=1 and holdout is 86%, which is why
the first run produced `floors>=1` with 0 OOF fires and rank buckets covering
99.7% of rows. Any bucket difference would measure the 2025/2026 regime shift.

Re-run this once there are ~3+ years of post-2025 coverage. Until then the
honest answer is "not enough data", not a precision number.

The percentile-rank buckets below also need rethinking before a re-run: with a
zero-inflated series, `(today >= window).mean()` scores a ZERO-floor day at ~0.89
because it ties every other zero, which inverts the rank's meaning.


The MR-LGBM (BAT DAY) signal tries to buy capitulation bounces. Everything used
to time it so far is a PROXY built from rolling returns:
  * regime-gate            REJECTED (2026-07-22)
  * RSI-direction split    REJECTED — deep-oversold fires are LESS reliable
  * breadth inflection     PROMISING but did NOT replicate on a wider holdout

`floors` from the exchange's own DailyIndex feed (backfilled 2026-08-11,
2016-01-04..2026-08-10) is not a proxy: it is the count of names that actually
closed limit-down. That is capitulation measured directly rather than inferred,
and nothing in this system has had access to it before.

Hypothesis: MR fires on days with an ELEVATED limit-down count are the real
capitulation bounces and should show higher precision than fires on ordinary
down days.

METHOD — deliberately identical to analyze_mr_breadth_inflection.py so the
result is comparable to the breadth-inflection number it is meant to beat:
the same purged-OOF machinery, the same shipped tau*, the same reliability
floor, the same chronological train/holdout split, and the SAME extended
out-of-training window that killed breadth inflection. `floors` is joined by
DATE only and is strictly backward-looking (it is the settled EOD count for
that session), so it is leak-free by construction — and it is NOT added as a
training feature, exactly as breadth was not.

Reads only. No model, no artifact, no parquet is written.

Run: python scripts/analyze_mr_limit_down_context.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.data.market_breadth_crawler import _DEFAULT_PARQUET as BREADTH_PARQUET  # noqa: E402
from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features  # noqa: E402
from src.models.train_mr_lgbm import (  # noqa: E402
    EMBARGO_BARS,
    N_SPLITS,
    _spw,
    chrono_split,
    label_3d_bounce,
    load_ohlcv,
    make_lgbm,
    purged_oof,
)

SHIPPED_TAU = 0.96          # production tau* (models/mr/mr_threshold.json)
MIN_RELIABLE_FIRES = 10     # below this precision is too noisy to read
EXTENDED_HOLDOUT_DAYS = 1095  # same ~3y window that failed to confirm breadth
INDEX_ID = "VNINDEX"        # the ONLY index whose A/D counts are populated;
# ceilings/floors are populated for VN30/VN100 too, but VNINDEX is the whole
# market and therefore the right capitulation denominator.


def _limit_down_series() -> pd.DataFrame:
    """Per-date limit-down / limit-up counts plus a rolling percentile rank.

    The RANK is what the buckets use, not the raw count: `floors` is
    zero-inflated (nonzero on only 809 of 7200 rows) and its scale drifts with
    the number of listed names over a decade, so an absolute cut like ">5" is
    not comparable across 2016 and 2026. A trailing-window percentile is.
    """
    path = Path(BREADTH_PARQUET)
    if not path.exists():
        raise SystemExit(
            f"{path} missing — run scripts/backfill_market_breadth.py --commit first.")
    pdf = (pl.read_parquet(path)
             .filter(pl.col("index_id") == INDEX_ID)
             .select(["date", "floors", "ceilings", "advances", "declines"])
             .sort("date")
             .to_pandas())
    # Trailing 1-year percentile of TODAY's floor count within the preceding
    # window — strictly backward-looking (the window ends at the current row),
    # and min_periods keeps the warm-up out rather than ranking against a
    # half-empty window.
    pdf["floors_rank_1y"] = (
        pdf["floors"].rolling(252, min_periods=120)
        .apply(lambda w: (w[-1] >= w).mean(), raw=True)
    )
    pdf["net_ad"] = pdf["advances"] - pdf["declines"]
    return pdf


def _print_bucket_table(title: str, buckets: dict, p: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'bucket':<30}{'n_rows':>8}{'base_rate':>11}{'fires':>8}"
          f"{'precision':>11}{'recall':>9}  note")
    for label, mask in buckets.items():
        pr, yr = p[mask], y[mask]
        if len(pr) == 0:
            print(f"{label:<30}{'(empty)':>8}")
            continue
        fr = pr >= SHIPPED_TAU
        nfr = int(fr.sum())
        tpr = int((fr & (yr == 1)).sum())
        prec = tpr / nfr if nfr else float("nan")
        rec = tpr / max(int(yr.sum()), 1)
        note = "" if nfr >= MIN_RELIABLE_FIRES else f"<{MIN_RELIABLE_FIRES} fires, noisy"
        print(f"{label:<30}{len(pr):>8}{yr.mean():>11.4f}{nfr:>8}"
              f"{prec:>11.3f}{rec:>9.3f}  {note}")


def _buckets(d: pd.DataFrame) -> dict:
    rank = d["floors_rank_1y"].to_numpy()
    floors = d["floors"].to_numpy()
    return {
        "ANY floor (baseline)": np.ones(len(d), dtype=bool),
        "floors == 0 (calm)": floors == 0,
        "floors >= 1": floors >= 1,
        "floors rank top 10%": rank >= 0.90,
        "floors rank top 25%": rank >= 0.75,
        "floors rank bottom 50%": rank < 0.50,
    }


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building MR features + 3d-bounce labels ...")
    feat = build_mr_features(ohlcv)
    feat = label_3d_bounce(feat)

    ld = _limit_down_series()
    print(f"limit-down series: {len(ld)} sessions "
          f"{ld['date'].min()} .. {ld['date'].max()}")

    feat["date"] = pd.to_datetime(feat["date"])
    ld["date"] = pd.to_datetime(ld["date"])
    feat = feat.merge(ld, on="date", how="left")
    before = len(feat)
    feat = feat.dropna(subset=["floors", "floors_rank_1y"])
    print(f"rows with a limit-down reading: {len(feat)}/{before} "
          f"(the breadth parquet starts 2016, MR features may reach further back)")

    # purged_oof needs [date, ticker] chronological order. Feature builders
    # leave [ticker, date]; without this re-sort the folds collapse to a few
    # hundred rows and OOF precision silently reads ~0.34 against a known 0.578.
    feat = feat.sort_values(["date", "ticker"]).reset_index(drop=True)

    train, test = chrono_split(feat)
    cols = list(MR_FEATURE_COLUMNS)   # floors is CONTEXT, never a training feature

    def X(d: pd.DataFrame) -> np.ndarray:
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    print(f"\ntrain={len(train)} rows   holdout={len(test)} rows")
    print(f"Running purged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS}) "
          f"— identical to production ...")
    ytr = train["y"].to_numpy(np.int64)
    # `end` is the label's barrier-touch date (`t1`), not the entry date — the
    # embargo is defined on when a label RESOLVES, so passing the entry date
    # twice would purge the wrong rows.
    oof = purged_oof(X(train), ytr,
                     pd.to_datetime(train["date"]).to_numpy(),
                     pd.to_datetime(train["t1"]).to_numpy())

    # purged_oof leaves NaN wherever a row never landed in a validation fold.
    # Buckets MUST be computed on the same mask or the context flags misalign
    # with the probabilities row-for-row.
    mask = np.isfinite(oof)
    print(f"  OOF coverage: {int(mask.sum())}/{len(oof)} rows scored")
    _print_bucket_table(
        f"OOF (train) — MR precision by limit-down context, tau*={SHIPPED_TAU}",
        _buckets(train[mask]), oof[mask], ytr[mask])

    print("\nFitting on train, scoring the chronological holdout ...")
    model = make_lgbm(_spw(ytr))
    model.fit(X(train), ytr)
    p_te = model.predict_proba(X(test))[:, 1]
    yte = test["y"].to_numpy(np.int64)
    _print_bucket_table(
        "HOLDOUT — same buckets, out of training",
        _buckets(test), p_te, yte)

    # The extended window is the step that killed breadth inflection: the
    # 1-year holdout was too thin to separate signal from noise.
    cutoff = feat["date"].max() - pd.Timedelta(days=EXTENDED_HOLDOUT_DAYS)
    ext = feat[feat["date"] >= cutoff]
    if len(ext) > len(test):
        print(f"\nExtended out-of-training window: {cutoff.date()} .. "
              f"{feat['date'].max().date()}  ({len(ext)} rows)")
        p_ext = model.predict_proba(X(ext))[:, 1]
        _print_bucket_table(
            f"EXTENDED ({EXTENDED_HOLDOUT_DAYS}d) — the check breadth inflection failed",
            _buckets(ext), p_ext, ext["y"].to_numpy(np.int64))

    print(f"\nRead this against the bars the MR model already clears: OOF 0.578 "
          f"overall and a 0.60 target. A bucket only matters if it beats those "
          f"AND survives the extended window with >={MIN_RELIABLE_FIRES} fires — "
          f"breadth inflection cleared the first and failed the second.")


if __name__ == "__main__":
    main()
