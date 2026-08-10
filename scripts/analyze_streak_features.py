"""Do UP/DOWN streak features carry signal the model does not already have?
(09-08-26, user-proposed.)

THE IDEA
--------
Streak LENGTH is a different quantity from momentum MAGNITUDE. A name up 10
pct across 3 violent sessions and one up 10 pct across 10 quiet consecutive
sessions share a `mom20` but describe different behaviour. Nothing in the
live 13-feature set encodes run length: `imp_up_streak` exists only inside
`impulse_features.py`, which belongs to the REJECTED fast-attack sub-model,
and a DOWN-streak has never been built at all.

VN-specific reason to care: HOSE caps a session at +/-7 pct, so a name
pinned near the ceiling (or floor) for N consecutive sessions is a real
structural state rather than a smooth drift.

TWO GATES, BOTH MUST PASS
-------------------------
1. SIGNAL -- cross-sectional IC vs forward returns, same methodology as the
   foreign-flow tests (per-date Spearman, then mean IC with an
   overlap-corrected t-stat; a 20d forward window makes consecutive dates
   ~95 pct redundant and inflates a naive t by ~sqrt(20)).
2. ORTHOGONALITY -- correlation against the 13 features already in the
   model. This is the gate that actually matters: this session has now
   rejected six OHLCV-derived ideas in a row, and the recurring reason is
   that new price features re-encode information `mom20_xsz` / `rs_10_xsz`
   / `rs_20_xsz` / `overext_*` already carry. A streak feature correlating
   ~0.9 with existing momentum is not new information, however good its IC
   looks, and adding it would mostly buy PBO inflation (macro-GBM went
   42.7 -> 87 pct PBO on exactly that mistake).

Only a feature that clears BOTH is worth a recipe bump + full retrain.

READ-ONLY. Run: python scripts/analyze_streak_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

HORIZONS = (5, 20)
MIN_TICKERS_PER_DATE = 30
STREAK_CAP = 15          # beyond this the bucket is too rare to rank on

STREAK_COLS = [
    "up_streak", "down_streak", "signed_streak",
    "streak_return", "streak_intensity",
]

# The live model's feature set (checkpoint-verified 08-08-26).
MODEL_FEATURES = [
    "close_fd_xsz", "volume_fd_xsz", "mom20_xsz", "overext_5_xsz",
    "overext_20_xsz", "rs_10_xsz", "rs_20_xsz", "smart_money_20_xsz",
    "vol_squeeze_xsz", "gap_risk_xsz", "amihud_liquidity_xsz",
    "hl_range_ratio_xsz", "market_regime",
]


def build_streaks(panel: pl.DataFrame) -> pl.DataFrame:
    """Consecutive up/down run lengths ending at t, plus run magnitude.

    Leak-safe by construction: every quantity is a function of closes up to
    and including t. Run length is computed with the standard
    cumulative-sum-of-breaks trick (a break resets the group id), matching
    `impulse_features.py::imp_up_streak`.
    """
    p = panel.sort(["ticker", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret1")
    )
    p = p.with_columns([
        (pl.col("ret1") > 0).fill_null(False).alias("is_up"),
        (pl.col("ret1") < 0).fill_null(False).alias("is_dn"),
    ])
    # Group id increments whenever the run breaks; rank within group = length.
    p = p.with_columns([
        (~pl.col("is_up")).cum_sum().over("ticker").alias("_up_grp"),
        (~pl.col("is_dn")).cum_sum().over("ticker").alias("_dn_grp"),
    ])
    p = p.with_columns([
        pl.when(pl.col("is_up"))
          .then(pl.col("is_up").cum_sum().over(["ticker", "_up_grp"]))
          .otherwise(0).clip(0, STREAK_CAP).alias("up_streak"),
        pl.when(pl.col("is_dn"))
          .then(pl.col("is_dn").cum_sum().over(["ticker", "_dn_grp"]))
          .otherwise(0).clip(0, STREAK_CAP).alias("down_streak"),
    ])
    p = p.with_columns([
        (pl.col("up_streak") - pl.col("down_streak")).alias("signed_streak"),
        # Cumulative return earned across the CURRENT run (length x magnitude).
        pl.when(pl.col("is_up"))
          .then(pl.col("ret1").cum_sum().over(["ticker", "_up_grp"]))
          .when(pl.col("is_dn"))
          .then(pl.col("ret1").cum_sum().over(["ticker", "_dn_grp"]))
          .otherwise(0.0).alias("streak_return"),
    ])
    # Average daily move inside the run -- separates "10 quiet days" from
    # "3 violent ones" even when the run lengths match.
    denom = (pl.col("up_streak") + pl.col("down_streak")).clip(1, None)
    p = p.with_columns((pl.col("streak_return") / denom).alias("streak_intensity"))
    return p.drop(["_up_grp", "_dn_grp"])


def ic_report(df: pl.DataFrame, col: str, ret_col: str, horizon: int) -> tuple:
    ics: list[float] = []
    for (_d,), grp in df.sort("date").group_by(["date"], maintain_order=True):
        if grp.height < MIN_TICKERS_PER_DATE:
            continue
        v, r = grp[col].to_numpy(), grp[ret_col].to_numpy()
        ok = np.isfinite(v) & np.isfinite(r)
        if ok.sum() < MIN_TICKERS_PER_DATE or np.std(v[ok]) == 0 or np.std(r[ok]) == 0:
            continue
        ic, _ = stats.spearmanr(v[ok], r[ok])
        if np.isfinite(ic):
            ics.append(float(ic))
    if len(ics) < 30:
        return (float("nan"), float("nan"), float("nan"), len(ics), 0)
    a = np.array(ics)
    mean_ic = a.mean()
    t_naive = mean_ic / (a.std(ddof=1) / np.sqrt(len(a))) if a.std(ddof=1) > 0 else 0.0
    indep = a[::horizon]                      # de-overlap the forward window
    t_indep = (indep.mean() / (indep.std(ddof=1) / np.sqrt(len(indep)))
               if len(indep) >= 3 and indep.std(ddof=1) > 0 else float("nan"))
    return (mean_ic, t_naive, t_indep, len(a), len(indep))


def main() -> None:
    from src.backtest.pipeline import RunConfig, materialize_dataset

    print("Materializing dataset (need the live 13 features for the orthogonality gate) ...")
    ds = materialize_dataset(RunConfig())
    panel = ds.panel
    have_model_feats = [c for c in MODEL_FEATURES if c in panel.columns]
    print(f"  panel rows={panel.height}  model features present={len(have_model_feats)}/{len(MODEL_FEATURES)}")

    df = build_streaks(panel)
    for h in HORIZONS:
        df = df.with_columns(
            (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0).alias(f"fwd_{h}d")
        )
    df = df.drop_nulls(["fwd_20d"])
    print(f"  rows after forward-return join: {df.height}")

    print(f"\n{'=' * 84}\nGATE 1 -- SIGNAL (cross-sectional IC vs forward return)\n{'=' * 84}")
    print(f"{'feature':<20}{'horizon':>8}{'mean IC':>10}{'t(naive)':>10}{'t(indep)':>10}{'dates':>8}{'indep':>7}")
    signal_ok: dict[str, bool] = {}
    for col in STREAK_COLS:
        best_t = 0.0
        for h in HORIZONS:
            ic, tn, ti, nd, ni = ic_report(df, col, f"fwd_{h}d", h)
            if np.isnan(ic):
                print(f"{col:<20}{h:>8}{'--':>10}{'--':>10}{'--':>10}{nd:>8}{ni:>7}")
                continue
            print(f"{col:<20}{h:>8}{ic:>+10.4f}{tn:>+10.2f}{ti:>+10.2f}{nd:>8}{ni:>7}")
            if np.isfinite(ti):
                best_t = max(best_t, abs(ti))
        signal_ok[col] = best_t >= 2.0

    print(f"\n{'=' * 84}\nGATE 2 -- ORTHOGONALITY (|corr| vs the 13 live model features)\n{'=' * 84}")
    if not have_model_feats:
        print("  Model features are NOT in the panel — cannot run this gate here.")
        print("  Without it a good IC is uninterpretable: it may just be momentum re-encoded.")
        return

    sample = df.sample(n=min(200_000, df.height), seed=42)
    print(f"{'feature':<20}{'max |corr|':>12}  worst-offender")
    orth_ok: dict[str, bool] = {}
    for col in STREAK_COLS:
        v = sample[col].to_numpy().astype(float)
        worst_c, worst_n = 0.0, "-"
        for mf in have_model_feats:
            m = sample[mf].to_numpy().astype(float)
            ok = np.isfinite(v) & np.isfinite(m)
            if ok.sum() < 1000 or np.std(v[ok]) == 0 or np.std(m[ok]) == 0:
                continue
            c = abs(float(np.corrcoef(v[ok], m[ok])[0, 1]))
            if c > worst_c:
                worst_c, worst_n = c, mf
        orth_ok[col] = worst_c < 0.70
        flag = "" if worst_c < 0.70 else "   <-- REDUNDANT"
        print(f"{col:<20}{worst_c:>12.3f}  {worst_n}{flag}")

    print(f"\n{'=' * 84}\nVERDICT\n{'=' * 84}")
    winners = [c for c in STREAK_COLS if signal_ok.get(c) and orth_ok.get(c)]
    for c in STREAK_COLS:
        s = "signal OK" if signal_ok.get(c) else "no signal"
        o = "orthogonal" if orth_ok.get(c) else "redundant"
        print(f"  {c:<20} {s:<11} + {o}")
    print(f"\n  clears BOTH gates: {winners if winners else 'NONE'}")
    if winners:
        print("  -> worth a recipe bump + retrain A/B. Nothing here proves it")
        print("     improves the BOOK; that still needs the backtest.")
    else:
        print("  -> not worth a retrain. Either no cross-sectional signal, or the")
        print("     signal is already carried by an existing feature.")
    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
