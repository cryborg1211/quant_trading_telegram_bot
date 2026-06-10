"""Per-trade net-edge profiler — score decile x horizon x regime.

Loads the frozen training checkpoint, re-materializes OOS data, scores every
OOS row, then decomposes P&L by model-confidence decile and HMM regime.

Answers: "Which trades should I stop taking?"

Run:
    python scripts/edge_profiler.py
    python scripts/edge_profiler.py --checkpoint models/saved/v3_training_checkpoint.joblib
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
os.chdir(_REPO)
sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.backtest.pipeline import (
    RunConfig,
    AlignedData,
    TRADING_DAYS,
    configure_logging,
    phase,
    materialize_dataset,
    subset_features,
)
from src.models.tabular_ensemble import TabularEnsemble
from src.models.macro_risk_hmm import build_market_proxy_returns
from src.execution.vn_cost_model import FeeSchedule

LOGGER = logging.getLogger("edge_profiler")
CHECKPOINT_PATH = Path("models/saved/v3_training_checkpoint.joblib")

FEES = FeeSchedule()
ROUND_TRIP_PCT = FEES.round_trip_pct()  # ~0.43%


# ---------------------------------------------------------------------------
# Build per-trade OOS frame
# ---------------------------------------------------------------------------

def _build_oos_trades(
    aligned: AlignedData,
    ensemble: TabularEnsemble,
    test_mask: np.ndarray,
    panel: pl.DataFrame,
    p_bull_series: pd.Series | None,
) -> pd.DataFrame:
    """Score every OOS row, join with label outcomes and regime."""
    X_oos = aligned.X[test_mask]
    y_oos = aligned.y[test_mask].astype(int)
    dates_oos = aligned.dates[test_mask]
    tickers_oos = aligned.tickers[test_mask]

    p_up = np.asarray(ensemble.predict_proba(X_oos), dtype=np.float32).ravel()

    df = pd.DataFrame({
        "date": dates_oos,
        "ticker": tickers_oos,
        "p_up": p_up,
        "y_true": y_oos,  # 0=DOWN, 1=FLAT, 2=UP
    })

    # Score decile (1=lowest conviction, 10=highest)
    df["score_decile"] = pd.qcut(df["p_up"], 10, labels=False, duplicates="drop") + 1

    # HMM regime overlay
    if p_bull_series is not None:
        bull_map = {
            pd.Timestamp(d).date(): float(v)
            for d, v in p_bull_series.dropna().items()
        }
        df["p_bull"] = df["date"].map(bull_map)
        df["regime"] = np.where(df["p_bull"] >= 0.5, "BULL", "BEAR")
    else:
        df["p_bull"] = 1.0
        df["regime"] = "UNKNOWN"

    # Forward returns from OHLCV panel (T+3, T+5, T+10, T+20 from close)
    pdf = panel.to_pandas() if isinstance(panel, pl.DataFrame) else panel.copy()
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.date
    pdf = pdf.sort_values(["ticker", "date"])

    for horizon in [3, 5, 10, 20]:
        col = f"fwd_ret_t{horizon}"
        pdf[col] = (
            pdf.groupby("ticker", sort=False)["close"]
            .transform(lambda s: s.shift(-horizon) / s - 1.0)
        )

    price_cols = ["ticker", "date", "close", "open", "volume",
                  "fwd_ret_t3", "fwd_ret_t5", "fwd_ret_t10", "fwd_ret_t20"]
    existing = [c for c in price_cols if c in pdf.columns]
    prices = pdf[existing].drop_duplicates(subset=["ticker", "date"])

    df = df.merge(prices, on=["ticker", "date"], how="left")

    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-decile summary: count, win_rate, gross T+N returns, net returns."""
    rows = []
    for decile in sorted(df["score_decile"].unique()):
        sub = df[df["score_decile"] == decile]
        n = len(sub)
        p_up_mean = sub["p_up"].mean()
        p_up_min = sub["p_up"].min()
        p_up_max = sub["p_up"].max()

        row = {
            "decile": int(decile),
            "n": n,
            "p_up_mean": round(p_up_mean, 4),
            "p_up_range": f"{p_up_min:.3f}-{p_up_max:.3f}",
            "label_up_pct": round((sub["y_true"] == 2).mean() * 100, 1),
            "label_down_pct": round((sub["y_true"] == 0).mean() * 100, 1),
        }

        for h in [3, 5, 10, 20]:
            col = f"fwd_ret_t{h}"
            if col in sub.columns:
                valid = sub[col].dropna()
                gross = valid.mean() * 100 if len(valid) > 0 else None
                net = (gross - ROUND_TRIP_PCT * 100) if gross is not None else None
                wr = (valid > 0).mean() * 100 if len(valid) > 0 else None
                row[f"gross_t{h}"] = round(gross, 3) if gross is not None else None
                row[f"net_t{h}"] = round(net, 3) if net is not None else None
                row[f"wr_t{h}"] = round(wr, 1) if wr is not None else None

        rows.append(row)
    return pd.DataFrame(rows)


def _regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime x top-decile breakdown."""
    top = df[df["score_decile"] >= 8]  # top 30% conviction
    rows = []
    for regime in sorted(df["regime"].unique()):
        sub = top[top["regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        row = {"regime": regime, "n": n, "p_up_mean": round(sub["p_up"].mean(), 4)}
        for h in [5, 20]:
            col = f"fwd_ret_t{h}"
            if col in sub.columns:
                valid = sub[col].dropna()
                gross = valid.mean() * 100 if len(valid) > 0 else None
                net = (gross - ROUND_TRIP_PCT * 100) if gross is not None else None
                row[f"gross_t{h}"] = round(gross, 3) if gross is not None else None
                row[f"net_t{h}"] = round(net, 3) if net is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def _stability_analysis(df: pd.DataFrame, thr: float = 0.45) -> None:
    """Concentration + temporal stability of the >= thr cohort.

    A +2% net edge on 1% of signals is only tradeable if it is spread across
    years and tickers — not one rally month or one hot ticker.
    """
    bar = "=" * 80
    sub = df[df["p_up"] >= thr].copy()
    if len(sub) == 0:
        print(f"\nNo signals at threshold {thr}.")
        return
    sub["year"] = pd.to_datetime(sub["date"].astype(str)).dt.year

    print(f"\n{bar}")
    print(f" STABILITY CHECK — cohort p_up >= {thr}  (n={len(sub):,})")
    print(bar)

    # Per-year breakdown
    rows = []
    for year, grp in sub.groupby("year"):
        valid = grp["fwd_ret_t20"].dropna()
        gross = valid.mean() * 100 if len(valid) else float("nan")
        rows.append({
            "year": int(year),
            "n": len(grp),
            "n_days": grp["date"].nunique(),
            "n_tickers": grp["ticker"].nunique(),
            "gross_t20": round(gross, 3),
            "net_t20": round(gross - ROUND_TRIP_PCT * 100, 3),
            "wr_t20": round((valid > 0).mean() * 100, 1) if len(valid) else None,
        })
    print("\n  PER-YEAR:")
    print(pd.DataFrame(rows).to_string(index=False))

    # Ticker concentration
    by_ticker = sub.groupby("ticker").agg(
        n=("p_up", "size"),
        net_t20=("fwd_ret_t20", lambda s: s.mean() * 100 - ROUND_TRIP_PCT * 100),
    ).sort_values("n", ascending=False)
    top5_share = by_ticker["n"].head(5).sum() / len(sub) * 100
    print(f"\n  TICKER CONCENTRATION: {sub['ticker'].nunique()} tickers, "
          f"top-5 hold {top5_share:.1f}% of signals")
    print(by_ticker.head(10).round(3).to_string())

    # Date concentration
    by_day = sub.groupby("date").size().sort_values(ascending=False)
    top10_day_share = by_day.head(10).sum() / len(sub) * 100
    print(f"\n  DATE CONCENTRATION: {len(by_day)} distinct days, "
          f"top-10 days hold {top10_day_share:.1f}% of signals")
    print(f"  busiest days: {dict(by_day.head(5))}")

    # Net edge excluding the single best ticker and best day (fragility probe)
    if len(by_ticker) > 1:
        ex_ticker = sub[sub["ticker"] != by_ticker.index[0]]["fwd_ret_t20"].dropna()
        print(f"\n  net_t20 excluding top ticker ({by_ticker.index[0]}): "
              f"{ex_ticker.mean() * 100 - ROUND_TRIP_PCT * 100:+.3f}%")
    if len(by_day) > 1:
        ex_day = sub[sub["date"] != by_day.index[0]]["fwd_ret_t20"].dropna()
        print(f"  net_t20 excluding busiest day ({by_day.index[0]}): "
              f"{ex_day.mean() * 100 - ROUND_TRIP_PCT * 100:+.3f}%")
    print(bar)


def _threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative edge at various P(UP) thresholds — the dispatch cutoff study."""
    rows = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = df[df["p_up"] >= thr]
        n = len(sub)
        if n == 0:
            continue
        row = {
            "threshold": thr,
            "n_signals": n,
            "pct_of_total": round(n / len(df) * 100, 1),
            "label_up_pct": round((sub["y_true"] == 2).mean() * 100, 1),
        }
        for h in [5, 20]:
            col = f"fwd_ret_t{h}"
            if col in sub.columns:
                valid = sub[col].dropna()
                gross = valid.mean() * 100 if len(valid) > 0 else None
                net = (gross - ROUND_TRIP_PCT * 100) if gross is not None else None
                wr = (valid > 0).mean() * 100 if len(valid) > 0 else None
                row[f"gross_t{h}"] = round(gross, 3) if gross is not None else None
                row[f"net_t{h}"] = round(net, 3) if net is not None else None
                row[f"wr_t{h}"] = round(wr, 1) if wr is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _print_table(title: str, df: pd.DataFrame) -> None:
    bar = "=" * 80
    print(f"\n{bar}")
    print(f" {title}")
    print(bar)
    print(df.to_string(index=False))
    print(bar)


def _render_verdict(decile_df: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    bar = "=" * 80
    print(f"\n{bar}")
    print(" ACTIONABLE FINDINGS")
    print(bar)

    # 1. Which deciles destroy value?
    if "net_t20" in decile_df.columns:
        net_col = "net_t20"
        horizon_label = "T+20"
    elif "net_t5" in decile_df.columns:
        net_col = "net_t5"
        horizon_label = "T+5"
    else:
        print("  Insufficient forward-return data for verdict.")
        return

    losers = decile_df[decile_df[net_col].notna() & (decile_df[net_col] < 0)]
    winners = decile_df[decile_df[net_col].notna() & (decile_df[net_col] > 0)]

    if len(losers) > 0:
        loser_deciles = losers["decile"].tolist()
        loser_n = losers["n"].sum()
        total_n = decile_df["n"].sum()
        print(f"\n  FINDING 1: Deciles {loser_deciles} are NET NEGATIVE at {horizon_label}.")
        print(f"  These represent {loser_n:,} / {total_n:,} OOS trades ({loser_n/total_n*100:.0f}%).")
        print(f"  RECOMMENDATION: raise the dispatch threshold to exclude them.")

    if len(winners) > 0:
        best = winners.loc[winners[net_col].idxmax()]
        print(f"\n  FINDING 2: Best decile = {int(best['decile'])} with {horizon_label} "
              f"net = {best[net_col]:+.3f}%")

    # 2. Optimal threshold
    if "net_t20" in threshold_df.columns:
        viable = threshold_df[threshold_df["net_t20"].notna() & (threshold_df["net_t20"] > 0)]
        if len(viable) > 0:
            best_thr = viable.iloc[0]  # lowest threshold that's still positive
            print(f"\n  FINDING 3: Minimum viable dispatch threshold = {best_thr['threshold']:.2f}")
            print(f"  At this gate: {int(best_thr['n_signals']):,} signals "
                  f"({best_thr['pct_of_total']:.1f}% of universe), "
                  f"T+20 net = {best_thr['net_t20']:+.3f}%")

            optimal = viable.loc[viable["net_t20"].idxmax()]
            if optimal["threshold"] != best_thr["threshold"]:
                print(f"  Optimal threshold (max net) = {optimal['threshold']:.2f} "
                      f"with T+20 net = {optimal['net_t20']:+.3f}% "
                      f"({int(optimal['n_signals']):,} signals)")
        else:
            print("\n  FINDING 3: NO threshold produces positive net T+20 returns.")
            print("  The model does not generate cost-clearing alpha at any conviction level.")
            print("  Before adding features or retraining, verify label quality and cost assumptions.")

    # 3. Cost budget
    print(f"\n  COST BUDGET: round-trip = {ROUND_TRIP_PCT*100:.3f}% "
          f"(buy {FEES.buy_fee_pct()*100:.3f}% + sell {FEES.sell_fee_pct()*100:.3f}%)")
    print(f"  Any signal with gross < {ROUND_TRIP_PCT*100:.3f}% is a guaranteed loss.")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRADES_CACHE = Path("data/edge_profiler_trades.parquet")


def main(checkpoint_path: Path = CHECKPOINT_PATH, use_cache: bool = False) -> None:
    configure_logging()
    t0 = time.perf_counter()

    if use_cache and TRADES_CACHE.exists():
        trades = pd.read_parquet(TRADES_CACHE)
        print(f"Loaded cached trade frame: {len(trades):,} rows ({TRADES_CACHE})")
        _run_analyses(trades, t0)
        return

    # 1. Load checkpoint
    with phase("Load checkpoint"):
        if not checkpoint_path.exists():
            print(f"[error] Checkpoint not found: {checkpoint_path}")
            print("  Run `python train_models.py` first.")
            sys.exit(1)
        ckpt = joblib.load(checkpoint_path)
        cfg: RunConfig = ckpt["train_cfg"]
        features: list[str] = list(ckpt["tabular_features"])
        cutoff = ckpt["cutoff"]
        trained: list[tuple[int, TabularEnsemble]] = list(ckpt["ensembles"])
        macro_hmm = ckpt.get("macro_hmm")
        best_seed, best_ensemble = trained[0]  # use first seed
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Seeds: {[s for s, _ in trained]}  features: {len(features)}  cutoff: {cutoff}")
        print(f"  Horizon: T+{cfg.tb_horizon}  PT: {cfg.tb_pt}s  SL: {cfg.tb_sl}s")

    # 2. Re-materialize dataset
    ds = materialize_dataset(cfg)
    with phase("Subset features"):
        ds.aligned = subset_features(ds.aligned, ds.all_features, features)

    # 3. HMM regime
    p_bull_series = None
    if macro_hmm is not None:
        try:
            market_ret = build_market_proxy_returns(ds.panel)
            p_bull_series = macro_hmm.p_bull_series(market_ret, filtered=True)
        except Exception as exc:
            LOGGER.warning("HMM failed: %s — proceeding without regime overlay.", exc)

    # 4. Build OOS trade frame
    test_mask = ds.aligned.dates >= cutoff
    n_oos = int(test_mask.sum())
    print(f"\nOOS rows: {n_oos:,}  (cutoff: {cutoff})")

    with phase("Score OOS and build trade frame"):
        trades = _build_oos_trades(
            ds.aligned, best_ensemble, test_mask, ds.panel, p_bull_series
        )
    print(f"Trade frame: {len(trades):,} rows")
    print(f"Score range: [{trades['p_up'].min():.4f}, {trades['p_up'].max():.4f}]")
    print(f"Label distribution: DOWN={int((trades['y_true']==0).sum())}  "
          f"FLAT={int((trades['y_true']==1).sum())}  UP={int((trades['y_true']==2).sum())}")

    trades.to_parquet(TRADES_CACHE, index=False)
    print(f"Cached trade frame -> {TRADES_CACHE}")

    _run_analyses(trades, t0)


def _run_analyses(trades: pd.DataFrame, t0: float) -> None:
    with phase("Decile analysis"):
        decile_df = _decile_analysis(trades)
    _print_table("SCORE-DECILE EDGE PROFILE (per trade, OOS)", decile_df)

    with phase("Regime x top-decile analysis"):
        regime_df = _regime_analysis(trades)
    if len(regime_df) > 0:
        _print_table("REGIME x TOP-DECILE (deciles 8-10, highest conviction)", regime_df)

    with phase("Threshold dispatch study"):
        threshold_df = _threshold_analysis(trades)
    _print_table("DISPATCH THRESHOLD STUDY (cumulative above threshold)", threshold_df)

    with phase("Stability check (threshold 0.45)"):
        _stability_analysis(trades, thr=0.45)

    _render_verdict(decile_df, threshold_df)

    elapsed = time.perf_counter() - t0
    print(f"\nWall-clock: {elapsed:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-trade net-edge profiler")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--cached", action="store_true",
                   help="Reuse data/edge_profiler_trades.parquet instead of re-scoring")
    args = p.parse_args()
    main(args.checkpoint, use_cache=args.cached)
