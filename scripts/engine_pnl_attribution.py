"""PnL attribution for one walk-forward engine run.

The per-row profiler says the model has a real net edge (+1.1% net T+20 in
the liquid universe at thr 0.40), yet the engine loses ~85% with a full book.
This script runs the engine ONCE (seed 42, sig_thr 0.40 — the catastrophic
config) and decomposes the loss: cost drag vs gross selection PnL vs churn.

Run:
    python scripts/engine_pnl_attribution.py
    python scripts/engine_pnl_attribution.py --sig-thr 0.45
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
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
    configure_logging,
    phase,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.tabular_ensemble import make_ensemble_oracle
from src.models.macro_risk_hmm import build_market_proxy_returns
from src.backtest.walk_forward import WalkForwardEngine, WalkForwardConfig
from src.portfolio.construction import PortfolioConstraints
from src.execution.vn_cost_model import ExecutionConfig

CHECKPOINT_PATH = Path("models/saved/v3_training_checkpoint.joblib")


def _run_engine(sig_thr: float, rebalance_frequency: int | None = None) -> tuple:
    ckpt = joblib.load(CHECKPOINT_PATH)
    cfg: RunConfig = ckpt["train_cfg"]
    if rebalance_frequency is not None:
        cfg.rebalance_frequency = rebalance_frequency
    features = list(ckpt["tabular_features"])
    cutoff = ckpt["cutoff"]
    seed, ensemble = list(ckpt["ensembles"])[0]
    macro_hmm = ckpt.get("macro_hmm")
    print(f"seed={seed}  sig_thr={sig_thr}  cutoff={cutoff}")

    ds = materialize_dataset(cfg)
    ds.aligned = subset_features(ds.aligned, ds.all_features, features)
    corporate_actions = load_corporate_actions(cfg)

    p_bull_series = None
    if macro_hmm is not None:
        market_ret = build_market_proxy_returns(ds.panel)
        p_bull_series = macro_hmm.p_bull_series(market_ret, filtered=True)

    # Mirror run_backtest.run_oos exactly
    oracle = make_ensemble_oracle(ensemble)
    buffer = 80
    all_dates = sorted(ds.panel["date"].unique().to_list())
    cutoff_idx = next((i for i, d in enumerate(all_dates) if d >= cutoff), 0)
    buf_start = all_dates[max(0, cutoff_idx - buffer)]
    sub = ds.panel.filter(pl.col("date") >= buf_start)

    wf_cfg = WalkForwardConfig(
        seq_len=1,
        feature_cols=features,
        initial_capital=cfg.initial_capital, max_positions=cfg.max_positions,
        rebalance_frequency=cfg.rebalance_frequency, signal_threshold=sig_thr,
        cov_lookback=60, kelly_fraction=cfg.kelly_fraction,
        risk_aversion=cfg.risk_aversion,
        liquid_top_n=cfg.liquid_top_n,
        start_trading_date=cutoff,
        constraints=PortfolioConstraints(
            max_weight=cfg.max_weight, long_only=True,
            target_leverage=0.95, target_vol=cfg.target_vol),
        exec_config=ExecutionConfig(),
    )
    eng = WalkForwardEngine(wf_cfg, oracle)
    inference_cache: dict = {}
    with phase("Walk-forward run"):
        result = eng.run(sub, corporate_actions=corporate_actions,
                         p_bull_series=p_bull_series,
                         inference_cache=inference_cache)

    # Dump the engine's own per-day scores for train/serve parity checks.
    rows = [
        {"date": str(d), "ticker": t, "p_up_engine": float(p)}
        for d, (p_up, tickers) in inference_cache.items()
        for t, p in zip(tickers, p_up)
    ]
    scores_out = Path("data/engine_inference_scores.parquet")
    pd.DataFrame(rows).to_parquet(scores_out, index=False)
    print(f"engine inference scores dumped -> {scores_out}  ({len(rows):,} rows)")
    return result, cfg


def _fifo_round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """Match buys to sells FIFO per ticker → realized round trips."""
    trips = []
    for ticker, grp in fills.sort_values("date").groupby("ticker"):
        lots: deque = deque()  # (date, qty_remaining, price)
        for f in grp.itertuples(index=False):
            side = str(f.side).upper()
            if side == "BUY":
                lots.append([f.date, f.qty, f.price])
            elif side == "SELL":
                remaining = f.qty
                while remaining > 0 and lots:
                    lot = lots[0]
                    take = min(remaining, lot[1])
                    trips.append({
                        "ticker": ticker,
                        "entry_date": lot[0], "exit_date": f.date,
                        "qty": take,
                        "entry_price": lot[2], "exit_price": f.price,
                        "gross_pnl": (f.price - lot[2]) * take,
                        "hold_days": (pd.Timestamp(f.date) - pd.Timestamp(lot[0])).days,
                    })
                    lot[1] -= take
                    remaining -= take
                    if lot[1] <= 0:
                        lots.popleft()
    return pd.DataFrame(trips)


def main(sig_thr: float, rebalance_frequency: int | None = None) -> None:
    configure_logging()
    result, cfg = _run_engine(sig_thr, rebalance_frequency)

    eq = result.equity_curve
    fills = pd.DataFrame(result.fills)
    print(f"\nfills: {len(fills):,}   rejections: {len(result.rejections):,}")
    if len(fills) == 0:
        print("No fills — nothing to attribute.")
        return
    print(f"fill columns: {list(fills.columns)}")

    fills_out = Path(f"data/engine_fills_thr{sig_thr:.2f}.parquet")
    fills.to_parquet(fills_out, index=False)
    print(f"fills dumped -> {fills_out}")

    # Per-fill cost forensics: where does the 21%-of-notional cost come from?
    fills["notional"] = fills["qty"] * fills["price"]
    fills["cost_pct"] = fills["cost"] / fills["notional"] * 100
    print("\n  COST AS % OF FILLED NOTIONAL (per fill):")
    print(fills["cost_pct"].describe(percentiles=[0.5, 0.9, 0.99]).round(3).to_string())
    print("\n  TOP 15 FILLS BY ABSOLUTE COST:")
    print(fills.nlargest(15, "cost")[
        ["date", "ticker", "side", "qty", "price", "cash_flow", "cost",
         "participation", "cost_pct"]].to_string(index=False))

    bar = "=" * 78
    init = cfg.initial_capital
    final_nav = float(eq["nav"].iloc[-1])
    net_pnl = final_nav - init

    buys = fills[fills["side"].str.upper() == "BUY"]
    sells = fills[fills["side"].str.upper() == "SELL"]
    buy_notional = float((buys["qty"] * buys["price"]).sum())
    sell_notional = float((sells["qty"] * sells["price"]).sum())
    total_cost = float(fills["cost"].sum())
    n_days = len(eq)
    years = n_days / 252

    avg_nav = float(eq["nav"].mean())
    annual_turnover = (buy_notional + sell_notional) / 2 / avg_nav / years

    print(f"\n{bar}")
    print(f" ENGINE PnL ATTRIBUTION — sig_thr={sig_thr}  ({n_days} OOS days)")
    print(bar)
    print(f"  initial capital      : {init:>20,.0f} VND")
    print(f"  final NAV            : {final_nav:>20,.0f} VND")
    print(f"  net PnL              : {net_pnl:>20,.0f} VND  ({net_pnl / init * 100:+.2f}%)")
    print(f"  ---")
    print(f"  fills                : {len(fills):>10,}  (buys={len(buys):,}  sells={len(sells):,})")
    print(f"  buy notional         : {buy_notional:>20,.0f} VND")
    print(f"  sell notional        : {sell_notional:>20,.0f} VND")
    print(f"  TOTAL EXECUTION COST : {total_cost:>20,.0f} VND  "
          f"({total_cost / init * 100:.2f}% of initial capital)")
    print(f"  annualized turnover  : {annual_turnover:>10.1f}x")
    print(f"  cost / |net loss|    : {abs(total_cost / net_pnl) * 100 if net_pnl != 0 else 0:>10.1f}%")

    trips = _fifo_round_trips(fills)
    if len(trips):
        gross_realized = float(trips["gross_pnl"].sum())
        print(f"  ---")
        print(f"  round trips          : {len(trips):>10,}")
        print(f"  gross realized PnL   : {gross_realized:>20,.0f} VND  "
              f"(price selection, pre-cost)")
        print(f"  median hold          : {trips['hold_days'].median():>10.0f} days")
        print(f"  mean hold            : {trips['hold_days'].mean():>10.1f} days")
        hold_dist = trips["hold_days"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0)
        print(f"  hold p10/p25/p50/p75/p90: "
              f"{'/'.join(str(int(v)) for v in hold_dist)}")
        win_rate = (trips["gross_pnl"] > 0).mean() * 100
        print(f"  round-trip win rate  : {win_rate:>10.1f}%  (gross)")
        per_trip_ret = (trips["exit_price"] / trips["entry_price"] - 1.0) * 100
        print(f"  mean gross ret/trip  : {per_trip_ret.mean():>10.3f}%")
        print(f"  ---")
        print(f"  DECOMPOSITION: net = gross_selection + costs + unrealized/marks")
        print(f"    gross selection    : {gross_realized:>+20,.0f} VND ({gross_realized / init * 100:+.2f}%)")
        print(f"    execution costs    : {-total_cost:>+20,.0f} VND ({-total_cost / init * 100:+.2f}%)")
        resid = net_pnl - gross_realized + total_cost
        print(f"    residual (open marks/CA): {resid:>+15,.0f} VND ({resid / init * 100:+.2f}%)")
    print(bar)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sig-thr", type=float, default=0.40)
    p.add_argument("--rebalance-frequency", type=int, default=None)
    args = p.parse_args()
    main(args.sig_thr, args.rebalance_frequency)
