"""Measure the HMM regime layer's reaction lag at market drawdown starts.

Question (2026-07-14 regime debate): after the market puts in a local peak and
starts a >=4% slide, how many TRADING SESSIONS pass before the regime layer
turns defensive (NO_TRADE {0,7} / PENALTY {1,6} shares rise)?

Method — SERVE PARITY, causal by construction:
  * For every evaluated cutoff date D, the feature panel is rebuilt on each
    ticker's last 120 rows ENDING AT D (the serve window), and only the tail
    row's `market_regime` per ticker is consumed — identical to what
    `main._compute_v3_features` caches on a live run dated D. Within-window
    HMM smoothing therefore cannot leak future information into the reading.
  * Market proxy = equal-weight mean of normalized closes across the loaded
    universe. Episodes = first close <= 0.96 x running 20-session peak.
  * For the most recent episodes, defensive shares are printed for cutoffs
    peak-2 .. peak+7 sessions; "lag" = first session offset where the
    defensive share (NO_TRADE + PENALTY) crosses 30% and 50%.

READ-ONLY diagnostic: no DB writes, no model/artifact changes, no Gemini.
Runtime ~1 min (each cutoff is one 120-row panel build). Usage:
    python scripts/measure_regime_lag.py [--episodes 2] [--window-rows 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.backtest.pipeline import RunConfig, build_features
from src.features.alpha360_generator import Alpha360Generator
from src.trading.regime_policy import NO_TRADE_REGIMES, PENALTY_REGIMES

SERVE_WINDOW_ROWS = 120  # main._compute_v3_features live-window parity
DRAWDOWN_TRIGGER = 0.04  # episode starts when proxy is 4% off its 20d peak
PEAK_LOOKBACK = 20


def regime_shares_at(ohlcv: pl.DataFrame, cutoff) -> tuple[float, float, int]:
    """(no_trade_share, penalty_share, n_tickers) as serve would see at `cutoff`."""
    win = (
        ohlcv.filter(pl.col("date") <= cutoff)
        .sort(["ticker", "date"])
        .group_by("ticker", maintain_order=True)
        .tail(SERVE_WINDOW_ROWS)
    )
    if win.is_empty():
        return 0.0, 0.0, 0
    panel, _, _ = build_features(win, RunConfig())
    if "market_regime" not in panel.columns:
        raise RuntimeError("panel carries no market_regime column")
    tails = (
        panel.sort(["ticker", "date"])
        .group_by("ticker", maintain_order=True)
        .tail(1)
        .select(["ticker", "market_regime"])
        .drop_nulls()
    )
    n = tails.height
    if n == 0:
        return 0.0, 0.0, 0
    regs = [int(v) for v in tails["market_regime"].to_list()]
    nt = sum(1 for r in regs if r in NO_TRADE_REGIMES) / n
    pen = sum(1 for r in regs if r in PENALTY_REGIMES) / n
    return nt, pen, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--window-rows", type=int, default=500)
    args = ap.parse_args()

    print(f"Loading OHLCV window ({args.window_rows} rows/ticker)...")
    ohlcv = Alpha360Generator().load_live_ohlcv_window(window_rows=args.window_rows)
    print(f"rows={ohlcv.height} tickers={ohlcv['ticker'].n_unique()}")

    # Equal-weight normalized market proxy.
    proxy = (
        ohlcv.sort(["ticker", "date"])
        .with_columns((pl.col("close") / pl.col("close").first().over("ticker")).alias("norm"))
        .group_by("date")
        .agg(pl.col("norm").mean().alias("mkt"))
        .sort("date")
    )
    dates = proxy["date"].to_list()
    mkt = proxy["mkt"].to_list()

    # Drawdown episodes: first close <= (1-TRIGGER) x running 20d peak.
    episodes = []  # (peak_idx, trigger_idx)
    i = PEAK_LOOKBACK
    while i < len(mkt):
        window = mkt[max(0, i - PEAK_LOOKBACK): i + 1]
        peak_val = max(window)
        if mkt[i] <= peak_val * (1.0 - DRAWDOWN_TRIGGER):
            peak_idx = max(0, i - PEAK_LOOKBACK) + window.index(peak_val)
            episodes.append((peak_idx, i))
            i += PEAK_LOOKBACK  # skip forward — one episode per structure
        else:
            i += 1

    print(f"\nDrawdown episodes found (>= {DRAWDOWN_TRIGGER:.0%} off 20d peak): {len(episodes)}")
    for p, t in episodes:
        print(f"  peak {dates[p]} ({mkt[p]:.3f}) -> trigger {dates[t]} ({mkt[t]:.3f}, "
              f"{(mkt[t] / mkt[p] - 1) * 100:+.1f}%, {t - p} sessions)")

    for peak_idx, trig_idx in episodes[-args.episodes:]:
        print(f"\n=== Episode: peak {dates[peak_idx]} (drawdown confirmed {dates[trig_idx]}) ===")
        lag30 = lag50 = None
        for k in range(-2, 8):
            idx = peak_idx + k
            if idx < 0 or idx >= len(dates):
                continue
            nt, pen, n = regime_shares_at(ohlcv, dates[idx])
            defensive = nt + pen
            mret = (mkt[idx] / mkt[peak_idx] - 1) * 100
            flag = ""
            if lag30 is None and defensive >= 0.30 and k >= 0:
                lag30, flag = k, " <- crosses 30%"
            if lag50 is None and defensive >= 0.50 and k >= 0:
                lag50, flag = k, flag + " <- crosses 50%"
            print(f"  P{k:+d} {dates[idx]} mkt={mret:+5.1f}% | NO_TRADE={nt:5.1%} "
                  f"PENALTY={pen:5.1%} defensive={defensive:5.1%} (n={n}){flag}")
        print(f"  LAG: defensive>=30% at P{'+' + str(lag30) if lag30 is not None else ': never in P+7'}"
              f" | >=50% at P{'+' + str(lag50) if lag50 is not None else ': never in P+7'}")


if __name__ == "__main__":
    main()
