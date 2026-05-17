"""Cost-aware economic evaluation for the Quant V6 stacking ensemble.

WHY THIS REPLACES MACRO-F1
──────────────────────────
A classifier with a beautiful macro-F1 can still lose money: F1 rewards
getting the SIDEWAYS class right (the majority, ~34% even after the
intrabar fix) and is blind to transaction costs. What a quant fund is
actually graded on is **Net P&L** and **Net Sharpe** *after* paying VN
market frictions. Model / threshold selection must optimise the economic
metric, not the statistical one (de Prado, AFML Ch. 14 — "backtest
statistics that matter").

TRADING RULE (long-only — HOSE retail cannot easily/ cheaply short)
───────────────────────────────────────────────────────────────────
    predicted UP (class 2)  → enter long at the signal bar close,
                              exit at the triple-barrier event date t1.
    DOWN / SIDEWAYS         → stay flat (no trade ⇒ no cost, no P&L).

The realised gross return of that long trade is exactly
``target_return_{h}`` produced by the triple-barrier labeller (close at
signal → close at t1).

COST MODEL (deliberately aggressive — VN small-caps gap on the open)
───────────────────────────────────────────────────────────────────
    round-trip fee   = 2 * fee_rate            (buy + sell)
        fee_rate 0.002 ≈ HOSE ~0.15% brokerage + 0.1% sell tax + transfer
    round-trip slip  = 2 * slippage_per_side   (aggressive gap fill)
        VN HOSE frequently gaps through the intended fill, especially the
        illiquid small-caps the model will over-pick; we charge it hard so
        the selected model is robust to real execution, not paper-perfect.

    net_trade_return = gross_return − (round-trip fee + round-trip slip)

NET SHARPE
──────────
Per-trade net returns → mean / std, annualised by ``sqrt(252 / horizon)``
(a horizon-day position can be recycled ~252/horizon times per year).
This is the single number that drives ``beats_baseline`` and the
cost-aware probability-threshold selection.
"""

from __future__ import annotations

import numpy as np

UP_CLASS = 2
TRADING_DAYS = 252

# Aggressive VN-market defaults. Override per call if needed.
DEFAULT_FEE_RATE = 0.002          # per side
DEFAULT_SLIPPAGE_PER_SIDE = 0.002  # per side — aggressive gap penalty


def round_trip_cost(
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_per_side: float = DEFAULT_SLIPPAGE_PER_SIDE,
) -> float:
    """Total cost charged to one completed long trade (buy + sell)."""
    return 2.0 * fee_rate + 2.0 * slippage_per_side


def economic_report(
    decisions: np.ndarray,
    realized_return: np.ndarray,
    horizon: int,
    *,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_per_side: float = DEFAULT_SLIPPAGE_PER_SIDE,
) -> dict[str, float | int | bool]:
    """Net P&L / Net Sharpe of a long-only book.

    Parameters
    ----------
    decisions : array[int|bool], shape (n,)
        ``True``/1 where we GO LONG (predicted UP), else flat. An int class
        array is accepted too — it is reduced to ``== UP_CLASS``.
    realized_return : array[float], shape (n,)
        ``target_return_{h}`` — gross close→t1 return for that row.
    horizon : int
        Label horizon in trading days (annualisation factor).

    Returns
    -------
    dict with net_pnl, net_sharpe (headline), gross_pnl, n_trades,
    hit_rate, avg_net_trade, avg_gross_trade, cost_per_trade, no_trades.
    """
    dec = np.asarray(decisions)
    if dec.dtype != bool:
        long_mask = dec == UP_CLASS
    else:
        long_mask = dec
    gross = np.asarray(realized_return, dtype=np.float64)

    valid = long_mask & np.isfinite(gross)
    n_trades = int(valid.sum())
    cost = round_trip_cost(fee_rate, slippage_per_side)

    if n_trades == 0:
        # No edge taken ⇒ no economic value. Sharpe 0 (not NaN) so it sorts
        # below any profitable threshold without poisoning comparisons.
        return {
            "net_pnl": 0.0,
            "net_sharpe": 0.0,
            "gross_pnl": 0.0,
            "n_trades": 0,
            "hit_rate": 0.0,
            "avg_net_trade": 0.0,
            "avg_gross_trade": 0.0,
            "cost_per_trade": float(cost),
            "no_trades": True,
        }

    g = gross[valid]
    net = g - cost
    net_mean = float(net.mean())
    net_std = float(net.std(ddof=1)) if n_trades > 1 else 0.0
    ann = float(np.sqrt(TRADING_DAYS / max(horizon, 1)))
    net_sharpe = net_mean / net_std * ann if net_std > 1e-12 else 0.0

    return {
        "net_pnl": float(net.sum()),
        "net_sharpe": float(net_sharpe),
        "gross_pnl": float(g.sum()),
        "n_trades": n_trades,
        "hit_rate": float((net > 0.0).mean()),
        "avg_net_trade": net_mean,
        "avg_gross_trade": float(g.mean()),
        "cost_per_trade": float(cost),
        "no_trades": False,
    }


def select_pnl_threshold(
    p_up: np.ndarray,
    realized_return: np.ndarray,
    horizon: int,
    *,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_per_side: float = DEFAULT_SLIPPAGE_PER_SIDE,
    grid: np.ndarray | None = None,
    min_trades: int = 50,
) -> tuple[float, dict[str, float | int | bool]]:
    """Pick the P(UP) cut-off that maximises **Net Sharpe** (cost-aware).

    This is the Task-2 "selection prioritises the economic metric over the
    statistical one" mechanism. It MUST be called on leak-free
    out-of-fold probabilities (purged-CV meta predictions), never on
    in-sample fits.

    A ``min_trades`` floor rejects degenerate thresholds that cherry-pick a
    handful of lucky trades. Falls back to argmax-style τ=0.5 if no grid
    point clears the floor.
    """
    if grid is None:
        grid = np.round(np.arange(0.30, 0.901, 0.01), 4)

    p = np.asarray(p_up, dtype=np.float64)
    best_tau = 0.5
    best_sharpe = -np.inf
    best_report: dict[str, float | int | bool] = {}

    for tau in grid:
        decisions = p >= tau
        rep = economic_report(
            decisions, realized_return, horizon,
            fee_rate=fee_rate, slippage_per_side=slippage_per_side,
        )
        if int(rep["n_trades"]) < min_trades:
            continue
        s = float(rep["net_sharpe"])
        if s > best_sharpe:
            best_sharpe, best_tau, best_report = s, float(tau), rep

    if not best_report:  # nothing cleared the floor → safe default
        best_tau = 0.5
        best_report = economic_report(
            p >= best_tau, realized_return, horizon,
            fee_rate=fee_rate, slippage_per_side=slippage_per_side,
        )
    return best_tau, best_report


if __name__ == "__main__":
    # Smoke test: costs must turn a thin gross edge into a net loss, and a
    # fat edge must survive; threshold selection must beat naive τ=0.5.
    rng = np.random.default_rng(0)
    n = 5_000
    horizon = 5

    # Latent skill: higher p_up ⇒ higher expected gross return.
    p_up = rng.uniform(0, 1, n)
    gross = (p_up - 0.5) * 0.06 + rng.normal(0, 0.02, n)  # ±, edge in tails

    naive = economic_report(p_up >= 0.5, gross, horizon)
    tau, sel = select_pnl_threshold(p_up, gross, horizon, min_trades=50)

    print(f"cost/trade           : {round_trip_cost():.4f}  (0.8% round trip)")
    print(f"naive  tau=0.50      : sharpe={naive['net_sharpe']:.3f} "
          f"net_pnl={naive['net_pnl']:.3f} trades={naive['n_trades']}")
    print(f"chosen tau={tau:.2f}      : sharpe={sel['net_sharpe']:.3f} "
          f"net_pnl={sel['net_pnl']:.3f} trades={sel['n_trades']}")
    assert sel["net_sharpe"] >= naive["net_sharpe"], "selector must not worsen Sharpe"

    # A pure-noise signal must NOT produce a positive net Sharpe edge.
    noise = economic_report(rng.uniform(0, 1, n) >= 0.5,
                            rng.normal(0, 0.02, n), horizon)
    print(f"pure noise tau=0.50  : sharpe={noise['net_sharpe']:.3f} "
          f"net_pnl={noise['net_pnl']:.3f} (should be <= ~0)")
    assert noise["net_pnl"] < 0.0, "costs must make a no-edge book lose money"
    print("economic_metrics smoke test OK")
