"""Vol-scaled burst-budget A/B (22-07-26, idea #1 of the 4-idea batch).

Burst sizing (`c8e3300`) proved a FLAT nav/10 divisor recovers ~3x the
concentration A/B's PnL, DD landing just inside the ~-13% comfort band.
This asks: does a DYNAMIC divisor — bigger clips (smaller divisor) in calm
markets, smaller clips (bigger divisor) in stressed markets, via
`src.trading.vol_sizing` — beat the flat nav/10 winner on either axis
(more PnL/Sharpe at similar DD, or lower DD at similar PnL/Sharpe)?

All arms: T+20 GOLDEN serve artifact, tranche hold=30, absolute_gate
admission at the artifact's own floor. Shared inference cache (oracle
scoring is admission/budget-independent). READ-ONLY, zero artifact changes.

Run: python scripts/analyze_vol_scaled_burst_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402

from run_backtest import (  # noqa: E402
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    load_corporate_actions,
    materialize_dataset,
)
from src.trading.vol_sizing import (  # noqa: E402
    market_vol_time_series,
    vol_scaled_budget_days,
)

BUNDLE_20D = REPO / "models" / "saved" / "v3_ensemble_20d.joblib"


def main() -> None:
    print(f"Loading T+20 GOLDEN serve bundle {BUNDLE_20D.name} ...")
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    tabular_features = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    sig_thr = float(bundle["signal_threshold"])

    cfg = RunConfig()
    cfg = _apply_eval_overrides(cfg, {})
    print(f"  up_thr={up_thr:.2f}  max_positions={cfg.max_positions}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)
    cutoff = ds.cutoff

    print("Computing trailing market-vol time series for the vol-scaled arm ...")
    vol_series = market_vol_time_series(ds.panel)
    budget_series = vol_series.apply(lambda v: vol_scaled_budget_days(v, base_divisor=10))
    print(f"  vol-scaled divisor over OOS: min={budget_series.min()} "
          f"max={budget_series.max()} mean={budget_series.mean():.1f} "
          f"days_covered={len(budget_series)}  (flat burst winner = 10)")

    gate = dict(admission_mode="absolute_gate", admission_floor=up_thr)
    arms = [
        ("baseline cross_sectional (ref)", dict(admission_mode="cross_sectional"), None),
        ("gate 0.46, budget nav/30 (ref)", dict(**gate), None),
        ("gate 0.46, BURST nav/10 (flat, shipped)", dict(**gate, tranche_budget_days=10), None),
        ("gate 0.46, VOL-SCALED (dynamic 6-20)", dict(**gate), budget_series),
    ]

    cache: dict = {}
    results = []
    for label, kw, bseries in arms:
        cfg.signal_threshold = sig_thr
        print(f"\nRunning arm: {label} ...")
        eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                     cutoff, cfg, mode="tranche", hold_days=30,
                     inference_cache=cache, budget_days_series=bseries, **kw)
        m = equity_metrics(eq, cfg.initial_capital)
        zc = int(eq.attrs.get("zero_candidate_days", 0))
        results.append((label, m, zc, len(eq)))
        print(f"  NetPnL={m['net_pnl']:+,.0f}  Sharpe={m['net_sharpe']:+.3f}  "
              f"DD={m['max_drawdown'] * 100:.2f}%  cash days={zc}/{len(eq)}")

    print(f"\n{'=' * 96}\nVOL-SCALED BURST A/B — T+20 GOLDEN, absolute_gate >= {up_thr:.2f}, "
          f"hold=30\n{'=' * 96}")
    print(f"{'arm':<42}{'NetPnL (VND)':>18}{'Sharpe':>9}{'MaxDD':>9}{'cash days':>12}")
    for label, m, zc, n_days in results:
        print(f"{label:<42}{m['net_pnl']:>+18,.0f}{m['net_sharpe']:>+9.3f}"
              f"{m['max_drawdown'] * 100:>8.2f}%{zc:>8}/{n_days}")

    print("\nDecision lens: vol-scaled wins if it beats flat nav/10 on EITHER "
          "axis (higher PnL/Sharpe at similar DD, or lower DD at similar "
          "PnL/Sharpe) — otherwise the fixed divisor is simpler and just as good.")


if __name__ == "__main__":
    main()
