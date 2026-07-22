"""Burst-sizing A/B — spend the barbell's unused risk headroom (2026-07-22).

The concentration A/B showed the absolute_gate>=0.46 barbell holds Sharpe
(~0.60) at just -4.4% DD but 1/8 the PnL: the gate opens on ~5% of days and
the calendar budget (nav/30/day) barely deploys before going back to cash.
New `tranche_budget_days` engine knob decouples the budget divisor from the
hold length. This A/B asks: deploying nav/10 or nav/5 on gate-open days
(3x / 6x the calendar budget), does PnL scale toward baseline while DD
stays under the production comfort band (~-13%)?

All arms: T+20 GOLDEN serve artifact, tranche hold=30, absolute_gate
admission at the artifact's own 0.46 floor. Shared inference cache (oracle
scoring is admission/budget-independent). Baseline cross_sectional arm
included for the headline reference. READ-ONLY, zero artifact changes.

Run: python scripts/analyze_burst_sizing_ab.py
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

    gate = dict(admission_mode="absolute_gate", admission_floor=up_thr)
    arms = [
        ("baseline cross_sectional (ref)", dict(admission_mode="cross_sectional")),
        ("gate 0.46, budget nav/30 (ref)", dict(**gate)),
        ("gate 0.46, BURST nav/10", dict(**gate, tranche_budget_days=10)),
        ("gate 0.46, BURST nav/5", dict(**gate, tranche_budget_days=5)),
    ]

    cache: dict = {}
    results = []
    for label, kw in arms:
        cfg.signal_threshold = sig_thr
        print(f"\nRunning arm: {label} ...")
        eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                     cutoff, cfg, mode="tranche", hold_days=30,
                     inference_cache=cache, **kw)
        m = equity_metrics(eq, cfg.initial_capital)
        zc = int(eq.attrs.get("zero_candidate_days", 0))
        results.append((label, m, zc, len(eq)))
        print(f"  NetPnL={m['net_pnl']:+,.0f}  Sharpe={m['net_sharpe']:+.3f}  "
              f"DD={m['max_drawdown'] * 100:.2f}%  cash days={zc}/{len(eq)}")

    print(f"\n{'=' * 92}\nBURST-SIZING A/B — T+20 GOLDEN, absolute_gate >= {up_thr:.2f}, "
          f"hold=30, budget divisor only\n{'=' * 92}")
    print(f"{'arm':<34}{'NetPnL (VND)':>18}{'Sharpe':>9}{'MaxDD':>9}{'cash days':>12}")
    for label, m, zc, n_days in results:
        print(f"{label:<34}{m['net_pnl']:>+18,.0f}{m['net_sharpe']:>+9.3f}"
              f"{m['max_drawdown'] * 100:>8.2f}%{zc:>8}/{n_days}")

    print("\nDecision lens: a burst arm is interesting if PnL moves materially "
          "toward the baseline while MaxDD stays within the production comfort "
          "band (~-13%, the braked serve book's level) and Sharpe holds near "
          "the gate-0.46 reference (0.600).")


if __name__ == "__main__":
    main()
