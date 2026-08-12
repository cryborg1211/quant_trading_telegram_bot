"""Where between 0.41 and 0.46 does the gate stop paying for itself?

CONTEXT
───────
The validated engine gate is 0.41 (`signal_threshold`, cross_sectional) and it
earns Sharpe 0.600 / DD -31.00% / 4555 buys. Serve gates on 0.46 — which in the
backtest is only a CLASSIFICATION-metric threshold, never a trading rule — and
with its four defensive layers that lands at 5 buys in 920 days, retaining 1.7pct
of the validated PnL.

-31pct DD is outside this project's ~-13pct comfort band, so "just use 0.41" is
not the answer either. This sweep finds what the intermediate levels actually
buy, and separates THREE effects that were previously tangled:

  1. LEVEL   — the gate number itself, 0.41 .. 0.46
  2. MODE    — cross_sectional (floor + top-N, what was validated) vs
               absolute_gate (floor + pool cap, what serve does)
  3. LAYERS  — none vs all four production defensive layers

Reported per arm: Sharpe, MaxDD, NetPnL and BET COUNT. Bet count is not a
footnote — the DSR work found too few independent bets is the binding constraint
on ever validating this system, so a level that improves Sharpe by starving the
book is not progress.

Zero writes, zero model changes.

Run: python scripts/analyze_gate_level_sweep.py
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
from src.trading.breadth import breadth_time_series  # noqa: E402

BUNDLE_20D = REPO / "models" / "saved" / "v3_ensemble_20d.joblib"
GATES = (0.41, 0.42, 0.43, 0.44, 0.45, 0.46)
ALL_LAYERS = {"serve_exposure_brake": True, "serve_sector_cap": 2,
              "serve_cohort_dedup": True, "serve_hysteresis_days": 2}


def main() -> None:
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    feats = list(bundle["tabular_features"])
    print(f"GOLDEN: up_threshold={bundle['up_threshold']:.2f} "
          f"signal_threshold={bundle['signal_threshold']:.2f}")

    cfg = _apply_eval_overrides(RunConfig(), {})
    print(f"eval knobs: max_positions={cfg.max_positions} "
          f"liquid_top_n={cfg.liquid_top_n}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    ca = load_corporate_actions(cfg)
    try:
        breadth = breadth_time_series(ds.panel)
    except Exception as exc:  # noqa: BLE001
        print(f"breadth unavailable ({exc}) — breadth leg reads 1.0")
        breadth = None

    # One shared cache: every arm here uses the 1-D P(UP) oracle (no argmax
    # mode), so oracle scoring is paid exactly once.
    cache: dict = {}
    arms: list[tuple[str, str, float, dict]] = []
    for g in GATES:
        arms.append(("cross_sectional", "none", g, {}))
    for g in GATES:
        arms.append(("cross_sectional", "all four", g, dict(ALL_LAYERS)))
    for g in GATES:
        arms.append(("absolute_gate", "none", g, {}))

    rows = []
    for mode, layer_label, gate, layer_kw in arms:
        # cross_sectional reads `cfg.signal_threshold`; absolute_gate reads
        # `admission_floor`. Set BOTH to the same level so the comparison is of
        # the mode, not of two different numbers.
        cfg.signal_threshold = gate
        eq = run_oos(ds.panel, feats, ensemble, ca, ds.cutoff, cfg,
                     mode="tranche", hold_days=30, inference_cache=cache,
                     breadth_series=breadth,
                     admission_mode=mode, admission_floor=gate, **layer_kw)
        m = equity_metrics(eq, cfg.initial_capital)
        buys = int(eq.attrs.get("n_buys", 0))
        cash_d = int(eq.attrs.get("zero_candidate_days", 0))
        rows.append((mode, layer_label, gate, m, buys, cash_d, len(eq)))
        print(f"  {mode:16s} {layer_label:9s} gate={gate:.2f}  "
              f"Sharpe={m['net_sharpe']:+.3f} DD={m['max_drawdown']*100:6.2f}% "
              f"PnL={m['net_pnl']:+,.0f} buys={buys}")

    for mode, layer_label in (("cross_sectional", "none"),
                              ("cross_sectional", "all four"),
                              ("absolute_gate", "none")):
        sub = [r for r in rows if r[0] == mode and r[1] == layer_label]
        if not sub:
            continue
        print(f"\n{'=' * 92}")
        print(f"{mode}  |  layers: {layer_label}")
        print(f"{'=' * 92}")
        print(f"{'gate':>6}{'Sharpe':>9}{'MaxDD':>9}{'NetPnL (VND)':>18}"
              f"{'buys':>8}{'cash days':>12}")
        for _m, _l, g, met, buys, cash_d, n in sub:
            print(f"{g:>6.2f}{met['net_sharpe']:>+9.3f}"
                  f"{met['max_drawdown'] * 100:>8.2f}%{met['net_pnl']:>+18,.0f}"
                  f"{buys:>8}{cash_d:>8}/{n}")
        best = max(sub, key=lambda r: r[3]["net_sharpe"])
        print(f"  best Sharpe at gate={best[2]:.2f} ({best[3]['net_sharpe']:+.3f}, "
              f"DD {best[3]['max_drawdown'] * 100:.2f}%, {best[4]} buys)")

    print("\nRead the DD column against this project's ~-13pct comfort band, and "
          "the buys column against the DSR bet-count constraint. A gate that "
          "looks best on Sharpe while leaving a handful of trades cannot be "
          "validated forward no matter how good it looks here.")


if __name__ == "__main__":
    main()
