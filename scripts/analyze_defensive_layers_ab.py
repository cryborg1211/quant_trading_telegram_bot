"""What do the four production defensive layers actually cost or save?

THE GAP THIS CLOSES
───────────────────
Sector cap, open-cohort dedup, admission hysteresis and the drift/breadth
exposure brake all run in production. NONE of them was in the backtest that
produced the validated numbers. Each was added after a real loss, one at a
time, with no measurement — so it has never been known whether they protect the
edge or destroy it.

The second gap this closes is the threshold. The GOLDEN artifact stores
`up_threshold = 0.46` (what SERVE gates on) and `signal_threshold = 0.41` (what
the BACKTEST engine actually gated on — the sweep sets sig_thr = thr - 0.05 and
runs admission_mode="cross_sectional"). Serve is 5pp stricter than the rule that
was validated, and the two have never been compared with the defensive layers
present.

GRID — 2 thresholds x 4 layer combinations:

    threshold  0.41 cross_sectional   (the VALIDATED rule)
               0.46 absolute_gate     (what SERVE runs)

    layers     none                   (the backtest as it has always been)
               brake only             (drift + breadth)
               filters only           (sector cap 2 + dedup + hysteresis 2)
               all four               (the closest backtestable serve analogue)

The event-rescue path is deliberately absent: it admits on sentiment, which has
no point-in-time history and cannot be replayed. Everything else in serve's
entry stack is represented.

ACCEPTANCE: diagnostic, not a hunt for a winner. Read each layer's marginal
effect on Sharpe, MaxDD and BET COUNT — the DSR work found too FEW independent
bets is the binding constraint, so a layer that improves Sharpe while halving
the bet count is not obviously good.

Zero writes, zero model changes.

Run: python scripts/analyze_defensive_layers_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

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

LAYER_SETS: list[tuple[str, dict]] = [
    ("none", {}),
    ("brake", {"serve_exposure_brake": True}),
    ("filters", {"serve_sector_cap": 2, "serve_cohort_dedup": True,
                 "serve_hysteresis_days": 2}),
    ("all four", {"serve_exposure_brake": True, "serve_sector_cap": 2,
                  "serve_cohort_dedup": True, "serve_hysteresis_days": 2}),
]


def main() -> None:
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    feats = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    sig_thr = float(bundle["signal_threshold"])
    print(f"GOLDEN: up_threshold={up_thr:.2f} (serve gate)  "
          f"signal_threshold={sig_thr:.2f} (backtest gate)")

    cfg = _apply_eval_overrides(RunConfig(), {})
    print(f"eval knobs: max_positions={cfg.max_positions} "
          f"liquid_top_n={cfg.liquid_top_n}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    ca = load_corporate_actions(cfg)

    # The breadth leg needs a real per-date series; without it that leg would
    # silently read 1.0 and the "brake" arms would understate the brake.
    try:
        breadth = breadth_time_series(ds.panel)
        print(f"breadth series: {len(breadth)} dates")
    except Exception as exc:  # noqa: BLE001
        print(f"breadth series unavailable ({exc}) — breadth leg will read 1.0")
        breadth = None

    thresholds = [
        (f"0.41 cross_sectional (VALIDATED)",
         {"admission_mode": "cross_sectional"}, sig_thr),
        (f"0.46 absolute_gate  (SERVE)",
         {"admission_mode": "absolute_gate", "admission_floor": up_thr}, sig_thr),
    ]

    cache: dict = {}
    rows = []
    for thr_label, thr_kw, engine_sig in thresholds:
        for layer_label, layer_kw in LAYER_SETS:
            cfg.signal_threshold = engine_sig
            eq = run_oos(ds.panel, feats, ensemble, ca, ds.cutoff, cfg,
                         mode="tranche", hold_days=30, inference_cache=cache,
                         breadth_series=breadth, **thr_kw, **layer_kw)
            m = equity_metrics(eq, cfg.initial_capital)
            rows.append((thr_label, layer_label, m,
                         int(eq.attrs.get("n_buys", 0)),
                         int(eq.attrs.get("zero_candidate_days", 0)), len(eq)))
            print(f"  {thr_label:32s} | {layer_label:10s} "
                  f"NetPnL={m['net_pnl']:+,.0f} Sharpe={m['net_sharpe']:+.3f} "
                  f"DD={m['max_drawdown']*100:6.2f}% buys={rows[-1][3]}")

    print(f"\n{'=' * 104}")
    print("DEFENSIVE-LAYER A/B — each layer's marginal effect, at both thresholds")
    print(f"{'=' * 104}")
    print(f"{'threshold':<34}{'layers':<11}{'NetPnL (VND)':>18}{'Sharpe':>9}"
          f"{'MaxDD':>9}{'buys':>8}{'cash d':>8}")
    for thr_label, layer_label, m, buys, cash_d, n in rows:
        print(f"{thr_label:<34}{layer_label:<11}{m['net_pnl']:>+18,.0f}"
              f"{m['net_sharpe']:>+9.3f}{m['max_drawdown']*100:>8.2f}%"
              f"{buys:>8}{cash_d:>6}/{n}")

    print("\nmarginal effect of the layers, per threshold:")
    for thr_label, _, _ in thresholds:
        base = next(r for r in rows if r[0] == thr_label and r[1] == "none")
        print(f"\n  {thr_label}   baseline Sharpe {base[2]['net_sharpe']:+.3f}, "
              f"DD {base[2]['max_drawdown']*100:.2f}%, buys {base[3]}")
        for r in rows:
            if r[0] != thr_label or r[1] == "none":
                continue
            d_sharpe = r[2]["net_sharpe"] - base[2]["net_sharpe"]
            d_dd = (r[2]["max_drawdown"] - base[2]["max_drawdown"]) * 100
            keep = (r[2]["net_pnl"] / base[2]["net_pnl"] * 100
                    if base[2]["net_pnl"] else float("nan"))
            print(f"    {r[1]:<10} Sharpe {d_sharpe:+.3f}  DD {d_dd:+.2f}pp  "
                  f"PnL kept {keep:6.1f}%  buys {r[3]} (was {base[3]})")

    print("\nA layer that lifts Sharpe while collapsing the bet count is NOT "
          "obviously good: the DSR work found too few independent bets is the "
          "binding constraint on ever validating this system.")


if __name__ == "__main__":
    main()
