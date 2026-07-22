"""Concentration A/B — size on conviction, cash otherwise (2026-07-22, READ-ONLY).

Follow-up to `analyze_confluence_backtest.py`, which found the T+20 gate is
the ONLY quality filter in the stack (gated picks ~3x base rate, n=1,767 ≈
2/day) while the T+5 gate carries ~no edge. User asked for the ROI lever
that finding supports: CONCENTRATION — deploy the tranche budget only when
the strong gate fires (the engine's per-name notional = budget/len(picks)
concentrates automatically when survivors are few), stay cash otherwise.

Arms (all run the SAME T+20 GOLDEN serve artifact through the SAME
walk-forward engine — the only variable is admission):
  1. baseline      — cross_sectional (top-N above sig_thr 0.41, the
                     validated GOLDEN book: always deploys something)
  2. concentrated  — absolute_gate, floor = the artifact's own up_threshold
                     (0.46): only names the strong gate passes; zero-
                     candidate days stay cash
  3. hyper         — absolute_gate, floor 0.50: the confluence gradient
                     suggested returns keep rising with gate strictness

Uses the existing `admission_mode="absolute_gate"` engine support (shipped
04-07-26) — zero new engine code, just a differently-parameterized run.
The per-seed `inference_cache` is shared across arms (oracle scoring is
admission-independent), so arms 2-3 reuse arm 1's GBM scoring.

NOTE vs the 04-07 A/B: that one asked "is serve's absolute gate COSTING
money vs cross_sectional" (answer: no, roughly equal). This asks the
CONCENTRATION question with the NEW 17-07 artifact and its tighter 0.46
gate, measuring return-per-unit-deployment, not just headline Sharpe —
report includes zero-candidate-day counts as the cash-drag proxy.

Zero writes, zero model/artifact changes.

Run: python scripts/analyze_concentration_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

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

TRADING_DAYS = 252
BUNDLE_20D = REPO / "models" / "saved" / "v3_ensemble_20d.joblib"


def main() -> None:
    print(f"Loading T+20 GOLDEN serve bundle {BUNDLE_20D.name} ...")
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    tabular_features = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    sig_thr = float(bundle["signal_threshold"])
    print(f"  up_threshold={up_thr:.2f}  signal_threshold={sig_thr:.2f}  "
          f"features={len(tabular_features)}")

    cfg = RunConfig()
    cfg = _apply_eval_overrides(cfg, {})
    print(f"  eval knobs: max_positions={cfg.max_positions}  "
          f"liquid_top_n={cfg.liquid_top_n}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)
    cutoff = ds.cutoff
    print(f"  cutoff={cutoff}")

    arms = [
        ("baseline (cross_sectional)", dict(admission_mode="cross_sectional")),
        (f"concentrated (gate>={up_thr:.2f})",
         dict(admission_mode="absolute_gate", admission_floor=up_thr)),
        ("hyper (gate>=0.50)",
         dict(admission_mode="absolute_gate", admission_floor=0.50)),
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
        n_days = len(eq)
        results.append((label, m, zc, n_days))
        print(f"  NetPnL={m['net_pnl']:+,.0f}  Sharpe={m['net_sharpe']:+.3f}  "
              f"DD={m['max_drawdown'] * 100:.2f}%  zero-candidate days={zc}/{n_days}")

    print(f"\n{'=' * 90}\nCONCENTRATION A/B — T+20 GOLDEN artifact, tranche hold=30, "
          f"same engine, admission only\n{'=' * 90}")
    print(f"{'arm':<34}{'NetPnL (VND)':>18}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'cash days':>11}")
    for label, m, zc, n_days in results:
        print(f"{label:<34}{m['net_pnl']:>+18,.0f}{m['net_sharpe']:>+9.3f}"
              f"{m['max_drawdown'] * 100:>8.2f}%{zc:>7}/{n_days}")

    base = results[0][1]
    print(f"\nAcceptance lens (same gate as every A/B this session): an arm wins "
          f"only if Sharpe > {base['net_sharpe']:+.3f} AND MaxDD not worse than "
          f"{base['max_drawdown'] * 100:.2f}%. Zero-candidate days show the cash "
          f"drag each concentration level pays for its selectivity.")


if __name__ == "__main__":
    main()
