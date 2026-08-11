"""Argmax vs absolute-gate admission A/B (2026-08-11, READ-ONLY).

THE HYPOTHESIS AND WHERE IT CAME FROM
─────────────────────────────────────
Serve admits on `P(UP) >= tau` (tau=0.46). Two independent live-paperlog reads
say that is the wrong statistic:

  1. CALIBRATION. Across the range the gate operates in, P(UP) is
     ANTI-informative. Binned against realized 20d outcomes (n=1370):
         [0.2,0.3)  pred 0.281  actual 0.459
         [0.4,0.5)  pred 0.419  actual 0.293
     Platt slope −0.274 — higher P(UP), worse outcome.
  2. THE ARGMAX IS DIFFERENT. Rows whose 3-class argmax was UP returned
     +1.19% mean 20d vs the −2.43% baseline (n=33), and T+5=UP with
     T+20=DOWN returned −1.57% (n=8).

Argmax also cannot be starved. The absolute gate currently sits ABOVE the 90th
percentile of the model's own output (serve p90 0.423 vs tau 0.46, after drifting
down from 0.463), so the book stays empty regardless of signal quality. Argmax
compares the three classes against each other, so distribution drift moves all
three together and the rule keeps firing.

BOTH CAVEATS, STATED UP FRONT
─────────────────────────────
The paperlog evidence is n=33 on rows that are mostly NOT liquidity-filtered
(`liquid_at_log` only starts 2026-08-11). This backtest is the actual test: it
runs on the liquid top-N universe the strategy really trades, over the full OOS
window, through the same engine.

ARMS — one variable only, admission. Same T+20 GOLDEN artifact, same engine,
same hold, shared inference cache (oracle scoring is admission-independent):
  1. absolute_gate @ the artifact's own up_threshold — TODAY'S PRODUCTION RULE
  2. argmax — admit every name whose top class is UP, ranked by p_up within
  3. cross_sectional — the always-deploys reference, for context
  4. argmax + burst sizing (nav/10) — argmax fires more often than the starved
     gate, so the burst divisor that won on gate-open days is re-measured here
     rather than assumed to transfer

ACCEPTANCE (same lens as every A/B in this project): an arm wins only if
Sharpe improves AND MaxDD is not worse. Zero-candidate days are reported as the
cash-drag proxy, and dispatch count as the bet-count proxy — the binding
constraint identified in the assessment was too FEW bets, not bad ones.

Zero writes, zero model changes.

Run: python scripts/analyze_argmax_admission_ab.py
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

BUNDLE_20D = REPO / "models" / "saved" / "v3_ensemble_20d.joblib"


def _buy_count(eq: pd.DataFrame) -> int:
    """Buy fills — the bet-count proxy, surfaced by run_oos via eq.attrs."""
    return int(eq.attrs.get("n_buys", 0))


def main() -> None:
    print(f"Loading T+20 GOLDEN serve bundle {BUNDLE_20D.name} ...")
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    tabular_features = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    sig_thr = float(bundle["signal_threshold"])
    meta = bundle.get("metadata") or {}
    print(f"  up_threshold={up_thr:.2f}  signal_threshold={sig_thr:.2f}  "
          f"trained_at={meta.get('trained_at')}")

    cfg = RunConfig()
    cfg = _apply_eval_overrides(cfg, {})
    print(f"  eval knobs: max_positions={cfg.max_positions}  "
          f"liquid_top_n={cfg.liquid_top_n}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)
    print(f"  cutoff={ds.cutoff}")

    arms = [
        (f"PRODUCTION absolute_gate>={up_thr:.2f}",
         dict(admission_mode="absolute_gate", admission_floor=up_thr)),
        ("argmax (class decision)",
         dict(admission_mode="argmax")),
        ("cross_sectional (reference)",
         dict(admission_mode="cross_sectional")),
        ("argmax + burst nav/10",
         dict(admission_mode="argmax", tranche_budget_days=10)),
    ]

    # SEPARATE caches per oracle shape. `argmax` runs a 3-class oracle while the
    # other arms run the (n,) P(UP) one, and the cache entry carries the argmax
    # map. Sharing one cache would let a 1-D arm seed an EMPTY argmax that a
    # later argmax arm then reads back, silently admitting nothing — a
    # green-looking run with a meaningless result.
    caches: dict[str, dict] = {"1d": {}, "3class": {}}
    results = []
    for label, kw in arms:
        cfg.signal_threshold = sig_thr
        cache = caches["3class" if kw.get("admission_mode") == "argmax" else "1d"]
        print(f"\nRunning arm: {label} ...")
        eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                     ds.cutoff, cfg, mode="tranche", hold_days=30,
                     inference_cache=cache, **kw)
        m = equity_metrics(eq, cfg.initial_capital)
        zc = int(eq.attrs.get("zero_candidate_days", 0))
        results.append((label, m, zc, len(eq), _buy_count(eq)))
        print(f"  NetPnL={m['net_pnl']:+,.0f}  Sharpe={m['net_sharpe']:+.3f}  "
              f"DD={m['max_drawdown'] * 100:.2f}%  cash days={zc}/{len(eq)}  "
              f"buys={results[-1][4]}")

    print(f"\n{'=' * 96}")
    print("ARGMAX ADMISSION A/B — T+20 GOLDEN artifact, tranche hold=30, "
          "admission is the ONLY variable")
    print(f"{'=' * 96}")
    print(f"{'arm':<34}{'NetPnL (VND)':>18}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'cash days':>12}{'buys':>8}")
    for label, m, zc, n_days, buys in results:
        print(f"{label:<34}{m['net_pnl']:>+18,.0f}{m['net_sharpe']:>+9.3f}"
              f"{m['max_drawdown'] * 100:>8.2f}%{zc:>8}/{n_days:<4}{buys:>8}")

    prod = results[0][1]
    print(f"\nAcceptance: an arm beats production only with Sharpe "
          f"> {prod['net_sharpe']:+.3f} AND MaxDD not worse than "
          f"{prod['max_drawdown'] * 100:.2f}%.")
    print("`buys` is the bet-count proxy — the assessment found the binding "
          "constraint is too FEW independent bets to ever clear DSR, so an arm "
          "that matches production on risk while firing far more often is a "
          "real improvement even at equal Sharpe.")


if __name__ == "__main__":
    main()
