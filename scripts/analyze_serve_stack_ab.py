"""Does the SERVE stack earn what the backtest claims? (2026-08-11, READ-ONLY)

THE PROBLEM THIS MEASURES
─────────────────────────
The ~100pct/3yr OOS figure was earned by a rule that admits on `p_up >= tau`,
ranks by `p_up`, and holds `max_positions` (5). Production does none of those
three the same way — `main._select_candidates` dispatches only names where
`make_final_decision` returned BUY, which additionally requires the primary
horizon's ARGMAX to be UP, then ranks the survivors by SENTIMENT score
descending with p_up as a mere tiebreak, and slices the top 3.

So the validated number may not describe the deployed system at all. Each
divergence is isolated as its own arm, then arm 5 stacks them.

  1. BACKTEST-VALIDATED   gate>=tau, rank p_up,   top-5   ← the ~100pct config
  2. + arbitrator argmax  gate>=tau AND argmax==UP, rank p_up, top-5
  3. + random ranking     gate>=tau, rank RANDOM, top-5
  4. + top-3 cohort       gate>=tau, rank p_up,   top-3
  5. SERVE ANALOGUE       gate>=tau AND argmax==UP, rank RANDOM, top-3

WHY RANDOM IS THE RIGHT PROXY FOR SENTIMENT RANKING
───────────────────────────────────────────────────
Sentiment has no point-in-time history — that is precisely why this project
keeps a forward paper-log instead of backtesting it. So serve's actual ordering
cannot be replayed. What CAN be answered is whether the ordering rule matters at
all: if a random permutation of the admitted set performs like p_up ordering,
then serve's re-ordering is harmless. If p_up ordering is materially better,
serve is discarding the edge the backtest measured. Random is run over several
seeds because a single permutation is itself noise.

Arm 2 is the one that answers "is the arbitrator veto helping or hurting" — it
survived 69pct of tau-clearing paperlog rows being killed with nobody having
measured whether that was protection or lost return.

ACCEPTANCE: this is diagnostic, not a search for a winner. The question is how
much of arm 1's result survives each added constraint.

Zero writes, zero model changes.

Run: python scripts/analyze_serve_stack_ab.py
"""
from __future__ import annotations

import statistics
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
RANDOM_SEEDS = (0, 1, 2)   # a single permutation is noise; average three


def main() -> None:
    bundle = joblib.load(BUNDLE_20D)
    ensemble = bundle["ensemble"]
    tabular_features = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    sig_thr = float(bundle["signal_threshold"])
    print(f"T+20 GOLDEN: up_threshold={up_thr:.2f} signal_threshold={sig_thr:.2f} "
          f"trained_at={(bundle.get('metadata') or {}).get('trained_at')}")

    cfg = _apply_eval_overrides(RunConfig(), {})
    print(f"eval knobs: max_positions={cfg.max_positions} liquid_top_n={cfg.liquid_top_n}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    ca = load_corporate_actions(cfg)
    print(f"cutoff={ds.cutoff}")

    GATE = dict(admission_mode="absolute_gate", admission_floor=up_thr)
    GATE_ARGMAX = dict(admission_mode="absolute_gate_argmax", admission_floor=up_thr)

    arms: list[tuple[str, dict, tuple[int, ...]]] = [
        ("1 BACKTEST-VALIDATED (p_up, top5)", {**GATE, "max_positions": 5}, (0,)),
        ("2 + arbitrator argmax", {**GATE_ARGMAX, "max_positions": 5}, (0,)),
        ("3 + random ranking", {**GATE, "max_positions": 5,
                                "rank_mode": "random"}, RANDOM_SEEDS),
        ("4 + top-3 cohort", {**GATE, "max_positions": 3}, (0,)),
        ("5 SERVE ANALOGUE (all three)", {**GATE_ARGMAX, "max_positions": 3,
                                          "rank_mode": "random"}, RANDOM_SEEDS),
    ]

    # Separate caches per oracle shape: the argmax arms run a 3-class oracle and
    # the cache entry carries the argmax map, so sharing one cache would let a
    # 1-D arm seed an EMPTY argmax that an argmax arm then reads back — a
    # green-looking run admitting nothing.
    caches: dict[str, dict] = {"1d": {}, "3class": {}}
    rows = []
    for label, kw, seeds in arms:
        key = "3class" if "argmax" in kw["admission_mode"] else "1d"
        per_seed = []
        for seed in seeds:
            cfg.signal_threshold = sig_thr
            eq = run_oos(ds.panel, tabular_features, ensemble, ca, ds.cutoff, cfg,
                         mode="tranche", hold_days=30,
                         inference_cache=caches[key], rank_seed=seed, **kw)
            m = equity_metrics(eq, cfg.initial_capital)
            per_seed.append((m, int(eq.attrs.get("zero_candidate_days", 0)),
                             int(eq.attrs.get("n_buys", 0)), len(eq)))
            print(f"  {label} seed={seed}: NetPnL={m['net_pnl']:+,.0f} "
                  f"Sharpe={m['net_sharpe']:+.3f} DD={m['max_drawdown']*100:.2f}% "
                  f"buys={per_seed[-1][2]}")
        pnl = statistics.mean(s[0]["net_pnl"] for s in per_seed)
        shp = statistics.mean(s[0]["net_sharpe"] for s in per_seed)
        dd = statistics.mean(s[0]["max_drawdown"] for s in per_seed)
        buys = statistics.mean(s[2] for s in per_seed)
        rows.append((label, pnl, shp, dd, buys, per_seed[0][3], len(seeds)))

    print(f"\n{'=' * 104}")
    print("SERVE-STACK A/B — how much of the validated result survives each "
          "production constraint")
    print(f"{'=' * 104}")
    print(f"{'arm':<38}{'NetPnL (VND)':>18}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'buys':>8}{'seeds':>7}")
    for label, pnl, shp, dd, buys, _n, ns in rows:
        print(f"{label:<38}{pnl:>+18,.0f}{shp:>+9.3f}{dd*100:>8.2f}%"
              f"{buys:>8.0f}{ns:>7}")

    base_pnl, base_shp = rows[0][1], rows[0][2]
    print(f"\nvs arm 1 (the config the ~100pct figure came from):")
    for label, pnl, shp, _dd, _b, _n, _ns in rows[1:]:
        keep = (pnl / base_pnl * 100.0) if base_pnl else float("nan")
        print(f"  {label:<38} keeps {keep:6.1f}% of NetPnL   "
              f"Sharpe {shp:+.3f} vs {base_shp:+.3f}")

    print("\nArm 2 answers whether the arbitrator veto protects or costs. Arm 3 "
          "answers whether the ordering rule matters at all — if random ties "
          "p_up, serve's sentiment-first sort is harmless; if it does not, serve "
          "is discarding the measured edge. Arm 5 is the closest backtestable "
          "analogue of what actually runs.")


if __name__ == "__main__":
    main()
