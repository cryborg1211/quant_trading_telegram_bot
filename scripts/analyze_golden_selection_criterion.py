"""GOLDEN selection criterion A/B: max-NetPnL (current) vs max-Sharpe
(proposed) -- 08-08-26.

MOTIVATION (found while auditing the 08-08-26 weekly auto-retrain)
------------------------------------------------------------------
`run_backtest.main` picks the GOLDEN config by **max mean OOS Net PnL**
(`golden = max(sweep_results, key=lambda r: r["mean_net_pnl"])`). This
morning's real retrain sweep shows what that costs:

    up_thr   mean_NetPnL     mean_Sharpe   mean_DD
    0.50     +3,000,030,798      +1.191     -8.73%
    0.46     +3,050,587,153      +0.605    -20.16%   <- picked GOLDEN

The rule picked 0.46 for a **1.7% PnL edge** while giving up **half the
Sharpe** and **2.3x the drawdown**. The promote-gate then rejected that
very candidate for MaxDD regression (19.26% vs incumbent 12.60%) -- so
the whole weekly retrain was wasted, when a Sharpe-selected GOLDEN
(0.50, DD -8.73%) would have IMPROVED on the incumbent on every axis
and shipped.

HYPOTHESIS
----------
H1: Selecting GOLDEN by mean Sharpe instead of mean NetPnL produces a
    materially better artifact on risk-adjusted terms at negligible PnL
    cost.
H2 (the big one): the current selection rule is a primary reason DSR has
    never passed. The DSR hurdle this morning was SR0=0.995 against a
    GOLDEN Sharpe of 0.646 (p=0.2559 FAIL). The 0.50 config's mean
    Sharpe is 1.191 -- ABOVE the hurdle. If that config's own DSR clears
    0.95, the system's long-standing "paper-only" blocker is partly a
    config-selection artifact, not purely a signal-strength problem.

METHOD
------
Re-runs the FULL production sweep (same thresholds, same seeds, same
checkpoint) so multiplicity is identical, then scores DSR + PBO for the
GOLDEN chosen under each rule. n_trials is held at the true full-sweep
count (len(thresholds) x len(seeds)) for BOTH arms -- selecting
differently does not reduce how many configs were actually searched, and
quietly shrinking n_trials would inflate DSR (the exact self-deception
`scripts/analyze_dsr_calibration.py` was written to rule out).

READ-ONLY: never writes an artifact, never touches models/saved/.

Run: python scripts/analyze_golden_selection_criterion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_backtest import (  # noqa: E402
    CHECKPOINT_PATH,
    DEFAULT_SWEEP_THRESHOLDS,
    TRADING_DAYS,
    _apply_eval_overrides,
    equity_metrics,
    monthly_net_sharpe,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    load_corporate_actions,
    materialize_dataset,
)
from src.models.macro_risk_hmm import build_regime_observation  # noqa: E402
from src.models.statistical_gates import cscv_pbo, deflated_sharpe  # noqa: E402


def _score_golden(label: str, golden: dict, n_trials: int, cscv_s: int) -> None:
    best = max(golden["per_seed"], key=lambda p: p["net_sharpe"])
    daily_r = best["eq"]["daily_return"].to_numpy()
    dsr = deflated_sharpe(daily_r, n_trials=n_trials, annualisation=TRADING_DAYS)

    gm = golden["monthly_cols"]
    if len(gm) >= 2:
        M = pd.concat(gm, axis=1).fillna(0.0).sort_index()
        S = min(cscv_s, (len(M) // 2) * 2)
        pbo = cscv_pbo(M.to_numpy(), S=max(2, S))
    else:
        pbo = {"pbo": float("nan"), "valid": False}

    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
    print(f"  up_threshold        : {golden['up_threshold']:.2f}")
    print(f"  mean NetPnL         : {golden['mean_net_pnl']:+,.0f} VND")
    print(f"  mean Sharpe         : {golden['mean_sharpe']:+.3f}")
    print(f"  mean MaxDD          : {golden['mean_dd'] * 100:+.2f}%")
    print(f"  mean UP-precision   : {golden['mean_up_precision']:.4f}")
    print(f"  best seed           : {best['seed']}  (Sharpe {best['net_sharpe']:+.3f}, "
          f"DD {best['max_drawdown'] * 100:+.2f}%)")
    print(f"  Deflated Sharpe     : SR={dsr.get('sr_annualised', float('nan')):+.3f}  "
          f"SR0={dsr.get('sr0_annualised', float('nan')):+.3f}")
    p_dsr = dsr.get("p_dsr", float("nan"))
    print(f"  DSR p-value         : {p_dsr:.4f}   "
          f"({'PASS >=0.95' if p_dsr >= 0.95 else 'FAIL <0.95'})")
    pbo_v = pbo.get("pbo", float("nan"))
    print(f"  PBO (CSCV)          : {pbo_v * 100:.1f}%   "
          f"({'PASS <=10%' if pbo_v <= 0.10 else 'FAIL >10%'})")


def main() -> None:
    print(f"Loading checkpoint {CHECKPOINT_PATH} ...")
    ckpt = joblib.load(CHECKPOINT_PATH)
    tabular_features = list(ckpt["tabular_features"])
    trained = list(ckpt["ensembles"])
    macro_hmm = ckpt.get("macro_hmm")
    cutoff = ckpt["cutoff"]
    seeds = [s for s, _ in trained]
    print(f"  seeds={seeds}  features={len(tabular_features)}  cutoff={cutoff}")

    # Eval config comes from the checkpoint's OWN train_cfg (same as
    # run_backtest.main) -- a fresh RunConfig() could silently differ on
    # liquid_top_n / max_positions / initial_capital and make this A/B
    # incomparable to the production sweep it is auditing.
    cfg = _apply_eval_overrides(ckpt["train_cfg"], {})

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)

    # Leak-free filtered P(Bull), recomputed exactly as run_backtest.main does.
    p_bull_series = None
    if macro_hmm is not None:
        try:
            obs = build_regime_observation(
                ds.panel, use_macro=cfg.use_macro_in_hmm, macro_parquet=cfg.macro_parquet)
            p_bull_series = macro_hmm.p_bull_series(obs, filtered=True)
        except Exception as exc:  # noqa: BLE001 -- diagnostic must not die on the overlay
            print(f"  (p_bull_series unavailable: {exc})")

    thresholds = list(DEFAULT_SWEEP_THRESHOLDS)
    n_trials = len(thresholds) * max(1, len(seeds))
    print(f"\nRunning FULL sweep: {len(thresholds)} thresholds x {len(seeds)} seeds "
          f"(n_trials={n_trials}) ...")

    seed_caches: dict[int, dict] = {seed: {} for seed, _ in trained}
    sweep_results: list[dict] = []
    for thr in thresholds:
        sig_thr = thr - 0.05                      # mirrors run_backtest.main exactly
        cfg.signal_threshold = sig_thr
        per_seed: list[dict] = []
        monthly_cols: dict[str, pd.Series] = {}
        for seed, ensemble in trained:
            try:
                eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                             cutoff, cfg, p_bull_series=p_bull_series,
                             inference_cache=seed_caches[seed], mode="tranche", hold_days=30)
                m = equity_metrics(eq, cfg.initial_capital)
                per_seed.append({"seed": seed, "eq": eq, "up_threshold": thr, **m})
                monthly_cols[f"seed_{seed}"] = monthly_net_sharpe(eq)
            except Exception as exc:  # noqa: BLE001 -- one bad seed must not sink the sweep
                print(f"  thr={thr:.2f} seed={seed} FAILED: {exc}")
        if not per_seed:
            continue
        agg = {
            "up_threshold": thr,
            "mean_net_pnl": float(np.mean([p["net_pnl"] for p in per_seed])),
            "mean_sharpe": float(np.mean([p["net_sharpe"] for p in per_seed])),
            "mean_dd": float(np.mean([p["max_drawdown"] for p in per_seed])),
            "mean_up_precision": 0.0,   # not needed for this diagnostic
            "per_seed": per_seed,
            "monthly_cols": monthly_cols,
        }
        sweep_results.append(agg)
        print(f"  thr={thr:.2f}  mean_NetPnL={agg['mean_net_pnl']:+,.0f}  "
              f"mean_Sharpe={agg['mean_sharpe']:+.3f}  mean_DD={agg['mean_dd'] * 100:+.2f}%")

    if not sweep_results:
        print("Sweep produced nothing — aborting.")
        return

    golden_pnl = max(sweep_results, key=lambda r: r["mean_net_pnl"])
    golden_sharpe = max(sweep_results, key=lambda r: r["mean_sharpe"])

    print(f"\n{'=' * 78}\nGOLDEN SELECTION CRITERION A/B  (n_trials={n_trials} for BOTH arms)\n{'=' * 78}")
    _score_golden("ARM A — max mean NetPnL  (CURRENT production rule)",
                  golden_pnl, n_trials, cfg.cscv_S)
    _score_golden("ARM B — max mean Sharpe  (PROPOSED rule)",
                  golden_sharpe, n_trials, cfg.cscv_S)

    if golden_pnl["up_threshold"] == golden_sharpe["up_threshold"]:
        print("\nBoth rules selected the SAME config — no difference on this sweep.")
    else:
        d_pnl = (golden_sharpe["mean_net_pnl"] - golden_pnl["mean_net_pnl"]) / abs(golden_pnl["mean_net_pnl"])
        print(f"\n{'=' * 78}\nDELTA (ARM B vs ARM A)\n{'=' * 78}")
        print(f"  PnL     : {d_pnl * 100:+.2f}%  (cost of switching, if negative)")
        print(f"  Sharpe  : {golden_sharpe['mean_sharpe'] - golden_pnl['mean_sharpe']:+.3f}")
        print(f"  MaxDD   : {(golden_sharpe['mean_dd'] - golden_pnl['mean_dd']) * 100:+.2f}pp "
              f"(negative = deeper, positive = shallower)")

    print("\nNo artifacts written — read-only diagnostic.")


if __name__ == "__main__":
    main()
