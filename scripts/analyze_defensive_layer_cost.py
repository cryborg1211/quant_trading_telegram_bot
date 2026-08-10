"""What do the defensive layers actually COST? (10-08-26, user-driven.)

THE GAP THIS CLOSES
-------------------
The burst-sizing headline (seed-averaged thr=0.50 + nav/10: Sharpe 1.380,
DD -16.13 pct, +97 pct over 919 days) was measured with LESS defence than
production runs. Verified by grep:
    HMM p_bull overlay        backtest YES   serve YES
    regime sizing             backtest NO    serve YES  (use_regime_sizing
                                                        defaults False)
    GARCH+drift+breadth brake backtest NO    serve YES  (live_exposure_scalar
                                                        exists only in main.py
                                                        and src/bot/garch_brake.py
                                                        -- walk_forward never
                                                        calls it)
    portfolio guard           backtest NO    serve YES  (alert-only)
So -16.13 pct DD is an UPPER bound on live drawdown and +97 pct is an UPPER
bound on live PnL. Reporting one without the other overstates the config.

WHY THE COST MIGHT BE LARGE -- prior evidence in this repo
---------------------------------------------------------
The 14-06-26 regime-sizing A/B: MaxDD -23.3 -> -16.9 pct and Sharpe
+0.73 -> +0.88, but Net PnL +46 -> +42 pct. And the 14-07 measurement found
defensive regimes cover 70-82 pct of the universe PERMANENTLY -- so the
regime layer behaves as a per-name character classifier, not a market timer,
and its DD reduction comes from chronic de-risking. That is a standing tax,
not crash insurance, which is exactly the concern this script measures.

ARMS (seed-averaged thr=0.50, cross_sectional, hold=30, on the same
919-day OOS span -- only the defensive flags change):
  1. nav/hold(30), no regime sizing      -- the 1.231 reference
  2. BURST nav/10, no regime sizing      -- the 1.380 headline
  3. BURST nav/10, REGIME SIZING ON      -- closer to what serve would do
  4. nav/hold(30), REGIME SIZING ON      -- isolates the layer's cost alone

The 3-leg GARCH/drift/breadth brake still cannot be measured here: it is
serve-only code with no backtest path. So even arm 3 remains optimistic
versus live. Stated, not hidden.

READ-ONLY. Run: python scripts/analyze_defensive_layer_cost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import joblib  # noqa: E402

from analyze_dsr_pass_attempt import SeedEnsemble  # noqa: E402
from run_backtest import (  # noqa: E402
    TRADING_DAYS,
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    load_corporate_actions,
    materialize_dataset,
)
from src.models.macro_risk_hmm import build_regime_observation  # noqa: E402
from src.models.statistical_gates import deflated_sharpe  # noqa: E402

CKPT = REPO / "models" / "saved" / "t20_std_checkpoint.joblib"
WINNING_THR = 0.50
COMFORT_DD = 0.13          # the production drawdown comfort band


def main() -> None:
    if not CKPT.exists():
        print(f"Missing {CKPT} — run scripts/run_seed_average_clean.ps1 first.")
        return

    ckpt = joblib.load(CKPT)
    feats = list(ckpt["tabular_features"])
    trained = list(ckpt["ensembles"])
    macro_hmm = ckpt.get("macro_hmm")
    cutoff = ckpt["cutoff"]
    cfg = _apply_eval_overrides(ckpt["train_cfg"], {})
    model = SeedEnsemble([e for _, e in trained])
    print(f"Checkpoint: train_frac={ckpt['train_cfg'].train_frac}  cutoff={cutoff}")

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)

    p_bull = None
    if macro_hmm is not None:
        try:
            obs = build_regime_observation(
                ds.panel, use_macro=cfg.use_macro_in_hmm, macro_parquet=cfg.macro_parquet)
            p_bull = macro_hmm.p_bull_series(obs, filtered=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (p_bull unavailable: {exc})")

    cfg.signal_threshold = WINNING_THR - 0.05
    print(f"Config: seed-averaged, cross_sectional, signal_threshold="
          f"{cfg.signal_threshold:.2f}, hold=30\n")

    arms = [
        ("nav/hold(30)  regime OFF", None, False),
        ("BURST nav/10  regime OFF", 10, False),
        ("BURST nav/10  regime ON ", 10, True),
        ("nav/hold(30)  regime ON ", None, True),
    ]

    cache: dict = {}
    print(f"{'arm':<26}{'NetPnL':>18}{'TotRet':>9}{'Sharpe':>9}{'MaxDD':>9}{'DSRn15':>8}")
    res: dict[str, dict] = {}
    for label, bd, regime in arms:
        try:
            eq = run_oos(ds.panel, feats, model, corporate_actions, cutoff, cfg,
                         p_bull_series=p_bull, inference_cache=cache,
                         mode="tranche", hold_days=30,
                         tranche_budget_days=bd, use_regime_sizing=regime)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<26}FAILED: {type(exc).__name__}: {str(exc)[:50]}")
            continue
        m = equity_metrics(eq, cfg.initial_capital)
        d15 = deflated_sharpe(eq["daily_return"].to_numpy(), n_trials=15,
                              annualisation=TRADING_DAYS).get("p_dsr", float("nan"))
        res[label.strip()] = m
        print(f"{label:<26}{m['net_pnl']:>+18,.0f}{m['total_return'] * 100:>+8.2f}%"
              f"{m['net_sharpe']:>+9.3f}{m['max_drawdown'] * 100:>8.2f}%{d15:>8.3f}")

    b_off = res.get("BURST nav/10  regime OFF")
    b_on = res.get("BURST nav/10  regime ON")
    n_off = res.get("nav/hold(30)  regime OFF")
    n_on = res.get("nav/hold(30)  regime ON")

    print(f"\n{'=' * 82}\nWHAT REGIME SIZING COSTS\n{'=' * 82}")
    for tag, off, on in (("BURST nav/10", b_off, b_on), ("nav/hold(30)", n_off, n_on)):
        if not (off and on):
            continue
        pnl_keep = (on["net_pnl"] / off["net_pnl"]) if off["net_pnl"] else float("nan")
        print(f"\n  {tag}")
        print(f"    PnL   {off['net_pnl']:+,.0f} -> {on['net_pnl']:+,.0f}   "
              f"(keeps {pnl_keep * 100:.0f}% of it)")
        print(f"    Sharpe {off['net_sharpe']:+.3f} -> {on['net_sharpe']:+.3f}   "
              f"({on['net_sharpe'] - off['net_sharpe']:+.3f})")
        print(f"    MaxDD  {off['max_drawdown'] * 100:+.2f}% -> {on['max_drawdown'] * 100:+.2f}%   "
              f"({(on['max_drawdown'] - off['max_drawdown']) * 100:+.2f}pp)")

    if b_on:
        inside = abs(b_on["max_drawdown"]) <= COMFORT_DD
        print(f"\n  BURST nav/10 + regime ON is "
              f"{'INSIDE' if inside else 'OUTSIDE'} the {COMFORT_DD * 100:.0f}% comfort band "
              f"(DD {b_on['max_drawdown'] * 100:+.2f}%)")
        if b_off and b_on["net_sharpe"] < b_off["net_sharpe"] - 0.15:
            print("  -> the layer is eating the gain, not just the risk: this is the")
            print("     over-defence case. Levers: raise REGIME_PENALTY_FACTOR above")
            print("     0.5, or shrink NO_TRADE_REGIMES {0,7}.")

    print("\n  Still NOT measured here: the 3-leg GARCH/drift/breadth brake is")
    print("  serve-only with no backtest path, so every arm above remains")
    print("  optimistic relative to live.")
    print("\nNo artifacts written — read-only diagnostic.")


if __name__ == "__main__":
    main()
