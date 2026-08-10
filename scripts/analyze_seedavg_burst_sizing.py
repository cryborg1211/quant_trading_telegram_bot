"""Is the seed-averaged Sharpe 1.231 real edge, or just an artifact of
sitting in cash? (09-08-26, user-driven.)

THE CRITIQUE THIS ANSWERS
-------------------------
The clean seed-average run selected thr=0.50: Sharpe +1.231, MaxDD -7.85%,
+26.10 pct total over 919 days. That is roughly 6.4 pct a year -- around a VN
term-deposit rate -- while the book sits idle most of the time. A Sharpe
computed over mostly-cash days is flattered: cash contributes zero variance
to the denominator. So the number may say "good at not trading" rather than
"good at trading".

Burst sizing (`WalkForwardConfig.tranche_budget_days`, shipped c8e3300) is
the existing lever for exactly this: when a selective gate opens rarely, the
calendar budget (nav / hold_days) barely deploys before returning to cash.
It was A/B'd once, at gate 0.46, where nav/10 tripled PnL (+661M -> +1.95B)
for a flat Sharpe and DD -4.44 pct -> -12.86 pct. It has NEVER been tested
on the seed-averaged thr=0.50 config, which starts from a much better
Sharpe and a -7.85 pct DD -- i.e. more room to spend.

BOTH OUTCOMES ARE INFORMATIVE, which is why this is worth running:
  * PnL scales up and Sharpe holds -> the edge is real and was simply
    under-deployed.
  * Sharpe collapses once capital is actually at work -> the 1.231 was a
    low-exposure artifact, and we learn that cheaply instead of trusting it.

MULTIPLICITY, stated honestly: the threshold sweep was already 5 configs;
adding 3 budget values makes the real search 15. DSR is therefore reported
at n_trials=5 (budget fixed a priori) AND n_trials=15 (budget searched too).
The 15 number is the honest one for a decision made after seeing this table.

READ-ONLY. Run: python scripts/analyze_seedavg_burst_sizing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

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
WINNING_THR = 0.50          # the seed-averaged sweep's Sharpe-selected GOLDEN
BUDGET_ARMS = (None, 10, 5)  # None => nav/hold_days (the 1.231 baseline)


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
    print(f"Checkpoint: tb_horizon={ckpt['train_cfg'].tb_horizon}  "
          f"train_frac={ckpt['train_cfg'].train_frac}  cutoff={cutoff}  "
          f"seeds={[s for s, _ in trained]}")

    model = SeedEnsemble([e for _, e in trained])

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

    # Reproduce the winning config exactly: cross_sectional admission with
    # signal_threshold = thr - 0.05, same as the seed-average sweep.
    cfg.signal_threshold = WINNING_THR - 0.05
    print(f"\nConfig: seed-averaged, cross_sectional, signal_threshold="
          f"{cfg.signal_threshold:.2f} (thr={WINNING_THR:.2f}), hold=30\n")

    cache: dict = {}
    print(f"{'budget':<18}{'NetPnL':>18}{'TotRet':>9}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'DSR n=5':>9}{'DSR n=15':>10}")
    rows = []
    for bd in BUDGET_ARMS:
        label = "nav/hold(30)" if bd is None else f"BURST nav/{bd}"
        try:
            eq = run_oos(ds.panel, feats, model, corporate_actions, cutoff, cfg,
                         p_bull_series=p_bull, inference_cache=cache,
                         mode="tranche", hold_days=30, tranche_budget_days=bd)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<18}FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        m = equity_metrics(eq, cfg.initial_capital)
        r = eq["daily_return"].to_numpy()
        d5 = deflated_sharpe(r, n_trials=5, annualisation=TRADING_DAYS).get("p_dsr", float("nan"))
        d15 = deflated_sharpe(r, n_trials=15, annualisation=TRADING_DAYS).get("p_dsr", float("nan"))
        rows.append((label, m, d5, d15))
        print(f"{label:<18}{m['net_pnl']:>+18,.0f}{m['total_return'] * 100:>+8.2f}%"
              f"{m['net_sharpe']:>+9.3f}{m['max_drawdown'] * 100:>8.2f}%"
              f"{d5:>9.3f}{d15:>10.3f}")

    if len(rows) >= 2:
        base = rows[0][1]
        print(f"\n{'=' * 84}\nDELTA vs the nav/hold(30) baseline\n{'=' * 84}")
        for label, m, _d5, _d15 in rows[1:]:
            pnl_x = (m["net_pnl"] / base["net_pnl"]) if base["net_pnl"] else float("nan")
            print(f"  {label:<16} PnL x{pnl_x:.2f}   "
                  f"Sharpe {m['net_sharpe'] - base['net_sharpe']:+.3f}   "
                  f"DD {(m['max_drawdown'] - base['max_drawdown']) * 100:+.2f}pp")

    print("\nDecision lens: burst sizing is worth it only if PnL scales up while")
    print("Sharpe holds and DD stays inside the ~13 pct production comfort band.")
    print("A Sharpe collapse means the 1.231 came from low exposure, not edge.")
    print("\nNo artifacts written — read-only diagnostic.")


if __name__ == "__main__":
    main()
