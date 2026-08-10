"""Is thr=0.50 + burst nav/10 a stable config, or one lucky window?
(10-08-26 — the test that actually settles the PBO 39.2 pct question.)

WHY THIS AND NOT MORE GATE ARGUMENTS
------------------------------------
Burst sizing lifted the seed-averaged config to Sharpe 1.380 / +97 pct over
919 OOS days, and Sharpe ROSE with deployment, which is real evidence. But
two flags say the config may still be cherry-picked, and neither is fixed by
deploying more capital:
  * PBO 39.2 pct -- the CSCV estimate of the chance this config lands below
    the median out-of-sample.
  * thr=0.50 never fired ONCE on the longer 2020-2026 window (08-08 test).
Bigger size on a fake edge only makes the loss bigger, so the question is
not "how much did it make" but "does it survive a window it was not chosen
on".

METHOD -- same model, different windows, so window effect is isolated from
model effect (the 08-08 attempt confounded the two by retraining as well):
the 919-day OOS span is cut in half and each half is run on its own, with
the full span as reference. The model, features, threshold and budget are
byte-identical across arms; only the dates change.

READ IT LIKE THIS
  * Sharpe holds in BOTH halves -> PBO 39.2 pct is plausibly an artifact of
    how CSCV partitions, and the config deserves far more trust.
  * Sharpe good in one half and poor in the other -> confirmed lucky window,
    learned cheaply, and the burst-sizing headline should be discarded.

KNOWN ARTIFACT: hold=30 means cohorts opened near a window's end cannot
close inside it, so both halves carry some truncation drag. It applies to
both arms roughly equally and to the full-span reference at its own end, so
it does not favour either half.

READ-ONLY. Run: python scripts/analyze_window_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from analyze_dsr_pass_attempt import SeedEnsemble  # noqa: E402
from run_backtest import (  # noqa: E402
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    load_corporate_actions,
    materialize_dataset,
)
from src.models.macro_risk_hmm import build_regime_observation  # noqa: E402

CKPT = REPO / "models" / "saved" / "t20_std_checkpoint.joblib"
WINNING_THR = 0.50
BUDGETS = (None, 10)          # None => nav/hold_days baseline; 10 => the burst winner


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
    print(f"Checkpoint: train_frac={ckpt['train_cfg'].train_frac}  cutoff={cutoff}  "
          f"seeds={[s for s, _ in trained]}")

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

    # Midpoint of the OOS trading dates.
    oos_dates = np.array(sorted(
        d for d in ds.panel.select("date").unique().to_series().to_list() if d >= cutoff))
    mid = oos_dates[len(oos_dates) // 2]
    print(f"OOS span: {oos_dates[0]} .. {oos_dates[-1]}  ({len(oos_dates)} dates)")
    print(f"  split at {mid}\n")

    cfg.signal_threshold = WINNING_THR - 0.05

    # (label, panel, cutoff). Half 1 truncates the panel END (run_oos has no
    # end-date argument); run_oos still takes its own 80-bar warm-up buffer
    # from before the cutoff, which exists in both cases.
    windows = [
        ("FULL  ", ds.panel, cutoff),
        ("HALF 1", ds.panel.filter(pl.col("date") <= mid), cutoff),
        ("HALF 2", ds.panel, mid),
    ]

    print(f"{'window':<9}{'budget':<15}{'days':>6}{'NetPnL':>18}{'TotRet':>9}"
          f"{'Sharpe':>9}{'MaxDD':>9}")
    results: dict[tuple[str, str], dict] = {}
    for wlabel, panel, cut in windows:
        cache: dict = {}
        for bd in BUDGETS:
            blabel = "nav/hold(30)" if bd is None else f"BURST nav/{bd}"
            try:
                eq = run_oos(panel, feats, model, corporate_actions, cut, cfg,
                             p_bull_series=p_bull, inference_cache=cache,
                             mode="tranche", hold_days=30, tranche_budget_days=bd)
            except Exception as exc:  # noqa: BLE001
                print(f"{wlabel:<9}{blabel:<15}FAILED: {type(exc).__name__}: {str(exc)[:50]}")
                continue
            m = equity_metrics(eq, cfg.initial_capital)
            results[(wlabel.strip(), blabel)] = m
            print(f"{wlabel:<9}{blabel:<15}{m['n_days']:>6}{m['net_pnl']:>+18,.0f}"
                  f"{m['total_return'] * 100:>+8.2f}%{m['net_sharpe']:>+9.3f}"
                  f"{m['max_drawdown'] * 100:>8.2f}%")

    print(f"\n{'=' * 80}\nSTABILITY VERDICT\n{'=' * 80}")
    for blabel in ("nav/hold(30)", "BURST nav/10"):
        h1 = results.get(("HALF 1", blabel))
        h2 = results.get(("HALF 2", blabel))
        full = results.get(("FULL", blabel))
        if not (h1 and h2 and full):
            print(f"  {blabel}: incomplete")
            continue
        s1, s2 = h1["net_sharpe"], h2["net_sharpe"]
        both_pos = s1 > 0 and s2 > 0
        spread = abs(s1 - s2)
        print(f"\n  {blabel}")
        print(f"    Sharpe  full={full['net_sharpe']:+.3f}  h1={s1:+.3f}  h2={s2:+.3f}"
              f"   spread={spread:.3f}")
        print(f"    MaxDD   full={full['max_drawdown'] * 100:+.2f}%  "
              f"h1={h1['max_drawdown'] * 100:+.2f}%  h2={h2['max_drawdown'] * 100:+.2f}%")
        if both_pos and spread < 0.5:
            print("    -> STABLE: positive in both halves and close together.")
        elif both_pos:
            print("    -> MIXED: positive in both halves but far apart — the full-span")
            print("       number is an average of quite different regimes.")
        else:
            print("    -> UNSTABLE: one half is non-positive. The full-span headline")
            print("       rests on a single favourable stretch — treat it as luck.")

    print("\nNo artifacts written — read-only diagnostic.")


if __name__ == "__main__":
    main()
