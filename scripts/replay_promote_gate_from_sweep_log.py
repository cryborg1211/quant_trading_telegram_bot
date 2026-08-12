"""Replay the promote-gate against a sweep log, WITHOUT saving anything.

WHY
───
`run_backtest.py --no-save` never reaches `_persist_bot_payload`, so a dry sweep
tells you the metrics but not whether the gate would have let them through. That
is the one thing worth knowing before an unattended Saturday retrain: four
consecutive rejections had already frozen the live T+20 artifact at 2026-07-17.

This reconstructs the metadata `_persist_bot_payload` WOULD have stamped from the
log's per-seed lines, then runs the real `_promote_decision` against the real
incumbent artifact — reporting both the current decision and what the pre-12-08
gate (bare relative Sharpe, no sweep-basis check) would have said, so the
difference is visible rather than asserted.

Read-only: loads the incumbent to read its metadata and writes nothing.

    python scripts/replay_promote_gate_from_sweep_log.py logs/parity_sweep_full_*.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_backtest import _promote_decision, _sweep_basis  # noqa: E402

# `thr=0.44 seed=42  NetPnL=+520,982,142  Sharpe=+0.21  DD=-13.85%  predUP=71623  prec=0.4574`
_SEED_RE = re.compile(
    r"thr=(?P<thr>[\d.]+)\s+seed=(?P<seed>\d+)\s+NetPnL=(?P<pnl>[+-][\d,]+)\s+"
    r"Sharpe=(?P<sharpe>[+-][\d.]+)\s+DD=(?P<dd>[+-][\d.]+)%\s+"
    r"predUP=(?P<predup>\d+)\s+prec=(?P<prec>[\d.]+)"
)
# `★ THR=0.43  mean_NetPnL=+1,555,924,715  mean_Sharpe=+0.469 ... mean_UPprec=0.4545`
_THR_RE = re.compile(
    r"THR=(?P<thr>[\d.]+)\s+mean_NetPnL=(?P<pnl>[+-][\d,]+)\s+"
    r"mean_Sharpe=(?P<sharpe>[+-][\d.]+).*?mean_UPprec=(?P<prec>[\d.]+)"
)
_LAYERS_RE = re.compile(r"up_threshold=(?P<thr>[\d.]+)\s+engine_gate=(?P<gate>[\d.]+)\s+"
                        r"layers=(?P<layers>\w+)")


def _parse(log: Path) -> tuple[dict, dict, dict]:
    """→ (per-threshold seed rows, per-threshold means, gate/layer conditions)."""
    text = log.read_text(encoding="utf-8", errors="replace")
    seeds: dict[float, list[dict]] = {}
    for m in _SEED_RE.finditer(text):
        thr = float(m["thr"])
        seeds.setdefault(thr, []).append({
            "seed": int(m["seed"]),
            "net_pnl": float(m["pnl"].replace(",", "")),
            "net_sharpe": float(m["sharpe"]),
            "max_drawdown": float(m["dd"]) / 100.0,
            "n_pred_up": int(m["predup"]),
            "up_precision": float(m["prec"]),
        })
    means = {float(m["thr"]): {"mean_net_pnl": float(m["pnl"].replace(",", "")),
                               "mean_sharpe": float(m["sharpe"]),
                               "mean_up_precision": float(m["prec"])}
             for m in _THR_RE.finditer(text)}
    cond = {}
    for m in _LAYERS_RE.finditer(text):
        cond[float(m["thr"])] = {"engine_gate": float(m["gate"]),
                                 "layers": m["layers"]}
    return seeds, means, cond


def main() -> None:
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not log.exists():
        print(f"log not found: {log}")
        raise SystemExit(2)

    seeds, means, cond = _parse(log)
    if not means:
        print("No completed threshold arms in the log yet "
              f"({len(seeds)} threshold(s) with partial seed rows). Still running?")
        for thr in sorted(seeds):
            print(f"  thr={thr:.2f}  seeds done={len(seeds[thr])}")
        raise SystemExit(1)

    print(f"{'thr':>6} {'mean Sharpe':>12} {'best Sharpe':>12} {'best DD':>9} "
          f"{'mean PnL':>18} {'UPprec':>8} {'gate':>6} {'layers':>7}")
    for thr in sorted(means, reverse=True):
        rows = seeds.get(thr, [])
        best = max(rows, key=lambda r: r["net_sharpe"]) if rows else None
        c = cond.get(thr, {})
        print(f"{thr:6.2f} {means[thr]['mean_sharpe']:+12.3f} "
              f"{(best['net_sharpe'] if best else float('nan')):+12.3f} "
              f"{(best['max_drawdown'] if best else float('nan')):9.2%} "
              f"{means[thr]['mean_net_pnl']:+18,.0f} "
              f"{means[thr]['mean_up_precision']:8.4f} "
              f"{c.get('engine_gate', float('nan')):6.2f} {c.get('layers', '?'):>7}")

    # GOLDEN = max mean OOS Net PnL across seeds (run_backtest.py step 4).
    golden_thr = max(means, key=lambda t: means[t]["mean_net_pnl"])
    rows = seeds[golden_thr]
    best = max(rows, key=lambda r: r["net_sharpe"])
    print(f"\nGOLDEN (max mean NetPnL) = thr {golden_thr:.2f}   "
          f"best seed {best['seed']}  Sharpe {best['net_sharpe']:+.3f}  "
          f"DD {best['max_drawdown']:.2%}")

    # Exactly the fields `_persist_bot_payload` stamps from best_seed_record.
    new_meta = {
        "oos_sharpe": best["net_sharpe"],
        "oos_max_dd": best["max_drawdown"],
        "golden_mean_up_precision": means[golden_thr]["mean_up_precision"],
        "trained_at": "REPLAY",
        "sweep_conditions": {
            "gate_offset": 0.0,
            "engine_gate": cond.get(golden_thr, {}).get("engine_gate", golden_thr),
            "admission_mode": "cross_sectional",
            "serve_sector_cap": 2, "serve_cohort_dedup": True,
            "serve_hysteresis_days": 2, "serve_exposure_brake": True,
            "use_regime_sizing": False, "max_positions": 5,
        },
    }

    for horizon in (20, 5):
        path = Path(f"models/saved/v3_ensemble_{horizon}d.joblib")
        if not path.exists():
            print(f"\nT+{horizon}: no incumbent on disk")
            continue
        old_meta = joblib.load(path).get("metadata") or {}
        old_sharpe = float(old_meta.get("oos_sharpe", float("nan")))
        print(f"\n── T+{horizon} incumbent: Sharpe {old_sharpe:+.3f}  "
              f"DD {abs(float(old_meta.get('oos_max_dd', 0))):.2%}  "
              f"basis={_sweep_basis(old_meta)}")

        # Pre-12-08 behaviour: bare relative Sharpe, no basis check.
        would_old_reject = new_meta["oos_sharpe"] < old_sharpe - 0.10
        print(f"   OLD gate (bare relative): "
              f"{'REJECT' if would_old_reject else 'promote'}  "
              f"(needs >= {old_sharpe - 0.10:+.3f}, has "
              f"{new_meta['oos_sharpe']:+.3f})")

        promote, reason = _promote_decision(new_meta, old_meta, -0.10, 3.0, 0.35)
        print(f"   NEW gate: {'PROMOTE' if promote else 'REJECT'}")
        print(f"     {reason}")


if __name__ == "__main__":
    main()
