"""
scripts/sweep_garch_hmm_brake.py — Robustness grid for the GARCH-HMM linear brake.

The single A/B (seed 0, floor 0.2, cap 0.96) showed Sharpe −0.36→−0.15. This
sweep checks that gain is NOT threshold-mined: it walks a grid of
  • min_exposure (floor)  ∈ floors
  • max_persistence (cap) ∈ caps
and reports Sharpe / MaxDD / TotalReturn per cell against ONE shared baseline.

Efficiency
──────────
Baseline walk-forward (no brake) is INVARIANT across every cell, and the
dataset materialization is expensive — both run ONCE. Only the braked arm
re-runs per cell. The GARCH-HMM refit per `cap` is cheap (~seconds); the cost
is the per-cell braked walk-forward.

Leakage discipline is inherited from validate_garch_hmm_brake.py: GARCH-HMM is
trained strictly on obs < cutoff; the scaler is filtered (leak-free).

Run
───
    python scripts/sweep_garch_hmm_brake.py
    python scripts/sweep_garch_hmm_brake.py --floors 0.1,0.2,0.3 --caps 0.94,0.96,0.98
    python scripts/sweep_garch_hmm_brake.py --seed-idx 1 --hold-days 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure repo root on sys.path ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    configure_logging,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.garch_hmm_regime import train_garch_hmm  # noqa: E402

# Reuse the validated building blocks (no duplicated WF wiring).
from scripts.validate_garch_hmm_brake import (  # noqa: E402
    _run_wf,
    _build_macro_obs,
    _metrics,
    _CHECKPOINT_5D,
    _MACRO_PARQUET,
)

LOGGER = logging.getLogger("quant.sweep_garch_hmm")

_OUT = Path("process/features/macro-integration/reports/garch_hmm_brake_sweep.csv")


def main(
    *,
    checkpoint_path: Path,
    macro_parquet: Path,
    n_states: int,
    floors: list[float],
    caps: list[float],
    hold_days: int,
    seed_idx: int,
    out_path: Path,
) -> pd.DataFrame:
    configure_logging()
    t0 = time.perf_counter()

    LOGGER.info("=" * 72)
    LOGGER.info(" GARCH-HMM BRAKE SWEEP | floors=%s  caps=%s  states=%d  seed_idx=%d",
                floors, caps, n_states, seed_idx)
    LOGGER.info("=" * 72)

    # ── Load checkpoint + materialize once ───────────────────────────────
    ckpt = joblib.load(checkpoint_path)
    cfg: RunConfig = ckpt["train_cfg"]
    tabular_features: list[str] = ckpt["tabular_features"]
    cutoff = ckpt["cutoff"]
    ensembles = ckpt["ensembles"]
    if seed_idx >= len(ensembles):
        raise ValueError(f"seed_idx={seed_idx} but only {len(ensembles)} seeds")
    seed, ensemble = ensembles[seed_idx]
    LOGGER.info("Checkpoint seed=%d  cutoff=%s  features=%d", seed, cutoff, len(tabular_features))

    ds = materialize_dataset(cfg)
    ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
    corporate_actions = load_corporate_actions(cfg)

    obs = _build_macro_obs(ds.panel, macro_parquet)
    cutoff_ts = pd.Timestamp(cutoff)
    obs_train = obs[obs.index < cutoff_ts]
    ic = cfg.initial_capital

    # ── Baseline ONCE (invariant) ────────────────────────────────────────
    LOGGER.info("Baseline walk-forward (no brake)...")
    eq_base = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                      cutoff, cfg, p_bull_series=None, hold_days=hold_days)
    m_base = _metrics(eq_base, ic)
    LOGGER.info("Baseline | Sharpe=%.3f  MaxDD=%.2f%%  Ret=%.2f%%",
                m_base["sharpe"], m_base["max_dd"], m_base["total_ret"])

    # ── Grid: refit GARCH-HMM per cap, braked WF per (cap, floor) ────────
    rows: list[dict] = []
    for cap in caps:
        garch = train_garch_hmm(obs_train, n_states=n_states, seed=cfg.seed,
                                n_restarts=20, max_persistence=cap)
        persistence = garch.garch_params["alpha"] + garch.garch_params["beta"]
        for floor in floors:
            scaler = garch.exposure_scaler(obs, min_exposure=floor, max_exposure=1.0)
            oos_scaler = scaler[scaler.index >= cutoff_ts]
            LOGGER.info("Cell cap=%.2f floor=%.2f | persist=%.4f mean_exp=%.3f → braked WF...",
                        cap, floor, persistence, float(oos_scaler.mean()))
            eq = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                         cutoff, cfg, p_bull_series=scaler.rename("p_bull"),
                         hold_days=hold_days)
            m = _metrics(eq, ic)
            rows.append({
                "cap": cap, "floor": floor, "persistence": round(persistence, 4),
                "mean_exposure": round(float(oos_scaler.mean()), 3),
                "sharpe": m["sharpe"], "max_dd": m["max_dd"], "total_ret": m["total_ret"],
                "d_sharpe": round(m["sharpe"] - m_base["sharpe"], 3),
                "d_max_dd": round(m["max_dd"] - m_base["max_dd"], 2),
                "d_total_ret": round(m["total_ret"] - m_base["total_ret"], 2),
            })

    grid = pd.DataFrame(rows)

    # ── Print + save ─────────────────────────────────────────────────────
    sep = "─" * 84
    print(f"\n{sep}")
    print(f"  GARCH-HMM BRAKE ROBUSTNESS SWEEP | seed_idx={seed_idx}  hold={hold_days}d")
    print(f"  BASELINE: Sharpe={m_base['sharpe']:.3f}  MaxDD={m_base['max_dd']:.2f}%  "
          f"Ret={m_base['total_ret']:.2f}%")
    print(sep)
    print(f"  {'cap':>5} {'floor':>6} {'persist':>8} {'mean_exp':>9} "
          f"{'Sharpe':>8} {'ΔSh':>7} {'MaxDD%':>8} {'ΔDD':>7} {'Ret%':>8} {'ΔRet':>7}")
    print(sep)
    for r in rows:
        print(f"  {r['cap']:>5.2f} {r['floor']:>6.2f} {r['persistence']:>8.4f} "
              f"{r['mean_exposure']:>9.3f} {r['sharpe']:>8.3f} {r['d_sharpe']:>+7.3f} "
              f"{r['max_dd']:>8.2f} {r['d_max_dd']:>+7.2f} {r['total_ret']:>8.2f} "
              f"{r['d_total_ret']:>+7.2f}")
    print(sep)

    # Robustness verdict: fraction of cells improving BOTH Sharpe and MaxDD.
    improved = sum(1 for r in rows if r["d_sharpe"] > 0 and r["d_max_dd"] > 0)
    frac = improved / max(len(rows), 1)
    print(f"  Cells improving BOTH Sharpe & MaxDD: {improved}/{len(rows)} ({frac*100:.0f}%)")
    if frac >= 0.8:
        verdict = "ROBUST — gain holds across the grid, not threshold-mined"
    elif frac >= 0.5:
        verdict = "PARTIAL — gain is grid-sensitive; pick floor/cap carefully"
    else:
        verdict = "FRAGILE — gain does not generalize; likely overfit"
    print(f"  VERDICT: {verdict}")
    print(f"{sep}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_path, index=False)
    LOGGER.info("Sweep grid saved → %s", out_path)
    LOGGER.info("Wall-clock: %.1fs (%d cells + 1 baseline)", time.perf_counter() - t0, len(rows))
    return grid


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _cli() -> dict:
    p = argparse.ArgumentParser(description="Robustness grid for the GARCH-HMM linear brake.")
    p.add_argument("--checkpoint", type=Path, default=_CHECKPOINT_5D)
    p.add_argument("--macro-parquet", type=Path, default=_MACRO_PARQUET)
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--floors", type=_parse_floats, default=[0.1, 0.2, 0.3])
    p.add_argument("--caps", type=_parse_floats, default=[0.94, 0.96, 0.98])
    p.add_argument("--hold-days", type=int, default=5)
    p.add_argument("--seed-idx", type=int, default=0)
    p.add_argument("--out", type=Path, default=_OUT)
    a = p.parse_args()
    return {
        "checkpoint_path": a.checkpoint,
        "macro_parquet": a.macro_parquet,
        "n_states": a.n_states,
        "floors": a.floors,
        "caps": a.caps,
        "hold_days": a.hold_days,
        "seed_idx": a.seed_idx,
        "out_path": a.out,
    }


if __name__ == "__main__":
    try:
        main(**_cli())
    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(level=logging.INFO)
        LOGGER.error("FATAL: %s", exc, exc_info=True)
        sys.exit(1)
