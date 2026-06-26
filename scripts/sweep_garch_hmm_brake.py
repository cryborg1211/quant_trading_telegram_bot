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


def _cell_key(cap: float, floor: float) -> str:
    """Stable id for a (cap, floor) grid cell."""
    return f"{cap:.2f}_{floor:.2f}"


def _baseline_path(out_path: Path, seed_idx: int, hold_days: int) -> Path:
    """Sidecar JSON caching the (invariant) baseline metrics for this run."""
    return out_path.with_name(f"{out_path.stem}_baseline_s{seed_idx}_h{hold_days}.json")


def _load_done(out_path: Path) -> tuple[list[dict], set[str]]:
    """Read already-computed grid rows (resume after a crash). Empty if absent."""
    if not out_path.exists():
        return [], set()
    try:
        df = pd.read_csv(out_path)
    except Exception:  # noqa: BLE001 — corrupt/partial CSV → start fresh
        return [], set()
    rows = df.to_dict("records")
    done = {_cell_key(float(r["cap"]), float(r["floor"])) for r in rows}
    return rows, done


def _append_row(out_path: Path, row: dict, write_header: bool) -> None:
    """Append one completed cell to the CSV immediately (crash-durable)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)


def _const_key(mean_exposure: float) -> str:
    """Match a braked cell to its flat-leverage control by rounded mean exposure."""
    return f"{round(float(mean_exposure), 2):.2f}"


def _constants_path(out_path: Path, seed_idx: int, hold_days: int) -> Path:
    """Sidecar JSON caching constant-exposure control arms {const: metrics}."""
    return out_path.with_name(f"{out_path.stem}_const_s{seed_idx}_h{hold_days}.json")


def _load_constants(path: Path) -> dict:
    if not path.exists():
        return {}
    import json  # noqa: PLC0415
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt sidecar → recompute
        return {}


def _save_constants(path: Path, consts: dict) -> None:
    import json  # noqa: PLC0415
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(consts), encoding="utf-8")


def _timing_alpha(row: dict, consts: dict) -> float | None:
    """Braked Sharpe − matched flat-leverage Sharpe (isolates timing vs de-leverage)."""
    c = consts.get(_const_key(row["mean_exposure"]))
    if c is None:
        return None
    return round(row["sharpe"] - c["sharpe"], 3)


def _print_summary(rows: list[dict], m_base: dict, consts: dict,
                   seed_idx: int, hold_days: int) -> None:
    import statistics  # noqa: PLC0415

    sep = "─" * 96
    print(f"\n{sep}")
    print(f"  GARCH-HMM BRAKE ROBUSTNESS SWEEP | seed_idx={seed_idx}  hold={hold_days}d")
    print(f"  BASELINE: Sharpe={m_base['sharpe']:.3f}  MaxDD={m_base['max_dd']:.2f}%  "
          f"Ret={m_base['total_ret']:.2f}%")
    print(sep)
    print(f"  {'cap':>5} {'floor':>6} {'mean_exp':>9} {'Sharpe':>8} {'ΔSh':>7} "
          f"{'MaxDD%':>8} {'ΔDD':>7} | {'const_Sh':>9} {'TIMING_α':>9}")
    print(sep)
    timing_alphas: list[float] = []
    for r in sorted(rows, key=lambda x: (x["cap"], x["floor"])):
        c = consts.get(_const_key(r["mean_exposure"]))
        c_sh = c["sharpe"] if c else float("nan")
        ta = _timing_alpha(r, consts)
        if ta is not None:
            timing_alphas.append(ta)
        ta_str = f"{ta:>+9.3f}" if ta is not None else f"{'n/a':>9}"
        print(f"  {r['cap']:>5.2f} {r['floor']:>6.2f} {r['mean_exposure']:>9.3f} "
              f"{r['sharpe']:>8.3f} {r['d_sharpe']:>+7.3f} {r['max_dd']:>8.2f} "
              f"{r['d_max_dd']:>+7.2f} | {c_sh:>9.3f} {ta_str}")
    print(sep)

    # De-leverage robustness (vs no-brake baseline) — the original (inflated) check.
    improved = sum(1 for r in rows if r["d_sharpe"] > 0 and r["d_max_dd"] > 0)
    frac = improved / max(len(rows), 1)
    print(f"  vs baseline: {improved}/{len(rows)} cells improve both Sharpe & MaxDD "
          f"({frac*100:.0f}%)")

    # The REAL test: does timing beat flat de-leverage at the SAME exposure?
    if timing_alphas:
        med_ta = statistics.median(timing_alphas)
        n_pos = sum(1 for a in timing_alphas if a > 0)
        print(f"  TIMING vs flat de-leverage: median α={med_ta:+.3f}  "
              f"({n_pos}/{len(timing_alphas)} cells beat their matched constant)")
        if med_ta > 0.02 and n_pos >= 0.7 * len(timing_alphas):
            verdict = "TIMING ADDS — brake beats flat de-leverage at equal exposure"
        elif med_ta > -0.02:
            verdict = ("NO TIMING EDGE — the gain is just holding less; a flat "
                       "constant-exposure book matches it")
        else:
            verdict = "TIMING HURTS — brake is worse than flat de-leverage"
    else:
        verdict = "constants pending — re-run to fill the control arm"
    print(f"  VERDICT: {verdict}")
    print(f"{sep}\n")


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

    # Per-seed grid CSV so multi-seed runs never mix rows into one file. The
    # baseline/const sidecars are already seed-keyed via _baseline_path /
    # _constants_path (which derive from the un-suffixed out_path).
    grid_path = out_path.with_name(
        f"{out_path.stem}_s{seed_idx}_h{hold_days}{out_path.suffix}")

    LOGGER.info("=" * 72)
    LOGGER.info(" GARCH-HMM BRAKE SWEEP | floors=%s  caps=%s  states=%d  seed_idx=%d",
                floors, caps, n_states, seed_idx)
    LOGGER.info(" grid CSV: %s", grid_path.name)
    LOGGER.info("=" * 72)

    # ── Resume bookkeeping (crash-durable; restarts are cheap) ───────────
    all_cells = [(cap, floor) for cap in caps for floor in floors]
    done_rows, done_keys = _load_done(grid_path)
    remaining = [(c, f) for (c, f) in all_cells if _cell_key(c, f) not in done_keys]
    baseline_path = _baseline_path(out_path, seed_idx, hold_days)
    consts_path = _constants_path(out_path, seed_idx, hold_days)
    consts = _load_constants(consts_path)
    m_base: dict | None = None
    if baseline_path.exists():
        import json  # noqa: PLC0415
        m_base = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Constant-exposure control arms needed = one per distinct cell mean_exposure.
    needed_consts = {_const_key(r["mean_exposure"]) for r in done_rows}
    consts_missing = needed_consts - set(consts)

    LOGGER.info("Resume | %d/%d cells done, baseline %s, %d/%d const arms cached",
                len(done_rows), len(all_cells), "cached" if m_base else "pending",
                len(needed_consts & set(consts)), len(needed_consts) or 0)

    # Everything cached → print from sidecars, skip the heavy materialize.
    if not remaining and m_base is not None and not consts_missing:
        LOGGER.info("Nothing to compute — emitting cached grid + controls.")
        _print_summary(done_rows, m_base, consts, seed_idx, hold_days)
        return pd.DataFrame(done_rows)

    # ── Load checkpoint + materialize once (only when work remains) ──────
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

    # ── Baseline ONCE (cached across restarts) ───────────────────────────
    if m_base is None:
        LOGGER.info("Baseline walk-forward (no brake)...")
        eq_base = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                          cutoff, cfg, p_bull_series=None, hold_days=hold_days)
        m_base = _metrics(eq_base, ic)
        import json  # noqa: PLC0415
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(m_base), encoding="utf-8")
    LOGGER.info("Baseline | Sharpe=%.3f  MaxDD=%.2f%%  Ret=%.2f%%",
                m_base["sharpe"], m_base["max_dd"], m_base["total_ret"])

    # ── Grid: refit GARCH-HMM per cap, braked WF per remaining cell ──────
    # Group remaining cells by cap so each cap's GARCH-HMM is fit once.
    rows: list[dict] = list(done_rows)
    caps_remaining = sorted({c for (c, _f) in remaining})
    for cap in caps_remaining:
        garch = train_garch_hmm(obs_train, n_states=n_states, seed=cfg.seed,
                                n_restarts=20, max_persistence=cap)
        persistence = garch.garch_params["alpha"] + garch.garch_params["beta"]
        for (c, floor) in [(c, f) for (c, f) in remaining if c == cap]:
            scaler = garch.exposure_scaler(obs, min_exposure=floor, max_exposure=1.0)
            oos_scaler = scaler[scaler.index >= cutoff_ts]
            LOGGER.info("Cell cap=%.2f floor=%.2f | persist=%.4f mean_exp=%.3f → braked WF...",
                        cap, floor, persistence, float(oos_scaler.mean()))
            eq = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                         cutoff, cfg, p_bull_series=scaler.rename("p_bull"),
                         hold_days=hold_days)
            m = _metrics(eq, ic)
            row = {
                "cap": cap, "floor": floor, "persistence": round(persistence, 4),
                "mean_exposure": round(float(oos_scaler.mean()), 3),
                "sharpe": m["sharpe"], "max_dd": m["max_dd"], "total_ret": m["total_ret"],
                "d_sharpe": round(m["sharpe"] - m_base["sharpe"], 3),
                "d_max_dd": round(m["max_dd"] - m_base["max_dd"], 2),
                "d_total_ret": round(m["total_ret"] - m_base["total_ret"], 2),
            }
            # Append IMMEDIATELY so a power-off only loses the in-flight cell.
            _append_row(grid_path, row, write_header=not grid_path.exists())
            rows.append(row)
            LOGGER.info("Cell saved (%d/%d total)", len(rows), len(all_cells))

    # ── Constant-exposure control arms (THE control vs the bear-OOS confound) ──
    # For each distinct mean-exposure the brake produced, run a FLAT-leverage book
    # at that same level. timing_alpha = braked_Sharpe − const_Sharpe isolates
    # whether the brake's TIMING adds anything over just holding less. Cached +
    # crash-durable like the rest.
    needed = {_const_key(r["mean_exposure"]) for r in rows}
    for ckey in sorted(needed - set(consts)):
        cval = float(ckey)
        LOGGER.info("Control arm: flat exposure=%.2f → WF...", cval)
        flat = pd.Series(cval, index=obs.index, name="p_bull")
        eqc = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                      cutoff, cfg, p_bull_series=flat, hold_days=hold_days)
        consts[ckey] = _metrics(eqc, ic)
        _save_constants(consts_path, consts)  # durable per-arm
        LOGGER.info("Control %.2f | Sharpe=%.3f", cval, consts[ckey]["sharpe"])

    _print_summary(rows, m_base, consts, seed_idx, hold_days)
    LOGGER.info("Wall-clock: %.1fs (%d new cells, %d controls, %d total cells)",
                time.perf_counter() - t0, len(remaining), len(needed), len(rows))
    return pd.DataFrame(rows)


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
