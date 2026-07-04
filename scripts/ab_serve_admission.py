"""ab_serve_admission.py — Serve-mirror admission A/B (Task A).

Runs the PRE-COMMITTED 7-config grid from
`process/general-plans/active/serve-admission-tranche-ab_PLAN_04-07-26.md`
against the SAME frozen T+20 checkpoint, then reports whether replacing the
tranche book's cross-sectional top-N admission with serve's absolute-gate
(`p_up >= admission_floor` → cap the pool at 6 → top-N) mechanics would have been
net-better on the OOS sample.

The grid is hardcoded IN this script (no CLI args) so there is no way to
accidentally run an unplanned 8th config (Guardrail 7 — DSR trial-count
discipline). All 7 configs share ONE per-seed inference cache: oracle scoring is
admission-mode-independent, so only the first config's first-per-seed run pays
the full daily-inference cost; configs 2-7 reuse the cache and re-run just the
(cheap) admission/allocation/execution path.

Reads the checkpoint READ-ONLY via `run_oos` — NEVER calls `_persist_bot_payload`
and NEVER writes to `models/saved/` (it does not go through `run_backtest.main`).

    python scripts/ab_serve_admission.py

Emits an ASCII comparison table to stdout AND writes a UTF-8 markdown report to
`process/general-plans/reports/serve-admission-ab-result_<run-date>.md`.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on path so `python scripts/ab_serve_admission.py` resolves `src`/root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    TRADING_DAYS,
    configure_logging,
    phase,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.macro_risk_hmm import build_regime_observation  # noqa: E402
from src.models.statistical_gates import deflated_sharpe, cscv_pbo  # noqa: E402
from src.models.tabular_ensemble import TabularEnsemble  # noqa: E402
from run_backtest import (  # noqa: E402
    CHECKPOINT_PATH,
    _load_checkpoint,
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)

LOGGER = logging.getLogger("quant.ab_admission")

REPORT_DIR = REPO_ROOT / "process" / "general-plans" / "reports"

# ── PRE-COMMITTED GRID (7 configs — DO NOT EXPAND, see Guardrail 7) ───────────
# Each config differs ONLY in admission fields (+ max_positions). Everything else
# (checkpoint, seeds, dataset, corporate actions, P(Bull), cost model) is shared.
GRID: list[dict] = [
    {"label": "1. Baseline (incumbent)",   "admission_mode": "cross_sectional",
     "admission_floor": None, "max_positions": 5, "admission_pool_cap": 6},
    {"label": "2. Serve-mirror @0.45/N5",  "admission_mode": "absolute_gate",
     "admission_floor": 0.45, "max_positions": 5, "admission_pool_cap": 6},
    {"label": "3. Serve-mirror @0.45/N3",  "admission_mode": "absolute_gate",
     "admission_floor": 0.45, "max_positions": 3, "admission_pool_cap": 6},
    {"label": "4. Floor sweep @0.40/N3",   "admission_mode": "absolute_gate",
     "admission_floor": 0.40, "max_positions": 3, "admission_pool_cap": 6},
    {"label": "5. Floor sweep @0.40/N5",   "admission_mode": "absolute_gate",
     "admission_floor": 0.40, "max_positions": 5, "admission_pool_cap": 6},
    {"label": "6. Floor sweep @0.35/N3",   "admission_mode": "absolute_gate",
     "admission_floor": 0.35, "max_positions": 3, "admission_pool_cap": 6},
    {"label": "7. Floor sweep @0.35/N5",   "admission_mode": "absolute_gate",
     "admission_floor": 0.35, "max_positions": 5, "admission_pool_cap": 6},
]

# Baseline keeps the GOLDEN cross_sectional signal_threshold (0.40); absolute_gate
# configs pass their floor as admission_floor and leave signal_threshold inert
# (the absolute_gate branch never reads it). We still set signal_threshold=0.40
# for parity with GOLDEN so the two paths only differ by the admission rule.
BASELINE_SIGNAL_THRESHOLD = 0.40


def _monthly_net_return(eq: pd.DataFrame) -> pd.Series:
    """Compounded NET RETURN per calendar month (generalizes run_backtest's
    monthly_net_sharpe to a return series so a June-pattern month is visible)."""
    s = eq.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["period"] = s["date"].dt.to_period("M")
    out: dict = {}
    for p, g in s.groupby("period"):
        r = g["daily_return"].to_numpy()
        out[str(p)] = float(np.prod(1.0 + r) - 1.0)
    return pd.Series(out).sort_index()


def _run_config(cfg_grid: dict, *, panel, tabular_features, trained,
                corporate_actions, cutoff, base_cfg, p_bull_series,
                seed_inference_caches) -> dict:
    """Run one grid config across all seeds; return the 4-seed aggregate."""
    per_seed: list[dict] = []
    monthly_cols: dict[str, pd.Series] = {}
    admission_mode = cfg_grid["admission_mode"]
    floor = cfg_grid["admission_floor"]
    # Fresh eval config per grid row so max_positions is per-config; dataset knobs
    # stay locked to the checkpoint (via _apply_eval_overrides).
    cfg = _apply_eval_overrides(base_cfg, {"max_positions": cfg_grid["max_positions"]})
    cfg.signal_threshold = BASELINE_SIGNAL_THRESHOLD

    for seed, ensemble in trained:
        try:
            eq = run_oos(
                panel, tabular_features, ensemble, corporate_actions,
                cutoff, cfg, p_bull_series=p_bull_series,
                inference_cache=seed_inference_caches[seed],
                mode="tranche", hold_days=30,
                admission_mode=admission_mode,
                admission_floor=(floor if floor is not None else 0.45),
                admission_pool_cap=cfg_grid["admission_pool_cap"],
            )
            m = equity_metrics(eq, cfg.initial_capital)
            zcd = int(eq.attrs.get("zero_candidate_days", 0))
            mean_gross = float(eq["gross_exposure"].mean()) if len(eq) else 0.0
            per_seed.append({"seed": seed, "eq": eq, "zero_candidate_days": zcd,
                             "mean_gross_exposure": mean_gross, **m})
            monthly_cols[f"seed_{seed}"] = _monthly_net_return(eq)
            LOGGER.info(
                "    %-22s seed=%d  NetPnL=%s  Sharpe=%+.2f  DD=%.2f%%  "
                "zeroCandDays=%d  meanGross=%.3f",
                cfg_grid["label"], seed, f"{m['net_pnl']:+,.0f}", m["net_sharpe"],
                m["max_drawdown"] * 100, zcd, mean_gross)
        except Exception as exc:  # noqa: BLE001 — a failed seed must not kill the grid
            LOGGER.warning("    %s seed=%d FAILED: %s", cfg_grid["label"], seed, exc)

    if not per_seed:
        LOGGER.warning("All seeds failed for config %s", cfg_grid["label"])
        return {"label": cfg_grid["label"], "n_seeds_ok": 0, "per_seed": [],
                "monthly_cols": {}}

    return {
        "label": cfg_grid["label"],
        "admission_mode": admission_mode,
        "admission_floor": floor,
        "max_positions": cfg_grid["max_positions"],
        "mean_net_pnl": float(np.mean([p["net_pnl"] for p in per_seed])),
        "mean_sharpe": float(np.mean([p["net_sharpe"] for p in per_seed])),
        "mean_dd": float(np.mean([p["max_drawdown"] for p in per_seed])),
        "mean_zero_candidate_days": float(np.mean([p["zero_candidate_days"] for p in per_seed])),
        "mean_gross_exposure": float(np.mean([p["mean_gross_exposure"] for p in per_seed])),
        "n_days": int(np.max([p["n_days"] for p in per_seed])),
        "n_seeds_ok": len(per_seed),
        "per_seed": per_seed,
        "monthly_cols": monthly_cols,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _build_ascii_table(results: list[dict], winner_label: str) -> str:
    """ASCII comparison table (cp1252-safe -- no non-ASCII glyphs)."""
    header = (f"{'Config':<24} {'NetPnL(VND)':>16} {'Sharpe':>8} {'MaxDD':>9} "
              f"{'ZeroCandDay':>12} {'MeanGross':>10} {'N':>3}")
    lines = [header, "-" * len(header)]
    for r in results:
        if r["n_seeds_ok"] == 0:
            lines.append(f"{r['label']:<24} {'FAILED (all seeds)':>16}")
            continue
        star = " *" if r["label"] == winner_label else "  "
        lines.append(
            f"{r['label']:<24} {r['mean_net_pnl']:>+16,.0f} "
            f"{r['mean_sharpe']:>+8.3f} {r['mean_dd'] * 100:>8.2f}% "
            f"{r['mean_zero_candidate_days']:>12.1f} "
            f"{r['mean_gross_exposure']:>10.3f} {r['n_seeds_ok']:>3d}{star}")
    lines.append("")
    lines.append("(* = winner by mean Net PnL; NetPnL/Sharpe/MaxDD/ZeroCandDay/"
                 "MeanGross are 4-seed means. N = seeds OK.)")
    return "\n".join(lines)


def _build_monthly_table(results: list[dict]) -> str:
    """Per-config mean monthly NET RETURN across seeds (rows=month, cols=config)."""
    cols: dict[str, pd.Series] = {}
    for r in results:
        if r["n_seeds_ok"] == 0:
            continue
        M = pd.concat(r["monthly_cols"], axis=1)
        cols[r["label"].split(".")[0].strip()] = M.mean(axis=1)  # short label (config #)
    if not cols:
        return "(no monthly data -- all configs failed)"
    table = pd.concat(cols, axis=1).sort_index()
    # Render as an ASCII grid (percent). Months are the index.
    header = f"{'Month':<9}" + "".join(f"{c:>9}" for c in table.columns)
    lines = [header, "-" * len(header)]
    for month, row in table.iterrows():
        cells = "".join(f"{v * 100:>8.2f}%" if pd.notna(v) else f"{'--':>9}"
                        for v in row.to_numpy())
        lines.append(f"{month:<9}{cells}")
    return "\n".join(lines)


def _compute_dsr_pbo(winner: dict, n_trials: int, cscv_s: int) -> tuple[dict, dict]:
    """DSR (n_trials multiplicity) + PBO on the WINNER's per-seed curves — same
    machinery as run_backtest's teardown. Winner's best-Sharpe seed drives DSR."""
    best_seed = max(winner["per_seed"], key=lambda p: p["net_sharpe"])
    daily_r = best_seed["eq"]["daily_return"].to_numpy()
    dsr = deflated_sharpe(daily_r, n_trials=n_trials, annualisation=TRADING_DAYS)
    monthly = {k: _monthly_from_eq(p["eq"]) for k, p in
               [(f"seed_{p['seed']}", p) for p in winner["per_seed"]]}
    if len(monthly) >= 2:
        M = pd.concat(monthly, axis=1).fillna(0.0).sort_index()
        S = min(cscv_s, (len(M) // 2) * 2)
        pbo = cscv_pbo(M.to_numpy(), S=max(2, S)) if len(M) >= 2 else {
            "pbo": float("nan"), "valid": False, "warning": "too few periods"}
    else:
        pbo = {"pbo": float("nan"), "valid": False,
               "warning": "single seed — need >=2 for PBO"}
    return dsr, pbo


def _monthly_from_eq(eq: pd.DataFrame) -> pd.Series:
    """Monthly net Sharpe (the CSCV performance metric run_backtest uses)."""
    s = eq.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["period"] = s["date"].dt.to_period("M")
    out: dict = {}
    for p, g in s.groupby("period"):
        r = g["daily_return"].to_numpy()
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        out[p] = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    return pd.Series(out).sort_index()


def main() -> None:
    configure_logging()
    t_start = time.perf_counter()

    with phase("Load training checkpoint (READ-ONLY)"):
        ckpt = _load_checkpoint(CHECKPOINT_PATH)
        train_cfg: RunConfig = ckpt["train_cfg"]
        tabular_features: list[str] = list(ckpt["tabular_features"])
        cutoff: date = ckpt["cutoff"]
        trained: list[tuple[int, TabularEnsemble]] = list(ckpt["ensembles"])
        macro_hmm = ckpt.get("macro_hmm")
        seeds = [s for s, _ in trained]
        LOGGER.info("Checkpoint | seeds=%s  features=%d  cutoff=%s  HMM=%s",
                    seeds, len(tabular_features), cutoff, macro_hmm is not None)

    base_cfg = _apply_eval_overrides(train_cfg, {})

    with phase("Re-materialize dataset + subset features"):
        ds = materialize_dataset(base_cfg)
        ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
    corporate_actions = load_corporate_actions(base_cfg)

    p_bull_series = None
    if macro_hmm is not None:
        try:
            obs = build_regime_observation(
                ds.panel, use_macro=base_cfg.use_macro_in_hmm,
                macro_parquet=base_cfg.macro_parquet)
            p_bull_series = macro_hmm.p_bull_series(obs, filtered=True)
            oos_pb = p_bull_series[p_bull_series.index >= pd.Timestamp(cutoff)]
            LOGGER.info("HMM P(Bull) | OOS mean=%.3f  OOS min=%.3f",
                        float(oos_pb.mean()), float(oos_pb.min()))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("P(Bull) recompute failed (%s) — full exposure.", exc)
            p_bull_series = None

    # ONE cache per seed, SHARED across all 7 configs. Oracle scoring is
    # admission-mode-independent, so only the first config pays the daily-oracle
    # cost per seed; configs 2-7 reuse it.
    seed_inference_caches: dict = {seed: {} for seed, _ in trained}

    results: list[dict] = []
    for cfg_grid in GRID:
        with phase(f"Grid config | {cfg_grid['label']}"):
            results.append(_run_config(
                cfg_grid, panel=ds.panel, tabular_features=tabular_features,
                trained=trained, corporate_actions=corporate_actions, cutoff=cutoff,
                base_cfg=base_cfg, p_bull_series=p_bull_series,
                seed_inference_caches=seed_inference_caches))

    ok_results = [r for r in results if r["n_seeds_ok"] > 0]
    if not ok_results:
        LOGGER.error("Every config failed — no report to write.")
        return

    winner = max(ok_results, key=lambda r: r["mean_net_pnl"])
    n_trials = 7 * max(1, len(seeds))              # 28 — DSR multiplicity convention
    dsr, pbo = _compute_dsr_pbo(winner, n_trials, base_cfg.cscv_S)

    ascii_table = _build_ascii_table(results, winner["label"])
    monthly_table = _build_monthly_table(ok_results)
    elapsed = time.perf_counter() - t_start

    # ── stdout (ASCII only — console is cp1252) ──────────────────────────────
    bar = "=" * 78
    print(f"\n{bar}\n SERVE-MIRROR ADMISSION A/B -- 7-config grid (T+20, {len(seeds)} seeds)\n{bar}")
    print(ascii_table)
    print(f"\nWinner by mean Net PnL: {winner['label']}")
    dsr_str = (f"p_dsr={dsr['p_dsr']:.4f} ({'PASS>=0.95' if dsr.get('valid') and dsr['p_dsr'] >= 0.95 else 'FAIL<0.95'})"
               if dsr.get("valid") else f"N/A ({dsr.get('warning', 'invalid')})")
    pbo_str = (f"pbo={pbo['pbo']:.1%} ({'PASS<=10%' if pbo.get('valid') and pbo['pbo'] <= 0.10 else 'FAIL>10%'})"
               if pbo.get("valid") else f"N/A ({pbo.get('warning', 'invalid')})")
    print(f"  DSR (n_trials={n_trials}): {dsr_str}")
    print(f"  PBO (CSCV): {pbo_str}")
    print(f"\nMonthly NET RETURN by config (4-seed mean):\n{monthly_table}")
    print(f"\nWall-clock: {elapsed:.1f}s\n{bar}\n")

    # ── markdown report (UTF-8) ──────────────────────────────────────────────
    run_date = datetime.now().strftime("%d-%m-%y")
    report_path = REPORT_DIR / f"serve-admission-ab-result_{run_date}.md"
    _write_report(report_path, results, winner, dsr, pbo, n_trials, seeds,
                  cutoff, ascii_table, monthly_table, elapsed)
    LOGGER.info("Report written: %s", report_path)


def _write_report(path: Path, results, winner, dsr, pbo, n_trials, seeds,
                  cutoff, ascii_table, monthly_table, elapsed) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dsr_line = (f"SR={dsr['sr_annualised']:+.3f}, SR0={dsr['sr0_annualised']:+.3f}, "
                f"p_dsr={dsr['p_dsr']:.4f} "
                f"({'PASS >=0.95' if dsr['p_dsr'] >= 0.95 else 'FAIL <0.95'})"
                if dsr.get("valid") else f"N/A ({dsr.get('warning', 'invalid')})")
    pbo_line = (f"{pbo['pbo']:.1%} "
                f"({'PASS <=10%' if pbo['pbo'] <= 0.10 else 'FAIL >10%'})"
                if pbo.get("valid") else f"N/A ({pbo.get('warning', 'invalid')})")

    md = [
        "# Serve-Mirror Admission A/B — Result",
        "",
        f"- **Run date:** {datetime.now().strftime('%d-%m-%y %H:%M')}",
        f"- **Plan:** `process/general-plans/active/serve-admission-tranche-ab_PLAN_04-07-26.md` (Task A)",
        f"- **Checkpoint:** `models/saved/v3_training_checkpoint.joblib` (T+20, seeds={seeds})",
        f"- **OOS cutoff:** {cutoff}",
        f"- **Command:** `python scripts/ab_serve_admission.py`",
        f"- **DSR convention:** n_trials={n_trials} (7 configs x {len(seeds)} seeds)",
        "",
        "## Comparison table (4-seed means)",
        "",
        "| # | Config | admission_mode | floor | top-N | Mean Net PnL (VND) | Mean Sharpe | Mean MaxDD | Mean Zero-Cand Days | Mean Gross Exp | Seeds OK |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        if r["n_seeds_ok"] == 0:
            md.append(f"| {i} | {r['label']} | — | — | — | FAILED | — | — | — | — | 0 |")
            continue
        win = " **<= WINNER**" if r["label"] == winner["label"] else ""
        floor = "n/a" if r["admission_floor"] is None else f"{r['admission_floor']:.2f}"
        md.append(
            f"| {i} | {r['label']}{win} | {r['admission_mode']} | {floor} | "
            f"{r['max_positions']} | {r['mean_net_pnl']:+,.0f} | {r['mean_sharpe']:+.3f} | "
            f"{r['mean_dd'] * 100:.2f}% | {r['mean_zero_candidate_days']:.1f} | "
            f"{r['mean_gross_exposure']:.3f} | {r['n_seeds_ok']} |")

    md += [
        "",
        f"**Winner by mean Net PnL:** {winner['label']}",
        "",
        f"- **DSR (winner, n_trials={n_trials}):** {dsr_line}",
        f"- **PBO (CSCV, winner per-seed):** {pbo_line}",
        "",
        "## Monthly NET RETURN by config (4-seed mean)",
        "",
        "Isolates narrow-leadership / negative-breadth months in the OOS window.",
        "",
        "```",
        monthly_table,
        "```",
        "",
        "## Raw stdout comparison table",
        "",
        "```",
        ascii_table,
        "```",
        "",
        "## Decision-rule read (Task B gate — see plan)",
        "",
        _decision_read(results, winner),
        "",
        f"_Wall-clock: {elapsed:.1f}s._",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def _decision_read(results: list[dict], winner: dict) -> str:
    """Apply the plan's Task-B decision rule mechanically to the numbers."""
    baseline = next((r for r in results
                     if r.get("admission_mode") == "cross_sectional"
                     and r["n_seeds_ok"] > 0), None)
    if baseline is None:
        return "Baseline (cross_sectional) config failed — decision rule cannot be evaluated."
    gates = [r for r in results if r.get("admission_mode") == "absolute_gate"
             and r["n_seeds_ok"] > 0]
    if not gates:
        return "All absolute_gate configs failed — no comparison possible."
    best_gate = max(gates, key=lambda r: r["mean_sharpe"])
    sharpe_delta = best_gate["mean_sharpe"] - baseline["mean_sharpe"]
    dd_better = best_gate["mean_dd"] > baseline["mean_dd"]  # less negative = better
    max_zcd = max(r["mean_zero_candidate_days"] for r in gates)
    n_days = max(r["n_days"] for r in results if r["n_seeds_ok"] > 0)
    zcd_frac = max_zcd / n_days if n_days else 0.0

    lines = [
        f"- Baseline (cross_sectional) mean Sharpe: {baseline['mean_sharpe']:+.3f}, "
        f"mean MaxDD: {baseline['mean_dd'] * 100:.2f}%",
        f"- Best absolute_gate config: {best_gate['label']} "
        f"(Sharpe {best_gate['mean_sharpe']:+.3f}, MaxDD {best_gate['mean_dd'] * 100:.2f}%)",
        f"- Best-gate Sharpe delta vs baseline: {sharpe_delta:+.3f}",
        f"- Does the gate earn its lost return via lower risk? "
        f"{'YES (MaxDD better)' if dd_better else 'NO (MaxDD not better)'}",
        f"- Max zero-candidate-day fraction (any gate config): {zcd_frac:.1%} of OOS days",
        "",
    ]
    # Plan's Task-B "proceed" condition: absolute_gate WORSE on BOTH
    # (Sharpe delta < -0.05 AND MaxDD not meaningfully better) AND materially
    # nonzero zero-candidate days (> ~5% of OOS days).
    worse_sharpe = sharpe_delta < -0.05
    proceed = worse_sharpe and (not dd_better) and (zcd_frac > 0.05)
    if proceed:
        lines.append(
            "**Read:** Evidence SUPPORTS a Task-B follow-up plan — the absolute "
            "gate costs risk-adjusted return (Sharpe worse by >0.05) without "
            "earning it back via lower drawdown, while blocking a material "
            "fraction of OOS trading days. A floor+top-N serve-admission change "
            "(behind a default-OFF kill-switch) is warranted, per the plan's "
            "Task-B decision rule. NOTE: Task B is OUT of scope for this plan and "
            "requires its own RESEARCH + PLAN pass.")
    else:
        lines.append(
            "**Read:** Evidence does NOT clearly support a Task-B serve-path "
            "change. The absolute gate is close to (or better than) baseline on "
            "risk-adjusted terms and/or does not block a material fraction of "
            "days. Serve's current defensiveness — while it happened to block "
            "June's SSB winner — is not measurably costly on average across this "
            "OOS sample. File the June episode as an acceptable false-defensive "
            "instance; no serve-path change is warranted from this evidence alone.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
