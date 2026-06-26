"""
scripts/benchmark_defense_layers.py — head-to-head of every defensive overlay.

Answers "which defense, alone or combined, actually wins?" by running each layer
through the SAME walk-forward (seed-0 ensemble) and reporting Sharpe / MaxDD /
Return / mean-exposure, plus a flat-leverage TIMING control for every soft arm.

Arms
────
  baseline          no defense
  regime_policy     rule-based 8-regime sizing (current serve default; engine-side)
  macro_hmm         2-state macro HMM P(Bull)  (market-wide weight scaler)
  garch_hmm         GARCH(1,1)+HMM exposure_scaler clip(P(Bull), floor, 1.0)
  macro+garch_min   element-wise min(macro_pbull, garch_scaler)  (most-conservative)
  regime+garch      regime_policy AND garch scaler
  all_min           regime_policy AND min(macro, garch)

Mechanism note: the engine has TWO injection points — `use_regime_sizing` (bool,
per-ticker 8-regime) and `p_bull_series` (one market-wide daily weight scaler).
Two soft scalers can't both ride `p_bull_series`, so combos use min() into one.

TIMING control: for each arm carrying a soft scaler, a flat-leverage book at that
arm's realized mean exposure isolates regime timing from plain de-leverage
(constant leverage is Sharpe-neutral, so a positive gap == real timing).

No lookahead: GARCH-HMM + macro HMM trained strictly on obs < cutoff; all scalers
filtered. This is pure EVALUATION — pick ONE winner (or the min-combine) to deploy
as a FIXED default. It is NOT a dynamic selector (that would be lookahead/overfit).

Resumable: every arm's metrics cached to a sidecar JSON the moment it finishes,
so a power-off only loses the in-flight arm.

Run
───
    python scripts/benchmark_defense_layers.py
    python scripts/benchmark_defense_layers.py --floor 0.1 --seed-idx 0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    configure_logging,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.garch_hmm_regime import train_garch_hmm  # noqa: E402
from src.models.macro_risk_hmm import (  # noqa: E402
    build_market_proxy_returns,
    train_macro_risk_hmm,
)

from scripts.validate_garch_hmm_brake import (  # noqa: E402
    _run_wf,
    _build_macro_obs,
    _metrics,
    _CHECKPOINT_5D,
    _MACRO_PARQUET,
)

LOGGER = logging.getLogger("quant.benchmark_defense")

_OUT = Path("process/features/macro-integration/reports/defense_benchmark.json")


def _cache_path(out_path: Path, seed_idx: int, hold_days: int) -> Path:
    return out_path.with_name(f"{out_path.stem}_s{seed_idx}_h{hold_days}.json")


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def main(
    *,
    checkpoint_path: Path,
    macro_parquet: Path,
    n_states: int,
    floor: float,
    max_persistence: float,
    hold_days: int,
    seed_idx: int,
    out_path: Path,
) -> dict:
    configure_logging()
    t0 = time.perf_counter()
    cache_path = _cache_path(out_path, seed_idx, hold_days)
    cache = _load_cache(cache_path)

    LOGGER.info("=" * 78)
    LOGGER.info(" DEFENSE BENCHMARK | floor=%.2f cap=%.2f states=%d seed_idx=%d hold=%dd",
                floor, max_persistence, n_states, seed_idx, hold_days)
    LOGGER.info(" cached arms: %s", sorted(cache) or "none")
    LOGGER.info("=" * 78)

    # Arms that need a walk-forward; we add their flat-leverage controls below.
    soft_arms = ["macro_hmm", "garch_hmm", "macro+garch_min", "regime+garch", "all_min"]
    arm_names = ["baseline", "regime_policy", *soft_arms]

    # ── Load checkpoint + materialize once (only if any arm/control missing) ──
    need_run = any(a not in cache for a in arm_names) or "__controls_done" not in cache
    if need_run:
        ckpt = joblib.load(checkpoint_path)
        cfg: RunConfig = ckpt["train_cfg"]
        tabular_features: list[str] = ckpt["tabular_features"]
        cutoff = ckpt["cutoff"]
        ensembles = ckpt["ensembles"]
        if seed_idx >= len(ensembles):
            raise ValueError(f"seed_idx={seed_idx} but only {len(ensembles)} seeds")
        seed, ensemble = ensembles[seed_idx]
        LOGGER.info("Checkpoint seed=%d cutoff=%s features=%d", seed, cutoff, len(tabular_features))

        ds = materialize_dataset(cfg)
        ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
        corporate_actions = load_corporate_actions(cfg)
        obs = _build_macro_obs(ds.panel, macro_parquet)
        cutoff_ts = pd.Timestamp(cutoff)
        ic = cfg.initial_capital

        # ── Train the two macro overlays on the in-sample split ──────────
        market_ret = obs["market_ret"]
        mr_train = market_ret[market_ret.index < cutoff_ts]
        macro_hmm = train_macro_risk_hmm(mr_train, n_states=2, seed=cfg.seed)
        macro_pbull = macro_hmm.p_bull_series(market_ret, filtered=True).rename("p_bull")

        garch = train_garch_hmm(obs[obs.index < cutoff_ts], n_states=n_states,
                                seed=cfg.seed, n_restarts=20, max_persistence=max_persistence)
        garch_scaler = garch.exposure_scaler(obs, min_exposure=floor, max_exposure=1.0)

        # Align both scalers on a common index, then build the combine.
        common = macro_pbull.dropna().index.intersection(garch_scaler.dropna().index)
        macro_pbull = macro_pbull.reindex(garch_scaler.index).ffill()
        macro_garch_min = pd.concat([macro_pbull, garch_scaler], axis=1).min(axis=1).rename("p_bull")

        # Arm spec: name → (use_regime_sizing, p_bull_series_or_None)
        spec = {
            "baseline":        (False, None),
            "regime_policy":   (True,  None),
            "macro_hmm":       (False, macro_pbull),
            "garch_hmm":       (False, garch_scaler.rename("p_bull")),
            "macro+garch_min": (False, macro_garch_min),
            "regime+garch":    (True,  garch_scaler.rename("p_bull")),
            "all_min":         (True,  macro_garch_min),
        }

        def _oos_mean_exp(series: "pd.Series | None") -> float:
            if series is None:
                return float("nan")
            s = series[series.index >= cutoff_ts].dropna()
            return float(s.mean()) if len(s) else float("nan")

        # ── Run each arm (skip cached) ───────────────────────────────────
        for name in arm_names:
            if name in cache:
                LOGGER.info("[%s] cached — skip", name)
                continue
            use_rs, pbull = spec[name]
            LOGGER.info("[%s] WF (regime_sizing=%s, scaler=%s)...",
                        name, use_rs, "yes" if pbull is not None else "no")
            eq = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                         cutoff, cfg, p_bull_series=pbull, hold_days=hold_days,
                         use_regime_sizing=use_rs)
            m = _metrics(eq, ic)
            m["mean_exposure"] = round(_oos_mean_exp(pbull), 3)
            cache[name] = m
            _save_cache(cache_path, cache)
            LOGGER.info("[%s] Sharpe=%.3f MaxDD=%.2f%% Ret=%.2f%% mean_exp=%s",
                        name, m["sharpe"], m["max_dd"], m["total_ret"], m["mean_exposure"])

        # ── Flat-leverage TIMING controls for each soft arm's mean exposure ──
        controls = cache.get("__controls", {})
        for name in soft_arms:
            me = cache[name].get("mean_exposure")
            if me is None or not np.isfinite(me):
                continue
            ckey = f"{round(float(me), 2):.2f}"
            if ckey in controls:
                continue
            LOGGER.info("[control] flat exposure=%.2f → WF...", float(ckey))
            flat = pd.Series(float(ckey), index=obs.index, name="p_bull")
            eqc = _run_wf(ds.panel, tabular_features, ensemble, corporate_actions,
                          cutoff, cfg, p_bull_series=flat, hold_days=hold_days)
            controls[ckey] = _metrics(eqc, ic)
            cache["__controls"] = controls
            _save_cache(cache_path, cache)
            LOGGER.info("[control %.2f] Sharpe=%.3f", float(ckey), controls[ckey]["sharpe"])

        cache["__controls_done"] = True
        _save_cache(cache_path, cache)

    _print_benchmark(cache, seed_idx, hold_days)
    LOGGER.info("Wall-clock: %.1fs", time.perf_counter() - t0)
    return cache


def _print_benchmark(cache: dict, seed_idx: int, hold_days: int) -> None:
    base = cache.get("baseline", {})
    controls = cache.get("__controls", {})
    arm_names = ["baseline", "regime_policy", "macro_hmm", "garch_hmm",
                 "macro+garch_min", "regime+garch", "all_min"]

    sep = "─" * 92
    print(f"\n{sep}")
    print(f"  DEFENSE LAYER BENCHMARK | seed_idx={seed_idx}  hold={hold_days}d")
    print(sep)
    print(f"  {'arm':<18}{'Sharpe':>8}{'ΔSh':>8}{'MaxDD%':>9}{'Ret%':>9}"
          f"{'mean_exp':>10}{'TIMING_α':>10}")
    print(sep)
    b_sh = base.get("sharpe", float("nan"))
    best = None
    for name in arm_names:
        m = cache.get(name)
        if not m:
            print(f"  {name:<18}{'pending':>8}")
            continue
        d_sh = m["sharpe"] - b_sh
        me = m.get("mean_exposure")
        ta = float("nan")
        if me is not None and np.isfinite(me):
            ck = f"{round(float(me), 2):.2f}"
            c = controls.get(ck)
            if c:
                ta = m["sharpe"] - c["sharpe"]
        me_str = f"{me:>10.3f}" if (me is not None and np.isfinite(me)) else f"{'—':>10}"
        ta_str = f"{ta:>+10.3f}" if np.isfinite(ta) else f"{'—':>10}"
        print(f"  {name:<18}{m['sharpe']:>8.3f}{d_sh:>+8.3f}{m['max_dd']:>9.2f}"
              f"{m['total_ret']:>9.2f}{me_str}{ta_str}")
        if best is None or m["sharpe"] > cache[best]["sharpe"]:
            best = name
    print(sep)
    if best:
        print(f"  BEST Sharpe: {best} ({cache[best]['sharpe']:.3f})")
        print("  NOTE: pick ONE winner (or min-combine) as a FIXED default — "
              "do NOT auto-select per-period (lookahead).")
    print(f"{sep}\n")


def _cli() -> dict:
    p = argparse.ArgumentParser(description="Benchmark defensive overlays head-to-head.")
    p.add_argument("--checkpoint", type=Path, default=_CHECKPOINT_5D)
    p.add_argument("--macro-parquet", type=Path, default=_MACRO_PARQUET)
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--floor", type=float, default=0.2, help="GARCH-HMM exposure floor")
    p.add_argument("--max-persistence", type=float, default=0.96)
    p.add_argument("--hold-days", type=int, default=5)
    p.add_argument("--seed-idx", type=int, default=0)
    p.add_argument("--out", type=Path, default=_OUT)
    a = p.parse_args()
    return {
        "checkpoint_path": a.checkpoint,
        "macro_parquet": a.macro_parquet,
        "n_states": a.n_states,
        "floor": a.floor,
        "max_persistence": a.max_persistence,
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
