"""
scripts/validate_garch_hmm_brake.py — A/B backtest of the GARCH-HMM exposure brake.

Loads the existing T+5 training checkpoint, runs the walk-forward engine TWICE
on the OOS period — once without any regime brake (baseline), once with the
GARCH-HMM exposure_brake gating signals to zero — and prints a side-by-side
comparison of equity curves, Sharpe, MaxDD, and total return.

This is a PURE COMPARISON script. It does not retrain anything. It reuses the
frozen ensemble oracle from the checkpoint so the only variable is the brake.

Pipeline
────────
    1. Load T+5 checkpoint (ensembles, features, cutoff)
    2. Build macro observation matrix (market_ret + sp500/dxy/usdvnd)
    3. Train GARCH-HMM on the IN-SAMPLE macro obs (< cutoff)
    4. Compute filtered P(Bull) + binary exposure_brake over the FULL series
    5. Run walk-forward OOS: baseline (no brake) vs braked (zeroed signals)
    6. Print metrics table + save equity curve comparison plot

Run
───
    python scripts/validate_garch_hmm_brake.py
    python scripts/validate_garch_hmm_brake.py --n-states 4 --threshold 0.4
    python scripts/validate_garch_hmm_brake.py --horizon 20 --hold-days 30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

# ── Project imports ──────────────────────────────────────────────────────
from src.backtest.pipeline import (
    RunConfig,
    TRADING_DAYS,
    configure_logging,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.tabular_ensemble import make_ensemble_oracle
from src.models.macro_risk_hmm import build_market_proxy_returns
from src.models.garch_hmm_regime import train_garch_hmm, GarchHmmRegime
from src.backtest.walk_forward import WalkForwardEngine, WalkForwardConfig
from src.portfolio.construction import PortfolioConstraints
from src.execution.vn_cost_model import ExecutionConfig

LOGGER = logging.getLogger("quant.validate_garch_hmm")

_CHECKPOINT_5D = Path("models/saved/v3_training_checkpoint.joblib")
_MACRO_PARQUET = Path("data/macro_daily.parquet")


# ── Equity metrics ───────────────────────────────────────────────────────

def _metrics(eq: pd.DataFrame, initial_capital: float) -> dict:
    nav = eq["nav"].to_numpy()
    if len(nav) == 0:
        return {"sharpe": 0.0, "max_dd": 0.0, "total_ret": 0.0, "final_nav": initial_capital}
    r = eq["daily_return"].to_numpy()
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    dd = float((nav / np.maximum.accumulate(nav) - 1.0).min())
    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(dd * 100, 2),
        "total_ret": round((nav[-1] / initial_capital - 1.0) * 100, 2),
        "final_nav": round(float(nav[-1]), 0),
        "n_days": len(eq),
    }


# ── Walk-forward runner ─────────────────────────────────────────────────

def _run_wf(
    panel: pl.DataFrame,
    tabular_features: list[str],
    ensemble,
    corporate_actions: list,
    cutoff: date,
    cfg: RunConfig,
    *,
    p_bull_series: pd.Series | None = None,
    hold_days: int = 5,
    use_regime_sizing: bool = False,
) -> pd.DataFrame:
    """Single walk-forward pass returning the OOS equity curve."""
    oracle = make_ensemble_oracle(ensemble)
    buffer = 80
    all_dates = sorted(panel["date"].unique().to_list())
    cutoff_idx = next((i for i, d in enumerate(all_dates) if d >= cutoff), 0)
    buf_start = all_dates[max(0, cutoff_idx - buffer)]
    sub = panel.filter(pl.col("date") >= buf_start)

    wf_cfg = WalkForwardConfig(
        seq_len=1,
        feature_cols=tabular_features,
        initial_capital=cfg.initial_capital,
        max_positions=cfg.max_positions,
        rebalance_frequency=cfg.rebalance_frequency,
        signal_threshold=cfg.signal_threshold,
        cov_lookback=60,
        kelly_fraction=cfg.kelly_fraction,
        risk_aversion=cfg.risk_aversion,
        liquid_top_n=cfg.liquid_top_n,
        start_trading_date=cutoff,
        rebalance_mode="tranche",
        tranche_hold_days=hold_days,
        use_regime_sizing=use_regime_sizing,
        constraints=PortfolioConstraints(
            max_weight=cfg.max_weight, long_only=True,
            target_leverage=0.95, target_vol=cfg.target_vol),
        exec_config=ExecutionConfig(),
    )
    eng = WalkForwardEngine(wf_cfg, oracle)
    result = eng.run(sub, corporate_actions=corporate_actions, p_bull_series=p_bull_series)
    eq = result.equity_curve
    eq = eq[pd.to_datetime(eq["date"]).dt.date >= cutoff].reset_index(drop=True)
    return eq


# ── Macro observation builder ───────────────────────────────────────────

def _build_macro_obs(panel: pl.DataFrame, macro_parquet: Path) -> pd.DataFrame:
    """Join market-breadth proxy with macro returns → 4-col obs DataFrame."""
    market_ret = build_market_proxy_returns(panel)

    if not macro_parquet.exists():
        raise FileNotFoundError(
            f"{macro_parquet} not found. Run 'python main.py --task crawl_macro' first.")

    macro = pd.read_parquet(macro_parquet)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    obs = pd.DataFrame({"market_ret": market_ret})
    for c in ("sp500_ret", "dxy_ret", "usdvnd_ret"):
        obs[c] = macro[c].reindex(obs.index).ffill(limit=3)
    return obs.dropna()


# ── Main ─────────────────────────────────────────────────────────────────

def main(
    *,
    checkpoint_path: Path = _CHECKPOINT_5D,
    macro_parquet: Path = _MACRO_PARQUET,
    n_states: int = 3,
    threshold: float = 0.5,
    min_exposure: float = 0.2,
    max_exposure: float = 1.0,
    hold_days: int = 5,
    use_regime_sizing: bool = False,
    seed_idx: int = 0,
) -> dict:
    configure_logging()
    t0 = time.perf_counter()

    LOGGER.info("=" * 72)
    LOGGER.info(" GARCH-HMM LINEAR BRAKE VALIDATION | states=%d  exposure=[%.2f, %.2f]  hold=%dd",
                n_states, min_exposure, max_exposure, hold_days)
    LOGGER.info("=" * 72)

    # ── 1. Load checkpoint ───────────────────────────────────────────────
    ckpt = joblib.load(checkpoint_path)
    cfg: RunConfig = ckpt["train_cfg"]
    tabular_features: list[str] = ckpt["tabular_features"]
    cutoff: date = ckpt["cutoff"]
    ensembles = ckpt["ensembles"]

    if seed_idx >= len(ensembles):
        raise ValueError(f"seed_idx={seed_idx} but only {len(ensembles)} seeds in checkpoint")
    seed, ensemble = ensembles[seed_idx]
    LOGGER.info("Checkpoint: %s | seed=%d  cutoff=%s  features=%d",
                checkpoint_path.name, seed, cutoff, len(tabular_features))

    # ── 2. Materialize dataset ───────────────────────────────────────────
    ds = materialize_dataset(cfg)
    ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
    corporate_actions = load_corporate_actions(cfg)

    # ── 3. Build macro obs + train GARCH-HMM on in-sample ───────────────
    obs = _build_macro_obs(ds.panel, macro_parquet)
    cutoff_ts = pd.Timestamp(cutoff)
    obs_train = obs[obs.index < cutoff_ts]
    LOGGER.info("Macro obs: %d total, %d train (< %s)", len(obs), len(obs_train), cutoff)

    garch_hmm = train_garch_hmm(obs_train, n_states=n_states, seed=cfg.seed, n_restarts=12)

    # ── 4. Compute LINEAR exposure scaler over full series ───────────────
    # Continuous braking: exposure = clip(P(Bull), min_exp, max_exp). Replaces
    # the binary hard cash-out (which stuck ~60% of days at 100% cash and tanked
    # the Sharpe) with smooth downsizing toward a residual floor.
    scaler = garch_hmm.exposure_scaler(
        obs, min_exposure=min_exposure, max_exposure=max_exposure, filtered=True)
    oos_scaler = scaler[scaler.index >= cutoff_ts]
    LOGGER.info(
        "OOS exposure scaler: mean=%.3f  min=%.3f  max=%.3f  (floor=%.2f)",
        float(oos_scaler.mean()), float(oos_scaler.min()),
        float(oos_scaler.max()), min_exposure)
    # The engine multiplies daily target weights by this series. None = baseline.
    p_bull_scaled = scaler.rename("p_bull")

    # ── 5A. Baseline run (no brake) ──────────────────────────────────────
    LOGGER.info("Running BASELINE (no brake)...")
    eq_base = _run_wf(
        ds.panel, tabular_features, ensemble, corporate_actions, cutoff, cfg,
        p_bull_series=None, hold_days=hold_days, use_regime_sizing=use_regime_sizing)

    # ── 5B. Linear-braked run ────────────────────────────────────────────
    LOGGER.info("Running GARCH-HMM LINEAR BRAKE (exposure=[%.2f, %.2f])...",
                min_exposure, max_exposure)
    eq_brake = _run_wf(
        ds.panel, tabular_features, ensemble, corporate_actions, cutoff, cfg,
        p_bull_series=p_bull_scaled, hold_days=hold_days, use_regime_sizing=use_regime_sizing)

    # ── 6. Compare ───────────────────────────────────────────────────────
    initial_capital = cfg.initial_capital
    m_base = _metrics(eq_base, initial_capital)
    m_brake = _metrics(eq_brake, initial_capital)

    header = f"{'Metric':<20} {'Baseline':>12} {'GARCH-HMM':>12} {'Delta':>12}"
    sep = "─" * 58

    print(f"\n{sep}")
    print(f"  GARCH-HMM LINEAR BRAKE A/B  |  states={n_states}  "
          f"exposure=[{min_exposure}, {max_exposure}]  hold={hold_days}d")
    print(sep)
    print(header)
    print(sep)
    for key in ("sharpe", "max_dd", "total_ret"):
        vb, vk = m_base[key], m_brake[key]
        delta = vk - vb
        unit = "%" if key in ("max_dd", "total_ret") else ""
        sign = "+" if delta >= 0 else ""
        label = {"sharpe": "Sharpe", "max_dd": "Max DD (%)", "total_ret": "Total Return (%)"}[key]
        print(f"  {label:<18} {vb:>11.2f}{unit} {vk:>11.2f}{unit} {sign}{delta:>10.2f}{unit}")
    print(sep)

    # Brake-specific stats
    print(f"  {'OOS Days':<18} {m_base['n_days']:>12d} {m_brake['n_days']:>12d}")
    print(f"  {'Mean exposure':<18} {'':>12} {oos_scaler.mean():>11.2f}x")
    print(f"  {'Min exposure':<18} {'':>12} {oos_scaler.min():>11.2f}x")
    print(f"  {'Bull state':<18} {'':>12} {garch_hmm.bull_state:>12d}")
    persistence = garch_hmm.garch_params["alpha"] + garch_hmm.garch_params["beta"]
    print(f"  {'GARCH persist.':<18} {'':>12} {persistence:>11.4f}")
    print(sep)

    # Verdict
    better_sharpe = m_brake["sharpe"] > m_base["sharpe"]
    better_dd = m_brake["max_dd"] > m_base["max_dd"]  # less negative = better
    if better_sharpe and better_dd:
        verdict = "KEEP — improves both Sharpe and MaxDD"
    elif better_sharpe:
        verdict = "MIXED — better Sharpe, worse MaxDD"
    elif better_dd:
        verdict = "MIXED — worse Sharpe, better MaxDD"
    else:
        verdict = "KILL — degrades both Sharpe and MaxDD"
    print(f"  VERDICT: {verdict}")
    print(f"{sep}\n")

    # ── Save equity curves for plotting ──────────────────────────────────
    out_dir = Path("process/features/macro-integration/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_base["variant"] = "baseline"
    eq_brake["variant"] = "garch_hmm_brake"
    combined = pd.concat([eq_base, eq_brake], ignore_index=True)
    csv_path = out_dir / "garch_hmm_brake_equity_curves.csv"
    combined.to_csv(csv_path, index=False)
    LOGGER.info("Equity curves saved → %s", csv_path)

    wall = time.perf_counter() - t0
    LOGGER.info("Wall-clock: %.1fs", wall)

    return {
        "baseline": m_base,
        "garch_hmm_brake": m_brake,
        "verdict": verdict,
        "garch_params": garch_hmm.garch_params,
        "n_states": n_states,
        "min_exposure": min_exposure,
        "max_exposure": max_exposure,
        "mean_oos_exposure": float(oos_scaler.mean()),
    }


def _cli() -> dict:
    p = argparse.ArgumentParser(
        description="A/B backtest: T+5 signals with vs without GARCH-HMM exposure brake.")
    p.add_argument("--checkpoint", type=Path, default=_CHECKPOINT_5D)
    p.add_argument("--macro-parquet", type=Path, default=_MACRO_PARQUET)
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="(legacy binary brake) P(Bull) cutoff — unused by linear scaler")
    p.add_argument("--min-exposure", type=float, default=0.2,
                   help="exposure floor for linear brake clip (default: 0.2)")
    p.add_argument("--max-exposure", type=float, default=1.0,
                   help="exposure cap for linear brake clip (default: 1.0)")
    p.add_argument("--hold-days", type=int, default=5,
                   help="Tranche hold period (default: 5 for T+5)")
    p.add_argument("--seed-idx", type=int, default=0,
                   help="Which seed from checkpoint to use (default: 0)")
    p.add_argument("--regime-sizing", action="store_true",
                   help="Also apply existing regime-conditional sizing")
    a = p.parse_args()
    return {
        "checkpoint_path": a.checkpoint,
        "macro_parquet": a.macro_parquet,
        "n_states": a.n_states,
        "threshold": a.threshold,
        "min_exposure": a.min_exposure,
        "max_exposure": a.max_exposure,
        "hold_days": a.hold_days,
        "use_regime_sizing": a.regime_sizing,
        "seed_idx": a.seed_idx,
    }


if __name__ == "__main__":
    try:
        main(**_cli())
    except Exception as exc:
        logging.basicConfig(level=logging.INFO)
        LOGGER.error("FATAL: %s", exc, exc_info=True)
        sys.exit(1)
