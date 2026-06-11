"""
run_backtest.py — V4.0 "Fast Evaluator".

The cheap, iterate-all-day half of the pipeline.  It NEVER retrains a GBM: it
loads the frozen ensembles + HMM from `models/saved/v3_training_checkpoint.joblib`
(written by `train_models.py`), re-materializes the SAME dataset, then runs the
walk-forward threshold sweep, statistical rigor gates, and persists the live-bot
payload.  Tune walk-forward / sweep parameters here as often as you like — each
run is ~minutes, not the ~40 you'd pay to retrain.

Pipeline
────────
    1. Load training checkpoint           (ensembles, HMM, feature list, cutoff,
                                           train_cfg)
    2. Re-materialize the dataset         (ingest → features → labels → align),
                                          then subset to the checkpoint's feature
                                          list  →  train/serve parity by replay
    3. Threshold sweep                    (WalkForwardEngine, VN50 gate
                                           liquid_top_n=50) over up_threshold grid
    4. GOLDEN config                      (max mean OOS Net PnL across seeds)
    5. Signal evaluation                  (OOS UP-precision, pre-PnL)
    6. Persist the live-bot payload       → models/saved/v3_ensemble_{H}d.joblib
    7. Deflated Sharpe + PBO (CSCV)       on the GOLDEN's per-seed equity curves

Dataset hyper-params (horizon, PT/SL, train_frac, data sources) are LOCKED to
whatever `train_models.py` used — they ride in the checkpoint's `train_cfg`.
Only walk-forward / portfolio / sweep knobs are tunable here.

Run
───
    python run_backtest.py
    python run_backtest.py --liquid-top-n 50 --max-positions 10
    python run_backtest.py --sweep-thresholds 0.55,0.50,0.45 --no-save
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.backtest.pipeline import (
    RunConfig,
    AlignedData,
    TRADING_DAYS,
    FEATURE_RECIPE_VERSION,
    configure_logging,
    phase,
    materialize_dataset,
    subset_features,
    load_corporate_actions,
)
from src.models.tabular_ensemble import TabularEnsemble, make_ensemble_oracle
from src.models.macro_risk_hmm import build_market_proxy_returns
from src.backtest.walk_forward import WalkForwardEngine, WalkForwardConfig
from src.portfolio.construction import PortfolioConstraints
from src.execution.vn_cost_model import ExecutionConfig
from src.models.statistical_gates import deflated_sharpe, cscv_pbo

LOGGER = logging.getLogger("quant.backtest")

CHECKPOINT_PATH = Path("models/saved/v3_training_checkpoint.joblib")
DEFAULT_SWEEP_THRESHOLDS = [0.50, 0.45, 0.40, 0.35]
REQUIRED_CKPT_KEYS = ("ensembles", "tabular_features", "cutoff", "train_cfg")


# ─────────────────────────────────────────────────────────────────────────────
# OOS walk-forward (Phase 8) — frozen ensemble oracle
# ─────────────────────────────────────────────────────────────────────────────

def _build_wf_config(tabular_features: list[str], cutoff: date, cfg: RunConfig,
                     mode: str = "tranche", hold_days: int = 30,
                     pt_sigma: float | None = None,
                     sl_sigma: float | None = None) -> WalkForwardConfig:
    """Pure WalkForwardConfig builder — extracted from `run_oos` so the
    mode/hold-days plumbing is unit-testable without running the engine.

    `mode="grid"` reproduces the legacy delta-rebalance book byte-for-byte;
    `mode="tranche"` runs the staggered AFML cohort book (the evaluator default
    since the grid-date study showed the grid book destroys the per-trade edge).
    """
    return WalkForwardConfig(
        seq_len=1,                               # V3/V4: pure tabular — single-bar inputs
        feature_cols=tabular_features,
        initial_capital=cfg.initial_capital, max_positions=cfg.max_positions,
        rebalance_frequency=cfg.rebalance_frequency, signal_threshold=cfg.signal_threshold,
        cov_lookback=60, kelly_fraction=cfg.kelly_fraction, risk_aversion=cfg.risk_aversion,
        liquid_top_n=cfg.liquid_top_n,            # VN50 top-N ADV gate (tradeable universe)
        start_trading_date=cutoff,
        rebalance_mode=mode,
        tranche_hold_days=hold_days,
        tranche_pt_sigma=pt_sigma,
        tranche_sl_sigma=sl_sigma,
        constraints=PortfolioConstraints(
            max_weight=cfg.max_weight, long_only=True,
            target_leverage=0.95, target_vol=cfg.target_vol),
        exec_config=ExecutionConfig(),
    )


def run_oos(panel, tabular_features: list[str], ensemble: TabularEnsemble,
            corporate_actions: list, cutoff: date, cfg: RunConfig,
            p_bull_series: pd.Series | None = None,
            inference_cache: dict | None = None,
            mode: str = "tranche", hold_days: int = 30,
            pt_sigma: float | None = None,
            sl_sigma: float | None = None) -> pd.DataFrame:
    """Walk-forward OOS using the pure-tabular ensemble oracle.

    The engine builds (n, 1, F) single-bar tensors internally (seq_len=1) and the
    oracle slices the trailing bar before scoring.  `cfg.signal_threshold` is set
    by the sweep loop before each call.

    `inference_cache` — a PER-SEED ``{date: (p_up, tickers)}`` map reused across
    the threshold sweep.  Because oracle scoring is threshold-independent, only
    the first threshold for a given seed pays the GBM cost; the rest hit the
    cache and re-run just the (cheap) allocation/execution path.  NOTE: tranche
    mode scores DAILY (~900 OOS days), so the first threshold per seed is the
    expensive one (~15 min); grid mode only scores on rebalance days.
    """
    oracle = make_ensemble_oracle(ensemble)
    # Lookback buffer ahead of cutoff so OOS day-1 has feature warm-up + cov history.
    buffer = 80                                  # ~20d feature warm-up + 60d cov lookback
    all_dates = sorted(panel["date"].unique().to_list())
    cutoff_idx = next((i for i, d in enumerate(all_dates) if d >= cutoff), 0)
    buf_start = all_dates[max(0, cutoff_idx - buffer)]
    sub = panel.filter(pl.col("date") >= buf_start)

    wf_cfg = _build_wf_config(tabular_features, cutoff, cfg, mode, hold_days,
                              pt_sigma, sl_sigma)
    eng = WalkForwardEngine(wf_cfg, oracle)
    # Soft HMM regime scaling: P(Bull) multiplies the daily target weights.
    result = eng.run(sub, corporate_actions=corporate_actions, p_bull_series=p_bull_series,
                     inference_cache=inference_cache)
    eq = result.equity_curve
    eq = eq[pd.to_datetime(eq["date"]).dt.date >= cutoff].reset_index(drop=True)
    return eq


# ─────────────────────────────────────────────────────────────────────────────
# Equity metrics
# ─────────────────────────────────────────────────────────────────────────────

def monthly_net_sharpe(eq: pd.DataFrame) -> pd.Series:
    s = eq.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["period"] = s["date"].dt.to_period("M")
    out = {}
    for p, g in s.groupby("period"):
        r = g["daily_return"].to_numpy()
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        out[p] = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    return pd.Series(out).sort_index()


def equity_metrics(eq: pd.DataFrame, initial_capital: float) -> dict:
    nav = eq["nav"].to_numpy()
    r = eq["daily_return"].to_numpy()
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    dd = (nav / np.maximum.accumulate(nav) - 1.0).min() if len(nav) else 0.0
    return {
        "net_pnl": float(nav[-1] - initial_capital) if len(nav) else 0.0,
        "total_return": float(nav[-1] / initial_capital - 1.0) if len(nav) else 0.0,
        "net_sharpe": sharpe,
        "max_drawdown": float(dd),
        "final_nav": float(nav[-1]) if len(nav) else initial_capital,
        "n_days": int(len(eq)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal evaluation — pure OOS directional accuracy, BEFORE PnL
# ─────────────────────────────────────────────────────────────────────────────

def signal_evaluation_report(ensemble: TabularEnsemble, aligned: AlignedData,
                             test_mask: np.ndarray, *, up_threshold: float = 0.40) -> dict:
    """
    Is the alpha REAL? Score the ensemble's directional calls on the OOS labels,
    independent of portfolio sizing / PnL.  Headline metric = PRECISION of UP.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    Xte = aligned.X[test_mask]
    if len(Xte) == 0:
        LOGGER.warning("Signal evaluation skipped — empty OOS set.")
        return {}

    p_up = np.asarray(ensemble.predict_proba(Xte)).ravel()

    y_true = aligned.y[test_mask].astype(int)          # 3-class {0:DOWN,1:FLAT,2:UP}
    y_true_up = (y_true == 2).astype(int)              # UP-vs-rest ground truth
    y_pred_up = (p_up >= up_threshold).astype(int)

    bar = "=" * 72                                     # ASCII — cp1252-safe regardless of wrap
    rep = classification_report(y_true_up, y_pred_up, labels=[0, 1],
                                target_names=["NOT_UP", "UP(2)"], digits=4, zero_division=0)
    cm = confusion_matrix(y_true_up, y_pred_up, labels=[0, 1])   # [[TN,FP],[FN,TP]]

    up_sel = y_pred_up == 1
    n_pred_up = int(up_sel.sum())
    true_among_up = np.bincount(y_true[up_sel], minlength=3) if n_pred_up else np.zeros(3, int)
    not_sel = ~up_sel
    true_among_not = np.bincount(y_true[not_sel], minlength=3) if not_sel.any() else np.zeros(3, int)
    up_precision = float(true_among_up[2] / n_pred_up) if n_pred_up else 0.0
    down_among_up = float(true_among_up[0] / n_pred_up) if n_pred_up else 0.0

    LOGGER.info("\n%s\n SIGNAL EVALUATION — V4 Ensemble OOS directional accuracy "
                "(pre-PnL)\n%s", bar, bar)
    LOGGER.info("OOS samples=%d  predicted-UP=%d  (P(UP) ≥ %.2f)  true-UP base rate=%.3f",
                len(Xte), n_pred_up, up_threshold, float(y_true_up.mean()))
    LOGGER.info("Confusion matrix (UP-vs-rest) [rows=true, cols=pred]:\n%s", cm)
    LOGGER.info("Predicted-UP → TRUE label breakdown:  DOWN=%d (%.1f%%)  FLAT=%d  UP=%d (%.1f%%)",
                int(true_among_up[0]), 100 * down_among_up, int(true_among_up[1]),
                int(true_among_up[2]), 100 * up_precision)
    LOGGER.info("Not-predicted-UP → TRUE breakdown:    DOWN=%d  FLAT=%d  UP=%d",
                int(true_among_not[0]), int(true_among_not[1]), int(true_among_not[2]))
    LOGGER.info("Classification report (UP-vs-rest):\n%s", rep)
    LOGGER.info("★ PRECISION OF UP (Class 2) = %.4f   ← the 'is the alpha real?' number", up_precision)
    LOGGER.info("%s", bar)
    return {
        "up_precision": up_precision,
        "down_rate_among_pred_up": down_among_up,
        "n_pred_up": n_pred_up,
        "n_oos": int(len(Xte)),
        "true_up_base_rate": float(y_true_up.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {path}\n"
            f"Run `python train_models.py` first (the heavy lifter writes it).")
    ckpt = joblib.load(path)
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint {path} missing required keys: {missing}")
    return ckpt


def _apply_eval_overrides(cfg: RunConfig, overrides: dict) -> RunConfig:
    """Make the walk-forward / portfolio knobs reflect the CURRENT code defaults
    (plus any CLI override) — NOT the values frozen into the checkpoint at train
    time.

    The split's whole purpose is to iterate on eval/sizing parameters WITHOUT
    retraining, so eval knobs must track the live `RunConfig` defaults (e.g. the
    max_weight/max_positions edited in pipeline.py) and explicit CLI flags.
    Dataset / model fields (horizon, PT/SL, frac_diff_d, train_frac, sources) are
    NOT in `eval_fields`, so they stay LOCKED to whatever the checkpoint trained
    on — preserving feature/label parity while letting risk sizing move freely.
    """
    eval_fields = {
        "initial_capital", "max_positions", "rebalance_frequency", "max_weight",
        "target_vol", "kelly_fraction", "risk_aversion", "liquid_top_n", "cscv_S",
    }
    fresh = RunConfig()                       # current code defaults
    # 1) Refresh every eval knob to the CURRENT default (drop the checkpoint's
    #    frozen eval values), so editing pipeline.py takes effect with no retrain.
    for f in eval_fields:
        setattr(cfg, f, getattr(fresh, f))
    # 2) Apply explicit CLI overrides on top.
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in eval_fields:
            raise ValueError(f"_apply_eval_overrides: '{k}' is not a tunable eval field")
        setattr(cfg, k, v)
    LOGGER.info(
        "Eval knobs (refreshed from current defaults + CLI) | max_weight=%.2f  "
        "max_positions=%d  worst-case gross=%.0f%%  liquid_top_n=%s  target_vol=%.2f  "
        "(dataset/model fields stay locked to the checkpoint)",
        cfg.max_weight, cfg.max_positions, cfg.max_weight * cfg.max_positions * 100,
        cfg.liquid_top_n, cfg.target_vol)
    return cfg


def main(checkpoint_path: Path = CHECKPOINT_PATH, *,
         eval_overrides: dict | None = None,
         sweep_thresholds: list[float] | None = None,
         save_bot_payload: bool = True,
         export_only: bool = False,
         mode: str = "tranche",
         hold_days: int = 30,
         pt_sigma: float | None = None,
         sl_sigma: float | None = None) -> None:
    configure_logging()
    t_start = time.perf_counter()
    sweep_thresholds = list(sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS)

    # ── 1. Load the frozen training checkpoint ───────────────────────────────
    with phase("Load training checkpoint"):
        ckpt = _load_checkpoint(checkpoint_path)
        train_cfg: RunConfig = ckpt["train_cfg"]
        tabular_features: list[str] = list(ckpt["tabular_features"])
        cutoff: date = ckpt["cutoff"]
        trained: list[tuple[int, TabularEnsemble]] = list(ckpt["ensembles"])
        macro_hmm = ckpt.get("macro_hmm")
        seeds = [s for s, _ in trained]
        LOGGER.info("Checkpoint | schema=%s  seeds=%s  features=%d  cutoff=%s  HMM=%s",
                    ckpt.get("schema_version", "?"), seeds, len(tabular_features),
                    cutoff, macro_hmm is not None)

    # Dataset params LOCKED to train; only eval knobs are tunable here.
    cfg = _apply_eval_overrides(train_cfg, eval_overrides or {})

    LOGGER.info("=" * 100)
    LOGGER.info(" RUN_BACKTEST (Fast Evaluator) | mode=%s%s  horizon=T+%d  PT=%.1fσ  SL=%.1fσ  "
                "VN50_gate(liquid_top_n)=%d  seeds=%d",
                mode, f" hold={hold_days}d" if mode == "tranche" else "",
                cfg.tb_horizon, cfg.tb_pt, cfg.tb_sl, cfg.liquid_top_n, len(seeds))
    LOGGER.info("=" * 100)

    # ── EXPORT-ONLY fast path ────────────────────────────────────────────────
    # Skip the entire walk-forward + sweep + DSR/PBO simulation (and even the
    # expensive dataset materialization) — just repackage the live-bot artifact
    # straight from the frozen checkpoint.  Finishes in seconds.
    if export_only:
        _export_only(cfg, tabular_features, trained, macro_hmm,
                     mode=mode, hold_days=hold_days)
        LOGGER.info(" Wall-clock: %.1fs", time.perf_counter() - t_start)
        return

    # ── 2. Re-materialize the SAME dataset, then replay the feature selection ─
    ds = materialize_dataset(cfg)
    if ds.cutoff != cutoff:
        LOGGER.warning("Re-materialized cutoff %s != checkpoint cutoff %s — using checkpoint's "
                       "(train/serve split parity).", ds.cutoff, cutoff)
    with phase("Subset features to the checkpoint's selected pool"):
        ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
        LOGGER.info("Eval matrix | rows=%d  features=%d  (%s)",
                    ds.aligned.X.shape[0], ds.aligned.X.shape[1], tabular_features)

    corporate_actions = load_corporate_actions(cfg)

    # Recompute the leak-free filtered P(Bull) from the FROZEN HMM over this panel.
    p_bull_series = None
    if macro_hmm is not None:
        try:
            market_ret = build_market_proxy_returns(ds.panel)
            p_bull_series = macro_hmm.p_bull_series(market_ret, filtered=True)
            oos_pb = p_bull_series[p_bull_series.index >= pd.Timestamp(cutoff)]
            LOGGER.info("HMM P(Bull) | bull_state=%d  OOS mean=%.3f  OOS min=%.3f",
                        macro_hmm.bull_state, float(oos_pb.mean()), float(oos_pb.min()))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("P(Bull) recompute failed (%s) — full exposure (no scaling).", exc)
            p_bull_series = None

    # ── 3. THRESHOLD SWEEP (the cheap goal-seeker) ───────────────────────────
    test_mask = ds.aligned.dates >= cutoff
    Xte_all = ds.aligned.X[test_mask]
    y_true_all = ds.aligned.y[test_mask].astype(int)
    y_true_up_all = (y_true_all == 2).astype(int)

    # Per-seed inference caches — populated on the FIRST threshold, then reused
    # (no GBM re-scoring) on every threshold after.  Keyed by seed so each frozen
    # ensemble keeps its own {date: (p_up, tickers)} map; oracle scoring is
    # threshold-independent, so this turns an O(thresholds × seeds) inference cost
    # into O(seeds).
    seed_inference_caches: dict[int, dict] = {seed: {} for seed, _ in trained}

    sweep_results: list[dict] = []
    for thr in sweep_thresholds:
        sig_thr = thr - 0.05                            # engine gate 5pp below model threshold
        cfg.signal_threshold = sig_thr                  # WalkForwardConfig pulls from here
        with phase(f"Sweep | mode={mode}  up_threshold={thr:.2f}  signal_threshold={sig_thr:.2f}"):
            per_seed: list[dict] = []
            monthly_cols_thr: dict[str, pd.Series] = {}
            for seed, ensemble in trained:
                try:
                    eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                                 cutoff, cfg, p_bull_series=p_bull_series,
                                 inference_cache=seed_inference_caches[seed],
                                 mode=mode, hold_days=hold_days,
                                 pt_sigma=pt_sigma, sl_sigma=sl_sigma)
                    m = equity_metrics(eq, cfg.initial_capital)
                    # Inline UP-precision @thr (no log spam)
                    if len(Xte_all) > 0:
                        p_up = np.asarray(ensemble.predict_proba(Xte_all)).ravel()
                        y_pred_up = (p_up >= thr).astype(int)
                        n_pred_up = int(y_pred_up.sum())
                        tp = int(((y_pred_up == 1) & (y_true_up_all == 1)).sum())
                        up_precision = float(tp / n_pred_up) if n_pred_up else 0.0
                    else:
                        n_pred_up, up_precision = 0, 0.0
                    per_seed.append({"seed": seed, "eq": eq, "ensemble": ensemble,
                                     "up_threshold": thr, **m,
                                     "n_pred_up": n_pred_up,
                                     "up_precision": up_precision})
                    monthly_cols_thr[f"seed_{seed}"] = monthly_net_sharpe(eq)
                    LOGGER.info(
                        "    thr=%.2f seed=%d  NetPnL=%s  Sharpe=%+.2f  "
                        "DD=%.2f%%  predUP=%d  prec=%.4f",
                        thr, seed, f"{m['net_pnl']:+,.0f}", m["net_sharpe"],
                        m["max_drawdown"] * 100, n_pred_up, up_precision)
                except Exception as exc:                # noqa: BLE001 — sweep must NEVER crash
                    LOGGER.warning("    thr=%.2f seed=%d FAILED: %s", thr, seed, exc)
        if not per_seed:
            LOGGER.warning("All seeds failed at threshold %.2f", thr)
            continue
        agg = {
            "up_threshold": thr,
            "signal_threshold": sig_thr,
            "mean_net_pnl": float(np.mean([p["net_pnl"] for p in per_seed])),
            "mean_sharpe": float(np.mean([p["net_sharpe"] for p in per_seed])),
            "mean_dd": float(np.mean([p["max_drawdown"] for p in per_seed])),
            "total_pred_up": int(sum(p["n_pred_up"] for p in per_seed)),
            "mean_up_precision": float(np.mean([p["up_precision"] for p in per_seed])),
            "n_seeds_ok": len(per_seed),
            "per_seed": per_seed,
            "monthly_cols": monthly_cols_thr,
        }
        sweep_results.append(agg)
        LOGGER.info(
            "  ★ THR=%.2f  mean_NetPnL=%s  mean_Sharpe=%+.3f  total_predUP=%d  mean_UPprec=%.4f",
            thr, f"{agg['mean_net_pnl']:+,.0f}", agg["mean_sharpe"],
            agg["total_pred_up"], agg["mean_up_precision"])

    if not sweep_results:
        LOGGER.error("Sweep produced no successful results — aborting teardown.")
        return

    # ── 4. GOLDEN CONFIG — threshold that maximises mean OOS Net PnL ─────────
    golden = max(sweep_results, key=lambda r: r["mean_net_pnl"])
    best_seed_record = max(golden["per_seed"], key=lambda p: p["net_sharpe"])

    # ── 5. Full verbose signal-eval JUST for the GOLDEN ──────────────────────
    with phase(f"Signal evaluation — GOLDEN (up_threshold={golden['up_threshold']:.2f})"):
        signal_evaluation_report(best_seed_record["ensemble"], ds.aligned, test_mask,
                                 up_threshold=golden["up_threshold"])

    # ── 5b. Calibrated P(UP) distribution vs Kelly cap-onset (R-tuning study) ─
    with phase("Probability-distribution diagnostic (calibrated P(UP) vs Kelly cap-onset)"):
        _plot_prob_distribution(best_seed_record["ensemble"], Xte_all, golden)

    # ── 6. Persist the live-bot payload (bot_inference.V3BotInference schema) ─
    if save_bot_payload:
        _persist_bot_payload(cfg, tabular_features, golden, best_seed_record,
                             macro_hmm, sweep_results, mode=mode, hold_days=hold_days,
                             pt_sigma=pt_sigma, sl_sigma=sl_sigma)

    # ── 7. Phase 4 DSR + PBO on the GOLDEN's per-seed equity curves ──────────
    # Multiplicity for DSR = TOTAL configs swept (thresholds × seeds).
    with phase("Phase 4 — Deflated Sharpe + PBO (CSCV) on GOLDEN"):
        daily_r = best_seed_record["eq"]["daily_return"].to_numpy()
        n_trials_total = len(sweep_thresholds) * max(1, len(seeds))
        dsr = deflated_sharpe(daily_r, n_trials=n_trials_total, annualisation=TRADING_DAYS)
        gm = golden["monthly_cols"]
        if len(gm) >= 2:
            M = pd.concat(gm, axis=1).fillna(0.0).sort_index()
            S = min(cfg.cscv_S, (len(M) // 2) * 2)
            pbo = cscv_pbo(M.to_numpy(), S=max(2, S)) if len(M) >= 2 else \
                {"pbo": float("nan"), "valid": False, "warning": "too few periods"}
        else:
            pbo = {"pbo": float("nan"), "valid": False,
                   "warning": "single config — train ≥2 seeds for PBO"}

    _print_sweep_report(sweep_results, golden, mode=mode, hold_days=hold_days)
    _teardown_report(cfg, best_seed_record, dsr, pbo, len(seeds),
                     time.perf_counter() - t_start, mode=mode, hold_days=hold_days)


def _persist_bot_payload(cfg: RunConfig, tabular_features: list[str], golden: dict,
                         best_seed_record: dict, macro_hmm, sweep_results: list[dict],
                         mode: str = "tranche", hold_days: int = 30,
                         pt_sigma: float | None = None,
                         sl_sigma: float | None = None) -> None:
    """Single self-contained joblib the bot loads (src/bot/bot_inference.py):
    ensemble + feature-column order + decision thresholds + HMM overlay + provenance.
    Horizon-aware filename so dual-horizon bots can run side by side."""
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = save_dir / f"v3_ensemble_{cfg.tb_horizon}d.joblib"

    # Back up the EXISTING artifact before overwriting.  models/saved/ is
    # git-untracked, unversioned, and un-backed-up, so a bad/partial run would
    # otherwise clobber the live model with no rollback (a synthetic smoke run did
    # exactly this once).  Timestamped copies land in models/saved/backups/.
    if artifact_path.exists():
        import shutil
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_dir = save_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"v3_ensemble_{cfg.tb_horizon}d_{ts}.joblib"
        shutil.copy2(artifact_path, backup_path)
        LOGGER.info("Backed up existing artifact → %s", backup_path)

    bundle = {
        "schema_version": "v3.0",
        "ensemble": best_seed_record["ensemble"],
        "tabular_features": list(tabular_features),          # column order is load-bearing
        "up_threshold": float(golden["up_threshold"]),
        "signal_threshold": float(golden["signal_threshold"]),
        # ADDITIVE — the validated portfolio construction these thresholds were
        # tuned under.  The serve path does not consume this yet (Phase 2:
        # NAV/hold_days/max_positions sizing + exit-after-hold alerts); loaders
        # validate required keys only, so unknown keys are safe.
        "strategy": {
            "mode": mode,
            "hold_days": int(hold_days),
            "signal_threshold": float(golden["signal_threshold"]),
            "pt_sigma": pt_sigma,                        # None ⇒ barrier off
            "sl_sigma": sl_sigma,
        },
        "macro_hmm": macro_hmm,                              # may be None
        "metadata": {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "best_seed": int(best_seed_record["seed"]),
            "n_seeds_in_golden": int(golden["n_seeds_ok"]),
            "oos_net_pnl_vnd": float(best_seed_record["net_pnl"]),
            "oos_total_return": float(best_seed_record["total_return"]),
            "oos_sharpe": float(best_seed_record["net_sharpe"]),
            "oos_max_dd": float(best_seed_record["max_drawdown"]),
            "oos_days": int(best_seed_record["n_days"]),
            "oos_final_nav_vnd": float(best_seed_record["final_nav"]),
            "golden_total_pred_up": int(golden["total_pred_up"]),
            "golden_mean_up_precision": float(golden["mean_up_precision"]),
            "tb_horizon": int(cfg.tb_horizon),
            "tb_pt": float(cfg.tb_pt),
            "tb_sl": float(cfg.tb_sl),
            # FEATURE-RECIPE LOCK (train/serve parity): the only config-driven
            # feature hyper-param in build_features.  The bot reads this back and
            # passes it into V3FeatureConfig at serve time — never a default.
            # (All other recipe knobs are fixed literals inside the SHARED
            # build_features, so they cannot drift between train and serve.)
            "frac_diff_d": float(cfg.frac_diff_d),
            "feature_recipe_version": FEATURE_RECIPE_VERSION,   # structural recipe stamp (tripwire)
            "categorical_features": list(getattr(best_seed_record["ensemble"], "categorical_features", [])),
            "liquid_top_n": int(cfg.liquid_top_n) if cfg.liquid_top_n else None,
            "sweep_results": [
                {k: v for k, v in r.items() if k not in ("per_seed", "monthly_cols")}
                for r in sweep_results
            ],
        },
    }
    joblib.dump(bundle, artifact_path, compress=3)
    size_kb = artifact_path.stat().st_size / 1024
    LOGGER.info("V4 GOLDEN bot payload persisted → %s  (%.1f KB)  seed=%d  thr=%.2f",
                artifact_path, size_kb, best_seed_record["seed"], golden["up_threshold"])


def _plot_prob_distribution(
    ensemble, X_oos: np.ndarray, golden: dict, *,
    save_path: Path = Path("models/saved/prob_distribution.png"),
) -> None:
    """Histogram of the GOLDEN config's CALIBRATED OOS P(UP), with half-Kelly
    cap-onset thresholds for R ∈ {1.2, 1.5, 2.0} overlaid.

    cap-onset p = (0.2R + 1) / (R + 1)  — the win-probability above which
    half-Kelly (×0.5, 10% cap) PINS at the NAV cap; below it, sizing scales
    smoothly.  Overlaying these on the actual calibrated mass tells us which R
    keeps the predictions inside the differentiating (sub-cap) band.

    Prints a percentile summary to the console and saves a PNG.  Plotting is
    best-effort: a missing matplotlib or any draw error never aborts the
    teardown — the text summary is always emitted.
    """
    if X_oos is None or len(X_oos) == 0:
        LOGGER.warning("prob-distribution diagnostic skipped — empty OOS matrix.")
        return
    p_up = np.asarray(ensemble.predict_proba(X_oos), dtype=float).ravel()
    p_up = p_up[np.isfinite(p_up)]
    if p_up.size == 0:
        LOGGER.warning("prob-distribution diagnostic skipped — no finite P(UP).")
        return

    R_grid = [1.2, 1.5, 2.0]
    cap_onset = {R: (0.2 * R + 1.0) / (R + 1.0) for R in R_grid}   # closed form
    up_thr = float(golden.get("up_threshold", 0.0))

    # ── console text summary ────────────────────────────────────────────────
    q = np.percentile(p_up, [0, 10, 25, 50, 75, 90, 100])
    bar = "=" * 72
    LOGGER.info("\n%s\n CALIBRATED OOS P(UP) DISTRIBUTION — GOLDEN (up_thr=%.2f, n=%d)\n%s",
                bar, up_thr, p_up.size, bar)
    LOGGER.info("  min=%.3f  p10=%.3f  p25=%.3f  MEDIAN=%.3f  p75=%.3f  p90=%.3f  max=%.3f",
                q[0], q[1], q[2], q[3], q[4], q[5], q[6])
    LOGGER.info("  mean=%.3f  std=%.3f", float(p_up.mean()), float(p_up.std()))
    for R in R_grid:
        c = cap_onset[R]
        pinned = 100.0 * float((p_up >= c).mean())
        LOGGER.info("  R=%.1f | cap-onset p≥%.3f  →  %.1f%% PIN at 10%% cap  |  "
                    "%.1f%% size sub-cap", R, c, pinned, 100.0 - pinned)
    LOGGER.info("%s", bar)

    # ── PNG ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")                          # headless backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(p_up, bins=np.linspace(0.0, 1.0, 51), color="#4C72B0",
                alpha=0.75, edgecolor="white", linewidth=0.5)
        line_colors = {1.2: "#2ca02c", 1.5: "#ff7f0e", 2.0: "#d62728"}
        for R in R_grid:
            c = cap_onset[R]
            ax.axvline(c, color=line_colors[R], linestyle="--", linewidth=2.0,
                       label=f"R={R}  cap-onset p={c:.3f}")
        if up_thr > 0:
            ax.axvline(up_thr, color="black", linestyle=":", linewidth=1.5,
                       label=f"BUY threshold p={up_thr:.2f}")
        ax.set_title(f"Calibrated OOS P(UP) — GOLDEN (n={p_up.size})  •  "
                     f"left of a line = sizes sub-cap for that R")
        ax.set_xlabel("Calibrated P(UP)")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        LOGGER.info("Probability-distribution plot saved → %s", save_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("prob-distribution PNG failed (%s: %s) — text summary above still valid.",
                       type(exc).__name__, exc)


def _print_sweep_report(sweep_results: list[dict], golden: dict,
                        mode: str = "tranche", hold_days: int = 30) -> None:
    """Threshold-sweep summary table — one row per up_threshold, GOLDEN tagged."""
    bar = "=" * 100
    mode_tag = f"mode={mode}" + (f" hold={hold_days}d" if mode == "tranche" else "")
    print(f"\n{bar}")
    print(f" THRESHOLD SWEEP REPORT ({mode_tag})  —  goal: maximise mean OOS Net PnL across the seed pool")
    print(bar)
    header = (f" {'up_thr':>7} {'sig_thr':>8} {'seeds':>6} "
              f"{'mean_NetPnL (VND)':>22} {'mean_Sharpe':>13} {'mean_DD':>10} "
              f"{'total_predUP':>14} {'mean_UPprec':>13}")
    print(header)
    print(" " + "-" * (len(header) - 1))
    for r in sweep_results:
        mark = "  <- GOLDEN" if r is golden else ""
        print(f" {r['up_threshold']:>7.2f} {r['signal_threshold']:>8.2f} {r['n_seeds_ok']:>6d} "
              f"{r['mean_net_pnl']:>+22,.0f} {r['mean_sharpe']:>+13.3f} "
              f"{r['mean_dd']*100:>+9.2f}% {r['total_pred_up']:>14,d} "
              f"{r['mean_up_precision']:>13.4f}{mark}")
    print(bar)
    print(f" ★ GOLDEN CONFIG  ->  {mode_tag}  up_threshold={golden['up_threshold']:.2f}  "
          f"signal_threshold={golden['signal_threshold']:.2f}  "
          f"mean_NetPnL={golden['mean_net_pnl']:+,.0f} VND  "
          f"mean_Sharpe={golden['mean_sharpe']:+.3f}  "
          f"total_predUP={golden['total_pred_up']:,d}")
    print(bar + "\n")


def _teardown_report(cfg, best, dsr, pbo, n_configs, elapsed,
                     mode: str = "tranche", hold_days: int = 30):
    bar = "=" * 70                                     # ASCII — cp1252-safe regardless of wrap
    mode_tag = mode + (f" (hold={hold_days}d)" if mode == "tranche" else "")
    print(f"\n{bar}\n QUANT ENGINE V4.0 — OOS BACKTEST TEARDOWN REPORT\n{bar}")
    print(f" Best config        : seed={best['seed']}  mode={mode_tag}  ({n_configs} configs tried)")
    print(f" OOS trading days   : {best['n_days']}")
    print(f" Initial capital    : {cfg.initial_capital:,.0f} VND")
    print(f" Final NAV          : {best['final_nav']:,.0f} VND")
    print("─" * 70)
    print(f" Total Net PnL      : {best['net_pnl']:+,.0f} VND  ({best['total_return']:+.2%})")
    print(f" Net Sharpe (ann.)  : {best['net_sharpe']:+.3f}")
    print(f" Max Drawdown       : {best['max_drawdown']:.2%}")
    print("─" * 70)
    if dsr.get("valid"):
        print(f" Deflated Sharpe    : SR={dsr['sr_annualised']:+.3f}  SR0={dsr['sr0_annualised']:+.3f}")
        print(f" DSR p-value        : {dsr['p_dsr']:.4f}   "
              f"({'PASS ≥0.95' if dsr['p_dsr'] >= 0.95 else 'FAIL <0.95'})")
    else:
        print(f" Deflated Sharpe    : N/A — {dsr.get('warning', 'invalid')}")
    if pbo.get("valid"):
        print(f" PBO (CSCV)         : {pbo['pbo']:.1%}   "
              f"({'PASS ≤10%' if pbo['pbo'] <= 0.10 else 'FAIL >10%'})  "
              f"[T={pbo['n_periods']} months, N={pbo['n_configs']} configs]")
    else:
        print(f" PBO (CSCV)         : N/A — {pbo.get('warning', 'invalid')}")
    print("─" * 70)
    fit = (dsr.get("valid") and dsr["p_dsr"] >= 0.95
           and pbo.get("valid") and pbo["pbo"] <= 0.10)
    print(f" PRODUCTION VERDICT : {'✓ FIT' if fit else '✗ UNFIT'} FOR PRODUCTION")
    print(f" Wall-clock         : {elapsed:.1f}s")
    print(f"{bar}\n")


def _export_only(cfg: RunConfig, tabular_features: list[str],
                 trained: list[tuple[int, TabularEnsemble]], macro_hmm,
                 mode: str = "tranche", hold_days: int = 30) -> None:
    """Repackage the live-bot artifact from the frozen checkpoint WITHOUT running
    the walk-forward backtest.

    No OOS simulation runs, so there are no fresh PnL/Sharpe metrics and no
    sweep-tuned threshold.  Thresholds are PRESERVED from the existing artifact
    when one is on disk (a pure post-retrain refresh keeps the last tuned gate);
    otherwise sensible defaults.  OOS metrics are stamped NaN to mark the
    artifact as export-only.  Reuses `_persist_bot_payload` so the bundle schema
    (and the backup-on-write) stay identical to the full path.
    """
    artifact_path = Path("models/saved") / f"v3_ensemble_{cfg.tb_horizon}d.joblib"

    # Preserve last-tuned thresholds from an existing artifact if present.
    up_thr = 0.50
    sig_thr = float(getattr(cfg, "signal_threshold", 0.40) or 0.40)
    if artifact_path.exists():
        try:
            prev = joblib.load(artifact_path)
            up_thr = float(prev.get("up_threshold", up_thr))
            sig_thr = float(prev.get("signal_threshold", sig_thr))
            LOGGER.info("Export-only | preserving thresholds from existing artifact "
                        "(up=%.2f, signal=%.2f)", up_thr, sig_thr)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Export-only | could not read existing artifact (%s) — defaults "
                           "(up=%.2f, signal=%.2f)", exc, up_thr, sig_thr)
    else:
        LOGGER.info("Export-only | no existing artifact — defaults (up=%.2f, signal=%.2f)",
                    up_thr, sig_thr)

    # No OOS ranking available → pick the seed matching train cfg, else the first.
    seed_to_ens = {s: e for s, e in trained}
    seed = cfg.seed if cfg.seed in seed_to_ens else trained[0][0]
    LOGGER.info("Export-only | packaging seed=%s (of %s) — NO OOS ranking (backtest skipped)",
                seed, list(seed_to_ens))

    _NA = float("nan")
    golden = {
        "up_threshold": up_thr, "signal_threshold": sig_thr,
        "n_seeds_ok": 1, "total_pred_up": 0, "mean_up_precision": _NA,
    }
    best_seed_record = {
        "ensemble": seed_to_ens[seed], "seed": seed,
        "net_pnl": _NA, "total_return": _NA, "net_sharpe": _NA,
        "max_drawdown": _NA, "n_days": 0, "final_nav": _NA,
    }
    _persist_bot_payload(cfg, tabular_features, golden, best_seed_record, macro_hmm, [],
                         mode=mode, hold_days=hold_days)

    LOGGER.info("=" * 100)
    LOGGER.info(" Export complete. Walk-forward backtest skipped.")
    LOGGER.info(" Artifact → %s  (horizon=T+%d, up=%.2f/signal=%.2f, OOS metrics=NaN)",
                artifact_path, cfg.tb_horizon, up_thr, sig_thr)
    LOGGER.info("=" * 100)


def _cli() -> tuple[Path, dict, list[float], bool, bool, str, int, float | None, float | None]:
    p = argparse.ArgumentParser(
        description="V4.0 Fast Evaluator — sweep + DSR/PBO on a frozen training checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH,
                   help="training checkpoint written by train_models.py")
    # Walk-forward / portfolio knobs (the whole point — iterate freely here).
    p.add_argument("--mode", choices=("tranche", "grid"), default="tranche",
                   help="portfolio construction: 'tranche' = staggered AFML cohort book "
                        "(default; daily inference — first threshold per seed ~15 min), "
                        "'grid' = legacy concentrated delta-rebalance book")
    p.add_argument("--hold-days", type=int, default=30,
                   help="tranche holding period in trading days (per-day net edge "
                        "peaks at 30; ignored in grid mode)")
    p.add_argument("--tranche-pt", type=float, default=None,
                   help="tranche profit-take barrier in entry-vol multiples "
                        "(labels use 3.0; default off)")
    p.add_argument("--tranche-sl", type=float, default=None,
                   help="tranche stop-loss barrier in entry-vol multiples "
                        "(labels use 2.0; default off)")
    p.add_argument("--liquid-top-n", type=int, default=None, help="VN50 ADV gate (default 50)")
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--rebalance-frequency", type=int, default=None)
    p.add_argument("--max-weight", type=float, default=None)
    p.add_argument("--target-vol", type=float, default=None)
    p.add_argument("--kelly-fraction", type=float, default=None)
    p.add_argument("--risk-aversion", type=float, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--cscv-s", type=int, default=None)
    p.add_argument("--sweep-thresholds", type=str, default=None,
                   help="comma list of up_thresholds, e.g. '0.55,0.50,0.45,0.40'")
    p.add_argument("--no-save", action="store_true",
                   help="skip persisting the live-bot payload (pure iteration)")
    p.add_argument("--export-only", "--skip-backtest", action="store_true", dest="export_only",
                   help="skip walk-forward + sweep; repackage the live-bot artifact from the "
                        "checkpoint in seconds (preserves existing tuned thresholds)")
    a = p.parse_args()

    overrides = {
        "liquid_top_n": a.liquid_top_n,
        "max_positions": a.max_positions,
        "rebalance_frequency": a.rebalance_frequency,
        "max_weight": a.max_weight,
        "target_vol": a.target_vol,
        "kelly_fraction": a.kelly_fraction,
        "risk_aversion": a.risk_aversion,
        "initial_capital": a.initial_capital,
        "cscv_S": a.cscv_s,
    }
    sweep = ([float(x) for x in a.sweep_thresholds.split(",")]
             if a.sweep_thresholds else None)
    return (a.checkpoint, overrides, sweep, (not a.no_save), a.export_only,
            a.mode, a.hold_days, a.tranche_pt, a.tranche_sl)


if __name__ == "__main__":
    (_ckpt, _overrides, _sweep, _save, _export,
     _mode, _hold, _pt, _sl) = _cli()
    main(_ckpt, eval_overrides=_overrides, sweep_thresholds=_sweep,
         save_bot_payload=_save, export_only=_export, mode=_mode, hold_days=_hold,
         pt_sigma=_pt, sl_sigma=_sl)
