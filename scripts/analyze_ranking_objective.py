"""Ranking-objective A/B for the tranche admission signal (2026-07-22, READ-ONLY).

Idea: admission only ever consumes RANK ORDER (top-K by score within a day's
cross-section) — `_tranche_day` never uses the ensemble's P(UP) as a real
probability, only to sort tickers and compare against a threshold. Yet
`TabularEnsemble` is trained with a POINTWISE multiclass objective (predict
each ticker's class independently) — tonight's own knife-catch research also
found the pointwise probability isn't even monotonically reliable (deep-
oversold MR-LGBM fires scored WORSE than moderate ones). A model trained
DIRECTLY on cross-sectional ranking might order tickers better without any
new features or data.

Design: LightGBM's `lambdarank` objective, `group` = trading date (all
tickers on the same day form one ranking group), relevance label = the
EXISTING ordinal triple-barrier `y` ∈ {0=DOWN, 1=SIDE, 2=UP} (already a valid
ordinal relevance scale — DOWN < SIDE < UP — no label engineering needed).

Scope (deliberately minimal for a first test — see the note at the bottom
before considering production wiring):
  * ONE `LGBMRanker`, not the full 3-model-stack + PurgedKFold-OOF +
    LogisticRegression-meta machinery. This tests the OBJECTIVE, not a
    production-ready replacement architecture.
  * Same frozen checkpoint's feature pool + chronological cutoff
    (`materialize_dataset` + `subset_features`, identical to
    `run_backtest.py`'s own steps 1-2 — train/serve parity by construction).
  * Same AFML sample weights (`aligned.w`) as the real ensemble uses.
  * Wrapped in a `TabularEnsemble`-compatible adapter (`.predict_proba(X) ->
    (n,) score in [0,1]`) via a PER-CALL cross-sectional percentile-rank
    transform of the ranker's raw score — ranking objectives don't produce
    calibrated probabilities, but the walk-forward admission logic only ever
    consumes rank + a threshold comparison, so percentile-rank is a faithful,
    mechanically drop-in substitute (same n, same call granularity: the
    engine already calls the oracle once per trading day with that day's
    full cross-section — see `WalkForwardEngine._inference`).
  * Runs the EXACT SAME `run_oos` / `WalkForwardEngine` the production sweep
    uses, for a directly comparable Sharpe/DD/PBO readout against the known
    GOLDEN baseline. Zero writes, zero checkpoint/artifact changes.

Run: python scripts/analyze_ranking_objective.py [--horizon 5|20] [--thresholds 0.5,0.7,0.9]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRanker  # noqa: E402

from run_backtest import (  # noqa: E402
    CHECKPOINT_PATH,
    _apply_eval_overrides,
    equity_metrics,
    monthly_net_sharpe,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    load_corporate_actions,
    materialize_dataset,
    subset_features,
)
from src.models.statistical_gates import cscv_pbo, deflated_sharpe  # noqa: E402

TRADING_DAYS = 252
SEED = 42


class _RankerAsEnsemble:
    """Adapts a fitted LGBMRanker to `TabularEnsemble.predict_proba`'s
    contract: `(n,) score in [0,1]` via per-call cross-sectional percentile
    rank. The walk-forward engine calls the oracle once per trading day with
    that day's full ticker cross-section (see `WalkForwardEngine._inference`),
    so "per call" == "per day" — exactly the granularity ranking needs.
    """

    def __init__(self, ranker: LGBMRanker):
        self.ranker = ranker

    def predict_proba(self, X) -> np.ndarray:
        raw = np.asarray(self.ranker.predict(X), dtype=np.float64)
        n = len(raw)
        if n <= 1:
            return np.full(n, 0.5, dtype=np.float64)
        order = np.argsort(raw)  # ascending: worst -> best
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        return ranks / (n - 1)  # 0..1, higher = better rank


def _lambdarank_groups(sorted_dates: np.ndarray) -> np.ndarray:
    """Run-length group sizes for consecutive equal dates.

    Caller MUST pass an ALREADY date-sorted array — LightGBM's `group`
    parameter is a list of CONTIGUOUS block sizes, not an arbitrary grouping
    key, so this only produces a valid `group` array for sorted input.
    """
    _, counts = np.unique(sorted_dates, return_counts=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=str, default="0.50,0.70,0.80,0.90")
    a = ap.parse_args()
    thresholds = [float(x) for x in a.thresholds.split(",")]

    print(f"Loading checkpoint {CHECKPOINT_PATH} ...")
    import joblib
    ckpt = joblib.load(CHECKPOINT_PATH)
    train_cfg: RunConfig = ckpt["train_cfg"]
    tabular_features: list[str] = list(ckpt["tabular_features"])
    cutoff = ckpt["cutoff"]
    print(f"  tb_horizon={train_cfg.tb_horizon}  cutoff={cutoff}  "
          f"features={len(tabular_features)}")

    # Eval/sizing knobs (max_positions, liquid_top_n, ...) track the CURRENT
    # code defaults, not whatever was frozen at train time — same refresh
    # run_backtest.main() applies before its own sweep (dataset/model fields
    # like tb_horizon are NOT in this set, so they stay locked to the
    # checkpoint — train/serve parity preserved).
    train_cfg = _apply_eval_overrides(train_cfg, {})
    print(f"  max_positions={train_cfg.max_positions}  "
          f"liquid_top_n={train_cfg.liquid_top_n}  "
          f"(refreshed to current code defaults)")

    print("Materializing dataset (same as run_backtest.py) ...")
    ds = materialize_dataset(train_cfg)
    ds.aligned = subset_features(ds.aligned, ds.all_features, tabular_features)
    corporate_actions = load_corporate_actions(train_cfg)

    aligned = ds.aligned
    train_mask = ds.train_mask  # authoritative — aligned.dates < cutoff

    X_tr = aligned.X[train_mask]
    y_tr = aligned.y[train_mask]
    w_tr = aligned.w[train_mask]
    d_tr = aligned.dates[train_mask]

    print(f"Train rows={len(y_tr)}  OOS rows={(~train_mask).sum()}")

    # Sort TRAIN by date so lambdarank's `group` sizes correspond to
    # CONTIGUOUS blocks (LightGBM's own requirement — a group is a run of
    # consecutive rows, not an arbitrary label).
    order = np.argsort(d_tr, kind="stable")
    X_tr, y_tr, w_tr, d_tr = X_tr[order], y_tr[order], w_tr[order], d_tr[order]
    groups = _lambdarank_groups(d_tr)
    assert groups.sum() == len(y_tr), "group sizes must sum to n_train_rows"
    print(f"Lambdarank groups (trading days): {len(groups)}  "
          f"(min/median/max tickers-per-day = {groups.min()}/"
          f"{int(np.median(groups))}/{groups.max()})")

    print("Fitting LGBMRanker (objective=lambdarank) ...")
    ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=63,
        max_bin=255,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )
    ranker.fit(X_tr, y_tr, sample_weight=w_tr, group=groups)
    wrapped = _RankerAsEnsemble(ranker)

    print(f"\nRunning walk-forward OOS across thresholds {thresholds} "
          f"(mode=tranche hold=30d, cross_sectional admission) ...")
    # inference_cache: the ranker's per-day scoring is threshold-independent
    # (percentile-rank happens inside predict_proba, before any threshold
    # comparison), so cache it across the sweep exactly like the production
    # sweep loop does — pays the LightGBM scoring cost once, not once/threshold.
    inference_cache: dict = {}
    results = []
    for thr in thresholds:
        train_cfg.signal_threshold = thr  # _build_wf_config reads cfg.signal_threshold — set BEFORE run_oos
        eq = run_oos(
            ds.panel, tabular_features, wrapped, corporate_actions,
            cutoff, train_cfg, mode="tranche", hold_days=30,
            inference_cache=inference_cache,
        )
        m = equity_metrics(eq, train_cfg.initial_capital)
        monthly = monthly_net_sharpe(eq)
        results.append({"threshold": thr, "eq": eq, "monthly": monthly, **m})
        print(f"  thr={thr:.2f}  NetPnL={m['net_pnl']:+,.0f}  "
              f"Sharpe={m['net_sharpe']:+.3f}  DD={m['max_drawdown']*100:.2f}%")

    best = max(results, key=lambda r: r["net_sharpe"])
    daily_r = best["eq"]["daily_return"].to_numpy()
    dsr = deflated_sharpe(daily_r, n_trials=len(thresholds), annualisation=TRADING_DAYS)
    M = pd.concat([r["monthly"] for r in results], axis=1).fillna(0.0).sort_index()
    S = min(4, (len(M) // 2) * 2)
    pbo = cscv_pbo(M.to_numpy(), S=max(2, S)) if len(M) >= 2 else {"pbo": float("nan")}

    print(f"\n{'=' * 76}\nBEST (thr={best['threshold']:.2f}) — RANKER vs KNOWN BASELINE\n{'=' * 76}")
    print(f"  RANKER   : Sharpe={best['net_sharpe']:+.3f}  DD={best['max_drawdown']*100:.2f}%  "
          f"NetPnL={best['net_pnl']:+,.0f}  DSR_p={dsr.get('p_dsr', float('nan')):.3f}  "
          f"PBO={pbo.get('pbo', float('nan')):.1%}")
    print(f"  BASELINE : Sharpe=+0.545  DD=-13.31%  (T+5 GOLDEN, tonight's 20-07 retrain, "
          f"pointwise ensemble — read from logs, not recomputed here)")
    print(f"\nNOTE: this is a SINGLE-SEED, single-ranker first test — not a production "
          f"replacement (no 3-model stack, no PurgedKFold-OOF, no meta-learner). "
          f"Promising here would justify the bigger integration; a loss here kills "
          f"the idea before that investment.")


if __name__ == "__main__":
    main()
