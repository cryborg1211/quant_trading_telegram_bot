"""Meta-labelling (AFML 3.6) on the T+20 primary -- 09-08-26.

THE IDEA
--------
The primary model decides the SIDE (which names look UP). A secondary binary
model decides WHETHER TO BET, trained only on the primary's own bets with
label = "did that bet actually win". It adds no new data -- it re-uses the
primary's output plus market state -- so it attacks PRECISION, which is this
system's measured weakness (UP-precision ~0.55). V4 has no meta-labeller;
the pre-V4 stacking architecture had one (`meta_labeler.joblib`) and it was
dropped in the rewrite.

Why it might NOT be enough, stated up front: better precision usually means
FEWER bets, and fewer bets shortens the effective sample, which hurts the
sqrt(T) term in DSR. Precision and DSR can move in opposite directions here.
That is exactly what the backtest arms below measure.

LEAK-FREE DESIGN (and its cost)
-------------------------------
Training a meta-labeller properly needs leak-free PRIMARY predictions on the
primary's own training set, i.e. its purged OOF matrix. `TabularEnsemble`
computes that internally but does not store it (only `oof_meta_feature_names`
survives), and recomputing it means refitting 4 seeds x 5 folds (~70 min).

So this test uses the cheap, honest alternative: the primary never saw ANY
OOS row, so its OOS predictions are already leak-free. The OOS window is
split chronologically -- meta trains on the FIRST half, and is evaluated on
the SECOND half, which neither model has seen. Cost: the meta gets ~half the
OOS window to learn from and the verdict rests on the other half. This is a
feasibility read, not a production fit; if it works, redo it with the full
purged-OOF version.

ARMS COMPARED (on the meta-test window only)
  A. primary alone                    -- current production behaviour
  B. primary gated by the meta-labeller

Run: python scripts/analyze_meta_labelling.py [checkpoint_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from run_backtest import (  # noqa: E402
    TRADING_DAYS,
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    load_corporate_actions,
    materialize_dataset,
    subset_features,
)
from src.models.macro_risk_hmm import build_regime_observation  # noqa: E402
from src.models.statistical_gates import deflated_sharpe  # noqa: E402
from src.models.tabular_ensemble import UP_CLASS  # noqa: E402

DEFAULT_CKPT = REPO / "models" / "saved" / "t20_std_checkpoint.joblib"

# Market-state columns fed to the meta ALONGSIDE the primary's output. Kept
# deliberately small: handing the meta the full 13-feature vector invites it
# to just re-learn the primary instead of learning WHEN the primary is right.
META_CONTEXT_COLS = ["market_regime", "vol_squeeze_xsz",
                     "amihud_liquidity_xsz", "mom20_xsz"]
META_TAUS = (0.30, 0.40, 0.50, 0.60, 0.70)


class SeedEnsemble:
    """Average P(UP) and the 3-class distribution across every seed."""

    def __init__(self, ensembles) -> None:
        self._e = list(ensembles)
        self.feature_names = list(self._e[0].feature_names)

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([np.asarray(e.predict_proba(X)).ravel() for e in self._e], axis=0)

    def predict_proba_3class(self, X) -> np.ndarray:
        return np.mean([np.asarray(e.predict_proba_3class(X)) for e in self._e], axis=0)


def build_meta_features(p3: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """[p_down, p_side, p_up, conviction, spread] + market-state context."""
    p_dn, p_sd, p_up = p3[:, 0], p3[:, 1], p3[:, 2]
    conviction = p_up - np.maximum(p_sd, p_dn)
    spread = p_up - p_dn
    core = np.column_stack([p_dn, p_sd, p_up, conviction, spread])
    return np.hstack([core, ctx]) if ctx.size else core


class MetaGatedModel:
    """Primary, vetoed by the meta-labeller. predict_proba returns the
    primary's P(UP) where BOTH gates agree and 0.0 otherwise -- 0.0 sits
    below every admission floor, so a veto simply removes the name."""

    def __init__(self, primary, meta, ctx_idx: list[int],
                 tau_primary: float, tau_meta: float) -> None:
        self.primary, self.meta = primary, meta
        self.ctx_idx, self.tau_primary, self.tau_meta = ctx_idx, tau_primary, tau_meta
        self.feature_names = list(primary.feature_names)

    def predict_proba(self, X) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        p3 = self.primary.predict_proba_3class(Xa)
        p_up = p3[:, UP_CLASS]
        ctx = Xa[:, self.ctx_idx] if self.ctx_idx else np.empty((len(Xa), 0))
        p_win = self.meta.predict_proba(build_meta_features(p3, ctx))[:, 1]
        return np.where((p_up >= self.tau_primary) & (p_win >= self.tau_meta), p_up, 0.0)


def main() -> None:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading {ckpt_path.name} ...")
    ckpt = joblib.load(ckpt_path)
    feats = list(ckpt["tabular_features"])
    trained = list(ckpt["ensembles"])
    macro_hmm = ckpt.get("macro_hmm")
    cutoff = ckpt["cutoff"]
    train_cfg = ckpt["train_cfg"]
    print(f"  tb_horizon={train_cfg.tb_horizon}  train_frac={train_cfg.train_frac}  "
          f"cutoff={cutoff}  seeds={[s for s, _ in trained]}")

    cfg = _apply_eval_overrides(train_cfg, {})
    primary = SeedEnsemble([e for _, e in trained])

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)
    ds.aligned = subset_features(ds.aligned, ds.all_features, feats)

    dates = pd.to_datetime(ds.aligned.dates)
    oos = dates >= pd.Timestamp(cutoff)
    X_oos = ds.aligned.X[oos]
    y_oos = ds.aligned.y[oos]
    d_oos = dates[oos]
    print(f"OOS rows: {len(X_oos)}  dates {d_oos.min().date()} .. {d_oos.max().date()}")

    # Chronological split of the OOS window.
    uniq = np.array(sorted(set(d_oos)))
    mid = uniq[len(uniq) // 2]
    tr_m, te_m = d_oos < mid, d_oos >= mid
    print(f"  meta-TRAIN: {tr_m.sum()} rows (< {pd.Timestamp(mid).date()})")
    print(f"  meta-TEST : {te_m.sum()} rows (>= {pd.Timestamp(mid).date()})")

    print("\nScoring the primary over OOS (seed-averaged) ...")
    p3 = primary.predict_proba_3class(X_oos)
    p_up = p3[:, UP_CLASS]

    tau_p = float(ckpt.get("up_threshold") or cfg.signal_threshold or 0.46)
    print(f"  primary tau (up_threshold) = {tau_p:.2f}")

    ctx_idx = [feats.index(c) for c in META_CONTEXT_COLS if c in feats]
    print(f"  meta context cols: {[feats[i] for i in ctx_idx]}")
    ctx = X_oos[:, ctx_idx] if ctx_idx else np.empty((len(X_oos), 0))
    MF = build_meta_features(p3, ctx)

    # Meta trains ONLY on the primary's own bets (AFML 3.6).
    bet_tr = tr_m & (p_up >= tau_p)
    if bet_tr.sum() < 200:
        print(f"\nOnly {bet_tr.sum()} primary bets in the meta-train half — too few to fit. "
              "Lower tau or widen the window.")
        return
    y_meta_tr = (y_oos[bet_tr] == UP_CLASS).astype(int)
    print(f"\nMeta-train bets: {bet_tr.sum()}  win rate: {y_meta_tr.mean():.3f}")

    meta = LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.03,
        max_depth=3, num_leaves=7, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    meta.fit(MF[bet_tr], y_meta_tr)

    # ── Precision on the held-out half ───────────────────────────────────
    bet_te = te_m & (p_up >= tau_p)
    y_te = (y_oos[bet_te] == UP_CLASS).astype(int)
    p_win_te = meta.predict_proba(MF[bet_te])[:, 1]
    print(f"\n{'=' * 78}\nPRECISION on meta-TEST (primary bets only)\n{'=' * 78}")
    print(f"  primary alone : n={bet_te.sum():>6}  precision={y_te.mean():.4f}")
    for tm in META_TAUS:
        keep = p_win_te >= tm
        if keep.sum() < 20:
            print(f"  meta>={tm:.2f}     : n={keep.sum():>6}  (too few)")
            continue
        print(f"  meta>={tm:.2f}     : n={keep.sum():>6}  precision={y_te[keep].mean():.4f}"
              f"   kept={keep.mean():.1%}")

    # ── The real test: does it improve the BOOK, not just precision? ─────
    print(f"\n{'=' * 78}\nBACKTEST on the meta-test window (cutoff={pd.Timestamp(mid).date()})\n{'=' * 78}")
    p_bull = None
    if macro_hmm is not None:
        try:
            obs = build_regime_observation(
                ds.panel, use_macro=cfg.use_macro_in_hmm, macro_parquet=cfg.macro_parquet)
            p_bull = macro_hmm.p_bull_series(obs, filtered=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (p_bull unavailable: {exc})")

    mid_date = pd.Timestamp(mid).date()
    arms: list[tuple[str, object]] = [("A: primary alone", primary)]
    for tm in (0.40, 0.50, 0.60):
        arms.append((f"B: primary + meta>={tm:.2f}",
                     MetaGatedModel(primary, meta, ctx_idx, tau_p, tm)))

    cache: dict = {}
    print(f"{'arm':<26}{'NetPnL':>18}{'Sharpe':>9}{'MaxDD':>9}{'DSR p':>8}")
    for label, model in arms:
        cfg.signal_threshold = tau_p - 0.05
        try:
            eq = run_oos(ds.panel, feats, model, corporate_actions, mid_date, cfg,
                         p_bull_series=p_bull,
                         inference_cache=cache if model is primary else None,
                         mode="tranche", hold_days=30)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<26}FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        m = equity_metrics(eq, cfg.initial_capital)
        dsr = deflated_sharpe(eq["daily_return"].to_numpy(),
                              n_trials=len(META_TAUS), annualisation=TRADING_DAYS)
        print(f"{label:<26}{m['net_pnl']:>+18,.0f}{m['net_sharpe']:>+9.3f}"
              f"{m['max_drawdown'] * 100:>8.2f}%{dsr.get('p_dsr', float('nan')):>8.3f}")

    print("\nRead it this way: arm B is only interesting if Sharpe RISES while")
    print("drawdown does not worsen. Higher precision with fewer trades and a")
    print("flat-or-lower Sharpe means the meta is just shrinking the book.")
    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
