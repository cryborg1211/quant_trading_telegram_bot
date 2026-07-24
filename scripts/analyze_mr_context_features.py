"""MR context features research — volume-exhaustion + sector-relative
oversold vs the shipped MR-LGBM feature set (22-07-26 knife-catch follow-up).

Round 2 of the regime-gate research REJECTED market_regime and raw
RSI-direction as additive to MR-LGBM's own oversold features — both were
redundant "how stretched is price" information. Stated path forward: NEW
features. This script tests two NEW families (`src/features/
mr_context_features.py`): volume-exhaustion (is capitulation volume fading
or surging) and sector-relative oversold (worse than sector peers, or a
whole-sector selloff). Mirrors `train_mr_lgbm.py`'s exact validated scaffold
(purged OOF, embargo=3, strict-tau selection, 1-year chrono holdout) and the
SAME `label_3d_bounce` label the shipped model uses — apples-to-apples,
BASELINE (11 cols) vs EXTENDED (11 + 6 context cols).

Zero model artifacts, zero serve wiring — a verdict script. Ship-worthy
only if the extended set's OOF precision clears the baseline's by a real
margin (not noise) at the same 0.60 target.

Run: python scripts/analyze_mr_context_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.mr_context_features import (  # noqa: E402
    MR_CONTEXT_FEATURE_COLUMNS,
    build_mr_context_features,
)
from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features  # noqa: E402
from src.models.train_mr_lgbm import (  # noqa: E402
    EMBARGO_BARS,
    N_SPLITS,
    _spw,
    chrono_split,
    label_3d_bounce,
    load_ohlcv,
    make_lgbm,
    purged_oof,
    select_strict_tau,
)

SHIPPED_MR_OOF_REF = 0.578
SHIPPED_MR_HOLDOUT_REF = 0.647


def _eval_feature_set(
    name: str, cols: list[str], train: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, float, int]:
    def X(d: pd.DataFrame) -> np.ndarray:
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    # purged_oof needs chronologically-contiguous folds. `train` inherits
    # build_mr_features'/build_mr_context_features' [ticker, date] sort
    # (ticker-primary) — re-sort [date, ticker] first, same corrective step
    # analyze_impulse_attack.py uses, or folds silently degenerate into
    # near-empty ticker-blocks instead of real time slices.
    train = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_tr, y_tr = X(train), train["y"].to_numpy(np.int64)
    start = pd.to_datetime(train["date"]).to_numpy()
    end = pd.to_datetime(train["t1"]).to_numpy()

    print(f"\nRunning purged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS}) — "
          f"{name} ({len(cols)} features) ...")
    oof = purged_oof(x_tr, y_tr, start, end)
    tau, tau_info = select_strict_tau(oof, y_tr)

    final = make_lgbm(_spw(y_tr))
    final.fit(x_tr, y_tr)
    x_te, y_te = X(test), test["y"].to_numpy(np.int64)
    p_te = final.predict_proba(x_te)[:, 1]
    fire_te = p_te >= tau
    n_fire = int(fire_te.sum())
    tp = int((fire_te & (y_te == 1)).sum())
    prec_te = tp / n_fire if n_fire else float("nan")
    rec_te = tp / max(int(y_te.sum()), 1)

    print(f"  [{name}] tau*={tau:.2f}  OOF precision={tau_info['oof_precision']:.3f}  "
          f"recall={tau_info['oof_recall']:.3f}  fires={tau_info['oof_fires']}  "
          f"target(0.60)_met={tau_info['target_precision_met']}")
    print(f"  [{name}] HOLDOUT fires={n_fire}  precision={prec_te:.3f}  recall={rec_te:.3f}")
    return float(tau_info["oof_precision"]), prec_te, n_fire


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building MR + context features ...")
    feat = build_mr_features(ohlcv)
    feat = build_mr_context_features(feat)

    sect_coverage = feat["mrctx_sect_rsi_rel"].notna().mean()
    print(f"Sector-relative coverage: {sect_coverage:.1%} of rows "
          f"(rest = OTHER/thin-sector, NaN — LightGBM handles natively)")

    feat = label_3d_bounce(feat)
    train, test = chrono_split(feat)

    baseline_cols = list(MR_FEATURE_COLUMNS)
    extended_cols = list(MR_FEATURE_COLUMNS) + list(MR_CONTEXT_FEATURE_COLUMNS)

    base_oof, base_hold, base_n = _eval_feature_set(
        "BASELINE", baseline_cols, train, test)
    ext_oof, ext_hold, ext_n = _eval_feature_set(
        "EXTENDED (+context)", extended_cols, train, test)

    print(f"\n{'=' * 72}\nMR CONTEXT FEATURES VERDICT (vs shipped MR-LGBM's own bar)\n{'=' * 72}")
    print(f"  shipped MR ref (17-07 retrain): OOF {SHIPPED_MR_OOF_REF:.3f}, "
          f"holdout {SHIPPED_MR_HOLDOUT_REF:.3f}")
    print(f"  BASELINE (11 cols, reproduced): OOF {base_oof:.3f}, "
          f"holdout {base_hold:.3f}  (holdout fires={base_n})")
    print(f"  EXTENDED (17 cols, +context):   OOF {ext_oof:.3f}, "
          f"holdout {ext_hold:.3f}  (holdout fires={ext_n})")
    print(f"\n  OOF precision delta (extended - baseline): {ext_oof - base_oof:+.3f}")
    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
