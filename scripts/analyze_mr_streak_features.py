"""Do streak features improve the MR knife-catch model? (09-08-26)

WHY MR-LGBM AND NOT THE T+20 PRIMARY
------------------------------------
`scripts/analyze_streak_features.py` cleared streaks through both gates but
located the signal at the SHORT horizon: overlap-corrected t-stats of 7-13
at T+5, collapsing to 0.67-2.34 at T+20. Signs are internally consistent
(up-streak negative, down-streak positive, streak_return/intensity most
negative) = short-horizon MEAN REVERSION. That is MR-LGBMs own thesis (3-day
bounce label), and MR-LGBM sits at OOF precision 0.578, under its own 0.60
target. So this is where run-length information belongs.

ARMS (identical scaffold to train_mr_lgbm: purged OOF, embargo=3, strict-tau
selection, strict chronological hold-out -- nothing about the validation
machinery changes between arms):
  1. BASELINE          11 mr_* features            (reproduces the shipped model)
  2. +STREAK           11 + 5 stk_*
  3. +CONTEXT+STREAK   11 + 6 mrctx_* + 5 stk_*    (stacks the two survivors)

Arm 3 exists because MR context features (volume-exhaustion +
sector-relative oversold) already reached OOF 0.642 on this same scaffold
but were left PROMISING-UNCONFIRMED on a thin hold-out. If run-length is
genuinely orthogonal to those too, the combination is the interesting one.

Ship-worthy only if OOF precision clears the models own 0.60 bar AND the
hold-out has enough fires to be believable -- the exact standard that kept
MR context features out of serve.

Zero artifacts, zero serve wiring. Run:
  python scripts/analyze_mr_streak_features.py
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
from src.features.streak_features import (  # noqa: E402
    STREAK_FEATURE_COLUMNS,
    build_streak_features,
)
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

SHIPPED_OOF_REF = 0.578
SHIPPED_HOLDOUT_REF = 0.647


def eval_arm(name: str, cols: list[str],
             train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float, int]:
    def X(d: pd.DataFrame) -> np.ndarray:
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    # purged_oof needs chronologically-contiguous folds. The feature builders
    # leave the frame [ticker, date]-sorted; without this re-sort the folds
    # degenerate into near-empty ticker blocks (this exact bug produced a
    # bogus OOF 0.340 in the MR-context run before it was caught).
    train = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_tr, y_tr = X(train), train["y"].to_numpy(np.int64)
    start = pd.to_datetime(train["date"]).to_numpy()
    end = pd.to_datetime(train["t1"]).to_numpy()

    print(f"\nPurged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS}) — {name} "
          f"({len(cols)} features) ...")
    oof = purged_oof(x_tr, y_tr, start, end)
    tau, info = select_strict_tau(oof, y_tr)

    final = make_lgbm(_spw(y_tr))
    final.fit(x_tr, y_tr)
    x_te, y_te = X(test), test["y"].to_numpy(np.int64)
    fire = final.predict_proba(x_te)[:, 1] >= tau
    n_fire = int(fire.sum())
    tp = int((fire & (y_te == 1)).sum())
    prec = tp / n_fire if n_fire else float("nan")

    print(f"  [{name}] tau*={tau:.2f}  OOF precision={info['oof_precision']:.3f}  "
          f"recall={info['oof_recall']:.3f}  fires={info['oof_fires']}  "
          f"target(0.60)_met={info['target_precision_met']}")
    print(f"  [{name}] HOLDOUT fires={n_fire}  precision={prec:.3f}")
    return float(info["oof_precision"]), prec, n_fire


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building MR + context + streak features ...")
    feat = build_mr_features(ohlcv)
    feat = build_mr_context_features(feat)
    feat = build_streak_features(feat)
    feat = label_3d_bounce(feat)

    train, test = chrono_split(feat)

    base = list(MR_FEATURE_COLUMNS)
    arms = [
        ("BASELINE", base),
        ("+STREAK", base + list(STREAK_FEATURE_COLUMNS)),
        ("+CONTEXT+STREAK",
         base + list(MR_CONTEXT_FEATURE_COLUMNS) + list(STREAK_FEATURE_COLUMNS)),
    ]

    results = []
    for name, cols in arms:
        results.append((name, len(cols), *eval_arm(name, cols, train, test)))

    print(f"\n{'=' * 78}\nMR + STREAK VERDICT\n{'=' * 78}")
    print(f"  shipped reference (17-07 retrain): OOF {SHIPPED_OOF_REF:.3f}  "
          f"holdout {SHIPPED_HOLDOUT_REF:.3f}")
    print(f"\n{'arm':<20}{'n_feat':>8}{'OOF prec':>10}{'holdout':>10}{'h_fires':>9}")
    for name, nf, oof_p, hold_p, hf in results:
        print(f"{name:<20}{nf:>8}{oof_p:>10.3f}{hold_p:>10.3f}{hf:>9}")

    base_oof = results[0][2]
    print(f"\n  OOF delta vs BASELINE:")
    for name, _nf, oof_p, _hp, _hf in results[1:]:
        print(f"    {name:<18} {oof_p - base_oof:+.3f}")

    print("\n  Bar to clear: OOF >= 0.60 AND enough holdout fires to believe it.")
    print("  MR context features already hit OOF 0.642 but were left unconfirmed")
    print("  on a 5-fire hold-out — a thin hold-out here means the same verdict.")
    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
