"""Fast-attack (impulse continuation) research — tail label + volume filter.

READ-ONLY research (2026-07-22, user-requested "fast attack" direction):
can a dedicated sub-model catch momentum IGNITION the way MR-LGBM catches
capitulation? Mirrors `train_mr_lgbm.py`'s exact validated scaffold —
purged OOF (embargo = label horizon), strict-τ selection, chronological
1-year holdout — swapping only the feature set and the label:

LABEL (tail event, attack side — the mirror of `label_3d_bounce`):
    y = 1  IFF
        (1) 3-bar fwd return > +3%          [the continuation], AND
        (2) an IMPULSE setup fired at t     [the ignition]:
              (imp_ret1_z > 2.0  AND imp_vol_z > 1.0)     big day + volume
           OR (imp_brk20_dist > 0 AND imp_vol_z > 2.0)    breakout + heavy volume
    The setup gate REQUIRES volume confirmation in both branches (the
    user's "filter by volume" — participation-less spikes never qualify).
    3-bar horizon = the T+2.5-settlement practical minimum, same as MR.

Zero model artifacts, zero serve wiring — a verdict script. Ship-worthy
only if OOF precision clears the same 0.60 bar MR-LGBM is held to.

Run: python scripts/analyze_impulse_attack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.impulse_features import (  # noqa: E402
    IMPULSE_FEATURE_COLUMNS,
    build_impulse_features,
)
from src.models.train_mr_lgbm import (  # noqa: E402
    BOUNCE_THRESHOLD,
    EMBARGO_BARS,
    HORIZON,
    N_SPLITS,
    _spw,
    chrono_split,
    load_ohlcv,
    make_lgbm,
    purged_oof,
    select_strict_tau,
)

SHIPPED_MR_TAU_REF = 0.96  # informational reference only


def label_3d_continuation(df: pd.DataFrame) -> pd.DataFrame:
    """OUTCOME ∧ IMPULSE-SETUP-at-t — structural mirror of label_3d_bounce.

    Requires build_impulse_features() to have run. Rows without a full
    +HORIZON-bar window are dropped (can't label / purge).
    """
    need = {"imp_ret1_z", "imp_vol_z", "imp_brk20_dist"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"label_3d_continuation needs impulse columns {sorted(need - set(df.columns))}"
            " — call build_impulse_features(df) BEFORE labeling."
        )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False, group_keys=False)
    fwd_close = g["close"].shift(-HORIZON)
    df["target_return_3d"] = fwd_close / df["close"] - 1.0
    df["t1"] = g["date"].shift(-HORIZON)
    df = df.dropna(subset=["target_return_3d", "t1"]).reset_index(drop=True)

    ret_z = pd.to_numeric(df["imp_ret1_z"], errors="coerce")
    vol_z = pd.to_numeric(df["imp_vol_z"], errors="coerce")
    brk = pd.to_numeric(df["imp_brk20_dist"], errors="coerce")
    impulse = (
        ((ret_z > 2.0) & (vol_z > 1.0)).fillna(False)
        | ((brk > 0.0) & (vol_z > 2.0)).fillna(False)
    )

    continuation = df["target_return_3d"] > BOUNCE_THRESHOLD
    df["y"] = (continuation & impulse).astype(np.int8)
    df["_impulse_setup"] = impulse.astype(np.int8)
    return df


def main() -> None:
    print("Loading OHLCV ...")
    ohlcv = load_ohlcv()

    print("Building impulse features + continuation label ...")
    feat = build_impulse_features(ohlcv)
    feat = label_3d_continuation(feat)

    setup_rate = feat["_impulse_setup"].mean()
    cont_given_setup = feat.loc[feat["_impulse_setup"] == 1, "target_return_3d"]
    print(f"Impulse-setup rate: {setup_rate:.3%} of all rows")
    print(f"P(fwd3 > +{BOUNCE_THRESHOLD:.0%} | setup): "
          f"{(cont_given_setup > BOUNCE_THRESHOLD).mean():.3f}  "
          f"(n={len(cont_given_setup)}, mean fwd3={cont_given_setup.mean() * 100:+.2f}%)")
    base_all = (feat["target_return_3d"] > BOUNCE_THRESHOLD).mean()
    print(f"P(fwd3 > +{BOUNCE_THRESHOLD:.0%}) unconditional: {base_all:.3f}")

    train, test = chrono_split(feat)
    cols = list(IMPULSE_FEATURE_COLUMNS)

    def X(d: pd.DataFrame) -> np.ndarray:
        return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

    train = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    x_tr, y_tr = X(train), train["y"].to_numpy(np.int64)
    start = pd.to_datetime(train["date"]).to_numpy()
    end = pd.to_datetime(train["t1"]).to_numpy()

    print(f"\nRunning purged OOF ({N_SPLITS} folds, embargo={EMBARGO_BARS}) — "
          f"identical machinery to MR-LGBM ...")
    oof = purged_oof(x_tr, y_tr, start, end)
    tau, tau_info = select_strict_tau(oof, y_tr)

    print("\nFit FINAL LGBM on full train, score STRICT 1-year holdout ...")
    final = make_lgbm(_spw(y_tr))
    final.fit(x_tr, y_tr)
    x_te, y_te = X(test), test["y"].to_numpy(np.int64)
    p_te = final.predict_proba(x_te)[:, 1]
    fire_te = p_te >= tau
    n_fire = int(fire_te.sum())
    tp = int((fire_te & (y_te == 1)).sum())
    prec_te = tp / n_fire if n_fire else float("nan")
    rec_te = tp / max(int(y_te.sum()), 1)

    print(f"\n{'=' * 72}\nFAST-ATTACK VERDICT (vs MR-LGBM's own bar)\n{'=' * 72}")
    print(f"  OOF     : tau*={tau:.2f}  precision={tau_info['oof_precision']:.3f}  "
          f"recall={tau_info['oof_recall']:.3f}  fires={tau_info['oof_fires']}  "
          f"target(0.60)_met={tau_info['target_precision_met']}")
    print(f"  HOLDOUT : fires={n_fire}  precision={prec_te:.3f}  recall={rec_te:.3f}  "
          f"(pos_base_rate={y_te.mean():.3%})")
    print(f"  MR ref  : shipped tau*={SHIPPED_MR_TAU_REF:.2f}, OOF precision 0.578, "
          f"holdout 0.647 (17-07 retrain) — the incumbent tail sleeve this "
          f"must at least match to earn a place.")
    print("\nNo artifacts written — research verdict only.")


if __name__ == "__main__":
    main()
