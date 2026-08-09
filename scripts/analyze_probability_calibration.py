"""Is the model's P(UP) honest? Calibration check on live paperlog data
(09-08-26).

WHY THIS MATTERS MOST FOR A HUMAN-IN-THE-LOOP PRODUCT
-----------------------------------------------------
If the operator sizes a position off the displayed probability, that number
has to mean what it says. A model claiming 55% that really wins 45% of the
time does not just mis-forecast -- it makes the operator confident in
exactly the wrong places. `TabularEnsemble` already wraps the meta-LR in
CalibratedClassifierCV, but that calibration was fitted on the TRAINING
OOF matrix and has never been verified against live outcomes.

DATA: `sentiment_entry_paperlog`, settled rows only (outcome_filled=TRUE).
Every row holds the model's own p_up_5d / p_up_20d as displayed at
decision time plus the realised ret_3d / ret_20d -- a genuine
prediction-vs-outcome record, not a backtest reconstruction.

LABEL MISMATCH -- stated up front, it bounds what this can prove
---------------------------------------------------------------
The model outputs P(triple-barrier class UP) = P(3sigma profit target is hit
BEFORE the 2sigma stop). The paperlog stores a simple realised return. So
this measures calibration against "did it go up", NOT against the event the
model was trained on. Those differ: a model perfectly calibrated for its own
barrier label can look miscalibrated here. `ret_20d > 0` is the metric an
OPERATOR actually cares about, so it is the right thing to report -- but a
gap shown here is not automatically proof the training-time calibration is
broken.

METRICS
  * Reliability table  -- predicted bin vs realised hit rate (with per-bin n,
    so thin high-probability bins are visible rather than hidden)
  * Brier score        -- overall probabilistic accuracy (lower is better)
  * ECE                -- size-weighted mean |predicted - realised|
  * Platt slope        -- logistic refit of outcome on logit(p). slope < 1
    means OVER-confident (probabilities too extreme), > 1 under-confident

READ-ONLY. Run: python scripts/analyze_probability_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.data.db_engine import DuckDBEngine  # noqa: E402

MIN_BIN = 10          # below this a bin's hit rate is noise; flagged not hidden


def _reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> None:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    print(f"  {'bin':<14}{'n':>7}{'pred':>9}{'actual':>9}{'gap':>9}")
    ece, total = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        n = int(m.sum())
        pred, act = float(p[m].mean()), float(y[m].mean())
        gap = act - pred
        ece += (n / total) * abs(gap)
        flag = "" if n >= MIN_BIN else "  (thin)"
        print(f"  [{lo:.1f},{hi:.1f})    {n:>7}{pred:>9.3f}{act:>9.3f}{gap:>+9.3f}{flag}")
    print(f"\n  ECE (size-weighted mean |gap|): {ece:.4f}")


def _platt(p: np.ndarray, y: np.ndarray) -> None:
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) < 2:
        print("  Platt: outcome has a single class — cannot fit")
        return
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    verdict = ("OVER-confident (probs too extreme)" if slope < 0.9
               else "UNDER-confident (probs too timid)" if slope > 1.1
               else "slope near 1 — shape is about right")
    print(f"  Platt slope={slope:+.3f}  intercept={intercept:+.3f}   -> {verdict}")


def _report(name: str, p: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{'=' * 74}\n{name}   n={len(p)}\n{'=' * 74}")
    if len(p) < 30:
        print("  too few settled rows to say anything")
        return
    print(f"  mean predicted P(UP) : {p.mean():.4f}")
    print(f"  realised hit rate    : {y.mean():.4f}")
    print(f"  Brier score          : {np.mean((p - y) ** 2):.4f}  "
          f"(baseline always-predict-base-rate = {np.mean((y.mean() - y) ** 2):.4f})")
    _platt(p, y)
    print()
    _reliability(p, y)


def main() -> None:
    con = DuckDBEngine().get_connection()
    rows = con.execute("""
        SELECT p_up_20d, ret_20d, p_up_5d, ret_3d,
               COALESCE(final_decision, decision_5d) AS dec
        FROM sentiment_entry_paperlog
        WHERE outcome_filled AND p_up_20d IS NOT NULL AND ret_20d IS NOT NULL
    """).fetchall()
    if not rows:
        print("No settled paperlog rows with a stored p_up_20d — nothing to check.")
        return

    p20 = np.array([float(r[0]) for r in rows])
    r20 = np.array([float(r[1]) for r in rows])
    dec = np.array([r[4] if r[4] is not None else -1 for r in rows])
    y20 = (r20 > 0).astype(int)

    print(f"Settled paperlog rows: {len(rows)}")
    print(f"  decision mix: " + ", ".join(
        f"{d}={int((dec == d).sum())}" for d in sorted(set(dec.tolist()))))
    print("  (0=SELL/EXIT, 1=HOLD, 2=BUY per the arbitrator encoding)")

    _report("T+20 P(UP) vs realised ret_20d > 0  — ALL settled rows", p20, y20)

    # The BUY slice is what an operator acts on; report it separately even
    # though it is thin, because an aggregate dominated by 1000+ SELL rows
    # says nothing about the probabilities that actually trigger a trade.
    buy = dec == 2
    if buy.sum() >= 30:
        _report("T+20 P(UP) — BUY decisions only", p20[buy], y20[buy])
    else:
        print(f"\n{'=' * 74}\nBUY-only slice: n={int(buy.sum())} — too thin to calibrate")
        print(f"{'=' * 74}")
        if buy.sum():
            print(f"  mean predicted {p20[buy].mean():.4f} vs realised hit rate "
                  f"{y20[buy].mean():.4f}  (n={int(buy.sum())}, treat as anecdote)")

    # Short horizon, if stored (drives /verify).
    p5 = np.array([float(r[2]) for r in rows if r[2] is not None and r[3] is not None])
    r3 = np.array([float(r[3]) for r in rows if r[2] is not None and r[3] is not None])
    if len(p5) >= 30:
        _report("T+5 P(UP) vs realised ret_3d > 0", p5, (r3 > 0).astype(int))

    print("\nHow to act on this:")
    print("  * slope < 0.9 with a positive ECE -> probabilities are too extreme;")
    print("    a Platt/isotonic recalibration on live outcomes fixes the NUMBER")
    print("    without retraining the model.")
    print("  * a large negative gap in the HIGH bins is the dangerous case: it")
    print("    means confident BUY calls are the least trustworthy ones.")
    print("\nNo artifacts written — read-only diagnostic.")


if __name__ == "__main__":
    main()
