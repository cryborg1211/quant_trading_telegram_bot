"""T+5/T+20 confluence — BACKTEST-POWERED version (2026-07-22, READ-ONLY).

The live-paperlog version (`analyze_confluence_signal.py`) came back
INCONCLUSIVE: the "both horizons say UP" bucket had ZERO occurrences in
1002 settled rows — the recent defensive stretch never produced the case.
This version gets statistical power the paperlog can't: score the ENTIRE
~4-year historical OOS window (~300K ticker-day rows) with BOTH shipped
GOLDEN serve artifacts and bucket realized forward returns by agreement.

Design notes:
  * Loads the two SERVE bundles (`models/saved/v3_ensemble_{5,20}d.joblib`)
    — each carries its own fitted TabularEnsemble + its own feature list
    (the pools differ slightly per horizon), exactly what production serves.
  * One `materialize_dataset` pass builds the full feature pool; each
    artifact's matrix is sliced by ITS OWN feature list via column indices
    (no `subset_features` — that mutates the shared AlignedData in place,
    which would destroy the pool for the second artifact).
  * OOS discipline: both artifacts were trained with a chronological cutoff
    of 2022-11-11/14 (17-07-26 retrain logs); rows scored here are
    `>= ds.cutoff` (the re-materialized boundary, within days of both) —
    genuinely out-of-sample for both models' weights.
  * Outcomes = RAW forward returns from panel closes (5 and 20 TRADING bars
    ahead via per-ticker shift), the same quantity `sentiment_entry_paperlog`
    settles with (`ret_20d`) — directly comparable to the live version's
    buckets. Rows lacking a full forward window are dropped.
  * Decisions = argmax over each model's 3-class output (same encoding as
    the paperlog / arbitrator: 0=DOWN 1=SIDE 2=UP), plus a second,
    serve-relevant gate test: p_up >= each artifact's own up_threshold.

Zero writes, zero model/artifact changes.

Run: python scripts/analyze_confluence_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.backtest.pipeline import RunConfig, materialize_dataset  # noqa: E402

MIN_RELIABLE_N = 30  # bucket reliability floor (bigger than live version — we have volume)
BUNDLES = {
    5: REPO / "models" / "saved" / "v3_ensemble_5d.joblib",
    20: REPO / "models" / "saved" / "v3_ensemble_20d.joblib",
}


def _score_horizon(bundle_path: Path, X_full: np.ndarray,
                   all_features: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """(decisions argmax {0,1,2}, p_up, up_threshold) for one artifact."""
    bundle = joblib.load(bundle_path)
    ens = bundle["ensemble"]
    feats = list(bundle["tabular_features"])
    up_thr = float(bundle["up_threshold"])
    idx = [all_features.index(f) for f in feats]
    X = X_full[:, idx]
    proba3 = ens.predict_proba_3class(X)          # (n, 3) in fixed {0,1,2} layout
    decisions = np.argmax(proba3, axis=1).astype(np.int64)
    p_up = proba3[:, 2].astype(np.float64)
    print(f"  {bundle_path.name}: features={len(feats)}  up_thr={up_thr:.2f}  "
          f"decision mix: DOWN={np.mean(decisions == 0):.1%} "
          f"SIDE={np.mean(decisions == 1):.1%} UP={np.mean(decisions == 2):.1%}")
    return decisions, p_up, up_thr


def _fwd_returns(panel: pl.DataFrame, bars: int) -> pl.DataFrame:
    return (
        panel.lazy()
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close").shift(-bars).over("ticker") / pl.col("close") - 1.0)
            .alias(f"fwd_{bars}")
        )
        .select(["ticker", "date", f"fwd_{bars}"])
        .collect()
    )


def _bucket(label: str, rets: np.ndarray) -> None:
    n = len(rets)
    if n == 0:
        print(f"{label:<40}{'(empty)':>8}")
        return
    note = "" if n >= MIN_RELIABLE_N else f"<{MIN_RELIABLE_N}, noisy"
    print(f"{label:<40}{n:>8}{rets.mean() * 100:>10.2f}%"
          f"{float((rets > 0).mean()):>10.1%}  {note}")


def main() -> None:
    print("Materializing dataset (one pass, full feature pool) ...")
    ds = materialize_dataset(RunConfig())
    aligned = ds.aligned
    oos = ~ds.train_mask
    X_oos = aligned.X[oos]
    dates_oos = aligned.dates[oos]
    tickers_oos = aligned.tickers[oos]
    print(f"OOS rows={len(dates_oos)}  cutoff={ds.cutoff}  "
          f"range={dates_oos.min()}..{dates_oos.max()}")

    print("Scoring both GOLDEN artifacts on the SAME OOS cross-section ...")
    d5, pup5, thr5 = _score_horizon(BUNDLES[5], X_oos, ds.all_features)
    d20, pup20, thr20 = _score_horizon(BUNDLES[20], X_oos, ds.all_features)

    print("Computing realized forward returns from panel closes ...")
    f20 = _fwd_returns(ds.panel, 20)
    key_to_ret20: dict[tuple, float] = {
        (t, d): r for t, d, r in f20.iter_rows() if r is not None
    }
    ret20 = np.array(
        [key_to_ret20.get((t, d), np.nan) for t, d in zip(tickers_oos, dates_oos)],
        dtype=np.float64,
    )
    have = np.isfinite(ret20)
    print(f"Rows with a full 20-bar forward window: {have.sum()} / {len(ret20)}")

    d5, d20, pup5, pup20, ret20 = d5[have], d20[have], pup5[have], pup20[have], ret20[have]

    print(f"\n{'=' * 84}\nARGMAX-DECISION BUCKETS (mirrors the live paperlog version)\n{'=' * 84}")
    print(f"{'bucket':<40}{'n':>8}{'mean ret20':>11}{'hit rate':>10}  note")
    _bucket("Confluence UP (5d=UP, 20d=UP)", ret20[(d5 == 2) & (d20 == 2)])
    _bucket("Confluence DOWN (5d=DN, 20d=DN)", ret20[(d5 == 0) & (d20 == 0)])
    _bucket("Divergent (5d=UP, 20d=DN)", ret20[(d5 == 2) & (d20 == 0)])
    _bucket("Divergent (5d=DN, 20d=UP)", ret20[(d5 == 0) & (d20 == 2)])
    _bucket("5d=UP, 20d=SIDE", ret20[(d5 == 2) & (d20 == 1)])
    _bucket("ALL rows (base rate)", ret20)

    print(f"\n{'=' * 84}\nTHE DIRECT TEST (argmax): 20d confirmation on top of 5d=UP\n{'=' * 84}")
    up5 = d5 == 2
    _bucket("5d=UP, ANY 20d", ret20[up5])
    _bucket("5d=UP AND 20d=UP (confluence-gated)", ret20[up5 & (d20 == 2)])
    _bucket("5d=UP but 20d != UP (filtered out)", ret20[up5 & (d20 != 2)])

    print(f"\n{'=' * 84}\nSERVE-GATE TEST: p_up >= each artifact's own up_threshold "
          f"({thr5:.2f} / {thr20:.2f})\n{'=' * 84}")
    g5 = pup5 >= thr5
    g20 = pup20 >= thr20
    _bucket("5d gate PASS, ANY 20d", ret20[g5])
    _bucket("BOTH gates PASS (confluence)", ret20[g5 & g20])
    _bucket("5d PASS but 20d FAIL (filtered out)", ret20[g5 & ~g20])
    _bucket("20d gate PASS, ANY 5d", ret20[g20])
    _bucket("20d PASS but 5d FAIL", ret20[g20 & ~g5])

    print("\nDone. Forward returns are GROSS (no costs) — bucket COMPARISONS are "
          "the signal, absolute levels are not tradable numbers.")


if __name__ == "__main__":
    main()
