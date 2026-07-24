"""DSR/PBO calibration diagnostic (22-07-26, idea #3 of the 4-idea batch).

Every experiment run tonight died on the Sharpe<0.95 deflated hurdle (DSR)
or PBO overfit — nobody targeted the STAT GATE itself. Two READ-ONLY
diagnostics, both reusing the T+20 GOLDEN artifact (no retrain, no
multi-seed sweep — cheap):

  1. SWEEP-CLONE CHECK — production's own DSR call
     (`run_backtest.py::main`, n_trials = len(sweep_thresholds) * n_seeds)
     counts every swept (threshold, seed) pair as an independent trial. If
     several thresholds produce near-identical books (the sweep-plateau
     artifact already documented for PBO — max_positions=10 saturation),
     the DSR multiplicity penalty is inflated by trials that bought no real
     search diversity. Scores the CURRENT production sweep grid
     [0.50,0.48,0.46,0.44,0.42] ONCE (shared inference cache — oracle
     scoring is threshold-independent) and reports pairwise correlation of
     each threshold's daily-return series.

  2. BLOCK-BOOTSTRAP SHARPE/DSR STABILITY — resamples the GOLDEN's own OOS
     daily-return series in contiguous blocks (preserves autocorrelation,
     unlike i.i.d. resampling) and recomputes DSR at the artifact's OWN
     real n_trials (read from its persisted metadata, not assumed) on each
     redraw. Answers: is the DSR failure ROBUST (nearly every resample
     fails) or BORDERLINE (a meaningful fraction would pass) — i.e. is this
     a real edge-too-weak problem or a small-sample measurement-noise one.

Neither diagnostic touches the production DSR/PBO FORMULA — read-only
measurement, not a proposed fix. A clear finding either way is actionable:
sweep-clones -> trim the grid; borderline bootstrap -> more OOS data (time)
will likely close the gap; robust-fail bootstrap -> stop chasing sizing/
admission knobs, the edge itself needs to be bigger.

Run: python scripts/analyze_dsr_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_backtest import (  # noqa: E402
    DEFAULT_SWEEP_THRESHOLDS,
    TRADING_DAYS,
    _apply_eval_overrides,
    equity_metrics,
    run_oos,
)
from src.backtest.pipeline import (  # noqa: E402
    RunConfig,
    load_corporate_actions,
    materialize_dataset,
)
from src.models.statistical_gates import deflated_sharpe

N_BOOTSTRAP = 2000
BLOCK_LEN = 20   # ~1 trading month, preserves short-run autocorrelation
RNG_SEED = 42


def main() -> None:
    print("Loading T+20 GOLDEN serve bundle v3_ensemble_20d.joblib ...")
    bundle = joblib.load(REPO / "models" / "saved" / "v3_ensemble_20d.joblib")
    ensemble = bundle["ensemble"]
    tabular_features = list(bundle["tabular_features"])
    sig_thr = float(bundle["signal_threshold"])
    golden_thr = float(bundle["up_threshold"])
    n_seeds = int(bundle["metadata"].get("n_seeds_in_golden", 1))
    n_trials_prod = len(DEFAULT_SWEEP_THRESHOLDS) * max(1, n_seeds)
    print(f"  golden up_thr={golden_thr:.2f}  n_seeds_in_golden={n_seeds}  "
          f"production n_trials={n_trials_prod}")

    cfg = RunConfig()
    cfg = _apply_eval_overrides(cfg, {})

    print("Materializing dataset ...")
    ds = materialize_dataset(cfg)
    corporate_actions = load_corporate_actions(cfg)
    cutoff = ds.cutoff

    print(f"\n{'=' * 80}\nDIAGNOSTIC 1 — sweep-clone check "
          f"({len(DEFAULT_SWEEP_THRESHOLDS)} thresholds, ONE artifact, "
          f"shared inference cache)\n{'=' * 80}")
    cache: dict = {}
    daily_returns: dict[float, pd.Series] = {}
    for thr in DEFAULT_SWEEP_THRESHOLDS:
        cfg.signal_threshold = sig_thr
        eq = run_oos(ds.panel, tabular_features, ensemble, corporate_actions,
                     cutoff, cfg, mode="tranche", hold_days=30,
                     inference_cache=cache, admission_mode="absolute_gate",
                     admission_floor=thr)
        m = equity_metrics(eq, cfg.initial_capital)
        daily_returns[thr] = pd.Series(
            eq["daily_return"].to_numpy(), index=pd.to_datetime(eq["date"]))
        print(f"  thr={thr:.2f}  NetPnL={m['net_pnl']:+,.0f}  "
              f"Sharpe={m['net_sharpe']:+.3f}  DD={m['max_drawdown'] * 100:.2f}%")

    corr = pd.DataFrame(daily_returns).corr()
    print("\nPairwise daily-return correlation across swept thresholds:")
    print(corr.round(3).to_string())
    off_diag = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    mean_corr = float(np.nanmean(off_diag))
    print(f"\nMean off-diagonal correlation: {mean_corr:.3f}  "
          f"(>=0.95 ~ near-clone configs sharing one effective trial; "
          f"<0.7 ~ genuinely diverse search)")

    print(f"\n{'=' * 80}\nDIAGNOSTIC 2 — block-bootstrap DSR stability "
          f"(GOLDEN thr={golden_thr:.2f}, {N_BOOTSTRAP} resamples, "
          f"block={BLOCK_LEN}d, n_trials={n_trials_prod})\n{'=' * 80}")
    golden_r = daily_returns[golden_thr].dropna().to_numpy()
    n = len(golden_r)
    rng = np.random.default_rng(RNG_SEED)
    n_blocks = int(np.ceil(n / BLOCK_LEN))

    sharpes = np.empty(N_BOOTSTRAP)
    dsr_pass = np.zeros(N_BOOTSTRAP, dtype=bool)
    for i in range(N_BOOTSTRAP):
        starts = rng.integers(0, max(1, n - BLOCK_LEN), size=n_blocks)
        sample = np.concatenate([golden_r[s:s + BLOCK_LEN] for s in starts])[:n]
        mu, sd = sample.mean(), sample.std(ddof=1)
        sharpes[i] = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
        dsr = deflated_sharpe(sample, n_trials=n_trials_prod, annualisation=TRADING_DAYS)
        dsr_pass[i] = dsr.get("p_dsr", 0.0) >= 0.95

    print(f"  Original GOLDEN daily series: n={n} days")
    print(f"  Bootstrap annualized Sharpe: mean={sharpes.mean():+.3f}  "
          f"p5={np.percentile(sharpes, 5):+.3f}  p50={np.percentile(sharpes, 50):+.3f}  "
          f"p95={np.percentile(sharpes, 95):+.3f}")
    pass_frac = float(dsr_pass.mean())
    print(f"  Fraction of resamples where DSR PASSES (p_dsr>=0.95): {pass_frac:.1%}")

    print(f"\n{'=' * 80}\nVERDICT\n{'=' * 80}")
    if mean_corr >= 0.95:
        print("  Sweep grid is near-clone across thresholds — n_trials multiplicity "
              "is inflated by configs with no real search diversity. Trimming the "
              "grid (fewer, more-different thresholds) would lower the DSR penalty "
              "honestly.")
    else:
        print("  Sweep grid thresholds are NOT near-clones — the current n_trials "
              "multiplicity count is methodologically fair, not artificially "
              "inflating the DSR penalty.")
    if pass_frac >= 0.20:
        print(f"  DSR failure looks BORDERLINE — {pass_frac:.0%} of resampled "
              "histories would pass. More OOS data (time, not more knobs) is likely "
              "to close the gap.")
    else:
        print(f"  DSR failure looks ROBUST — only {pass_frac:.0%} of resampled "
              "histories pass. The edge itself needs to be bigger; further sizing/"
              "admission knobs are unlikely to fix this.")

    print("\nNo artifacts written, no formula changed — read-only diagnostic.")


if __name__ == "__main__":
    main()
