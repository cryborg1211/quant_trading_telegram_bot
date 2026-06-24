"""
scripts/walk_forward_macro_pipeline.py — Rolling-window walk-forward validation
of the GARCH-HMM regime brake for BOTH swing horizons (T+5, T+20).

Design
──────
  • Fixed 504-day (~2y) rolling TRAIN window; refit at every `step_days` stride.
  • For each refit, generate CAUSAL daily signals across the test slice by
    feeding the model the expanding context [window_start : day] and reading
    its leak-free filtered P(Bull) at the last bar (== p_bull_latest).
  • A try/except fallback prevents a single numerically-singular window from
    aborting the whole sweep; fallback days are FLAGGED so the ON/OFF ratio can
    exclude them.

Methodology caveats (read before trusting the numbers)
──────────────────────────────────────────────────────
  1. A 504-day window may not contain BOTH a bull and a bear regime → some
     refits degenerate and fall back. Watch the printed fallback rate.
  2. Refitting every 5/20 days re-runs GARCH+HMM from scratch; `bull_state`
     identity can flip between refits → some signal churn is MODEL instability,
     not real regime change. This is a diagnostic, not a production signal path.

Paths
─────
Spec named data/macro/macro_series.parquet + data/equity/ohlcv_panel.parquet,
which DO NOT exist in this repo. Real layout is data/macro_daily.parquet plus
per-ticker data/ohlcv_*.parquet shards, loaded via the canonical pipeline
helpers. Override with --macro-parquet if needed.

Run
───
    python scripts/walk_forward_macro_pipeline.py
    python scripts/walk_forward_macro_pipeline.py --train-window 504 --threshold 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure repo root on sys.path (avoid 'No module named src') ───────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.pipeline import RunConfig, configure_logging, load_ohlcv  # noqa: E402
from src.models.macro_risk_hmm import build_market_proxy_returns  # noqa: E402
from src.models.garch_hmm_regime import train_garch_hmm  # noqa: E402

LOGGER = logging.getLogger("quant.walk_forward_macro")

_MACRO_COLS = ("sp500_ret", "dxy_ret", "usdvnd_ret")
_HORIZONS = (5, 20)
_OUT_DIR = Path("process/features/macro-integration")


# ── Data assembly ────────────────────────────────────────────────────────

def build_macro_obs(cfg: RunConfig, macro_parquet: Path) -> pd.DataFrame:
    """4-column macro frame: market_ret + sp500/dxy/usdvnd, date-indexed, sorted."""
    panel = load_ohlcv(cfg)
    market_ret = build_market_proxy_returns(panel)

    if not macro_parquet.exists():
        raise FileNotFoundError(
            f"{macro_parquet} missing. Run 'python main.py --task crawl_macro' first.")
    macro = pd.read_parquet(macro_parquet)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    obs = pd.DataFrame({"market_ret": market_ret})
    for c in _MACRO_COLS:
        obs[c] = macro[c].reindex(obs.index).ffill(limit=3)
    return obs.dropna().sort_index()


# ── Per-horizon rolling walk-forward ─────────────────────────────────────

def run_horizon(
    obs: pd.DataFrame,
    *,
    horizon: int,
    train_window: int,
    n_states: int,
    n_restarts: int,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Rolling refit every `horizon` days; emit causal daily brake signals."""
    n = len(obs)
    rows: list[dict] = []
    n_windows = 0
    n_fallback = 0

    for i in range(train_window, n, horizon):
        train = obs.iloc[i - train_window:i]
        test_end = min(i + horizon, n)
        n_windows += 1

        try:
            model = train_garch_hmm(
                train, n_states=n_states, n_restarts=n_restarts, seed=seed)
            ok = True
        except Exception as exc:  # noqa: BLE001 — singular window → neutral fallback
            LOGGER.warning("[T+%d] window @%s fallback: %s",
                           horizon, obs.index[i].date(), exc)
            model = None
            ok = False
            n_fallback += 1

        for j in range(i, test_end):
            day = obs.index[j]
            if ok:
                # CAUSAL: feed expanding context [window_start : day] so GARCH
                # warms up and the filtered HMM posterior peeks no future.
                ctx = obs.iloc[i - train_window:j + 1]
                p_bull = float(model.p_bull_latest(ctx))
                brake = 1.0 if p_bull >= threshold else 0.0   # == exposure_brake row logic
            else:
                p_bull, brake = 0.5, 1.0   # neutral: stay invested, but flagged

            rows.append({
                "date": day,
                "horizon": horizon,
                "p_bull": p_bull,
                "exposure_brake": brake,
                "is_fallback": not ok,
            })

    LOGGER.info("[T+%d] windows=%d  fallback=%d (%.0f%%)  signal_days=%d",
                horizon, n_windows, n_fallback,
                100 * n_fallback / max(n_windows, 1), len(rows))
    return pd.DataFrame(rows)


# ── Summary ──────────────────────────────────────────────────────────────

def _ratio_block(df: pd.DataFrame) -> dict:
    valid = df[~df["is_fallback"]]
    n = len(valid)
    on = float((valid["exposure_brake"] == 1.0).mean()) if n else float("nan")
    return {
        "signal_days": len(df),
        "valid_days": n,
        "fallback_days": int(df["is_fallback"].sum()),
        "brake_on_pct": on * 100,
        "brake_off_pct": (1 - on) * 100,
        "mean_p_bull": float(valid["p_bull"].mean()) if n else float("nan"),
    }


def print_summary(results: dict[int, pd.DataFrame], threshold: float) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  GARCH-HMM WALK-FORWARD | rolling refit | brake threshold={threshold}")
    print(sep)
    print(f"  {'Metric':<22}{'T+5':>18}{'T+20':>18}")
    print(sep)
    b5, b20 = _ratio_block(results[5]), _ratio_block(results[20])
    rows = [
        ("Signal days", "signal_days", "d"),
        ("Valid days", "valid_days", "d"),
        ("Fallback days", "fallback_days", "d"),
        ("Brake ON (invested)", "brake_on_pct", "%"),
        ("Brake OFF (cash)", "brake_off_pct", "%"),
        ("Mean P(Bull)", "mean_p_bull", "f"),
    ]
    for label, key, fmt in rows:
        if fmt == "d":
            print(f"  {label:<22}{b5[key]:>18d}{b20[key]:>18d}")
        elif fmt == "%":
            print(f"  {label:<22}{b5[key]:>17.1f}%{b20[key]:>17.1f}%")
        else:
            print(f"  {label:<22}{b5[key]:>18.4f}{b20[key]:>18.4f}")
    print(f"{sep}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main(
    *,
    macro_parquet: Path,
    train_window: int,
    n_states: int,
    n_restarts: int,
    threshold: float,
    seed: int,
    out_dir: Path,
) -> dict[int, pd.DataFrame]:
    configure_logging()
    t0 = time.perf_counter()

    LOGGER.info("=" * 64)
    LOGGER.info(" WALK-FORWARD MACRO | train_window=%d  horizons=%s  states=%d",
                train_window, list(_HORIZONS), n_states)
    LOGGER.info("=" * 64)

    obs = build_macro_obs(RunConfig(), macro_parquet)
    LOGGER.info("Macro obs: %d days (%s .. %s)",
                len(obs), obs.index.min().date(), obs.index.max().date())
    if len(obs) <= train_window + max(_HORIZONS):
        raise ValueError(
            f"obs ({len(obs)}) too short for train_window={train_window} + horizon")

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, pd.DataFrame] = {}
    for h in _HORIZONS:
        LOGGER.info("Running T+%d rolling walk-forward...", h)
        df = run_horizon(
            obs, horizon=h, train_window=train_window, n_states=n_states,
            n_restarts=n_restarts, threshold=threshold, seed=seed)
        suffix = "t5" if h == 5 else "t20"
        path = out_dir / f"walk_forward_signals_{suffix}.parquet"
        df.to_parquet(path, index=False)
        LOGGER.info("Saved → %s (%d rows)", path, len(df))
        results[h] = df

    print_summary(results, threshold)
    LOGGER.info("Wall-clock: %.1fs", time.perf_counter() - t0)
    return results


def _cli() -> dict:
    p = argparse.ArgumentParser(
        description="Rolling walk-forward validation of the GARCH-HMM brake (T+5, T+20).")
    p.add_argument("--macro-parquet", type=Path, default=Path("data/macro_daily.parquet"))
    p.add_argument("--train-window", type=int, default=504, help="rolling train days (~2y)")
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    a = p.parse_args()
    return {
        "macro_parquet": a.macro_parquet,
        "train_window": a.train_window,
        "n_states": a.n_states,
        "n_restarts": a.n_restarts,
        "threshold": a.threshold,
        "seed": a.seed,
        "out_dir": a.out_dir,
    }


if __name__ == "__main__":
    try:
        main(**_cli())
    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(level=logging.INFO)
        LOGGER.error("FATAL: %s", exc, exc_info=True)
        sys.exit(1)
