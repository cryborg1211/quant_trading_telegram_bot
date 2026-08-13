"""Read-only drift + sampling-bias monitor (2026-07-20).

Answers, on demand, the questions a stale model hides:
  1. PREDICTION drift — is serve-time P(UP) distributed like the OOS backtest
     reference? (features are cross-sectionally z-scored per date, so classic
     feature-PSI is self-normalized away — the model OUTPUT is where drift
     shows.)
  2. LABEL base-rate drift — trailing realized UP rate vs the train-period
     rate (sampling-bias check: is the world the model was fit on still the
     world it trades in?).
  3. REGIME mix drift — defensive-share trend (NO_TRADE {0,7} + PENALTY {1,6}).

Zero writes. Reads sentiment_entry_paperlog (read-only DuckDB) + OHLCV
parquets. Run:  python scripts/check_drift.py [--days 20]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

# OOS reference from the 17-07-26 T+20 retrain teardown (calibrated P(UP)).
REF_PUP = {"p10": 0.321, "median": 0.408, "p90": 0.463}
# Train-period true-UP base rate (T+20 labels, same teardown).
REF_UP_BASE_RATE = 0.415
# Fallback ONLY. The strategy admits on an ABSOLUTE threshold, so the statistic
# that decides whether it ever fires is the UPPER TAIL, not the median — a
# distribution can hold its centre while its p90 sinks below tau, which shuts the
# gate with no median drift to warn about. Observed 10-08-26: median shift only
# -0.015 ("OK") while p90 fell 0.463 -> 0.423, i.e. below tau.
_FALLBACK_TAU_T20 = 0.46
DB = REPO / "data" / "quant_v6_core.duckdb"
DATA = REPO / "data"


def live_tau_t20() -> tuple[float, str]:
    """Read the T+20 gate from the LIVE artifact. Never pin it.

    This was a hardcoded `REF_TAU_T20 = 0.46` until 13-08-26, and it silently went
    wrong the moment an artifact was promoted: the 13-08 retrain moved the gate to
    0.43, and the monitor kept reporting "GATE STARVED ... below tau=0.46" against
    a threshold that no longer existed.

    The distinction that was missed: `REF_PUP` genuinely IS a frozen reference —
    you compare today's serve scores against the distribution the model was
    validated on, and you re-pin it after a retrain. `tau` is NOT a reference. It
    is the operative gate, it lives in the artifact, and pinning it makes the
    monitor describe a system that is not deployed.
    """
    try:
        import joblib  # noqa: PLC0415

        b = joblib.load(DATA.parent / "models" / "saved" / "v3_ensemble_20d.joblib")
        tau = float(b["up_threshold"])
        stamped = (b.get("metadata") or {}).get("trained_at") or "?"
        return tau, f"live artifact (trained_at={stamped})"
    except Exception as exc:  # noqa: BLE001 — a monitor must not die on a bad read
        return _FALLBACK_TAU_T20, f"FALLBACK {_FALLBACK_TAU_T20:.2f} (artifact unreadable: {exc})"


def _warn_if_scores_predate_artifact(days: int) -> None:
    """Flag the window where a fresh gate is being judged by the old model's scores.

    The paperlog is written by the daily cron using whatever artifact was live that
    day. Right after a promotion the whole window predates the new model, so both
    the p90 and the %-above-tau describe the OUTGOING model. Without this note the
    output reads as a current verdict on the new gate.
    """
    try:
        import joblib  # noqa: PLC0415
        from datetime import datetime as _dt  # noqa: PLC0415

        b = joblib.load(DATA.parent / "models" / "saved" / "v3_ensemble_20d.joblib")
        raw = str(b.get("trained_at") or (b.get("metadata") or {}).get("trained_at") or "")
        if not raw:
            return
        trained = _dt.fromisoformat(raw.replace("Z", "+00:00")).date()
        age_days = (date.today() - trained).days
        if age_days < days:
            print(f"  ⚠️  ARTIFACT IS {age_days}d OLD but this window is {days}d — "
                  f"the paperlog rows above were scored by the PREVIOUS model, so "
                  f"the tail-vs-gate verdict describes the OUTGOING gate, not the "
                  f"new one. Re-run after ~{days - age_days} more daily crons.")
    except Exception:  # noqa: BLE001 — advisory only
        return


def prediction_drift(days: int) -> None:
    print(f"\n── 1. Prediction drift (paperlog, last {days}d vs OOS reference) ──")
    with duckdb.connect(str(DB), read_only=True) as conn:
        # REF_PUP is the T+20 teardown reference, so this must read the T+20
        # probability. It previously read `p_up_20d`, which despite its name
        # held the SECONDARY (T+5) probability — the monitor compared T+5
        # serve output against a T+20 reference and reported "OK" (fixed
        # 10-08-26 along with the column rename). `source='daily'` is required
        # because /verify rows have T+5 as their PRIMARY.
        rows = conn.execute(
            """
            SELECT p_up_primary FROM sentiment_entry_paperlog
            WHERE log_date >= current_date - ? * INTERVAL '1 day'
              AND p_up_primary IS NOT NULL
              AND source = 'daily'
              AND COALESCE(primary_horizon_days, 20) = 20
            """,
            [days],
        ).fetchall()
    if len(rows) < 30:
        print(f"  only {len(rows)} rows — need >=30 for a read; wait for the "
              "un-starved paperlog to accumulate (fix shipped 20-07).")
        return
    p = np.array([r[0] for r in rows], dtype=float)
    p10, med, p90 = np.percentile(p, [10, 50, 90])
    print(f"  serve  p10={p10:.3f} median={med:.3f} p90={p90:.3f}  (n={len(p)})")
    print(f"  OOSref p10={REF_PUP['p10']:.3f} median={REF_PUP['median']:.3f} "
          f"p90={REF_PUP['p90']:.3f}")
    shift = med - REF_PUP["median"]
    flag = "OK" if abs(shift) < 0.03 else ("⚠️ DRIFT" if abs(shift) < 0.06 else "🔴 SEVERE")
    print(f"  median shift {shift:+.3f} → {flag}"
          f"{'  (retrain or recalibrate before trusting the gate)' if flag != 'OK' else ''}")

    # Tail-vs-gate check. This is the one that actually predicts whether the
    # strategy can trade at all, and it is independent of the median.
    tau, tau_src = live_tau_t20()
    tail_shift = p90 - REF_PUP["p90"]
    pct_above = float((p >= tau).mean()) * 100.0
    print(f"  p90 shift {tail_shift:+.3f}  |  {pct_above:.1f}% of scores reach "
          f"tau={tau:.2f}  [{tau_src}]")
    if p90 < tau:
        print(f"  🔴 GATE STARVED: p90={p90:.3f} is BELOW tau={tau:.2f} — "
              f"the top decile of the model's own output no longer clears its "
              f"gate, so the book stays empty regardless of signal quality. "
              f"Recalibrate tau on the TRADEABLE universe, or widen the "
              f"universe, before reading anything into the low dispatch count.")
    elif pct_above < 2.0:
        print(f"  ⚠️  gate rarely reachable ({pct_above:.1f}% of scores) — "
              f"expect very few dispatches; too few bets to evaluate.")

    # AFTER A RETRAIN this comparison is stale in a second, subtler way: the
    # paperlog rows above were scored by the PREVIOUS model, so a new tau is being
    # judged against an old score distribution. Say so rather than let the verdict
    # read as current.
    _warn_if_scores_predate_artifact(days)


def label_base_rate(days: int) -> None:
    print(f"\n── 2. Label base-rate drift (realized 20d fwd return > 0) ──")
    files = sorted(DATA.glob("ohlcv_*.parquet"))
    if not files:
        print("  no OHLCV parquets found.")
        return
    ups = total = 0
    cutoff = date.today() - timedelta(days=days + 40)
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["date", "close"]).sort("date")
        except Exception:
            continue
        closes = df["close"].to_list()
        dates = df["date"].to_list()
        for i in range(len(closes) - 20):
            d = dates[i]
            d = d.date() if hasattr(d, "date") else d
            if d < cutoff:
                continue
            if closes[i] and closes[i] > 0:
                total += 1
                ups += closes[i + 20] > closes[i]
    if total < 500:
        print(f"  only {total} matured windows — inconclusive.")
        return
    rate = ups / total
    shift = rate - REF_UP_BASE_RATE
    flag = "OK" if abs(shift) < 0.05 else ("⚠️ SHIFT" if abs(shift) < 0.10 else "🔴 SEVERE")
    print(f"  trailing UP rate {rate:.3f} vs train {REF_UP_BASE_RATE:.3f} "
          f"(Δ{shift:+.3f}, n={total}) → {flag}")


def regime_mix() -> None:
    print("\n── 3. Regime defensive-share (from today's serve log lines) ──")
    log = REPO / "logs" / "quant_v6.log"
    if not log.exists():
        print("  logs/quant_v6.log absent.")
        return
    lines = [ln for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines()
             if "NO_TRADE {0,7}" in ln]
    for ln in lines[-5:]:
        print("  " + ln.split(" — ")[-1][:150])
    if not lines:
        print("  no regime pulse lines found.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    a = ap.parse_args()
    print("DRIFT / SAMPLING-BIAS MONITOR (read-only)")
    prediction_drift(a.days)
    label_base_rate(a.days)
    regime_mix()
    print("\nReference constants pinned to the 17-07-26 retrain teardown — "
          "update REF_* after every retrain.")


if __name__ == "__main__":
    main()
