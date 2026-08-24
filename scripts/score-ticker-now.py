"""Score tickers with the CURRENT artifact, without touching DuckDB.

Deliberately DB-free: the bot holds an exclusive DuckDB lock while it runs
(src/data/db_engine.py's singleton), so any diagnostic that opens the core DB
forces the bot down. This reads parquet + the joblib artifact only.

    python scripts/score-ticker-now.py SSI GMD PNJ
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from main import predict_v3_horizon
from src.features.alpha360_generator import Alpha360Generator

WANT = [t.upper() for t in sys.argv[1:]] or ["SSI"]

_win = Alpha360Generator().load_live_ohlcv_window(window_rows=120)
latest_df = _win.to_pandas() if hasattr(_win, "to_pandas") else _win
print(f"OHLCV window loaded: rows={len(latest_df)}  "
      f"max_date={latest_df['date'].max()}")

preds, thr, _bot, _feats, gate = predict_v3_horizon(latest_df, 20)
tau = float(thr.get("pnl_threshold_tau"))
print(f"live T+20 tau = {tau}\n")

print(f"{'ticker':>7} {'p_down':>8} {'p_flat':>8} {'p_up':>8}  gate  vs0.42  vs0.43")
print("-" * 60)
for t in WANT:
    p = preds.get(t)
    if not p:
        print(f"{t:>7}   (not in the scored universe)")
        continue
    print(f"{t:>7} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f}  "
          f"{'PASS' if gate.get(t) else 'fail':>4}  "
          f"{'PASS' if p[2] >= 0.42 else 'fail':>6}  "
          f"{'PASS' if p[2] >= 0.43 else 'fail':>6}")

clearing = sorted(((t, p[2]) for t, p in preds.items() if p[2] >= tau),
                  key=lambda kv: -kv[1])
print(f"\ntickers clearing tau={tau} across the WHOLE scored set: {len(clearing)}")
for t, p in clearing[:15]:
    print(f"   {t:<6} {p:.4f}")
