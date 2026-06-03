"""Smoke test (synthetic data): drive the V4.0 two-script flow end-to-end.

    PHASE A  train_models.main()  -> v3_training_checkpoint.joblib  (heavy lifter)
    PHASE B  run_backtest.main()  -> threshold sweep + DSR + PBO     (fast evaluator)

Both the checkpoint write and the bot-payload write are redirected to a temp
dir / disabled, so the smoke NEVER touches the live models/saved/ registry.
Green signal = both phases complete without error and the checkpoint round-trips.
"""
import sys, io, os, tempfile, ast
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

for _f in ("src/backtest/pipeline.py", "train_models.py", "run_backtest.py"):
    with open(_f, encoding="utf-8") as _fh:
        ast.parse(_fh.read())
print("AST parse OK (pipeline + train_models + run_backtest)")

import numpy as np
import pandas as pd
import duckdb

import train_models
import run_backtest
from src.backtest.pipeline import RunConfig

rng = np.random.default_rng(0)

with tempfile.TemporaryDirectory() as tmp:
    core = os.path.join(tmp, "core.duckdb")
    # ── Synthetic stock_ohlcv: 18 tickers × ~260 business days ──
    days = pd.bdate_range("2022-01-03", periods=260).date.tolist()
    rows = []
    for i in range(18):
        tk = f"SYN{i:02d}"
        px = 20_000.0 + i * 1500
        for d in days:
            px *= (1 + rng.normal(0.0004, 0.018))
            o = px * (1 + rng.normal(0, 0.002))
            rows.append((tk, d, o, max(o, px) * 1.005, min(o, px) * 0.995, px,
                         int(rng.uniform(500_000, 2_000_000))))
    ohlcv_df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    # ── Synthetic macro_daily ──
    macro_rows = []
    dxy, sp, usd = 100.0, 4000.0, 24000.0
    for d in days:
        dxy *= (1 + rng.normal(0, 0.004)); sp *= (1 + rng.normal(0.0003, 0.01))
        usd *= (1 + rng.normal(0, 0.002))
        macro_rows.append((d, sp, dxy, usd, None, None, 3.5))
    macro_df = pd.DataFrame(macro_rows, columns=[
        "date", "sp500_close", "dxy_close", "usd_vnd", "interbank_on_rate", "vnibor", "inflation_yoy"])

    con = duckdb.connect(core)
    con.execute("CREATE TABLE stock_ohlcv AS SELECT * FROM ohlcv_df")
    con.execute("CREATE TABLE macro_daily AS SELECT * FROM macro_df")
    con.close()
    print(f"Synthetic core DuckDB: {len(ohlcv_df)} OHLCV rows, {len(macro_df)} macro rows")

    # Redirect the checkpoint into the temp dir so we never touch models/saved/.
    ckpt_path = Path(tmp) / "v3_training_checkpoint.joblib"
    train_models.CHECKPOINT_PATH = ckpt_path

    # Tiny TRAIN config — V4.0 defaults (T+20, PT=3.0σ, SL=2.0σ) inherited; only
    # the dataset wiring + a small seed pool are overridden for speed.
    train_cfg = RunConfig(
        bitemporal_duckdb=os.path.join(tmp, "nonexistent.duckdb"),  # force fallback
        core_duckdb=core,
        parquet_glob=os.path.join(tmp, "none_*.parquet"),
        min_history=120,
        train_frac=0.6,
        n_configs=2,         # ≥2 → PBO computable
    )

    print("\n=== PHASE A: train_models.main() (heavy lifter) ===\n")
    train_models.main(train_cfg)
    assert ckpt_path.exists(), f"training checkpoint not written to {ckpt_path}"
    print(f"\n=== checkpoint written OK: {ckpt_path.stat().st_size / 1024:.1f} KB ===")

    print("\n=== PHASE B: run_backtest.main() (fast evaluator) ===\n")
    run_backtest.main(
        ckpt_path,
        eval_overrides={"max_positions": 4, "max_weight": 0.35,
                        "target_vol": 0.30, "cscv_S": 4},
        sweep_thresholds=[0.50, 0.45],
        save_bot_payload=False,   # SMOKE: never write the live bot payload
    )
    print("\n=== run_backtest.main() completed without error ===")

print("\nSMOKE TEST PASSED")
