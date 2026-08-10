"""Multi-shard OHLCV readers must tolerate per-file dtype divergence (10-08-26).

WHY THIS FILE EXISTS
────────────────────
`data/ohlcv_*.parquet` is written by two sources: vnstock hands back int64
`volume`, SSI FastConnect float64. After one EOD crawl the 359 shards held two
different schemas (345 double / 15 int64). Every reader that unifies schemas
across files is an independent crash site, and polars unifies on the FIRST
shard's schema — so the failure depends on glob order and looks random.

This bit twice in one day. The first fix (ded6e4c) patched only
`alpha360_generator`, and `backtest.pipeline.load_ohlcv` then failed the same
way — taking down all three exposure-brake legs at once, silently, because the
meta-controller fails open to 1.0 (full exposure) on a panel-load error. So the
contract is pinned here for every reader rather than per-site.

The 15 int64 shards belong to inactive tickers that will never be rewritten,
so canonicalizing on write is NOT sufficient. Readers must cope.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _write_mixed_shards(directory: Path, n_float: int = 3, n_int: int = 2) -> None:
    """Shards identical except for `volume`'s dtype, named so the float ones
    sort FIRST — that is the order in which polars picks the target schema and
    then chokes on the int64 arrivals."""
    from src.backtest.pipeline import RunConfig

    directory.mkdir(parents=True, exist_ok=True)
    # Long enough to clear `_post_ohlcv`'s min_history filter, read from the
    # config rather than hardcoded — otherwise every ticker is dropped and the
    # tests pass for the wrong reason (no SchemaError, but nothing loaded).
    rows = int(RunConfig().min_history) + 20
    start = date.today() - timedelta(days=rows + 5)

    def frame(ticker: str, vol_dtype) -> pl.DataFrame:
        return pl.DataFrame({
            "ticker": [ticker] * rows,
            "date": [start + timedelta(days=i) for i in range(rows)],
            "open": [10.0 + i * 0.1 for i in range(rows)],
            "high": [10.5 + i * 0.1 for i in range(rows)],
            "low": [9.5 + i * 0.1 for i in range(rows)],
            "close": [10.2 + i * 0.1 for i in range(rows)],
            "volume": pl.Series([1000 + i for i in range(rows)], dtype=vol_dtype),
            "adj_close": [10.2 + i * 0.1 for i in range(rows)],
        })

    for i in range(n_float):
        frame(f"AA{i}", pl.Float64).write_parquet(directory / f"ohlcv_AA{i}.parquet")
    for i in range(n_int):
        frame(f"ZZ{i}", pl.Int64).write_parquet(directory / f"ohlcv_ZZ{i}.parquet")


@pytest.fixture()
def shard_dir(tmp_path, monkeypatch):
    """Mixed-dtype shards under a RELATIVE glob.

    Both readers call `Path().glob(pattern)`, which rejects absolute patterns,
    so the test has to cd into a sandbox rather than pass a tmp_path glob.
    """
    _write_mixed_shards(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    return "data/ohlcv_*.parquet"


def test_the_mismatch_this_guards_is_real(tmp_path):
    """A plain multi-file scan really does raise — otherwise the tests below
    would pass for the wrong reason if polars ever changed its defaults."""
    _write_mixed_shards(tmp_path)
    files = sorted(str(p) for p in tmp_path.glob("ohlcv_*.parquet"))
    with pytest.raises(pl.exceptions.SchemaError):
        pl.scan_parquet(files).select(["ticker", "volume"]).collect()


def test_backtest_pipeline_load_ohlcv_tolerates_mixed_dtypes(shard_dir):
    """`load_ohlcv` feeds backtests, training AND the live exposure brake.
    When it raised, all three brake legs silently read 1.0 = full exposure."""
    from src.backtest.pipeline import RunConfig, load_ohlcv

    cfg = RunConfig()
    cfg.parquet_glob = shard_dir
    df = load_ohlcv(cfg)

    assert df.height > 0
    assert df.schema["volume"] == pl.Float64
    assert df["ticker"].n_unique() == 5      # no shard silently dropped


def test_train_mr_lgbm_load_ohlcv_tolerates_mixed_dtypes(shard_dir, monkeypatch):
    """This one runs inside the UNATTENDED weekly retrain (Sat 09:00), so a
    mismatch here kills the retrain with nobody watching."""
    from src.models import train_mr_lgbm

    monkeypatch.setattr(train_mr_lgbm, "OHLCV_GLOB", shard_dir)
    df = train_mr_lgbm.load_ohlcv()

    assert len(df) > 0
    assert str(df["volume"].dtype) == "float64"
    assert df["ticker"].nunique() == 5


def test_readers_agree_on_the_canonical_volume_dtype(shard_dir, monkeypatch):
    """Both readers must land on the SAME dtype, or features computed by one
    path and consumed by the other diverge on train/serve parity."""
    from src.backtest.pipeline import RunConfig, load_ohlcv
    from src.models import train_mr_lgbm

    cfg = RunConfig()
    cfg.parquet_glob = shard_dir
    monkeypatch.setattr(train_mr_lgbm, "OHLCV_GLOB", shard_dir)

    assert load_ohlcv(cfg).schema["volume"] == pl.Float64
    assert str(train_mr_lgbm.load_ohlcv()["volume"].dtype) == "float64"


def test_all_int64_shards_still_load(tmp_path, monkeypatch):
    """The pre-FastConnect world: every shard int64. Must keep working, or the
    fix would have traded one breakage for another."""
    from src.backtest.pipeline import RunConfig, load_ohlcv

    _write_mixed_shards(tmp_path / "data", n_float=0, n_int=4)
    monkeypatch.chdir(tmp_path)
    cfg = RunConfig()
    cfg.parquet_glob = "data/ohlcv_*.parquet"
    df = load_ohlcv(cfg)
    assert df.height > 0 and df.schema["volume"] == pl.Float64
