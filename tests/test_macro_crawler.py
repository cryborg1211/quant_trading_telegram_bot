"""Unit tests for the macro daily crawler (Macro-Integration P1).

The yfinance network call is mocked at ``macro_crawler._fetch_one`` so these
tests need neither the network nor the ``yfinance`` package. Cover: join schema,
return math, single-symbol degradation, parquet build, and idempotent +
incremental updates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import macro_crawler

_BASE = {"^GSPC": 4000.0, "DX-Y.NYB": 100.0, "VND=X": 24000.0}


def _series(symbol: str, n: int = 6, start: str = "2023-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    base = _BASE[symbol]
    return pd.Series([base + i for i in range(n)], index=idx, name="Close")


def _fake_fetch(symbol: str, start: str, end: str | None) -> pd.Series:
    return _series(symbol)


def _patch(monkeypatch, fn=_fake_fetch) -> None:
    monkeypatch.setattr(macro_crawler, "_fetch_one", fn)


# --------------------------------------------------------------------------- #
# fetch_macro_history
# --------------------------------------------------------------------------- #
def test_history_schema(monkeypatch) -> None:
    _patch(monkeypatch)
    out = macro_crawler.fetch_macro_history()
    assert {"date", "sp500", "dxy", "usdvnd",
            "sp500_ret", "dxy_ret", "usdvnd_ret"}.issubset(out.columns)
    assert len(out) == 6


def test_history_return_math(monkeypatch) -> None:
    _patch(monkeypatch)
    out = macro_crawler.fetch_macro_history()
    # sp500 goes 4000, 4001, ... → first ret NaN, second = 1/4000.
    assert np.isnan(out["sp500_ret"].iloc[0])
    assert out["sp500_ret"].iloc[1] == pytest.approx(1.0 / 4000.0)


def test_single_symbol_failure_degrades_to_nan(monkeypatch) -> None:
    def _flaky(symbol, start, end):
        if symbol == "DX-Y.NYB":
            raise RuntimeError("feed down")
        return _series(symbol)

    _patch(monkeypatch, _flaky)
    out = macro_crawler.fetch_macro_history()  # must not raise
    assert out["dxy"].isna().all()
    assert out["sp500"].notna().any()
    assert out["usdvnd"].notna().any()


def test_history_ffill_limit(monkeypatch) -> None:
    # sp500 has a 1-row gap → ffilled (limit 3); a >3 gap would stay NaN.
    def _gappy(symbol, start, end):
        s = _series(symbol)
        if symbol == "^GSPC":
            s.iloc[2] = np.nan
        return s

    _patch(monkeypatch, _gappy)
    out = macro_crawler.fetch_macro_history()
    assert out["sp500"].iloc[2] == pytest.approx(4001.0)  # carried from row 1


# --------------------------------------------------------------------------- #
# update_macro_daily
# --------------------------------------------------------------------------- #
def test_update_writes_parquet(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch)
    p = tmp_path / "macro_daily.parquet"
    n = macro_crawler.update_macro_daily(p)
    assert n == 6
    assert p.exists()
    df = pd.read_parquet(p)
    assert {"date", "sp500", "dxy", "usdvnd", "sp500_ret"}.issubset(df.columns)
    assert len(df) == 6


def test_update_is_idempotent(monkeypatch, tmp_path) -> None:
    _patch(monkeypatch)
    p = tmp_path / "macro_daily.parquet"
    macro_crawler.update_macro_daily(p)
    n2 = macro_crawler.update_macro_daily(p)  # same dates → dedup, no growth
    assert n2 == 6
    assert len(pd.read_parquet(p)) == 6


def test_update_incremental_merges_new_dates(monkeypatch, tmp_path) -> None:
    p = tmp_path / "macro_daily.parquet"
    # First vintage: 6 days from 2023-01-02.
    _patch(monkeypatch, lambda s, st, e: _series(s, n=6, start="2023-01-02"))
    macro_crawler.update_macro_daily(p)
    # Later vintage: a fresh window that overlaps + extends.
    _patch(monkeypatch, lambda s, st, e: _series(s, n=6, start="2023-01-06"))
    macro_crawler.update_macro_daily(p)
    df = pd.read_parquet(p)
    # Union of 2023-01-02..09 business days (overlap deduped) → strictly > 6 rows.
    assert len(df) > 6
    assert df["date"].is_monotonic_increasing
    assert df["date"].duplicated().sum() == 0
