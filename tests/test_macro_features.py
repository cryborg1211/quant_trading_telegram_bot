"""Tests for the flagged GBM macro-feature variant (Macro-Integration P3).

The load-bearing invariant: with the flag OFF the feature recipe is BYTE-IDENTICAL
to the baseline (`v2-sha8:53b5bd85`) — a true no-op, so the serve path and all
existing artifacts are untouched. With the flag ON the recipe hash changes (so a
macro-trained artifact is rejected by the price-only serve tripwire) and the
macro columns are appended before the categorical.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pandas as pd
import pytest

from src.backtest.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_RECIPE_VERSION,
    FEATURE_SCHEMA,
    _MACRO_FEATURE_NAMES,
    _join_macro_features,
    effective_feature_schema,
    effective_recipe_version,
)

_BASELINE_HASH = "v2-sha8:53b5bd85"


# --------------------------------------------------------------------------- #
# Recipe-hash discipline
# --------------------------------------------------------------------------- #
def test_baseline_recipe_is_unchanged() -> None:
    # P3 must NOT disturb the shipped baseline recipe.
    assert FEATURE_RECIPE_VERSION == _BASELINE_HASH


def test_recipe_off_is_byte_identical_to_baseline() -> None:
    assert effective_recipe_version(False) == FEATURE_RECIPE_VERSION


def test_recipe_on_differs_from_baseline() -> None:
    assert effective_recipe_version(True) != FEATURE_RECIPE_VERSION


# --------------------------------------------------------------------------- #
# Effective schema
# --------------------------------------------------------------------------- #
def test_schema_off_is_baseline() -> None:
    assert effective_feature_schema(False) == FEATURE_SCHEMA


def test_schema_on_inserts_macro_before_categorical() -> None:
    sch = effective_feature_schema(True)
    names = [n for n, _ in sch]
    # Categorical stays last; macro dims sit between continuous and categorical.
    assert names[-len(CATEGORICAL_FEATURES):] == CATEGORICAL_FEATURES
    for m in _MACRO_FEATURE_NAMES:
        assert names.index(m) < names.index(CATEGORICAL_FEATURES[0])
    assert len(sch) == len(FEATURE_SCHEMA) + len(_MACRO_FEATURE_NAMES)
    # macro dims are continuous Float32.
    dtypes = dict(sch)
    assert all(dtypes[m] == "Float32" for m in _MACRO_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# _join_macro_features
# --------------------------------------------------------------------------- #
def _macro_parquet(tmp_path, dates) -> str:
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "sp500": 4000.0, "dxy": 100.0, "usdvnd": 24000.0,
        "sp500_ret": [0.01, 0.02], "dxy_ret": [0.001, 0.002], "usdvnd_ret": [0.0, 0.001],
    })
    p = tmp_path / "macro.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def test_join_broadcasts_macro_across_tickers(tmp_path) -> None:
    d1, d2 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    df = pl.DataFrame({
        "date": [d1, d2, d1, d2],
        "ticker": ["A", "A", "B", "B"],
        "close": [1.0, 2.0, 3.0, 4.0],
    })
    out = _join_macro_features(df, _macro_parquet(tmp_path, [d1, d2]))
    assert set(_MACRO_FEATURE_NAMES).issubset(out.columns)
    # Market-level: A and B on the same date carry the SAME macro value.
    a = out.filter((pl.col("ticker") == "A") & (pl.col("date") == d1))["sp500_ret"][0]
    b = out.filter((pl.col("ticker") == "B") & (pl.col("date") == d1))["sp500_ret"][0]
    assert a == b == pytest.approx(0.01)
    assert out.schema["sp500_ret"] == pl.Float32
    assert out.height == df.height  # left join keeps every panel row


def test_join_raises_when_parquet_missing(tmp_path) -> None:
    df = pl.DataFrame({"date": [dt.date(2024, 1, 2)], "ticker": ["A"], "close": [1.0]})
    with pytest.raises(FileNotFoundError):
        _join_macro_features(df, tmp_path / "nope.parquet")
