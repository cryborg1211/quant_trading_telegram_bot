"""Tests for the rule-based 8-market-regime classifier + regime-aware sizing.

Covers the parts that are cheap + deterministic to assert:
  • build_regime_features → valid, non-null, Int8 0..7; scratch columns dropped
  • a clear strong-uptrend yields some Regime 3 rows
  • leak-free: mutating the FUTURE must not move a PAST regime
  • the Vietnamese label map + safe lookup
  • suggested_weight() structural regime overrides
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.features.market_regime import (
    build_regime_features,
    REGIME_LABELS_VI,
    regime_label_vi,
)
from src.bot.sizing import suggested_weight, REGIME_PENALTY_CAP


def _synthetic(n: int = 200, seed: int = 0) -> pl.DataFrame:
    """Calm → strong uptrend → sharp reversal, so several regimes appear."""
    rng = np.random.default_rng(seed)
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    px = [100.0]
    for i in range(1, n):
        if i < 70:
            px.append(px[-1] * (1 + rng.normal(0, 0.001)))           # calm
        elif i < 140:
            px.append(px[-1] * (1 + 0.012 + rng.normal(0, 0.003)))   # strong uptrend
        else:
            px.append(px[-1] * (1 - 0.015 + rng.normal(0, 0.006)))   # reversal
    px = np.array(px)
    return pl.DataFrame({
        "ticker": ["T"] * n, "date": days,
        "open": px * (1 + rng.normal(0, 0.002, n)),
        "high": px * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": px * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": px,
        "volume": (1e6 * (1 + np.abs(rng.normal(0, 0.5, n)))).astype(float),
    })


def test_regime_valid_range_and_nonnull():
    out = build_regime_features(_synthetic().lazy()).collect()
    assert "market_regime" in out.columns
    reg = out["market_regime"]
    assert reg.dtype == pl.Int8
    assert reg.null_count() == 0
    assert int(reg.min()) >= 0 and int(reg.max()) <= 7
    # scratch indicator columns must not leak into the panel
    assert not any(c.startswith("_") for c in out.columns)


def test_regime_detects_strong_trend():
    out = build_regime_features(_synthetic().lazy()).collect()
    assert (out["market_regime"] == 3).sum() > 0   # the uptrend block → Strong Trend


def test_regime_leak_free():
    frame = _synthetic()
    n = frame.height
    base = build_regime_features(frame.lazy()).collect()["market_regime"].to_numpy()
    mutated = frame.with_columns(
        pl.when(pl.arange(0, n) >= n - 30).then(pl.col("close") * 1.5)
          .otherwise(pl.col("close")).alias("close")
    )
    after = build_regime_features(mutated.lazy()).collect()["market_regime"].to_numpy()
    assert (base[: n - 31] == after[: n - 31]).all()   # past regimes unchanged


def test_labels_cover_all_eight_regimes():
    assert set(REGIME_LABELS_VI) == set(range(8))
    assert regime_label_vi(3) == "Xu Hướng Mạnh"
    assert regime_label_vi(None) == "Không xác định"
    assert regime_label_vi(99) == "Không xác định"


@pytest.mark.parametrize("regime", [0, 7])
def test_sizing_no_trade_regimes(regime):
    # Freeze (0) / Liquidity Sweep (7) → stand aside regardless of P(UP).
    assert suggested_weight(0.65, market_regime=regime) == 0.0


@pytest.mark.parametrize("regime", [1, 6])
def test_sizing_penalty_regimes_capped(regime):
    base = suggested_weight(0.65)
    w = suggested_weight(0.65, market_regime=regime)
    assert w <= REGIME_PENALTY_CAP + 1e-9     # Squeeze (1) / Choppy (6) → harsh cap
    assert w < base


def test_sizing_strong_trend_is_full_kelly():
    assert suggested_weight(0.65, market_regime=3) == suggested_weight(0.65)


def test_sizing_none_is_backward_compatible():
    for p in (0.45, 0.55, 0.65):
        assert suggested_weight(p, market_regime=None) == suggested_weight(p)
