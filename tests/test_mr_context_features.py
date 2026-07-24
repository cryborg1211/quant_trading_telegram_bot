"""MR context features — volume-exhaustion + sector-relative oversold
(22-07-26 knife-catch research follow-up).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.mr_context_features import (
    MR_CONTEXT_FEATURE_COLUMNS,
    build_mr_context_features,
)
from src.features.mr_features import build_mr_features

N = 40


def _bars(ticker: str, n: int, prices: list[float], volumes: list[float]) -> pd.DataFrame:
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "ticker": ticker, "date": days,
        "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices,
        "volume": volumes,
    })


def _flat_walk(n: int, start: float = 100.0) -> list[float]:
    px = [start]
    for i in range(n - 1):
        px.append(px[-1] + (0.3 if i % 2 == 0 else -0.3))
    return px


def _panel_with(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _ready(panel: pd.DataFrame) -> pd.DataFrame:
    return build_mr_context_features(build_mr_features(panel))


def test_missing_required_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_mr_context_features(pd.DataFrame({"ticker": ["A"], "date": [dt.date(2025, 1, 1)]}))


def test_output_has_all_context_columns() -> None:
    panel = _bars("VCB", N, _flat_walk(N), [1_000_000.0] * N)
    res = _ready(panel)
    for col in MR_CONTEXT_FEATURE_COLUMNS:
        assert col in res.columns


def test_volume_spike_gets_high_z() -> None:
    vols = [1_000_000.0] * (N - 1) + [5_000_000.0]   # spike on the last bar
    panel = _bars("VCB", N, _flat_walk(N), vols)
    res = _ready(panel)
    assert res["mrctx_vol_z20"].iloc[-1] > 2.0


def test_all_down_days_down_vol_ratio_near_one() -> None:
    px = [100.0 - 0.5 * i for i in range(N)]          # monotonic decline
    panel = _bars("VCB", N, px, [1_000_000.0] * N)
    res = _ready(panel)
    assert res["mrctx_down_vol_ratio"].iloc[-1] == pytest.approx(1.0)


def test_all_up_days_down_vol_ratio_near_zero() -> None:
    px = [100.0 + 0.5 * i for i in range(N)]          # monotonic rise
    panel = _bars("VCB", N, px, [1_000_000.0] * N)
    res = _ready(panel)
    assert res["mrctx_down_vol_ratio"].iloc[-1] == pytest.approx(0.0)


def test_fading_volume_during_decline_is_negative_trend() -> None:
    px = [100.0 - 0.3 * i for i in range(N)]
    vols = [2_000_000.0] * 35 + [500_000.0] * 5      # volume dries up recently
    panel = _bars("VCB", N, px, vols)
    res = _ready(panel)
    assert res["mrctx_vol_trend_3v10"].iloc[-1] < 0.0


def test_sector_relative_more_oversold_than_peer_is_negative() -> None:
    # VCB, BID, CTG are all BANKS (>=3 present clears the reliability
    # floor). VCB crashes harder -> lower RSI14 -> its sector-relative
    # reading should be NEGATIVE (more oversold than its peers).
    crash_px = [100.0] * 25 + [100.0 * (0.98 ** i) for i in range(1, N - 24)]
    calm_px = _flat_walk(N)
    panel = _panel_with([
        _bars("VCB", N, crash_px, [1_000_000.0] * N),
        _bars("BID", N, calm_px, [1_000_000.0] * N),
        _bars("CTG", N, calm_px, [1_000_000.0] * N),
    ])
    res = _ready(panel)
    vcb_last = res[res["ticker"] == "VCB"].iloc[-1]
    assert vcb_last["mrctx_sect_rsi_rel"] < 0.0


def test_unmapped_ticker_is_other_and_nan() -> None:
    panel = _panel_with([
        _bars("ZZZTICKER", N, _flat_walk(N), [1_000_000.0] * N),
        _bars("QQQTICKER", N, _flat_walk(N), [1_000_000.0] * N),
    ])
    res = _ready(panel)
    assert res["mrctx_sect_rsi_rel"].isna().all()


def test_sector_group_below_min_names_is_nan() -> None:
    # AGRI_AQUA has 6 members; using only 2 keeps the group below the
    # min-3-names reliability floor -> NaN, not a noisy 2-name median.
    panel = _panel_with([
        _bars("HAG", N, _flat_walk(N), [1_000_000.0] * N),
        _bars("HNG", N, _flat_walk(N), [1_000_000.0] * N),
    ])
    res = _ready(panel)
    assert res["mrctx_sect_rsi_rel"].isna().all()


def test_leak_free_past_unaffected_by_future_mutation() -> None:
    panel = _panel_with([
        _bars("VCB", N, _flat_walk(N), [1_000_000.0] * N),
        _bars("BID", N, _flat_walk(N), [1_000_000.0] * N),
        _bars("CTG", N, _flat_walk(N), [1_000_000.0] * N),
    ])
    res1 = _ready(panel)
    cutoff = N - 5
    cutoff_date = pd.Timestamp(dt.date(2025, 1, 1) + dt.timedelta(days=cutoff))
    mutated = panel.copy()
    mask = pd.to_datetime(mutated["date"]) >= cutoff_date
    mutated.loc[mask, "close"] *= 2.0
    mutated.loc[mask, "volume"] *= 5.0
    res2 = _ready(mutated)
    before = pd.to_datetime(res1["date"]) < cutoff_date
    a = res1.loc[before, MR_CONTEXT_FEATURE_COLUMNS].to_numpy(dtype=float)
    b = res2.loc[before, MR_CONTEXT_FEATURE_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, equal_nan=True)
