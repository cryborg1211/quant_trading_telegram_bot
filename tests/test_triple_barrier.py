"""Characterization tests for `triple_barrier_pipeline` (V4.1 Structural Debt P3).

Hub node `triple_barrier_pipeline` (src/labels/triple_barrier.py) and its
building blocks had ZERO direct coverage. These tests pin labeling correctness:
first-touch detection, the conservative same-bar PT∧SL tie-break, no look-ahead
(`t1 >= t0`), the 012/raw bin schemes, ruthless excision of unlabelable rows,
and weight normalization.

Pure pandas/numpy/polars — no ML framework, no DuckDB, no mocks. Touch
scenarios use a flat base price with explicit single-bar high/low spikes so the
fired barrier is deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.labels.triple_barrier import (
    TripleBarrierConfig,
    apply_pt_sl_on_t1,
    get_bins,
    get_daily_vol,
    get_vertical_barriers,
    triple_barrier_pipeline,
)


def _bdays(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _touch_case(
    *, spike_bar: int = 2, hi: float = 100.0, lo: float = 100.0,
    n: int = 8, trgt: float = 0.01, vbar: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Flat-100 path with a single-bar high/low spike at `spike_bar`."""
    idx = _bdays(n)
    close = pd.Series(100.0, index=idx)
    high = pd.Series(100.0, index=idx)
    low = pd.Series(100.0, index=idx)
    high.iloc[spike_bar] = hi
    low.iloc[spike_bar] = lo
    events = pd.DataFrame({"t1": [idx[vbar]], "trgt": [trgt]}, index=[idx[0]])
    return close, high, low, events


def _panel_df(
    tickers: tuple[str, ...] = ("AAA",), n: int = 60, base: float = 20_000.0, seed: int = 42,
) -> pl.DataFrame:
    """Seeded random-walk OHLCV panel (σ > 0 so events are labelable)."""
    idx = _bdays(n)
    parts = []
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(seed + i)
        rets = rng.normal(0.0, 0.02, n)
        close = base * np.exp(np.cumsum(rets))
        parts.append(pd.DataFrame({
            "ticker": t, "date": idx, "open": close,
            "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": 1_000_000.0,
        }))
    return pl.from_pandas(pd.concat(parts, ignore_index=True))


# --------------------------------------------------------------------------- #
# 2.2 — get_daily_vol
# --------------------------------------------------------------------------- #
class TestGetDailyVol:
    def test_returns_pandas_series(self) -> None:
        out = get_daily_vol(pd.Series(np.linspace(100, 110, 30), index=_bdays(30)))
        assert isinstance(out, pd.Series)

    def test_length_matches_input(self) -> None:
        close = pd.Series(np.linspace(100, 110, 30), index=_bdays(30))
        assert len(get_daily_vol(close)) == len(close)

    def test_flat_price_gives_near_zero_vol(self) -> None:
        out = get_daily_vol(pd.Series(100.0, index=_bdays(40)))
        assert out.dropna().abs().max() < 1e-9

    def test_raises_on_non_series(self) -> None:
        with pytest.raises(TypeError):
            get_daily_vol([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_non_positive_prices_masked(self) -> None:
        close = pd.Series([100.0, 0.0, 100.0, -5.0, 100.0] * 6, index=_bdays(30))
        out = get_daily_vol(close)
        assert not np.isinf(out.to_numpy()).any()


# --------------------------------------------------------------------------- #
# 2.3 — get_vertical_barriers
# --------------------------------------------------------------------------- #
class TestGetVerticalBarriers:
    def test_t1_is_within_close_index(self) -> None:
        close = pd.Series(100.0, index=_bdays(10))
        t1 = get_vertical_barriers(close, close.index[:5], horizon=3)
        assert all(d in close.index for d in t1.to_numpy())

    def test_horizon_5_gives_correct_offset(self) -> None:
        close = pd.Series(100.0, index=_bdays(10))
        t1 = get_vertical_barriers(close, close.index[[0]], horizon=5)
        assert t1.iloc[0] == close.index[5]

    def test_censored_at_end_of_history(self) -> None:
        close = pd.Series(100.0, index=_bdays(10))
        t1 = get_vertical_barriers(close, close.index[[8]], horizon=20)
        assert t1.iloc[0] == close.index[-1]


# --------------------------------------------------------------------------- #
# 2.4 — apply_pt_sl_on_t1
# --------------------------------------------------------------------------- #
class TestApplyPtSlOnT1:
    def test_pt_hit_before_vertical_barrier(self) -> None:
        close, high, low, events = _touch_case(hi=102.0, lo=100.0)
        out = apply_pt_sl_on_t1(close, events, (1.5, 1.5), high=high, low=low)
        assert out["pt"].notna().any()

    def test_sl_hit_before_vertical_barrier(self) -> None:
        close, high, low, events = _touch_case(hi=100.0, lo=98.0)
        out = apply_pt_sl_on_t1(close, events, (1.5, 1.5), high=high, low=low)
        assert out["sl"].notna().any()

    def test_vertical_barrier_when_no_touch(self) -> None:
        close, high, low, events = _touch_case(hi=100.0, lo=100.0)
        out = apply_pt_sl_on_t1(close, events, (1.5, 1.5), high=high, low=low)
        assert out["pt"].isna().all()
        assert out["sl"].isna().all()
        assert (out["t1"].to_numpy() == events["t1"].to_numpy()).all()

    def test_conservative_tiebreak_sl_wins(self) -> None:
        # Same bar trips both +PT and -SL → SL wins, PT dropped to NaT.
        close, high, low, events = _touch_case(hi=102.0, lo=98.0)
        out = apply_pt_sl_on_t1(close, events, (1.5, 1.5), high=high, low=low)
        assert out["pt"].isna().all()
        assert out["sl"].notna().any()

    def test_t1_is_never_before_t0(self) -> None:
        close, high, low, events = _touch_case(hi=102.0, lo=100.0)
        out = apply_pt_sl_on_t1(close, events, (1.5, 1.5), high=high, low=low)
        assert (out["t1"].to_numpy() >= events.index.to_numpy()).all()

    def test_empty_events_returns_empty_dataframe(self) -> None:
        events = pd.DataFrame(
            {"t1": pd.Series([], dtype="datetime64[ns]"), "trgt": pd.Series([], dtype=float)}
        )
        out = apply_pt_sl_on_t1(pd.Series(100.0, index=_bdays(5)), events, (1.5, 1.5))
        assert len(out) == 0
        assert {"pt", "sl", "t1"}.issubset(out.columns)


# --------------------------------------------------------------------------- #
# 2.5 — get_bins
# --------------------------------------------------------------------------- #
def _bins_events(pt, sl, t1, t0) -> pd.DataFrame:
    return pd.DataFrame({"t1": [t1], "pt": [pt], "sl": [sl]}, index=[t0])


class TestGetBins:
    def test_bin_values_in_012_scheme(self) -> None:
        idx = _bdays(6)
        close = pd.Series(np.linspace(100, 110, 6), index=idx)  # rising → ret>0
        events = _bins_events(pd.NaT, pd.NaT, idx[2], idx[0])
        out = get_bins(events, close, label_scheme="012")
        assert set(out["bin"].dropna().unique()).issubset({0.0, 1.0, 2.0})

    def test_bin_values_in_raw_scheme(self) -> None:
        idx = _bdays(6)
        close = pd.Series(np.linspace(100, 110, 6), index=idx)
        events = _bins_events(pd.NaT, pd.NaT, idx[2], idx[0])
        out = get_bins(events, close, label_scheme="raw")
        assert set(out["bin"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_invalid_scheme_raises(self) -> None:
        idx = _bdays(6)
        close = pd.Series(100.0, index=idx)
        events = _bins_events(pd.NaT, pd.NaT, idx[2], idx[0])
        with pytest.raises(ValueError):
            get_bins(events, close, label_scheme="nope")

    def test_pt_hit_gives_up_bin(self) -> None:
        idx = _bdays(6)
        close = pd.Series(100.0, index=idx)
        events = _bins_events(idx[2], pd.NaT, idx[2], idx[0])  # PT first
        out = get_bins(events, close, label_scheme="012")
        assert out["bin"].iloc[0] == 2.0

    def test_sl_hit_gives_down_bin(self) -> None:
        idx = _bdays(6)
        close = pd.Series(100.0, index=idx)
        events = _bins_events(pd.NaT, idx[2], idx[2], idx[0])  # SL first
        out = get_bins(events, close, label_scheme="012")
        assert out["bin"].iloc[0] == 0.0

    def test_invalid_price_gives_nan_bin(self) -> None:
        idx = _bdays(6)
        close = pd.Series(100.0, index=idx)
        close.iloc[0] = 0.0  # invalid entry price
        events = _bins_events(pd.NaT, pd.NaT, idx[2], idx[0])
        out = get_bins(events, close, label_scheme="012")
        assert np.isnan(out["bin"].iloc[0])


# --------------------------------------------------------------------------- #
# 2.6 — triple_barrier_pipeline
# --------------------------------------------------------------------------- #
class TestTripleBarrierPipeline:
    def test_returns_polars_dataframe(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False)
        assert isinstance(out, pl.DataFrame)

    def test_output_columns_complete(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False)
        expected = {"ticker", "t0", "t1", "trgt", "ret", "bin", "num_co_events", "uniqueness", "w"}
        assert expected.issubset(set(out.columns))

    def test_bin_dtype_is_int64(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False)
        assert out.schema["bin"] == pl.Int64

    def test_no_look_ahead_t1_ge_t0(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False)
        assert out.select((pl.col("t1") >= pl.col("t0")).all()).item()

    def test_no_nan_bin_in_output(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False)
        assert out["bin"].null_count() == 0

    def test_normalize_weights_gives_mean_1(self) -> None:
        out = triple_barrier_pipeline(_panel_df(), progress=False, normalize_weights=True)
        assert abs(float(out["w"].mean()) - 1.0) < 1e-6

    def test_two_tickers_both_present(self) -> None:
        out = triple_barrier_pipeline(_panel_df(tickers=("AAA", "BBB")), progress=False)
        assert set(out["ticker"].unique().to_list()) == {"AAA", "BBB"}

    def test_raises_on_missing_close_column(self) -> None:
        with pytest.raises(ValueError):
            triple_barrier_pipeline(_panel_df(), close_col="nonexistent", progress=False)

    def test_raises_when_all_tickers_skipped(self) -> None:
        # 3 bars < vol_span+horizon+5 = 35 → every ticker skipped.
        with pytest.raises(ValueError):
            triple_barrier_pipeline(_panel_df(n=3), progress=False)

    def test_t5_vs_t20_horizons_produce_different_outputs(self) -> None:
        panel = _panel_df()
        res5 = triple_barrier_pipeline(panel, TripleBarrierConfig(horizon=5), progress=False)
        res20 = triple_barrier_pipeline(panel, TripleBarrierConfig(horizon=20), progress=False)
        assert res5["t1"].to_list() != res20["t1"].to_list()
