"""Unit tests for the pure-technical fan-chart forecast.

``project_fan`` and ``build_fan_figure`` are Streamlit-free, so they are tested
directly without an AppTest harness. The contract that matters for the
visualization: the band widens with the horizon (T+5 tight, T+20 wide), the fan
anchors at the last close, and degenerate inputs degrade cleanly.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import pytest

from dashboard.utils.fan_chart import build_fan_figure, project_fan


def _ramp(n: int = 60, start: float = 50.0, step: float = 0.1) -> list[float]:
    return [start + step * i for i in range(n)]


def test_projection_anchors_at_last_close() -> None:
    closes = _ramp()
    proj = project_fan(closes, horizon=20)
    assert proj.s0 == closes[-1]
    assert len(proj.days) == 20
    assert len(proj.median) == 20


def test_band_widens_with_horizon() -> None:
    # Noisy series → non-zero sigma → the fan must fan OUT: the T+20 spread of
    # the widest band strictly exceeds its T+5 spread.
    closes = [50.0, 51.0, 49.5, 52.0, 50.5, 53.0, 49.0, 54.0, 50.0, 55.0]
    proj = project_fan(closes, horizon=20)
    widest = proj.bands[-1]
    spread_t5 = widest.upper[4] - widest.lower[4]
    spread_t20 = widest.upper[19] - widest.lower[19]
    assert spread_t20 > spread_t5 > 0.0


def test_bands_nested_narrow_to_wide() -> None:
    closes = [50.0, 51.0, 49.5, 52.0, 50.5, 53.0, 49.0, 54.0]
    proj = project_fan(closes, horizon=20)
    # Bands ordered narrow→wide: each outer band brackets the inner at T+20.
    for inner, outer in zip(proj.bands, proj.bands[1:]):
        assert outer.upper[-1] >= inner.upper[-1]
        assert outer.lower[-1] <= inner.lower[-1]


def test_flat_series_collapses_band_to_median() -> None:
    # Zero volatility → bands degenerate onto the median (sigma == 0).
    proj = project_fan([50.0] * 30, horizon=20)
    assert proj.sigma == 0.0
    for band in proj.bands:
        assert band.upper == pytest.approx(proj.median)
        assert band.lower == pytest.approx(proj.median)


def test_too_few_closes_raises() -> None:
    with pytest.raises(ValueError):
        project_fan([50.0], horizon=20)
    with pytest.raises(ValueError):
        project_fan([], horizon=20)


def test_non_positive_prices_filtered() -> None:
    # 0 / negative closes are dropped; a single survivor pair still projects.
    proj = project_fan([0.0, -1.0, 50.0, 51.0], horizon=5)
    assert proj.s0 == 51.0
    assert math.isfinite(proj.median[0])


def _ohlc(n: int = 60, start: float = 50.0, step: float = 0.1) -> list:
    """Synthetic ascending OHLC rows (date, open, high, low, close)."""
    from datetime import date, timedelta

    d0 = date(2026, 1, 1)
    out = []
    for i in range(n):
        c = start + step * i
        out.append((d0 + timedelta(days=i), c - 0.05, c + 0.1, c - 0.1, c))
    return out


def test_build_figure_candlestick_and_mc_paths() -> None:
    ohlc = _ohlc()
    closes = [row[4] for row in ohlc]
    proj = project_fan(closes, horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG", n_paths=12)
    assert isinstance(fig, go.Figure)
    # 1 candlestick + 12 Monte Carlo paths + 1 neon median = 14 traces.
    assert len(fig.data) == 14
    assert any(isinstance(t, go.Candlestick) for t in fig.data)
    # No legacy shaded fan band survives.
    assert not any(getattr(t, "fill", None) == "tonexty" for t in fig.data)
    # The bold neon median is present at width 3.
    assert any(
        getattr(getattr(t, "line", None), "color", None) == "#00F2FE"
        and getattr(getattr(t, "line", None), "width", None) == 3
        for t in fig.data
    )


def test_simulate_gbm_paths_shape_and_determinism() -> None:
    from dashboard.utils.fan_chart import simulate_gbm_paths

    proj = project_fan(_ramp(), horizon=20)
    a = simulate_gbm_paths(proj, 12, ticker="HPG")
    b = simulate_gbm_paths(proj, 12, ticker="HPG")
    assert len(a) == 12 and all(len(p) == 20 for p in a)
    assert a == b  # same ticker → deterministic across calls (no rerun jitter)


def test_simulate_gbm_paths_differ_by_ticker() -> None:
    # Root-cause guard: a shared seed made every ticker clone one wiggle shape.
    # Normalise out s0/mu/sigma scaling by comparing the standardised SHAPE of
    # the first path — different tickers must yield different shapes.
    from dashboard.utils.fan_chart import simulate_gbm_paths

    proj = project_fan(_ramp(), horizon=20)
    dcm = simulate_gbm_paths(proj, 12, ticker="DCM")
    vhm = simulate_gbm_paths(proj, 12, ticker="VHM")
    assert dcm != vhm
    # Even path-0 (same proj, so same mu/sigma/s0) must diverge → proves the
    # underlying random draws differ, not just a scalar rescale.
    assert dcm[0] != vhm[0]


def test_ticker_seed_is_process_stable() -> None:
    # crc32-based seed must NOT depend on PYTHONHASHSEED (unlike builtin hash).
    from dashboard.utils.fan_chart import _ticker_seed

    assert _ticker_seed("DCM") == _ticker_seed("dcm")  # normalised upper
    assert _ticker_seed("DCM") != _ticker_seed("VHM")
