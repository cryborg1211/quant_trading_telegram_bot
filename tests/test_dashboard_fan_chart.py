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


def test_build_figure_returns_traces() -> None:
    closes = _ramp()
    proj = project_fan(closes, horizon=20)
    fig = build_fan_figure(closes, proj, ticker="HPG")
    assert isinstance(fig, go.Figure)
    # history + median + 2 traces per band (upper edge + filled lower edge).
    assert len(fig.data) == 2 + 2 * len(proj.bands)
    assert any(getattr(t, "fill", None) == "tonexty" for t in fig.data)
