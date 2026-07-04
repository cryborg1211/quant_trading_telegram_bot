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
    """Synthetic ascending OHLCV rows (date, open, high, low, close, volume)."""
    from datetime import date, timedelta

    d0 = date(2026, 1, 1)
    out = []
    for i in range(n):
        c = start + step * i
        out.append((d0 + timedelta(days=i), c - 0.05, c + 0.1, c - 0.1, c, 1000.0 + i))
    return out


def test_build_figure_default_is_clean_scenario_mode() -> None:
    ohlc = _ohlc()
    closes = [row[4] for row in ohlc]
    proj = project_fan(closes, horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Candlestick) for t in fig.data)
    # Exactly ONE filled cone (the 80% band) — not the old 3-band stack.
    fills = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]
    assert len(fills) == 1
    # Broker-style default: labelled bull/bear scenario edges, NO MC spaghetti.
    names = {t.name for t in fig.data if t.name}
    assert "Kịch bản lạc quan (P90)" in names
    assert "Kịch bản thận trọng (P10)" in names
    assert not any(
        getattr(t, "legendgroup", None) == "mc_paths" for t in fig.data
    )
    # MA 10/20/50 overlays present (60 closes ≥ every window).
    ma_names = {n for n in names if n.startswith("MA ")}
    assert ma_names == {"MA 10", "MA 20", "MA 50"}
    # The bold neon median is present at width 3.
    assert any(
        getattr(getattr(t, "line", None), "color", None) == "#00F2FE"
        and getattr(getattr(t, "line", None), "width", None) == 3
        for t in fig.data
    )


def test_mc_paths_opt_in_via_show_paths() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG", n_paths=12, show_paths=True)
    paths = [t for t in fig.data if getattr(t, "legendgroup", None) == "mc_paths"]
    assert len(paths) == 12


def test_volume_pane_present_and_colored() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    bars = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bars) == 1  # volume pane only (no flow passed)
    vol = bars[0]
    assert list(vol.y) == [row[5] for row in ohlc]
    # Rising synthetic candles (close > open) → every bar TV-teal.
    assert set(vol.marker.color) == {"#26a69a"}


def test_foreign_flow_pane_optional() -> None:
    from datetime import date

    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    # Thousand-VND net values: +1.5 tỷ then −0.86 tỷ.
    flow = [(date(2026, 2, 26), 1_500_000.0), (date(2026, 2, 27), -860_510.0)]
    fig = build_fan_figure(ohlc, proj, ticker="GVR", flow=flow)
    bars = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bars) == 2  # volume + foreign flow
    flow_bar = bars[-1]
    assert list(flow_bar.y) == pytest.approx([1.5, -0.860510])  # tỷ VND
    assert list(flow_bar.marker.color) == ["#26a69a", "#ef5350"]
    # Without flow the third pane (and its bar) is absent.
    fig2 = build_fan_figure(ohlc, proj, ticker="GVR")
    assert len([t for t in fig2.data if isinstance(t, go.Bar)]) == 1


def test_ohlc_readout_annotation() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    texts = [a.text for a in fig.layout.annotations if a.text]
    assert any(t.startswith("O <b>") and "%" in t for t in texts)


def test_cone_uses_80pct_band_and_anchors_at_s0() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    cone_upper = next(t for t in fig.data if getattr(t, "fill", None) == "tonexty")
    # Anchored at S0 today, then exactly the 80% (index-1) analytic band.
    assert cone_upper.y[0] == proj.s0
    assert list(cone_upper.y[1:]) == pytest.approx(proj.bands[1].upper)


def test_figure_guides_and_terminal_labels() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    annot_texts = [a.text for a in fig.layout.annotations if a.text]
    # Horizon + today guides are labelled.
    assert any("T+5" in t for t in annot_texts)
    assert any("T+20" in t for t in annot_texts)
    assert any("Hôm nay" in t for t in annot_texts)
    # 3 terminal outcome labels carry a signed % change.
    pct_labels = [t for t in annot_texts if "%" in t and ("+" in t or "−" in t or "-" in t)]
    assert len(pct_labels) >= 3
    # Guide shapes: forecast wash + today/T+5/T+20 vlines + last-price hline.
    assert len(fig.layout.shapes) >= 5


def test_gapless_tape_rangebreaks() -> None:
    from datetime import date, timedelta

    # Weekday-only history with one missing Wednesday (exchange holiday).
    d0 = date(2026, 1, 5)  # a Monday
    rows, d, i = [], d0, 0
    holiday = date(2026, 1, 14)  # the skipped Wednesday
    while len(rows) < 40:
        if d.weekday() < 5 and d != holiday:
            c = 50.0 + 0.1 * i
            rows.append((d, c - 0.05, c + 0.1, c - 0.1, c))
            i += 1
        d += timedelta(days=1)
    proj = project_fan([r[4] for r in rows], horizon=20)
    fig = build_fan_figure(rows, proj, ticker="HPG")

    breaks = fig.layout.xaxis.rangebreaks
    assert any(tuple(b.bounds or ()) == ("sat", "mon") for b in breaks)
    holiday_values = [v for b in breaks if b.values for v in b.values]
    assert holiday in holiday_values


def test_tradingview_conventions() -> None:
    ohlc = _ohlc()
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    # Price scale on the right, TV canvas color, symbol watermark present.
    assert fig.layout.yaxis.side == "right"
    assert fig.layout.plot_bgcolor == "#131722"
    assert any(
        a.text == "HPG" and (a.font.size or 0) >= 48
        for a in fig.layout.annotations
    )


def test_default_viewport_focuses_recent_history_plus_fan() -> None:
    ohlc = _ohlc(n=120)  # more history than the 45-session default view
    proj = project_fan([row[4] for row in ohlc], horizon=20)
    fig = build_fan_figure(ohlc, proj, ticker="HPG")
    lo, hi = fig.layout.xaxis.range
    # Viewport starts inside the history (not at bar 0) and ends past the
    # last candle (the fan is fully visible by default).
    assert lo > ohlc[0][0]
    assert hi > ohlc[-1][0]
    # Candlestick's forced rangeslider is OFF (terminal look; it also ignores
    # rangebreaks) — full history stays reachable via the ALL button.
    assert fig.layout.xaxis.rangeslider.visible is False


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
