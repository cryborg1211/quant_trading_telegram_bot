"""Pure-technical fan-chart forecast (no AI / news / sentiment).

A deliberately simple, instant, deterministic forecast: a Geometric Brownian
Motion projection estimated from the ticker's own recent log-returns. There is
NO Gemini, no LLM, no news pipeline anywhere in this module — that is the whole
point of the "Tầm Nhìn Thuật Toán" tab (renders in milliseconds).

Model
-----
From the last ``len(closes)`` daily closes:

* ``mu``    = mean daily log-return (drift)
* ``sigma`` = sample stdev of daily log-returns (volatility)

Project forward ``t = 1..H`` trading days under GBM:

* median(t) = S0 · exp(mu·t)
* band(t)   = S0 · exp(mu·t ± z·sigma·√t)

The ``√t`` term makes the band widen with the horizon — tight at T+5, wide at
T+20 — producing the probability fan.

Figure layout (broker-terminal style, 02-07-26)
-----------------------------------------------
``build_fan_figure`` renders a TradingView-like multi-pane terminal:

* pane 1 — candlesticks + MA10/20/50 overlays + 80% cone + Monte Carlo paths
  + neon median + OHLC readout + symbol watermark + last-price tag;
* pane 2 — session volume bars (green/red by candle direction);
* pane 3 — foreign net flow (Khối ngoại, tỷ VND) from
  ``data/foreign_flow_daily.parquet`` — only when the caller passes ``flow``
  rows (the parquet accumulates one row per ticker per day, forward-only).

Weekends and inferred exchange holidays are removed from the time axis
(``rangebreaks``), so the tape is gapless like a real trading terminal.

Both functions are Streamlit-free (plotly is a pure library), so they are unit
tested directly.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# z multipliers ≈ 50% / 80% / 95% central intervals (narrow → wide).
# `project_fan` computes all three analytic bands (unit-tested); the figure
# draws ONLY the 80% one as a single faint cone under the Monte Carlo paths
# (three nested fills was the old clutter; zero cone made the paths read as
# noise — one subtle band is the middle ground).
_DEFAULT_Z_LEVELS = (0.674, 1.282, 1.960)


@dataclass(frozen=True)
class FanBand:
    """One nested confidence band of the fan."""

    z: float
    upper: list[float]
    lower: list[float]


@dataclass(frozen=True)
class FanProjection:
    """Forward GBM projection result (days are 1-indexed trading days)."""

    days: list[int]
    median: list[float]
    bands: list[FanBand]  # ordered narrow → wide
    s0: float
    mu: float
    sigma: float


def _log_returns(closes: list[float]) -> list[float]:
    """Daily log-returns of a positive-price series (skips non-positive pairs)."""
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def project_fan(
    closes: list[float],
    horizon: int = 20,
    z_levels: tuple[float, ...] = _DEFAULT_Z_LEVELS,
) -> FanProjection:
    """Project a GBM fan ``horizon`` trading days forward from the last close.

    Raises ``ValueError`` if fewer than two positive closes are supplied (the
    caller is expected to show a clean "not enough data" message instead).
    """
    clean = [float(c) for c in closes if c and c > 0]
    if len(clean) < 2:
        raise ValueError("need at least two positive closes to project")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    rets = _log_returns(clean)
    s0 = clean[-1]
    mu = sum(rets) / len(rets) if rets else 0.0
    if len(rets) >= 2:
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0

    days = list(range(1, horizon + 1))
    median = [s0 * math.exp(mu * t) for t in days]
    bands = [
        FanBand(
            z=z,
            upper=[s0 * math.exp(mu * t + z * sigma * math.sqrt(t)) for t in days],
            lower=[s0 * math.exp(mu * t - z * sigma * math.sqrt(t)) for t in days],
        )
        for z in sorted(z_levels)  # narrow → wide
    ]
    return FanProjection(
        days=days, median=median, bands=bands, s0=s0, mu=mu, sigma=sigma
    )


# TradingView palette — the real one, not an approximation. Candle teal/red,
# #131722 chart canvas, #787b86 axis ink are TV's exact dark-theme values.
_CANDLE_UP = "#26a69a"
_CANDLE_DOWN = "#ef5350"
_PATH_UP = "rgba(38, 166, 154, 0.22)"   # semi-transparent teal (winning paths)
_PATH_DOWN = "rgba(239, 83, 80, 0.22)"  # semi-transparent red (losing paths)
_MEDIAN_NEON = "#00F2FE"                # bold neon median trajectory
_CONE_FILL = "rgba(0, 242, 254, 0.07)"  # faint cyan 80% probability cone
_CONE_EDGE = "rgba(0, 242, 254, 0.25)"  # barely-visible cone edge lines
_FORECAST_ZONE = "rgba(45, 212, 167, 0.04)"  # subtle forecast-region wash
_GUIDE_COLOR = "rgba(120, 123, 134, 0.55)"   # T+5 / T+20 / today guide lines
_TV_BG = "#131722"                       # TradingView chart canvas
_TV_GRID = "rgba(42, 46, 57, 0.55)"      # TV hairline grid
_SPIKE_COLOR = "#787b86"
_TEXT_MUTED = "#787b86"                  # TV axis/legend ink
_PRICE_BADGE_BG = "#363a45"              # last-price tag on the right axis
_WATERMARK = "rgba(120, 123, 134, 0.14)"  # big faint symbol behind the chart
_FONT_FAMILY = "Trebuchet MS, Inter, Segoe UI, Arial"  # TV's chart font first
# Moving-average overlays (window → TV-ish line color).
_MA_STYLES: tuple[tuple[int, str], ...] = (
    (10, "#fbc02d"),   # amber
    (20, "#ab47bc"),   # purple
    (50, "#2962ff"),   # TV blue
)
_N_MC_PATHS = 12
# Default viewport: last N history sessions + the full forecast. 120 sessions of
# history squashed the 20-day fan into the right edge — the fan is the point of
# this chart, so the default zoom keeps it readable (the 1M/3M/6M/ALL buttons
# still give the full history on demand).
_DEFAULT_VIEW_SESSIONS = 45
# Base seed mixed with a per-ticker hash. A SINGLE shared seed would make every
# ticker draw the identical standard-normal sequence, so all fans would share
# one wiggle shape (only scaled by mu/sigma/s0). Mixing the ticker in gives each
# symbol its own stable-but-distinct stream — still jitter-free across reruns.
_MC_SEED_BASE = 7


def _ticker_seed(ticker: str, base: int = _MC_SEED_BASE) -> int:
    """Stable per-ticker seed (process-independent, unlike ``hash``).

    ``zlib.crc32`` is deterministic across interpreter runs (Python's built-in
    ``hash`` is salted per-process), so the same ticker reproduces the same fan
    on every launch while different tickers diverge.
    """
    return (base ^ zlib.crc32(ticker.strip().upper().encode())) & 0xFFFFFFFF


def _business_days_after(last: date, n: int) -> list[date]:
    """The next ``n`` weekday dates strictly after ``last`` (skips Sat/Sun).

    A cheap stand-in for the VN trading calendar — good enough to give the
    forecast a real datetime x-axis so the rangeselector month buttons work.
    """
    out: list[date] = []
    cur = last
    while len(out) < n:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:  # 0=Mon … 4=Fri
            out.append(cur)
    return out


def _sma(values: list[float], window: int) -> list[float | None]:
    """Simple moving average; ``None`` until the window fills (plotly skips)."""
    out: list[float | None] = [None] * len(values)
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out[i] = acc / window
    return out


def simulate_gbm_paths(
    proj: FanProjection,
    n_paths: int = _N_MC_PATHS,
    *,
    ticker: str = "",
    seed: int | None = None,
) -> list[list[float]]:
    """Simulate ``n_paths`` GBM price walks from S0 over the projection horizon.

    Each path draws ``horizon`` i.i.d. daily log-returns ``~ N(mu, sigma)`` and
    compounds them: ``price_t = S0·exp(Σ r)``. The RNG seed defaults to a
    per-``ticker`` value so each symbol gets its own distinct path shape while
    staying stable across Streamlit reruns; pass ``seed`` to override (tests).
    Returns one list of ``horizon`` prices per path (S0 anchor added by caller).
    """
    horizon = len(proj.days)
    rng_seed = _ticker_seed(ticker) if seed is None else seed
    rng = np.random.default_rng(rng_seed)
    rets = rng.normal(proj.mu, proj.sigma, size=(n_paths, horizon))
    walks = proj.s0 * np.exp(np.cumsum(rets, axis=1))
    return [row.tolist() for row in walks]


def build_fan_figure(
    ohlc: list[tuple],
    proj: FanProjection,
    *,
    ticker: str = "",
    n_paths: int = _N_MC_PATHS,
    flow: list[tuple[object, float]] | None = None,
    show_paths: bool = False,
) -> go.Figure:
    """Build the broker-terminal figure: price+forecast / volume / foreign flow.

    ``ohlc`` is ascending ``(date, open, high, low, close[, volume])`` history
    (the volume field is optional for backward compat — absent volume renders
    an empty volume pane). ``flow`` is optional ascending ``(date,
    foreign_net_val)`` rows in THOUSANDS of VND (the on-disk convention of
    ``foreign_flow_daily.parquet``); when present a third "Khối ngoại" pane is
    added, rendered in tỷ VND. The forecast extends real business days past the
    last candle; weekends + inferred exchange holidays are hidden via
    rangebreaks so the tape is gapless.
    """
    has_flow = bool(flow)
    n_rows = 3 if has_flow else 2
    row_heights = [0.58, 0.18, 0.24] if has_flow else [0.76, 0.24]
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=row_heights,
    )

    hist_dates = [row[0] for row in ohlc]
    last_date = hist_dates[-1] if hist_dates else date.today()
    closes = [row[4] for row in ohlc]

    # --- Non-trading-day gap removal (VN market: Mon–Fri) ---------------------
    # Weekends are closed via a static rangebreak; exchange holidays (Tết, …)
    # are inferred as any weekday inside the history window with no candle,
    # so the tape reads gapless like TradingView.
    holiday_breaks: list[date] = []
    if hist_dates:
        present = set(hist_dates)
        cur = hist_dates[0]
        while cur < last_date:
            if cur.weekday() < 5 and cur not in present:
                holiday_breaks.append(cur)
            cur = cur + timedelta(days=1)

    # --- Pane 1: candlestick history ------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=hist_dates,
            open=[row[1] for row in ohlc],
            high=[row[2] for row in ohlc],
            low=[row[3] for row in ohlc],
            close=closes,
            name="Lịch sử giá (OHLC)",
            increasing_line_color=_CANDLE_UP,
            increasing_fillcolor=_CANDLE_UP,
            decreasing_line_color=_CANDLE_DOWN,
            decreasing_fillcolor=_CANDLE_DOWN,
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # --- Pane 1: MA10/20/50 overlays (drawn under the forecast traces) --------
    for window, color in _MA_STYLES:
        if len(closes) >= window:
            fig.add_trace(
                go.Scatter(
                    x=hist_dates, y=_sma(closes, window), mode="lines",
                    name=f"MA {window}",
                    line=dict(color=color, width=1),
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

    s0 = proj.s0
    fwd_dates = _business_days_after(last_date, len(proj.days))
    fan_x = [last_date, *fwd_dates]  # anchor every forward series at today (S0)

    # --- Pane 1: bull/base/bear scenario cone (broker-research style) --------
    # `proj.bands` is narrow→wide (≈50/80/95%); index 1 is the 80% interval.
    # Default look = the analyst target-range visual every brokerage uses:
    # a shaded cone whose edges ARE the labelled optimistic (P90) and
    # defensive (P10) scenarios, plus the bold median. The 12-path Monte
    # Carlo spaghetti confused non-quant users as the default — it stays
    # available behind ``show_paths`` (a toggle in the tab).
    cone = proj.bands[1] if len(proj.bands) >= 2 else proj.bands[-1]
    _edge_w = 1 if show_paths else 1.5
    fig.add_trace(
        go.Scatter(
            x=fan_x, y=[s0, *cone.lower], mode="lines",
            line=dict(color=_CANDLE_DOWN, width=_edge_w, dash="dot"),
            name="Kịch bản thận trọng (P10)", legendgroup="cone",
            customdata=[0.0] + [
                (v / s0 - 1.0) * 100.0 if s0 else 0.0 for v in cone.lower
            ],
            hovertemplate=(
                "%{x|%d/%m}: %{y:,.2f} "
                "(%{customdata:+.1f}%)<extra>Thận trọng</extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fan_x, y=[s0, *cone.upper], mode="lines",
            line=dict(color=_CANDLE_UP, width=_edge_w, dash="dot"),
            fill="tonexty", fillcolor=_CONE_FILL,
            name="Kịch bản lạc quan (P90)", legendgroup="cone",
            customdata=[0.0] + [
                (v / s0 - 1.0) * 100.0 if s0 else 0.0 for v in cone.upper
            ],
            hovertemplate=(
                "%{x|%d/%m}: %{y:,.2f} "
                "(%{customdata:+.1f}%)<extra>Lạc quan</extra>"
            ),
        ),
        row=1, col=1,
    )

    # --- Pane 1: optional Monte Carlo paths (opt-in; branch from T_now) ------
    if show_paths:
        paths = simulate_gbm_paths(proj, n_paths, ticker=ticker)
        for i, walk in enumerate(paths):
            terminal_up = walk[-1] >= s0
            color = _PATH_UP if terminal_up else _PATH_DOWN
            fig.add_trace(
                go.Scatter(
                    x=fan_x,
                    y=[s0, *walk],
                    mode="lines",
                    line=dict(color=color, width=1),
                    name="Đường giả lập (Simulated Paths)" if i == 0 else None,
                    legendgroup="mc_paths",
                    showlegend=(i == 0),  # one legend entry for the bundle
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # --- Pane 1: median trajectory (bold neon dashed) --------------------------
    median_pct = [0.0] + [
        (m / s0 - 1.0) * 100.0 if s0 else 0.0 for m in proj.median
    ]
    fig.add_trace(
        go.Scatter(
            x=fan_x,
            y=[s0, *proj.median],
            mode="lines",
            name="Kịch bản trung vị (Expected Median)",
            line=dict(color=_MEDIAN_NEON, width=3, dash="dash"),
            customdata=median_pct,
            hovertemplate=(
                "%{x|%d/%m}: %{y:,.2f} "
                "(%{customdata:+.1f}%)<extra>Trung vị</extra>"
            ),
        ),
        row=1, col=1,
    )

    # --- Pane 2: session volume ------------------------------------------------
    volumes = [
        (row[5] if len(row) > 5 and row[5] is not None else 0.0) for row in ohlc
    ]
    vol_colors = [
        _CANDLE_UP if row[4] >= row[1] else _CANDLE_DOWN for row in ohlc
    ]
    fig.add_trace(
        go.Bar(
            x=hist_dates, y=volumes, marker_color=vol_colors,
            marker_line_width=0, opacity=0.85, showlegend=False,
            hovertemplate="%{x|%d/%m}: %{y:,.0f} cp<extra>Khối lượng</extra>",
        ),
        row=2, col=1,
    )

    # --- Pane 3: foreign net flow (Khối ngoại, tỷ VND) -------------------------
    if has_flow:
        flow_dates = [r[0] for r in flow]
        # thousands of VND on disk → tỷ VND for display (÷ 1e6).
        flow_ty = [float(r[1]) / 1e6 if r[1] is not None else 0.0 for r in flow]
        flow_colors = [_CANDLE_UP if v >= 0 else _CANDLE_DOWN for v in flow_ty]
        fig.add_trace(
            go.Bar(
                x=flow_dates, y=flow_ty, marker_color=flow_colors,
                marker_line_width=0, opacity=0.9, showlegend=False,
                hovertemplate="%{x|%d/%m}: %{y:+,.2f} tỷ<extra>KLNN ròng</extra>",
            ),
            row=3, col=1,
        )
        fig.add_hline(y=0, row=3, col=1, line_color=_TV_GRID, line_width=1)

    # --- Guides: today divider, forecast wash, T+5 / T+20 markers ------------
    # NOTE: explicit add_shape/add_annotation instead of the add_vline/add_hline
    # helpers — those helpers compute an annotation midpoint as (x0+x1)/2, which
    # raises TypeError on datetime.date x values (plotly quirk). yref="paper"
    # makes every vertical guide span ALL panes, terminal-style.
    def _v_guide(d: date, label: str, dash: str, position_top: bool) -> None:
        fig.add_shape(
            type="line", x0=d, x1=d, y0=0, y1=1, yref="paper", layer="below",
            line=dict(color=_GUIDE_COLOR, width=1, dash=dash),
        )
        # Bottom labels sit slightly INSIDE the plot area (y=0.02) — at y=0
        # they collide with the x-axis date tick labels.
        fig.add_annotation(
            x=d, y=1.0 if position_top else 0.02, yref="paper",
            yanchor="bottom",
            text=label, showarrow=False,
            font=dict(size=10, color=_TEXT_MUTED, family=_FONT_FAMILY),
        )

    fig.add_shape(  # forecast-region wash
        type="rect", x0=last_date, x1=fwd_dates[-1], y0=0, y1=1, yref="paper",
        fillcolor=_FORECAST_ZONE, line_width=0, layer="below",
    )
    _v_guide(last_date, "Hôm nay", "dash", position_top=True)
    if len(fwd_dates) >= 5:
        _v_guide(fwd_dates[4], "T+5", "dot", position_top=False)
    _v_guide(fwd_dates[-1], f"T+{len(fwd_dates)}", "dot", position_top=False)
    # Dotted last-price reference across the price pane (TradingView habit).
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=s0, y1=s0, yref="y",
        layer="below", line=dict(color=_GUIDE_COLOR, width=1, dash="dot"),
    )

    # --- Terminal outcome labels at the cone edges + median endpoint ---------
    def _pct_txt(v: float) -> str:
        pct = (v / s0 - 1.0) * 100.0 if s0 else 0.0
        return f"{v:,.1f} ({pct:+.1f}%)"

    _annot_common = dict(
        x=fwd_dates[-1], xanchor="left", xshift=4, yref="y", showarrow=False,
        font=dict(size=10, family=_FONT_FAMILY),
    )
    fig.add_annotation(
        y=cone.upper[-1], text=_pct_txt(cone.upper[-1]),
        font_color=_CANDLE_UP, **_annot_common,
    )
    fig.add_annotation(
        y=cone.lower[-1], text=_pct_txt(cone.lower[-1]),
        font_color=_CANDLE_DOWN, **_annot_common,
    )
    fig.add_annotation(
        y=proj.median[-1], text=_pct_txt(proj.median[-1]),
        font_color=_MEDIAN_NEON, **_annot_common,
    )

    # --- TradingView touches: OHLC readout, watermark, last-price tag --------
    if ohlc:
        last_row = ohlc[-1]
        prev_close = ohlc[-2][4] if len(ohlc) > 1 else last_row[4]
        chg = (last_row[4] / prev_close - 1.0) if prev_close else 0.0
        day_color = _CANDLE_UP if last_row[4] >= last_row[1] else _CANDLE_DOWN
        fig.add_annotation(
            x=0.01, y=0.99, xref="x domain", yref="y domain",
            xanchor="left", yanchor="top", showarrow=False, align="left",
            text=(
                f"O <b>{last_row[1]:,.2f}</b>  H <b>{last_row[2]:,.2f}</b>  "
                f"L <b>{last_row[3]:,.2f}</b>  C <b>{last_row[4]:,.2f}</b>  "
                f"({chg:+.2%})"
            ),
            font=dict(size=11, color=day_color, family=_FONT_FAMILY),
        )
    if ticker:
        fig.add_annotation(
            x=0.5, y=0.55, xref="x domain", yref="y domain", showarrow=False,
            text=ticker, font=dict(size=64, color=_WATERMARK, family=_FONT_FAMILY),
        )
    fig.add_annotation(
        x=1.0, xref="paper", xanchor="left", y=s0, yref="y", yanchor="middle",
        text=f"{s0:,.2f}", showarrow=False,
        font=dict(size=10, color="#e6e8eb", family=_FONT_FAMILY),
        bgcolor=_PRICE_BADGE_BG, borderpad=3,
    )

    # --- Pane captions (top-left of the volume / flow panes) ------------------
    fig.add_annotation(
        x=0.005, y=0.97, xref="x2 domain", yref="y2 domain",
        xanchor="left", yanchor="top", showarrow=False,
        text="Khối lượng", font=dict(size=10, color=_TEXT_MUTED),
    )
    if has_flow:
        fig.add_annotation(
            x=0.005, y=0.97, xref="x3 domain", yref="y3 domain",
            xanchor="left", yanchor="top", showarrow=False,
            text="Khối ngoại ròng (tỷ VND)", font=dict(size=10, color=_TEXT_MUTED),
        )

    title = (
        f"Quỹ đạo dự phóng biến động giá — {ticker}"
        if ticker
        else "Quỹ đạo dự phóng biến động giá"
    )
    # Default viewport: recent history + the whole fan (see _DEFAULT_VIEW_SESSIONS).
    view_start = (
        hist_dates[-_DEFAULT_VIEW_SESSIONS]
        if len(hist_dates) > _DEFAULT_VIEW_SESSIONS else hist_dates[0]
    ) if hist_dates else last_date
    view_end = fwd_dates[-1] + timedelta(days=4)  # pad for the terminal labels

    fig.update_layout(
        template="plotly_dark",
        # Title pinned top-left; the horizontal legend is right-aligned and
        # pushed clear of the title.
        title=dict(text=title, y=0.99, yanchor="top", x=0),
        # Solid TV canvas (not transparent): the chart reads as an embedded
        # TradingView panel against the app's #0e1117 background.
        paper_bgcolor=_TV_BG,
        plot_bgcolor=_TV_BG,
        font=dict(family=_FONT_FAMILY, color=_TEXT_MUTED),
        margin=dict(t=90, b=30, l=10, r=95),  # r: terminal % labels + price scale
        height=700 if has_flow else 620,
        hovermode="x unified",
        bargap=0.25,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
            font=dict(size=11, color=_TEXT_MUTED),
        ),
    )
    # Shared-axis styling: gapless tape + crosshair spikes on every pane.
    # NOTE: a Candlestick trace force-enables its rangeslider by default —
    # explicitly off (broker terminals don't use one; it also ignores
    # rangebreaks, which would reintroduce the weekend gaps in miniature).
    fig.update_xaxes(
        type="date",
        gridcolor=_TV_GRID,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            *([dict(values=holiday_breaks)] if holiday_breaks else []),
        ],
        showspikes=True, spikemode="toaxis+across", spikethickness=1,
        spikecolor=_SPIKE_COLOR, spikedash="dot", spikesnap="cursor",
        rangeslider_visible=False,
    )
    fig.update_xaxes(range=[view_start, view_end], row=1, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="#1e222d", activecolor="#2dd4a7",
            font=dict(color="rgb(230,232,235)"),
            bordercolor=_TV_GRID, borderwidth=1,
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
        ),
        row=1, col=1,
    )
    # TV convention: price scale on the RIGHT, no axis titles (units live in
    # the tab caption / pane captions instead).
    fig.update_yaxes(side="right", gridcolor=_TV_GRID)
    fig.update_yaxes(
        showspikes=True, spikemode="toaxis+across", spikethickness=1,
        spikecolor=_SPIKE_COLOR, spikedash="dot",
        row=1, col=1,
    )
    return fig
