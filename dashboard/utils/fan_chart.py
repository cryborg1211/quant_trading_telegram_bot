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
T+20 — producing the probability fan. Several nested ``z`` levels (≈50/80/95%)
give the layered look: the narrow inner band is drawn darkest, the wide outer
band lightest.

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

# z multipliers ≈ 50% / 80% / 95% central intervals (narrow → wide). Retained
# because `project_fan` still computes the analytic bands (unit-tested); the
# figure no longer DRAWS them — it renders Monte Carlo paths instead.
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


# TradingView-style palette.
_CANDLE_UP = "#22c55e"
_CANDLE_DOWN = "#ef4444"
_PATH_UP = "rgba(16, 185, 129, 0.2)"    # semi-transparent green (winning paths)
_PATH_DOWN = "rgba(239, 68, 68, 0.2)"   # semi-transparent red (losing paths)
_MEDIAN_NEON = "#00F2FE"                # bold neon median trajectory
_SPIKE_COLOR = "#A3AED0"
_TEXT_MUTED = "#A3AED0"                  # enterprise muted slate (axis/legend/title)
_FONT_FAMILY = "Inter, Segoe UI, San Francisco, Arial"
_N_MC_PATHS = 12
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
    ohlc: list[tuple[object, float, float, float, float]],
    proj: FanProjection,
    *,
    ticker: str = "",
    n_paths: int = _N_MC_PATHS,
) -> go.Figure:
    """Build the TradingView-style candlestick + Monte-Carlo-path forecast.

    ``ohlc`` is ascending ``(date, open, high, low, close)`` history. The
    forecast extends real business days past the last candle so the x-axis is a
    true datetime axis (required by the rangeselector month buttons). Instead of
    shaded confidence bands, the forecast is drawn as ``n_paths`` thin GBM walks
    (green = closes up, red = closes down) plus one bold neon median line.
    """
    fig = go.Figure()

    hist_dates = [row[0] for row in ohlc]
    last_date = hist_dates[-1] if hist_dates else date.today()

    # --- Candlestick history --------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=hist_dates,
            open=[row[1] for row in ohlc],
            high=[row[2] for row in ohlc],
            low=[row[3] for row in ohlc],
            close=[row[4] for row in ohlc],
            name="Lịch sử giá (OHLC)",
            increasing_line_color=_CANDLE_UP,
            increasing_fillcolor=_CANDLE_UP,
            decreasing_line_color=_CANDLE_DOWN,
            decreasing_fillcolor=_CANDLE_DOWN,
            whiskerwidth=0.5,
        )
    )

    s0 = proj.s0
    fwd_dates = _business_days_after(last_date, len(proj.days))
    fan_x = [last_date, *fwd_dates]  # anchor every forward series at today (S0)

    # --- Monte Carlo paths (thin, semi-transparent, branch from T_now) --------
    # Seed off the ticker so DCM and VHM get distinct shapes (not one cloned
    # wiggle scaled by price), yet each stays stable across reruns.
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
                showlegend=(i == 0),  # one legend entry for the whole bundle
                hoverinfo="skip",
            )
        )

    # --- Median trajectory (bold neon dashed) --------------------------------
    fig.add_trace(
        go.Scatter(
            x=fan_x,
            y=[s0, *proj.median],
            mode="lines",
            name="Kịch bản trung vị (Expected Median)",
            line=dict(color=_MEDIAN_NEON, width=3, dash="dash"),
            hovertemplate="%{x|%d/%m}: %{y:,.2f}<extra></extra>",
        )
    )

    title = (
        f"Quỹ đạo dự phóng biến động giá — {ticker}"
        if ticker
        else "Quỹ đạo dự phóng biến động giá"
    )
    fig.update_layout(
        template="plotly_dark",
        # Title pinned top-left; the horizontal legend is right-aligned and
        # pushed clear of the title AND the bottom range-selector buttons.
        title=dict(text=title, y=0.98, yanchor="top", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Enterprise font enforced globally (axis / legend / title inherit).
        font=dict(family=_FONT_FAMILY, color=_TEXT_MUTED),
        margin=dict(t=100, b=40, l=60, r=40),
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
            font=dict(size=11, color=_TEXT_MUTED),
        ),
        xaxis=dict(
            type="date",
            gridcolor="rgba(42,49,66,0.6)",
            # Professional crosshair: a dotted spike that tracks the cursor.
            showspikes=True, spikemode="toaxis+across", spikethickness=1,
            spikecolor=_SPIKE_COLOR, spikedash="dot", spikesnap="cursor",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                bgcolor="#1a1f2b", activecolor="#2dd4a7",
                font=dict(color="rgb(230,232,235)"),
                bordercolor="rgba(42,49,66,0.8)", borderwidth=1,
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
            ),
        ),
        yaxis=dict(
            title="Giá (nghìn đồng)", gridcolor="rgba(42,49,66,0.6)",
            showspikes=True, spikemode="toaxis+across", spikethickness=1,
            spikecolor=_SPIKE_COLOR, spikedash="dot",
        ),
    )
    return fig
