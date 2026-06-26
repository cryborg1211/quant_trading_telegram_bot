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
from dataclasses import dataclass

import plotly.graph_objects as go

# Accent teal (matches dashboard.theme.ACCENT = #2dd4a7) as an RGB triple so we
# can vary the fill alpha per band.
_ACCENT_RGB = (45, 212, 167)
_MEDIAN_RGB = (45, 212, 167)
_HIST_RGB = (139, 149, 165)  # theme.MUTED

# z multipliers ≈ 50% / 80% / 95% central intervals (narrow → wide).
_DEFAULT_Z_LEVELS = (0.674, 1.282, 1.960)
# Per-band fill alpha, keyed by position narrow→wide: inner darkest, outer faint.
_BAND_ALPHAS = (0.34, 0.20, 0.10)


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


def build_fan_figure(
    closes: list[float],
    proj: FanProjection,
    *,
    ticker: str = "",
) -> go.Figure:
    """Build the dark-themed Plotly fan chart.

    The x-axis is a trading-session index: history runs ``-(N-1)…0`` (0 = today
    = T_now), the forecast runs ``0…H``. Anchoring every forward series at
    ``(0, S0)`` makes the fan emanate from a single point at today.
    """
    fig = go.Figure()

    # --- History (solid muted line up to today) ------------------------------
    hist_x = list(range(-(len(closes) - 1), 1))  # … -2, -1, 0
    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=list(closes),
            mode="lines",
            name="Lịch sử",
            line=dict(color=f"rgb{_HIST_RGB}", width=2),
            hovertemplate="Phiên %{x}: %{y:,.2f}<extra></extra>",
        )
    )

    s0 = proj.s0
    fwd_x = [0, *proj.days]  # anchor the fan at today (x=0, y=S0)

    # --- Fan bands (wide → narrow so the narrow inner band paints on top) -----
    # alphas are keyed narrow→wide; reverse alongside the reversed band order.
    n_bands = len(proj.bands)
    for idx in range(n_bands - 1, -1, -1):
        band = proj.bands[idx]
        alpha = _BAND_ALPHAS[idx] if idx < len(_BAND_ALPHAS) else 0.10
        upper = [s0, *band.upper]
        lower = [s0, *band.lower]
        # Upper edge: invisible line, no fill.
        fig.add_trace(
            go.Scatter(
                x=fwd_x, y=upper, mode="lines",
                line=dict(width=0), hoverinfo="skip",
                showlegend=False,
            )
        )
        # Lower edge: fill up to the immediately-preceding upper trace.
        fig.add_trace(
            go.Scatter(
                x=fwd_x, y=lower, mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba{(*_ACCENT_RGB, alpha)}",
                name=f"±{band.z:.2f}σ",
                hoverinfo="skip",
            )
        )

    # --- Median (dashed accent line) -----------------------------------------
    fig.add_trace(
        go.Scatter(
            x=fwd_x, y=[s0, *proj.median], mode="lines",
            name="Trung vị dự báo",
            line=dict(color=f"rgb{_MEDIAN_RGB}", width=2, dash="dash"),
            hovertemplate="T+%{x}: %{y:,.2f}<extra></extra>",
        )
    )

    # --- T_now / T+5 / T+20 reference markers --------------------------------
    fig.add_vline(x=0, line=dict(color="rgba(230,232,235,0.45)", width=1))
    for h, label in ((5, "T+5"), (proj.days[-1], f"T+{proj.days[-1]}")):
        if h <= proj.days[-1]:
            fig.add_vline(
                x=h, line=dict(color="rgba(139,149,165,0.45)", width=1, dash="dot")
            )
            fig.add_annotation(
                x=h, y=1.0, yref="paper", showarrow=False, text=label,
                font=dict(color="rgb(139,149,165)", size=11), yanchor="bottom",
            )

    title = f"Quạt dự báo kỹ thuật — {ticker}" if ticker else "Quạt dự báo kỹ thuật"
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgb(230,232,235)"),
        margin=dict(l=10, r=10, t=46, b=10),
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Phiên giao dịch (0 = hôm nay)",
            gridcolor="rgba(42,49,66,0.6)", zeroline=False,
        ),
        yaxis=dict(title="Giá (nghìn đồng)", gridcolor="rgba(42,49,66,0.6)"),
    )
    return fig
