"""Tầm Nhìn Thuật Toán — pure-technical fan-chart forecast.

Enter a ticker → instantly render a GBM probability fan (T+5 tight, T+20 wide)
built ONLY from the ticker's own recent closes. No Gemini, no LLM, no news
sentiment — so it renders in milliseconds and needs no API key. This tab is the
one place a user sees the raw quant forecast with the AI overlays stripped out.

Degrades cleanly: an empty input shows a hint; a ticker with no shard / < 2
closes shows a warning instead of a chart.
"""

from __future__ import annotations

import streamlit as st

from dashboard.utils.fan_chart import build_fan_figure, project_fan

_HISTORY_SESSIONS = 120
_HORIZON = 20


def _load_ohlc(ticker: str) -> list[tuple[object, float, float, float, float]]:
    """Recent ``(date, open, high, low, close)`` history (``[]`` on failure)."""
    from src.data import price_lookup  # noqa: PLC0415 — lazy heavy import

    return price_lookup.ohlc_history(ticker, n=_HISTORY_SESSIONS)


def render() -> None:
    """Render the Tầm Nhìn Thuật Toán (technical fan-chart) tab."""
    st.header("Tầm Nhìn Thuật Toán")
    st.caption(
        "Hệ thống mô phỏng Monte Carlo dựa trên mô hình hình học Geometric "
        "Brownian Motion (GBM) từ chuỗi tỷ suất sinh lợi lịch sử."
    )

    ticker = st.text_input("Mã cổ phiếu", key="fan_ticker").strip().upper()
    if not ticker:
        st.info("Nhập một mã để xem quạt dự báo T+5 / T+20.")
        return

    ohlc = _load_ohlc(ticker)
    if len(ohlc) < 2:
        st.warning(
            f"Không đủ dữ liệu giá cho {ticker} để dựng dự báo "
            "(cần ít nhất 2 phiên)."
        )
        return

    closes = [row[4] for row in ohlc]
    try:
        proj = project_fan(closes, horizon=_HORIZON)
    except ValueError:
        st.warning(f"Dữ liệu giá của {ticker} không hợp lệ để dự báo.")
        return

    # --- Metric strip (BELOW input, ABOVE chart) -----------------------------
    sigma_daily = proj.sigma
    current_price = proj.s0
    has_t5 = len(proj.median) >= 5
    t5_median = proj.median[4] if has_t5 else current_price
    t5_pct = (t5_median / current_price - 1.0) if current_price else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric(
        label="Độ biến động ngày (Volatility)",
        value=f"{sigma_daily:.2%}",
        help="σ tính từ chuỗi log-returns",
    )
    m2.metric(
        label="Expected T+5 Median",
        value=f"{t5_median:,.2f}",
        delta=f"{t5_pct:.1%}",
    )
    m3.metric(label=f"Thị giá hiện tại ({ticker})", value=f"{current_price:,.2f}")

    fig = build_fan_figure(ohlc, proj, ticker=ticker)
    # theme=None: keep the figure's own dark layout (Streamlit's theme override
    # would otherwise restyle the fan colors).
    st.plotly_chart(fig, use_container_width=True, theme=None)
