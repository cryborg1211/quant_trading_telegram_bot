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


def _load_closes(ticker: str) -> list[float]:
    """Recent close series for ``ticker`` (empty list on any failure)."""
    from src.data import price_lookup  # noqa: PLC0415 — lazy heavy import

    history = price_lookup.close_history(ticker, n=_HISTORY_SESSIONS)
    return [c for _date, c in history]


def render() -> None:
    """Render the Tầm Nhìn Thuật Toán (technical fan-chart) tab."""
    st.header("Tầm Nhìn Thuật Toán")
    st.caption(
        "Dự báo kỹ thuật thuần túy (mô hình GBM từ chính lịch sử giá) — "
        "KHÔNG dùng AI / tin tức, vẽ tức thì."
    )

    ticker = st.text_input("Mã cổ phiếu", key="fan_ticker").strip().upper()
    if not ticker:
        st.info("Nhập một mã để xem quạt dự báo T+5 / T+20.")
        return

    closes = _load_closes(ticker)
    if len(closes) < 2:
        st.warning(
            f"Không đủ dữ liệu giá cho {ticker} để dựng dự báo "
            "(cần ít nhất 2 phiên)."
        )
        return

    try:
        proj = project_fan(closes, horizon=_HORIZON)
    except ValueError:
        st.warning(f"Dữ liệu giá của {ticker} không hợp lệ để dự báo.")
        return

    fig = build_fan_figure(closes, proj, ticker=ticker)
    # theme=None: keep the figure's own dark layout (Streamlit's theme override
    # would otherwise restyle the fan colors).
    st.plotly_chart(fig, use_container_width=True, theme=None)

    drift_pct = (proj.median[4] / proj.s0 - 1.0) * 100.0 if len(proj.median) >= 5 else 0.0
    st.caption(
        f"σ ngày ≈ {proj.sigma * 100:.2f}%  ·  trung vị T+5 ≈ {proj.median[4]:,.2f} "
        f"({drift_pct:+.1f}%)  ·  giá hiện tại {proj.s0:,.2f}"
        if len(proj.median) >= 5
        else f"σ ngày ≈ {proj.sigma * 100:.2f}%  ·  giá hiện tại {proj.s0:,.2f}"
    )
