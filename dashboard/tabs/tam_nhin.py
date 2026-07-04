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

# DISPLAY window vs ESTIMATION window are deliberately different:
#   • display — ~2 years of candles so the ALL button shows real history;
#   • estimation — GBM mu/sigma from the LAST 120 sessions only. Widening the
#     estimation window would silently change the forecast (2-year-old drift
#     is not today's regime); widening the display window must not.
_HISTORY_SESSIONS_DISPLAY = 500
_ESTIMATION_SESSIONS = 120
_HORIZON = 20


def _load_ohlc(ticker: str) -> list[tuple[object, float, float, float, float, float]]:
    """Recent ``(date, O, H, L, C, volume)`` history (``[]`` on failure)."""
    from src.data import price_lookup  # noqa: PLC0415 — lazy heavy import

    return price_lookup.ohlc_history(ticker, n=_HISTORY_SESSIONS_DISPLAY)


def _load_foreign_flow(ticker: str) -> list[tuple[object, float]]:
    """Ascending ``(date, foreign_net_val)`` rows for the Khối ngoại pane.

    Reads ``data/foreign_flow_daily.parquet`` (thousands of VND, one row per
    ticker per day, forward-accumulated by the daily cron). Degrades to ``[]``
    on any failure — the chart simply renders without the flow pane.
    """
    from pathlib import Path  # noqa: PLC0415

    path = Path("data/foreign_flow_daily.parquet")
    if not path.exists():
        return []
    try:
        import polars as pl  # noqa: PLC0415 — lazy heavy import

        df = (
            pl.read_parquet(path)
            .filter(pl.col("ticker") == ticker)
            .select("date", "foreign_net_val")
            .drop_nulls()
            .sort("date")
        )
        return [(r[0], float(r[1])) for r in df.iter_rows()]
    except Exception:  # noqa: BLE001 — pane is optional, never break the tab
        return []


def render() -> None:
    """Render the Tầm Nhìn Thuật Toán (technical fan-chart) tab."""
    st.header("Tầm Nhìn Thuật Toán")
    st.caption(
        "Hệ thống mô phỏng Monte Carlo dựa trên mô hình hình học Geometric "
        "Brownian Motion (GBM) từ chuỗi tỷ suất sinh lợi lịch sử."
    )

    ticker = st.text_input(
        "Mã cổ phiếu", key="fan_ticker", placeholder="VD: HPG, FPT, VNM…",
    ).strip().upper()
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
        # Forecast estimated from the recent-regime window ONLY (see the
        # display-vs-estimation note on the constants above).
        proj = project_fan(closes[-_ESTIMATION_SESSIONS:], horizon=_HORIZON)
    except ValueError:
        st.warning(f"Dữ liệu giá của {ticker} không hợp lệ để dự báo.")
        return

    # --- Metric strip (BELOW input, ABOVE chart) -----------------------------
    sigma_daily = proj.sigma
    current_price = proj.s0
    has_t5 = len(proj.median) >= 5
    t5_median = proj.median[4] if has_t5 else current_price
    t5_pct = (t5_median / current_price - 1.0) if current_price else 0.0
    t20_median = proj.median[-1]
    t20_pct = (t20_median / current_price - 1.0) if current_price else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label=f"Thị giá hiện tại ({ticker})", value=f"{current_price:,.2f}")
    m2.metric(
        label="Trung vị T+5",
        value=f"{t5_median:,.2f}",
        delta=f"{t5_pct:.1%}",
    )
    m3.metric(
        label=f"Trung vị T+{len(proj.days)}",
        value=f"{t20_median:,.2f}",
        delta=f"{t20_pct:.1%}",
    )
    m4.metric(
        label="Biến động ngày (σ)",
        value=f"{sigma_daily:.2%}",
        help=(
            f"σ tính từ log-returns của {_ESTIMATION_SESSIONS} phiên gần nhất "
            "(cửa sổ ước lượng của dự phóng). "
            f"Quy năm ≈ {sigma_daily * (252 ** 0.5):.1%} (σ×√252)."
        ),
    )

    show_paths = st.toggle(
        "Hiện 12 kịch bản mô phỏng Monte Carlo",
        value=False,
        key="fan_show_paths",
        help=(
            "Mặc định biểu đồ chỉ vẽ 3 kịch bản (lạc quan / trung vị / thận "
            "trọng) theo phong cách báo cáo phân tích — bật để xem thêm 12 "
            "đường mô phỏng ngẫu nhiên."
        ),
    )
    fig = build_fan_figure(
        ohlc, proj, ticker=ticker,
        flow=_load_foreign_flow(ticker), show_paths=show_paths,
    )
    # theme=None: keep the figure's own dark layout (Streamlit's theme override
    # would otherwise restyle the fan colors). Modebar trimmed to the TV-like
    # essentials; scroll-to-zoom matches the TradingView muscle memory.
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d",
            ],
        },
    )
    st.caption(
        "Đơn vị giá: nghìn đồng · trục thời gian đã ẩn ngày nghỉ (T7/CN, lễ) "
        "· vùng tô = khoảng 80% xác suất · 3 kịch bản: lạc quan P90 (xanh) / "
        "trung vị (neon) / thận trọng P10 (đỏ) · nhãn phải = giá và "
        "%-thay-đổi kỳ vọng tại chân trời dự phóng."
    )
    with st.expander("📖 Cách đọc biểu đồ"):
        st.markdown(
            "- **Nến (xanh/đỏ):** lịch sử giá thực tế; mặc định hiển thị "
            "~45 phiên gần nhất — bấm 1M/3M/6M/ALL (≈2 năm) để xem xa hơn, "
            "lăn chuột để phóng to/thu nhỏ.\n"
            "- **Dự phóng** luôn ước lượng từ 120 phiên gần nhất, bất kể "
            "khung nhìn — xem xa hơn không làm đổi quạt dự báo.\n"
            "- **MA 10/20/50:** ba đường trung bình động (vàng/tím/xanh "
            "dương) — xu hướng ngắn/trung/dài hạn.\n"
            "- **Vùng tô nhạt:** dải mà giá có ~80% xác suất nằm trong, "
            "loe rộng dần theo √t (T+5 hẹp, T+20 rộng).\n"
            "- **3 kịch bản:** lạc quan P90 (viền xanh) / trung vị (nét đứt "
            "neon) / thận trọng P10 (viền đỏ) — cùng phong cách vùng giá "
            "mục tiêu trong báo cáo phân tích của các công ty chứng khoán.\n"
            "- **Kịch bản mô phỏng (tùy chọn):** bật công tắc phía trên để "
            "vẽ thêm 12 đường ngẫu nhiên GBM — xanh nếu kết thúc trên giá "
            "hiện tại, đỏ nếu dưới.\n"
            "- **Khung Khối lượng:** thanh khoản từng phiên, tô màu theo "
            "chiều nến.\n"
            "- **Khung Khối ngoại:** giá trị MUA/BÁN ròng của nhà đầu tư "
            "nước ngoài (tỷ VND) — dữ liệu tự thu thập mỗi phiên từ "
            "01/07/2026, dày dần theo thời gian.\n"
            "- **Mốc T+5 / T+20:** hai chân trời của hệ thống — T+5 dùng để "
            "xác nhận nhanh, T+20 là tín hiệu chính.\n"
            "- *Dự phóng thuần kỹ thuật (GBM) từ giá lịch sử — không gồm "
            "tin tức/AI; không phải khuyến nghị đầu tư.*"
        )
