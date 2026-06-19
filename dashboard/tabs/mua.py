"""MUA tab — buy signals.

P1: static skeleton with STUB data. Renders a T+5 / T+20 horizon toggle and a
list of buy-signal ticker cards. The quick-add button renders but does nothing
in P1 (session_state wiring is P2).

P2 TODO: replace STUB_SIGNALS with a read-only call to a preview-safe variant
of ``main.daily_inference(broadcast=False, horizon=H)`` (see P0 GOTCHA 1 — the
current daily_inference still mutates DuckDB; a persist=False path is needed
before this tab calls it). The returned ``list[dict]`` fields map 1:1 onto
``render_ticker_card`` args.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.ticker_card import render_ticker_card

# --- STUB DATA (P1 only) ------------------------------------------------------
# Each dict matches the render_ticker_card signature plus a horizon key so the
# toggle can filter. In P2 this comes from real inference output.
STUB_SIGNALS: list[dict] = [
    {
        "ticker": "HPG",
        "action": "MUA",
        "price": 27500.0,
        "prob_up": 0.62,
        "prob_side": 0.25,
        "prob_down": 0.13,
        "sentiment": 0.41,
        "weight_pct": 7.2,
        "hold_days": 20,
        "horizon": 20,
    },
    {
        "ticker": "VHM",
        "action": "MUA",
        "price": 41200.0,
        "prob_up": 0.55,
        "prob_side": 0.31,
        "prob_down": 0.14,
        "sentiment": 0.18,
        "weight_pct": 6.0,
        "hold_days": 20,
        "horizon": 20,
    },
    {
        "ticker": "TCB",
        "action": "MUA",
        "price": 23150.0,
        "prob_up": 0.58,
        "prob_side": 0.22,
        "prob_down": 0.20,
        "sentiment": -0.05,
        "weight_pct": 5.5,
        "hold_days": 5,
        "horizon": 5,
    },
]


def render() -> None:
    """Render the MUA (buy signals) tab."""
    st.header("MUA — Tín hiệu mua")
    st.caption("Danh sách cổ phiếu được mô hình khuyến nghị mua (dữ liệu mẫu — P1).")

    horizon_label = st.radio(
        "Khung thời gian",
        options=["T+5", "T+20"],
        index=1,
        horizontal=True,
        key="mua_horizon",
    )
    horizon = 5 if horizon_label == "T+5" else 20

    visible = [s for s in STUB_SIGNALS if s["horizon"] == horizon]
    if not visible:
        st.info("Không có tín hiệu mua cho khung thời gian này (dữ liệu mẫu).")
        return

    for sig in visible:
        clicked = render_ticker_card(
            ticker=sig["ticker"],
            action=sig["action"],
            price=sig["price"],
            prob_up=sig["prob_up"],
            prob_side=sig["prob_side"],
            prob_down=sig["prob_down"],
            sentiment=sig["sentiment"],
            weight_pct=sig["weight_pct"],
            hold_days=sig["hold_days"],
            on_add_click=True,
        )
        if clicked:
            # P1: no-op acknowledgement. P2 pre-fills the GIỮ add form via
            # st.session_state and switches tabs.
            st.toast(f"(P1 mẫu) Đã chọn thêm {sig['ticker']} — chưa kết nối GIỮ.")
