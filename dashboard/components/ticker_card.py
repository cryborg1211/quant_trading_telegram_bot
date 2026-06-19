"""Reusable per-ticker signal card component.

Renders a single ticker's signal as a card: action badge (MUA/GIỮ/BÁN),
price, the 3-segment up/side/down probability bar, sentiment, suggested
weight and hold/exit info. Optionally renders a "đã mua → thêm" quick-add
button.

P1: pure UI helper. Data comes in as plain args (stub data in P1; real
``dispatched_signals`` dict fields in P2).
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.signal_bar import render_signal_bar

# Action badge colors keyed by action label.
_ACTION_COLORS = {
    "MUA": "#1f9d55",
    "GIỮ": "#d69e2e",
    "GIU": "#d69e2e",
    "BÁN": "#e53e3e",
    "BAN": "#e53e3e",
}


def _action_badge(action: str) -> str:
    color = _ACTION_COLORS.get(action.upper(), "#4a5568")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-weight:700;font-size:13px;">{action}</span>'
    )


def render_ticker_card(
    ticker: str,
    action: str,
    price: float,
    prob_up: float,
    prob_side: float,
    prob_down: float,
    sentiment: float,
    weight_pct: float,
    hold_days: int,
    on_add_click: bool = False,
) -> bool:
    """Render a single ticker signal card.

    Args mirror the P2 ``dispatched_signals`` dict fields so wiring is a
    straight pass-through later. Returns True if the quick-add button was
    clicked this run (only possible when ``on_add_click`` is True), else False.
    """
    clicked = False
    with st.container(border=True):
        header_cols = st.columns([2, 1])
        with header_cols[0]:
            st.markdown(
                f"### {ticker} &nbsp; {_action_badge(action)}",
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            st.metric("Giá", f"{price:,.0f}")

        render_signal_bar(prob_up, prob_side, prob_down)

        info_cols = st.columns(3)
        info_cols[0].caption(f"Sentiment: **{sentiment:+.2f}**")
        info_cols[1].caption(f"Tỷ trọng: **{weight_pct:.1f}%**")
        info_cols[2].caption(f"Nắm giữ: **{hold_days} ngày**")

        if on_add_click:
            # P1: button renders and reports its click, but app does not wire
            # the result into session_state yet (that is P2 quick-add).
            clicked = st.button(
                "Đã mua → thêm",
                key=f"add_{ticker}",
                use_container_width=True,
            )
    return clicked
