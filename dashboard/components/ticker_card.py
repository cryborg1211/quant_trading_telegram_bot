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
from dashboard.theme import action_color


def render_skeleton_cards(n: int = 3) -> None:
    """Render ``n`` shimmering placeholder cards (pre-load empty state)."""
    st.markdown(
        '<div class="qv-skel"></div>' * max(0, n),
        unsafe_allow_html=True,
    )


def _action_badge(action: str) -> str:
    color = action_color(action)
    return (
        f'<span style="background:{color};color:#06120e;padding:3px 12px;'
        f'border-radius:999px;font-weight:800;font-size:12px;'
        f'letter-spacing:0.02em;">{action}</span>'
    )


def _render_oversight_row(sig: dict) -> None:
    """Second info row: the Telegram card's 'Lớp giám sát' fields."""
    regime_label = sig.get("regime_label")
    regime_id = sig.get("market_regime")
    tier_label = sig.get("risk_tier_label_vi")
    tier_pct = sig.get("risk_tier_pct")
    garch = sig.get("garch_scalar")

    cols = st.columns(3)
    if regime_label and regime_id is not None:
        cols[0].caption(f"Pha thị trường: **{regime_label}** (R{int(regime_id)})")
    else:
        cols[0].caption("Pha thị trường: —")
    if tier_label and tier_pct is not None:
        cols[1].caption(f"Hạng rủi ro: **{tier_label}** (trần {int(tier_pct)}%)")
    else:
        cols[1].caption("Hạng rủi ro: —")
    garch_txt = "—"
    if garch is not None:
        try:
            g = float(garch)
            garch_txt = f"**×{g:.2f}**" if g < 1.0 else "**tắt (×1.00)**"
        except (TypeError, ValueError):
            garch_txt = "—"
    cols[2].caption(f"Phanh GARCH: {garch_txt}")


def _render_analysis_expander(sig: dict) -> None:
    """Collapsible full-detail section — parity with the Telegram BUY card."""
    with st.expander("Phân tích chi tiết & tin tức"):
        base = sig.get("base_decision_vi")
        if base:
            st.caption(f"Phân loại gốc của mô hình: **{base}**")
        regime_action = sig.get("regime_action_vi")
        if regime_action:
            st.caption(f"Điều chỉnh theo pha: {regime_action}")
        arb = sig.get("arb_note_vi")
        if arb:
            st.caption(f"Trọng tài tin tức: {arb}")
        sent_status = sig.get("sentiment_status")
        if sent_status:
            st.caption(f"Tin tức & tâm lý: **{sent_status}**")

        conclusion = sig.get("conclusion") or sig.get("gemini_summary")
        if conclusion:
            st.markdown(f"**Nhận định:** {conclusion}")

        pos = sig.get("top_pos_features")
        neg = sig.get("top_neg_features")
        if pos:
            st.caption(pos)
        if neg:
            st.caption(neg)
        exit_rule = sig.get("exit_rule")
        if exit_rule:
            st.caption(f"Quy tắc thoát: {exit_rule}")

        urls = sig.get("article_urls") or []
        if urls:
            links = " · ".join(
                f"[{i}]({u})" for i, u in enumerate(urls[:6], start=1)
            )
            st.caption(f"Nguồn tham khảo: {links}")


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
    sig: dict | None = None,
) -> bool:
    """Render a single ticker signal card.

    Args mirror the P2 ``dispatched_signals`` dict fields so wiring is a
    straight pass-through later. When the full ``sig`` dict is supplied, the
    card also renders the Telegram-card parity sections: event banner, the
    'Lớp giám sát' oversight row (regime / risk tier / GARCH brake) and a
    collapsible full-analysis block (nhận định, tin tức, nguồn). Returns True
    if the quick-add button was clicked this run (only possible when
    ``on_add_click`` is True), else False.
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

        # Event-rescue banner (non-standard signals only) — mirrors the
        # Telegram card's ⚡ line.
        if sig:
            status = str(sig.get("status") or "")
            if status and status != "MUA":
                st.warning(f"⚡ {status} — {sig.get('ly_do', '')}")

        render_signal_bar(prob_up, prob_side, prob_down)

        info_cols = st.columns(3)
        info_cols[0].caption(f"Sentiment: **{sentiment:+.2f}**")
        info_cols[1].caption(f"Tỷ trọng: **{weight_pct:.1f}%**")
        info_cols[2].caption(f"Nắm giữ: **{hold_days} ngày**")

        if sig:
            _render_oversight_row(sig)
            _render_analysis_expander(sig)

        if on_add_click:
            # P1: button renders and reports its click, but app does not wire
            # the result into session_state yet (that is P2 quick-add).
            clicked = st.button(
                "Đã mua → thêm",
                key=f"add_{ticker}",
                use_container_width=True,
            )
    return clicked
