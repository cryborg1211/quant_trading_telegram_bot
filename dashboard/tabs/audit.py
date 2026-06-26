"""Audit tab — post-mortem hit-rate (P2 wired).

Tuần / Tháng toggle → ``run_post_mortem(LOCAL_USER_ID, days)``
(HEADLESS-OK per P0; needs GEMINI_API_KEY; never raises — degrades to an inline
Vietnamese message). The result HTML is TTL-cached (``st.cache_data ttl=300``)
and rendered via ``st.markdown(unsafe_allow_html=True)``.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.report_card import render_report_html
from dashboard.utils.headless import LOCAL_USER_ID

_WINDOW_DAYS = {"Tuần": 7, "Tháng": 30}


@st.cache_data(ttl=300)
def _cached_postmortem(user_id: str, days: int) -> str:
    """TTL-cached post-mortem HTML (heavy import + Gemini call deferred)."""
    from src.utils.audit_evaluator import run_post_mortem  # noqa: PLC0415 — lazy

    return run_post_mortem(user_id, days=days)


def render() -> None:
    """Render the Audit (post-mortem) tab."""
    st.header("Audit — Đánh giá lại")
    st.caption("Tỷ lệ đúng/sai của khuyến nghị đã qua.")

    window = st.radio(
        "Khoảng thời gian",
        options=list(_WINDOW_DAYS.keys()),
        index=0,
        horizontal=True,
        key="audit_window",
    )
    days = _WINDOW_DAYS[window]

    html = _cached_postmortem(LOCAL_USER_ID, days)
    render_report_html(html)

    st.info(
        "Audit chỉ hiển thị dữ liệu từ phiên giao dịch dashboard — các lệnh "
        "từ bot Telegram sử dụng user_id khác."
    )
