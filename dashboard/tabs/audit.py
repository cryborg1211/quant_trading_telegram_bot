"""Audit tab — post-mortem hit-rate.

P1: static skeleton with STUB data. Renders a Tuần / Tháng toggle, a stub
post-mortem table, and two summary metrics (hit-rate, vs VNINDEX).

P2 TODO:
  - post-mortem → ``audit_evaluator.run_post_mortem(user_id, days)``
    (HEADLESS-OK per P0; needs GEMINI_API_KEY; pass the FIXED local user_id so
    rows written by the dashboard are found — see P0 GOTCHA 2).
  - Tuần = days≈7, Tháng = days≈30.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# --- STUB DATA (P1 only) ------------------------------------------------------
STUB_POSTMORTEM: list[dict] = [
    {
        "ma": "HPG",
        "lenh": "MUA",
        "gia_vao": 26800.0,
        "net_return": "+3.1%",
        "dung_sai": "Đúng",
    },
    {
        "ma": "VHM",
        "lenh": "MUA",
        "gia_vao": 40100.0,
        "net_return": "-1.4%",
        "dung_sai": "Sai",
    },
    {
        "ma": "FPT",
        "lenh": "GIỮ",
        "gia_vao": 132000.0,
        "net_return": "+5.0%",
        "dung_sai": "Đúng",
    },
    {
        "ma": "SSI",
        "lenh": "BÁN",
        "gia_vao": 31500.0,
        "net_return": "+0.8%",
        "dung_sai": "Đúng",
    },
]

# Stub summary keyed by window label.
STUB_SUMMARY = {
    "Tuần": {"hit_rate": "75%", "vs_vnindex": "+1.9%"},
    "Tháng": {"hit_rate": "68%", "vs_vnindex": "+3.2%"},
}


def render() -> None:
    """Render the Audit (post-mortem) tab."""
    st.header("Audit — Đánh giá lại")
    st.caption("Tỷ lệ đúng/sai của khuyến nghị đã qua (dữ liệu mẫu — P1).")

    window = st.radio(
        "Khoảng thời gian",
        options=["Tuần", "Tháng"],
        index=0,
        horizontal=True,
        key="audit_window",
    )

    st.subheader("Bảng đánh giá")
    df = pd.DataFrame(STUB_POSTMORTEM)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Tổng kết")
    summary = STUB_SUMMARY[window]
    cols = st.columns(2)
    cols[0].metric("Tỷ lệ đúng (hit-rate)", summary["hit_rate"])
    cols[1].metric("So với VNINDEX", summary["vs_vnindex"])
