"""BÁN tab — sell verdicts + rebalance advice (P2 wired).

Fetches the local portfolio holdings, then runs
``inference_for_holdings_headless`` (→ ``main.inference_for_holdings``,
HEADLESS-OK: no DB write, no Telegram) on a background thread. The serve path
returns a combined SELL/HOLD + rebalance HTML report, rendered directly via
``st.markdown(unsafe_allow_html=True)`` (D3: HTML→markdown for this tab).
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.report_card import render_report_html
from dashboard.utils.headless import (
    LOCAL_USER_ID,
    inference_for_holdings_headless,
    portfolio_list,
)
from dashboard.utils.thread_runner import clear_cached, load_gate, run_in_thread


def render() -> None:
    """Render the BÁN (sell / rebalance) tab."""
    st.header("BÁN — Khuyến nghị bán")
    st.caption("Đánh giá bán/giữ cho từng vị thế và gợi ý tái cân bằng.")

    holdings = portfolio_list(LOCAL_USER_ID)
    if not holdings:
        st.info("Chưa có vị thế nào trong danh mục — thêm từ tab GIỮ.")
        return

    tickers = [h["ticker"] for h in holdings]

    # Defer the holdings inference until requested (it re-runs on every rerun
    # otherwise — same hang as MUA).
    if not load_gate(
        "ban",
        prompt="Bấm để phân tích bán/giữ cho danh mục hiện tại.",
        button_label="Phân tích danh mục",
    ):
        return

    if st.button("🔄 Làm mới", key="ban_refresh", type="tertiary"):
        clear_cached(inference_for_holdings_headless, tickers)
        st.rerun()

    html = run_in_thread(
        inference_for_holdings_headless,
        tickers,
        label="Phân tích bán/giữ...",
        ttl=300,
    )

    if not html:
        st.info("Không có đánh giá nào cho danh mục hiện tại.")
        return

    render_report_html(html)
