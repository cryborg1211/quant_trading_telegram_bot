"""Quant V4 Dashboard — Streamlit entry point.

P1: static 6-tab shell. Sets page config, loads .env at startup (per P0
GOTCHA 3 — must happen before any serve module import, which arrives in P2),
renders a sidebar with status dots, then dispatches to the six tab modules via
``st.tabs``. Each ``render()`` call is wrapped in a try/except error boundary so
one broken tab cannot crash the whole app.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

# Load .env BEFORE importing any serve modules. P1 imports none, but this
# establishes the required order for P2 (config hot-reload, P0 GOTCHA 3).
load_dotenv(override=True)

from dashboard.tabs import (  # noqa: E402 - import after load_dotenv by design
    audit,
    ban,
    giu,
    mua,
    settings,
    verify,
)

st.set_page_config(page_title="Quant V4 Dashboard", layout="wide")

# Tab order matches the approved design.
_TAB_LABELS = ["MUA", "GIỮ", "BÁN", "Verify", "Audit", "Settings"]


def _render_sidebar() -> None:
    """Render the sidebar nav legend + static status dots (P1 placeholders)."""
    with st.sidebar:
        st.title("Quant V4")
        st.caption("Bảng điều khiển nội bộ — P1 (dữ liệu mẫu)")

        st.subheader("Trạng thái")
        # P1: static placeholder dots. P2 wires real checks (data freshness,
        # Gemini key present, Telegram reachable).
        st.markdown("🟢 Dữ liệu (mẫu)")
        st.markdown("🟡 Gemini (chưa kiểm tra)")
        st.markdown("🟡 Telegram (chưa kiểm tra)")

        st.divider()
        st.caption("Điều hướng bằng các tab phía trên.")


def main() -> None:
    """Build the dashboard shell and dispatch to each tab."""
    _render_sidebar()
    st.title("Quant V4 Dashboard")

    tab_mua, tab_giu, tab_ban, tab_verify, tab_audit, tab_settings = st.tabs(
        _TAB_LABELS
    )

    # Map each tab container to its render fn; isolate failures per tab.
    tab_renderers = [
        (tab_mua, mua.render),
        (tab_giu, giu.render),
        (tab_ban, ban.render),
        (tab_verify, verify.render),
        (tab_audit, audit.render),
        (tab_settings, settings.render),
    ]
    for tab, render_fn in tab_renderers:
        with tab:
            try:
                render_fn()
            except Exception as exc:  # noqa: BLE001 - error boundary per tab
                st.error(f"Lỗi tab: {exc}")


main()
