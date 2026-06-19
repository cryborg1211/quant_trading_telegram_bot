"""GIỮ tab — portfolio holdings.

P1: static skeleton with STUB data. Renders an add-position form, a holdings
table, summary metric cards, and a "Chạy lại" button. The add form writes to
``st.session_state["pending_add"]`` but does NOT touch the portfolio DB in P1.
Remove buttons and "Chạy lại" render but are no-ops in P1.

P2 TODO:
  - add/remove → raw DuckDB INSERT/DELETE on the ``portfolio`` table via
    ``DuckDBEngine()`` under a FIXED local user_id (see P0 GOTCHA 2). There is
    no add/remove API on PortfolioManager; the bot does raw SQL.
  - holdings table + PnL + sell verdict → ``main.inference_for_holdings(tickers)``.
  - exit countdown → ``signal_ledger.list_open()`` / ``check_exits_due()``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# --- STUB DATA (P1 only) ------------------------------------------------------
STUB_HOLDINGS: list[dict] = [
    {
        "ma": "HPG",
        "KL": 1000,
        "gia_vao": 26800.0,
        "PnL": "+2.6%",
        "lenh": "GIỮ",
        "thoat_countdown": "còn 12 phiên",
    },
    {
        "ma": "FPT",
        "KL": 300,
        "gia_vao": 132000.0,
        "PnL": "+5.1%",
        "lenh": "GIỮ",
        "thoat_countdown": "còn 4 phiên",
    },
    {
        "ma": "SSI",
        "KL": 1500,
        "gia_vao": 31500.0,
        "PnL": "-1.8%",
        "lenh": "BÁN",
        "thoat_countdown": "đến hạn",
    },
]

# Stub summary values (P2: computed from live holdings + PnL).
STUB_SUMMARY = {
    "von_vao": "126.000.000 ₫",
    "PnL_today": "+1.2%",
    "PnL_total": "+3.4%",
    "lenh_mo": "3",
}


def render() -> None:
    """Render the GIỮ (portfolio holdings) tab."""
    st.header("GIỮ — Danh mục")
    st.caption("Quản lý vị thế đang nắm giữ (dữ liệu mẫu — P1).")

    # --- Add-position form ----------------------------------------------------
    with st.expander("➕ Thêm vị thế", expanded=False):
        with st.form("add_position"):
            cols = st.columns(3)
            new_ticker = cols[0].text_input("Mã", key="add_ticker").strip().upper()
            new_volume = cols[1].number_input(
                "Khối lượng (KL)", min_value=0, step=100, value=0, key="add_volume"
            )
            new_price = cols[2].number_input(
                "Giá vào", min_value=0.0, step=100.0, value=0.0, key="add_price"
            )
            submitted = st.form_submit_button("Thêm vào danh mục")
        if submitted:
            if new_ticker and new_volume > 0 and new_price > 0:
                # P1: stage only — no PortfolioManager / DB write yet.
                st.session_state["pending_add"] = {
                    "ticker": new_ticker,
                    "volume": int(new_volume),
                    "price": float(new_price),
                }
                st.success(
                    f"(P1 mẫu) Đã ghi tạm {new_ticker} — chưa lưu vào danh mục."
                )
            else:
                st.warning("Nhập đủ Mã, Khối lượng và Giá vào (> 0).")

    # --- Holdings table -------------------------------------------------------
    st.subheader("Vị thế hiện tại")
    df = pd.DataFrame(STUB_HOLDINGS)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-row remove buttons (P1: no-op stubs).
    st.caption("Gỡ vị thế (mẫu — chưa hoạt động ở P1):")
    remove_cols = st.columns(len(STUB_HOLDINGS))
    for col, row in zip(remove_cols, STUB_HOLDINGS):
        if col.button(f"Gỡ {row['ma']}", key=f"remove_{row['ma']}"):
            st.toast(f"(P1 mẫu) Gỡ {row['ma']} — chưa kết nối DB.")

    # --- Summary cards --------------------------------------------------------
    st.subheader("Tổng quan")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Vốn vào", STUB_SUMMARY["von_vao"])
    summary_cols[1].metric("PnL hôm nay", STUB_SUMMARY["PnL_today"])
    summary_cols[2].metric("PnL tổng", STUB_SUMMARY["PnL_total"])
    summary_cols[3].metric("Lệnh mở", STUB_SUMMARY["lenh_mo"])

    # --- Re-run button (P1: no-op) -------------------------------------------
    if st.button("🔄 Chạy lại", key="giu_rerun"):
        st.toast("(P1 mẫu) Chạy lại — chưa kết nối inference.")
