"""GIỮ tab — portfolio holdings (P2 wired).

Live portfolio CRUD against the ``portfolio`` DuckDB table (under a fixed local
user_id, P0 GOTCHA 2), live PnL via ``price_lookup.latest_close`` (VN
thousands-VND scale applied in ``_pnl_ratio``), and exit countdown via
``signal_ledger.list_open()``.

The add form pre-fills from the MUA tab's quick-add
(``st.session_state["giu_prefill"]``). Holdings reads are TTL-cached
(``st.cache_data ttl=30``); any add/remove clears the cache.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.utils.headless import (
    LOCAL_USER_ID,
    _pnl_ratio,
    portfolio_add,
    portfolio_list,
    portfolio_remove,
)


@st.cache_data(ttl=30)
def _cached_holdings(user_id: str) -> list[dict]:
    """TTL-cached holdings read (cleared on add/remove)."""
    return portfolio_list(user_id)


def _latest_close(ticker: str) -> float | None:
    """Lazy wrapper around price_lookup.latest_close (heavy import deferred)."""
    from src.data import price_lookup  # noqa: PLC0415 — lazy heavy import

    return price_lookup.latest_close(ticker)


def _open_positions() -> dict[str, dict]:
    """Map ticker → open-signal row (for exit countdown). Empty on failure."""
    try:
        from src.trading import signal_ledger  # noqa: PLC0415 — lazy heavy import

        return {p["ticker"]: p for p in signal_ledger.list_open()}
    except Exception:  # noqa: BLE001 — countdown is best-effort
        return {}


def render() -> None:
    """Render the GIỮ (portfolio holdings) tab."""
    st.header("GIỮ — Danh mục")
    st.caption("Quản lý vị thế đang nắm giữ.")

    # --- MUA quick-add pre-fill ----------------------------------------------
    prefill = st.session_state.pop("giu_prefill", None)
    if prefill:
        st.session_state["add_ticker"] = str(prefill.get("ticker", ""))
        st.session_state["add_price"] = float(prefill.get("price", 0.0) or 0.0)
        st.info(f"Đã điền sẵn {prefill.get('ticker')} từ tab MUA — nhập khối lượng và lưu.")

    # --- Add-position form ----------------------------------------------------
    with st.expander("➕ Thêm vị thế", expanded=bool(prefill)):
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
                try:
                    portfolio_add(LOCAL_USER_ID, new_ticker, int(new_volume), float(new_price))
                    _cached_holdings.clear()
                    st.success(f"Đã thêm {new_ticker} vào danh mục.")
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
            else:
                st.warning("Nhập đủ Mã, Khối lượng và Giá vào (> 0).")

    # --- Holdings + PnL + exit countdown -------------------------------------
    st.subheader("Vị thế hiện tại")
    holdings = _cached_holdings(LOCAL_USER_ID)
    if not holdings:
        st.info("Chưa có vị thế nào trong danh mục.")
        if st.button("🔄 Chạy lại", key="giu_rerun"):
            _cached_holdings.clear()
            st.rerun()
        return

    open_positions = _open_positions()

    rows: list[dict] = []
    ratios: list[float] = []
    von_vao = 0.0
    for h in holdings:
        ticker = h["ticker"]
        entry = float(h["price"] or 0.0)
        volume = int(h["volume"] or 0)
        von_vao += entry * volume
        ratio = _pnl_ratio(entry, _latest_close(ticker))
        if ratio is not None:
            ratios.append(ratio)
        countdown = open_positions.get(ticker, {}).get("sessions_remaining", "-")
        rows.append({
            "Mã": ticker,
            "KL": volume,
            "Giá vào": f"{entry:,.0f}",
            "PnL": f"{ratio:+.1%}" if ratio is not None else "N/A",
            "Còn lại (phiên)": countdown,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-row remove buttons.
    st.caption("Gỡ vị thế:")
    remove_cols = st.columns(len(holdings))
    for col, h in zip(remove_cols, holdings):
        if col.button(f"Gỡ {h['ticker']}", key=f"remove_{h['ticker']}"):
            portfolio_remove(LOCAL_USER_ID, h["ticker"])
            _cached_holdings.clear()
            st.toast(f"Đã gỡ {h['ticker']}.")
            st.rerun()

    # --- Summary cards --------------------------------------------------------
    st.subheader("Tổng quan")
    avg_pnl = (sum(ratios) / len(ratios)) if ratios else None
    summary_cols = st.columns(4)
    summary_cols[0].metric("Vốn vào", f"{von_vao:,.0f} ₫")
    summary_cols[1].metric(
        "PnL trung bình", f"{avg_pnl:+.1%}" if avg_pnl is not None else "N/A"
    )
    summary_cols[2].metric("Số mã có giá", f"{len(ratios)}/{len(holdings)}")
    summary_cols[3].metric("Lệnh mở", str(len(holdings)))

    # --- Re-run button --------------------------------------------------------
    if st.button("🔄 Chạy lại", key="giu_rerun"):
        _cached_holdings.clear()
        st.rerun()
