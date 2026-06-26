"""MUA tab — buy signals (P2 wired).

Renders a T+5 / T+20 horizon toggle and live buy-signal ticker cards from a
preview-safe inference run (``daily_inference_headless`` → ``main.daily_inference
(broadcast=False, persist=False)``). The heavy inference runs on a background
thread via ``run_in_thread`` (TTL-cached) so the UI does not freeze.

Quick-add: clicking "Đã mua → thêm" on a card stages the ticker + price in
``st.session_state["giu_prefill"]`` and reruns so the GIỮ tab pre-fills its add
form.
"""

from __future__ import annotations

import re

import streamlit as st

from dashboard.components.ticker_card import render_skeleton_cards, render_ticker_card
from dashboard.utils.headless import daily_inference_headless
from dashboard.utils.thread_runner import clear_cached, load_gate, run_in_thread

# Pull a leading integer out of a hold label like "30 phiên" / "T+20" / "20".
_HOLD_DAYS_RE = re.compile(r"(\d+)")


def _hold_days(sig: dict, default_horizon: int) -> int:
    """Best-effort numeric hold-days from the signal dict."""
    label = sig.get("hold_label")
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        m = _HOLD_DAYS_RE.search(label)
        if m:
            return int(m.group(1))
    return int(default_horizon)


def render() -> None:
    """Render the MUA (buy signals) tab."""
    st.header("MUA — Tín hiệu mua")
    st.caption("Danh sách cổ phiếu được mô hình khuyến nghị mua.")

    horizon_label = st.radio(
        "Khung thời gian",
        options=["T+5", "T+20"],
        index=1,
        horizontal=True,
        key="mua_horizon",
    )
    horizon = 5 if horizon_label == "T+5" else 20

    # Defer the ~1-min inference until the user asks — otherwise it would fire
    # the moment the app opens (every tab body runs on every rerun).
    if not load_gate(
        "mua",
        prompt="Bấm để tải tín hiệu mua (chạy mô hình, có thể mất ~1 phút).",
        button_label="Tải tín hiệu MUA",
    ):
        render_skeleton_cards(3)
        return

    if st.button("🔄 Làm mới", key="mua_refresh", type="tertiary"):
        clear_cached(daily_inference_headless, horizon)
        st.rerun()

    # Background, TTL-cached inference. Returns (report_html, signal_list).
    _html, signal_list = run_in_thread(
        daily_inference_headless,
        horizon,
        label=f"Tính tín hiệu MUA (T+{horizon})...",
        ttl=300,
    )

    if not signal_list:
        st.info("Không có tín hiệu mua cho khung thời gian này.")
        return

    for sig in signal_list:
        clicked = render_ticker_card(
            ticker=str(sig.get("ticker", "N/A")),
            action=str(sig.get("action", "MUA")),
            price=float(sig.get("price", 0.0) or 0.0),
            prob_up=float(sig.get("prob_up", 0.0) or 0.0),
            prob_side=float(sig.get("prob_side", 0.0) or 0.0),
            prob_down=float(sig.get("prob_down", 0.0) or 0.0),
            sentiment=float(sig.get("sentiment_score", 0.0) or 0.0),
            weight_pct=float(sig.get("suggested_weight", 0.0) or 0.0) * 100.0,
            hold_days=_hold_days(sig, horizon),
            on_add_click=True,
        )
        if clicked:
            st.session_state["giu_prefill"] = {
                "ticker": str(sig.get("ticker", "")),
                "price": float(sig.get("price", 0.0) or 0.0),
            }
            st.toast(f"Đã chọn {sig.get('ticker')} → mở tab GIỮ để xác nhận.")
            st.rerun()
