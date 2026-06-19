"""Verify tab — single-ticker dual-horizon check.

P1: static skeleton with STUB data. Renders a ticker text input; when a symbol
is entered, shows a stub dual-horizon (T+5 / T+20) result card built from
metric columns. A "Gửi Telegram" push button renders but is a no-op in P1.

P2 TODO:
  - result → ``main.verify_single_ticker(ticker)`` (HEADLESS-OK per P0; only
    side-effect is an optional paperlog write wrapped in try/except).
  - push → ``telegram_alerter.TelegramBot().send_text_alert(html, label)``
    (send-only, no polling). NEVER call build_application() here.
"""

from __future__ import annotations

import streamlit as st


def _stub_horizon_result(ticker: str, horizon: int) -> dict:
    """Deterministic stub result so the same ticker looks stable across reruns."""
    seed = sum(ord(c) for c in ticker) + horizon
    prob_up = 0.40 + (seed % 25) / 100.0
    prob_down = 0.20 + (seed % 17) / 100.0
    sentiment = ((seed % 21) - 10) / 20.0
    return {
        "prob_up": round(prob_up, 2),
        "prob_down": round(prob_down, 2),
        "sentiment": round(sentiment, 2),
    }


def render() -> None:
    """Render the Verify (single-ticker) tab."""
    st.header("Verify — Kiểm tra cổ phiếu")
    st.caption("Nhập mã để xem dự báo hai khung thời gian (dữ liệu mẫu — P1).")

    ticker = st.text_input("Mã cổ phiếu", key="verify_ticker").strip().upper()

    if not ticker:
        st.info("Nhập một mã cổ phiếu để xem kết quả.")
        return

    st.subheader(f"Kết quả cho {ticker}")
    cols = st.columns(2)
    for col, (label, horizon) in zip(cols, [("T+5", 5), ("T+20", 20)]):
        result = _stub_horizon_result(ticker, horizon)
        with col:
            st.markdown(f"**{label}**")
            st.metric("Xác suất tăng", f"{result['prob_up'] * 100:.0f}%")
            st.metric("Xác suất giảm", f"{result['prob_down'] * 100:.0f}%")
            st.metric("Sentiment", f"{result['sentiment']:+.2f}")

    if st.button("📤 Gửi Telegram", key="verify_push"):
        # P1: no-op. P2 wires telegram_alerter.send_text_alert (send-only).
        st.toast(f"(P1 mẫu) Gửi Telegram cho {ticker} — chưa kết nối alerter.")
