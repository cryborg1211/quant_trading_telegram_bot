"""Verify tab — single-ticker dual-horizon check (P2 wired).

Enter a ticker, click "Kiểm tra" to run ``verify_single_ticker_headless``
(→ ``main.verify_single_ticker``, HEADLESS-OK) on a background thread. The
result HTML renders via ``st.markdown(unsafe_allow_html=True)`` and is cached in
session_state so the "Gửi Telegram" button can push it send-only via
``TelegramBot().send_text_alert`` (NEVER ``build_application`` — no polling). The
verify RUN logs a ``verify`` audit row under the local user_id (best-effort) so
the post-mortem Audit tab sees every check, not only the rare Telegram push.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.report_card import render_report_html
from dashboard.utils.headless import LOCAL_USER_ID, verify_single_ticker_headless
from dashboard.utils.thread_runner import run_in_thread


def render() -> None:
    """Render the Verify (single-ticker) tab."""
    st.header("Verify — Kiểm tra cổ phiếu")
    st.caption("Nhập mã để xem dự báo hai khung thời gian (T+5 / T+20).")

    ticker = st.text_input("Mã cổ phiếu", key="verify_ticker").strip().upper()

    if not ticker:
        st.info("Nhập một mã cổ phiếu rồi bấm Kiểm tra.")
        return

    result_key = f"verify_result_{ticker}"

    if st.button("🔍 Kiểm tra", key="verify_run"):
        # Force a fresh run for this ticker (drop any stale cached HTML).
        st.session_state.pop(result_key, None)
        html = run_in_thread(
            verify_single_ticker_headless,
            ticker,
            label=f"Kiểm tra {ticker}...",
            ttl=120,
        )
        st.session_state[result_key] = html or ""
        # Audit-log the verify at the moment it RUNS (not only on Telegram push)
        # so the post-mortem Audit tab reflects every recommendation the user
        # actually pulled. Best-effort — a logging failure must not break the tab.
        if html:
            try:
                from src.data.db_engine import DuckDBEngine  # noqa: PLC0415

                DuckDBEngine().log_user_action(LOCAL_USER_ID, "verify", ticker)
            except Exception:  # noqa: BLE001 — audit logging is best-effort
                pass

    html = st.session_state.get(result_key)
    if html:
        render_report_html(html, title=f"Kết quả cho {ticker}")
    elif html == "":
        st.warning(f"Không có kết quả cho {ticker} (mã ngoài vũ trụ giao dịch?).")
    else:
        # Ticker entered but not checked yet — a friendly placeholder beats a
        # blank gap, and the push button stays hidden until there is a result.
        st.info(f"Bấm **🔍 Kiểm tra** để xem dự báo T+5 / T+20 cho **{ticker}**.")
        return

    # Telegram push only makes sense once a real result exists.
    if html and st.button("📤 Gửi Telegram", key="verify_push", type="tertiary"):
        try:
            from src.utils.telegram_alerter import TelegramBot  # noqa: PLC0415

            TelegramBot().send_text_alert(html, label=ticker)
            st.success("Đã gửi.")
            # Note: the audit row is written on the verify RUN above, not here —
            # a Telegram push is optional and must not be the only audit trigger.
        except Exception as exc:  # noqa: BLE001 — surface send failure, no retry
            st.error(f"Gửi thất bại: {exc}")
