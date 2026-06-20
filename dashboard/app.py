"""Quant V4 Dashboard — Streamlit entry point.

P1: static 6-tab shell. Sets page config, loads .env at startup (per P0
GOTCHA 3 — must happen before any serve module import, which arrives in P2),
renders a sidebar with status dots, then dispatches to the six tab modules via
``st.tabs``. Each ``render()`` call is wrapped in a try/except error boundary so
one broken tab cannot crash the whole app.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env BEFORE importing any serve modules so env-keyed config + the status
# dots below see the user's secrets (P0 GOTCHA 3).
load_dotenv(override=True)

# Repo root = one level up from this file (dashboard/app.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
_DATA_FRESH_SECONDS = 86400 * 3  # parquet considered fresh if < 3 days old

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


def _data_freshness_dot() -> str:
    """Green if any data/ohlcv_*.parquet is < 3 days old, yellow otherwise."""
    try:
        shards = list(_DATA_DIR.glob("ohlcv_*.parquet"))
        if not shards:
            return "🟡 Dữ liệu (không tìm thấy parquet)"
        newest = max(p.stat().st_mtime for p in shards)
        age = time.time() - newest
        if age < _DATA_FRESH_SECONDS:
            return "🟢 Dữ liệu (mới)"
        return f"🟡 Dữ liệu (cũ {age / 86400:.0f} ngày)"
    except Exception:  # noqa: BLE001 — status must never crash the sidebar
        return "🟡 Dữ liệu (chưa kiểm tra)"


def _env_dot(var: str, label: str) -> str:
    """Green if the env var is present and non-empty, red otherwise."""
    return f"🟢 {label}" if os.environ.get(var) else f"🔴 {label} (thiếu khóa)"


def _render_sidebar() -> None:
    """Render the sidebar nav legend + real status dots."""
    with st.sidebar:
        st.title("Quant V4")
        st.caption("Bảng điều khiển nội bộ")

        st.subheader("Trạng thái")
        st.markdown(_data_freshness_dot())
        st.markdown(_env_dot("GEMINI_API_KEY", "Gemini"))
        st.markdown(_env_dot("TELEGRAM_BOT_TOKEN", "Telegram"))

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
