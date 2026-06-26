"""Streamlit boot/render smoke for the local dashboard (P2 live-render gate).

The P2 logic suite (``test_dashboard_persist_gate.py`` et al.) pins the headless
*logic* contracts, but it never actually rendered ``dashboard/app.py`` under
Streamlit — streamlit was not installed, so the "does the app boot and render
all six tabs without a render-time crash" gate stayed open in the handoff.

This module closes that gate deterministically with Streamlit's in-process
``AppTest`` harness. Every heavy seam is stubbed AT THE TAB USE-SITE (the tabs
bind the names at import time), so the smoke needs no models, no parquet, no
DuckDB, no Gemini, and no Telegram:

  * ``mua.run_in_thread``           → synchronous passthrough (no executor/rerun)
  * ``mua.daily_inference_headless``→ empty signal list (no ML inference)
  * ``ban.portfolio_list``          → empty book (BÁN short-circuits, no inference)
  * ``giu._cached_holdings``        → configurable holdings (bypasses st.cache_data)
  * ``giu._latest_close`` / ``_open_positions`` → no price/ledger I/O
  * ``audit._cached_postmortem``    → empty HTML (no Gemini / DB post-mortem)

What it proves: imports resolve under ``streamlit run``, ``set_page_config`` +
sidebar + ``st.tabs`` build, and all six ``render()`` functions execute their
widget construction without tripping the per-tab error boundary
(``st.error("Lỗi tab: …")``) or raising an uncaught exception.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip cleanly on a bare runner that lacks streamlit (conftest does not stub it).
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
_TAB_LABELS = ["MUA", "GIỮ", "BÁN", "Verify", "Audit", "Settings"]


def _sync_run_in_thread(fn, *args, label: str = "", ttl=None, **kwargs):
    """Deterministic stand-in for ``thread_runner.run_in_thread``.

    Calls ``fn`` inline — no ThreadPoolExecutor, no ``st.rerun`` polling loop,
    no ``time.sleep`` — so the boot is fast and free of thread/rerun flakiness.
    """
    return fn(*args, **kwargs)


def _boot(
    giu_holdings: list[dict] | None = None,
    *,
    requests: list[str] | None = None,
) -> AppTest:
    """Render ``dashboard/app.py`` under AppTest with all heavy seams stubbed.

    ``requests`` pre-sets the per-tab load-gate flags (e.g. ``["mua"]``) so a
    test can exercise the post-click render path; by default every gate is
    closed, matching a fresh app open.
    """
    holdings = list(giu_holdings or [])
    with ExitStack() as stack:
        # MUA — synchronous, empty buy-signal list (no daily_inference / models).
        stack.enter_context(
            patch("dashboard.tabs.mua.run_in_thread", _sync_run_in_thread)
        )
        stack.enter_context(
            patch("dashboard.tabs.mua.daily_inference_headless", lambda h: ("", []))
        )
        # BÁN — empty book → returns before any inference.
        stack.enter_context(patch("dashboard.tabs.ban.portfolio_list", lambda uid: []))
        # GIỮ — patch the cached wrapper directly to dodge cross-test cache reuse.
        stack.enter_context(
            patch("dashboard.tabs.giu._cached_holdings", lambda uid: list(holdings))
        )
        stack.enter_context(patch("dashboard.tabs.giu._latest_close", lambda t: 50.0))
        stack.enter_context(patch("dashboard.tabs.giu._open_positions", lambda: {}))
        # Audit — no Gemini / DB post-mortem.
        stack.enter_context(
            patch("dashboard.tabs.audit._cached_postmortem", lambda uid, days: "")
        )
        at = AppTest.from_file(_APP_PATH)
        for name in requests or []:
            at.session_state[f"_load_requested_{name}"] = True
        return at.run(timeout=30)


def _tab_boundary_errors(at: AppTest) -> list[str]:
    """Messages from the per-tab error boundary (``st.error('Lỗi tab: …')``)."""
    return [e.value for e in at.error if "Lỗi tab" in str(e.value)]


def test_app_boots_six_tabs_clean() -> None:
    """App renders all six tabs with no uncaught exception and no boundary error."""
    at = _boot()

    assert not at.exception, f"uncaught: {[e.value for e in at.exception]}"
    assert [t.label for t in at.tabs] == _TAB_LABELS
    assert not _tab_boundary_errors(at), (
        f"tab error boundary fired: {_tab_boundary_errors(at)}"
    )
    # Main shell rendered. Title now renders via the styled theme.page_header
    # (st.markdown), not st.title.
    assert any("Quant V4 Dashboard" in m.value for m in at.markdown)
    # Heavy tabs defer behind a load-gate on a fresh open — MUA shows its prompt
    # and does NOT run inference until requested.
    assert any("Bấm để tải tín hiệu mua" in i.value for i in at.info)


def test_gated_tabs_render_when_requested() -> None:
    """With the load-gate flags pre-set, the heavy tab seams run without error."""
    at = _boot(requests=["mua", "ban", "audit"])

    assert not at.exception, f"uncaught: {[e.value for e in at.exception]}"
    assert not _tab_boundary_errors(at), (
        f"tab error boundary fired: {_tab_boundary_errors(at)}"
    )
    # MUA seam executed (empty signals → "no signals" info, past the gate).
    assert any("Không có tín hiệu" in i.value for i in at.info)


def test_app_boots_with_one_holding_renders_giu() -> None:
    """A single GIỮ holding drives the holdings/PnL render path without error."""
    at = _boot(giu_holdings=[{"ticker": "VCB", "volume": 1000, "price": 50000.0}])

    assert not at.exception, f"uncaught: {[e.value for e in at.exception]}"
    assert not _tab_boundary_errors(at), (
        f"tab error boundary fired: {_tab_boundary_errors(at)}"
    )
    # GIỮ rendered its holdings rows (ticker + remove button) + summary metric.
    assert any("VCB" in m.value for m in at.markdown)
    assert any(b.label == "Gỡ" for b in at.button)
    assert any(m.label == "Vốn vào" for m in at.metric)
