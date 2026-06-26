"""Unit tests for the Telegram-HTML → dashboard-HTML report adapter.

The adapter is a pure string transform, so it is unit-testable without a
Streamlit runtime. ``render_report_html`` itself is a thin ``st.markdown``
wrapper exercised by the app smoke test.
"""

from __future__ import annotations

from dashboard.components.report_card import telegram_html_to_dashboard


def test_newlines_become_breaks() -> None:
    assert telegram_html_to_dashboard("a\nb") == "a<br>b"


def test_rule_bar_becomes_hr() -> None:
    # A run of box-drawing chars (the builders' separator) collapses to one <hr/>.
    assert telegram_html_to_dashboard("══════════════════════════════") == "<hr/>"


def test_rule_bar_absorbs_surrounding_breaks() -> None:
    # The \n\n padding the builders put around a rule must not leave dangling
    # <br> either side of the <hr/>.
    out = telegram_html_to_dashboard("top\n\n══════════════\n\nbottom")
    assert out == "top<hr/>bottom"


def test_inline_tags_pass_through() -> None:
    # <b>/<a>/<code> are kept verbatim — already html.escaped by the builder.
    raw = '<b>HPG</b> <a href="x">link</a> <code>X</code>'
    assert telegram_html_to_dashboard(raw) == raw


def test_excess_breaks_collapse_to_double() -> None:
    assert telegram_html_to_dashboard("a\n\n\n\nb") == "a<br><br>b"


def test_empty_input_is_empty() -> None:
    assert telegram_html_to_dashboard("") == ""
