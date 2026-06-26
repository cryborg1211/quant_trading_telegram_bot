"""Adapter: render serve-path Telegram HTML inside the dark dashboard theme.

The BÁN / Verify / Audit tabs reuse the live bot's report builders
(``src/reports/builders.py``). Those builders emit *Telegram-flavored* HTML —
only ``<b>``/``<i>``/``<code>``/``<a>`` inline tags, literal ``\\n`` line breaks,
and ``══════`` rule-bars. Telegram honors the newlines and renders the runes as
separators; a browser does NOT (single ``\\n`` collapses, the runes show raw),
so dumping that string straight into ``st.markdown`` looks broken and clashes
with the dark theme.

This component is the dashboard-side fix. It must NEVER change the builder
output (that string is still sent verbatim to Telegram by the live bot). Instead
it translates the Telegram dialect into themed HTML at render time:

* ``══════`` rule-bars  → a single ``<hr/>``
* literal ``\\n``        → ``<br>`` (so line breaks survive)
* the whole thing       → wrapped in a ``.qv-report`` card (styled in theme.py)
"""

from __future__ import annotations

import re

import streamlit as st

# A Telegram rule-bar: a run of box-drawing / equals chars, optionally padded by
# the surrounding blank lines the builders emit around it.
_RULE_RE = re.compile(r"\n*[═=]{6,}\n*")
# 3+ consecutive breaks → at most a double break (kills the vertical gaps the
# Telegram \n\n spacing leaves behind once converted to <br>).
_EXCESS_BREAKS_RE = re.compile(r"(?:<br>\s*){3,}")
# Breaks hugging an <hr/> add nothing once the rule itself carries margin.
_RULE_PADDING_RE = re.compile(r"(?:<br>\s*)*<hr/>(?:<br>\s*)*")


def telegram_html_to_dashboard(raw_html: str) -> str:
    """Translate one Telegram-flavored report string into themed dashboard HTML.

    Pure string transform; the inline tags (``<b>`` etc.) pass through untouched
    and every dynamic field was already ``html.escape``d by the builder.
    """
    body = _RULE_RE.sub("<hr/>", raw_html)
    body = body.replace("\n", "<br>")
    body = _EXCESS_BREAKS_RE.sub("<br><br>", body)
    body = _RULE_PADDING_RE.sub("<hr/>", body)
    return body.strip()


def render_report_html(raw_html: str, title: str | None = None) -> None:
    """Render a serve-path Telegram HTML report as a themed dark card.

    No-op on empty input so callers can pass through ``""`` safely.
    """
    if not raw_html:
        return
    head = (
        f'<div class="qv-report-title">{title}</div>' if title else ""
    )
    body = telegram_html_to_dashboard(raw_html)
    st.markdown(
        f'<div class="qv-report">{head}{body}</div>',
        unsafe_allow_html=True,
    )
