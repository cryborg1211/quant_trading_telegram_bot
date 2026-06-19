"""3-segment up/side/down probability bar component.

Renders a single horizontal bar split into three colored segments whose widths
are proportional to the up / side / down class probabilities. Used by
``ticker_card`` and any tab that wants to visualize a model's directional
distribution.

P1: pure UI helper, no state, no heavy imports.
"""

from __future__ import annotations

import streamlit as st

# Segment colors (up = green, side = amber, down = red).
_UP_COLOR = "#1f9d55"
_SIDE_COLOR = "#d69e2e"
_DOWN_COLOR = "#e53e3e"


def _normalize(prob_up: float, prob_side: float, prob_down: float) -> tuple[float, float, float]:
    """Clamp negatives to 0 and rescale so the three probabilities sum to 1.0.

    Falls back to an even three-way split when the inputs are degenerate
    (all zero / all negative), so the bar always renders something sensible.
    """
    up = max(0.0, float(prob_up))
    side = max(0.0, float(prob_side))
    down = max(0.0, float(prob_down))
    total = up + side + down
    if total <= 0.0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (up / total, side / total, down / total)


def render_signal_bar(prob_up: float, prob_side: float, prob_down: float) -> None:
    """Render a 3-segment horizontal bar for the up/side/down distribution.

    Proportions are normalized to sum to 1.0 (clamping any negatives first).
    """
    up, side, down = _normalize(prob_up, prob_side, prob_down)

    up_pct = round(up * 100)
    side_pct = round(side * 100)
    down_pct = round(down * 100)

    bar_html = (
        '<div style="display:flex;width:100%;height:18px;border-radius:4px;'
        'overflow:hidden;font-size:11px;line-height:18px;color:#fff;'
        'text-align:center;font-weight:600;">'
        f'<div style="width:{up * 100:.2f}%;background:{_UP_COLOR};">'
        f'{up_pct if up_pct >= 12 else ""}</div>'
        f'<div style="width:{side * 100:.2f}%;background:{_SIDE_COLOR};">'
        f'{side_pct if side_pct >= 12 else ""}</div>'
        f'<div style="width:{down * 100:.2f}%;background:{_DOWN_COLOR};">'
        f'{down_pct if down_pct >= 12 else ""}</div>'
        "</div>"
    )
    st.markdown(bar_html, unsafe_allow_html=True)
    st.caption(f"Tăng {up_pct}%  ·  Đi ngang {side_pct}%  ·  Giảm {down_pct}%")
