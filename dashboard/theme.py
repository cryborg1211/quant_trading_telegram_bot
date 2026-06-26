"""Central dark-premium theme: palette constants + global CSS injection.

Single source of truth for dashboard colors so `app.py` and every component
agree. `inject_global_css()` is called once at the top of `app.py` to skin the
default Streamlit chrome into a Bloomberg-ish dark terminal.
"""

from __future__ import annotations

import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────
BG = "#0e1117"            # app background (charcoal)
SURFACE = "#1a1f2b"       # raised cards / panels
SURFACE_2 = "#222936"     # hover / secondary surface
BORDER = "#2a3142"        # hairline borders
TEXT = "#e6e8eb"          # primary text
MUTED = "#8b95a5"         # secondary / caption text

ACCENT = "#2dd4a7"        # teal — MUA / UP / primary action
AMBER = "#e0b341"         # GIỮ / SIDE
DANGER = "#f87171"        # BÁN / DOWN

# Action label → accent color (handles unaccented fallbacks too).
ACTION_COLORS = {
    "MUA": ACCENT,
    "GIỮ": AMBER, "GIU": AMBER,
    "BÁN": DANGER, "BAN": DANGER,
}


def action_color(action: str) -> str:
    return ACTION_COLORS.get(action.upper(), MUTED)


# ── Global CSS ───────────────────────────────────────────────────────────

_GLOBAL_CSS = f"""
<style>
  /* hide default Streamlit chrome */
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
  header[data-testid="stHeader"] {{background: transparent;}}

  /* base canvas */
  .stApp {{ background: {BG}; }}
  html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    color: {TEXT};
  }}

  /* headings */
  h1, h2, h3 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.7rem; }}

  /* tabs → pill row with accent underline on active */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.95rem;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; background: {SURFACE}; }}
  .stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT};
    background: {SURFACE};
  }}

  /* bordered containers → raised cards */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE};
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    padding: 4px 6px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    transition: border-color .15s ease, transform .15s ease;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: {ACCENT} !important;
    transform: translateY(-1px);
  }}

  /* metrics */
  div[data-testid="stMetric"] {{
    background: transparent;
    padding: 0;
  }}
  div[data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 700; }}
  div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}

  /* buttons → accent fill */
  .stButton > button {{
    background: {ACCENT};
    color: #06120e;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    transition: filter .15s ease;
  }}
  .stButton > button:hover {{ filter: brightness(1.08); color: #06120e; }}

  /* sidebar */
  section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
  }}

  /* captions */
  .stCaption, small {{ color: {MUTED}; }}

  /* dividers */
  hr {{ border-color: {BORDER}; }}

  /* serve-path report card (BÁN / Verify / Audit Telegram-HTML adapter) */
  .qv-report {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 20px;
    color: {TEXT};
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
  }}
  .qv-report-title {{
    font-size: 1.05rem; font-weight: 800; color: {TEXT}; margin-bottom: 10px;
  }}
  .qv-report b {{ color: {TEXT}; font-weight: 700; }}
  .qv-report a {{ color: {ACCENT}; text-decoration: none; }}
  .qv-report a:hover {{ text-decoration: underline; }}
  .qv-report code {{
    background: {SURFACE_2}; color: {AMBER}; padding: 1px 6px;
    border-radius: 6px; font-size: 0.85em;
  }}
  .qv-report hr {{
    border: none; border-top: 1px solid {BORDER}; margin: 12px 0;
  }}
</style>
"""


def inject_global_css() -> None:
    """Inject the dark-premium global stylesheet. Call once near app start."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a polished gradient page header band."""
    sub = (
        f'<div style="color:{MUTED};font-size:0.9rem;margin-top:2px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="padding:14px 18px;border-radius:14px;margin-bottom:14px;
             background:linear-gradient(135deg,{SURFACE} 0%,{BG} 100%);
             border:1px solid {BORDER};">
          <div style="font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;
               color:{TEXT};">{title}</div>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )
