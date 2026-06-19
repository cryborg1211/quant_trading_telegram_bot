"""BÁN tab — sell verdicts + rebalance advice.

P1: static skeleton with STUB data. Renders per-holding sell/hold verdict cards
(reusing the ticker card with action="BÁN"/"GIỮ") and a stub rebalance advice
block in an ``st.info`` box.

P2 TODO:
  - sell/hold verdicts → ``main.inference_for_holdings(holding_tickers)`` which
    returns the SELL/HOLD report (HEADLESS-OK per P0 — no DB write, no telegram).
  - rebalance advice → same fn through the Gemini arbitrator path.
  - holdings list comes from the portfolio table (shared with GIỮ).
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.ticker_card import render_ticker_card

# --- STUB DATA (P1 only) ------------------------------------------------------
STUB_VERDICTS: list[dict] = [
    {
        "ticker": "HPG",
        "action": "GIỮ",
        "price": 27500.0,
        "prob_up": 0.51,
        "prob_side": 0.34,
        "prob_down": 0.15,
        "sentiment": 0.30,
        "weight_pct": 7.2,
        "hold_days": 12,
    },
    {
        "ticker": "FPT",
        "action": "GIỮ",
        "price": 138700.0,
        "prob_up": 0.49,
        "prob_side": 0.33,
        "prob_down": 0.18,
        "sentiment": 0.22,
        "weight_pct": 6.4,
        "hold_days": 4,
    },
    {
        "ticker": "SSI",
        "action": "BÁN",
        "price": 30930.0,
        "prob_up": 0.21,
        "prob_side": 0.27,
        "prob_down": 0.52,
        "sentiment": -0.34,
        "weight_pct": 0.0,
        "hold_days": 0,
    },
]

STUB_REBALANCE = (
    "Gợi ý tái cân bằng (mẫu — P1): Giảm SSI do tín hiệu giảm chiếm ưu thế "
    "(prob_down 52%, sentiment âm). Cân nhắc luân chuyển tỷ trọng sang HPG/FPT "
    "vốn vẫn ở trạng thái GIỮ. Đây là dữ liệu mẫu — P2 sẽ thay bằng khuyến nghị "
    "từ Gemini arbitrator."
)


def render() -> None:
    """Render the BÁN (sell / rebalance) tab."""
    st.header("BÁN — Khuyến nghị bán")
    st.caption("Đánh giá bán/giữ cho từng vị thế và gợi ý tái cân bằng (dữ liệu mẫu — P1).")

    st.subheader("Đánh giá bán/giữ")
    for v in STUB_VERDICTS:
        render_ticker_card(
            ticker=v["ticker"],
            action=v["action"],
            price=v["price"],
            prob_up=v["prob_up"],
            prob_side=v["prob_side"],
            prob_down=v["prob_down"],
            sentiment=v["sentiment"],
            weight_pct=v["weight_pct"],
            hold_days=v["hold_days"],
            on_add_click=False,
        )

    st.subheader("Tái cân bằng")
    st.info(STUB_REBALANCE)
