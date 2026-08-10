"""Transparent checklist score + disqualifier warnings for a signal card.

WHY A CHECKLIST AND NOT A NUMBER FROM A MODEL
─────────────────────────────────────────────
The operator decides every trade, so what they need is not another
probability — they already have two — but a fast read on how many
independent things agree, and which things actively disagree. Every item
below is a boolean the operator can verify against the card's own
CẢNH BÁO and BỐI CẢNH blocks, so the score can never look more precise
than the evidence behind it.

Deliberately NOT a weighted composite: weights would be invented, and this
session established that P(UP) itself is mis-calibrated at T+20 (live check
10-08-26: predicted 39.2 pct vs realised 23.6 pct). Counting agreements is
honest; blending them into a fake probability is not.

WARNINGS are the disqualifiers established by this session's research:
  * room exhausted — T+20 forward return +0.52 pct vs +1.54 pct when room is
    available (p<1e-6, 207 tickers, 15.4 pct of the universe)
  * T+5-only — the T+20 gate does all the quality filtering; T+5-gated picks
    are ~noise, and all 13 losing July-2026 dispatches came through this door
  * no-trade regime — regime_policy says stand aside entirely
"""

from __future__ import annotations

from typing import Any

from src.trading.regime_policy import NO_TRADE_REGIMES, PENALTY_REGIMES

# Each entry: (key, label). `key` is looked up in the facts dict; a True value
# counts toward the score. Absent/None counts as NOT satisfied (never as a
# silent pass) so a missing input can only ever understate the score.
_CHECKS: tuple[tuple[str, str], ...] = (
    ("gate_20d", "Qua cổng T+20"),
    ("gate_5d", "Qua cổng T+5"),
    ("room_available", "Còn room ngoại"),
    ("regime_ok", "Pha thị trường không xấu"),
    ("no_brake", "Không bị phanh biến động"),
    ("breadth_ok", "Độ rộng thị trường ổn"),
    ("arbitrator_positive", "Trọng tài tin tức tích cực"),
    ("mr_fired", "Tín hiệu bắt đáy kích hoạt"),
)

TOTAL_CHECKS = len(_CHECKS)


def build_facts(
    *,
    p_up_20d: float | None = None,
    p_up_5d: float | None = None,
    tau_20d: float | None = None,
    tau_5d: float | None = None,
    room_exhausted: bool | None = None,
    market_regime: int | None = None,
    garch_scalar: float | None = None,
    breadth: float | None = None,
    breadth_low_cut: float = 0.41,
    sentiment_score: float | None = None,
    mr_fired: bool | None = None,
) -> dict[str, bool | None]:
    """Reduce raw serve-path values to the booleans the score counts.

    None is preserved (rather than coerced to False) wherever the input was
    genuinely unavailable, so callers can distinguish "failed the check" from
    "could not evaluate it" when rendering.
    """
    def _gate(p: float | None, tau: float | None) -> bool | None:
        if p is None or tau is None:
            return None
        return float(p) >= float(tau)

    return {
        "gate_20d": _gate(p_up_20d, tau_20d),
        "gate_5d": _gate(p_up_5d, tau_5d),
        "room_available": (None if room_exhausted is None else not room_exhausted),
        "regime_ok": (
            None if market_regime is None
            else int(market_regime) not in (NO_TRADE_REGIMES | PENALTY_REGIMES)
        ),
        "no_brake": (None if garch_scalar is None else float(garch_scalar) >= 0.999),
        "breadth_ok": (None if breadth is None else float(breadth) >= breadth_low_cut),
        "arbitrator_positive": (
            None if sentiment_score is None else float(sentiment_score) > 0.0
        ),
        "mr_fired": mr_fired,
    }


def score_signal(facts: dict[str, Any]) -> tuple[int, int]:
    """(passed, total) over the fixed checklist."""
    passed = sum(1 for key, _ in _CHECKS if facts.get(key) is True)
    return passed, TOTAL_CHECKS


def score_line(facts: dict[str, Any]) -> str:
    """`ĐIỂM TỔNG: 4/8` — the headline number only."""
    passed, total = score_signal(facts)
    return f"ĐIỂM TỔNG: {passed}/{total}"


def warning_lines(facts: dict[str, Any], *, market_regime: int | None = None) -> list[str]:
    """Disqualifiers worth interrupting the operator for. Empty when clean.

    Only the three research-backed ones are promoted to CẢNH BÁO; everything
    else stays in the context block, because a warning list that fires on
    every card trains the reader to ignore it.
    """
    out: list[str] = []
    if facts.get("room_available") is False:
        out.append("Hết room ngoại — nhóm này lịch sử kém hơn rõ rệt")
    if facts.get("gate_20d") is False and facts.get("gate_5d") is True:
        out.append("Chỉ qua cổng T+5, T+20 chưa mở — cửa này từng gây toàn bộ "
                   "lệnh lỗ tháng 7/2026")
    if market_regime is not None and int(market_regime) in NO_TRADE_REGIMES:
        out.append("Pha thị trường thuộc nhóm KHÔNG giao dịch")
    return out


__all__ = [
    "build_facts",
    "score_signal",
    "score_line",
    "warning_lines",
    "TOTAL_CHECKS",
]
