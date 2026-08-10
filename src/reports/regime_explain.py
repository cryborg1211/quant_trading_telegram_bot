"""Plain-Vietnamese explanation of WHY a market regime forces a size action.

The signal card used to print only "Pha thị trường: Đi Ngang (Regime 6)" and
a bare multiplier, which tells an operator what happened but not why. This
module supplies the missing sentence: what the regime physically IS (the
detector's own criterion, from `src/features/market_regime.py`) and what the
sizing policy does about it (from `src/trading/regime_policy.py`).

Both sources are authoritative and imported, never restated as literals, so
a policy change (e.g. moving a regime in or out of PENALTY_REGIMES) flows
into the wording automatically instead of silently going stale.
"""

from __future__ import annotations

from src.features.market_regime import REGIME_LABELS_VI
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_FACTOR,
    STRONG_TREND_REGIME,
)

# What the DETECTOR actually looks for, per regime — phrased from the criteria
# documented in market_regime.py's module docstring, not invented.
_WHAT_IT_IS: dict[int, str] = {
    0: "ATR và khối lượng cực thấp — thị trường gần như không giao dịch",
    1: "dải Bollinger co hẹp — giá đang nén, chưa chọn hướng",
    2: "giá vừa phá dải trên kèm khối lượng tăng",
    3: "hiệu suất xu hướng cao, các đường trung bình xếp thẳng hàng",
    4: "bùng nổ: giá vượt xa dải trên + khối lượng đột biến + RSI nóng",
    5: "RSI ở vùng cực trị (dưới 30 hoặc trên 70)",
    6: "hiệu suất xu hướng thấp, không có cấu trúc rõ ràng",
    7: "nến biên độ rộng nhưng thân nhỏ — bóng nến dài hai đầu",
}

# Why that structure justifies the action the policy takes.
_WHY_ACTION: dict[int, str] = {
    0: "không có thanh khoản để vào/ra, lệnh dễ bị kẹt",
    1: "chưa biết nén xong sẽ bung lên hay xuống",
    2: "xu hướng mới hình thành, chưa được xác nhận",
    3: "điều kiện thuận lợi nhất cho việc giữ vị thế",
    4: "giai đoạn dễ đảo chiều đột ngột sau khi cạn lực mua",
    5: "giá đã đi quá xa, khả năng bật lại cao",
    6: "giá dao động không hướng, tín hiệu dễ bị nhiễu",
    7: "dấu hiệu quét stop-loss — giá đâm sâu rồi hồi ngay trong phiên",
}


def regime_action_label(regime: int | None) -> str:
    """Short label for what the sizing policy does in this regime."""
    if regime is None:
        return "không xác định"
    r = int(regime)
    if r in NO_TRADE_REGIMES:
        return "KHÔNG giao dịch (bỏ qua mã này)"
    if r in PENALTY_REGIMES:
        return f"giảm tỷ trọng ×{REGIME_PENALTY_FACTOR:.2f}"
    if r == STRONG_TREND_REGIME:
        return "cho phép tỷ trọng tối đa"
    return "giữ tỷ trọng bình thường"


def regime_explain_lines(regime: int | None) -> list[str]:
    """Two-line explanation: what the regime is, and why it drives the action.

    Returns [] for an unknown/absent regime so the caller can omit the block
    entirely rather than print a placeholder.
    """
    if regime is None:
        return []
    r = int(regime)
    if r not in REGIME_LABELS_VI:
        return []

    label = REGIME_LABELS_VI[r]
    what = _WHAT_IT_IS.get(r, "")
    why = _WHY_ACTION.get(r, "")
    action = regime_action_label(r)

    lines = [f"Pha thị trường: {label} (Regime {r}) — {action}"]
    if what and why:
        lines.append(f"    {what}; {why}")
    elif what:
        lines.append(f"    {what}")
    return lines


__all__ = ["regime_explain_lines", "regime_action_label"]
