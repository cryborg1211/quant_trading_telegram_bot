"""Pure-formatter tests for src.reports.builders.build_regret_report.

The builder does zero I/O: every row is a plain fixture dict — a cancelled ledger
row pre-enriched with evaluate_regret_pnl's GROSS `pct` / `matured` / `gap_flag`
(+ optional `error`). Covers exact line format, gap-flag variants, the
hypothetical/non-recommendation disclaimer surviving truncation (hard
requirement), section omission, error filtering, sort order, and 4096-char safety.
"""
from __future__ import annotations

from datetime import date

from src.reports.builders import build_regret_report

TODAY = date(2026, 7, 10)
LOOKBACK = 7


def _open_row(**kw) -> dict:
    base = {"ticker": "HPG", "screen_date": date(2026, 7, 3), "p_up": 0.30,
            "reason": "Cửa tăng thấp", "sessions_remaining": 4,
            "pct": 5.34, "matured": False}
    base.update(kw)
    return base


def _closed_row(**kw) -> dict:
    base = {"ticker": "FPT", "screen_date": date(2026, 7, 1), "p_up": 0.25,
            "reason": "Trọng tài từ chối", "pct": -2.67, "matured": True}
    base.update(kw)
    return base


def test_open_line_exact_format() -> None:
    out = build_regret_report([_open_row()], TODAY, LOOKBACK)
    assert ("Đã huỷ ngày 03/07/2026: mã HPG (P(tăng)=30%, lý do: Cửa tăng thấp) "
            "— nếu mua thì hiện lãi 5.3% (còn 4 ngày)") in out
    assert "⏳ <b>ĐANG THEO DÕI</b>" in out
    assert "ĐÃ KẾT THÚC" not in out            # matured section omitted when empty


def test_closed_line_exact_format() -> None:
    out = build_regret_report([_closed_row()], TODAY, LOOKBACK)
    assert "Đã huỷ ngày 01/07/2026: mã FPT — nếu mua thì đã lỗ 2.7%" in out
    assert "🏁 <b>ĐÃ KẾT THÚC (7 NGÀY QUA)</b>" in out
    assert "ĐANG THEO DÕI" not in out          # tracking section omitted when empty


def test_gap_flag_line_variant_open() -> None:
    out = build_regret_report(
        [_open_row(gap_flag=True, pct=-0.30, adjustment_factor=19.47 / 33.3)],
        TODAY, LOOKBACK)
    assert "lỗ 0.3%" in out
    assert "đã tự động điều chỉnh sự kiện DN" in out
    assert "hệ số 0.585" in out
    assert "giá bất thường" not in out
    assert "(còn 4 ngày)" in out                # suffix retained
    assert "P(tăng)=30%" in out                 # screening metadata still shown


def test_gap_flag_line_variant_closed() -> None:
    out = build_regret_report(
        [_closed_row(gap_flag=True, pct=4.70, adjustment_factor=0.36)],
        TODAY, LOOKBACK)
    assert "nếu mua thì đã lãi 4.7%" in out
    assert "đã tự động điều chỉnh sự kiện DN" in out
    assert "hệ số 0.360" in out
    assert "giá bất thường" not in out


def test_disclaimer_present_and_survives_truncation() -> None:
    # Synthetic 50-row fixture with long reasons forces report-level truncation;
    # the non-recommendation disclaimer must STILL be present (hard requirement).
    rows = [_open_row(ticker=f"T{i:02d}", sessions_remaining=i,
                      reason="lý do rất dài " * 10) for i in range(50)]
    out = build_regret_report(rows, TODAY, LOOKBACK)
    assert "KHÔNG phải khuyến nghị" in out
    assert len(out) <= 4096
    assert "tín hiệu khác (rút gọn)" in out


def test_both_empty_returns_empty_string() -> None:
    assert build_regret_report([], TODAY, LOOKBACK) == ""


def test_error_rows_excluded() -> None:
    out = build_regret_report(
        [_open_row(), {"ticker": "ERR", "error": "no price", "matured": False}],
        TODAY,
        LOOKBACK,
    )
    assert "HPG" in out
    assert "ERR" not in out


def test_open_rows_sorted_by_sessions_remaining() -> None:
    rows = [_open_row(ticker="AAA", sessions_remaining=5),
            _open_row(ticker="BBB", sessions_remaining=1),
            _open_row(ticker="CCC", sessions_remaining=3)]
    out = build_regret_report(rows, TODAY, LOOKBACK)
    assert out.index("mã BBB") < out.index("mã CCC") < out.index("mã AAA")


def test_truncation_under_telegram_limit() -> None:
    rows = [_open_row(ticker=f"T{i:02d}", sessions_remaining=i) for i in range(50)]
    out = build_regret_report(rows, TODAY, LOOKBACK)
    assert len(out) <= 4096
    assert "tín hiệu khác (rút gọn)" in out
