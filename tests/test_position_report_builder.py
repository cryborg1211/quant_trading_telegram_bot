"""Pure-formatter tests for src.reports.builders.build_position_report.

The builder does zero I/O: every row is a plain fixture dict pre-enriched with
evaluate_signal_pnl's `pct` (+ optional `error`) merged onto the ledger row's
own fields. Covers exact line format, verdict/verb tie-break at pct==0, section
omission, error-row filtering, sort order, and Telegram-4096 truncation safety.
"""
from __future__ import annotations

from datetime import date

from src.reports.builders import build_position_report

TODAY = date(2026, 6, 10)
LOOKBACK = 7


def _open_row(**kw) -> dict:
    base = {"ticker": "HPG", "horizon": 20, "pct": 5.34,
            "dispatch_date": date(2026, 6, 1), "sessions_remaining": 3}
    base.update(kw)
    return base


def _closed_row(**kw) -> dict:
    base = {"ticker": "FPT", "horizon": 5, "pct": -2.67}
    base.update(kw)
    return base


def test_open_line_exact_format() -> None:
    out = build_position_report([_open_row()], [], TODAY, LOOKBACK)
    assert ("Dự báo ngày 01/06/2026: Mô hình T20 mã HPG hiện đang lãi 5.3% "
            "(còn 3 ngày kết thúc dự đoán)") in out
    assert "📈 <b>ĐANG MỞ</b>" in out
    assert "ĐÃ ĐÓNG" not in out            # closed section omitted when empty


def test_closed_line_exact_format() -> None:
    out = build_position_report([], [_closed_row()], TODAY, LOOKBACK)
    assert "Mô hình T5 — mã FPT: đã dự báo sai và lỗ 2.7%" in out
    assert "🏁 <b>ĐÃ ĐÓNG (7 NGÀY QUA)</b>" in out
    assert "HÔM NAY" not in out            # header reflects the rolling window now
    assert "ĐANG MỞ" not in out            # open section omitted when empty


def test_closed_header_reflects_lookback() -> None:
    out = build_position_report([], [_closed_row()], TODAY, 14)
    assert "🏁 <b>ĐÃ ĐÓNG (14 NGÀY QUA)</b>" in out


def test_pct_zero_tie_break_is_positive() -> None:
    out_open = build_position_report([_open_row(pct=0.0)], [], TODAY, LOOKBACK)
    assert "hiện đang lãi 0.0%" in out_open
    out_closed = build_position_report([], [_closed_row(pct=0.0)], TODAY, LOOKBACK)
    assert "đã dự báo đúng và lãi 0.0%" in out_closed


def test_both_empty_returns_empty_string() -> None:
    assert build_position_report([], [], TODAY, LOOKBACK) == ""


def test_error_rows_excluded() -> None:
    out = build_position_report(
        [_open_row(), {"ticker": "ERR", "error": "no price"}],
        [{"ticker": "ERR2", "error": "bad"}],
        TODAY,
        LOOKBACK,
    )
    assert "HPG" in out
    assert "ERR" not in out
    assert "ĐÃ ĐÓNG" not in out            # all closed rows were errors → section gone


def test_all_error_rows_returns_empty() -> None:
    out = build_position_report(
        [{"ticker": "E1", "error": "x"}],
        [{"ticker": "E2", "error": "y"}],
        TODAY,
        LOOKBACK,
    )
    assert out == ""


def test_open_rows_sorted_by_sessions_remaining() -> None:
    rows = [_open_row(ticker="AAA", sessions_remaining=5),
            _open_row(ticker="BBB", sessions_remaining=1),
            _open_row(ticker="CCC", sessions_remaining=3)]
    out = build_position_report(rows, [], TODAY, LOOKBACK)
    assert out.index("mã BBB") < out.index("mã CCC") < out.index("mã AAA")


def test_truncation_under_telegram_limit() -> None:
    rows = [_open_row(ticker=f"T{i:02d}", sessions_remaining=i) for i in range(50)]
    out = build_position_report(rows, [], TODAY, LOOKBACK)
    assert len(out) <= 4096
    assert "tín hiệu khác (rút gọn)" in out


def test_gap_flag_open_line_renders_adjusted_annotation() -> None:
    out = build_position_report(
        [_open_row(gap_flag=True, pct=-0.30, adjustment_factor=19.47 / 33.3)],
        [], TODAY, LOOKBACK)
    assert "lỗ 0.3%" in out                                    # corrected pct IS shown (trusted)
    assert "đã tự động điều chỉnh sự kiện DN" in out
    assert "hệ số 0.585" in out
    assert "giá bất thường" not in out                         # old warning line gone
    assert "(còn 3 ngày kết thúc dự đoán)" in out               # suffix retained


def test_gap_flag_closed_line_renders_adjusted_annotation() -> None:
    out = build_position_report(
        [], [_closed_row(gap_flag=True, pct=4.70, adjustment_factor=0.36)],
        TODAY, LOOKBACK)
    assert "đã dự báo đúng và lãi 4.7%" in out
    assert "đã tự động điều chỉnh sự kiện DN" in out
    assert "hệ số 0.360" in out
    assert "giá bất thường" not in out
    assert "Mô hình T5 — mã FPT:" in out
