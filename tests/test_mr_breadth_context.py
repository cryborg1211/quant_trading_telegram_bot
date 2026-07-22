"""Breadth-inflection annotation on the MR knife-catch display (22-07-26).

Informational only — `_mr_breadth_context_line` must NEVER change whether
the [BAT DAY] tag / veto fires (that stays gated purely on mr["fired"]), it
only appends a short, always-labeled-unconfirmed VI note when a fire's
breadth context is known. Covers the 3 call sites (fallback report,
sell/hold veto, /verify state line) plus the pure helper directly.
"""
from __future__ import annotations

from src.reports.builders import (
    _build_fallback_observability_report_vi,
    _build_sell_hold_report,
    _mr_breadth_context_line,
    _mr_state_line,
    _MR_SELL_VETO,
)

_FAVORABLE = {"fired": True, "breadth_favorable": True}
_UNFAVORABLE = {"fired": True, "breadth_favorable": False}
_UNKNOWN = {"fired": True, "breadth_favorable": None}
_NOT_FIRED = {"fired": False, "breadth_favorable": True}


# ---------------------------------------------------------------------------
# _mr_breadth_context_line — pure helper
# ---------------------------------------------------------------------------


def test_empty_when_not_fired():
    assert _mr_breadth_context_line(_NOT_FIRED) == ""


def test_empty_when_none_mr():
    assert _mr_breadth_context_line(None) == ""
    assert _mr_breadth_context_line({}) == ""


def test_empty_when_context_unavailable():
    assert _mr_breadth_context_line(_UNKNOWN) == ""


def test_favorable_line_says_improving_and_unconfirmed():
    line = _mr_breadth_context_line(_FAVORABLE)
    assert "cải thiện" in line
    assert "✅" in line
    assert "chưa xác nhận" in line  # always labeled unconfirmed


def test_unfavorable_line_says_not_improving():
    line = _mr_breadth_context_line(_UNFAVORABLE)
    assert "⚠️" in line
    assert "chưa xác nhận" in line


# ---------------------------------------------------------------------------
# _mr_state_line (/verify)
# ---------------------------------------------------------------------------


def test_verify_state_line_appends_favorable_context():
    line = _mr_state_line(_FAVORABLE)
    assert "CẢNH BÁO HOẢNG LOẠN" in line  # base fired text unchanged
    assert "cải thiện" in line


def test_verify_state_line_no_context_when_not_fired():
    line = _mr_state_line({"fired": False})
    assert "Chưa đạt" in line
    assert "cải thiện" not in line and "✅" not in line and "⚠️" not in line


def test_verify_state_line_none_state_unaffected():
    assert _mr_state_line(None) == "\U0001f52a <b>Trạng thái Bắt đáy:</b> Không khả dụng"


# ---------------------------------------------------------------------------
# _build_fallback_observability_report_vi
# ---------------------------------------------------------------------------

_PREDS = {"SSI": [0.564, 0.010, 0.426]}
_SENTS = {"SSI": {"sentiment_score": -0.25, "reasoning_vi": "Tin xau.", "source_urls": []}}
_REASONS = {"SSI": "Cua tang duoi nguong an toan."}


def test_fallback_report_tag_unaffected_by_breadth():
    # The [BAT DAY] TAG itself must still key off fired alone.
    for mr in (_FAVORABLE, _UNFAVORABLE, _UNKNOWN):
        html_out = _build_fallback_observability_report_vi(
            ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": mr},
        )
        assert "BẮT ĐÁY" in html_out

    html_out = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": _NOT_FIRED},
    )
    assert "BẮT ĐÁY" not in html_out


def test_fallback_report_includes_breadth_note_only_when_fired_and_known():
    html_fav = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": _FAVORABLE},
    )
    assert "cải thiện" in html_fav

    html_unknown = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": _UNKNOWN},
    )
    assert "cải thiện" not in html_unknown and "chưa xác nhận" not in html_unknown


def test_fallback_report_no_mr_scores_unaffected():
    html_out = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS,
    )
    assert "BẮT ĐÁY" not in html_out
    assert "KHÔNG GIAO DỊCH" in html_out


# ---------------------------------------------------------------------------
# _build_sell_hold_report — MR veto line
# ---------------------------------------------------------------------------

_SELL_DECISION = 0


def test_sell_hold_veto_includes_breadth_note_when_favorable():
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": _SELL_DECISION},
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.6, 0.1, 0.3]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": _FAVORABLE},
    )
    assert _MR_SELL_VETO.split("<b>")[0] in html_out  # base veto text present
    assert "cải thiện" in html_out


def test_sell_hold_veto_present_without_breadth_context():
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": _SELL_DECISION},
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.6, 0.1, 0.3]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": _UNKNOWN},
    )
    assert "CẢNH BÁO BÁN ĐÚNG ĐÁY" in html_out  # veto still fires
    assert "chưa xác nhận" not in html_out


def test_sell_hold_no_veto_when_not_sell_decision():
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": 2},  # not SELL
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.1, 0.1, 0.8]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": _FAVORABLE},
    )
    assert "CẢNH BÁO BÁN ĐÚNG ĐÁY" not in html_out
