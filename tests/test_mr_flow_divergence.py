"""Foreign-flow divergence annotation on the MR knife-catch display (08-08-26).

Informational only — `_mr_flow_divergence_line` must NEVER change whether
the [BAT DAY] tag / veto fires (that stays gated purely on mr["fired"]), it
only appends a short VI note when a fire's flow context is known. Covers
the 3 call sites (fallback report, sell/hold veto, /verify state line) plus
the pure helper directly. Mirrors test_mr_breadth_context.py's structure.
"""
from __future__ import annotations

from src.reports.builders import (
    _build_fallback_observability_report_vi,
    _build_sell_hold_report,
    _mr_flow_divergence_line,
    _mr_state_line,
    _MR_SELL_VETO,
)

_DIVERGENT = {"fired": True, "flow_divergence": True}
_NOT_DIVERGENT = {"fired": True, "flow_divergence": False}
_UNKNOWN = {"fired": True, "flow_divergence": None}
_NOT_FIRED = {"fired": False, "flow_divergence": True}


# ---------------------------------------------------------------------------
# _mr_flow_divergence_line — pure helper
# ---------------------------------------------------------------------------


def test_empty_when_not_fired():
    assert _mr_flow_divergence_line(_NOT_FIRED) == ""


def test_empty_when_none_mr():
    assert _mr_flow_divergence_line(None) == ""
    assert _mr_flow_divergence_line({}) == ""


def test_empty_when_context_unavailable():
    assert _mr_flow_divergence_line(_UNKNOWN) == ""


def test_divergent_line_says_gom_manh():
    line = _mr_flow_divergence_line(_DIVERGENT)
    assert "gom mạnh" in line
    assert "✅" in line
    assert "ảnh hưởng nhỏ" in line  # always labeled small-effect


def test_not_divergent_line_says_no_clear_signal():
    line = _mr_flow_divergence_line(_NOT_DIVERGENT)
    assert "⚪" in line
    assert "ảnh hưởng nhỏ" in line


# ---------------------------------------------------------------------------
# _mr_state_line (/verify)
# ---------------------------------------------------------------------------


def test_verify_state_line_appends_divergence_context():
    line = _mr_state_line(_DIVERGENT)
    assert "CẢNH BÁO HOẢNG LOẠN" in line  # base fired text unchanged
    assert "gom mạnh" in line


def test_verify_state_line_no_context_when_not_fired():
    line = _mr_state_line({"fired": False})
    assert "Chưa đạt" in line
    assert "gom mạnh" not in line


# ---------------------------------------------------------------------------
# _build_fallback_observability_report_vi
# ---------------------------------------------------------------------------

_PREDS = {"SSI": [0.564, 0.010, 0.426]}
_SENTS = {"SSI": {"sentiment_score": -0.25, "reasoning_vi": "Tin xau.", "source_urls": []}}
_REASONS = {"SSI": "Cua tang duoi nguong an toan."}


def test_fallback_report_tag_unaffected_by_flow_divergence():
    # The [BAT DAY] TAG itself must still key off fired alone.
    for mr in (_DIVERGENT, _NOT_DIVERGENT, _UNKNOWN):
        html_out = _build_fallback_observability_report_vi(
            ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": mr},
        )
        assert "BẮT ĐÁY" in html_out


def test_fallback_report_includes_flow_note_only_when_fired_and_known():
    html_div = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": _DIVERGENT},
    )
    assert "gom mạnh" in html_div

    html_unknown = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, mr_scores={"SSI": _UNKNOWN},
    )
    assert "gom mạnh" not in html_unknown and "ảnh hưởng nhỏ" not in html_unknown


# ---------------------------------------------------------------------------
# _build_sell_hold_report — MR veto line
# ---------------------------------------------------------------------------

_SELL_DECISION = 0


def test_sell_hold_veto_includes_flow_note_when_divergent():
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": _SELL_DECISION},
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.6, 0.1, 0.3]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": _DIVERGENT},
    )
    assert _MR_SELL_VETO.split("<b>")[0] in html_out  # base veto text present
    assert "gom mạnh" in html_out


def test_sell_hold_veto_present_without_flow_context():
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": _SELL_DECISION},
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.6, 0.1, 0.3]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": _UNKNOWN},
    )
    assert "CẢNH BÁO BÁN ĐÚNG ĐÁY" in html_out  # veto still fires
    assert "ảnh hưởng nhỏ" not in html_out


def test_breadth_and_flow_notes_can_both_appear():
    both = {"fired": True, "breadth_favorable": True, "flow_divergence": True}
    html_out = _build_sell_hold_report(
        ["SSI"],
        final_decisions={"SSI": _SELL_DECISION},
        all_sentiments={"SSI": {}},
        stacking_predictions={"5d": {"SSI": [0.6, 0.1, 0.3]}},
        live_exec_prices={"SSI": 20000.0},
        mr_scores={"SSI": both},
    )
    assert "cải thiện" in html_out  # breadth note
    assert "gom mạnh" in html_out   # flow note
