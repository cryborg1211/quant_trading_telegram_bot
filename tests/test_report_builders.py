"""Regression tests for src/reports/builders.py.

Guards the horizon-label bug (03-07-26): /suggest_buy20's monitoring report
rendered T+20 probabilities under a hardcoded "(5 ngày tới)" label because
`_build_fallback_observability_report_vi` was horizon-blind. The prediction
dict arg is named `stacking_predictions_5d` but holds the PRIMARY horizon's
probs — the label must come from `horizon_days`, never from the legacy name.
"""

from src.reports.builders import _build_fallback_observability_report_vi

_PREDS = {"SSI": [0.564, 0.010, 0.426]}
_SENTS = {"SSI": {"sentiment_score": -0.25, "reasoning_vi": "Tin xấu.", "source_urls": []}}
_REASONS = {"SSI": "Cửa tăng dưới ngưỡng an toàn."}


def test_fallback_report_labels_requested_horizon() -> None:
    html = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, horizon_days=20,
    )
    assert "(20 ngày tới)" in html
    assert "(T+20)" in html
    assert "(5 ngày tới)" not in html


def test_fallback_report_defaults_to_short_horizon() -> None:
    html = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS,
    )
    assert "(5 ngày tới)" in html
    assert "(T+5)" in html


def test_fallback_report_still_flags_no_trade() -> None:
    html = _build_fallback_observability_report_vi(
        ["SSI"], _PREDS, _SENTS, _REASONS, horizon_days=20,
    )
    assert "KHÔNG GIAO DỊCH" in html
    assert "HỦY BỎ TÍN HIỆU" in html
