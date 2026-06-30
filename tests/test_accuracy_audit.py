"""Unit tests for the model-accuracy confusion-matrix auditor.

Pure read-side logic over mocked ``sentiment_entry_paperlog`` rows — no real
DuckDB, no parquet. Pins the confusion-matrix mapping, the precision/recall
math, the settled-row query contract, and the rendered digest.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.utils.accuracy_audit import (
    build_accuracy_report,
    classify_outcome,
    fetch_settled_predictions,
    summarize_accuracy,
)

# Decision encoding: 0=SELL, 1=HOLD, 2=BUY.
_SELL, _HOLD, _BUY = 0, 1, 2


# --------------------------------------------------------------------------- #
# classify_outcome — the confusion matrix.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "decision, ret, expected",
    [
        (_BUY, 0.05, "TP"),    # acted, rose
        (_BUY, -0.03, "FP"),   # acted, fell
        (_BUY, 0.0, "FP"),     # acted, flat → not a win
        (_SELL, -0.04, "TN"),  # stayed out, fell → correct
        (_HOLD, 0.0, "TN"),    # stayed out, flat → correct
        (_HOLD, 0.06, "FN"),   # stayed out, rose → missed
        (_SELL, 0.02, "FN"),   # stayed out, rose → missed
        (None, 0.05, None),    # ungradable
        (_BUY, None, None),    # ungradable
    ],
)
def test_classify_outcome_matrix(decision, ret, expected) -> None:
    assert classify_outcome(decision, ret) == expected


# --------------------------------------------------------------------------- #
# summarize_accuracy — precision / recall.
# --------------------------------------------------------------------------- #

def test_summarize_precision_and_recall() -> None:
    rows = [
        {"decision": _BUY, "ret": 0.05},    # TP
        {"decision": _BUY, "ret": 0.02},    # TP
        {"decision": _BUY, "ret": -0.01},   # FP
        {"decision": _SELL, "ret": -0.03},  # TN
        {"decision": _HOLD, "ret": -0.02},  # TN
        {"decision": _HOLD, "ret": 0.04},   # FN
    ]
    s = summarize_accuracy(rows)
    assert s["counts"] == {"TP": 2, "FP": 1, "TN": 2, "FN": 1}
    assert s["total_settled"] == 6
    # BUY precision = 2 / (2 + 1)
    assert s["buy_precision"] == pytest.approx(2 / 3)
    # Defensive recall = 2 / (2 + 1)
    assert s["defensive_recall"] == pytest.approx(2 / 3)


def test_summarize_zero_denominators_are_none() -> None:
    # All HOLD with losses → no BUY calls and no FN → both ratios undefined.
    s = summarize_accuracy([{"decision": _HOLD, "ret": -0.01}])
    assert s["counts"] == {"TP": 0, "FP": 0, "TN": 1, "FN": 0}
    assert s["buy_precision"] is None       # 0 buy calls
    assert s["defensive_recall"] == 1.0     # 1 / (1 + 0)
    s2 = summarize_accuracy([])
    assert s2["buy_precision"] is None and s2["defensive_recall"] is None


# --------------------------------------------------------------------------- #
# fetch_settled_predictions — query contract.
# --------------------------------------------------------------------------- #

def _db_with_rows(rows: list[tuple]) -> MagicMock:
    db = MagicMock()
    db.conn.execute.return_value.fetchall.return_value = rows
    return db


def test_fetch_maps_rows_and_prefers_arbitrated_decision() -> None:
    # (log_date, ticker, COALESCE(final_decision, decision_5d), entry_close, ret_20d)
    db = _db_with_rows([
        (date(2026, 6, 1), "hpg", 2, 25.0, 0.08),
        (date(2026, 5, 28), "ssi", 0, 30.0, -0.02),
    ])
    out = fetch_settled_predictions(db)
    assert [r["ticker"] for r in out] == ["HPG", "SSI"]  # normalised upper
    assert out[0]["decision"] == 2 and out[0]["ret"] == pytest.approx(0.08)
    # The query must only settle terminal rows.
    sql = db.conn.execute.call_args[0][0]
    assert "outcome_filled = TRUE" in sql
    assert "ret_20d IS NOT NULL" in sql
    assert "COALESCE(final_decision, decision_5d)" in sql


def test_fetch_lookback_param_is_bound() -> None:
    db = _db_with_rows([])
    fetch_settled_predictions(db, lookback_days=30)
    sql, params = db.conn.execute.call_args[0]
    assert "CURRENT_DATE" in sql
    assert params == [30]


def test_fetch_degrades_to_empty_on_error() -> None:
    db = MagicMock()
    db.conn.execute.side_effect = RuntimeError("no such table")
    assert fetch_settled_predictions(db) == []


# --------------------------------------------------------------------------- #
# build_accuracy_report — rendered digest.
# --------------------------------------------------------------------------- #

def test_report_renders_header_table_and_emojis() -> None:
    db = _db_with_rows([
        (date(2026, 6, 1), "HPG", _BUY, 25.0, 0.08),    # TP 🟢
        (date(2026, 5, 30), "VHM", _BUY, 50.0, -0.04),  # FP 🔴
        (date(2026, 5, 28), "SSI", _HOLD, 30.0, -0.02), # TN 🔵
    ])
    out = build_accuracy_report(db=db, last_n=15)
    assert "HẬU KIỂM ĐỘ CHÍNH XÁC MÔ HÌNH" in out
    assert "BUY Precision:</b> 50.0%" in out   # 1 TP / (1 TP + 1 FP)
    assert "Defensive Recall:</b> 100.0%" in out
    assert "HPG" in out and "VHM" in out and "SSI" in out
    assert "🟢" in out and "🔴" in out and "🔵" in out
    assert "<pre>" in out and "</pre>" in out


def test_report_empty_when_no_settled_rows() -> None:
    db = _db_with_rows([])
    assert build_accuracy_report(db=db) == ""
