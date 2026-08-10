"""Tests for scripts/analyze_rank_sleeve.py::sleeve_verdict (item-1 frozen criteria)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_rank_sleeve import sleeve_verdict  # noqa: E402

_D0 = dt.date(2026, 6, 16)


def _rows(day_specs: list[list[tuple[str, float, float | None, bool]]]) -> pl.DataFrame:
    """Build a paperlog frame: one inner list per day of (ticker, p_up, ret_20d, filled)."""
    recs = []
    for i, day in enumerate(day_specs):
        d = _D0 + dt.timedelta(days=i)
        for ticker, p_up, ret, filled in day:
            recs.append({"log_date": d, "ticker": ticker, "p_up_secondary": p_up,
                         "ret_20d": ret, "outcome_filled": filled})
    return pl.DataFrame(recs, schema={"log_date": pl.Date, "ticker": pl.Utf8,
                                      "p_up_secondary": pl.Float64, "ret_20d": pl.Float64,
                                      "outcome_filled": pl.Boolean})


def test_empty_frame_is_insufficient() -> None:
    out = sleeve_verdict(pl.DataFrame(schema={"log_date": pl.Date, "ticker": pl.Utf8,
                                              "p_up_secondary": pl.Float64, "ret_20d": pl.Float64,
                                              "outcome_filled": pl.Boolean}))
    assert out["verdict"] == "INSUFFICIENT_DATA"
    assert out["settled_sleeve_n"] == 0


def test_sleeve_is_top3_by_pup_per_day() -> None:
    # 5 names one day; top-3 by p_up = A, B, C. Only sleeve rows count.
    day = [("A", 0.48, 0.10, True), ("B", 0.47, 0.10, True), ("C", 0.46, 0.10, True),
           ("D", 0.30, -0.50, True), ("E", 0.20, -0.50, True)]
    out = sleeve_verdict(_rows([day]), min_settled_n=3)
    assert out["settled_sleeve_n"] == 3
    assert abs(out["sleeve_mean"] - 0.10) < 1e-12  # D/E's -50% never enters the sleeve


def test_unsettled_sleeve_rows_do_not_count() -> None:
    day = [("A", 0.48, None, False), ("B", 0.47, 0.10, True), ("C", 0.46, 0.10, True),
           ("D", 0.30, 0.99, True)]
    out = sleeve_verdict(_rows([day]), min_settled_n=2)
    # A is IN the sleeve (top-3 by p_up) but unsettled -> evaluated rows = B, C only.
    # D is settled but NOT in the sleeve.
    assert out["settled_sleeve_n"] == 2
    assert abs(out["sleeve_mean"] - 0.10) < 1e-12


def test_success_requires_beating_control() -> None:
    # Sleeve positive but BELOW the cross-section mean -> FAILURE (ranking adds nothing).
    day = [("A", 0.48, 0.01, True), ("B", 0.47, 0.01, True), ("C", 0.46, 0.01, True),
           ("D", 0.30, 0.50, True), ("E", 0.20, 0.50, True)]
    out = sleeve_verdict(_rows([day]), min_settled_n=3)
    assert out["verdict"] == "FAILURE"


def test_success_path() -> None:
    day = [("A", 0.48, 0.10, True), ("B", 0.47, 0.08, True), ("C", 0.46, 0.06, True),
           ("D", 0.30, -0.10, True), ("E", 0.20, -0.20, True)]
    out = sleeve_verdict(_rows([day]), min_settled_n=3)
    assert out["verdict"] == "SUCCESS"
    assert out["sleeve_mean"] > out["control_mean"]


def test_min_n_gate_blocks_verdict() -> None:
    day = [("A", 0.48, 0.10, True), ("B", 0.47, 0.08, True), ("C", 0.46, 0.06, True)]
    out = sleeve_verdict(_rows([day]))  # default min 60 >> 3 settled
    assert out["sufficient"] is False
    assert out["verdict"] == "INSUFFICIENT_DATA"
    assert out["settled_sleeve_n"] == 3  # numbers still reported, verdict withheld
