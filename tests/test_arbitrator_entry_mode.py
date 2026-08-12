"""Arbitrator role at the ENTRY gate — `main._eligible_entries` (11-08-26).

WHY THIS EXISTS
───────────────
"gate" (the original rule) dispatched only names where `make_final_decision`
returned BUY, which requires the primary horizon's ARGMAX to be UP. Measured
over the full 920-day OOS window (scripts/analyze_serve_stack_ab.py), adding
that condition to the validated config takes it from 46 buys to ZERO: p_up's
99th percentile (0.4523) sits below p_down's MEDIAN (0.5957), so UP cannot win
the argmax on this model. It is an off switch, not a filter — and it forced
every real dispatch through the far-less-validated event-rescue path.

"veto" restores parity with the backtest: every tau-clearing candidate is
eligible and the arbitrator only BLOCKS on strongly bearish news.
"""
from __future__ import annotations

import pytest

import main
from config.settings import CONFIG
from main import _eligible_entries

# Decision encoding shared with the arbitrator: 0=SELL, 1=HOLD, 2=BUY.
_SELL, _HOLD, _BUY = 0, 1, 2


def _sent(**scores) -> dict[str, dict]:
    return {t: {"sentiment_score": s} for t, s in scores.items()}


class TestGateMode:
    def test_requires_a_buy_verdict(self):
        decs = {"AAA": _BUY, "BBB": _HOLD, "CCC": _SELL}
        eligible, vetoed = _eligible_entries(
            ["AAA", "BBB", "CCC"], decs, {}, mode="gate")
        assert eligible == ["AAA"] and vetoed == []

    def test_blocks_everything_when_no_argmax_is_up(self):
        # The measured reality: the primary argmax is DOWN on ~99.985pct of
        # name-days, so the book is empty no matter how well tau was cleared.
        cands = ["AAA", "BBB", "CCC"]
        eligible, _ = _eligible_entries(
            cands, dict.fromkeys(cands, _SELL), {}, mode="gate")
        assert eligible == []

    def test_gate_mode_ignores_sentiment_entirely(self):
        # In gate mode the veto lives inside make_final_decision, not here.
        eligible, vetoed = _eligible_entries(
            ["AAA"], {"AAA": _BUY}, _sent(AAA=-0.99), mode="gate")
        assert eligible == ["AAA"] and vetoed == []


class TestVetoMode:
    def test_admits_a_candidate_the_arbitrator_called_sell(self):
        # THE WHOLE POINT: the class verdict no longer gates entry.
        eligible, vetoed = _eligible_entries(["AAA"], {"AAA": _SELL}, {})
        assert eligible == ["AAA"] and vetoed == []

    def test_blocks_on_strongly_bearish_sentiment(self):
        eligible, vetoed = _eligible_entries(
            ["AAA", "BBB"], {}, _sent(AAA=-0.80, BBB=0.10))
        assert eligible == ["BBB"] and vetoed == ["AAA"]

    def test_boundary_is_inclusive(self):
        # `<=`, matching EVENT_BEAR_SENTIMENT's own comparison so the two bear
        # vetoes in the codebase cannot disagree at the edge.
        assert _eligible_entries(["A"], {}, _sent(A=-0.50))[0] == []
        assert _eligible_entries(["A"], {}, _sent(A=-0.49))[0] == ["A"]

    def test_missing_sentiment_does_not_veto(self):
        """A failed news fetch is not a bearish signal.

        The arbitrator returns no sentiment whenever the scrape or the Gemini
        call fails — not rare — and vetoing on that would reinstate the empty
        book this mode exists to fix.
        """
        assert _eligible_entries(["A"], {}, {})[0] == ["A"]
        assert _eligible_entries(["A"], {}, {"A": {}})[0] == ["A"]
        assert _eligible_entries(["A"], {}, {"A": {"sentiment_score": None}})[0] == ["A"]

    def test_unparseable_sentiment_does_not_veto(self):
        assert _eligible_entries(["A"], {}, {"A": {"sentiment_score": "n/a"}})[0] == ["A"]

    def test_neutral_and_positive_pass(self):
        eligible, vetoed = _eligible_entries(
            ["A", "B", "C"], {}, _sent(A=0.0, B=0.35, C=0.90))
        assert eligible == ["A", "B", "C"] and vetoed == []

    def test_preserves_input_order(self):
        # The caller sorts and slices the top 3 afterwards, so a reordering here
        # would silently change which names ship.
        eligible, _ = _eligible_entries(
            ["AAA", "BBB", "CCC", "DDD"], {}, _sent(BBB=-0.99))
        assert eligible == ["AAA", "CCC", "DDD"]

    def test_explicit_threshold_overrides_config(self):
        assert _eligible_entries(["A"], {}, _sent(A=-0.20), bear_veto=-0.10)[0] == []
        assert _eligible_entries(["A"], {}, _sent(A=-0.20), bear_veto=-0.90)[0] == ["A"]

    def test_empty_candidate_list(self):
        assert _eligible_entries([], {}, {}) == ([], [])


class TestDualHorizonMerge:
    """One dispatch is recorded TWICE — primary (T+20, hold 30) plus a T+5
    tracking row for the same ticker and date. While both are OPEN they carry
    the SAME percentage (same entry price, same current price), so the EOD
    report printed the number twice and made T+5 read as a standalone signal.
    """

    @staticmethod
    def _rows(d):
        return [
            {"ticker": "HPG", "dispatch_date": d, "horizon": 20, "hold_days": 30,
             "pct": 2.4, "sessions_remaining": 26},
            {"ticker": "HPG", "dispatch_date": d, "horizon": 5, "hold_days": 5,
             "pct": 2.4, "sessions_remaining": 1},
        ]

    def test_pair_collapses_to_one_row(self):
        from datetime import date as _d

        from src.reports.builders import _merge_dual_horizon
        merged = _merge_dual_horizon(self._rows(_d(2026, 8, 10)))
        assert len(merged) == 1
        assert merged[0]["horizons"] == [(5, 1), (20, 26)]

    def test_merged_row_keeps_the_longest_remaining_window(self):
        # Sorting is by sessions_remaining, so the merged row must represent
        # when the WHOLE dispatch closes, not its first leg.
        from datetime import date as _d

        from src.reports.builders import _merge_dual_horizon
        merged = _merge_dual_horizon(self._rows(_d(2026, 8, 10)))
        assert merged[0]["sessions_remaining"] == 26

    def test_unpaired_row_passes_through(self):
        from datetime import date as _d

        from src.reports.builders import _merge_dual_horizon
        rows = [{"ticker": "SSI", "dispatch_date": _d(2026, 8, 10), "horizon": 20,
                 "hold_days": 30, "pct": 0.8, "sessions_remaining": 26}]
        merged = _merge_dual_horizon(rows)
        assert len(merged) == 1 and merged[0]["horizons"] == [(20, 26)]

    def test_different_dispatch_dates_do_not_merge(self):
        from datetime import date as _d

        from src.reports.builders import _merge_dual_horizon
        rows = [
            {"ticker": "HPG", "dispatch_date": _d(2026, 8, 10), "horizon": 20,
             "hold_days": 30, "pct": 1.0, "sessions_remaining": 26},
            {"ticker": "HPG", "dispatch_date": _d(2026, 8, 5), "horizon": 20,
             "hold_days": 30, "pct": 3.0, "sessions_remaining": 21},
        ]
        assert len(_merge_dual_horizon(rows)) == 2

    def test_report_prints_one_line_naming_both_windows(self):
        from datetime import date as _d

        from src.reports.builders import build_position_report
        out = build_position_report(self._rows(_d(2026, 8, 10)), [],
                                    _d(2026, 8, 12), 7)
        body = [ln for ln in out.splitlines() if "HPG" in ln]
        assert len(body) == 1, out
        assert "T5 còn 1 ngày" in body[0] and "T20 còn 26 ngày" in body[0]
        # The percentage appears ONCE, not twice.
        assert body[0].count("2.4%") == 1


class TestConfigWiring:
    def test_default_mode_is_veto(self):
        assert str(CONFIG.trading.arbitrator_entry_mode).lower() == "veto"

    def test_bear_threshold_matches_the_event_layer(self):
        # Two independent bear vetoes exist (this gate and build_event_overrides);
        # drift between their thresholds would make one silently stricter.
        assert (float(CONFIG.trading.arbitrator_entry_bear_veto)
                == pytest.approx(main.EVENT_BEAR_SENTIMENT))

    def test_unknown_mode_falls_back_to_veto_not_gate(self):
        # A typo in settings.json must not silently re-enable the off switch.
        assert _eligible_entries(["A"], {"A": _SELL}, {}, mode="typo")[0] == ["A"]
