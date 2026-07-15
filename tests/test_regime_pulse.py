"""Tests for the regime observability pulse (2026-07-14).

`build_regime_pulse` renders the serve regime cache into one log/report line;
`_build_combined_report` optionally prepends it. Added after the 14-07 regime
debate exposed that serve never records which regimes it saw.
"""
from __future__ import annotations

from src.reports.builders import _build_combined_report, build_regime_pulse
from src.utils.telegram_alerter import TelegramBot


def test_empty_cache_returns_empty_string():
    assert build_regime_pulse({}) == ""


def test_distribution_shares_and_defensive_aggregates():
    cache = (
        {f"T{i}": 3 for i in range(5)}      # 50% regime 3
        | {f"U{i}": 6 for i in range(3)}    # 30% regime 6 (PENALTY)
        | {f"V{i}": 0 for i in range(2)}    # 20% regime 0 (NO_TRADE)
    )
    out = build_regime_pulse(cache)
    assert "(10 mã)" in out
    assert "R3" in out and "50%" in out
    assert "R6" in out and "30%" in out
    assert "NO_TRADE {0,7}: 20%" in out
    assert "PENALTY {1,6}: 30%" in out


def test_top3_cap_and_html_safety():
    cache = {f"T{i}": i % 5 for i in range(25)}  # 5 distinct regimes
    out = build_regime_pulse(cache)
    assert out.count(" · ") >= 2  # top-3 joined
    assert sum(f"R{r} " in out for r in range(5)) == 3  # only 3 regimes named
    assert "<" not in out and ">" not in out  # valid inside parse_mode=HTML


def test_combined_report_prepends_pulse(monkeypatch):
    monkeypatch.setattr(TelegramBot, "_build_message", staticmethod(lambda sd: "MSG"))
    with_pulse = _build_combined_report([{"a": 1}], regime_pulse="PULSE-LINE")
    assert with_pulse.startswith("PULSE-LINE\n")
    assert "MSG" in with_pulse


def test_combined_report_unchanged_without_pulse(monkeypatch):
    monkeypatch.setattr(TelegramBot, "_build_message", staticmethod(lambda sd: "MSG"))
    assert _build_combined_report([{"a": 1}]) == "MSG"


def test_combined_report_empty_list_short_circuits_even_with_pulse():
    assert _build_combined_report([], regime_pulse="PULSE-LINE") == ""
