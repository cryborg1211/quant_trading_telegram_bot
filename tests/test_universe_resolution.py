"""Degrade-path tests for main._resolve_candidate_universe().

Covers the serve_universe_mode precedence / fail-safe wiring from Design
Decision #3 of serve-universe-adv-alignment_PLAN_05-07-26.md:

  - "vn30" mode NEVER calls liquid_universe.
  - "adv_top_n" with an EMPTY liquid_universe result degrades to _VN30_UNIVERSE.
  - "adv_top_n" with liquid_universe RAISING degrades to _VN30_UNIVERSE (no crash).
  - an invalid mode string behaves like "vn30" (fail toward the proven-safe set).

Uses monkeypatch to swap CONFIG.trading.serve_universe_mode and to intercept
main.liquid_universe so no real ADV math / DB access is needed.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import main
from main import _VN30_UNIVERSE, _resolve_candidate_universe


def _dummy_panel() -> pl.DataFrame:
    """Minimal valid OHLCV panel (contents irrelevant — liquid_universe is stubbed)."""
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "open": [10.0, 20.0],
            "high": [10.0, 20.0],
            "low": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [100.0, 200.0],
        }
    )


def test_universe_mode_vn30_bypasses_adv_computation(monkeypatch):
    """mode='vn30' must return _VN30_UNIVERSE and never call liquid_universe."""
    monkeypatch.setattr(main.CONFIG.trading, "serve_universe_mode", "vn30")

    calls: list = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return frozenset({"SHOULD_NOT_BE_USED"})

    monkeypatch.setattr(main, "liquid_universe", _spy)

    result = _resolve_candidate_universe(_dummy_panel())
    assert result == _VN30_UNIVERSE
    assert calls == []  # liquid_universe never invoked


def test_universe_mode_adv_degrades_to_vn30_on_empty(monkeypatch):
    """mode='adv_top_n' + empty liquid_universe result → falls back to VN30."""
    monkeypatch.setattr(main.CONFIG.trading, "serve_universe_mode", "adv_top_n")
    monkeypatch.setattr(main, "liquid_universe", lambda *a, **k: frozenset())

    result = _resolve_candidate_universe(_dummy_panel())
    assert result == _VN30_UNIVERSE


def test_universe_mode_adv_degrades_to_vn30_on_exception(monkeypatch):
    """mode='adv_top_n' + liquid_universe raising → caught, falls back to VN30."""
    monkeypatch.setattr(main.CONFIG.trading, "serve_universe_mode", "adv_top_n")

    def _boom(*args, **kwargs):
        raise RuntimeError("schema break")

    monkeypatch.setattr(main, "liquid_universe", _boom)

    # Must NOT propagate the exception — degrade-not-crash.
    result = _resolve_candidate_universe(_dummy_panel())
    assert result == _VN30_UNIVERSE


def test_invalid_mode_string_falls_back_to_vn30(monkeypatch):
    """An unrecognized mode string behaves like 'vn30' (no ADV computation)."""
    monkeypatch.setattr(main.CONFIG.trading, "serve_universe_mode", "bogus")

    calls: list = []
    monkeypatch.setattr(
        main, "liquid_universe",
        lambda *a, **k: calls.append(1) or frozenset({"X"}),
    )

    result = _resolve_candidate_universe(_dummy_panel())
    assert result == _VN30_UNIVERSE
    assert calls == []  # invalid mode never attempts ADV computation


def test_adv_mode_uses_liquid_universe_result_when_nonempty(monkeypatch):
    """Sanity: a non-empty ADV result is used verbatim (not degraded)."""
    monkeypatch.setattr(main.CONFIG.trading, "serve_universe_mode", "adv_top_n")
    expected = frozenset({"FPT", "HPG", "MWG"})
    monkeypatch.setattr(main, "liquid_universe", lambda *a, **k: expected)

    result = _resolve_candidate_universe(_dummy_panel())
    assert result == expected
    assert result != _VN30_UNIVERSE
