"""Horizon mapping + room lookup for the operator card (10-08-26).

THE REGRESSION THIS GUARDS
──────────────────────────
`stacking_predictions` uses legacy keys: "5d" holds the PRIMARY horizon and
"20d" the SECONDARY, whatever those horizons actually are. Mapping them onto
the card by key name would label T+20 probabilities as T+5 and INVERT the
"T+5-only" warning — turning the single most important disqualifier into a
false all-clear. These tests pin the mapping to the real `horizon` value so a
future rename cannot silently flip it.
"""
from __future__ import annotations

from unittest.mock import patch

import main
from main import SHORT_HORIZON

# probs are [p_down, p_side, p_up]
_PRIMARY = {"AAA": [0.20, 0.20, 0.60]}
_SECONDARY = {"AAA": [0.40, 0.25, 0.35]}
_PREDS = {"5d": _PRIMARY, "20d": _SECONDARY}


def test_primary_is_t20_when_horizon_is_20():
    out = main._horizon_card_fields("AAA", _PREDS, 20, 0.46, 0.44)
    assert out["p_up_20d"] == 0.60 and out["tau_20d"] == 0.46   # PRIMARY
    assert out["p_up_5d"] == 0.35 and out["tau_5d"] == 0.44     # SECONDARY


def test_primary_is_t5_when_horizon_is_short():
    # Mirror image: with the short horizon as PRIMARY the same "5d" key now
    # legitimately holds T+5, and the mapping must follow the horizon.
    out = main._horizon_card_fields("AAA", _PREDS, SHORT_HORIZON, 0.44, 0.46)
    assert out["p_up_5d"] == 0.60 and out["tau_5d"] == 0.44     # PRIMARY
    assert out["p_up_20d"] == 0.35 and out["tau_20d"] == 0.46   # SECONDARY


def test_missing_ticker_yields_none_not_zero():
    # None means "could not evaluate"; 0.0 would read as a failed gate and
    # silently understate, or worse be compared against a threshold.
    out = main._horizon_card_fields("ZZZ", _PREDS, 20, 0.46, 0.44)
    assert out["p_up_20d"] is None and out["p_up_5d"] is None


def test_missing_secondary_horizon_degrades_cleanly():
    out = main._horizon_card_fields("AAA", {"5d": _PRIMARY}, 20, 0.46, None)
    assert out["p_up_20d"] == 0.60
    assert out["p_up_5d"] is None and out["tau_5d"] is None


def test_malformed_probs_yield_none():
    out = main._horizon_card_fields("AAA", {"5d": {"AAA": []}}, 20, 0.46, 0.44)
    assert out["p_up_20d"] is None


# ── _room_exhausted ─────────────────────────────────────────────────────────

def test_room_exhausted_true_when_room_is_zero():
    with patch("src.trading.flow_context.latest_foreign_room", return_value=0.0):
        assert main._room_exhausted("AAA") is True


def test_room_exhausted_true_when_room_is_negative():
    # Over-limit foreign ownership is a real (0.14 pct of rows) edge case.
    with patch("src.trading.flow_context.latest_foreign_room", return_value=-5000.0):
        assert main._room_exhausted("AAA") is True


def test_room_available_when_room_is_positive():
    with patch("src.trading.flow_context.latest_foreign_room", return_value=1_000_000.0):
        assert main._room_exhausted("AAA") is False


def test_room_unknown_when_lookup_returns_none():
    with patch("src.trading.flow_context.latest_foreign_room", return_value=None):
        assert main._room_exhausted("AAA") is None


def test_room_lookup_failure_fails_open_to_none():
    # A data-layer failure must never block a dispatch, and must never render
    # as "room available".
    with patch("src.trading.flow_context.latest_foreign_room",
               side_effect=RuntimeError("parquet gone")):
        assert main._room_exhausted("AAA") is None
