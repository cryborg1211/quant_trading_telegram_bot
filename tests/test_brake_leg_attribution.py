"""Per-leg brake attribution on the operator card (10-08-26).

THE TWO DEFECTS THESE GUARD
───────────────────────────
Found by driving the real `_dispatch_signals` with real market data rather
than a hand-written payload — an offline dict could not have surfaced either.

1. `live_exposure_scalar()` computed all three legs, logged them, then returned
   only `min(...)`. `_dispatch_signals` therefore had one number to pass on and
   the card's "Độ rộng thị trường" and "Xu hướng chỉ số" lines could NEVER
   render. Two of the four context explanations were dead in practice.

2. The combined minimum was passed in as `garch_scalar`, so a breadth- or
   drift-driven size cut was labelled a volatility brake — the card would read
   "Phanh biến động GARCH ×0.50" on a day the GARCH leg was 1.00.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.bot import garch_brake


# ── live_exposure_legs ──────────────────────────────────────────────────────

def _all_legs_off():
    """Disable every leg so the combiner runs with no model/panel dependency."""
    return patch.multiple(
        garch_brake.CONFIG.trading,
        garch_brake_enabled=False,
        drift_brake_enabled=False,
        breadth_brake_enabled=False,
    )


def test_legs_exposes_every_leg_not_just_the_minimum():
    with _all_legs_off():
        legs = garch_brake.live_exposure_legs()
    for key in ("combined", "garch", "drift", "breadth", "breadth_raw", "binding"):
        assert key in legs, f"missing {key}"


def test_all_legs_disabled_is_full_exposure_and_no_binding_leg():
    with _all_legs_off():
        legs = garch_brake.live_exposure_legs()
    assert legs["combined"] == 1.0
    assert legs["binding"] is None      # None, not "garch" — nothing is braking


def test_binding_leg_is_named_when_something_brakes():
    with _all_legs_off(), \
         patch.object(garch_brake.CONFIG.trading, "breadth_brake_enabled", True), \
         patch.object(garch_brake, "_breadth_leg_scalar", return_value=0.4), \
         patch.object(garch_brake, "_breadth_raw", return_value=0.31):
        legs = garch_brake.live_exposure_legs()
    assert legs["binding"] == "breadth"
    assert legs["combined"] == pytest.approx(0.4)
    assert legs["garch"] == 1.0          # untouched — this is the mislabel guard
    assert legs["breadth_raw"] == pytest.approx(0.31)


def test_breadth_raw_is_the_fraction_not_the_multiplier():
    # The card renders breadth_raw as "N% mã tăng"; showing the multiplier
    # there would report a 40%-breadth market as "50% mã tăng".
    with _all_legs_off(), \
         patch.object(garch_brake.CONFIG.trading, "breadth_brake_enabled", True), \
         patch.object(garch_brake, "_breadth_leg_scalar", return_value=0.5), \
         patch.object(garch_brake, "_breadth_raw", return_value=0.22):
        legs = garch_brake.live_exposure_legs()
    assert legs["breadth"] == pytest.approx(0.5)      # multiplier
    assert legs["breadth_raw"] == pytest.approx(0.22)  # fraction


def test_scalar_wrapper_still_returns_the_combined_float():
    # 20 existing tests depend on this narrow contract.
    with _all_legs_off():
        assert garch_brake.live_exposure_scalar() == 1.0


def test_breadth_raw_fails_open_to_none_on_a_missing_panel():
    assert garch_brake._breadth_raw(None) is None


# ── card payload wiring ─────────────────────────────────────────────────────

def _card_from(brake_legs: dict | None, exposure: float) -> dict:
    """Only the brake-attribution slice of the payload `_dispatch_signals`
    builds — asserted directly so the test does not need the full serve path."""
    return {
        "garch_scalar": float((brake_legs or {}).get("garch", exposure)),
        "drift_scalar": (brake_legs or {}).get("drift"),
        "breadth": (brake_legs or {}).get("breadth_raw"),
        "exposure_scalar": float(exposure),
    }


def test_card_reports_the_garch_leg_not_the_combined_minimum():
    legs = {"combined": 0.5, "garch": 1.0, "drift": 0.5,
            "breadth": 1.0, "breadth_raw": 0.44, "binding": "drift"}
    card = _card_from(legs, 0.5)
    assert card["garch_scalar"] == 1.0      # NOT 0.5 — GARCH is not braking
    assert card["drift_scalar"] == 0.5      # the leg that actually is
    assert card["exposure_scalar"] == 0.5   # combined still available


def test_card_gets_breadth_and_drift_so_both_lines_can_render():
    legs = {"combined": 0.72, "garch": 0.72, "drift": 0.94,
            "breadth": 1.0, "breadth_raw": 0.38, "binding": "garch"}
    card = _card_from(legs, 0.72)
    assert card["breadth"] == pytest.approx(0.38)
    assert card["drift_scalar"] == pytest.approx(0.94)


def test_missing_breakdown_falls_back_to_the_old_behaviour():
    # Brake unavailable (import/compute failure) must not blank the card or
    # raise — it degrades to exactly what shipped before the breakdown existed.
    card = _card_from(None, 0.65)
    assert card["garch_scalar"] == 0.65
    assert card["drift_scalar"] is None and card["breadth"] is None


@pytest.mark.parametrize("scalar,expect_braking", [
    (0.72, True),      # a real brake
    (0.99, True),      # prints ×0.99 — a brake claim here is honest
    (0.997, False),    # rounds to ×1.00; the value observed live on 10-08-26
    (1.0, False),
])
def test_garch_line_label_matches_the_number_it_prints(scalar, expect_braking):
    """`×{:.2f}` and the label must agree. Comparing against 1.0 exactly made a
    routine 0.997 render as "Đang phanh — hạ tỷ trọng ×1.00" — a brake claim
    next to a no-brake number. The leg is a continuous P(Bull) clip, so values
    just under 1.0 are the common case, not an edge case.
    """
    from src.utils.telegram_alerter import TelegramBot

    card = TelegramBot._build_message(
        {"ticker": "AAA", "price": "1 VND", "garch_scalar": scalar})
    assert ("Đang phanh" in card) is expect_braking
    if not expect_braking:
        assert "Không kích hoạt (×1.00)" in card


def test_rendered_card_shows_all_three_legs():
    from src.utils.telegram_alerter import TelegramBot

    legs = {"combined": 0.72, "garch": 0.72, "drift": 0.94,
            "breadth": 1.0, "breadth_raw": 0.38, "binding": "garch"}
    payload = {"ticker": "AAA", "price": "10,000 VND", "horizon_label": "T+20",
               "market_regime": 6, "regime_label": "Đi Ngang",
               **_card_from(legs, 0.72)}
    card = TelegramBot._build_message(payload)
    assert "38%" in card              # breadth line
    assert "0.94" in card             # drift line
    assert "0.72" in card             # garch line
