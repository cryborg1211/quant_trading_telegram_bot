"""Pure-function tests for price_lookup.derive_ca_adjustment_factor.

A pure numeric transform on an ordered close list — no I/O, no stubs needed.
Sibling of has_ca_gap (same 10% default threshold, same input list). Covers
no-gap identity, single-gap ratio, multi-gap cumulative composition (in date
order), degenerate inputs, the prev>0 guard, and the custom-threshold param.
"""
from __future__ import annotations

import pytest

from src.data.price_lookup import derive_ca_adjustment_factor


def test_no_gap_returns_one() -> None:
    # All consecutive moves < 10% → no corporate action → identity factor.
    assert derive_ca_adjustment_factor([100.0, 105.0, 101.0, 108.0]) == 1.0


def test_single_gap_returns_ratio() -> None:
    # First pair gaps at -40% (a CA reset), second pair is a +5% organic move.
    assert derive_ca_adjustment_factor([100.0, 60.0, 63.0]) == pytest.approx(0.6)


def test_multi_gap_composes_product_in_date_order() -> None:
    # Two gaps (0.6 each) with an organic +5% move between them compose as a
    # cumulative product 0.6 * 0.6 = 0.36 (plan Decision 2 worked example).
    assert derive_ca_adjustment_factor([100.0, 60.0, 63.0, 37.8]) == pytest.approx(0.36)


def test_empty_and_single_element_returns_one() -> None:
    assert derive_ca_adjustment_factor([]) == 1.0
    assert derive_ca_adjustment_factor([100.0]) == 1.0


def test_zero_or_negative_prev_skipped() -> None:
    # Mirrors has_ca_gap's own prev>0 guard — no ZeroDivisionError, no factor
    # change from a corrupted zero/negative close.
    assert derive_ca_adjustment_factor([0.0, 50.0]) == 1.0
    assert derive_ca_adjustment_factor([-10.0, 50.0]) == 1.0


def test_custom_threshold_param() -> None:
    # A -6% move: default 10% threshold sees no gap; a tighter 5% threshold does.
    assert derive_ca_adjustment_factor([100.0, 94.0]) == 1.0
    assert derive_ca_adjustment_factor([100.0, 94.0], max_session_move=0.05) == pytest.approx(0.94)
