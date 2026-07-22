"""Serve-mirror admission A/B — tranche-engine admission branching.

Task A of `serve-admission-tranche-ab_PLAN_04-07-26.md`. The tranche book gains
an opt-in `admission_mode="absolute_gate"` that mirrors serve's `main.predict_v3
_horizon` meta-gate (absolute P(UP) floor first, cap the survivor pool at
`admission_pool_cap`, then the existing top-`max_positions` slice). The default
`admission_mode="cross_sectional"` must reproduce the pre-A/B admission (top-N
above the relative `signal_threshold`) byte-for-byte.

Fixtures mirror `tests/test_regime_tranche_sizing.py`: a synthetic constant-price
panel whose per-ticker `feat` value doubles as the oracle P(UP) score, a trivial
oracle that returns that feature, and a tranche engine on a 1B NAV / HOLD=5 book.

Four required cases (plan checklist):
  1. default-off byte-identical equity curve on a fixed panel + seed
  2. absolute_gate zero-candidate-day cash carry (all P(UP) below the floor)
  3. absolute_gate pool-cap enforcement (>cap survivors capped before top-N)
  4. absolute_gate floor boundary inclusive (`>=`, matching serve's meta_gate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine

N_DAYS = 20
HOLD = 5
PRICE = 20.0          # thousand-VND scale → 20,000 VND after _prepare
MAX_POS = 2


def _panel(scores: dict[str, float]) -> pd.DataFrame:
    """Constant-price panel; each ticker's `feat` == its (constant) P(UP)."""
    days = pd.bdate_range("2024-01-02", periods=N_DAYS).date
    frames = []
    for tk, score in scores.items():
        frames.append(pd.DataFrame({
            "ticker": tk, "date": days,
            "open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE,
            "volume": 10_000_000, "feat": float(score),
        }))
    return pd.concat(frames, ignore_index=True)


def _oracle(X: np.ndarray) -> np.ndarray:
    return X[:, -1, 0].astype(np.float64)   # p_up = the feature value


def _engine(*, admission_mode: str = "cross_sectional",
            admission_floor: float = 0.45,
            admission_pool_cap: int = 6,
            max_positions: int = MAX_POS,
            signal_threshold: float = 0.40) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=max_positions, signal_threshold=signal_threshold,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode=admission_mode, admission_floor=admission_floor,
        admission_pool_cap=admission_pool_cap,
    )
    return WalkForwardEngine(cfg, _oracle)


def _bought(res) -> set[str]:
    fills = pd.DataFrame(res.fills)
    if fills.empty:
        return set()
    return set(fills.query("side == 'buy'")["ticker"])


# ── 1. Default-off byte-identical ────────────────────────────────────────────

# 6-name cross-section so the absolute-gate pool cap (6) and the top-N slice (2)
# are both exercised by the same fixture. Scores span the 0.40 signal floor.
_SCORES6 = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.70,
            "DDD": 0.60, "EEE": 0.50, "FFF": 0.10}


def test_default_matches_explicit_cross_sectional_byte_identical() -> None:
    # The field default and an explicit admission_mode="cross_sectional" must be
    # the SAME code path → bit-identical NAV series on a fixed panel + oracle.
    default_res = WalkForwardEngine(
        WalkForwardConfig(
            seq_len=1, feature_cols=["feat"], rebalance_mode="tranche",
            tranche_hold_days=HOLD, max_positions=MAX_POS, signal_threshold=0.40,
            liquid_top_n=None, initial_capital=1_000_000_000.0,
        ),  # admission_* left at defaults
        _oracle,
    ).run(_panel(_SCORES6))
    explicit_res = _engine(admission_mode="cross_sectional").run(_panel(_SCORES6))

    np.testing.assert_array_equal(
        default_res.equity_curve["nav"].to_numpy(),
        explicit_res.equity_curve["nav"].to_numpy(),
    )
    # cross_sectional never touches the absolute-gate counter.
    assert default_res.zero_candidate_days == 0
    assert explicit_res.zero_candidate_days == 0


def test_cross_sectional_is_deterministic() -> None:
    # Byte-identical guardrail: re-running the unchanged default admission on the
    # same panel/oracle reproduces the exact NAV curve (no hidden nondeterminism).
    a = _engine(admission_mode="cross_sectional").run(_panel(_SCORES6))
    b = _engine(admission_mode="cross_sectional").run(_panel(_SCORES6))
    np.testing.assert_array_equal(
        a.equity_curve["nav"].to_numpy(), b.equity_curve["nav"].to_numpy())


def test_non_binding_absolute_gate_equals_cross_sectional() -> None:
    # When the absolute floor admits everyone the cross_sectional path already
    # keeps (floor ≤ signal_threshold, pool_cap ≥ candidate count), the two
    # branches must produce the IDENTICAL book — proving absolute_gate only ever
    # SUBTRACTS names, never reorders or adds relative to the incumbent.
    cross = _engine(admission_mode="cross_sectional").run(_panel(_SCORES6))
    gate = _engine(admission_mode="absolute_gate", admission_floor=0.0,
                   admission_pool_cap=10).run(_panel(_SCORES6))
    np.testing.assert_array_equal(
        cross.equity_curve["nav"].to_numpy(), gate.equity_curve["nav"].to_numpy())
    assert gate.zero_candidate_days == 0


# ── 2. Zero-candidate-day cash carry ─────────────────────────────────────────

def test_absolute_gate_zero_candidate_day_carries_cash() -> None:
    # Every score below the floor → NOTHING clears the gate on any trading day.
    # No buys fill, NAV stays flat at initial capital, and every post-warm-up
    # trading day increments zero_candidate_days.
    res = _engine(admission_mode="absolute_gate", admission_floor=0.95).run(
        _panel(_SCORES6))
    assert _bought(res) == set()
    nav = res.equity_curve["nav"].to_numpy()
    assert np.allclose(nav, 1_000_000_000.0)
    # Warm-up needs seq_len (1) history, so day index 0 never trades; the
    # remaining N_DAYS-1 days are all zero-candidate.
    assert res.zero_candidate_days == N_DAYS - 1


def test_cross_sectional_never_counts_zero_candidate_days() -> None:
    # Even a panel where every name is below the ABSOLUTE floor still clears the
    # relative signal_threshold in cross_sectional mode → the counter stays 0
    # (the counter is absolute_gate-only, by contract).
    res = _engine(admission_mode="cross_sectional", signal_threshold=0.40).run(
        _panel({"AAA": 0.42, "BBB": 0.41}))
    assert res.zero_candidate_days == 0
    assert _bought(res) == {"AAA", "BBB"}


# ── 3. Pool-cap enforcement ──────────────────────────────────────────────────

def test_absolute_gate_pool_cap_limits_survivors_before_topn() -> None:
    # 8 names all clear a low floor; pool_cap=3 keeps only the top-3 by P(UP)
    # BEFORE the top-N slice. With max_positions=5 (> cap), the book can still
    # only ever buy the 3 highest-scored names — never the 4th/5th.
    scores = {"S90": 0.90, "S85": 0.85, "S80": 0.80, "S75": 0.75,
              "S70": 0.70, "S65": 0.65, "S60": 0.60, "S55": 0.55}
    res = _engine(admission_mode="absolute_gate", admission_floor=0.40,
                  admission_pool_cap=3, max_positions=5).run(_panel(scores))
    bought = _bought(res)
    assert bought == {"S90", "S85", "S80"}
    assert "S75" not in bought and "S70" not in bought


def test_pool_cap_only_caps_when_it_binds() -> None:
    # pool_cap larger than the survivor count is inert: with 2 survivors above
    # the floor and cap=6, both are admitted (then top-N=2 keeps both).
    res = _engine(admission_mode="absolute_gate", admission_floor=0.50,
                  admission_pool_cap=6, max_positions=2).run(
        _panel({"AAA": 0.90, "BBB": 0.80, "LOW": 0.10}))
    assert _bought(res) == {"AAA", "BBB"}


# ── 4. Floor boundary inclusive (>=) ─────────────────────────────────────────
#
# The engine builds float32 feature tensors (`walk_forward.py:534`), so the
# oracle's P(UP) is float32-precision — exactly as serve's `predict_proba`
# probabilities are finite-precision floats compared against a float threshold.
# The floor and the "at floor" score therefore use values that round-trip
# EXACTLY through float32 (0.5 / 0.75 / 0.625 are dyadic; 0.45 is NOT and would
# widen to 0.44999998, spuriously failing an otherwise-correct `>=`). Picking
# representable values tests the inclusive-`>=` semantics honestly rather than a
# float-representation accident.

def test_absolute_gate_floor_is_inclusive() -> None:
    # A name scored EXACTLY at the floor must be ADMITTED (serve's meta_gate uses
    # P(UP) >= up_threshold). AT_FLOOR == floor is bought; BELOW just under is not.
    res = _engine(admission_mode="absolute_gate", admission_floor=0.5,
                  max_positions=2).run(
        _panel({"AT_FLOOR": 0.5, "ABOVE": 0.75, "BELOW": 0.49}))
    bought = _bought(res)
    assert "AT_FLOOR" in bought
    assert "ABOVE" in bought
    assert "BELOW" not in bought
    assert res.zero_candidate_days == 0


def test_absolute_gate_just_below_floor_excluded() -> None:
    # The single name sits below the floor → empty survivor list →
    # zero-candidate cash carry on every trading day.
    res = _engine(admission_mode="absolute_gate", admission_floor=0.5).run(
        _panel({"NEAR": 0.49}))
    assert _bought(res) == set()
    assert res.zero_candidate_days == N_DAYS - 1


# ── Burst sizing (tranche_budget_days, 22-07-26) ─────────────────────────────

def _engine_budget(budget_days: int | None) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=1, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        tranche_budget_days=budget_days,
    )
    return WalkForwardEngine(cfg, _oracle)


def _first_buy_notional(res) -> float:
    fills = pd.DataFrame(res.fills)
    buys = fills.query("side == 'buy'").sort_values("date")
    first = buys.iloc[0]
    return float(first["qty"]) * float(first["price"])


def test_budget_days_none_is_default_hold_divisor() -> None:
    # None ⇒ byte-identical to the pre-knob engine (nav/hold_days budget).
    default_res = _engine_budget(None).run(_panel({"AAA": 0.90}))
    explicit_res = _engine_budget(HOLD).run(_panel({"AAA": 0.90}))
    np.testing.assert_array_equal(
        default_res.equity_curve["nav"].to_numpy(),
        explicit_res.equity_curve["nav"].to_numpy(),
    )


def test_budget_days_smaller_divisor_deploys_more() -> None:
    # budget_days=1 deploys nav/1 on day one vs nav/HOLD — the first buy's
    # notional must be ~HOLD× bigger (lot rounding makes it approximate).
    slow = _first_buy_notional(_engine_budget(None).run(_panel({"AAA": 0.90})))
    burst = _first_buy_notional(_engine_budget(1).run(_panel({"AAA": 0.90})))
    assert burst > slow * (HOLD - 1)  # ≈ HOLD× bigger, tolerant of lot rounding


def test_budget_days_burst_capped_by_cash() -> None:
    # Even nav/1 every day cannot deploy more than available cash — engine's
    # existing cash constraint must keep NAV finite/sane (no negative cash).
    res = _engine_budget(1).run(_panel({"AAA": 0.90}))
    assert res.equity_curve["nav"].to_numpy().min() > 0
