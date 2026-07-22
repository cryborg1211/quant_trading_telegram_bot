"""rank_breadth admission mode (22-07-26) — breadth-conditioned K, no absolute floor.

Mirrors tests/test_admission_ab.py's fixture style (constant-price panel,
oracle returns the `feat` column as P(UP), `_bought` extracts fills).

Core contract under test: NO absolute P(UP) floor exists in this mode — the
engine always takes the top-K ranked names, where K = round(max_positions x
breadth_scalar(breadth)). K responds ONLY to breadth, never to how low the
top names' scores are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardEngine

N_DAYS = 20
HOLD = 5
PRICE = 20.0
MAX_POS = 4


def _panel(scores: dict[str, float]) -> pd.DataFrame:
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
    return X[:, -1, 0].astype(np.float64)


def _breadth_series(value: float) -> pd.Series:
    """Constant breadth across the whole panel's date range."""
    days = pd.bdate_range("2024-01-02", periods=N_DAYS)
    return pd.Series(value, index=days)


def _engine(*, breadth_trigger: float = 0.40, floor_level: float = 0.25,
            floor: float = 0.5, max_positions: int = MAX_POS) -> WalkForwardEngine:
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"],
        rebalance_mode="tranche", tranche_hold_days=HOLD,
        max_positions=max_positions, signal_threshold=0.0,  # irrelevant in this mode
        liquid_top_n=None, initial_capital=1_000_000_000.0,
        admission_mode="rank_breadth",
        rank_breadth_trigger=breadth_trigger,
        rank_breadth_floor_level=floor_level,
        rank_breadth_floor=floor,
    )
    return WalkForwardEngine(cfg, _oracle)


def _bought(res) -> set[str]:
    fills = pd.DataFrame(res.fills)
    if fills.empty:
        return set()
    return set(fills.query("side == 'buy'")["ticker"])


_SCORES6 = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.70,
            "DDD": 0.60, "EEE": 0.50, "FFF": 0.10}


# ── No absolute floor — the defining property ───────────────────────────────

def test_full_breadth_admits_top_k_regardless_of_absolute_score() -> None:
    # Breadth at/above trigger -> K=max_positions=4. Even FFF at 0.10 (far
    # below any sane "signal_threshold") is admitted since it's rank #4.
    res = _engine().run(_panel(_SCORES6), breadth_series=_breadth_series(0.60))
    assert _bought(res) == {"AAA", "BBB", "CCC", "DDD"}
    assert res.zero_candidate_days == 0


def test_very_low_scores_still_admitted_when_ranked_in_k() -> None:
    # ALL scores are tiny (would fail any absolute_gate floor) — rank_breadth
    # doesn't care, it only cares about RANK.
    scores = {"A": 0.02, "B": 0.015, "C": 0.01, "D": 0.005}
    res = _engine(max_positions=2).run(_panel(scores), breadth_series=_breadth_series(0.60))
    assert _bought(res) == {"A", "B"}


# ── K sizing from the breadth scalar ─────────────────────────────────────────

def test_floor_level_breadth_halves_k() -> None:
    # breadth <= floor_level -> K = round(max_positions * floor) = round(4*0.5) = 2.
    res = _engine(floor=0.5).run(_panel(_SCORES6), breadth_series=_breadth_series(0.10))
    assert _bought(res) == {"AAA", "BBB"}


def test_famine_breadth_zero_k_carries_cash() -> None:
    # floor=0.0 is a DEGENERATE breadth_scalar input (fails open to 1.0 by
    # that function's own contract) — use a small but VALID floor instead:
    # round(max_positions=4 * floor=0.05) = round(0.2) = 0 -> true K=0,
    # pure cash-carry, same diagnostic counter as absolute_gate's.
    res = _engine(floor=0.05).run(_panel(_SCORES6), breadth_series=_breadth_series(0.05))
    assert _bought(res) == set()
    nav = res.equity_curve["nav"].to_numpy()
    assert np.allclose(nav, 1_000_000_000.0)
    assert res.zero_candidate_days == N_DAYS - 1


def test_ramp_midpoint_rounds_k() -> None:
    # trigger=0.40, floor_level=0.20, floor=0.5 -> midpoint breadth=0.30 gives
    # scalar=0.75 -> K=round(4*0.75)=3.
    res = _engine(breadth_trigger=0.40, floor_level=0.20, floor=0.5).run(
        _panel(_SCORES6), breadth_series=_breadth_series(0.30))
    assert _bought(res) == {"AAA", "BBB", "CCC"}


# ── Fail-open contract ────────────────────────────────────────────────────────

def test_missing_breadth_series_fails_open_to_full_k() -> None:
    # No breadth_series at all -> self._breadth stays empty -> .get(D, 1.0)
    # defaults to full exposure, same fail-open contract as p_bull_series.
    res = _engine().run(_panel(_SCORES6))  # breadth_series omitted
    assert _bought(res) == {"AAA", "BBB", "CCC", "DDD"}


def test_date_missing_from_breadth_series_fails_open_for_that_day() -> None:
    # A breadth_series that doesn't cover the panel's dates at all -> every
    # lookup misses -> fail-open default 1.0 -> full K throughout. Uses a
    # small but VALID floor (0.05, not the degenerate 0.0) so the ONLY
    # reason this isn't famine is the missing-date fail-open, not a second,
    # unrelated fail-open path inside breadth_scalar itself.
    stale_index = pd.bdate_range("2020-01-01", periods=5)
    stale_series = pd.Series(0.05, index=stale_index)  # would be famine IF matched
    res = _engine(floor=0.05).run(_panel(_SCORES6), breadth_series=stale_series)
    assert _bought(res) == {"AAA", "BBB", "CCC", "DDD"}  # NOT famine — fail-open


# ── Determinism / isolation from other admission modes ──────────────────────

def test_rank_breadth_is_deterministic() -> None:
    a = _engine().run(_panel(_SCORES6), breadth_series=_breadth_series(0.60))
    b = _engine().run(_panel(_SCORES6), breadth_series=_breadth_series(0.60))
    np.testing.assert_array_equal(
        a.equity_curve["nav"].to_numpy(), b.equity_curve["nav"].to_numpy())


def test_cross_sectional_default_unaffected_by_new_fields() -> None:
    # Adding rank_breadth_* fields to WalkForwardConfig must not perturb the
    # existing default admission path.
    cfg = WalkForwardConfig(
        seq_len=1, feature_cols=["feat"], rebalance_mode="tranche",
        tranche_hold_days=HOLD, max_positions=MAX_POS, signal_threshold=0.40,
        liquid_top_n=None, initial_capital=1_000_000_000.0,
    )
    res = WalkForwardEngine(cfg, _oracle).run(_panel(_SCORES6))
    assert res.zero_candidate_days == 0
    assert _bought(res) == {"AAA", "BBB", "CCC", "DDD"}
