"""Look-ahead-leak + oversold-detection tests for the mean-reversion features.

Ports the synthetic V-shape capitulation check that previously lived only in
``src/features/mr_features.py``'s ``__main__`` block (so CI never ran it). The
two guarantees under test:

1. **Oversold detection** — at a sharp capitulation low the MR features must
   scream: RSI deeply oversold, %B below the lower Bollinger band, price far
   below SMA20, Williams %R near -100.
2. **Look-ahead safety** — a feature at bar ``t`` must NOT change when only
   FUTURE bars (``> t``) are mutated. This is the leak guard for the whole
   module's rolling / ewm / shift contract.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features


def _v_shape_frame(n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic single-ticker OHLCV: calm → 2-bar crash → sharp V recovery.

    40 calm bars (tiny ±0.2% wiggle → small but positive BB std) then a sudden
    ~-13% / -11% plunge that pierces the lower band before it can widen, then a
    +6%/bar rebound. Returns ``(frame, close_array)``.
    """
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    px = [100.0 + (0.2 if i % 2 else -0.2) for i in range(40)]
    px.append(px[-1] * 0.87)   # -13% crash bar
    px.append(px[-1] * 0.89)   # -11% follow-through (the low)
    while len(px) < n:
        px.append(px[-1] * 1.06)  # sharp V rebound
    arr = np.array(px[:n], dtype=float)
    frame = pd.DataFrame(
        {
            "ticker": ["KNIFE"] * n,
            "date": days,
            "open": arr * 0.999,
            "high": arr * 1.01,
            "low": arr * 0.98,
            "close": arr,
            "volume": [1_000_000] * n,
        }
    )
    return frame, arr


def test_features_scream_oversold_at_capitulation_low() -> None:
    frame, px = _v_shape_frame()
    res = build_mr_features(frame)
    row = res.iloc[int(np.argmin(px))]

    assert row["mr_rsi_9"] < 20, "RSI9 should be deeply oversold at the low"
    assert row["mr_bb_pctb"] < 0.0, "%B should be below the lower band"
    assert row["mr_bb_below_lower"] == 1, "price should pierce the lower BB"
    assert row["mr_dma_sma20"] < -0.05, "price should be far below SMA20"
    assert row["mr_williams_r_14"] < -80, "Williams %R should be near -100"


def test_no_look_ahead_leak_when_future_mutated() -> None:
    frame, px = _v_shape_frame()
    bottom = int(np.argmin(px))
    base = build_mr_features(frame)

    mutated = frame.copy()
    mutated.loc[bottom + 3:, "close"] *= 1.5  # alter ONLY the future

    after = build_mr_features(mutated)

    # Flag columns are Int8 and compared separately to keep the float allclose clean.
    float_cols = [c for c in MR_FEATURE_COLUMNS if c not in ("mr_bb_below_lower", "mr_gap_down")]
    a = base.loc[:bottom, float_cols].to_numpy()
    b = after.loc[:bottom, float_cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True), (
        "LOOK-AHEAD LEAK: past features moved when only future bars changed"
    )


@pytest.mark.parametrize("missing", ["open", "high", "low", "close", "ticker", "date"])
def test_missing_required_column_raises(missing: str) -> None:
    frame, _ = _v_shape_frame()
    with pytest.raises(ValueError, match="missing required columns"):
        build_mr_features(frame.drop(columns=[missing]))
