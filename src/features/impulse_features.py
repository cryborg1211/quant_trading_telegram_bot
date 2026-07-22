"""Impulse ("fast-attack") feature engineering — momentum-burst microstructure.

WHY THIS EXISTS
───────────────
The mirror twin of `mr_features.py`. That module sees panic bottoms the slow
Alpha360-lineage stack cannot; THIS module sees momentum IGNITION the same
stack cannot: its shortest window is 5 days and FracDiff smooths single-day
impulses away, so a +2σ volume-confirmed breakout bar looks like any other
bar to the main ensemble. Built for a dedicated fast-attack sub-model
(research phase: `scripts/analyze_impulse_attack.py`), deliberately NOT
mixed into the main feature recipe — zero impact on
`FEATURE_RECIPE_VERSION` or the serve tripwire, exactly like the MR sleeve.

VN-MARKET REALITIES BAKED IN
────────────────────────────
  • HOSE caps a session at ±7%: `imp_ret1_z` uses a per-ticker rolling σ so
    "how extreme is today" is scale-free; names pinned at the ceiling show
    up as `imp_close_pos` ≈ 1.0 + max `imp_ret1_pct`.
  • T+2.5 settlement: the companion label horizon is 3 bars minimum (same
    reason MR's bounce label is 3d) — no feature here encodes anything a
    T+0 scalper would need.

LOOK-AHEAD SAFETY (same audited discipline as mr_features.py)
─────────────────────────────────────────────────────────────
A feature at bar t is consumed at bar t's close: every rolling/ewm/diff
window ENDS at t, previous values come from `shift(1)` within ticker,
grouping is per-ticker throughout. No value dated > t anywhere. Warm-up
rows are NaN; LightGBM handles NaN natively downstream.

OUTPUT CONTRACT
───────────────
``build_impulse_features(df)`` returns a COPY of the input OHLCV frame with
``IMPULSE_FEATURE_COLUMNS`` appended. Accepts pandas or polars; requires
``ticker, date, open, high, low, close, volume`` (volume is REQUIRED here —
half the point of an impulse is volume confirmation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.schema_hash import compute_feature_schema_hash

IMPULSE_FEATURE_COLUMNS: list[str] = [
    "imp_ret1_pct",        # today's close-to-close %
    "imp_ret1_z",          # ...as z-score vs the ticker's own 60d return σ
    "imp_vol_z",           # volume z-score vs 20d mean/σ
    "imp_range_exp",       # today's true range / ATR14 (range expansion)
    "imp_gap_up_pct",      # open vs prior close, positive gaps
    "imp_accel_3_10",      # 3d mean return − 10d mean return (acceleration)
    "imp_up_streak",       # consecutive up-closes ending at t (capped 10)
    "imp_close_pos",       # (close − low) / (high − low): 1.0 = closed at high
    "imp_brk20_dist",      # close / prior 20d high − 1 (breakout distance)
    "imp_vol_price_conf",  # imp_ret1_z × imp_vol_z when both > 0, else 0
]

_IMPULSE_SCHEMA: list[tuple[str, str]] = [
    (col, "float64") for col in IMPULSE_FEATURE_COLUMNS
]
IMPULSE_SCHEMA_HASH: str = compute_feature_schema_hash(_IMPULSE_SCHEMA, None)

_REQUIRED = ("ticker", "date", "open", "high", "low", "close", "volume")


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing == EWM with alpha = 1/n (matches mr_features.py)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.where(den != 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def build_impulse_features(df) -> pd.DataFrame:
    """Append vectorized impulse/momentum-burst features. Not mutated — a
    sorted copy is returned (same contract as build_mr_features)."""
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"build_impulse_features: missing required columns {missing}")

    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    g = out.groupby("ticker", sort=False, group_keys=False)
    prev_close = g["close"].shift(1)

    # 1-day return, raw and scale-free.
    ret1 = _safe_div(out["close"] - prev_close, prev_close)
    ret1_sd60 = ret1.groupby(out["ticker"]).transform(
        lambda s: s.rolling(60, min_periods=30).std())
    out["imp_ret1_pct"] = ret1
    out["imp_ret1_z"] = _safe_div(ret1, ret1_sd60)

    # Volume z vs 20d.
    vol_ma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    vol_sd20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["imp_vol_z"] = _safe_div(out["volume"] - vol_ma20, vol_sd20)

    # Range expansion: today's true range vs Wilder ATR14.
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.groupby(out["ticker"]).transform(lambda s: _wilder(s, 14))
    out["imp_range_exp"] = _safe_div(tr, atr14)

    # Positive opening gap only (attack side — gap-downs live in mr_features).
    gap = _safe_div(out["open"] - prev_close, prev_close)
    out["imp_gap_up_pct"] = gap.clip(lower=0.0)

    # Momentum acceleration: short-window mean return minus longer-window.
    ret_ma3 = ret1.groupby(out["ticker"]).transform(
        lambda s: s.rolling(3, min_periods=3).mean())
    ret_ma10 = ret1.groupby(out["ticker"]).transform(
        lambda s: s.rolling(10, min_periods=10).mean())
    out["imp_accel_3_10"] = ret_ma3 - ret_ma10

    # Consecutive up-closes ending at t (vectorized run length, capped at 10).
    up = (ret1 > 0).astype(np.int64)
    streak_groups = (up == 0).groupby(out["ticker"]).cumsum()
    out["imp_up_streak"] = (
        up.groupby([out["ticker"], streak_groups]).cumsum().clip(upper=10)
        .astype(np.float64)
    )

    # Close position within the day's range: 1.0 = buyers held into the close.
    out["imp_close_pos"] = _safe_div(out["close"] - out["low"],
                                     out["high"] - out["low"])

    # Breakout distance above the PRIOR 20d high (shifted — excludes today).
    prior_high20 = g["high"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).max())
    out["imp_brk20_dist"] = _safe_div(out["close"], prior_high20) - 1.0

    # Price×volume confirmation: product only when BOTH push the same
    # (bullish) way — the classic "impulse with participation" signature.
    both_pos = (out["imp_ret1_z"] > 0) & (out["imp_vol_z"] > 0)
    out["imp_vol_price_conf"] = np.where(
        both_pos, out["imp_ret1_z"] * out["imp_vol_z"], 0.0)

    return out
