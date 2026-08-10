"""Run-length (streak) features — consecutive up/down sessions.

WHY THIS EXISTS
───────────────
Run LENGTH is not run MAGNITUDE. A name up 10% across 3 violent sessions and
one up 10% across 10 quiet consecutive sessions share a `mom20` but describe
different behaviour, and nothing in the live 13-feature recipe encodes the
distinction (`impulse_features.imp_up_streak` exists only inside the
REJECTED fast-attack sub-model, and a DOWN-streak had never been built).

Validated by `scripts/analyze_streak_features.py` (09-08-26) against two
gates: cross-sectional IC (overlap-corrected) and orthogonality to the 13
live model features. All five cleared both — max |corr| 0.45-0.69, always
against `overext_5_xsz`.

HORIZON — READ THIS BEFORE USING
────────────────────────────────
The signal is SHORT-horizon. Overlap-corrected t-stats run 7-13 at T+5 and
collapse to 0.67-2.34 at T+20. Signs are internally consistent (up-streak
negative, down-streak positive, streak_return/intensity most negative) =
short-horizon MEAN REVERSION. So these belong with the MR knife-catch sleeve
(3-day bounce label), NOT the T+20 tranche book.

VN-SPECIFIC NOTE
────────────────
HOSE caps a session at ±7%, so a name pinned near the ceiling (or floor) for
N consecutive sessions is a structural state, not smooth drift — which is
part of why run length carries information the magnitude features miss.

LOOK-AHEAD SAFETY
─────────────────
Every quantity is a function of closes up to and including bar t. Run length
uses the standard cumulative-sum-of-breaks trick (a break resets the group
id) computed per ticker; no value dated > t is touched anywhere.

OUTPUT CONTRACT
───────────────
``build_streak_features(df)`` returns a COPY with ``STREAK_FEATURE_COLUMNS``
appended. Accepts pandas or polars; requires ``ticker, date, close``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.schema_hash import compute_feature_schema_hash

STREAK_FEATURE_COLUMNS: list[str] = [
    "stk_up_streak",        # consecutive up closes ending at t (capped)
    "stk_down_streak",      # consecutive down closes ending at t (capped)
    "stk_signed_streak",    # up − down: one signed run-length number
    "stk_streak_return",    # cumulative return earned across the CURRENT run
    "stk_streak_intensity",  # streak_return / run length = avg daily move in the run
]

_STREAK_SCHEMA: list[tuple[str, str]] = [
    (col, "float64") for col in STREAK_FEATURE_COLUMNS
]
STREAK_SCHEMA_HASH: str = compute_feature_schema_hash(_STREAK_SCHEMA, None)

_REQUIRED = ("ticker", "date", "close")

# Runs longer than this are rare enough that the bucket stops being rankable;
# matches the cap used in the validating EDA.
STREAK_CAP = 15


def build_streak_features(df) -> pd.DataFrame:
    """Append run-length features. Not mutated — a sorted copy is returned
    (same contract as build_mr_features / build_impulse_features)."""
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"build_streak_features: missing required columns {missing}")

    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce").astype(float)

    g = out.groupby("ticker", sort=False, group_keys=False)
    ret1 = g["close"].pct_change()
    is_up = (ret1 > 0).fillna(False)
    is_dn = (ret1 < 0).fillna(False)

    # Group id increments on every break, so the within-group cumulative count
    # IS the current run length.
    up_grp = (~is_up).groupby(out["ticker"]).cumsum()
    dn_grp = (~is_dn).groupby(out["ticker"]).cumsum()

    up_len = is_up.groupby([out["ticker"], up_grp]).cumsum()
    dn_len = is_dn.groupby([out["ticker"], dn_grp]).cumsum()
    out["stk_up_streak"] = np.where(is_up, up_len, 0).clip(0, STREAK_CAP).astype(float)
    out["stk_down_streak"] = np.where(is_dn, dn_len, 0).clip(0, STREAK_CAP).astype(float)
    out["stk_signed_streak"] = out["stk_up_streak"] - out["stk_down_streak"]

    # Cumulative return across the CURRENT run only (resets when the run breaks).
    up_cum = ret1.groupby([out["ticker"], up_grp]).cumsum()
    dn_cum = ret1.groupby([out["ticker"], dn_grp]).cumsum()
    streak_ret = np.where(is_up, up_cum, np.where(is_dn, dn_cum, 0.0))
    out["stk_streak_return"] = pd.Series(streak_ret, index=out.index).astype(float)

    # Average daily move inside the run — separates "10 quiet days" from
    # "3 violent ones" even when the run lengths match.
    run_len = (out["stk_up_streak"] + out["stk_down_streak"]).clip(lower=1.0)
    out["stk_streak_intensity"] = out["stk_streak_return"] / run_len

    return out
