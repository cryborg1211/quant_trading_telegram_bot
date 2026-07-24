"""MR context features — volume-exhaustion + sector-relative oversold.

WHY THIS EXISTS
───────────────
Round 2 of the regime-gate research (22-07-26) REJECTED both `market_regime`
and raw RSI-direction as additive to MR-LGBM's own oversold features — both
turned out to be redundant "how stretched is price" information the model
already has. The stated path forward was NEW features, not more slicing of
the existing ones. This module adds two families neither of
`mr_features.py`'s price-only oscillators can see:

  1. VOLUME EXHAUSTION — is capitulation selling volume FADING (climax
     drying up, more likely a real bottom) or still SURGING (distribution
     continuing, less likely a bottom)? `mr_features.py` has no volume
     features at all.
  2. SECTOR-RELATIVE OVERSOLD — is this ticker MORE oversold than its
     sector peers (idiosyncratic capitulation, a cleaner mean-reversion
     candidate) or is the whole sector selling off together (systemic, may
     persist longer)? Reuses `src.trading.sector_map.sector_of` (the
     ~100-ticker/13-sector map shipped for the July admission guards).
     Tickers mapped to OTHER (a mixed, non-cluster bucket) and sectors with
     fewer than 3 names present on a date get NaN here — a "sector" of 1-2
     unrelated names isn't a real peer group.

INPUT CONTRACT — call AFTER `mr_features.build_mr_features`: requires
`ticker, date, close, volume, mr_rsi_14, mr_dma_sma20` already present. NOT
mixed into `MR_FEATURE_COLUMNS` / `MR_SCHEMA_HASH` — the shipped MR-LGBM
artifact's input shape is untouched; this is an ADDITIVE research feature
set for a re-trained comparison only (`scripts/analyze_mr_context_features.py`).

LOOK-AHEAD SAFETY: volume features use the same shift/rolling-ends-at-t
discipline as mr_features.py. Sector-relative features are a same-date
cross-sectional median/rank across OTHER tickers' bars at the SAME date t —
legitimate (all t-dated data is known by t's close) and the same class of
computation as `src.trading.breadth`'s market-wide reads; no ticker's own
FUTURE value leaks into another ticker's reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.trading.sector_map import OTHER, sector_of

MR_CONTEXT_FEATURE_COLUMNS: list[str] = [
    "mrctx_vol_z20",
    "mrctx_vol_trend_3v10",
    "mrctx_down_vol_ratio",
    "mrctx_sect_rsi_rel",
    "mrctx_sect_dma_rel",
    "mrctx_sect_oversold_pctile",
]

_REQUIRED = ("ticker", "date", "close", "volume", "mr_rsi_14", "mr_dma_sma20")
_MIN_SECTOR_NAMES = 3


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.where(den != 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def build_mr_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append volume-exhaustion + sector-relative oversold columns.

    Not mutated — a sorted copy is returned (same contract as
    `build_mr_features`).
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_mr_context_features: missing required columns {missing}"
            " — call build_mr_features(df) first."
        )

    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    g = out.groupby("ticker", sort=False, group_keys=False)
    close, vol = out["close"], out["volume"]

    # ── 1. Volume exhaustion ─────────────────────────────────────────────
    vol_ma20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    vol_sd20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
    out["mrctx_vol_z20"] = _safe_div(vol - vol_ma20, vol_sd20)

    vol_ma3 = g["volume"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    # Prior window shifted 3 bars back so it doesn't overlap the recent-3d window.
    vol_ma10_prior = g["volume"].transform(
        lambda s: s.shift(3).rolling(10, min_periods=7).mean())
    out["mrctx_vol_trend_3v10"] = _safe_div(vol_ma3 - vol_ma10_prior, vol_ma10_prior)

    prev_close = g["close"].shift(1)
    down_vol = vol * (close < prev_close).astype(float)
    down_vol_sum10 = out.assign(_dv=down_vol).groupby("ticker", sort=False)["_dv"].transform(
        lambda s: s.rolling(10, min_periods=7).sum())
    vol_sum10 = g["volume"].transform(lambda s: s.rolling(10, min_periods=7).sum())
    out["mrctx_down_vol_ratio"] = _safe_div(down_vol_sum10, vol_sum10)

    # ── 2. Sector-relative oversold ──────────────────────────────────────
    out["_sector"] = out["ticker"].map(sector_of)
    grp = out.groupby(["date", "_sector"])
    sect_size = grp["mr_rsi_14"].transform("count")
    valid = (out["_sector"] != OTHER) & (sect_size >= _MIN_SECTOR_NAMES)

    sect_rsi_med = grp["mr_rsi_14"].transform("median")
    out["mrctx_sect_rsi_rel"] = np.where(valid, out["mr_rsi_14"] - sect_rsi_med, np.nan)

    sect_dma_med = grp["mr_dma_sma20"].transform("median")
    out["mrctx_sect_dma_rel"] = np.where(valid, out["mr_dma_sma20"] - sect_dma_med, np.nan)

    # Percentile rank of OVERSOLDNESS within sector/date: low RSI = more
    # oversold, so 1 - ascending-rank(RSI) is directly "how oversold" pctile.
    sect_pctile = 1.0 - grp["mr_rsi_14"].rank(pct=True)
    out["mrctx_sect_oversold_pctile"] = np.where(valid, sect_pctile, np.nan)

    out = out.drop(columns=["_sector"])
    return out
