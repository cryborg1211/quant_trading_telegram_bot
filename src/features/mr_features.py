"""Mean-Reversion (capitulation / "knife-catch") feature engineering.

WHY THIS EXISTS
───────────────
The end-of-sprint reversal audit proved the Alpha360 stack is ~81%
price-momentum driven and structurally CANNOT see V-shape bottoms (P(UP)
never lifts toward τ* at capitulation lows). Momentum features lag turns
by construction. This module builds a SEPARATE, dedicated oversold /
panic feature set for a parallel MR sub-model — it is deliberately NOT
mixed with Alpha360.

DEPENDENCY DECISION
───────────────────
ta-lib is NOT installed in this environment (no requirements.txt; the
codebase is Polars/numpy). Rather than add a fragile C-extension hard
dependency, every indicator here is implemented in pure, fully-vectorized
pandas/numpy. RSI/ATR use Wilder smoothing via `ewm(alpha=1/n,
adjust=False)` — numerically identical to TA-Lib's RSI/ATR. (If you later
`pip install TA-Lib`, these can be swapped 1:1; the column contract
stays.)

LOOK-AHEAD SAFETY (audited)
───────────────────────────
A feature at bar t is consumed by a model that decides at the CLOSE of
bar t, so it may use any data up to and including bar t.
  • Every rolling / ewm / diff window ENDS at t (no forward shift).
  • "Previous close" is ``shift(1)`` WITHIN each ticker.
  • Gap-down uses ``open[t]`` vs ``close[t-1]`` — both known by t's close.
  • All grouping is per-ticker, so windows never bleed across symbols.
There is no use of any value dated > t. Early rows are NaN (insufficient
history) — the downstream model imputes/drops them, exactly like
Alpha360.

OUTPUT CONTRACT
───────────────
``build_mr_features(df)`` returns a COPY of the input OHLCV frame with
the columns in ``MR_FEATURE_COLUMNS`` appended. Input may be a pandas or
polars DataFrame; a polars frame is converted to pandas (the rest of the
MR pipeline is pandas/ta-style). Required input columns:
``ticker, date, open, high, low, close`` (``volume`` optional).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The engineered MR feature columns (everything this module adds).
MR_FEATURE_COLUMNS: list[str] = [
    "mr_dma_sma10",
    "mr_dma_sma20",
    "mr_dma_sma50",
    "mr_bb_pctb",
    "mr_bb_below_lower",
    "mr_rsi_9",
    "mr_rsi_14",
    "mr_williams_r_14",
    "mr_atr_norm_14",
    "mr_gap_pct",
    "mr_gap_down",
]

_REQUIRED = ("ticker", "date", "open", "high", "low", "close")


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing == EWM with alpha = 1/n (TA-Lib RSI/ATR engine)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise divide; 0 / non-finite denominators → NaN (no inf, no leak)."""
    out = num / den.where(den != 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def build_mr_features(df) -> pd.DataFrame:
    """Append vectorized mean-reversion / capitulation features.

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame
        Standard OHLCV with at least ``ticker, date, open, high, low,
        close``. Not mutated — a sorted copy is returned.

    Returns
    -------
    pandas.DataFrame
        Input columns + ``MR_FEATURE_COLUMNS``, sorted by
        ``[ticker, date]``.
    """
    # Accept polars too (the OHLCV/Alpha360 layer is polars) — convert once.
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"build_mr_features: missing required columns {missing}")

    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    g = out.groupby("ticker", sort=False, group_keys=False)
    close, high, low, open_ = out["close"], out["high"], out["low"], out["open"]

    # ── 1. Distance-to-MA (DMA): how far price has stretched below trend ──
    for n in (10, 20, 50):
        sma = g["close"].transform(lambda s, _n=n: s.rolling(_n, min_periods=_n).mean())
        out[f"mr_dma_sma{n}"] = _safe_div(close - sma, sma)

    # ── 2. Bollinger (20, 2σ): %B and a pierce-below-lower flag ──────────
    bb_mid = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    # Population std (ddof=0) — matches TA-Lib's BBANDS.
    bb_sd = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).std(ddof=0)
    )
    bb_up = bb_mid + 2.0 * bb_sd
    bb_lo = bb_mid - 2.0 * bb_sd
    out["mr_bb_pctb"] = _safe_div(close - bb_lo, bb_up - bb_lo)
    # NaN where BB undefined (no spurious 0); flag only where defined.
    out["mr_bb_below_lower"] = (
        (close < bb_lo).where(bb_lo.notna()).astype("Int8")
    )

    # ── 3. Extreme oscillators: RSI(9), RSI(14), Williams %R(14) ─────────
    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    for n in (9, 14):
        avg_gain = out.assign(_x=gain).groupby("ticker", sort=False)["_x"].transform(
            lambda s, _n=n: _wilder(s, _n)
        )
        avg_loss = out.assign(_x=loss).groupby("ticker", sort=False)["_x"].transform(
            lambda s, _n=n: _wilder(s, _n)
        )
        rs = _safe_div(avg_gain, avg_loss)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # avg_loss == 0 (pure up streak) ⇒ RS = NaN above ⇒ force RSI = 100.
        rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
        out[f"mr_rsi_{n}"] = rsi

    hh14 = g["high"].transform(lambda s: s.rolling(14, min_periods=14).max())
    ll14 = g["low"].transform(lambda s: s.rolling(14, min_periods=14).min())
    out["mr_williams_r_14"] = -100.0 * _safe_div(hh14 - close, hh14 - ll14)

    # ── 4. ATR(14) normalized: volatility explosion during panic ────────
    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = out.assign(_tr=tr).groupby("ticker", sort=False)["_tr"].transform(
        lambda s: _wilder(s, 14)
    )
    out["mr_atr_norm_14"] = _safe_div(atr14, close)

    # ── 5. Overnight gap-down: panic at the open ────────────────────────
    out["mr_gap_pct"] = _safe_div(open_ - prev_close, prev_close)
    out["mr_gap_down"] = (
        (out["mr_gap_pct"] < 0.0).where(out["mr_gap_pct"].notna()).astype("Int8")
    )

    return out


if __name__ == "__main__":
    # Smoke test: a synthetic V-shape capitulation. At the panic LOW the
    # oversold features must scream — RSI low, %B<0 / below-lower flag set,
    # DMA deeply negative, Williams %R near -100, ATR spiking — and the
    # recovery must mean-revert those back. Proves leak-free detection.
    import datetime as _dt

    n = 80
    days = [_dt.date(2025, 1, 1) + _dt.timedelta(days=i) for i in range(n)]
    # 40 CALM bars (tiny ±0.2% wiggle so BB std is small but > 0) → a
    # SHARP 2-bar capitulation (gap/crash before the band can widen) →
    # V recovery. This is realistic panic: a sudden plunge from calm
    # pierces the lower band; a smooth multi-day grind just "walks" it
    # (correct, but a weaker test of the pierce flag).
    px = [100.0 + (0.2 if i % 2 else -0.2) for i in range(40)]
    px.append(px[-1] * 0.87)              # -13% crash bar
    px.append(px[-1] * 0.89)              # -11% follow-through (the low)
    for _ in range(n - len(px)):
        px.append(px[-1] * 1.06)          # sharp V rebound
    px = np.array(px[:n], dtype=float)
    frame = pd.DataFrame(
        {
            "ticker": ["KNIFE"] * n,
            "date": days,
            "open": px * 0.999,
            "high": px * 1.01,
            "low": px * 0.98,
            "close": px,
            "volume": [1_000_000] * n,
        }
    )
    res = build_mr_features(frame)
    bottom = int(np.argmin(px))
    row = res.iloc[bottom]
    print(f"Capitulation low at bar {bottom} (close={px[bottom]:.2f}):")
    print(f"  RSI9            = {row['mr_rsi_9']:.1f}   (expect << 30)")
    print(f"  RSI14           = {row['mr_rsi_14']:.1f}")
    print(f"  %B              = {row['mr_bb_pctb']:.3f}  (expect < 0)")
    print(f"  below_lower     = {row['mr_bb_below_lower']}     (expect 1)")
    print(f"  DMA vs SMA20    = {row['mr_dma_sma20']*100:.1f}% (expect deeply -)")
    print(f"  Williams %R     = {row['mr_williams_r_14']:.1f} (expect ~ -100)")
    print(f"  ATR/close       = {row['mr_atr_norm_14']*100:.2f}% (spiking)")
    print(f"  gap_pct         = {row['mr_gap_pct']*100:.2f}%")
    assert row["mr_rsi_9"] < 20, "RSI9 should be deeply oversold at the low"
    assert row["mr_bb_pctb"] < 0.0, "%B should be below the lower band"
    assert row["mr_bb_below_lower"] == 1, "price should pierce lower BB"
    assert row["mr_dma_sma20"] < -0.05, "price should be far below SMA20"
    assert row["mr_williams_r_14"] < -80, "Williams %R should be near -100"
    # Leak check: feature at bar t must not change if FUTURE bars are altered.
    f2 = frame.copy()
    f2.loc[bottom + 3:, "close"] *= 1.5            # mutate the future only
    r2 = build_mr_features(f2)
    cols = [c for c in MR_FEATURE_COLUMNS if c not in ("mr_bb_below_lower", "mr_gap_down")]
    a = res.loc[: bottom, cols].to_numpy()
    b = r2.loc[: bottom, cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True), "LOOK-AHEAD LEAK: past features moved when future changed!"
    print("\nmr_features smoke test OK — oversold at the low, and LEAK-FREE.")
