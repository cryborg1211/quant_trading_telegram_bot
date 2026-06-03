"""Rule-based "8 Market Regimes" classifier — pure Polars, zero ML.

WHY RULE-BASED
──────────────
The V4 stack is a fast, Parquet-driven tabular ensemble.  A regime label is a
*structural context* signal (what kind of tape are we in?), not a prediction —
so it is computed by transparent, auditable `pl.when().then()` heuristics rather
than a black-box model.  No deep learning ⇒ nothing to overfit, and it runs in
one vectorized lazy pass over the whole panel.

ARCHITECTURE FIT
────────────────
The V4 model panel is built from RAW OHLCV (`src/backtest/pipeline.build_features`)
— it has no ADX/RSI/ATR columns of its own (those live in the MR sub-model).  So
this module computes its OWN regime indicators from `open/high/low/close/volume`
(guaranteed present), mirroring the audited formulas in `src/features/mr_features.py`.

LEAK-SAFETY
───────────
Every rolling / shift / diff window ENDS at bar t and is grouped `.over("ticker")`,
so a regime at bar t uses only data up to and including t — identical discipline
to `mr_features.py`.  Warm-up rows (insufficient history) fall through to the
neutral CHOPPY default and are non-null, so the downstream `market_regime`
column never injects NaN into the feature matrix.

OUTPUT
──────
`build_regime_features(lf)` returns the input LazyFrame with ONE integer column
`market_regime` ∈ {0..7} appended.  All scratch indicator columns are dropped.

    0 Freeze            extremely low ATR & volume — dead tape
    1 Squeeze           Bollinger bandwidth compressed — coiling
    2 Early Trend       price breaking a band on rising volume
    3 Strong Trend      high efficiency-ratio + aligned moving averages
    4 Climax            blow-off: far outside the upper band + volume spike + hot RSI
    5 Mean Reversion    RSI at an extreme (<30 or >70)
    6 Choppy            low efficiency-ratio / no structure  (also the default)
    7 Liquidity Sweep   big-range bar with a tiny body (long wicks)
"""
from __future__ import annotations

import polars as pl

# Integer regime → human-readable Vietnamese label (shown on the Telegram card).
REGIME_LABELS_VI: dict[int, str] = {
    0: "Đóng Băng",            # Freeze
    1: "Tích Lũy (Nén)",       # Squeeze
    2: "Khởi Đầu Xu Hướng",    # Early Trend
    3: "Xu Hướng Mạnh",        # Strong Trend
    4: "Cao Trào",             # Climax
    5: "Hồi Quy Trung Bình",   # Mean Reversion
    6: "Đi Ngang (Nhiễu)",     # Choppy
    7: "Quét Thanh Khoản",     # Liquidity Sweep
}

# English labels — handy for logs / English-locale callers.
REGIME_LABELS_EN: dict[int, str] = {
    0: "Freeze", 1: "Squeeze", 2: "Early Trend", 3: "Strong Trend",
    4: "Climax", 5: "Mean Reversion", 6: "Choppy", 7: "Liquidity Sweep",
}

# The default regime for warm-up / structureless rows.
DEFAULT_REGIME: int = 6  # Choppy

_EPS = 1e-12

# Scratch indicator columns this module creates then drops (kept explicit so the
# drop works in lazy mode without a schema round-trip).
_SCRATCH: tuple[str, ...] = (
    "_tr", "_abs_diff", "_gain", "_loss",
    "_sma10", "_sma20", "_sma50", "_sd20", "_vol_ma20", "_vol_sd20",
    "_atr_norm", "_bbw", "_pctb", "_avg_gain", "_avg_loss", "_er", "_vol_z",
    "_range_norm", "_body_frac", "_atr_norm_ma60", "_bbw_ma60", "_rsi",
)


def regime_label_vi(regime: int | None) -> str:
    """Vietnamese label for a regime id; safe on None / out-of-range."""
    if regime is None:
        return "Không xác định"
    return REGIME_LABELS_VI.get(int(regime), "Không xác định")


def build_regime_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append an integer `market_regime` ∈ {0..7} column, classified from raw
    OHLCV via vectorized, leak-free heuristics.

    Parameters
    ----------
    lf : pl.LazyFrame
        Must expose ``ticker, date, open, high, low, close, volume``.

    Returns
    -------
    pl.LazyFrame
        ``lf`` + ``market_regime`` (Int8, non-null); all scratch columns dropped.
    """
    g = "ticker"
    lf = lf.sort([g, "date"])

    prev_close = pl.col("close").shift(1).over(g)

    # ── Block 1: base series (raw-only + inline shift/diff) ──────────────────
    lf = lf.with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ).alias("_tr"),
        pl.col("close").diff().over(g).abs().alias("_abs_diff"),
        pl.when(pl.col("close").diff().over(g) > 0)
          .then(pl.col("close").diff().over(g)).otherwise(0.0).alias("_gain"),
        pl.when(pl.col("close").diff().over(g) < 0)
          .then(-pl.col("close").diff().over(g)).otherwise(0.0).alias("_loss"),
        pl.col("close").rolling_mean(10).over(g).alias("_sma10"),
        pl.col("close").rolling_mean(20).over(g).alias("_sma20"),
        pl.col("close").rolling_mean(50).over(g).alias("_sma50"),
        pl.col("close").rolling_std(20, ddof=0).over(g).alias("_sd20"),
        pl.col("volume").rolling_mean(20).over(g).alias("_vol_ma20"),
        pl.col("volume").rolling_std(20, ddof=0).over(g).alias("_vol_sd20"),
        ((pl.col("high") - pl.col("low")) / (pl.col("close") + _EPS)).alias("_range_norm"),
        ((pl.col("close") - pl.col("open")).abs()
         / ((pl.col("high") - pl.col("low")) + _EPS)).alias("_body_frac"),
    ])

    # ── Block 2: indicators that depend on Block 1 ───────────────────────────
    lf = lf.with_columns([
        (pl.col("_tr").rolling_mean(14).over(g) / (pl.col("close") + _EPS)).alias("_atr_norm"),
        ((4.0 * pl.col("_sd20")) / (pl.col("_sma20") + _EPS)).alias("_bbw"),
        ((pl.col("close") - (pl.col("_sma20") - 2.0 * pl.col("_sd20")))
         / ((4.0 * pl.col("_sd20")) + _EPS)).alias("_pctb"),
        pl.col("_gain").rolling_mean(14).over(g).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(14).over(g).alias("_avg_loss"),
        ((pl.col("close") - pl.col("close").shift(14).over(g)).abs()
         / (pl.col("_abs_diff").rolling_sum(14).over(g) + _EPS)).alias("_er"),
        ((pl.col("volume") - pl.col("_vol_ma20")) / (pl.col("_vol_sd20") + _EPS)).alias("_vol_z"),
    ])

    # ── Block 3: indicators that depend on Block 2 ───────────────────────────
    lf = lf.with_columns([
        pl.col("_atr_norm").rolling_mean(60).over(g).alias("_atr_norm_ma60"),
        pl.col("_bbw").rolling_mean(60).over(g).alias("_bbw_ma60"),
        (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + _EPS))).alias("_rsi"),
    ])

    # ── Boolean primitives (relative thresholds → scale-free across tickers) ─
    low_atr      = pl.col("_atr_norm") < (0.60 * pl.col("_atr_norm_ma60"))
    low_vol      = pl.col("_vol_z") < -0.50
    vol_spike    = pl.col("_vol_z") > 2.0
    vol_up       = pl.col("_vol_z") > 1.0
    squeeze      = pl.col("_bbw") < (0.50 * pl.col("_bbw_ma60"))
    above_band   = pl.col("_pctb") > 1.0
    below_band   = pl.col("_pctb") < 0.0
    strong_trend = pl.col("_er") > 0.50
    choppy       = pl.col("_er") < 0.30
    ma_up        = (pl.col("_sma10") > pl.col("_sma20")) & (pl.col("_sma20") > pl.col("_sma50"))
    ma_dn        = (pl.col("_sma10") < pl.col("_sma20")) & (pl.col("_sma20") < pl.col("_sma50"))
    ma_aligned   = ma_up | ma_dn
    rsi_hot      = pl.col("_rsi") > 70.0
    rsi_extreme  = (pl.col("_rsi") > 70.0) | (pl.col("_rsi") < 30.0)
    big_range    = pl.col("_range_norm") > (2.0 * pl.col("_atr_norm"))
    small_body   = pl.col("_body_frac") < 0.30

    # ── Classification — FIRST match wins (most-extreme / most-specific first) ─
    lf = lf.with_columns(
        pl.when(low_atr & low_vol).then(0)                    # Freeze
          .when(big_range & small_body).then(7)               # Liquidity Sweep
          .when(above_band & vol_spike & rsi_hot).then(4)     # Climax
          .when(strong_trend & ma_aligned).then(3)            # Strong Trend
          .when((above_band | below_band) & vol_up).then(2)   # Early Trend
          .when(squeeze).then(1)                              # Squeeze
          .when(rsi_extreme).then(5)                          # Mean Reversion
          .when(choppy).then(6)                               # Choppy
          .otherwise(DEFAULT_REGIME)                          # default → Choppy
          .fill_null(DEFAULT_REGIME)                          # warm-up rows → neutral
          .cast(pl.Int8)
          .alias("market_regime")
    )

    return lf.drop(list(_SCRATCH))


if __name__ == "__main__":
    # Smoke test: synthetic tape with a calm stretch, a strong uptrend, and a
    # capitulation. Asserts the output is a valid, non-null 0..7 integer column
    # and is LEAK-FREE (mutating the future must not move a past regime).
    import numpy as np
    import datetime as _dt

    rng = np.random.default_rng(0)
    n = 200
    days = [_dt.date(2025, 1, 1) + _dt.timedelta(days=i) for i in range(n)]
    px = [100.0]
    for i in range(1, n):
        if i < 70:           # calm: tiny noise (Freeze/Squeeze/Choppy territory)
            px.append(px[-1] * (1 + rng.normal(0, 0.001)))
        elif i < 140:        # strong uptrend
            px.append(px[-1] * (1 + 0.012 + rng.normal(0, 0.003)))
        else:                # sharp reversal / capitulation
            px.append(px[-1] * (1 - 0.015 + rng.normal(0, 0.006)))
    px = np.array(px)
    frame = pl.DataFrame({
        "ticker": ["TEST"] * n,
        "date": days,
        "open": px * (1 + rng.normal(0, 0.002, n)),
        "high": px * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": px * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": px,
        "volume": (1_000_000 * (1 + np.abs(rng.normal(0, 0.5, n)))).astype(float),
    })

    out = build_regime_features(frame.lazy()).collect()
    reg = out["market_regime"]
    assert "market_regime" in out.columns
    assert reg.null_count() == 0, "market_regime must be non-null"
    assert reg.min() >= 0 and reg.max() <= 7, "regime out of [0,7]"
    assert reg.dtype == pl.Int8
    assert not any(c.startswith("_") for c in out.columns), "scratch columns leaked"

    # Leak check: changing only the FUTURE must not alter a past regime.
    f2 = frame.clone()
    fut = pl.col("close")  # mutate the last 30 closes upward
    f2 = f2.with_columns(
        pl.when(pl.arange(0, n) >= n - 30).then(pl.col("close") * 1.5)
          .otherwise(pl.col("close")).alias("close")
    )
    out2 = build_regime_features(f2.lazy()).collect()
    a = out["market_regime"].to_numpy()[: n - 31]
    b = out2["market_regime"].to_numpy()[: n - 31]
    assert (a == b).all(), "LOOK-AHEAD LEAK: a past regime moved when the future changed!"

    dist = out["market_regime"].value_counts().sort("market_regime")
    print("market_regime smoke OK — non-null 0..7, leak-free. Distribution:")
    for row in dist.iter_rows(named=True):
        print(f"  regime {row['market_regime']} ({REGIME_LABELS_EN[row['market_regime']]:<15}): {row['count']}")
