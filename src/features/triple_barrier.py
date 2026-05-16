"""Volatility-scaled Triple-Barrier labeling (Lopez de Prado, AFML Ch. 3).

Drop-in replacement for the fixed ±3% logic in
``alpha360_generator._generate_targets``. Pure math/feature code — does NOT
touch DuckDB or any infrastructure.

Why this exists
───────────────
Fixed percentage bands treat a low-vol large-cap (VCB) and a high-vol
small-cap identically. The triple-barrier method scales every barrier by
the asset's *own* recent volatility, so a "win" is always a move that is
large relative to that name's noise.

Label semantics (de Prado convention):
    +1  → upper (profit-taking) barrier touched first
    -1  → lower (stop-loss) barrier touched first
     0  → vertical (time) barrier hit first  ⇒ "sideways"

Quant V6 classifier contract:
    The stacking model expects classes {0:DOWN, 1:SIDE, 2:UP}. We expose
    BOTH the de Prado bin (``target_bin_*``) and the mapped pipeline class
    (``target_class_*`` = bin + 1) so nothing downstream changes.

CRITICAL OUTPUT: ``t1_*`` (event-end date)
    Every labeled sample carries the date on which its label became known.
    This is the observation window end — ``PurgedKFold`` requires it to
    purge train/test overlap. Persist it to the feature parquet.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# de Prado bin → Quant V6 stacking class (DOWN=0, SIDE=1, UP=2)
BIN_TO_CLASS: dict[int, int] = {-1: 0, 0: 1, 1: 2}

# polars renamed ewm_std's `min_periods` → `min_samples` in 1.21.0.
# Resolve the correct kwarg once, version-robustly.
_PL_VERSION = tuple(int(p) for p in pl.__version__.split(".")[:2])
_EWM_MIN_KW = "min_samples" if _PL_VERSION >= (1, 21) else "min_periods"


def add_daily_vol(
    df: pl.DataFrame,
    *,
    close_col: str = "close",
    ticker_col: str = "ticker",
    date_col: str = "date",
    span: int = 20,
    min_periods: int = 10,
) -> pl.DataFrame:
    """Append ``_tb_sigma`` = EWM std of daily simple returns, per ticker.

    span=20 ≈ a 20-day exponentially-weighted daily-return volatility.
    The barriers at time *t* are scaled by ``_tb_sigma`` observed at *t*
    (causal — no look-ahead).
    """
    out = df.sort([ticker_col, date_col]).with_columns(
        (
            pl.col(close_col) / pl.col(close_col).shift(1).over(ticker_col) - 1.0
        ).alias("_tb_ret")
    )
    return out.with_columns(
        pl.col("_tb_ret")
        .ewm_std(span=span, ignore_nulls=True, **{_EWM_MIN_KW: min_periods})
        .over(ticker_col)
        .alias("_tb_sigma")
    )


def _first_touch_numpy(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
    use_intrabar_extremes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-ticker first-touch resolver. Vectorized across rows; the only
    Python loop is over the (tiny) horizon dimension.

    Returns
    -------
    bin_   : float ndarray  ({-1,0,1}, np.nan where unlabelable)
    t1_off : float ndarray  (#bars from t to the deciding bar; np.nan if NaN label)
    ret    : float ndarray  (realized close-to-close return at t1; diagnostics)
    """
    n = close.shape[0]
    idx0 = np.arange(n)

    pt = pt_mult * sigma  # upper barrier as a return threshold (>0)
    sl = sl_mult * sigma  # lower barrier magnitude (>0)

    first_up = np.full(n, np.inf)
    first_dn = np.full(n, np.inf)

    for k in range(1, horizon + 1):
        fwd = idx0 + k
        valid = fwd < n
        ok = valid & np.isfinite(pt) & np.isfinite(sl) & (sigma > 0)

        up_metric = np.full(n, np.nan)
        dn_metric = np.full(n, np.nan)
        if use_intrabar_extremes:
            up_metric[valid] = high[fwd[valid]] / close[valid] - 1.0
            dn_metric[valid] = low[fwd[valid]] / close[valid] - 1.0
        else:
            r = np.full(n, np.nan)
            r[valid] = close[fwd[valid]] / close[valid] - 1.0
            up_metric, dn_metric = r, r

        hit_up = np.zeros(n, dtype=bool)
        hit_dn = np.zeros(n, dtype=bool)
        hit_up[ok] = up_metric[ok] >= pt[ok]
        hit_dn[ok] = dn_metric[ok] <= -sl[ok]

        first_up[hit_up & np.isinf(first_up)] = k
        first_dn[hit_dn & np.isinf(first_dn)] = k

    touched = np.minimum(first_up, first_dn)
    finite = np.isfinite(touched)
    full_window = (idx0 + horizon) < n  # vertical barrier confirmable
    labelable = finite | full_window

    up_first = first_up < first_dn
    dn_first = first_dn < first_up

    bin_ = np.full(n, np.nan)
    t1_off = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    raw_bin = np.where(up_first, 1.0, np.where(dn_first, -1.0, 0.0))
    raw_off = np.where(finite, touched, float(horizon))

    bin_[labelable] = raw_bin[labelable]
    t1_off[labelable] = raw_off[labelable]

    end_idx = np.clip((idx0 + np.nan_to_num(t1_off, nan=0.0)).astype(np.int64), 0, n - 1)
    ret[labelable] = close[end_idx[labelable]] / close[labelable] - 1.0
    return bin_, t1_off, ret


def add_triple_barrier_labels(
    df: pl.DataFrame,
    *,
    horizon: int = 5,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    suffix: str = "5d",
    vol_span: int = 20,
    vol_min_periods: int = 10,
    use_intrabar_extremes: bool = False,
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "close",
) -> pl.DataFrame:
    """Return ``df`` with triple-barrier columns added for one horizon.

    Adds (``{suffix}`` defaults to ``"5d"``):
        target_bin_{suffix}     Int8   de Prado label  {-1,0,1}
        target_class_{suffix}   Int8   pipeline class  {0,1,2}  (= bin+1)
        target_return_{suffix}  Float64 realized close-to-close return at t1
        t1_{suffix}             Date   event-end date  ← REQUIRED by PurgedKFold

    Call once per horizon (``suffix="5d"``, then ``suffix="20d"``) to
    reproduce the existing dual-horizon column contract.

    ``use_intrabar_extremes=True`` uses next bars' high/low (more realistic:
    a 2σ take-profit is usually touched intrabar). Default False = the
    conservative close-path (also the only option if high/low absent).
    """
    if use_intrabar_extremes and not {"high", "low"}.issubset(df.columns):
        raise ValueError("use_intrabar_extremes=True requires 'high' and 'low' columns")

    work = add_daily_vol(
        df,
        close_col=close_col,
        ticker_col=ticker_col,
        date_col=date_col,
        span=vol_span,
        min_periods=vol_min_periods,
    )

    bin_col = f"target_bin_{suffix}"
    cls_col = f"target_class_{suffix}"
    ret_col = f"target_return_{suffix}"
    t1_col = f"t1_{suffix}"

    labeled_parts: list[pl.DataFrame] = []
    for part in work.partition_by(ticker_col, maintain_order=True):
        part = part.sort(date_col)
        close = part[close_col].to_numpy().astype(np.float64)
        high = (
            part["high"].to_numpy().astype(np.float64)
            if "high" in part.columns
            else close
        )
        low = (
            part["low"].to_numpy().astype(np.float64)
            if "low" in part.columns
            else close
        )
        sigma = part["_tb_sigma"].to_numpy().astype(np.float64)
        dates = part[date_col].to_numpy()

        bin_, t1_off, ret = _first_touch_numpy(
            close, high, low, sigma, horizon, pt_mult, sl_mult, use_intrabar_extremes
        )

        n = close.shape[0]
        end_pos = np.clip(
            (np.arange(n) + np.nan_to_num(t1_off, nan=0.0)).astype(np.int64), 0, n - 1
        )
        t1_dates = np.where(np.isfinite(t1_off), dates[end_pos], np.datetime64("NaT"))

        cls = np.where(np.isfinite(bin_), bin_ + 1.0, np.nan)

        labeled_parts.append(
            part.with_columns(
                [
                    pl.Series(bin_col, bin_, dtype=pl.Float64)
                    .round(0)
                    .cast(pl.Int8, strict=False),
                    pl.Series(cls_col, cls, dtype=pl.Float64)
                    .round(0)
                    .cast(pl.Int8, strict=False),
                    pl.Series(ret_col, ret, dtype=pl.Float64),
                    pl.Series(t1_col, t1_dates).cast(pl.Date, strict=False),
                ]
            )
        )

    return pl.concat(labeled_parts, how="vertical").drop(["_tb_ret", "_tb_sigma"])


if __name__ == "__main__":
    # Smoke test: a synthetic up-trender must label +1 (class 2).
    import datetime as _dt

    days = [_dt.date(2024, 1, 1) + _dt.timedelta(days=i) for i in range(40)]
    up = pl.DataFrame(
        {
            "ticker": ["AAA"] * 40,
            "date": days,
            "close": [100.0 * (1.02**i) for i in range(40)],
        }
    )
    out = add_triple_barrier_labels(up, horizon=5, pt_mult=2.0, sl_mult=2.0)
    head = out.head(8).select(["date", "target_bin_5d", "target_class_5d", "t1_5d"])
    for row in head.to_dicts():  # to_dicts() avoids polars' unicode table repr
        print(row)
    assert out["target_class_5d"].drop_nulls().mode().to_list()[0] == 2, "uptrend => class 2"
    print("triple_barrier smoke test OK")
