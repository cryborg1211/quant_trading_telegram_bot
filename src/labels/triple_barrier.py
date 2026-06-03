"""
src/labels/triple_barrier.py — Quant Engine V2.0, Phase 2

Dynamic labeling and sample-weighting for sequence models, faithful to
de Prado, *Advances in Financial Machine Learning* (2018), Chapters 3 & 4.

╔══════════════════════════════════════════════════════════════════════════════╗
║  Why we are tearing out V1's fixed-horizon labels                            ║
║                                                                              ║
║  V1 emits `target_class_5d`/`target_class_20d` via vol-scaled triple-barrier ║
║  but trains the model with EQUAL sample weights.  This silently inflates the ║
║  effective degrees of freedom: overlapping 5-day events share information,   ║
║  so the LSTM sees the SAME piece of price action labeled into N near-       ║
║  identical samples.  Cross-validation hit rates become optimistic; out-of-   ║
║  sample Sharpe collapses (the §11 "small-n" caveat is partly explained by   ║
║  this).                                                                      ║
║                                                                              ║
║  This module fixes the labeling AND adds the sample-weight layer:            ║
║                                                                              ║
║    • `get_daily_vol`           — EWMA σ_t for dynamic barrier width          ║
║    • `apply_pt_sl_on_t1`       — first-touch barrier resolver                ║
║    • `get_vertical_barriers`   — t1 (max-holding) helper                     ║
║    • `get_events`              — full event DataFrame builder                ║
║    • `get_bins`                — class labels {0, 1, 2}                       ║
║    • `get_num_co_events`       — bar-by-bar concurrent event count           ║
║    • `get_sample_tw`           — average uniqueness weight (AFML 4.2)         ║
║    • `get_sample_weights`      — return-attribution weight (AFML 4.10) ★     ║
║    • `triple_barrier_pipeline` — Polars panel → labels + weights, one call    ║
║                                                                              ║
║  ★ The return-attribution weight is the institutional dividing line:        ║
║    train_lstm.py MUST consume `sample_weight=w` in the loss; otherwise the   ║
║    upgrade to LSTM is mathematically pointless.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Vectorisation strategy
──────────────────────
The naïve AFML pseudocode is per-event Python.  Two algorithms are accelerated:

  • `get_num_co_events` — O(N_events × horizon)  →  O(N_events + N_bars)
    via cumulative-sum on a +1/−1 delta array.

  • `get_sample_weights` — same speedup: cumsum of (log-ret / c_t), then
    each event's weight = cumsum[t1] − cumsum[t0 − 1].

The first-touch detection in `apply_pt_sl_on_t1` is path-dependent and cannot
be fully vectorised across events; we vectorise across rows with a tight
inner loop over the (typically small) horizon dimension.

References
──────────
  AFML §3.1  Daily volatility estimator
  AFML §3.2  Triple-barrier first-touch
  AFML §3.3  Symmetric labels via `getBins`
  AFML §4.1  Number of concurrent labels
  AFML §4.2  Average uniqueness of a label
  AFML §4.10 Return-attribution sample weights
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm optional — degrade to a no-op pass-through
    def tqdm(iterable=None, **_kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else None

LOGGER = logging.getLogger(__name__)

__all__ = [
    "TripleBarrierConfig",
    "get_daily_vol",
    "apply_pt_sl_on_t1",
    "get_vertical_barriers",
    "get_events",
    "get_bins",
    "get_num_co_events",
    "get_sample_tw",
    "get_sample_weights",
    "triple_barrier_pipeline",
]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TripleBarrierConfig:
    """Hyper-parameters for the triple-barrier labeling pipeline."""
    pt_mult: float = 1.5
    """Upper-barrier multiplier: PT = +pt_mult × σ_t.  AFML default 2.0; 1.5 = swing."""

    sl_mult: float = 1.5
    """Lower-barrier multiplier: SL = −sl_mult × σ_t.  AFML default 2.0; 1.5 = swing."""

    vol_span: int = 20
    """EWMA span (in bars) for the daily volatility estimator."""

    horizon: int = 10
    """Vertical (time) barrier in bars.  T+10 swing horizon (was 5)."""

    min_ret: float = 0.0
    """Drop events with σ_t below this floor (sub-noise targets are skipped)."""

    label_scheme: str = "012"
    """
    "012": DOWN=0, FLAT=1, UP=2   (Quant V6 classifier contract)
    "raw": -1 / 0 / +1            (de Prado native)
    """

    use_intrabar_extremes: bool = True
    """
    If True and `high`/`low` columns are present, detect intrabar barrier
    touches.  Otherwise use close-only paths (under-detects PT/SL hits).
    """


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dynamic volatility (AFML §3.1)
# ─────────────────────────────────────────────────────────────────────────────

def get_daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponentially-weighted std of 1-bar simple returns.

    σ_t = EWMA_span( r_t ),    r_t = close_t / close_{t−1} − 1

    Properties:
        • Reacts faster than rolling std but smoother than instantaneous |r_t|.
        • EWMA places larger weight on recent observations, capturing the
          regime in which the next trade will be placed.
        • Computed per ticker by the caller (typically inside a groupby).

    Args:
        close: pandas Series of close prices, indexed by date.
        span:  EWMA span in bars (default 20 ≈ 1 trading month).

    Returns:
        pandas Series of EWMA std-of-returns, same index as `close`.
        The first `span` bars are warm-up and may be unreliable.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("get_daily_vol: `close` must be a pandas Series.")

    # Defensive: a zero/negative close makes pct_change blow up to ±inf and
    # poisons the EWMA σ (→ poisons every barrier width). Mask non-positive
    # prices to NaN first, then strip any residual inf before the EWMA.
    safe_close = close.where(close > 0)
    returns = safe_close.pct_change().replace([np.inf, -np.inf], np.nan)
    return returns.ewm(span=span, min_periods=max(1, span // 2)).std()


# ─────────────────────────────────────────────────────────────────────────────
# 2. First-touch barrier resolver (AFML §3.2)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    use_intrabar_extremes: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    For each event, return the first time the upper or lower barrier is touched.

    The event paths from `events.index` (entry date t0) up to `events['t1']`
    (vertical barrier) are scanned bar-by-bar.  The first bar whose forward
    return crosses ±(pt_sl × trgt) is recorded.

    Conservative tie-break (institutional risk-mgmt convention)
    ───────────────────────────────────────────────────────────
    If a single wide-range bar's high crosses +PT *and* its low crosses −SL,
    the intrabar order is unknowable from end-of-day data.  We assume the
    stop-loss fired first (`first_dn = first_up`) — this is the V1 convention
    and aligns with the AFML §3.4 risk-management argument.

    Args:
        close:                 pandas Series of close prices, indexed by date.
        events:                pandas DataFrame indexed by t0, with columns
                                 't1'   — vertical barrier (latest exit date)
                                 'trgt' — barrier width unit (σ_t)
        pt_sl:                 (upper_mult, lower_mult).  Zero disables that side.
        high, low:             optional intrabar extremes Series; if provided
                                and `use_intrabar_extremes=True`, used for the
                                touch check.
        use_intrabar_extremes: see above.

    Returns:
        DataFrame indexed identically to `events`, with columns:
            'pt' — first date upper barrier touched (NaT if never)
            'sl' — first date lower barrier touched (NaT if never)
            't1' — first-touch date (min of pt, sl, vertical barrier)
    """
    # ── VECTORISED FIRST-TOUCH (C-speed, no iterrows) ────────────────────────
    # Strategy (ports V1's `_first_touch_numpy`): the only Python loop is over
    # the SMALL forward-offset dimension `k = 0 … max_horizon`; every event is
    # processed simultaneously with NumPy. Complexity O(n_events · max_horizon)
    # but executed in C. For the fixed-horizon pipeline max_horizon ≈ cfg.horizon
    # (single digits), so this collapses the old 15-min iterrows scan to < 1 s.
    #
    # Functional equivalence with the prior per-event implementation is exact:
    #   • scan window is [t0, t1_v] INCLUSIVE (offset 0 = the entry bar);
    #   • barrier metrics use intrabar high/low (use_ohlc) or close otherwise;
    #   • non-positive / non-finite prices are ignored (cannot trip a barrier);
    #   • events with invalid σ, invalid entry price, or < 2 bars get no touch
    #     and t1 = t1_v;
    #   • first-touch = earliest of (PT, SL, vertical); same-bar PT∧SL tie →
    #     conservative SL (PT column dropped to NaT, SL kept, t1 = that bar).
    pt_mult, sl_mult = pt_sl
    use_ohlc = use_intrabar_extremes and high is not None and low is not None

    n = len(close)
    idx_values = close.index.values                  # datetime64[ns]
    last_bar = close.index[-1]
    n_ev = len(events)
    LOGGER.debug("apply_pt_sl_on_t1: vectorised first-touch over %d events", n_ev)

    NAT = np.datetime64("NaT", "ns")
    if n_ev == 0:
        empty = pd.DataFrame(index=events.index, columns=["pt", "sl", "t1"], dtype="datetime64[ns]")
        return empty

    close_arr = close.to_numpy(dtype=np.float64)
    high_arr = high.to_numpy(dtype=np.float64) if use_ohlc else close_arr
    low_arr = low.to_numpy(dtype=np.float64) if use_ohlc else close_arr

    # Entry positions (t0 must be a member of the close index — same contract as
    # the original's `close.at[t0]`); detect violations explicitly.
    pos_t0 = close.index.get_indexer(events.index)
    if (pos_t0 < 0).any():
        raise KeyError("apply_pt_sl_on_t1: some event t0 is not in the close index.")

    # Vertical-barrier positions: NaT t1 → last bar; otherwise the last index
    # position with date ≤ t1_v  (== `close.loc[t0:t1_v]` right edge).
    t1_raw = events["t1"]
    t1_filled = t1_raw.where(t1_raw.notna(), last_bar)
    t1_v_values = pd.to_datetime(t1_filled).to_numpy()
    pos_t1 = np.searchsorted(idx_values, t1_v_values, side="right") - 1
    pos_t1 = np.clip(pos_t1, pos_t0, n - 1)

    p0 = close_arr[pos_t0]
    trgt = events["trgt"].to_numpy(dtype=np.float64)
    horizon_i = pos_t1 - pos_t0                       # ≥ 0

    # Scannable = ≥ 2 bars AND finite-positive σ AND finite-positive entry price.
    valid_trgt = np.isfinite(trgt) & (trgt > 0)
    scannable = (horizon_i >= 1) & valid_trgt & np.isfinite(p0) & (p0 > 0)

    # Barrier thresholds (only on valid σ; elsewhere set so a touch can't fire).
    pt_thr = np.full(n_ev, np.inf, dtype=np.float64)   # up_metric ≥ +inf → False
    sl_thr = np.full(n_ev, -np.inf, dtype=np.float64)  # dn_metric ≤ −inf → False
    if pt_mult > 0:
        pt_thr[valid_trgt] = pt_mult * trgt[valid_trgt]
    if sl_mult > 0:
        sl_thr[valid_trgt] = -sl_mult * trgt[valid_trgt]

    first_up = np.full(n_ev, np.inf, dtype=np.float64)  # earliest PT offset (inf=never)
    first_dn = np.full(n_ev, np.inf, dtype=np.float64)  # earliest SL offset

    max_h = int(horizon_i[scannable].max()) if scannable.any() else 0
    for k in range(0, max_h + 1):
        in_range = scannable & (k <= horizon_i)
        if not in_range.any():
            continue
        fwd = np.clip(pos_t0 + k, 0, n - 1)             # bar at offset k

        if use_ohlc:
            hi_k = high_arr[fwd]
            lo_k = low_arr[fwd]
            valid_hi = in_range & np.isfinite(hi_k) & (hi_k > 0)
            valid_lo = in_range & np.isfinite(lo_k) & (lo_k > 0)
            up_metric = np.full(n_ev, -np.inf, dtype=np.float64)
            dn_metric = np.full(n_ev, np.inf, dtype=np.float64)
            up_metric[valid_hi] = hi_k[valid_hi] / p0[valid_hi] - 1.0
            dn_metric[valid_lo] = lo_k[valid_lo] / p0[valid_lo] - 1.0
        else:
            c_k = close_arr[fwd]
            valid_c = in_range & np.isfinite(c_k) & (c_k > 0)
            up_metric = np.full(n_ev, -np.inf, dtype=np.float64)
            dn_metric = np.full(n_ev, np.inf, dtype=np.float64)
            up_metric[valid_c] = c_k[valid_c] / p0[valid_c] - 1.0
            dn_metric[valid_c] = c_k[valid_c] / p0[valid_c] - 1.0

        if pt_mult > 0:
            newly_up = in_range & np.isinf(first_up) & (up_metric >= pt_thr)
            first_up[newly_up] = k
        if sl_mult > 0:
            newly_dn = in_range & np.isinf(first_dn) & (dn_metric <= sl_thr)
            first_dn[newly_dn] = k

    # ── Assemble pt / sl / t1 columns (conservative same-bar tie → SL) ───────
    fin_up = np.isfinite(first_up)
    fin_dn = np.isfinite(first_dn)
    tie = fin_up & fin_dn & (first_up == first_dn)      # PT∧SL same bar → drop PT

    def _dates(offsets: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pos = np.clip(pos_t0 + np.where(np.isfinite(offsets), offsets, 0).astype(np.int64), 0, n - 1)
        return np.where(mask, idx_values[pos], NAT)

    pt_dates = _dates(first_up, fin_up & ~tie)
    sl_dates = _dates(first_dn, fin_dn)

    touch_off = np.minimum(first_up, first_dn)          # earliest touch (inf=none)
    fin_touch = np.isfinite(touch_off)
    t1_touch = _dates(touch_off, fin_touch)
    # No touch → vertical barrier; a touch is always ≤ t1_v by construction.
    t1_dates = np.where(fin_touch, t1_touch, t1_v_values.astype("datetime64[ns]"))

    out = pd.DataFrame(
        {"pt": pt_dates, "sl": sl_dates, "t1": t1_dates}, index=events.index
    )
    out["pt"] = pd.to_datetime(out["pt"])
    out["sl"] = pd.to_datetime(out["sl"])
    out["t1"] = pd.to_datetime(out["t1"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vertical barrier helper
# ─────────────────────────────────────────────────────────────────────────────

def get_vertical_barriers(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    horizon: int,
) -> pd.Series:
    """
    For each event start t0 in `t_events`, return the bar `horizon` steps later.

    If t0 + horizon exceeds the data range, returns the last available bar
    (event is censored by the end of history).

    Args:
        close:    pandas Series of close prices, indexed by date.
        t_events: event start dates.
        horizon:  number of bars to the vertical barrier.

    Returns:
        pandas Series of t1 dates, indexed by t_events.
    """
    idx = close.index
    locs = idx.searchsorted(t_events)
    t1_locs = np.minimum(locs + horizon, len(idx) - 1)
    return pd.Series(idx[t1_locs], index=t_events, name="t1")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full event builder (AFML §3.3)
# ─────────────────────────────────────────────────────────────────────────────

def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    target: pd.Series,
    min_ret: float = 0.0,
    t1: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    use_intrabar_extremes: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Build the canonical events DataFrame used by `get_bins` and `get_sample_weights`.

    For each event start t0:
        - Skip if σ_t0 < min_ret.
        - Compute first-touch date via `apply_pt_sl_on_t1`.

    Args:
        close, high, low:      OHLC Series, indexed by date.
        t_events:              Candidate event start dates.
        pt_sl:                 (pt_mult, sl_mult) barrier multipliers.
        target:                σ_t Series (from `get_daily_vol`) indexed by date.
        min_ret:               Drop events with σ < min_ret.
        t1:                    Vertical barriers; if None, all events get NaT
                                 (only horizontal barriers will be checked).
        use_intrabar_extremes: Pass-through to `apply_pt_sl_on_t1`.

    Returns:
        DataFrame indexed by t0 with columns:
            't1'    — first-touch date (min of PT, SL, vertical)
            'trgt'  — σ_t0 (the unit barrier width)
            'pt'    — date PT was hit (NaT if not)
            'sl'    — date SL was hit (NaT if not)
    """
    trgt = target.reindex(t_events)
    trgt = trgt[trgt > min_ret].dropna()
    # Defensive: drop events whose ENTRY close is non-positive / non-finite.
    # A zero entry price would divide-by-zero downstream (apply_pt_sl_on_t1,
    # get_bins). Excise these events at the source so they never enter the
    # accumulators that AFML §4 weighting sums over.
    if not trgt.empty:
        entry_px = close.reindex(trgt.index).to_numpy(dtype=np.float64)
        valid_entry = np.isfinite(entry_px) & (entry_px > 0)
        trgt = trgt[valid_entry]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt", "pt", "sl"])

    if t1 is None:
        t1 = pd.Series(pd.NaT, index=trgt.index)
    else:
        t1 = t1.reindex(trgt.index)

    events = pd.DataFrame({"t1": t1, "trgt": trgt})
    touches = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=pt_sl,
        high=high,
        low=low,
        use_intrabar_extremes=use_intrabar_extremes,
        progress=progress,
    )

    events["pt"] = touches["pt"]
    events["sl"] = touches["sl"]
    events["t1"] = touches["t1"]
    return events


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bin / label assignment (AFML §3.3)
# ─────────────────────────────────────────────────────────────────────────────

def get_bins(
    events: pd.DataFrame,
    close: pd.Series,
    label_scheme: str = "012",
) -> pd.DataFrame:
    """
    Convert event endings into class labels.

    For each event:
        ret = close[t1] / close[t0] - 1

    Then map to label:
        scheme "raw":  sign(ret)               ∈ {-1, 0, 1}
        scheme "012":  DOWN=0, FLAT=1, UP=2    (V6 classifier contract)

    A FLAT label corresponds to a vertical-barrier exit with ret = 0 (rare),
    OR to the case where pt was NaT AND sl was NaT (no horizontal touch
    within the horizon, and the realized horizon return is exactly zero —
    in practice this is sign collapse to FLAT).

    We refine this: if the FIRST-TOUCH is the vertical barrier (no horizontal
    touch occurred), label by sign(ret) of the horizon return; if a horizontal
    barrier touched first, label by which side touched (+1 if pt < sl, etc.).

    Args:
        events:       output of `get_events`.
        close:        close-price Series.
        label_scheme: "012" (V6 contract) or "raw" (-1/0/+1).

    Returns:
        DataFrame indexed by t0 with columns:
            'ret' — realised close-to-close return at first-touch (NaN if the
                    entry/exit price is invalid → unlabelable).
            'bin' — class label as FLOAT, NaN for unlabelable events.  Kept
                    float (not int8) so NaN survives; `triple_barrier_pipeline`
                    drops the NaN rows then casts to int.
    """
    if events.empty:
        return pd.DataFrame(columns=["ret", "bin"])

    events_ = events.dropna(subset=["t1"]).copy()
    p_t0 = np.asarray(close.reindex(events_.index).values, dtype=np.float64)
    p_t1 = np.asarray(close.reindex(events_["t1"].values).values, dtype=np.float64)

    # RUTHLESS price-validity gate: a zero/non-finite entry (p_t0) or exit (p_t1)
    # makes the event UNLABELABLE. Compute the return only where BOTH legs are
    # finite & positive; elsewhere → NaN (NOT a synthetic 0.0 FLAT). The NaN
    # rows are excised entirely by triple_barrier_pipeline so no dirty proxy
    # label ever reaches the model.
    valid_px = np.isfinite(p_t0) & (p_t0 > 0) & np.isfinite(p_t1) & (p_t1 > 0)
    ret = np.full(len(p_t0), np.nan, dtype=np.float64)
    np.divide(p_t1, p_t0, out=ret, where=valid_px)
    ret = np.where(valid_px, ret - 1.0, np.nan)

    # If a horizontal barrier was hit first, force the bin to its sign;
    # otherwise let the realised return's sign decide (vertical exit).
    pt_first = events_["pt"].notna() & (
        events_["sl"].isna() | (events_["pt"] < events_["sl"])
    )
    sl_first = events_["sl"].notna() & (
        events_["pt"].isna() | (events_["sl"] < events_["pt"])
    )

    raw_bin = np.zeros(len(events_), dtype=np.float64)
    raw_bin[pt_first.values] = 1.0
    raw_bin[sl_first.values] = -1.0
    # Where neither barrier hit, take the vertical-exit sign.
    neither = ~(pt_first.values | sl_first.values)
    raw_bin[neither] = np.sign(ret[neither])
    # RUTHLESS: invalid-price events are unlabelable → NaN bin (never FLAT).
    raw_bin[~valid_px] = np.nan

    out = pd.DataFrame({"ret": ret, "bin": raw_bin}, index=events_.index)

    # Keep 'bin' FLOAT so NaN (unlabelable) survives to the pipeline's drop.
    # Mapping: "012" → -1/0/+1 become 0/1/2 ; "raw" keeps -1/0/+1. NaN→NaN.
    if label_scheme == "012":
        out["bin"] = out["bin"] + 1.0
    elif label_scheme == "raw":
        pass
    else:
        raise ValueError(f"label_scheme must be '012' or 'raw', got {label_scheme!r}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Concurrent event count (AFML §4.1) — VECTORISED
# ─────────────────────────────────────────────────────────────────────────────

def get_num_co_events(close_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """
    For each bar, count the number of events whose lifespan [t0, t1] covers it.

    Vectorised O(N_events + N_bars) implementation:
        • For each event i, mark +1 at position(t0_i) and −1 at position(t1_i + 1).
        • Cumulative sum across bars gives the active count at each bar.

    Two labels that share the same time span carry redundant information; this
    function provides the denominator for AFML's uniqueness-based weights.

    Args:
        close_index: full close-price DatetimeIndex (covers all events).
        t1:          Series of first-touch end dates, indexed by t0 (event start).

    Returns:
        pandas Series of concurrent-event counts, indexed by `close_index`.
        At any bar t,  count[t]  =  #{ i : t0_i ≤ t ≤ t1_i }.
    """
    if t1.empty:
        return pd.Series(0, index=close_index, dtype=np.int64)

    pos_t0 = close_index.searchsorted(t1.index.values)
    pos_t1 = close_index.searchsorted(t1.values) + 1
    pos_t1 = np.clip(pos_t1, 0, len(close_index))

    delta = np.zeros(len(close_index) + 1, dtype=np.int64)
    np.add.at(delta, pos_t0, 1)
    np.add.at(delta, pos_t1, -1)

    count = np.cumsum(delta[:-1])
    return pd.Series(count, index=close_index, name="num_co_events")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Average uniqueness (AFML §4.2)
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_tw(
    t1: pd.Series,
    num_co_events: pd.Series,
) -> pd.Series:
    """
    Compute the average uniqueness of each label over its lifespan.

    For an event i with span [t0_i, t1_i]:

        u_i = mean_{t ∈ [t0_i, t1_i]} ( 1 / c_t )

    where c_t = number of concurrent events at bar t.

    u_i ∈ (0, 1]:
        • 1.0  → bar-by-bar the event has no overlap (fully unique label).
        • near 0 → the event is one of many concurrent labels covering the
                    same bars (highly redundant).

    Compared to `get_sample_weights`, this ignores return magnitude and weighs
    purely by overlap.  Use this if you want each LABEL (not each return) to
    have equal information content.

    Args:
        t1:            first-touch end dates, indexed by t0 (event start).
        num_co_events: output of `get_num_co_events`.

    Returns:
        Series of average-uniqueness weights u_i ∈ (0, 1], indexed by t0.
    """
    if t1.empty:
        return pd.Series(dtype=np.float64)

    inv_co = (1.0 / num_co_events.replace(0, np.nan)).fillna(0.0)
    cum = inv_co.cumsum()
    idx = num_co_events.index

    pos_t0 = idx.searchsorted(t1.index.values)
    pos_t1 = idx.searchsorted(t1.values)
    pos_t1 = np.clip(pos_t1, 0, len(idx) - 1)

    # Inclusive sum over [t0, t1] using cumsum trick:
    #   sum[t0..t1] = cum[t1] − cum[t0 − 1]
    prev = np.where(pos_t0 > 0, pos_t0 - 1, 0)
    raw_sum = cum.values[pos_t1] - np.where(pos_t0 > 0, cum.values[prev], 0.0)
    span = (pos_t1 - pos_t0 + 1).clip(min=1)
    uniqueness = raw_sum / span
    return pd.Series(uniqueness, index=t1.index, name="tw")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Return-attribution sample weights (AFML §4.10) — ★ CRITICAL ★
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_weights(
    events: pd.DataFrame,
    close: pd.Series,
    num_co_events: pd.Series | None = None,
) -> pd.Series:
    """
    Return-attribution sample weights for overlapping labels (AFML §4.10).

    The institutional dividing line
    ────────────────────────────────
    Naïvely training an LSTM with equal sample weights on overlapping triple-
    barrier events double-counts the same price movement N times (where N is
    the average concurrent-event count).  The cross-entropy loss inflates its
    effective sample size and the model's degrees-of-freedom check (BIC, DSR)
    collapses to nonsense.

    AFML's solution: each event's weight is the SUM of per-bar log-returns
    *attributed* to that event:

                       t1_i
        w_i  =  |  Σ    log(p_t / p_{t-1})  /  c_t   |
                       t = t0_i

    Interpretation:
        • Each bar's log-return is split among all events that span it (c_t
          concurrent labels) — each gets 1/c_t of the move.
        • An event's weight is the magnitude of the price action it 'owns'.
        • Highly overlapping events get small weights; truly unique labels
          (events spanning calm periods with no other active labels) keep
          full weight.

    Properties:
        • The total weight Σ_i w_i = Σ_t |log-return_t| (no information is
          lost; it is redistributed across labels).
        • Use as `sample_weight=` in any framework that supports it
          (sklearn, LightGBM, XGBoost) or as a multiplier in the PyTorch loss
          (`loss = (w * ce_loss).sum() / w.sum()`).

    Vectorised O(N_events + N_bars):
        • Compute per-bar log-returns.
        • Divide by c_t → per-bar attribution score.
        • Cumulative sum → cum_attr.
        • Each event's signed weight = cum_attr[t1] − cum_attr[t0 − 1].
        • Take absolute value (AFML uses magnitude; sign is meaningless for
          loss weighting).

    Args:
        events:         output of `get_events`; must contain 't1'.
        close:          close-price Series, indexed by date, covering all events.
        num_co_events:  pre-computed concurrent count; if None, computed here.

    Returns:
        Series of |return-attribution| weights w_i ≥ 0, indexed by t0.
        Normalise to mean=1 if your loss expects that scale.
    """
    if events.empty:
        return pd.Series(dtype=np.float64)

    events_ = events.dropna(subset=["t1"])
    if num_co_events is None:
        num_co_events = get_num_co_events(close.index, events_["t1"])

    # Guard against dirty data: log of a zero/negative price ratio is −inf/NaN
    # and would poison the cumulative attribution. Mask non-positive prices,
    # then strip any residual inf before the cumsum.
    safe_close = close.where(close > 0)
    log_ret = (
        np.log(safe_close / safe_close.shift(1))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Per-bar attribution score: r_t / c_t.  Avoid 0/0 by treating c_t==0 → 0.
    co_safe = num_co_events.reindex(close.index).fillna(0).replace(0, np.nan)
    attribution = (log_ret / co_safe).fillna(0.0)
    cum_attr = attribution.cumsum()

    idx = close.index
    pos_t0 = idx.searchsorted(events_.index.values)
    pos_t1 = idx.searchsorted(events_["t1"].values)
    pos_t1 = np.clip(pos_t1, 0, len(idx) - 1)
    prev = np.where(pos_t0 > 0, pos_t0 - 1, 0)

    signed_w = cum_attr.values[pos_t1] - np.where(
        pos_t0 > 0, cum_attr.values[prev], 0.0
    )
    w = np.abs(signed_w)
    return pd.Series(w, index=events_.index, name="w")


# ─────────────────────────────────────────────────────────────────────────────
# 9. End-to-end pipeline — Polars panel → labels + weights
# ─────────────────────────────────────────────────────────────────────────────

def triple_barrier_pipeline(
    panel_df: pl.DataFrame | pd.DataFrame,
    cfg: TripleBarrierConfig | None = None,
    *,
    close_col: str = "close",
    high_col: str | None = "high",
    low_col: str | None = "low",
    ticker_col: str = "ticker",
    date_col: str = "date",
    normalize_weights: bool = True,
    progress: bool = True,
    inner_progress: bool = False,
) -> pl.DataFrame:
    """
    Run the full triple-barrier + sample-weighting pipeline over a panel.

    For each ticker independently:
        1. Sort by date.
        2. Compute σ_t = `get_daily_vol(close, vol_span)`.
        3. Define t_events = every date with finite σ_t (post warm-up).
        4. Build vertical barriers t1 = t0 + horizon bars.
        5. Compute first-touch events via `get_events`.
        6. Assign labels via `get_bins`.
        7. Compute concurrent counts via `get_num_co_events`.
        8. Compute return-attribution sample weights via `get_sample_weights`.

    Across tickers, sample weights are normalised so that:
        mean(w_i) = 1.0     (within the entire panel)
    when `normalize_weights=True`, so a downstream loss with `w` as a
    multiplier scales identically to the unweighted baseline.

    Args:
        panel_df:         Polars or pandas DataFrame with OHLC + ticker + date.
        cfg:              TripleBarrierConfig; defaults applied if None.
        close_col,
        high_col,
        low_col:          Column names for OHLC.  high/low may be None to
                          force close-only path.
        ticker_col,
        date_col:         Panel keys.
        normalize_weights: Rescale weights to mean 1 across the panel.

    Returns:
        Polars DataFrame with one row per (ticker, t0):
            ticker        — equity symbol
            t0            — event start (= prediction target date in Phase 1)
            t1            — first-touch end date
            trgt          — σ_t0 (vol target = barrier width unit)
            ret           — realised close-to-close return at t1
            bin           — class label (per cfg.label_scheme)
            num_co_events — average concurrent labels over [t0, t1]
            uniqueness    — AFML 4.2 average uniqueness u_i ∈ (0, 1]
            w             — AFML 4.10 return-attribution sample weight
                            (multiply your per-sample loss by this)
    """
    cfg = cfg or TripleBarrierConfig()
    if isinstance(panel_df, pl.DataFrame):
        pdf = panel_df.to_pandas()
    else:
        pdf = panel_df.copy()

    needed = {close_col, ticker_col, date_col}
    if not needed.issubset(pdf.columns):
        raise ValueError(f"triple_barrier_pipeline: missing columns {needed - set(pdf.columns)}")

    has_ohlc = (
        cfg.use_intrabar_extremes
        and high_col is not None
        and low_col is not None
        and high_col in pdf.columns
        and low_col in pdf.columns
    )

    pdf[date_col] = pd.to_datetime(pdf[date_col])
    parts: list[pd.DataFrame] = []
    skipped = 0

    # ── DIAGNOSTICS ──────────────────────────────────────────────────────────
    # This pipeline is SINGLE-THREADED: a plain pandas groupby loop, no joblib /
    # multiprocessing.Pool / threads. There is therefore no parallel deadlock to
    # hunt — a "hang" is a slow loop, and the dominant cost is the per-event
    # `.iterrows()` scan inside apply_pt_sl_on_t1 (see its PERF NOTE). The tqdm
    # bar below + per-phase timing pinpoint exactly where wall-clock goes.
    groups = list(pdf.groupby(ticker_col, sort=False))
    n_groups = len(groups)
    LOGGER.info("Phase 2: triple_barrier_pipeline | %d tickers | SINGLE-THREADED "
                "(no joblib/mp.Pool) | inner_progress=%s", n_groups, inner_progress)

    timings = {"daily_vol": 0.0, "events": 0.0, "bins": 0.0, "weights": 0.0}
    iterator = (
        tqdm(groups, total=n_groups, desc="Phase 2 triple-barrier",
             unit="ticker", mininterval=0.5)
        if progress else groups
    )

    for ticker, g in iterator:
        g = g.sort_values(date_col).drop_duplicates(subset=[date_col]).set_index(date_col)

        if len(g) < cfg.vol_span + cfg.horizon + 5:
            skipped += 1
            continue

        close = g[close_col].astype(np.float64)
        high = g[high_col].astype(np.float64) if has_ohlc else None
        low = g[low_col].astype(np.float64) if has_ohlc else None

        # --- 1+2. Vol target ---
        LOGGER.debug("[%s] get_daily_vol …", ticker)
        _t = time.perf_counter()
        sigma = get_daily_vol(close, span=cfg.vol_span)
        timings["daily_vol"] += time.perf_counter() - _t
        t_events_idx = sigma.dropna().index
        if len(t_events_idx) == 0:
            skipped += 1
            continue

        # --- 3. Vertical barriers ---
        t1 = get_vertical_barriers(close, t_events_idx, cfg.horizon)

        # --- 4. Events with first-touch (THE hot path: per-event iterrows) ---
        LOGGER.debug("[%s] get_events on %d candidate events …", ticker, len(t_events_idx))
        _t = time.perf_counter()
        events = get_events(
            close=close,
            t_events=t_events_idx,
            pt_sl=(cfg.pt_mult, cfg.sl_mult),
            target=sigma,
            min_ret=cfg.min_ret,
            t1=t1,
            high=high,
            low=low,
            use_intrabar_extremes=cfg.use_intrabar_extremes,
            progress=inner_progress,
        )
        timings["events"] += time.perf_counter() - _t
        if events.empty:
            skipped += 1
            continue

        # --- 5. Labels ---
        LOGGER.debug("[%s] get_bins …", ticker)
        _t = time.perf_counter()
        bins = get_bins(events, close, label_scheme=cfg.label_scheme)
        timings["bins"] += time.perf_counter() - _t

        # --- 6. AFML §4.10 concurrency + uniqueness + weights ---
        LOGGER.debug("[%s] AFML weights (co-events / uniqueness / attribution) …", ticker)
        _t = time.perf_counter()
        num_co = get_num_co_events(close.index, events["t1"])
        uniqueness = get_sample_tw(events["t1"], num_co)
        weights = get_sample_weights(events, close, num_co_events=num_co)
        timings["weights"] += time.perf_counter() - _t

        # Live per-phase cumulative timing — reveals the bottleneck in real time.
        if progress:
            iterator.set_postfix(
                tk=str(ticker), n_ev=len(events),
                vol=f"{timings['daily_vol']:.0f}s", evt=f"{timings['events']:.0f}s",
                bins=f"{timings['bins']:.0f}s", wgt=f"{timings['weights']:.0f}s",
            )

        # Average-concurrent-events-over-lifespan, useful diagnostic
        avg_co_over_lifespan = 1.0 / uniqueness.replace(0, np.nan)

        out = pd.DataFrame({
            "ticker": ticker,
            "t0": events.index,
            "t1": events["t1"].values,
            "trgt": events["trgt"].values,
            "ret": bins["ret"].reindex(events.index).values,
            "bin": bins["bin"].reindex(events.index).values,
            "num_co_events": avg_co_over_lifespan.reindex(events.index).values,
            "uniqueness": uniqueness.reindex(events.index).values,
            "w": weights.reindex(events.index).values,
        })
        parts.append(out)

    # Per-phase wall-clock totals — the headline diagnostic. Whichever number
    # dominates is your bottleneck (expect `events` to dwarf the rest on the
    # full universe due to the iterrows scan in apply_pt_sl_on_t1).
    LOGGER.info(
        "Phase 2 per-phase totals (s): daily_vol=%.1f  events=%.1f  bins=%.1f  weights=%.1f",
        timings["daily_vol"], timings["events"], timings["bins"], timings["weights"],
    )

    if not parts:
        raise ValueError(
            f"triple_barrier_pipeline: zero ticker had usable history "
            f"(need ≥ vol_span+horizon+5 = {cfg.vol_span + cfg.horizon + 5} bars). "
            f"Skipped={skipped}."
        )

    result = pd.concat(parts, ignore_index=True)

    # ── RUTHLESS excision: drop unlabelable events (NaN ret/bin from invalid
    # entry/exit prices) BEFORE anything else. They never enter the training set
    # as synthetic FLAT labels. Done before weight normalisation so the mean-1
    # scaling is computed on the clean set only.
    n_pre = len(result)
    result = result.dropna(subset=["ret", "bin"]).reset_index(drop=True)
    n_dropped = n_pre - len(result)
    if n_dropped:
        LOGGER.info("triple_barrier_pipeline: excised %d unlabelable events "
                    "(invalid entry/exit price).", n_dropped)
    if result.empty:
        raise ValueError(
            "triple_barrier_pipeline: every event was unlabelable after the "
            "price-validity drop — check the OHLCV source for dirty data.")
    # NaN gone → safe to lock the label dtype to integer.
    result["bin"] = result["bin"].astype(np.int64)

    if normalize_weights and result["w"].notna().any():
        mean_w = float(result["w"].mean())
        if mean_w > 0:
            result["w"] = result["w"] / mean_w

    LOGGER.info(
        "triple_barrier_pipeline | events=%d  tickers=%d  skipped=%d  "
        "label_dist=%s  mean_w=%.3f  mean_uniqueness=%.3f",
        len(result),
        result["ticker"].nunique(),
        skipped,
        result["bin"].value_counts().to_dict(),
        float(result["w"].mean()),
        float(result["uniqueness"].mean()),
    )

    # Polars output — easier to join with Phase 1 tensor metadata.
    return pl.from_pandas(result)
