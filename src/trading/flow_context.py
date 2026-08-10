"""Live foreign-flow divergence context for the MR knife-catch display
(08-08-26 -- SSI FastConnect research; see project memory / process/
general-plans/backlog/foreign-flow-fastconnect-integration_PLAN_24-07-26.md
for the full research trail).

Real, statistically confirmed but small: T+5 mean fwd return +0.774% vs
+0.304% baseline (Welch p=0.00004), T+20 +2.656% vs +1.387% (p<0.000001),
Cohen's d~0.08, broad across 345 tickers (233 with >=10 fires). Same
"display annotation, never gates the fire decision" treatment as the
22-07-26 breadth-inflection signal (src/trading/breadth.py) -- stronger
evidence behind this one, but the effect size still doesn't clear the bar
for a standalone trigger or a model feature/retrain.

Deliberately per-ticker, unlike breadth.py's market-wide reads: only ever
called for a ticker that JUST fired the MR trigger (main.mr_score_tickers),
so the cost of one live FastConnect call per fire is negligible against
the historical ~14 fires/month base rate. This sidesteps the separate,
still-open SOURCE 1 daily-cron coverage bug (see the same backlog plan,
item 3b) entirely -- no dependency on the broken 51-ticker daily snapshot,
fetches fresh per-ticker history directly from FastConnect on demand.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)

# Calendar days of history to request -- comfortably covers the ~35
# trading days flow_features.py's rolling-20d windows want (min_samples=5
# is the hard floor, but a fuller window gives a more reliable z-score).
_LOOKBACK_DAYS = 50


def latest_foreign_room(ticker: str) -> float | None:
    """Most recent `foreign_remain_room_vol` for `ticker`, or None.

    Reads the local backfill parquet (populated by
    scripts/backfill_ssi_foreign_flow.py) rather than calling FastConnect —
    room moves slowly and a dispatch must not wait on the network. None when
    the parquet, the ticker, or the column is missing; the caller treats that
    as "unknown", never as "room available".
    """
    try:
        from src.data.foreign_flow_crawler import _DEFAULT_PARQUET  # noqa: PLC0415

        path = Path(_DEFAULT_PARQUET)
        if not path.exists():
            return None
        rows = (
            pl.scan_parquet(path)
            .filter((pl.col("ticker") == ticker.upper().strip())
                    & pl.col("foreign_remain_room_vol").is_not_null())
            .select(["date", "foreign_remain_room_vol"])
            .sort("date", descending=True)
            .limit(1)
            .collect()
        )
        if rows.is_empty():
            return None
        return float(rows.row(0)[1])
    except Exception:  # noqa: BLE001 -- informational only, never break a dispatch
        LOGGER.warning("[flow-context] foreign-room lookup failed for %s", ticker,
                       exc_info=True)
        return None


def live_flow_divergence(ticker: str) -> dict | None:
    """{divergence: bool, flow_net_scaled_adv20: float | None} for
    `ticker`'s latest bar, or None on ANY failure (missing FastConnect
    credentials, network, insufficient local OHLCV history, etc). Fail-
    open -- this NEVER blocks or alters the MR fire decision, purely
    informational, same contract as breadth.live_breadth_inflection.
    """
    try:
        from src.data.foreign_flow_crawler import fetch_fastconnect_history_for_symbol
        from src.data.price_lookup import ohlc_history
        from src.features.flow_features import build_flow_features

        symbol = ticker.upper().strip()
        to_date = date.today()
        from_date = to_date - timedelta(days=_LOOKBACK_DAYS)

        flow = fetch_fastconnect_history_for_symbol(symbol, from_date, to_date)
        if flow.is_empty():
            return None

        bars = ohlc_history(symbol, n=40)
        if not bars:
            return None
        price = pl.DataFrame(
            bars, schema=["date", "open", "high", "low", "close", "volume"], orient="row",
        ).with_columns(pl.lit(symbol).alias("ticker"))

        joined = (
            flow.select(["ticker", "date", "foreign_buy_val", "foreign_sell_val", "foreign_net_val",
                        "prop_buy_val", "prop_sell_val", "prop_net_val"])
            .join(price.select(["ticker", "date", "close", "volume"]), on=["ticker", "date"], how="inner")
        )
        if joined.is_empty():
            return None

        feat = build_flow_features(joined).sort("date")
        latest = feat.tail(1)
        if latest.is_empty():
            return None

        row = latest.row(0, named=True)
        divergence = row.get("flow_knife_catch_divergence")
        if divergence is None:
            return None
        return {
            "divergence": bool(divergence == 1.0),
            "flow_net_scaled_adv20": row.get("flow_net_scaled_adv20"),
        }
    except Exception:  # noqa: BLE001 -- fail-open, never break the caller
        LOGGER.warning("[flow-context] live divergence check failed for %s", ticker, exc_info=True)
        return None
