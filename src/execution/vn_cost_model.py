"""
src/execution/vn_cost_model.py — Quant Engine V2.0, Phase 6

Realistic VN-microstructure execution and cost model.

╔══════════════════════════════════════════════════════════════════════════════╗
║  Why this replaces the flat 0.8% cost                                        ║
║                                                                              ║
║  Phases 1–5 give us pristine bitemporal data and statistically-honest        ║
║  validation, but a flat 0.8% round-trip cost lets the model write checks the ║
║  Vietnamese market will refuse to cash.  Real HOSE/HNX/UPCOM execution kills ║
║  a strategy by FOUR mechanisms not captured by a flat rate:                  ║
║                                                                              ║
║   1. Asymmetric tax — 0.1% TRANSFER TAX is charged on SELL leg only.         ║
║      A flat round-trip double-counts on buys, undercounts churn pain.       ║
║                                                                              ║
║   2. Square-root market impact — Q/ADV > 1% already shows in fills; > 10%    ║
║      explodes via the Almgren–Chriss-style sqrt(Q/ADV) law.  VN small-caps   ║
║      (which low-price filters force the model toward) are especially brittle.║
║                                                                              ║
║   3. Price-band walls (Trần/Sàn) — HOSE ±7%, HNX ±10%, UPCOM ±15%.  When     ║
║      a stock gaps to the ceiling on the day our model fires BUY, the order  ║
║      is "Trắng bên bán" (white sell-side) — no offer exists; the fill is    ║
║      flatly rejected.  A flat-cost backtest cheerfully books the +7% as a   ║
║      gain we could never have captured.                                     ║
║                                                                              ║
║   4. Lot-size truncation — HOSE/HNX/UPCOM trade in lots of 100 shares.       ║
║      A model that wants to buy 250 shares fills 200; the residual 50        ║
║      either dies or pays the odd-lot premium.  Backtests that allow         ║
║      fractional/odd-lot fills overstate hit rate at small notional.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

VN-rules cheat-sheet
────────────────────
   Exchange  Band/day   Lot     Tick (VND)                       Sell tax
   ──────── ─────────── ─────── ──────────────────────────────── ─────────
   HOSE      ±7%         100     10 (<10k), 50 (<50k), 100 (≥50k)  0.10%
   HNX       ±10%        100     100                                0.10%
   UPCOM     ±15%        100     100                                0.10%

   Brokerage 0.15% per side, VAT 10% on brokerage.  Reference price
   defines the price band — typically prior close (giá tham chiếu).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
import polars as pl

LOGGER = logging.getLogger("vn_cost_model")


# ─────────────────────────────────────────────────────────────────────────────
# Static VN rules
# ─────────────────────────────────────────────────────────────────────────────

class Exchange(str, Enum):
    HOSE = "HOSE"
    HNX = "HNX"
    UPCOM = "UPCOM"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ParticipationPolicy(str, Enum):
    REJECT = "reject"
    """Q/ADV > max → reject the entire order (most conservative)."""
    PENALIZE = "penalize"
    """Q/ADV > max → exponential extra slippage; fill at the worse price."""
    CAP = "cap_at_max"
    """Q/ADV > max → fill only `max_participation × ADV`, the rest dies."""


class RejectionReason(str, Enum):
    NONE = "none"
    PRICE_AT_CEILING_BUY = "price_at_ceiling_buy"        # Trắng bên bán
    PRICE_AT_FLOOR_SELL = "price_at_floor_sell"          # Trắng bên mua
    PRICE_OUTSIDE_BAND = "price_outside_band"            # Mathematically impossible
    BELOW_LOT_SIZE = "below_lot_size"                    # < 100 shares
    PARTICIPATION_REJECT = "participation_reject"        # Q/ADV > max under REJECT policy
    ZERO_VOLUME = "zero_volume"                          # ADV == 0 — illiquid; no fill
    INVALID_INPUT = "invalid_input"                      # NaN price/volume etc.
    INVENTORY_NOT_SETTLED = "inventory_not_settled"      # T+2.5 — buys not yet in account
    ATC_VOLUME_EXCEEDED = "atc_volume_exceeded"          # ATC matched volume < lot


# Per-exchange price band, lot size, default participation cap.
_EXCHANGE_RULES: dict[Exchange, dict[str, float | int]] = {
    Exchange.HOSE:  {"band": 0.07,  "lot": 100, "default_max_participation": 0.10},
    Exchange.HNX:   {"band": 0.10,  "lot": 100, "default_max_participation": 0.10},
    Exchange.UPCOM: {"band": 0.15,  "lot": 100, "default_max_participation": 0.08},
}


# ── VN tick schedule (Bước giá) ──────────────────────────────────────────────
# HOSE — three-tier:
#     price < 10,000           → 10  VND
#     10,000 ≤ price < 50,000  → 50  VND
#     price ≥ 50,000           → 100 VND
# HNX, UPCOM — flat 100 VND across all price levels.
#
# Codified as a tuple of (upper_exclusive, tick) so adding tiers in the future
# (e.g., a sub-1k bracket) is a one-line edit, no logic change.
_HOSE_TICK_TIERS: tuple[tuple[float, int], ...] = (
    (10_000.0, 10),
    (50_000.0, 50),
    (float("inf"), 100),
)


def tick_size_vnd(price: float, exchange: Exchange) -> int:
    """
    VN tick size (Bước giá) in VND.

    HOSE applies a three-tier schedule based on the price level.
    HNX and UPCOM use a flat 100 VND tick across the entire range.
    """
    if exchange == Exchange.HOSE:
        for upper, tick in _HOSE_TICK_TIERS:
            if price < upper:
                return tick
        return 100  # unreachable; sentinel kept for static-analysis safety
    return 100


# ── VN settlement calendar (T+2.5) ────────────────────────────────────────────
# Trade executes at T+0 (any time during the session).
# Cash & securities settle on the SECOND business day after T+0 at 13:00 ICT.
# Practical consequence:
#   Buy on Mon  → shares usable from Wed 13:00 onwards.
#   Buy on Fri  → shares usable from Tue 13:00 onwards (Sat/Sun skipped).
# Selling before settlement is FORBIDDEN by VSD rules; brokers reject it pre-trade.

SETTLEMENT_BUSINESS_DAYS: int = 2
SETTLEMENT_TIME_OF_DAY: dtime = dtime(13, 0)  # 13:00 ICT


def add_business_days(start: date, n: int) -> date:
    """Add `n` business days to `start`, skipping Sat/Sun (VN VSD calendar)."""
    cur = start
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:    # 0..4 = Mon..Fri
            added += 1
    return cur


def settlement_datetime(trade_date: date) -> datetime:
    """
    The exact instant at which a T+0 buy becomes sellable.

        trade_date + 2 business days,  at 13:00 ICT.
    """
    return datetime.combine(
        add_business_days(trade_date, SETTLEMENT_BUSINESS_DAYS),
        SETTLEMENT_TIME_OF_DAY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeeSchedule:
    """
    Per-side brokerage / sell-side transfer tax / VAT on brokerage.

    Defaults are mid-of-market retail rates (~15bps brokerage, 10bps sell
    transfer tax, 10% VAT on the brokerage fee).  Institutional rates can be
    materially lower; pass a custom FeeSchedule for production calibration.
    """
    brokerage_per_side: float = 0.0015
    sell_transfer_tax: float = 0.0010
    vat_on_brokerage: float = 0.10       # 10% of brokerage_fee

    def buy_fee_pct(self) -> float:
        """Total proportional cost on a BUY leg (brokerage + VAT-on-brokerage)."""
        return self.brokerage_per_side * (1.0 + self.vat_on_brokerage)

    def sell_fee_pct(self) -> float:
        """Total proportional cost on a SELL leg (brokerage + VAT + transfer tax)."""
        return self.brokerage_per_side * (1.0 + self.vat_on_brokerage) + self.sell_transfer_tax

    def round_trip_pct(self) -> float:
        """Sum of buy + sell legs as a fraction of notional (asymmetric tax INCLUDED)."""
        return self.buy_fee_pct() + self.sell_fee_pct()


@dataclass(frozen=True)
class SlippageModel:
    """
    Square-root market-impact model (Almgren–Chriss style).

        impact_pct = α · σ_daily · sqrt(Q / ADV)

    where:
        α          impact coefficient.  Default 1.0 — aggressive for VN.
                   Tier-1 US liquid: ~0.3.  VN large-caps: ~0.6.  VN small-cap: 1.0+.
        σ_daily    same units as price returns (e.g. 0.025 = 2.5% daily vol).
        Q          our trade quantity (shares).
        ADV        daily volume reference (shares).

    Participation-rate excess handling
    ──────────────────────────────────
    When Q/ADV > `max_participation`, behaviour depends on `policy`:
        REJECT    →  the order is killed; no fill, no cost booked.
        PENALIZE  →  multiplicative excess penalty on impact:
                       penalty = (1 + excess/max_participation) ** excess_exponent
                       excess  = max(0, Q/ADV − max_participation)
                       so at Q/ADV = 2·max → penalty = 2^exp; at 3·max → 3^exp.
        CAP       →  Q is hard-capped to max·ADV, the rest is discarded.
    """
    alpha: float = 1.0
    max_participation: float = 0.10
    excess_exponent: float = 2.0
    policy: ParticipationPolicy = ParticipationPolicy.PENALIZE

    def impact_pct(
        self,
        quantity: float,
        adv: float,
        daily_vol: float,
    ) -> tuple[float, float]:
        """
        Returns (impact_pct, participation_pct).

        Caller decides what to do with participation_pct relative to
        `max_participation`; this function only returns the raw + penalised
        proportional cost.
        """
        if adv <= 0:
            return float("inf"), float("inf")
        participation = float(quantity) / float(adv)
        base = self.alpha * daily_vol * math.sqrt(max(participation, 0.0))

        if participation <= self.max_participation:
            return base, participation

        # Excess participation triggers exponential penalty.
        excess_ratio = (participation - self.max_participation) / self.max_participation
        penalty = (1.0 + excess_ratio) ** self.excess_exponent
        return base * penalty, participation


@dataclass(frozen=True)
class ExecutionConfig:
    """
    End-to-end execution config = fees + slippage + per-exchange override knobs.
    """
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    slippage: SlippageModel = field(default_factory=SlippageModel)

    # Aggressive-fill mode: assume the model's target_price is the close.
    # If False, simulate at next-day open (T+1 fill) — adds gap risk but is
    # typically more realistic for end-of-day signals.
    allow_same_bar_fill: bool = True

    # If True, round impacted fill price to the legal VN tick grid.
    enforce_tick: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Order & Fill
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Order:
    """
    A single order intent.

    `target_price`      the model's intended fill price (e.g. today's close).
    `reference_price`   the prior session close — defines the ±band today.
    `daily_volume`      ADV or today's expected volume (shares).
    `daily_volatility`  daily σ of returns (e.g. 0.025) for impact scaling.

    VN-specific:
    `timestamp`         when the order is fired (datetime in ICT).  Required
                        when an `InventoryTracker` is passed to `simulate()`
                        because the T+2.5 settlement check needs the exact
                        instant (Wed 09:00 vs Wed 13:30 give opposite answers).
    `is_atc`            True ⇒ At-The-Close auction order.  No graduated
                        slippage; fills at the clearing price (= target_price)
                        up to `atc_volume`; the excess is rejected.
    `atc_volume`        ATC-session matched volume on the bar.  Only consulted
                        when `is_atc` is True.  None ⇒ no cap (use with care).
    """
    ticker: str
    side: OrderSide
    quantity: int
    target_price: float
    reference_price: float
    daily_volume: float
    daily_volatility: float
    exchange: Exchange
    timestamp: datetime | None = None
    is_atc: bool = False
    atc_volume: float | None = None


@dataclass(frozen=True)
class Fill:
    """
    The execution outcome.  Rejections set is_filled=False and rejection_reason ≠ NONE.
    """
    order: Order
    filled_quantity: int
    filled_price: float
    gross_notional: float
    brokerage_paid: float
    tax_paid: float
    vat_paid: float
    slippage_cost: float
    participation_pct: float
    rejection_reason: RejectionReason
    is_filled: bool

    @property
    def total_explicit_cost(self) -> float:
        """Brokerage + VAT + transfer tax (cash leaving the door)."""
        return self.brokerage_paid + self.tax_paid + self.vat_paid

    @property
    def total_cost(self) -> float:
        """Explicit cost + price impact (slippage)."""
        return self.total_explicit_cost + self.slippage_cost

    @property
    def cost_pct(self) -> float:
        """Total cost as a fraction of intended notional (target_price × intended_qty)."""
        intended = self.order.target_price * self.order.quantity
        return self.total_cost / intended if intended > 0 else 0.0

    @property
    def signed_cash_flow(self) -> float:
        """
        Cash leaving (BUY: negative) or entering (SELL: positive) the book.
        Includes all costs.  Returns 0 for a rejection.
        """
        if not self.is_filled:
            return 0.0
        if self.order.side == OrderSide.BUY:
            return -(self.gross_notional + self.total_explicit_cost)
        return self.gross_notional - self.total_explicit_cost


# ─────────────────────────────────────────────────────────────────────────────
# Corporate actions — Phase 6.5
# ─────────────────────────────────────────────────────────────────────────────

class CorporateActionType(str, Enum):
    CASH_DIVIDEND = "cash_dividend"     # VND per share credited to cash on ex-date
    STOCK_DIVIDEND = "stock_dividend"   # bonus shares; treated as a split factor
    SPLIT = "split"                     # share multiplier; price divides by factor


@dataclass(frozen=True)
class CorporateActionEvent:
    """
    A single corporate action applied on its ex-date.

    cash_dividend:  cash_per_share VND credited per net share held.
    split / stock_dividend:  every share becomes `split_factor` shares.
        A 1:1 bonus issue (cổ phiếu thưởng 100%) ⇒ split_factor = 2.0.
        A 2:1 split                                 ⇒ split_factor = 2.0.
        A 3-for-2 (50% bonus)                       ⇒ split_factor = 1.5.
    """
    ticker: str
    ex_date: date
    action_type: CorporateActionType
    cash_per_share: float = 0.0
    split_factor: float = 1.0


def _parse_ca_date(value) -> date:
    """Tolerant ex-date parser for the corporate_actions adapter."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value)[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised ex_date: {value!r}")


# ─────────────────────────────────────────────────────────────────────────────
# T+2.5 settlement inventory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PendingLot:
    """An executed buy lot waiting for VSD settlement."""
    trade_date: date
    settles_at: datetime         # exact instant the shares become sellable
    quantity: int
    fill_price: float

    def is_settled_at(self, t: datetime) -> bool:
        """True iff the lot has cleared by time `t`."""
        return t >= self.settles_at


class InventoryTracker:
    """
    Per-ticker T+2.5 settlement queue.

    Models the VN VSD reality: a buy fill on T+0 ENTERS a pending queue and is
    NOT sellable until the second business day at 13:00 ICT.  The execution
    engine consults this tracker on EVERY sell-side order; if available
    settled inventory < requested quantity at the order timestamp, the order
    is rejected with `INVENTORY_NOT_SETTLED`.

    Same-day round-trip
    ───────────────────
    A T+0 buy at 09:30 ICT followed by a sell intent at 14:00 ICT on the SAME
    day is rejected — the shares have not arrived.  This is the textbook VN
    case that flat-cost backtests silently allow, inflating intraday alpha.

    Concurrency
    ───────────
    Not async-safe by itself; if used across coroutines, wrap calls with an
    asyncio.Lock at the caller.  Backtests are single-threaded so no locking
    is needed there.
    """

    def __init__(self) -> None:
        self._pending: dict[str, list[PendingLot]] = {}
        self._sold: dict[str, list[tuple[datetime, int]]] = {}
        # Phase 6.5: realised cash from dividends (and any other cash credits).
        # This is what NEUTRALISES the ex-dividend price drop in the P&L.
        self.cash_balance: float = 0.0
        self._ca_log: list[dict] = []   # audit trail of applied corporate actions

    def record_buy(
        self,
        ticker: str,
        trade_date: date,
        quantity: int,
        fill_price: float,
    ) -> PendingLot:
        """
        Record a successful buy fill.  Settlement time is computed as
        T+2 business days at 13:00 ICT.
        """
        lot = PendingLot(
            trade_date=trade_date,
            settles_at=settlement_datetime(trade_date),
            quantity=quantity,
            fill_price=fill_price,
        )
        self._pending.setdefault(ticker, []).append(lot)
        return lot

    def record_sell(self, ticker: str, sell_ts: datetime, quantity: int) -> None:
        """Record a successful sell against settled inventory."""
        self._sold.setdefault(ticker, []).append((sell_ts, quantity))

    # ── Phase 6.5: corporate-action ledger ─────────────────────────────────
    def net_shares_at(self, ticker: str, t: datetime) -> int:
        """
        Net shares on the books (all lots, settled OR pending) minus shares sold
        on/before `t`.  This is the entitlement base for a dividend, and the
        quantity a split rescales.
        """
        held = sum(lot.quantity for lot in self._pending.get(ticker, []))
        sold = sum(q for ts, q in self._sold.get(ticker, []) if ts <= t)
        return max(0, held - sold)

    def apply_corporate_action(self, event: CorporateActionEvent) -> dict:
        """
        Apply ONE corporate action on its ex-date.

        cash_dividend
            cash_balance += net_shares × cash_per_share.
            This is the dividend trap fix: the ex-date close drops by ~the
            dividend, which a naive P&L books as a loss; the cash credit
            exactly offsets it so realised wealth is unchanged.

        split / stock_dividend (factor f)
            every pending lot:  quantity ×= f,  fill_price ÷= f
            every recorded sell (all pre-ex at this point in the timeline): ×= f
            ⇒ share count grows, cost basis preserved, the post-split price
              drop is purely mechanical and never hits P&L as a loss.

        Returns an audit dict (also appended to the internal ledger).
        """
        ticker = event.ticker
        # Entitlement / rescale base: net shares as of the ex-date open.
        ex_instant = datetime.combine(event.ex_date, dtime(0, 0))
        net_held = self.net_shares_at(ticker, ex_instant)

        record: dict = {
            "ticker": ticker,
            "ex_date": event.ex_date.isoformat(),
            "action_type": event.action_type.value,
            "net_held_pre": net_held,
        }

        if event.action_type == CorporateActionType.CASH_DIVIDEND:
            credited = net_held * event.cash_per_share
            self.cash_balance += credited
            record["cash_per_share"] = event.cash_per_share
            record["cash_credited"] = credited
            record["cash_balance_after"] = self.cash_balance

        elif event.action_type in (CorporateActionType.SPLIT,
                                    CorporateActionType.STOCK_DIVIDEND):
            f = event.split_factor
            if f <= 0:
                raise ValueError(f"split_factor must be > 0, got {f}")
            for lot in self._pending.get(ticker, []):
                lot.quantity = int(round(lot.quantity * f))
                lot.fill_price = lot.fill_price / f
            # Rescale historical sells so available_at()'s (held − sold) stays
            # consistent in post-split units.  At ex-date all recorded sells are
            # pre-split, so scaling them all is correct.
            if ticker in self._sold:
                self._sold[ticker] = [
                    (ts, int(round(q * f))) for ts, q in self._sold[ticker]
                ]
            record["split_factor"] = f
            record["shares_after"] = sum(
                lot.quantity for lot in self._pending.get(ticker, [])
            )
        else:
            raise ValueError(f"unknown action_type: {event.action_type}")

        self._ca_log.append(record)
        LOGGER.debug("corporate action applied: %s", record)
        return record

    def ingest_corporate_actions(
        self,
        events: Sequence[CorporateActionEvent],
    ) -> list[dict]:
        """
        Apply a batch of corporate actions in ex-date order (idempotent only if
        called once — splits compound, so do NOT replay).  Returns the audit log
        of what was applied.
        """
        return [self.apply_corporate_action(ev)
                for ev in sorted(events, key=lambda e: e.ex_date)]

    @staticmethod
    def parse_corporate_actions(
        df,
        *,
        ticker_col: str = "ticker",
        date_col: str = "event_date",
        type_col: str = "action_type",
        factor_col: str = "factor",
        cash_col: str = "cash_amount",
    ) -> list[CorporateActionEvent]:
        """
        Adapter from the Phase-5 `corporate_actions` table (Polars or pandas) to
        a list of `CorporateActionEvent`.

        Mapping:
            action_type 'dividend' / 'cash_dividend' → CASH_DIVIDEND (cash_amount = VND/share)
            action_type 'split'                      → SPLIT       (factor)
            action_type 'stock_dividend' / 'bonus'   → STOCK_DIVIDEND (factor)
        Rows with unrecognised types are skipped.
        """
        rows = df.iter_rows(named=True) if hasattr(df, "iter_rows") else (
            r._asdict() if hasattr(r, "_asdict") else r
            for _, r in df.iterrows()
        )
        events: list[CorporateActionEvent] = []
        for row in rows:
            raw_type = str(row[type_col]).lower()
            ex = row[date_col]
            ex_date = ex if isinstance(ex, date) and not isinstance(ex, datetime) else (
                ex.date() if isinstance(ex, datetime) else _parse_ca_date(ex)
            )
            if raw_type in ("dividend", "cash_dividend"):
                events.append(CorporateActionEvent(
                    ticker=str(row[ticker_col]), ex_date=ex_date,
                    action_type=CorporateActionType.CASH_DIVIDEND,
                    cash_per_share=float(row.get(cash_col, 0.0) or 0.0),
                ))
            elif raw_type == "split":
                events.append(CorporateActionEvent(
                    ticker=str(row[ticker_col]), ex_date=ex_date,
                    action_type=CorporateActionType.SPLIT,
                    split_factor=float(row.get(factor_col, 1.0) or 1.0),
                ))
            elif raw_type in ("stock_dividend", "bonus"):
                events.append(CorporateActionEvent(
                    ticker=str(row[ticker_col]), ex_date=ex_date,
                    action_type=CorporateActionType.STOCK_DIVIDEND,
                    split_factor=float(row.get(factor_col, 1.0) or 1.0),
                ))
        return events

    def position_value(
        self,
        ticker: str,
        mark_price: float,
        t: datetime,
    ) -> dict:
        """
        Mark-to-market a single position, INCLUDING dividend cash.

        Returns:
            dict with net_shares, mark_price, market_value, cash_balance,
            total_wealth (= market_value + cash_balance).

        The headline invariant the dividend/​split fix guarantees:
            total_wealth is CONTINUOUS across an ex-date — the price drop is
            offset by the cash credit (dividend) or the extra shares (split).
        """
        net = self.net_shares_at(ticker, t)
        mv = net * mark_price
        return {
            "net_shares": net,
            "mark_price": mark_price,
            "market_value": mv,
            "cash_balance": self.cash_balance,
            "total_wealth": mv + self.cash_balance,
        }

    def available_at(self, ticker: str, t: datetime) -> int:
        """
        Sellable shares (settled lots minus prior sells) for `ticker` at instant `t`.

        Returned value is always non-negative; over-selling is caller error.
        """
        settled = sum(
            lot.quantity for lot in self._pending.get(ticker, [])
            if lot.is_settled_at(t)
        )
        sold_already = sum(
            q for sell_ts, q in self._sold.get(ticker, [])
            if sell_ts <= t
        )
        return max(0, settled - sold_already)

    def pending_at(self, ticker: str, t: datetime) -> int:
        """Unsettled (still in T+2.5 pipeline) shares at `t`.  For audit / UI."""
        return sum(
            lot.quantity for lot in self._pending.get(ticker, [])
            if not lot.is_settled_at(t)
        )

    def snapshot(self, t: datetime) -> dict[str, dict[str, int]]:
        """{ticker: {settled, pending}} at time `t`.  Convenience for backtest reports."""
        out: dict[str, dict[str, int]] = {}
        all_tickers = set(self._pending.keys()) | set(self._sold.keys())
        for tk in all_tickers:
            out[tk] = {
                "settled": self.available_at(tk, t),
                "pending": self.pending_at(tk, t),
            }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — testable in isolation
# ─────────────────────────────────────────────────────────────────────────────

def round_down_to_lot(quantity: int, lot_size: int = 100) -> int:
    """
    HOSE/HNX/UPCOM round-down to the nearest 100-share lot.

    Quantities below `lot_size` round to 0 — the order is below the minimum
    fillable lot and must be rejected.
    """
    if quantity < 0:
        raise ValueError(f"quantity must be non-negative, got {quantity}")
    return (quantity // lot_size) * lot_size


def price_band_bounds(
    reference_price: float,
    exchange: Exchange,
) -> tuple[float, float]:
    """
    Inclusive (floor, ceiling) the price may settle at today.

    Computed against the prior session's reference price (giá tham chiếu).
    The band IS the legal trading range — orders outside it cannot match.
    """
    band = float(_EXCHANGE_RULES[exchange]["band"])
    floor = reference_price * (1.0 - band)
    ceiling = reference_price * (1.0 + band)
    return floor, ceiling


def is_at_ceiling(price: float, reference_price: float, exchange: Exchange,
                  tolerance_vnd: float = 1.0) -> bool:
    """True if `price` is at (or above) today's ceiling within `tolerance_vnd`."""
    _, ceiling = price_band_bounds(reference_price, exchange)
    return price >= ceiling - tolerance_vnd


def is_at_floor(price: float, reference_price: float, exchange: Exchange,
                tolerance_vnd: float = 1.0) -> bool:
    """True if `price` is at (or below) today's floor within `tolerance_vnd`."""
    floor, _ = price_band_bounds(reference_price, exchange)
    return price <= floor + tolerance_vnd


def round_to_tick(price: float, exchange: Exchange, *, side: OrderSide) -> float:
    """
    Round a continuous price to the legal VN tick grid.

    For a BUY we round UP (worse for us — can't fill at a price not on the
    grid below us); for a SELL we round DOWN (worse for us — can't sell at a
    grid price above us).  This is the conservative direction for cost modelling.
    """
    tick = tick_size_vnd(price, exchange)
    if side == OrderSide.BUY:
        return math.ceil(price / tick) * tick
    return math.floor(price / tick) * tick


# ─────────────────────────────────────────────────────────────────────────────
# The cost model itself
# ─────────────────────────────────────────────────────────────────────────────

class VNCostModel:
    """
    Single-order execution simulator with VN-microstructure rules.

    Usage:
        model = VNCostModel(ExecutionConfig())
        fill = model.simulate(order)
        if fill.is_filled:
            net_pnl = ...  # use fill.signed_cash_flow downstream
        else:
            log_rejection(fill.rejection_reason)

    For backtest integration, see `apply_to_signals()` (Polars-native batch path).
    """

    def __init__(self, cfg: ExecutionConfig | None = None) -> None:
        self.cfg = cfg or ExecutionConfig()

    # ── Single-order path ────────────────────────────────────────────────
    def simulate(
        self,
        order: Order,
        *,
        inventory: InventoryTracker | None = None,
    ) -> Fill:
        """
        Apply all VN-microstructure checks, in order:
            1. Input sanity (NaN / non-positive volume / non-positive price)
            2. Lot-size rounding + minimum 100-share quantity
            3. Price-band walls — outside band, Trắng bên bán, Trắng bên mua
            4. T+2.5 settlement check for SELL legs (if `inventory` is supplied)
            5. ATC branch  — clearing-price fill, hard cap at `atc_volume`
               OR  continuous market branch — sqrt impact + participation policy
            6. Tick rounding (HOSE three-tier, HNX/UPCOM flat 100 VND)

        When `inventory` is passed, the tracker is mutated on a successful
        fill (buys queued for settlement, sells deducted at settlement time).
        Backtests should pass a single shared `InventoryTracker` so the T+2.5
        chronology is enforced across the whole signal stream.
        """
        # ── 1. Input sanity ──
        if not (math.isfinite(order.target_price) and math.isfinite(order.reference_price)
                and math.isfinite(order.daily_volume) and math.isfinite(order.daily_volatility)):
            return self._rejected(order, RejectionReason.INVALID_INPUT,
                                  participation_pct=0.0)
        if order.target_price <= 0 or order.reference_price <= 0:
            return self._rejected(order, RejectionReason.INVALID_INPUT,
                                  participation_pct=0.0)
        if order.daily_volume <= 0:
            return self._rejected(order, RejectionReason.ZERO_VOLUME,
                                  participation_pct=0.0)

        # ── 2. Lot-size rounding ──
        lot = int(_EXCHANGE_RULES[order.exchange]["lot"])
        intended_qty = round_down_to_lot(order.quantity, lot)
        if intended_qty < lot:
            return self._rejected(order, RejectionReason.BELOW_LOT_SIZE,
                                  participation_pct=0.0)

        # ── 3. Price-band walls ──
        floor, ceiling = price_band_bounds(order.reference_price, order.exchange)
        if order.target_price > ceiling + 1e-9 or order.target_price < floor - 1e-9:
            return self._rejected(order, RejectionReason.PRICE_OUTSIDE_BAND,
                                  participation_pct=0.0)

        if order.side == OrderSide.BUY and is_at_ceiling(
                order.target_price, order.reference_price, order.exchange):
            return self._rejected(order, RejectionReason.PRICE_AT_CEILING_BUY,
                                  participation_pct=intended_qty / order.daily_volume)
        if order.side == OrderSide.SELL and is_at_floor(
                order.target_price, order.reference_price, order.exchange):
            return self._rejected(order, RejectionReason.PRICE_AT_FLOOR_SELL,
                                  participation_pct=intended_qty / order.daily_volume)

        # ── 4. T+2.5 settlement check for SELLs ─────────────────────────────
        # The VSD rule: shares bought at T+0 are not deliverable until T+2 at
        # 13:00 ICT.  Selling pre-settlement is forbidden — VSD blocks the
        # broker, and any backtest that allows it is fictitious.
        if order.side == OrderSide.SELL and inventory is not None:
            if order.timestamp is None:
                # An inventory-tracked simulation REQUIRES a timestamp to know
                # which lots have settled by the order instant.
                return self._rejected(order, RejectionReason.INVALID_INPUT,
                                      participation_pct=0.0)
            settled_available = inventory.available_at(order.ticker, order.timestamp)
            if settled_available < intended_qty:
                LOGGER.debug(
                    "T+2.5 reject  %s sell %d @ %s — only %d settled (pending=%d)",
                    order.ticker, intended_qty, order.timestamp.isoformat(),
                    settled_available,
                    inventory.pending_at(order.ticker, order.timestamp),
                )
                return self._rejected(order, RejectionReason.INVENTORY_NOT_SETTLED,
                                      participation_pct=intended_qty / order.daily_volume)

        # ── 5. ATC branch  vs  continuous-market branch ─────────────────────
        if order.is_atc:
            # ATC orders cross at a SINGLE clearing price during 14:30–14:45.
            # No graduated slippage: everyone gets the same price = target_price.
            # The only constraint is the matched volume — anything above
            # `atc_volume` cannot clear and is rejected.
            if order.atc_volume is not None:
                atc_cap_raw = int(order.atc_volume)
                if intended_qty > atc_cap_raw:
                    intended_qty = round_down_to_lot(atc_cap_raw, lot)
                    if intended_qty < lot:
                        return self._rejected(order, RejectionReason.ATC_VOLUME_EXCEEDED,
                                              participation_pct=order.quantity / order.daily_volume
                                              if order.daily_volume > 0 else 0.0)
            participation = intended_qty / order.daily_volume
            # No impact; fill at the clearing (target) price.
            raw_fill = order.target_price
            impact_pct = 0.0
        else:
            # Continuous-market branch: sqrt impact + participation policy.
            max_part = self.cfg.slippage.max_participation
            participation = intended_qty / order.daily_volume

            if (self.cfg.slippage.policy == ParticipationPolicy.REJECT
                    and participation > max_part):
                return self._rejected(order, RejectionReason.PARTICIPATION_REJECT,
                                      participation_pct=participation)

            if (self.cfg.slippage.policy == ParticipationPolicy.CAP
                    and participation > max_part):
                capped_raw = int(max_part * order.daily_volume)
                intended_qty = round_down_to_lot(capped_raw, lot)
                if intended_qty < lot:
                    return self._rejected(order, RejectionReason.BELOW_LOT_SIZE,
                                          participation_pct=participation)
                # Re-check T+2.5 against the capped qty (only relevant for sells).
                if (order.side == OrderSide.SELL and inventory is not None
                        and order.timestamp is not None):
                    if inventory.available_at(order.ticker, order.timestamp) < intended_qty:
                        return self._rejected(order, RejectionReason.INVENTORY_NOT_SETTLED,
                                              participation_pct=participation)
                participation = intended_qty / order.daily_volume

            impact_pct, _ = self.cfg.slippage.impact_pct(
                quantity=intended_qty,
                adv=order.daily_volume,
                daily_vol=order.daily_volatility,
            )
            impact_sign = 1.0 if order.side == OrderSide.BUY else -1.0
            raw_fill = order.target_price * (1.0 + impact_sign * impact_pct)

        # ── 6. Clip to band, then tick rounding ─────────────────────────────
        # You literally cannot fill past the wall; the impact is truncated at
        # the band.  Conservative: assume the order DOES fill at the wall.
        raw_fill = max(floor, min(ceiling, raw_fill))

        if self.cfg.enforce_tick:
            filled_price = round_to_tick(raw_fill, order.exchange, side=order.side)
        else:
            filled_price = raw_fill

        gross_notional = filled_price * intended_qty

        brokerage = gross_notional * self.cfg.fees.brokerage_per_side
        vat = brokerage * self.cfg.fees.vat_on_brokerage
        tax = gross_notional * self.cfg.fees.sell_transfer_tax \
            if order.side == OrderSide.SELL else 0.0

        slippage_cost = abs(filled_price - order.target_price) * intended_qty

        # ── Mutate the settlement queue on a successful fill ─────────────────
        # Buys join the pending queue (un-sellable until T+2 at 13:00 ICT).
        # Sells deduct from settled inventory (recorded at the order timestamp).
        if inventory is not None and order.timestamp is not None:
            if order.side == OrderSide.BUY:
                inventory.record_buy(
                    ticker=order.ticker,
                    trade_date=order.timestamp.date(),
                    quantity=intended_qty,
                    fill_price=filled_price,
                )
            else:
                inventory.record_sell(
                    ticker=order.ticker,
                    sell_ts=order.timestamp,
                    quantity=intended_qty,
                )

        return Fill(
            order=order,
            filled_quantity=intended_qty,
            filled_price=filled_price,
            gross_notional=gross_notional,
            brokerage_paid=brokerage,
            tax_paid=tax,
            vat_paid=vat,
            slippage_cost=slippage_cost,
            participation_pct=participation,
            rejection_reason=RejectionReason.NONE,
            is_filled=True,
        )

    def _rejected(self, order: Order, reason: RejectionReason,
                  *, participation_pct: float) -> Fill:
        return Fill(
            order=order,
            filled_quantity=0,
            filled_price=0.0,
            gross_notional=0.0,
            brokerage_paid=0.0,
            tax_paid=0.0,
            vat_paid=0.0,
            slippage_cost=0.0,
            participation_pct=participation_pct,
            rejection_reason=reason,
            is_filled=False,
        )

    # ── Batch path — for backtest integration ─────────────────────────────
    def simulate_batch(self, orders: Iterable[Order]) -> list[Fill]:
        """Vectorless batch.  Cost-model logic is per-order; readability beats µ-perf."""
        return [self.simulate(o) for o in orders]

    def apply_to_signals(
        self,
        df: pl.DataFrame,
        *,
        ticker_col: str = "ticker",
        side_col: str = "side",
        quantity_col: str = "quantity",
        target_price_col: str = "target_price",
        reference_price_col: str = "reference_price",
        daily_volume_col: str = "daily_volume",
        daily_vol_col: str = "daily_volatility",
        exchange_col: str = "exchange",
    ) -> pl.DataFrame:
        """
        Take a Polars signals DataFrame and append per-row execution outcomes.

        Required columns
            ticker (str), side ('buy'|'sell'), quantity (int), target_price,
            reference_price, daily_volume, daily_volatility, exchange.

        Appended columns
            filled_quantity, filled_price, gross_notional,
            brokerage_paid, tax_paid, vat_paid, slippage_cost,
            participation_pct, rejection_reason, is_filled,
            net_return_pct  (signed cash flow / target_price·quantity)
        """
        required = [ticker_col, side_col, quantity_col, target_price_col,
                    reference_price_col, daily_volume_col, daily_vol_col, exchange_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"apply_to_signals: missing columns {missing}")

        orders: list[Order] = []
        for row in df.iter_rows(named=True):
            orders.append(Order(
                ticker=str(row[ticker_col]),
                side=OrderSide(str(row[side_col]).lower()),
                quantity=int(row[quantity_col]),
                target_price=float(row[target_price_col]),
                reference_price=float(row[reference_price_col]),
                daily_volume=float(row[daily_volume_col]),
                daily_volatility=float(row[daily_vol_col]),
                exchange=Exchange(str(row[exchange_col]).upper()),
            ))
        fills = self.simulate_batch(orders)

        out_cols: dict[str, list] = {
            "filled_quantity": [], "filled_price": [], "gross_notional": [],
            "brokerage_paid": [], "tax_paid": [], "vat_paid": [],
            "slippage_cost": [], "participation_pct": [],
            "rejection_reason": [], "is_filled": [], "net_return_pct": [],
        }
        for f in fills:
            out_cols["filled_quantity"].append(f.filled_quantity)
            out_cols["filled_price"].append(f.filled_price)
            out_cols["gross_notional"].append(f.gross_notional)
            out_cols["brokerage_paid"].append(f.brokerage_paid)
            out_cols["tax_paid"].append(f.tax_paid)
            out_cols["vat_paid"].append(f.vat_paid)
            out_cols["slippage_cost"].append(f.slippage_cost)
            out_cols["participation_pct"].append(f.participation_pct)
            out_cols["rejection_reason"].append(f.rejection_reason.value)
            out_cols["is_filled"].append(f.is_filled)

            # net_return_pct: signed cash flow / intended notional.  For a buy
            # this is a negative cost; for a sell, positive proceeds-of-1-leg.
            intended = f.order.target_price * f.order.quantity
            out_cols["net_return_pct"].append(
                f.signed_cash_flow / intended if intended > 0 else 0.0
            )

        # Explicit dtypes — Polars Series inference is brittle when the first
        # row sets the type and a later row has a coerced 0/0.0.
        dtype_map: dict[str, pl.DataType] = {
            "filled_quantity": pl.Int64,
            "filled_price": pl.Float64,
            "gross_notional": pl.Float64,
            "brokerage_paid": pl.Float64,
            "tax_paid": pl.Float64,
            "vat_paid": pl.Float64,
            "slippage_cost": pl.Float64,
            "participation_pct": pl.Float64,
            "rejection_reason": pl.Utf8,
            "is_filled": pl.Boolean,
            "net_return_pct": pl.Float64,
        }
        return df.with_columns([
            pl.Series(name=k, values=v, dtype=dtype_map[k], strict=False)
            for k, v in out_cols.items()
        ])

    # ── Phase-4 drop-in: round-trip cost (%) for plugging into legacy paths ──
    def round_trip_cost_pct(
        self,
        *,
        notional_share: float = 1.0,
        adv: float = float("inf"),
        daily_vol: float = 0.02,
        exchange: Exchange = Exchange.HOSE,
    ) -> float:
        """
        Cheap aggregate round-trip cost for plugging into the Phase-4 net
        Sharpe path that currently uses 0.8%.  Includes asymmetric fees + the
        sqrt impact term at the given participation.

        Args:
            notional_share: Q / ADV proxy used for impact (default 1.0 = ADV).
            adv:            kept symbolic — we operate on the ratio.
            daily_vol:      σ_daily for impact.
            exchange:       drives band/lot/tick but NOT fees (same across VN).

        Returns:
            Round-trip cost as a fraction of one-side notional (so multiply by
            notional to get $).  Asymmetric: buy_fee + sell_fee + 2·impact.
        """
        fee_round_trip = self.cfg.fees.round_trip_pct()
        # Two crossings → twice the one-way impact (assuming symmetric exit).
        impact_one_side, _ = self.cfg.slippage.impact_pct(
            quantity=notional_share, adv=1.0, daily_vol=daily_vol,
        )
        return fee_round_trip + 2.0 * impact_one_side


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: ladder of rejection-reason counters for backtest audit
# ─────────────────────────────────────────────────────────────────────────────

def rejection_breakdown(fills: Sequence[Fill]) -> dict[str, int]:
    """Histogram of rejection reasons across a fill list (filled rows excluded)."""
    out: dict[str, int] = {r.value: 0 for r in RejectionReason if r != RejectionReason.NONE}
    for f in fills:
        if not f.is_filled:
            out[f.rejection_reason.value] += 1
    out["filled"] = sum(1 for f in fills if f.is_filled)
    return out
