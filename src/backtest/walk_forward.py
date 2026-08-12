"""
src/backtest/walk_forward.py — Quant Engine V2.0, Phase 8 (capstone)

Walk-forward paper-trading harness.  Wires Phases 1–7 into one strictly
chronological daily loop — the digital twin of running the fund day by day.

╔══════════════════════════════════════════════════════════════════════════════╗
║  The daily loop (no future peering)                                          ║
║                                                                              ║
║  For each trading day D in chronological order:                              ║
║                                                                              ║
║    1. MORNING (pre-market)                                                   ║
║       Apply every CorporateActionEvent with ex_date == D to the shared       ║
║       InventoryTracker: cash dividends credit cash, splits rescale the       ║
║       pending/settled share counts (Phase 6.5).  Done BEFORE marking so the  ║
║       ex-date price drop is already neutralised.                             ║
║                                                                              ║
║    2. INFERENCE (the Oracle)                                                 ║
║       Build the (n_eligible, seq_len, n_features) tensor from data STRICTLY  ║
║       BEFORE D (≤ D−1), incorporating the Phase 1.5 anti-FOMO features, and  ║
║       run the Phase 3 QuantLSTM → P(UP) per ticker.  The cutoff at D−1 is    ║
║       the leak firewall: today's bar is never visible to today's signal.    ║
║                                                                              ║
║    3. RISK & ALLOCATION (the PM)                                             ║
║       Ledoit-Wolf covariance over trailing returns (≤ D−1) → covariance-     ║
║       coupled fractional Kelly → constrained mean-variance (long-only,       ║
║       per-ticker + sector caps, vol target).  Output: target weights.        ║
║                                                                              ║
║    4. EXECUTION (the Trader)                                                 ║
║       Rebalance to target via the Phase 6 VNCostModel as ATC orders using    ║
║       D's OHLCV.  T+2.5 settlement, lot-size, price-band and ATC-volume       ║
║       rejections are LOGGED and RESPECTED — never silently filled.            ║
║                                                                              ║
║    5. CLOSING (the Accountant)                                               ║
║       Mark-to-market net shares at D's close + cash → NAV.  Daily Net PnL.   ║
║                                                                              ║
║  PRICE CONVENTION                                                            ║
║    The harness trades & marks on the panel's `close` column.  Use RAW        ║
║    (unadjusted) prices from the Phase-5 bitemporal store's `close_raw` and   ║
║    pass `corporate_actions` so the Phase-6.5 ledger neutralises ex-dates.    ║
║    (Do NOT pass corporate_actions if the panel is already back-adjusted —    ║
║    that would double-count.)  The price-band reference on an ex-date is      ║
║    auto-adjusted (prior close − dividend, or ÷ split factor) so the band     ║
║    check matches the exchange's reset reference.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from src.execution.vn_cost_model import (
    CorporateActionEvent,
    CorporateActionType,
    Exchange,
    ExecutionConfig,
    InventoryTracker,
    Order,
    OrderSide,
    RejectionReason,
    VNCostModel,
    round_down_to_lot,
)
from src.portfolio.construction import (
    PortfolioConstraints,
    get_ledoit_wolf_cov,
    kelly_optimize,
    mean_variance_optimize,
)
from src.trading.regime_policy import (
    NO_TRADE_REGIMES,
    PENALTY_REGIMES,
    REGIME_PENALTY_FACTOR,
)
from src.trading.breadth import breadth_scalar
from src.trading.cohort_weights import prob_scaled_weights
from src.trading.risk_tier import apply_nav_tier_cap, classify_risk_tier

LOGGER = logging.getLogger("backtest.walk_forward")

# An oracle maps an (n, seq_len, n_features) tensor → (n, 3) class probabilities
# [P(DOWN), P(FLAT), P(UP)]  (or an (n,) P(UP) vector).
SignalOracle = Callable[[np.ndarray], np.ndarray]

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Config & records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Walk-forward harness configuration."""
    seq_len: int = 20
    initial_capital: float = 10_000_000_000.0     # 10B VND

    feature_cols: list[str] = field(default_factory=list)

    # Universe selection (per rebalance)
    signal_threshold: float = 0.40                 # min P(UP) to consider a long
    max_positions: int = 10
    # Prob-scaled within-cohort weights (plan prob-scaled-tranche-weights_PLAN_20-07-26):
    # OFF = validated equal-weight cohorts. ON = weight picks by normalized
    # edge over the admission floor (src/trading/cohort_weights.py), capped at
    # 2× equal weight. A/B-only until the acceptance gate passes.
    use_prob_weights: bool = False

    # ── LIQUIDITY GATE (top-N ADV) ───────────────────────────────────────────
    # Restrict the daily candidate universe to the top-`liquid_top_n` names by
    # trailing-`adv_window` Average Daily $-Volume, ranked WITHIN each date.
    # Applied AFTER `_inference` and BEFORE `_allocate`, so the Kelly/MV
    # optimizer never sees an illiquid ticker.  None or ≤ 0 → off.
    # 50 ⇒ VN50-like cap on the tradeable universe.
    liquid_top_n: int | None = 50
    adv_window: int = 20

    # SOFT REGIME SCALING replaces the old hard `min_bull_prob` kill-switch.
    # The HMM Macro Risk Oracle supplies a per-day P(Bull) ∈ [0,1]; `_allocate`
    # multiplies the final target weights by it (continuous, differentiable —
    # no risk cliff). P(Bull) is passed to `run(p_bull_series=...)`; when absent
    # the engine defaults to 1.0 (full exposure, no scaling).

    # Covariance estimation
    cov_lookback: int = 60                         # trailing trading days
    cov_min_obs: int = 20                          # need ≥ this many aligned returns

    # Kelly + MV
    kelly_fraction: float = 0.5
    profit_factor: float = 1.8                     # global PF proxy for Kelly edge
    risk_aversion: float = 2.0

    # Rebalance cadence
    rebalance_frequency: int = 1                   # every N trading days

    # ── REBALANCE MODE ───────────────────────────────────────────────────────
    # "grid"    → classic delta-rebalance of one concentrated book every
    #             `rebalance_frequency` days.  ~45 correlated entry dates over
    #             a 4-year OOS window: timing variance dominates (the grid-date
    #             study measured picks at −0.83% T+20 vs +0.87% cohort).
    # "tranche" → AFML-style staggered book matched to the triple-barrier
    #             label horizon: EVERY day deploy NAV/`tranche_hold_days` into
    #             the day's top `max_positions` names (equal weight), hold each
    #             cohort exactly `tranche_hold_days` trading days, liquidate at
    #             the ATC.  ~880 partially-independent bets harvest the
    #             per-trade edge the per-row studies measured (+1.60% net T+20
    #             for within-day top-5).  `rebalance_frequency` is ignored;
    #             P(Bull) scales each NEW tranche's budget (no forced
    #             mid-cycle liquidation).
    rebalance_mode: str = "grid"
    tranche_hold_days: int = 20

    # ── TRANCHE BARRIER EXITS (triple-barrier replication) ──────────────────
    # The triple-barrier labels exit at PT (+pt·σ), SL (−sl·σ), or the vertical
    # barrier — the tranche book's fixed hold only replicates the vertical.
    # When set, each position records its entry price and entry-time daily vol;
    # a close beyond entry·(1 + pt·σ) or entry·(1 − sl·σ) flags the position
    # for exit at that day's ATC (retrying on rejection).  Match the label
    # convention (RunConfig: tb_pt=3.0, tb_sl=2.0) to trade what the model was
    # trained to predict.  None ⇒ barrier disabled (pure fixed-hold cohorts).
    tranche_pt_sigma: float | None = None
    tranche_sl_sigma: float | None = None

    # ── BURST SIZING (22-07-26 concentration follow-up — A/B, default off) ───
    # Decouples the DAILY BUDGET divisor from the hold length. The tranche
    # budget is normally nav/tranche_hold_days (calendar-based: full deployment
    # only if a cohort opens every single day). Under a selective admission
    # gate (absolute_gate ≥0.46 opens ~5% of days) that leaves ~95% of capital
    # permanently idle — the concentration A/B measured Sharpe 0.600 at just
    # −4.4% DD but 1/8 the PnL. `tranche_budget_days = 10` deploys nav/10 on
    # each day that DOES clear admission (3× the calendar budget), spending
    # the unused risk headroom. None ⇒ nav/tranche_hold_days — existing
    # behaviour byte-for-byte. Cash constraint still applies (a burst can
    # never deploy more than available cash).
    tranche_budget_days: int | None = None

    # ── REGIME-CONDITIONAL SIZING ────────────────────────────────────────────
    # When True, per-name allocations inside each tranche are modulated by the
    # ticker's `market_regime` (0-7, from build_regime_features), mirroring the
    # serve-path src/bot/sizing.py policy (single source: regime_policy.py):
    #   NO_TRADE_REGIMES {0,7} → exclude the name from that day's cohort;
    #   PENALTY_REGIMES  {1,6} → multiply its per-name notional by
    #                            REGIME_PENALTY_FACTOR (0.5×); freed half stays cash.
    # An absolute REGIME_PENALTY_CAP×NAV cap is inert at tranche scale
    # (per_name ≈ nav/(H·picks) ≈ 0.7% NAV ≪ 10% NAV), so the engine mirrors the
    # serve INTENT via the scale-invariant factor instead. Default False ⇒ no
    # allocation change (existing behaviour byte-for-byte).
    use_regime_sizing: bool = False

    # ── DISCRETE NAV-TIER PORTFOLIO CAP (A/B experiment — UNVALIDATED) ───────
    # When True, each day's NEW tranche budget is clamped so TOTAL deployment
    # (nav − cash) never exceeds the discrete tier cap from
    # src/trading/risk_tier.py: RISK_HIGH 20% / RISK_MID 60% / RISK_LOW 80% of
    # NAV, classified from that day's P(Bull). Existing positions are never
    # force-liquidated — a tier downgrade bleeds down as tranches expire.
    # Default False ⇒ existing behaviour byte-for-byte. Do not enable in the
    # GOLDEN eval until the A/B vs. use_regime_sizing says it earns its keep.
    use_nav_tier_cap: bool = False

    # ── SERVE-MIRROR ADMISSION (A/B experiment — default off) ────────────────
    # Gates WHICH NAMES enter a tranche (orthogonal to use_nav_tier_cap, which
    # gates HOW MUCH TOTAL NAV a tranche deploys). Two modes:
    #   "cross_sectional" → UNCHANGED default: rank by P(UP), take the top
    #       `max_positions` names above `signal_threshold` every trading day
    #       (the validated tranche book — a low relative floor that the top-N
    #       book saturates against, so a day almost always deploys SOMETHING).
    #   "absolute_gate"   → mirrors serve's `main.predict_v3_horizon` meta-gate:
    #       filter today's ranked candidates to `p_up >= admission_floor` FIRST
    #       (an ABSOLUTE floor, inclusive `>=` matching serve's `meta_gate`),
    #       cap the survivor list at `admission_pool_cap` (serve's
    #       `_ARBITRATOR_POOL=6`), THEN take the top `max_positions` of that
    #       capped/filtered list. When zero names clear the floor the day
    #       deploys NOTHING and the budget stays cash — mirroring serve's
    #       zero-candidate-day fall-to-cash economics (NOT its Vietnamese-text
    #       monitoring card, a display-layer concern out of scope here).
    # Default "cross_sectional" ⇒ existing `_tranche_day` admission byte-for-byte.
    #   "rank_breadth"    → (22-07-26) NO absolute floor at all — always take
    #       the best-ranked K names available. K = round(max_positions ×
    #       breadth_scalar(today's market breadth)) via the SAME piecewise-
    #       linear mapping already used for the exposure brake
    #       (src/trading/breadth.py). K=0 on a breadth-famine day (cash, same
    #       diagnostic as absolute_gate's zero-candidate days); K=max_positions
    #       once breadth clears `rank_breadth_trigger`. Attacks both known
    #       admission failure modes at once: absolute_gate's hard zero-outs on
    #       days the market still has relatively-better names, AND
    #       cross_sectional's chronic over-admission regardless of how thin
    #       the whole tape is.
    #   "argmax": admit every name whose 3-class ARGMAX is UP, ranked by p_up
    #       within that set. No absolute threshold anywhere, so it cannot be
    #       starved by output-distribution drift the way absolute_gate was
    #       (serve p90 fell 0.463 → 0.423 against a frozen tau=0.46). Motivated
    #       by live paperlog evidence that P(UP) is anti-informative across the
    #       gate's range (Platt slope −0.274) while argmax==UP is not.
    #       TESTED AND REJECTED 11-08-26 (scripts/analyze_argmax_admission_ab.py):
    #       bought NOTHING across 920 OOS days. The current model is bearish to
    #       the point that argmax==UP occurs on 0.015% of name-days — p_up's p99
    #       (0.4523) is below p_down's MEDIAN (0.5957), so UP cannot win the
    #       argmax. Kept in-tree behind the flag: the rule is sound, the model's
    #       class balance is what makes it unusable, so it is worth re-testing
    #       after any retrain that changes the label distribution.
    #   "absolute_gate_argmax": the tau floor AND the arbitrator's class
    #       condition (argmax==UP). This is what SERVE actually requires —
    #       `make_final_decision` only returns BUY on an UP argmax and
    #       `_select_candidates` dispatches only `final_decision == 2` — so the
    #       floor alone, which is all the backtest ever validated, admits more
    #       than production does.
    admission_mode: str = "cross_sectional"
    admission_floor: float = 0.45
    admission_pool_cap: int = 6
    # Ranking within the admitted set. "p_up" is what every validated result was
    # earned with; "random" is the backtestable proxy for serve's sentiment-first
    # ordering (sentiment has no point-in-time history, so it cannot be modelled
    # directly). Default preserves every existing result byte-for-byte.
    rank_mode: str = "p_up"
    rank_seed: int = 0

    # ── SERVE-PARITY DEFENSIVE LAYERS (all opt-in, all default OFF) ──────────
    # Each of these runs in production and NONE was in the backtest that
    # produced the validated numbers. Every one was added after a real loss,
    # one at a time, without measurement — so nobody knows whether they protect
    # the edge or destroy it. Encoded here so that can finally be answered;
    # defaults keep every existing result byte-for-byte.
    #
    # Sector cap: at most N admitted names per sector (serve:
    # CONFIG.trading.arbitrator_sector_cap = 2). 0 disables. Uses the SAME
    # src/trading/sector_map.py the serve path uses, so the two cannot drift.
    serve_sector_cap: int = 0
    # Open-cohort dedup: exclude names already held in an open tranche (serve:
    # signal_ledger.open_tickers). The July BSR incident — re-dispatched 4
    # consecutive days into a falling knife — is what motivated it.
    serve_cohort_dedup: bool = False
    # Admission hysteresis: require N CONSECUTIVE raw-qualifying days before a
    # name is admissible (serve: hysteresis_min_qualify_days = 2). 0 disables.
    # The streak is tracked on the RAW qualifying set, independent of the other
    # two filters, mirroring serve's candidate_hysteresis.
    serve_hysteresis_days: int = 0
    # 3-leg exposure brake (serve: src/bot/garch_brake.live_exposure_scalar).
    # drift and breadth legs are computed here from the panel with the SAME
    # piecewise ramps as serve; the GARCH leg is approximated by the engine's
    # existing `p_bull_series`, which already multiplies the budget, so it is
    # NOT re-applied here (that would double-count it).
    serve_exposure_brake: bool = False
    drift_brake_window: int = 10
    drift_brake_trigger: float = -0.03
    drift_brake_full: float = -0.06
    drift_brake_floor: float = 0.5
    breadth_brake_trigger: float = 0.40
    breadth_brake_floor_level: float = 0.25
    breadth_brake_floor: float = 0.5
    rank_breadth_window: int = 20
    rank_breadth_trigger: float = 0.40
    rank_breadth_floor_level: float = 0.25
    rank_breadth_floor: float = 0.5

    # OOS gate: only place trades on/after this date.  Days before it are still
    # iterated (NAV marked, corporate actions applied, features/cov built from
    # them) so the engine has lookback, but NO trade is initiated until the
    # out-of-sample period begins.  None ⇒ trade as soon as seq_len history exists.
    start_trading_date: date | None = None

    # Execution
    atc_participation: float = 0.15                # ATC matched vol = day_vol × this
    vol_lookback: int = 20                         # daily-vol estimate window
    atc_session: tuple[int, int] = (14, 35)        # HH, MM (ICT) for the ATC order ts
    fee_buffer: float = 0.015                      # 1.5% — absorbs sqrt impact + 0.1% sell tax + spread on LIQUID positions

    # ── PRICE UNIT (panel → absolute VND) ────────────────────────────────────
    # The parquet OHLCV shards store prices in THOUSANDS of VND (e.g. 13.45 =
    # 13,450 VND), but VNCostModel's tick grid (10/50/100 VND), the band
    # tolerances (±1 VND), and share-quantity math (qty = w·NAV / price) all
    # assume ABSOLUTE VND.  Feeding thousand-scale prices in unfixed cost
    # 5–100% of notional PER FILL in phantom tick-rounding "slippage" (e.g. a
    # 13.45 buy rounded UP to the 20 grid line, a 9.8 sell rounded DOWN to 0)
    # and inflated share counts 1000× (blowing through participation caps).
    # `_prepare` multiplies open/high/low/close by this factor so every
    # downstream computation runs in true VND.  Set to 1.0 for panels that are
    # already in absolute VND.
    price_unit_vnd: float = 1000.0

    # Constraints + cost model
    constraints: PortfolioConstraints = field(default_factory=lambda: PortfolioConstraints(
        max_weight=0.10,
        sector_caps={},
        long_only=True,
        target_leverage=0.95,                      # keep a small cash buffer
        target_vol=0.15,
    ))
    exec_config: ExecutionConfig = field(default_factory=ExecutionConfig)

    default_exchange: str = "HOSE"


@dataclass
class DailyRecord:
    """One row of the equity curve."""
    date: date
    nav: float
    cash: float
    market_value: float
    daily_return: float
    n_positions: int
    n_orders: int
    n_fills: int
    n_rejections: int
    dividend_cash: float
    gross_exposure: float


@dataclass
class WalkForwardResult:
    equity_curve: pd.DataFrame
    fills: list[dict]
    rejections: list[dict]
    corporate_action_log: list[dict]
    metrics: dict
    final_nav: float
    final_cash: float
    # Serve-mirror admission diagnostic (A/B experiment). Number of OOS trading
    # days where the `absolute_gate` admission filter produced an EMPTY survivor
    # list (nothing cleared `admission_floor`) so the tranche budget stayed cash.
    # Always 0 for `admission_mode="cross_sectional"` (no absolute floor), so the
    # default preserves the existing result contract for every other caller.
    zero_candidate_days: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# (make_lstm_oracle removed — V4 is pure-tabular; no LSTM/torch oracle path.)


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Chronological walk-forward simulator.  Construct with a config + oracle,
    then call `run(panel, corporate_actions)`.

    Cash model
        `self.cash` is the single source of truth for trading cash:
            buys  → cash += fill.signed_cash_flow  (negative)
            sells → cash += fill.signed_cash_flow  (positive)
            dividends → cash += cash_credited       (swept each morning)
        NAV = Σ net_shares·close + self.cash.  (InventoryTracker.cash_balance is
        an independent audit trail; not used for NAV to avoid double-counting.)
    """

    def __init__(self, config: WalkForwardConfig, oracle: SignalOracle) -> None:
        if not config.feature_cols:
            raise ValueError("WalkForwardConfig.feature_cols must be non-empty")
        self.config = config
        self.oracle = oracle
        self.model = VNCostModel(config.exec_config)

        # State (initialised in run())
        self.cash: float = 0.0
        self.inventory: InventoryTracker = InventoryTracker()
        self.records: list[DailyRecord] = []
        self.fills_log: list[dict] = []
        self.rejections_log: list[dict] = []
        self.ca_log: list[dict] = []
        self._held_tickers: set[str] = set()
        self._last_price: dict[str, float] = {}
        self._prev_nav: float = 0.0
        self._last_rebalance_idx: int = -10**9
        self._dividend_cash_today: float = 0.0
        # Tranche mode: each entry is {"entry_idx": int, "positions": {ticker: qty}}.
        # A tranche expires `tranche_hold_days` trading days after entry; its
        # positions are then sold (retrying daily on rejection / missing print).
        self._tranches: list[dict] = []
        # Serve-mirror admission diagnostic: OOS days where the `absolute_gate`
        # survivor list was empty (budget stayed cash). Reset per `run()`.
        self._zero_candidate_days: int = 0

    # ── Public entrypoint ──────────────────────────────────────────────────
    def run(
        self,
        panel: pl.DataFrame | pd.DataFrame,
        corporate_actions: Sequence[CorporateActionEvent] | None = None,
        p_bull_series: pd.Series | None = None,
        inference_cache: dict[date, tuple[np.ndarray, list[str]]] | None = None,
        breadth_series: pd.Series | None = None,
        budget_days_series: pd.Series | None = None,
    ) -> WalkForwardResult:
        """
        `p_bull_series` — date-indexed HMM P(Bull) (leak-free filtered). Each
        rebalance scales target weights by that day's P(Bull). When None, the
        engine uses 1.0 (full exposure, no soft scaling).

        `breadth_series` — date-indexed market breadth (see
        `src.trading.breadth.breadth_time_series`), consumed ONLY by
        `admission_mode="rank_breadth"`. When None (or a date is missing),
        the engine assumes breadth=1.0 (fail-open — full K, matching every
        other brake's fail-open contract in this codebase).

        `budget_days_series` — date-indexed burst-budget divisor (see
        `src.trading.vol_sizing.vol_scaled_budget_days`), overrides the flat
        `cfg.tranche_budget_days` constant per-day when supplied. When None
        (or a date is missing), falls back to `cfg.tranche_budget_days` /
        `cfg.tranche_hold_days` unchanged.

        `inference_cache` — optional ``{D: (p_up, tickers)}`` map, MUTATED in
        place.  The per-day oracle scoring (`_inference`) depends only on
        (oracle, D, panel) — NOT on `signal_threshold` — so a threshold sweep
        reusing the SAME frozen oracle can pass one shared cache to skip the
        expensive GBM re-inference on every threshold after the first.  Keep the
        cache PER-ORACLE (per seed): a different ensemble must use a different
        dict, or stale probabilities will leak across seeds.
        """
        self._prepare(panel, corporate_actions)
        # Per-run handle the cached `_inference` reads/writes (None ⇒ no caching).
        self._inference_cache = inference_cache

        # date → P(Bull) lookup; default 1.0 (full exposure) when absent.
        self._p_bull: dict[date, float] = {}
        if p_bull_series is not None:
            self._p_bull = {
                pd.Timestamp(d).date(): float(v)
                for d, v in p_bull_series.dropna().items()
            }

        # date → market breadth lookup; default 1.0 (fail-open, full K) when
        # absent — see `admission_mode="rank_breadth"` in `_tranche_day`.
        self._breadth: dict[date, float] = {}
        if breadth_series is not None:
            self._breadth = {
                pd.Timestamp(d).date(): float(v)
                for d, v in breadth_series.dropna().items()
            }

        # date → burst-budget divisor lookup; empty ⇒ every `.get(D)` call
        # returns None and the flat `cfg.tranche_budget_days` constant wins.
        self._budget_days: dict[date, int] = {}
        if budget_days_series is not None:
            self._budget_days = {
                pd.Timestamp(d).date(): int(v)
                for d, v in budget_days_series.dropna().items()
            }

        # Serve-parity layers. `_qualify_streak` mirrors serve's
        # candidate_hysteresis table: consecutive RAW-qualifying days per
        # ticker, tracked independently of the sector cap and dedup so a name
        # blocked by those does not lose its streak (the serve semantics).
        self._qualify_streak: dict[str, int] = {}
        # date → trailing cumulative market return, for the drift brake leg.
        self._drift_cum: dict[date, float] = {}
        if self.config.serve_exposure_brake:
            self._drift_cum = self._build_drift_series(
                window=self.config.drift_brake_window)

        self.cash = self.config.initial_capital
        self._prev_nav = self.config.initial_capital
        self._zero_candidate_days = 0

        for i, D in enumerate(self.calendar):
            self._dividend_cash_today = 0.0

            # 1. MORNING — corporate actions
            self._morning_routine(D)

            # 2–4. REBALANCE (gated by cadence + OOS start; needs seq_len history)
            n_orders = n_fills = n_rej = 0
            after_start = (self.config.start_trading_date is None
                           or D >= self.config.start_trading_date)
            if self.config.rebalance_mode == "tranche":
                if i >= self.config.seq_len and after_start:
                    n_orders, n_fills, n_rej = self._tranche_day(i, D)
            elif (i >= self.config.seq_len and after_start
                    and (i - self._last_rebalance_idx) >= self.config.rebalance_frequency):
                p_up, sig_tickers = self._inference(D)
                p_up, sig_tickers = self._apply_liquidity_filter(D, p_up, sig_tickers)
                if sig_tickers:
                    p_bull_today = self._p_bull.get(D, 1.0)   # soft regime weight
                    target_weights = self._allocate(D, p_up, sig_tickers, p_bull_today)
                    n_orders, n_fills, n_rej = self._execute(D, target_weights)
                    self._last_rebalance_idx = i

            # 5. CLOSING — mark-to-market
            self._closing(D, n_orders, n_fills, n_rej)

        return self._build_result()

    # ── Data preparation ───────────────────────────────────────────────────
    def _prepare(
        self,
        panel: pl.DataFrame | pd.DataFrame,
        corporate_actions: Sequence[CorporateActionEvent] | None,
    ) -> None:
        pdf = panel.to_pandas() if isinstance(panel, pl.DataFrame) else panel.copy()
        pdf["date"] = pd.to_datetime(pdf["date"]).dt.date

        required = {"ticker", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(pdf.columns)
        if missing:
            raise ValueError(f"panel missing columns: {missing}")
        miss_feat = [c for c in self.config.feature_cols if c not in pdf.columns]
        if miss_feat:
            raise ValueError(f"panel missing feature columns: {miss_feat}")

        if "exchange" not in pdf.columns:
            pdf["exchange"] = self.config.default_exchange

        pdf = pdf.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Convert panel prices (thousand-VND parquet convention) to ABSOLUTE VND
        # so tick rounding, band tolerances, and share-quantity math are correct.
        # Feature columns are untouched (z-scored derivatives, scale-free).
        scale = float(self.config.price_unit_vnd)
        if scale != 1.0:
            for col in ("open", "high", "low", "close"):
                pdf[col] = pdf[col] * scale
            LOGGER.info("Price unit | panel OHLC × %.0f → absolute VND", scale)

        # Per-ticker derived columns: prior close (band ref) + trailing daily vol.
        pdf["ref_price"] = pdf.groupby("ticker", sort=False)["close"].shift(1)
        rets = pdf.groupby("ticker", sort=False)["close"].pct_change()
        pdf["ret"] = rets
        pdf["vol"] = (
            rets.groupby(pdf["ticker"]).transform(
                lambda s: s.rolling(self.config.vol_lookback, min_periods=5).std()
            )
        )
        # Leak-safe ADV (trailing $-volume mean, shifted 1 day per ticker) so the
        # liquidity gate uses ONLY information available before today's open.
        pdf["dvol"] = pdf["close"] * pdf["volume"]
        pdf["adv20"] = (
            pdf.groupby("ticker", sort=False)["dvol"].transform(
                lambda s: s.rolling(self.config.adv_window,
                                    min_periods=self.config.adv_window).mean().shift(1)
            )
        )

        self.ticker_frames: dict[str, pd.DataFrame] = {
            tk: g.reset_index(drop=True) for tk, g in pdf.groupby("ticker", sort=False)
        }
        # Fast (date → {ticker → row}) lookup for execution / marking.
        self._day_index: dict[date, dict[str, dict]] = {}
        for tk, g in self.ticker_frames.items():
            for row in g.itertuples(index=False):
                row_dict = {
                    "open": row.open, "high": row.high, "low": row.low,
                    "close": row.close, "volume": row.volume,
                    "ref_price": row.ref_price, "vol": row.vol,
                    "adv20": row.adv20,
                    "exchange": row.exchange,
                }
                # Per-ticker regime label (only present when the panel was built
                # with build_regime_features). Used by regime-conditional sizing;
                # absent in minimal test panels → safe `.get(...)` fallback.
                if hasattr(row, "market_regime"):
                    row_dict["market_regime"] = int(row.market_regime)
                self._day_index.setdefault(row.date, {})[tk] = row_dict

        # Sector map (optional column)
        if "sector" in pdf.columns:
            self._sector_map = dict(pdf[["ticker", "sector"]].drop_duplicates().values)
        else:
            self._sector_map = {}

        self.calendar: list[date] = sorted(pdf["date"].unique())

        # Corporate actions bucketed by ex-date.
        self._ca_by_date: dict[date, list[CorporateActionEvent]] = {}
        for ev in (corporate_actions or []):
            self._ca_by_date.setdefault(ev.ex_date, []).append(ev)

        LOGGER.info(
            "Walk-forward prepared | tickers=%d  days=%d  range=%s..%s  CAs=%d",
            len(self.ticker_frames), len(self.calendar),
            self.calendar[0], self.calendar[-1],
            sum(len(v) for v in self._ca_by_date.values()),
        )
        if self.config.liquid_top_n is not None and self.config.liquid_top_n > 0:
            LOGGER.info("Liquidity gate ACTIVE | top-%d by trailing-%dd ADV (within-date rank)",
                        int(self.config.liquid_top_n), self.config.adv_window)

    def _build_drift_series(self, window: int = 10) -> dict[date, float]:
        """date → trailing cumulative equal-weight market return (drift leg).

        Mirrors the input to serve's `garch_brake._drift_scalar`: the
        market-proxy return over the trailing window. Built from
        `self.ticker_frames` (already normalised by `_prepare`) rather than the
        raw panel, because `run()` accepts either a polars OR a pandas frame and
        this must not care which.

        Strictly backward-looking: the value for D uses returns up to D-1's
        close only, since D's admission decision is made before D trades.
        """
        try:
            per_day: dict[date, list[float]] = {}
            for g in self.ticker_frames.values():
                closes = g["close"].to_numpy(dtype=float)
                dates = list(g["date"])
                for i in range(1, len(closes)):
                    prev = closes[i - 1]
                    if prev > 0:
                        per_day.setdefault(dates[i], []).append(closes[i] / prev - 1.0)
        except Exception:  # noqa: BLE001 — the brake must never break a run
            LOGGER.warning("[serve-parity] drift series build failed — leg disabled.",
                           exc_info=True)
            return {}

        ordered = sorted(per_day)
        rets = [sum(per_day[d]) / len(per_day[d]) for d in ordered]
        out: dict[date, float] = {}
        for i, d in enumerate(ordered):
            if i == 0:
                out[d] = 0.0          # no history yet → nothing to brake
                continue
            cum = 1.0
            for r in rets[max(0, i - window):i]:
                cum *= (1.0 + r)
            out[d] = cum - 1.0
        return out

    def _apply_serve_layers(self, admitted: list[str],
                            raw_qualified: list[str]) -> list[str]:
        """Serve's three admission filters, in serve's order. Order-preserving.

        Each is a no-op unless its config knob is set, so the default path is
        byte-identical to every result produced before these existed.

        `raw_qualified` is the PRE-filter qualifying set. The hysteresis streak
        is counted on it rather than on `admitted`, mirroring serve: a name held
        out by dedup or the sector cap must not lose the streak it would
        otherwise have built.
        """
        cfg = self.config

        # 1. Hysteresis — N consecutive raw-qualifying days before admissible.
        if cfg.serve_hysteresis_days > 0:
            qualified = set(raw_qualified)
            for tk in qualified:
                self._qualify_streak[tk] = self._qualify_streak.get(tk, 0) + 1
            for tk in list(self._qualify_streak):
                if tk not in qualified:
                    self._qualify_streak[tk] = 0
            admitted = [t for t in admitted
                        if self._qualify_streak.get(t, 0) >= cfg.serve_hysteresis_days]

        # 2. Open-cohort dedup — skip names already held in a live tranche.
        if cfg.serve_cohort_dedup:
            held = {tk for tr in self._tranches for tk in tr["positions"]}
            admitted = [t for t in admitted if t not in held]

        # 3. Sector cap — at most N per sector, keeping the best-ranked.
        if cfg.serve_sector_cap > 0:
            from src.trading.sector_map import sector_of  # noqa: PLC0415

            seen: dict[str, int] = {}
            capped: list[str] = []
            for t in admitted:
                sec = sector_of(t)
                # OTHER is uncapped, exactly as in serve's apply_sector_cap —
                # it is the unmapped bucket, not a real sector.
                if sec == "OTHER":
                    capped.append(t)
                    continue
                if seen.get(sec, 0) >= cfg.serve_sector_cap:
                    continue
                seen[sec] = seen.get(sec, 0) + 1
                capped.append(t)
            admitted = capped

        return admitted

    def _serve_exposure_scalar(self, D: date) -> float:
        """drift × breadth legs of serve's 3-leg brake, as a single multiplier.

        The GARCH leg is deliberately NOT included: the engine already
        multiplies the budget by `p_bull`, which is the same macro-regime
        posterior the serve GARCH leg reads, and applying it twice would
        double-count. Each leg fails open to 1.0 independently, matching
        `garch_brake.live_exposure_legs`.
        """
        cfg = self.config
        if not cfg.serve_exposure_brake:
            return 1.0

        def _ramp(x: float, trigger: float, full: float, floor: float) -> float:
            """Piecewise-linear 1.0 → floor between `trigger` and `full`."""
            if full == trigger:
                return 1.0
            if (x - trigger) / (full - trigger) <= 0.0:
                return 1.0
            frac = (trigger - x) / (trigger - full)
            return max(floor, 1.0 - min(1.0, frac) * (1.0 - floor))

        drift_s = 1.0
        cum = self._drift_cum.get(D)
        if cum is not None:
            drift_s = _ramp(float(cum), cfg.drift_brake_trigger,
                            cfg.drift_brake_full, cfg.drift_brake_floor)
        breadth_s = 1.0
        b = self._breadth.get(D)
        if b is not None:
            breadth_s = _ramp(float(b), cfg.breadth_brake_trigger,
                              cfg.breadth_brake_floor_level, cfg.breadth_brake_floor)
        return min(drift_s, breadth_s)

    # ── 1. Morning routine ─────────────────────────────────────────────────
    def _morning_routine(self, D: date) -> None:
        for ev in self._ca_by_date.get(D, []):
            result = self.inventory.apply_corporate_action(ev)
            credited = result.get("cash_credited", 0.0)
            if credited:
                self.cash += credited                 # sweep dividend into trading cash
                self._dividend_cash_today += credited
            self.ca_log.append({"date": D.isoformat(), **result})
            LOGGER.debug("morning CA %s", result)

    # ── 2. Inference ───────────────────────────────────────────────────────
    def _inference(self, D: date) -> tuple[np.ndarray, list[str]]:
        """Build the leak-safe (≤ D−1) tensor and run the oracle → P(UP).

        When an `inference_cache` was supplied to `run()`, the (threshold-
        independent) result for `D` is memoized so a threshold sweep over the
        same frozen oracle pays the GBM scoring cost exactly once per day.

        Also records `self._argmax_up_today`: {ticker: bool} for whether UP was
        the ARGMAX class, not merely above some P(UP) level. The two are very
        different admission rules — live paperlog evidence (11-08-26) has P(UP)
        anti-correlated with outcomes across the gate's operating range (Platt
        slope −0.274) while argmax==UP picks beat the baseline by ~3.6pp — so
        `admission_mode="argmax"` needs the class decision, which p_up alone
        cannot express.
        """
        cache = getattr(self, "_inference_cache", None)
        if cache is not None and D in cache:
            cached = cache[D]
            # 3-tuple since the argmax A/B; tolerate a legacy 2-tuple cache so a
            # caller holding an older dict cannot crash the run.
            if len(cached) == 3:
                p_up_c, tickers_c, argmax_c = cached
            else:
                p_up_c, tickers_c = cached
                argmax_c = {}
            self._argmax_up_today = dict(argmax_c)
            # Defensive copies: downstream (_apply_liquidity_filter / _allocate)
            # treats these as read-only, but copying guarantees a future mutation
            # can never poison the shared cache.
            return p_up_c.copy(), list(tickers_c)

        seq = self.config.seq_len
        feats = self.config.feature_cols
        X_list: list[np.ndarray] = []
        tickers: list[str] = []

        for tk, frame in self.ticker_frames.items():
            hist = frame[frame["date"] < D]
            if len(hist) < seq:
                continue
            window = hist[feats].to_numpy()[-seq:]
            if window.shape != (seq, len(feats)) or not np.isfinite(window).all():
                continue
            X_list.append(window)
            tickers.append(tk)

        if not X_list:
            self._argmax_up_today = {}
            if cache is not None:
                cache[D] = (np.array([]), [], {})
            return np.array([]), []

        X = np.stack(X_list).astype(np.float32)        # (n, seq, F)
        probs = self.oracle(X)
        probs = np.asarray(probs)
        p_up = probs[:, 2] if probs.ndim == 2 else probs.ravel()
        # UP is the argmax over [DOWN, SIDE, UP]. Ties resolve AGAINST admission
        # (strict >) so a flat 3-way split never counts as a BUY call.
        if probs.ndim == 2 and probs.shape[1] >= 3:
            argmax_up = {
                tk: bool(probs[i, 2] > probs[i, 0] and probs[i, 2] > probs[i, 1])
                for i, tk in enumerate(tickers)
            }
        else:
            # A 1-D oracle exposes no class structure; leave argmax unavailable
            # rather than inventing it from a threshold.
            argmax_up = {}
        self._argmax_up_today = argmax_up
        if cache is not None:
            cache[D] = (p_up, tickers, argmax_up)
        return p_up, tickers

    # ── 2b. Liquidity gate (top-N ADV filter) ──────────────────────────────
    def _apply_liquidity_filter(
        self, D: date, p_up: np.ndarray, tickers: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Restrict the candidate universe to the top-`liquid_top_n` names by
        trailing-window ADV (ranked WITHIN this date — highest ADV survives).
        No-op when the filter is disabled (`liquid_top_n` None or ≤ 0) or when
        fewer than 5 names have valid ADV (warm-up → fall back to the full set
        rather than trade nothing).  Returns the surviving (p_up, tickers) pair,
        masked to the top-N LIQUID slice.
        """
        top_n = self.config.liquid_top_n
        if not tickers or top_n is None or top_n <= 0:
            return p_up, tickers
        day = self._day_index.get(D, {})
        advs = pd.Series([day.get(t, {}).get("adv20") for t in tickers], dtype=float)
        if advs.notna().sum() < 5:
            return p_up, tickers
        # Keep the top-N by ADV (ties broken arbitrarily by Series.rank).  When
        # there are fewer than N valid ADVs, just keep all of them.
        k = min(int(top_n), int(advs.notna().sum()))
        # Rank descending so the LARGEST ADV gets rank 1, then keep ranks ≤ k.
        keep = (advs.rank(method="first", ascending=False, na_option="keep") <= k).to_numpy()
        if not keep.any():
            return np.array([]), []
        idx = np.flatnonzero(keep)
        return p_up[idx], [tickers[i] for i in idx]

    # ── 3. Risk & allocation ───────────────────────────────────────────────
    def _allocate(
        self, D: date, p_up: np.ndarray, tickers: list[str], p_bull_today: float = 1.0,
    ) -> dict[str, float]:
        cfg = self.config

        # Universe: top conviction longs above signal_threshold (no hard gate).
        order = np.argsort(p_up)[::-1]
        chosen = [(tickers[j], float(p_up[j])) for j in order
                  if p_up[j] >= cfg.signal_threshold][:cfg.max_positions]
        if not chosen:
            return {}
        sel = [t for t, _ in chosen]
        W = np.array([w for _, w in chosen], dtype=np.float64)

        Sigma = self._covariance(D, sel)
        if Sigma is None:
            return {}

        # Covariance-coupled fractional Kelly → long-only intent → constrained MV.
        PF = np.full(len(sel), cfg.profit_factor, dtype=np.float64)
        kelly_w = kelly_optimize(W, PF, Sigma, fraction=cfg.kelly_fraction)
        mu = np.clip(kelly_w, 0.0, None)
        if mu.sum() <= 0:
            return {}

        # Feasibility cap: with a thin universe, max_weight × n_sel can be below
        # the configured leverage (e.g. 2 names @ 25% cap can't deploy 90%).
        # Deploy what is feasible and leave the remainder in cash, rather than
        # letting the QP raise infeasibility and trading nothing.
        n_sel = len(sel)
        feasible_lev = min(
            cfg.constraints.target_leverage,
            cfg.constraints.max_weight * n_sel * 0.999,
        )
        constraints = replace(
            cfg.constraints,
            ticker_to_sector={t: self._sector_map.get(t, "OTHER") for t in sel},
            target_leverage=feasible_lev,
        )
        try:
            res = mean_variance_optimize(
                mu, Sigma, sel, constraints, risk_aversion=cfg.risk_aversion,
            )
        except ValueError as exc:
            LOGGER.warning("MV optimize failed on %s: %s", D, exc)
            return {}
        w = res["weights"]

        # ── SOFT REGIME SCALING (HMM Macro Risk Oracle) ─────────────────────
        # Multiply every base weight by P(Bull). Exposure scales continuously
        # with regime conviction (P(Bull)=0.2 → ~20% invested / 80% cash); no
        # non-differentiable cliff. p_bull_today defaults to 1.0 (no HMM).
        p_bull = float(np.clip(p_bull_today, 0.0, 1.0))
        if p_bull < 0.999:
            LOGGER.info("[%s] soft regime scaling: P(Bull)=%.3f → gross exposure ×%.2f",
                        D, p_bull, p_bull)
        return {sel[j]: float(w[j]) * p_bull for j in range(len(sel)) if w[j] > 1e-6}

    def _covariance(self, D: date, tickers: list[str]) -> np.ndarray | None:
        cfg = self.config
        series: dict[str, pd.Series] = {}
        for tk in tickers:
            frame = self.ticker_frames[tk]
            hist = frame[frame["date"] < D].tail(cfg.cov_lookback + 1)
            series[tk] = hist.set_index("date")["close"]
        rets = pd.DataFrame(series).pct_change().dropna(how="any")
        if len(rets) < cfg.cov_min_obs or rets.shape[1] != len(tickers):
            return None
        Sigma, _delta = get_ledoit_wolf_cov(rets.to_numpy())
        return Sigma

    # ── 4. Execution ───────────────────────────────────────────────────────
    def _execute(self, D: date, target_weights: dict[str, float]) -> tuple[int, int, int]:
        cfg = self.config
        ts = datetime.combine(D, dtime(*cfg.atc_session))
        day = self._day_index.get(D, {})
        if not day:
            return 0, 0, 0

        nav = self._compute_nav(D)

        # Target shares per ticker (lot-rounded).
        targets: dict[str, int] = {}
        for tk, w in target_weights.items():
            if tk not in day:
                continue
            px = day[tk]["close"]
            if px <= 0:
                continue
            targets[tk] = round_down_to_lot(int((w * nav) / px), 100)

        # Union of target names + currently-held names (held but not targeted → liquidate).
        universe = set(targets) | {t for t in self._held_tickers
                                   if self.inventory.net_shares_at(t, ts) > 0}

        orders: list[tuple[str, OrderSide, int]] = []
        for tk in universe:
            if tk not in day:
                continue                               # no print today → cannot trade
            current = self.inventory.net_shares_at(tk, ts)
            target = targets.get(tk, 0)
            delta = target - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append((tk, side, abs(delta)))

        # Sells first (free cash), then buys.
        orders.sort(key=lambda o: 0 if o[1] == OrderSide.SELL else 1)

        n_fills = n_rej = 0
        for tk, side, qty in orders:
            filled_qty, _px, rejected = self._place_atc_order(D, ts, tk, side, qty)
            if filled_qty > 0:
                n_fills += 1
            elif rejected:
                n_rej += 1

        return len(orders), n_fills, n_rej

    def _place_atc_order(
        self, D: date, ts: datetime, tk: str, side: OrderSide, qty: int,
    ) -> tuple[int, float, bool]:
        """Place one ATC order through the cost model and book the outcome.

        Shared by grid `_execute` and `_tranche_day`.  Returns
        ``(filled_quantity, filled_price, was_rejected)`` — ``(0, 0.0, False)``
        means the order was skipped pre-simulation (no print / unaffordable /
        sub-lot).
        """
        cfg = self.config
        day = self._day_index.get(D, {})
        row = day.get(tk)
        if row is None:
            return 0, 0.0, False                       # no print today → cannot trade

        # Cash guard for buys: never spend cash we don't have.  Two layers:
        #   (1) hard skip when cash is already non-positive (the bug fix —
        #       a negative `self.cash` would otherwise feed a negative qty into
        #       round_down_to_lot and crash the engine);
        #   (2) defensive `max(0, ...)` on the affordable calc so any residual
        #       transient negativity is floored to zero rather than propagated.
        if side == OrderSide.BUY:
            unit = row["close"] * (1.0 + cfg.fee_buffer)
            if unit <= 0 or self.cash <= 0:
                return 0, 0.0, False
            affordable = round_down_to_lot(max(0, int(self.cash / unit)), 100)
            qty = min(qty, affordable)
            if qty < 100:
                return 0, 0.0, False

        vol = row["vol"]
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.02                                  # fallback daily vol
        ref = row["ref_price"]
        if not np.isfinite(ref) or ref <= 0:
            ref = row["close"]
        ref = self._adjusted_reference(D, tk, ref)

        order = Order(
            ticker=tk, side=side, quantity=int(qty),
            target_price=float(row["close"]),
            reference_price=float(ref),
            daily_volume=float(row["volume"]),
            daily_volatility=float(vol),
            exchange=Exchange(str(row["exchange"]).upper()),
            timestamp=ts,
            is_atc=True,
            atc_volume=float(row["volume"]) * cfg.atc_participation,
        )
        fill = self.model.simulate(order, inventory=self.inventory)

        if fill.is_filled:
            self.cash += fill.signed_cash_flow
            if side == OrderSide.BUY:
                self._held_tickers.add(tk)
            self.fills_log.append({
                "date": D.isoformat(), "ticker": tk, "side": side.value,
                "qty": fill.filled_quantity, "price": fill.filled_price,
                "cash_flow": fill.signed_cash_flow,
                "cost": fill.total_cost, "participation": fill.participation_pct,
            })
            return int(fill.filled_quantity), float(fill.filled_price), False

        self.rejections_log.append({
            "date": D.isoformat(), "ticker": tk, "side": side.value,
            "qty": qty, "reason": fill.rejection_reason.value,
        })
        return 0, 0.0, True

    # ── 4b. Tranche-mode daily routine ──────────────────────────────────────
    def _tranche_day(self, i: int, D: date) -> tuple[int, int, int]:
        """One day of the staggered-tranche book.

        1. Barrier scan (when `tranche_pt_sigma`/`tranche_sl_sigma` set):
           flag any position whose close has crossed entry·(1 + pt·σ) or
           entry·(1 − sl·σ) — σ recorded at entry, mirroring the
           triple-barrier label target.
        2. SELL every position of every expired tranche (older than
           `tranche_hold_days`) plus every barrier-flagged position
           (retrying daily until empty — halts, band floors, and ATC volume
           caps just defer the exit).
        3. BUY today's tranche: NAV/`tranche_hold_days` × P(Bull), split
           equally across the day's admitted names. Admission depends on
           `admission_mode`: "cross_sectional" (default) takes the top
           `max_positions` above `signal_threshold`; "absolute_gate" filters to
           `p_up >= admission_floor` first, caps at `admission_pool_cap`, then
           takes the top `max_positions` (empty survivor list ⇒ zero orders,
           budget stays cash, `zero_candidate_days` incremented).

        Position record: {"qty", "entry_price", "entry_vol", "exit_pending"}.
        """
        cfg = self.config
        ts = datetime.combine(D, dtime(*cfg.atc_session))
        day = self._day_index.get(D, {})
        n_orders = n_fills = n_rej = 0

        # 1. Barrier scan — PT/SL exits ahead of the vertical barrier.
        if cfg.tranche_pt_sigma is not None or cfg.tranche_sl_sigma is not None:
            for tranche in self._tranches:
                for tk, pos in tranche["positions"].items():
                    if pos["exit_pending"]:
                        continue
                    row = day.get(tk)
                    if row is None or pos["entry_price"] <= 0:
                        continue
                    ret = row["close"] / pos["entry_price"] - 1.0
                    sigma = pos["entry_vol"]
                    hit_pt = (cfg.tranche_pt_sigma is not None
                              and ret >= cfg.tranche_pt_sigma * sigma)
                    hit_sl = (cfg.tranche_sl_sigma is not None
                              and ret <= -cfg.tranche_sl_sigma * sigma)
                    if hit_pt or hit_sl:
                        pos["exit_pending"] = True

        # 2. Exits (sells first → frees cash for buys): expired tranches in
        #    full, plus barrier-flagged positions in unexpired tranches.
        for tranche in self._tranches:
            expired = (i - tranche["entry_idx"]) >= cfg.tranche_hold_days
            for tk, pos in list(tranche["positions"].items()):
                if not (expired or pos["exit_pending"]):
                    continue
                if pos["qty"] < 1:
                    del tranche["positions"][tk]
                    continue
                n_orders += 1
                filled, _px, rejected = self._place_atc_order(
                    D, ts, tk, OrderSide.SELL, pos["qty"])
                if filled > 0:
                    n_fills += 1
                    remaining = pos["qty"] - filled
                    if remaining > 0:
                        pos["qty"] = remaining           # ATC cap → finish tomorrow
                        pos["exit_pending"] = True
                    else:
                        del tranche["positions"][tk]
                elif rejected:
                    n_rej += 1                           # retry tomorrow
                    pos["exit_pending"] = True
        self._tranches = [t for t in self._tranches if t["positions"]]

        # 3. Today's tranche.
        p_up, tickers = self._inference(D)
        p_up, tickers = self._apply_liquidity_filter(D, p_up, tickers)
        if len(tickers) == 0:
            return n_orders, n_fills, n_rej

        # Ranking rule. `p_up` (default) is what every validated result was
        # earned with. `random` is a deterministic per-day shuffle used as the
        # backtestable PROXY for serve's actual rule: `main._select_candidates`
        # sorts the arbitrated pool by SENTIMENT score descending with p_up only
        # as a tiebreak, and sentiment cannot be backtested (no point-in-time
        # history — that is why the paperlog exists). If random ranks as well as
        # p_up, serve's re-ordering is harmless; if p_up is materially better,
        # serve is discarding the very edge the backtest measured.
        if cfg.rank_mode == "random":
            rng = np.random.default_rng(cfg.rank_seed + D.toordinal())
            order_idx = rng.permutation(len(p_up))
        else:
            order_idx = np.argsort(p_up)[::-1]
        if cfg.admission_mode == "absolute_gate_argmax":
            # Serve's REAL admission: the tau floor AND the arbitrator's class
            # condition. `make_final_decision` returns BUY only when the primary
            # horizon's argmax is UP, and `_select_candidates` dispatches only
            # `final_decision == 2` — so the floor alone (what the backtest
            # validated) is not what production requires.
            _argmax = getattr(self, "_argmax_up_today", {}) or {}
            survivors = [tickers[j] for j in order_idx
                         if p_up[j] >= cfg.admission_floor
                         and _argmax.get(tickers[j], False)][:cfg.admission_pool_cap]
            if not survivors:
                self._zero_candidate_days += 1
                return n_orders, n_fills, n_rej
            admitted, cohort_n = survivors, cfg.max_positions
        elif cfg.admission_mode == "absolute_gate":
            # Serve-mirror admission: ABSOLUTE floor first (inclusive `>=`,
            # matching serve's `meta_gate`), cap the survivor pool at
            # `admission_pool_cap` (serve's `_ARBITRATOR_POOL`), THEN take the
            # top `max_positions` of that capped/filtered list. An empty survivor
            # list deploys nothing (budget stays cash) — mirrors serve's
            # zero-candidate-day fall-to-cash economics.
            survivors = [tickers[j] for j in order_idx
                         if p_up[j] >= cfg.admission_floor][:cfg.admission_pool_cap]
            if not survivors:
                self._zero_candidate_days += 1
                return n_orders, n_fills, n_rej
            admitted, cohort_n = survivors, cfg.max_positions
        elif cfg.admission_mode == "argmax":
            # Admit on the CLASS DECISION (UP is the argmax), not on a P(UP)
            # level. Motivation (11-08-26 paperlog): across the range the
            # absolute gate operates in, P(UP) is anti-informative — the
            # [0.2,0.3) bin realised a 45.9% hit rate vs 29.3% for [0.4,0.5),
            # Platt slope −0.274 — while argmax==UP rows beat the baseline
            # (+1.19% vs −2.43%). Argmax is also immune to the gate-starvation
            # failure mode, since it compares the three classes against each
            # other rather than against a frozen absolute threshold that the
            # output distribution can drift below.
            # Still ranked by p_up within the admitted set, so the cohort takes
            # the strongest of the qualifying names.
            _argmax = getattr(self, "_argmax_up_today", {}) or {}
            survivors = [tickers[j] for j in order_idx
                         if _argmax.get(tickers[j], False)][:cfg.admission_pool_cap]
            if not survivors:
                self._zero_candidate_days += 1
                return n_orders, n_fills, n_rej
            admitted, cohort_n = survivors, cfg.max_positions
        elif cfg.admission_mode == "rank_breadth":
            # NO absolute P(UP) floor at all — always take the best-ranked
            # names available. Only the COUNT (K) responds to market breadth,
            # via the SAME piecewise-linear scalar the exposure brake uses.
            today_breadth = float(self._breadth.get(D, 1.0))  # fail-open
            k_scalar = breadth_scalar(
                today_breadth, cfg.rank_breadth_trigger,
                cfg.rank_breadth_floor_level, cfg.rank_breadth_floor,
            )
            k = round(cfg.max_positions * k_scalar)
            if k <= 0:
                self._zero_candidate_days += 1
                return n_orders, n_fills, n_rej
            admitted, cohort_n = [tickers[j] for j in order_idx], k
        else:
            # "cross_sectional" — UNCHANGED default: top-N above the relative
            # `signal_threshold` floor (byte-for-byte the pre-A/B admission).
            admitted = [tickers[j] for j in order_idx
                        if p_up[j] >= cfg.signal_threshold]
            cohort_n = cfg.max_positions

        # ── SERVE-PARITY DEFENSIVE LAYERS ────────────────────────────────────
        # Applied to the ranked ADMITTED list BEFORE the cohort slice, so a name
        # a filter removes frees its slot for the next-best — the same ordering
        # `main._select_candidates` uses. All no-ops unless explicitly enabled,
        # so every prior result is byte-for-byte unchanged.
        #
        # `raw_qualified` is the pre-filter set the hysteresis streak is tracked
        # on: serve counts consecutive RAW-qualifying days independently of the
        # dedup/sector filters, so a name blocked by those keeps its streak.
        floor = (cfg.admission_floor
                 if cfg.admission_mode.startswith("absolute_gate")
                 else cfg.signal_threshold)
        raw_qualified = [tickers[j] for j in order_idx if p_up[j] >= floor]
        picks = self._apply_serve_layers(admitted, raw_qualified)[:cohort_n]

        if not picks:
            return n_orders, n_fills, n_rej

        p_bull = float(np.clip(self._p_bull.get(D, 1.0), 0.0, 1.0))
        nav = self._compute_nav(D)
        # Burst sizing: budget divisor decoupled from hold length when set
        # (None ⇒ calendar-based nav/hold_days, byte-identical default).
        # Vol-scaled override (22-07-26): a per-day divisor from
        # `self._budget_days` (see `vol_sizing.vol_scaled_budget_days`) wins
        # over the flat `cfg.tranche_budget_days` constant when supplied.
        budget_days = self._budget_days.get(D) or cfg.tranche_budget_days or cfg.tranche_hold_days
        budget = (nav / budget_days) * p_bull
        # Serve's drift+breadth exposure legs (GARCH is already in `p_bull`).
        # 1.0 unless serve_exposure_brake is on, so the default is unchanged.
        budget *= self._serve_exposure_scalar(D)
        if cfg.use_nav_tier_cap:
            # Discrete portfolio cap (risk_tier.py): clamp NEW deployment so
            # total invested (nav − cash) stays under the tier ceiling. Uses
            # the same p_bull the budget already scales by; no per-ticker
            # regime input here (market-wide cap, not a per-name rule).
            tier = classify_risk_tier(p_bull)
            deployed = nav - max(self.cash, 0.0)
            budget = apply_nav_tier_cap(budget, nav, deployed, tier)
        budget = min(budget, max(self.cash, 0.0) / (1.0 + cfg.fee_buffer))
        if budget <= 0:
            return n_orders, n_fills, n_rej
        # per_name divides the budget across the FULL cohort slot count. Under
        # regime sizing, a skipped/penalised name's share is NOT redistributed to
        # survivors — it stays as cash (smaller tranche on bad days = the
        # DD-reducing behaviour). So the denominator stays `len(picks)` — and the
        # prob-weighted map below is likewise frozen over the full pick list.
        if cfg.use_prob_weights:
            _p_by_ticker = {tickers[j]: float(p_up[j]) for j in range(len(tickers))}
            _floor = (cfg.admission_floor if cfg.admission_mode == "absolute_gate"
                      else cfg.signal_threshold)
            _fracs = prob_scaled_weights(
                [_p_by_ticker.get(tk, 0.0) for tk in picks], _floor)
            per_name_map = {tk: budget * f for tk, f in zip(picks, _fracs)}
        else:
            per_name_map = {tk: budget / len(picks) for tk in picks}

        positions: dict[str, dict] = {}
        for tk in picks:
            row = day.get(tk)
            if row is None or row["close"] <= 0:
                continue
            # Regime-conditional sizing (mirrors serve-path src/bot/sizing.py via
            # the shared regime_policy constants):
            #   NO_TRADE {0,7} → skip (its per_name share stays cash);
            #   PENALTY  {1,6} → 0.5× notional (the freed half stays cash).
            # market_regime is absent on minimal panels → `.get` returns None → no-op.
            notional = per_name_map[tk]
            if cfg.use_regime_sizing:
                regime = row.get("market_regime")
                if regime is not None:
                    if regime in NO_TRADE_REGIMES:
                        continue
                    if regime in PENALTY_REGIMES:
                        notional = notional * REGIME_PENALTY_FACTOR
            qty = round_down_to_lot(int(notional / row["close"]), 100)
            if qty < 100:
                continue
            n_orders += 1
            filled, px, rejected = self._place_atc_order(D, ts, tk, OrderSide.BUY, qty)
            if filled > 0:
                n_fills += 1
                vol = row["vol"]
                if not np.isfinite(vol) or vol <= 0:
                    vol = 0.02                           # fallback daily vol
                if tk in positions:
                    positions[tk]["qty"] += filled
                else:
                    positions[tk] = {
                        "qty": filled, "entry_price": px,
                        "entry_vol": float(vol), "exit_pending": False,
                    }
            elif rejected:
                n_rej += 1
        if positions:
            self._tranches.append({"entry_idx": i, "positions": positions})

        return n_orders, n_fills, n_rej

    def _adjusted_reference(self, D: date, ticker: str, raw_ref: float) -> float:
        """
        On a corporate-action ex-date the exchange RESETS the band reference.
        Mirror that so a legitimate ex-date gap is not falsely rejected as
        out-of-band: cash dividend → ref − div ; split → ref ÷ factor.
        """
        ref = raw_ref
        for ev in self._ca_by_date.get(D, []):
            if ev.ticker != ticker:
                continue
            if ev.action_type == CorporateActionType.CASH_DIVIDEND:
                ref = max(ref - ev.cash_per_share, 1.0)
            elif ev.action_type in (CorporateActionType.SPLIT,
                                     CorporateActionType.STOCK_DIVIDEND):
                ref = ref / ev.split_factor
        return ref

    # ── 5. Closing ─────────────────────────────────────────────────────────
    def _compute_nav(self, D: date) -> float:
        ts = datetime.combine(D, dtime(15, 0))         # after the close
        day = self._day_index.get(D, {})
        mv = 0.0
        for tk in self._held_tickers:
            shares = self.inventory.net_shares_at(tk, ts)
            if shares <= 0:
                continue
            if tk in day:
                px = day[tk]["close"]
                self._last_price[tk] = px
            else:
                px = self._last_price.get(tk, 0.0)     # stale mark (halt/delist)
            mv += shares * px
        return mv + self.cash

    def _closing(self, D: date, n_orders: int, n_fills: int, n_rej: int) -> None:
        nav = self._compute_nav(D)
        mv = nav - self.cash
        daily_ret = (nav / self._prev_nav - 1.0) if self._prev_nav > 0 else 0.0
        ts = datetime.combine(D, dtime(15, 0))
        n_pos = sum(1 for tk in self._held_tickers
                    if self.inventory.net_shares_at(tk, ts) > 0)
        self.records.append(DailyRecord(
            date=D, nav=nav, cash=self.cash, market_value=mv,
            daily_return=daily_ret, n_positions=n_pos,
            n_orders=n_orders, n_fills=n_fills, n_rejections=n_rej,
            dividend_cash=self._dividend_cash_today,
            gross_exposure=(mv / nav if nav > 0 else 0.0),
        ))
        self._prev_nav = nav

    # ── Result assembly ────────────────────────────────────────────────────
    def _build_result(self) -> WalkForwardResult:
        eq = pd.DataFrame([r.__dict__ for r in self.records])
        metrics = self._metrics(eq)
        return WalkForwardResult(
            equity_curve=eq,
            fills=self.fills_log,
            rejections=self.rejections_log,
            corporate_action_log=self.ca_log,
            metrics=metrics,
            final_nav=float(eq["nav"].iloc[-1]) if len(eq) else self.cash,
            final_cash=self.cash,
            zero_candidate_days=self._zero_candidate_days,
        )

    def _metrics(self, eq: pd.DataFrame) -> dict:
        if len(eq) < 2:
            return {"n_days": len(eq)}
        r = eq["daily_return"].to_numpy()
        nav = eq["nav"].to_numpy()
        ann_factor = np.sqrt(TRADING_DAYS)
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        sharpe = (mu / sd * ann_factor) if sd > 1e-12 else 0.0
        running_max = np.maximum.accumulate(nav)
        drawdown = nav / running_max - 1.0
        total_ret = float(nav[-1] / self.config.initial_capital - 1.0)
        years = len(eq) / TRADING_DAYS
        cagr = float((nav[-1] / self.config.initial_capital) ** (1 / years) - 1.0) if years > 0 else 0.0
        return {
            "n_days": int(len(eq)),
            "total_return": total_ret,
            "cagr": cagr,
            "ann_sharpe": float(sharpe),
            "ann_vol": float(sd * ann_factor),
            "max_drawdown": float(drawdown.min()),
            "final_nav": float(nav[-1]),
            "n_fills": len(self.fills_log),
            "n_rejections": len(self.rejections_log),
            "total_dividends": float(eq["dividend_cash"].sum()),
        }


