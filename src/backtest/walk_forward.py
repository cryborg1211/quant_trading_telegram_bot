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

    # ── Public entrypoint ──────────────────────────────────────────────────
    def run(
        self,
        panel: pl.DataFrame | pd.DataFrame,
        corporate_actions: Sequence[CorporateActionEvent] | None = None,
        p_bull_series: pd.Series | None = None,
        inference_cache: dict[date, tuple[np.ndarray, list[str]]] | None = None,
    ) -> WalkForwardResult:
        """
        `p_bull_series` — date-indexed HMM P(Bull) (leak-free filtered). Each
        rebalance scales target weights by that day's P(Bull). When None, the
        engine uses 1.0 (full exposure, no soft scaling).

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

        self.cash = self.config.initial_capital
        self._prev_nav = self.config.initial_capital

        for i, D in enumerate(self.calendar):
            self._dividend_cash_today = 0.0

            # 1. MORNING — corporate actions
            self._morning_routine(D)

            # 2–4. REBALANCE (gated by cadence + OOS start; needs seq_len history)
            n_orders = n_fills = n_rej = 0
            after_start = (self.config.start_trading_date is None
                           or D >= self.config.start_trading_date)
            if (i >= self.config.seq_len and after_start
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
                self._day_index.setdefault(row.date, {})[tk] = {
                    "open": row.open, "high": row.high, "low": row.low,
                    "close": row.close, "volume": row.volume,
                    "ref_price": row.ref_price, "vol": row.vol,
                    "adv20": row.adv20,
                    "exchange": row.exchange,
                }

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
        """
        cache = getattr(self, "_inference_cache", None)
        if cache is not None and D in cache:
            p_up_c, tickers_c = cache[D]
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
            if cache is not None:
                cache[D] = (np.array([]), [])
            return np.array([]), []

        X = np.stack(X_list).astype(np.float32)        # (n, seq, F)
        probs = self.oracle(X)
        probs = np.asarray(probs)
        p_up = probs[:, 2] if probs.ndim == 2 else probs.ravel()
        if cache is not None:
            cache[D] = (p_up, tickers)
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
            row = day[tk]

            # Cash guard for buys: never spend cash we don't have.  Two layers:
            #   (1) hard skip when cash is already non-positive (the bug fix —
            #       a negative `self.cash` would otherwise feed a negative qty into
            #       round_down_to_lot and crash the engine);
            #   (2) defensive `max(0, ...)` on the affordable calc so any residual
            #       transient negativity is floored to zero rather than propagated.
            if side == OrderSide.BUY:
                unit = row["close"] * (1.0 + cfg.fee_buffer)
                if unit <= 0 or self.cash <= 0:
                    continue
                affordable = round_down_to_lot(max(0, int(self.cash / unit)), 100)
                qty = min(qty, affordable)
                if qty < 100:
                    continue

            vol = row["vol"]
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.02                              # fallback daily vol
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
                n_fills += 1
            else:
                self.rejections_log.append({
                    "date": D.isoformat(), "ticker": tk, "side": side.value,
                    "qty": qty, "reason": fill.rejection_reason.value,
                })
                n_rej += 1

        return len(orders), n_fills, n_rej

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


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: rejection histogram
# ─────────────────────────────────────────────────────────────────────────────

def rejection_histogram(result: WalkForwardResult) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in result.rejections:
        out[r["reason"]] = out.get(r["reason"], 0) + 1
    return out
