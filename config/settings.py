from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    alpha360_parquet: Path = Path("data/alpha360_features.parquet")
    macro_parquet: Path = Path("data/macro_daily.parquet")
    duckdb_path: Path = Path("data/quant_v6_core.duckdb")

    def __post_init__(self) -> None:
        self.data_dir = _path(self.data_dir)
        self.models_dir = _path(self.models_dir)
        self.logs_dir = _path(self.logs_dir)
        self.alpha360_parquet = _path(self.alpha360_parquet)
        self.macro_parquet = _path(self.macro_parquet)
        self.duckdb_path = _path(self.duckdb_path)


@dataclass
class ModelConfig:
    lstm_seq_len: int = 20
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_num_classes: int = 3
    lstm_top_k_features: int = 50

    stacking_top_k_features: int = 70
    stacking_feature_selection_max_rows: int = 150_000
    stacking_n_splits: int = 3
    stacking_horizons: list[int] = field(default_factory=lambda: [5, 20])
    stacking_baseline_macro_f1: float = 0.20

    alpha360_lookback: int = 60


@dataclass
class TrainingConfig:
    seed: int = 42
    split_date: str = "2024-01-01"
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30


@dataclass
class TradingConfig:
    timezone: str = "Asia/Ho_Chi_Minh"
    market_close_hour: int = 14
    market_close_minute: int = 45
    stop_loss_pct: float = -0.07
    take_profit_pct: float = 0.15
    fee_rate: float = 0.002
    virtual_allocation_per_ticker: float = 10_000_000.0
    default_telegram_id: str = "default_user"
    # When True, the live dispatch (_dispatch_signals) applies the same
    # regime-conditional sizing the backtest validated: NO_TRADE regimes {0,7}
    # skip the name, PENALTY regimes {1,6} get a 0.5x weight. Settings kill-switch:
    # set "regime_sizing_enabled": false in settings.json + restart to disable.
    regime_sizing_enabled: bool = True
    # Sentiment-entry forward paper-log: when True, every arbitrated candidate's
    # prediction + sentiment + price is logged each pipeline run (daily) and each
    # /verify command (verify) into `sentiment_entry_paperlog`. Pure observability —
    # changes NO trading decision. Settings kill-switch: set
    # "sentiment_entry_enabled": false in settings.json + restart to disable.
    sentiment_entry_enabled: bool = True
    # Analysis-time REFERENCE constant only — the "DOWN & sentiment > threshold"
    # treatment filter applied by scripts/analyze_sentiment_paperlog.py. It does
    # NOT gate captures; the full candidate cross-section is always logged.
    sentiment_entry_threshold: float = 0.7
    # GARCH-HMM macro exposure brake: when True, _dispatch_signals scales each
    # MUA weight by a market-wide exposure multiplier = clip(P(Bull), floor, 1.0)
    # from src/bot/garch_brake.py (GARCH(1,1)+HMM overlay, models/saved/
    # garch_hmm_v4_weights.joblib). Benchmark (seed 0, T+5 bear OOS): regime+garch
    # was the best defense (Sharpe -0.36→+0.005). FAIL-OPEN: any data/model failure
    # returns 1.0 (full exposure) so the live pipeline never breaks. Stacks WITH
    # regime_sizing (complementary: price-regime × macro-breadth). Kill-switch:
    # set "garch_brake_enabled": false in settings.json + restart.
    garch_brake_enabled: bool = True
    garch_brake_floor: float = 0.2
    # Drift brake (19-07-26): the GARCH brake is VOL-triggered and read ~0.99
    # through the whole low-vol July grind-down (−12/13 dispatches red). This
    # companion brake watches trailing WINDOW-session cumulative market-proxy
    # return: ≥ trigger → ×1.0; ≤ full → ×floor; linear in between. Combined
    # exposure = min(garch_scalar, drift_scalar). Fail-open like the GARCH
    # brake. Kill-switch: "drift_brake_enabled": false in settings.json.
    drift_brake_enabled: bool = True
    drift_brake_window: int = 10
    drift_brake_trigger: float = -0.03
    drift_brake_full: float = -0.06
    drift_brake_floor: float = 0.5
    # Breadth brake (20-07-26 meta-controller upgrade): check_drift.py found
    # the July bleed was a BREADTH collapse (trailing 20d UP rate 29.5% vs
    # 41.5% train), not model decay — a direct signal none of the other
    # brakes read. breadth = fraction of the liquid universe with a positive
    # trailing `breadth_brake_window`-session return. First-cut thresholds
    # (not yet backtested — moderate floor 0.5 bounds downside if miscalibrated,
    # matching drift_brake_floor). Kill-switch: "breadth_brake_enabled": false.
    breadth_brake_enabled: bool = True
    breadth_brake_window: int = 20
    breadth_brake_trigger: float = 0.40
    breadth_brake_floor_level: float = 0.25
    breadth_brake_floor: float = 0.5
    # MR-LGBM knife-catch breadth-inflection ANNOTATION (22-07-26, separate
    # from the breadth_brake_* exposure-scalar leg above — different window,
    # different purpose). Informational only: appends whether current market
    # breadth is in the "Low + Rising" bucket the research found had
    # OOF precision 0.667 vs 0.542 for Low+Falling — a REAL but UNCONFIRMED
    # gap (strict hold-out was too thin, 6 vs 8 fires). NEVER gates a trade
    # or the MR fire decision itself — text-only enrichment on the existing
    # 🔪 tag/veto lines, always labeled as an unconfirmed research signal.
    # Kill-switch: "mr_breadth_context_enabled": false in settings.json.
    mr_breadth_context_enabled: bool = True
    mr_breadth_context_window: int = 20
    mr_breadth_context_delta_window: int = 5
    mr_breadth_context_low_cut: float = 0.41
    mr_breadth_context_rising_threshold: float = 0.03
    # Promote-gate for the weekly auto-retrain (20-07-26): retrain moved from
    # manual/occasional to EVERY Saturday (quant_weekly_retrain schtask), so a
    # single bad walk-forward run would otherwise auto-overwrite the live
    # artifact with no human in the loop. run_backtest.py compares the new
    # candidate's embedded OOS metrics against the CURRENT on-disk artifact's
    # own metadata before persisting; a regression beyond these tolerances
    # keeps the incumbent live and stashes the candidate under
    # models/saved/rejected/ for inspection. Kill-switch:
    # "promote_gate_enabled": false in settings.json.
    promote_gate_enabled: bool = True
    promote_gate_min_sharpe_delta: float = -0.10
    promote_gate_max_dd_regression_pp: float = 3.0
    promote_gate_min_up_precision: float = 0.35
    # Prob-scaled tranche cohort weights (plan prob-scaled-tranche-weights_PLAN_20-07-26):
    # scale within-cohort dispatch weights by normalized edge over the serve
    # signal gate (src/trading/cohort_weights.py, capped 2× equal weight).
    # Default OFF — flips ON only after the backtest A/B acceptance gate
    # (Sharpe up, MaxDD not >1pp worse, PBO not worse) passes.
    prob_weighted_cohorts_enabled: bool = False
    # Intraday attack scanner (monitoring-only): a repeating in-bot job that
    # rescores the model on PROVISIONAL intraday bars during HOSE trading hours
    # and sends a compact "top movers" card when something notable moves before
    # the 15:30 EOD crawl. Writes NOTHING to parquet/DuckDB/paperlog, invokes NO
    # arbitrator/Gemini — pure alert-only. Defaults OFF (this is a brand-new
    # always-on background job with a new runtime dependency, so unlike the other
    # kill-switches here it ships disabled; flip it on manually after restart).
    # Kill-switch: set "intraday_scanner_enabled": true in settings.json + restart.
    intraday_scanner_enabled: bool = False
    intraday_scan_interval_min: int = 15   # cadence in minutes; clamped to [10, 30] at job registration
    intraday_alert_delta_pp: float = 0.02  # |ΔP(UP)| >= this (in probability, 0.02 = 2pp) alerts a top-3 name
    # Serve candidate universe mode. "adv_top_n" (DEFAULT ON) resolves the live
    # candidate pool dynamically via src/trading/serve_universe.liquid_universe
    # (top serve_liquid_top_n names by trailing serve_adv_window-day $-ADV) —
    # the SAME universe definition the validated tranche backtest runs under
    # (WalkForwardConfig.liquid_top_n=50, adv_window=20). "vn30" is the
    # kill-switch: it uses the static _VN30_UNIVERSE frozenset in main.py (old
    # behavior) and is ALSO the degrade target. Any OTHER string → treated as
    # "vn30" plus a warning (fail toward the proven-safe old universe, never
    # toward the unvalidated new one). Kill-switch: set "serve_universe_mode":
    # "vn30" in settings.json + restart to disable.
    serve_universe_mode: str = "adv_top_n"
    serve_liquid_top_n: int = 50   # backtest-validated N (WalkForwardConfig.liquid_top_n)
    serve_adv_window: int = 20     # backtest-validated trailing window (WalkForwardConfig.adv_window)
    # Portfolio Guard (EOD, alert-only): after each full_pipeline/inference_only
    # EOD pass, notify_portfolio_guard() scans every non-cron user's /add
    # holdings against five deterministic triggers (hard stop-loss, take-profit,
    # trailing-stop, model-flip-to-SELL, NO_TRADE regime warning) and DMs a
    # protective card ONLY to users with a fired trigger. Writes NOTHING (no
    # parquet/DuckDB/paperlog/ledger), never auto-sells. Kill-switch: set
    # "portfolio_guard_enabled": false in settings.json + restart. When False the
    # guard fully short-circuits (zero DB reads, zero compute).
    portfolio_guard_enabled: bool = True
    # Trailing-stop trigger: alert when today's close has fallen >= this fraction
    # from the position's peak close since entry (0.08 = 8% off the high).
    portfolio_guard_trailing_pct: float = 0.08
    # Stage-2 enrichment gate: when True, a SINGLE arbitrator+sentiment batch runs
    # across the union of all triggered tickers per EOD run (bounded Gemini spend).
    # When False, only the quant/rule triggers run — alerts still ship, LLM-free.
    # Kill-switch: set "portfolio_guard_llm_enabled": false in settings.json + restart.
    portfolio_guard_llm_enabled: bool = True
    # EOD dual-horizon position/verdict report: after each EOD run,
    # notify_position_report() broadcasts every OPEN ledger row's live NET PnL% +
    # sessions-remaining and every row closed that run with a đúng/sai verdict.
    # Alert-only, event-gated (silent when nothing to report). Kill-switch: set
    # "eod_position_report_enabled": false in settings.json + restart to disable.
    # When False the report fully short-circuits (zero DB reads).
    eod_position_report_enabled: bool = True
    # Rolling lookback for the report's CLOSED section: rows closed within the
    # trailing N days (inclusive) are shown, not only those closed today. Lets
    # recently/backfilled-closed suggestions surface for a few days.
    eod_position_report_lookback_days: int = 7
    # /suggest_buy5 and /suggest_buy20 call daily_inference(broadcast=False,
    # persist=False): the persist gate correctly skips PortfolioManager/paperlog
    # writes, but that also means the cron-path signal_ledger.record_dispatch
    # (gated on broadcast) never fires, so these manual picks went completely
    # untracked. When True, the two suggest_buy commands paper-track their
    # delivered picks into dispatched_signals (dedup key (ticker, horizon) per
    # dispatch_date — repeated same-day calls no-op). Card delivery is never
    # blocked by a ledger failure. Kill-switch: set
    # "suggest_buy_ledger_tracking_enabled": false in settings.json + restart.
    suggest_buy_ledger_tracking_enabled: bool = True
    # Cancelled-signal regret tracking: when True, every daily_inference call that
    # lands in the weak-market fallback OR the post-arbitration no-buy monitor logs
    # its rejected Top-3 (probs + Vietnamese reason) into cancelled_signals, and the
    # EOD notify_regret_report() broadcasts a HYPOTHETICAL "if bought anyway" report.
    # SINGLE kill-switch: set "cancelled_signal_tracking_enabled": false in
    # settings.json + restart to disable — gates BOTH the capture writes AND the
    # regret report send (see plan Decision 4).
    cancelled_signal_tracking_enabled: bool = True
    # Rolling lookback (days, inclusive) for notify_regret_report's window — a
    # DEDICATED knob, independently tunable from eod_position_report_lookback_days
    # (plan Decision 8), defaulting to the same 7 for day-one consistency.
    regret_report_lookback_days: int = 7
    # July-2026 incident guards (BSR dispatched 4 consecutive days into a falling
    # knife; 8 of 13 July dispatches were one PVN energy/fertilizer complex).
    # (1) Open-cohort dedup: a ticker with ANY status='OPEN' dispatched_signals
    # row is excluded from NEW candidate pools — no averaging down / re-buying
    # while a cohort is already live. Kill-switch: set
    # "dispatch_open_cohort_dedup_enabled": false in settings.json + restart.
    dispatch_open_cohort_dedup_enabled: bool = True
    # (2) Sector cap: at most N names per sector (src/trading/sector_map.py) in
    # the arbitrator candidate pool; unmapped (OTHER) tickers are uncapped.
    # 0 disables the cap entirely.
    arbitrator_sector_cap: int = 2
    # (3) Admission hysteresis: require N consecutive raw-qualifying days
    # (tracked independent of dedup/sector-cap) before a name is admissible.
    # Kill-switch: "hysteresis_enabled": false in settings.json + restart.
    hysteresis_enabled: bool = True
    hysteresis_min_qualify_days: int = 2
    hysteresis_max_gap_days: int = 4


@dataclass
class CrawlerConfig:
    stock_start_date: str = "2016-01-01"
    macro_start_date: str = "2014-01-01"
    throttle_min_interval_seconds: float = 4.25
    rate_limit_cooldown_seconds: int = 75
    request_retry_total: int = 3
    request_backoff_factor: float = 1.5


@dataclass
class SentimentConfig:
    # GA pin — the floating "gemini-flash-latest" alias is banned (silently
    # became gemini-3.5-flash; caused the 2026-07-07 JSON-drift incident and
    # intermittent 503s). `.env GEMINI_MODEL` overrides this for crawler AND
    # arbitrator alike; settings.json "sentiment.gemini_model" overrides the
    # dataclass default only.
    gemini_model: str = "gemini-2.5-flash"
    rss_lookback_weekday_days: int = 1
    rss_lookback_monday_days: int = 3
    # Cost trim 14-07-26 (user budget): 30→15 tickers, 8→4 GNews results,
    # 4000→2000 chars/article — roughly halves daily crawl volume/tokens.
    max_tickers: int = 15
    gnews_max_results: int = 4
    gnews_sleep_seconds: float = 1.25
    article_char_limit: int = 2000


# (UniverseFilterConfig removed — superseded by the hardcoded VN30 gate
#  `_VN30_UNIVERSE` in main.py. The old exclude_vn30 knob conflicted with it.)


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    @classmethod
    def from_json(cls, path: str | Path = "config/settings.json") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        return cls(
            paths=PathConfig(**raw.get("paths", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            trading=TradingConfig(**raw.get("trading", {})),
            crawler=CrawlerConfig(**raw.get("crawler", {})),
            sentiment=SentimentConfig(**raw.get("sentiment", {})),
        )


CONFIG = Config.from_json()