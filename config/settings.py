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
    gemini_model: str = "models/gemini-flash-latest"
    rss_lookback_weekday_days: int = 1
    rss_lookback_monday_days: int = 3
    max_tickers: int = 30
    gnews_max_results: int = 8
    gnews_sleep_seconds: float = 1.25
    article_char_limit: int = 4000


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