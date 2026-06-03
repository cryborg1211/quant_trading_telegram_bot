import argparse
import html
import json
import logging
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import polars as pl                                          # V3 feature pipeline is polars-native

from config.settings import CONFIG
from src.features.alpha360_generator import Alpha360Generator
from src.features.mr_features import MR_FEATURE_COLUMNS, build_mr_features
from src.features.market_regime import REGIME_LABELS_VI, regime_label_vi
from src.data import price_lookup  # fresh-parquet price lookups (stock_ohlcv retired)
# ─── V3 V1-faithful Tabular Ensemble (locked-in GOLDEN, the only inference route) ─
from src.bot.bot_inference import V3BotInference
from src.bot.sizing import suggested_weight
# Train/serve parity: live inference builds features through the SAME pipeline
# used by train_models.py — single source of truth, no duplicated feature math.
from src.backtest.pipeline import (
    RunConfig as V3FeatureConfig,
    build_features as build_v3_feature_panel,
    FEATURE_RECIPE_VERSION,
)
from src.models.quant_agent_arbitrator import (
    evaluate_trades_batch,
    get_rebalance_advice,
    map_tickers_to_news,
    scrape_centralized_news,
)
from src.trading.portfolio_manager import PortfolioManager
from src.utils.telegram_alerter import TelegramBot, format_source_links

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
# OHLCV crawl guard: 15:00 ICT.
# Vietnamese market officially closes at 15:00. The daily candle is NOT
# considered finalized until then — fetching mid-day:
#   • returns an incomplete (still-moving) close price
#   • adds DB write pressure while reads are happening from the bot path
#   • poisons training data if the new partial candle gets persisted
# This must be 15:00 sharp (NOT 14:45 ATC) to ensure the close is final.
MARKET_CLOSE = dt_time(15, 0)

# ---------------------------------------------------------------------------
# Human-readable feature name mapping for Alpha360 technical indicators
# ---------------------------------------------------------------------------
FEATURE_HUMAN_NAMES: dict[str, str] = {
    # --- Raw Alpha360 OHLCV base keys (used as lag-column base labels) ---
    "close": "Nền giá đóng cửa",
    "open": "Giá mở cửa",
    "high": "Mức đỉnh giá",
    "low": "Mức đáy giá",
    "vwap": "Giá trung bình gia quyền (VWAP)",  # kept for backward compat with old artifacts
    "hlc3": "Giá HLC3 (Trung bình Cao-Thấp-Đóng)",
    "volume": "Khối lượng giao dịch",
    # RSI variants
    "rsi_14": "RSI 14 ngày",
    "rsi_7": "RSI 7 ngày",
    "rsi_21": "RSI 21 ngày",
    # MACD
    "macd": "MACD",
    "macd_signal": "MACD Signal",
    "macd_hist": "MACD Histogram",
    # Bollinger
    "bb_upper": "Bollinger trên",
    "bb_lower": "Bollinger dưới",
    "bb_width": "Độ rộng Bollinger",
    "bb_pct": "% Bollinger",
    # Volume composites
    "volume_ratio_5": "Tỷ lệ khối lượng 5 ngày",
    "volume_ratio_20": "Tỷ lệ khối lượng 20 ngày",
    "volume_ma_5": "Khối lượng TB 5 ngày",
    "volume_ma_20": "Khối lượng TB 20 ngày",
    "obv": "OBV (Khối lượng cân bằng)",
    "obv_ma": "OBV trung bình",
    # Moving averages
    "sma_5": "Đường MA 5 ngày",
    "sma_10": "Đường MA 10 ngày",
    "sma_20": "Đường MA 20 ngày",
    "sma_50": "Đường MA 50 ngày",
    "ema_12": "Đường EMA 12 ngày",
    "ema_26": "Đường EMA 26 ngày",
    # Returns
    "return_1d": "Lợi nhuận 1 ngày",
    "return_5d": "Lợi nhuận 5 ngày",
    "return_20d": "Lợi nhuận 20 ngày",
    # Volatility
    "volatility_5": "Biến động giá 5 ngày",
    "volatility_20": "Biến động giá 20 ngày",
    "atr_14": "ATR 14 ngày",
    # Price ratios
    "close_to_high_52w": "Giá so với đỉnh 52 tuần",
    "close_to_low_52w": "Giá so với đáy 52 tuần",
    # Momentum
    "momentum_10": "Động lượng 10 ngày",
    "roc_10": "Tốc độ thay đổi giá (ROC 10)",
    "williams_r": "Williams %R",
    "cci_20": "CCI 20 ngày",
    # Stochastic
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    # Legacy macro keys (direct column names)
    "vnindex_return": "Biến động VN-Index",
    "usd_vnd_change": "Biến động tỷ giá USD/VND",
    "gold_change": "Biến động giá vàng",
    "oil_change": "Biến động giá dầu",
}

# Professional Vietnamese labels for macro_ prefixed inner keys
# e.g. macro_sp500_close → inner "sp500_close" → looked up here first
_MACRO_INNER_NAMES: dict[str, str] = {
    "sp500_close": "Diễn biến chứng khoán Mỹ (S&P 500)",
    "sp500_return": "Lợi nhuận S&P 500",
    "dxy_close": "Chỉ số sức mạnh đồng USD (DXY)",
    "dxy": "Chỉ số sức mạnh đồng USD (DXY)",
    "usd_vnd": "Tỷ giá USD/VND",
    "usd_vnd_change": "Biến động tỷ giá USD/VND",
    "gold_close": "Giá vàng thế giới",
    "gold_change": "Biến động giá vàng",
    "oil_close": "Giá dầu thô (WTI)",
    "oil_change": "Biến động giá dầu",
    "vix_close": "Chỉ số sợ hãi thị trường (VIX)",
    "vix": "Chỉ số biến động thị trường (VIX)",
    "fed_rate": "Lãi suất Fed",
    "cpi": "Lạm phát (CPI)",
    "vnindex_return": "Biến động VN-Index",
    "vnindex_close": "VN-Index",
    "interbank_rate": "Lãi suất liên ngân hàng",
    "interbank_on_rate": "Lãi suất liên ngân hàng qua đêm (ON)",
    "vnibor": "Lãi suất liên ngân hàng 1 tháng (VNIBOR 1M)",
    "inflation_yoy": "Lạm phát YoY (VN CPI)",
    "deposit_rate": "Lãi suất tiền gửi",
}


# Regex: match trailing numeric suffix (Alpha360 lag index, e.g. close_38, vwap_12)
_NUMERIC_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)$")
# Regex: match macro_ prefix (e.g. macro_sp500_close, macro_vnindex_return)
_MACRO_PREFIX_RE = re.compile(r"^macro_(.+)$", re.IGNORECASE)


def _humanize_feature(feat: str) -> str:
    """
    Convert raw Alpha360/stacking feature name to professional Vietnamese trading label.

    Resolution order:
    1. Exact match in FEATURE_HUMAN_NAMES          (e.g. rsi_14 → "RSI 14 ngày")
    2. Macro prefix via _MACRO_INNER_NAMES          (e.g. macro_sp500_close → "Diễn biến CK Mỹ (S&P 500)")
    3. Explicit _lag_N suffix                       (e.g. rsi_14_lag_3 → "RSI 14 ngày cách đây 3 phiên")
    4. Alpha360 numeric suffix (OHLCV lag columns)  (e.g. vwap_12 → "Giá VWAP cách đây 12 phiên")
    5. Title-case fallback
    """
    # 1. Exact match
    if feat in FEATURE_HUMAN_NAMES:
        return FEATURE_HUMAN_NAMES[feat]

    # 2. Macro prefix — use professional Vietnamese names, not raw title-case
    m2 = _MACRO_PREFIX_RE.match(feat)
    if m2:
        inner = m2.group(1)
        # Try _MACRO_INNER_NAMES first, then FEATURE_HUMAN_NAMES, then title-case fallback
        label = _MACRO_INNER_NAMES.get(inner) or FEATURE_HUMAN_NAMES.get(inner)
        if label:
            return label
        return _MACRO_INNER_NAMES.get(inner, inner.replace("_", " ").title())

    # 3. Explicit _lag_N suffix (e.g. rsi_14_lag_3)
    if "_lag_" in feat:
        base_feat, lag = feat.rsplit("_lag_", 1)
        label = FEATURE_HUMAN_NAMES.get(base_feat, base_feat.replace("_", " ").title())
        return f"{label} cách đây {lag} phiên"

    # 4. Alpha360 numeric suffix (e.g. close_38, vwap_12, norm_vwap_5)
    m = _NUMERIC_SUFFIX_RE.match(feat)
    if m:
        base_feat, lag_idx = m.group(1), m.group(2)
        clean_base = base_feat.removeprefix("norm_")
        label = FEATURE_HUMAN_NAMES.get(clean_base) or FEATURE_HUMAN_NAMES.get(base_feat)
        if not label:
            label = clean_base.replace("_", " ").title()
        return f"{label} cách đây {lag_idx} phiên"

    # 5. Fallback
    return feat.replace("_", " ").title()


def _build_feature_explanation(model: Any, selected_features: list[str], top_k: int = 3) -> tuple[str, str]:
    """
    Use available tree-model feature_importances_ as SHAP-lite fallback for live alerts.
    Produces stable, human-readable drivers instead of raw column names.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return "Không có feature importance từ mô hình", "Theo dõi rủi ro tin tức/vĩ mô"

    arr = np.asarray(importances, dtype=np.float64)
    if arr.size == 0 or len(selected_features) == 0:
        return "Không có feature importance từ mô hình", "Theo dõi rủi ro tin tức/vĩ mô"

    n = min(arr.size, len(selected_features))
    arr = arr[:n]
    feats = selected_features[:n]
    order = np.argsort(arr)[::-1]

    top_positive = [_humanize_feature(feats[i]) for i in order[:top_k] if arr[i] > 0]
    top_risk = [_humanize_feature(feats[i]) for i in order[-min(top_k, n):][::-1]]

    pos_text = ", ".join(top_positive) if top_positive else "Không có động lực nổi bật"
    risk_text = ", ".join(top_risk) if top_risk else "Không có yếu tố rủi ro nổi bật"
    return f"Động lực: {pos_text}", f"Rủi ro: {risk_text}"


_VN_PRICE_SCALE_THRESHOLD = 1_000.0  # VN stocks quoted in thousands; raw < 1000 → multiply by 1000


def _get_live_exec_prices(latest_df: Any, tickers: list[str]) -> dict[str, float]:
    """
    Extract unscaled latest market prices from live feature frame.

    VN market convention: prices are stored in thousands (e.g. 10.5 = 10,500 VND).
    If extracted price < 1,000 we multiply by 1,000 to restore full VND value.
    """
    price_col = next((c for c in ("raw_close", "close", "price") if c in latest_df.columns), None)
    if price_col is None:
        LOGGER.warning("No raw price column found in latest_df. Portfolio/alerts will skip missing prices.")
        return {}

    out: dict[str, float] = {}
    for ticker in tickers:
        rows = latest_df[latest_df["ticker"].astype(str) == ticker]
        if rows.empty:
            continue
        price = float(rows.iloc[-1][price_col])
        if not np.isfinite(price) or price <= 0:
            continue
        # VN market stores prices in thousands (e.g. 10.5 → 10,500 VND)
        if price < _VN_PRICE_SCALE_THRESHOLD:
            price = price * 1_000.0
        out[ticker] = price
    return out


def _format_sentiment_status(sentiment_data: dict[str, Any]) -> str:
    """Differentiate true neutral from no-news/timeout fallback."""
    source_urls = sentiment_data.get("source_urls", []) or []
    score = float(sentiment_data.get("sentiment_score", 0.0) or 0.0)
    reason = str(sentiment_data.get("reasoning_vi", ""))

    if len(source_urls) == 0 and score == 0.0:
        if "timeout" in reason.lower() or "không có tin" in reason.lower() or "no news" in reason.lower():
            return "Không có tin tức / Timeout"
        return "Không có tin tức / Timeout"

    if score > 0.2:
        label = "Tích cực"
    elif score < -0.2:
        label = "Tiêu cực"
    else:
        label = "Trung tính"
    return f"{label} ({score:+.2f})"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=False,
    )
    return logging.getLogger(__name__)


LOGGER = setup_logging()


@contextmanager
def timed_step(message: str):
    start = time.perf_counter()
    LOGGER.info("%s started...", message)
    try:
        yield
    finally:
        LOGGER.info("%s finished in %.2fs.", message, time.perf_counter() - start)


def is_crawl_allowed(force_crawl: bool = False) -> bool:
    """Return True iff the OHLCV crawl may run.

    Skips with a clear log if current VN local time is before `MARKET_CLOSE`
    (15:00 ICT). The caller is expected to continue straight into inference
    using whatever data is already in the DuckDB / parquet store rather than
    pulling fresh (and incomplete) candles.

    `force_crawl=True` bypasses the guard — for operator-initiated rebuilds.
    """
    now = datetime.now(VN_TZ)
    if force_crawl:
        LOGGER.warning("force_crawl=True. Market-hour crawl guard bypassed.")
        return True
    if now.time() < MARKET_CLOSE:
        LOGGER.warning(
            "[Crawler] Skipped OHLCV fetch. Current time is before %s. "
            "Using existing DB data.",
            MARKET_CLOSE.strftime("%H:%M"),
        )
        LOGGER.info(
            "[Crawler] Current VN local time=%s (threshold=%s ICT).",
            now.strftime("%H:%M:%S %Z"),
            MARKET_CLOSE.strftime("%H:%M"),
        )
        return False
    return True


def crawl_hose(
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    data_dir: str = "data",
    force_crawl: bool = False,
    days_back: int | None = None,
) -> None:
    """Overnight-safe HOSE OHLCV crawl.

    `days_back` (INCREMENTAL mode): when set, only the last `days_back` calendar
    days are fetched — `start_date` is overridden to (today − days_back) in VN
    time.  Use `--days-back 1` for a previous-day-only daily refresh; a small
    window (3–5) adds overlap against late data corrections.  When None, a FULL
    crawl from `start_date` runs.
    """
    from src.data.crawlers import StockCrawler

    if not is_crawl_allowed(force_crawl=force_crawl):
        return

    if days_back is not None:
        start_date = (datetime.now(VN_TZ) - timedelta(days=int(days_back))).strftime("%Y-%m-%d")
        LOGGER.info("Incremental crawl | last %d day(s) → start_date=%s", int(days_back), start_date)

    LOGGER.info("Starting Quant V6 HOSE overnight crawler...")
    LOGGER.info("Start date: %s", start_date)
    LOGGER.info("End date: %s", end_date or datetime.now().strftime("%Y-%m-%d"))
    LOGGER.info("Data dir: %s", data_dir)
    LOGGER.info("Errors: logs/crawler_errors.txt")

    crawler = StockCrawler()
    with timed_step("HOSE overnight crawl"):
        summary = crawler.crawl_hose_overnight(
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
        )

    LOGGER.info("crawl_hose completed. Summary=%s", summary)


# (RETIRED: build_alpha360() + the Alpha360 feature factory were removed in the
#  parquet-first migration. The V4 pipeline recomputes features from raw OHLCV
#  via src/backtest/pipeline.build_features and never reads an Alpha360 matrix.)


# ─── Mean-Reversion (knife-catch) sub-model — shared live scorer ──────────
# Loaded once and cached (module-level) so /verify, /suggest_sell and
# /suggest_buy all reuse one in-memory model. Returns, per ticker:
#   {"prob": float, "fired": bool, "tau": float}
# `fired` == panic alert (prob >= the strict τ* from training).
_MR_MODEL: Any = None
_MR_TAU: float | None = None
_MR_ART = Path("models/mr")


def _load_mr() -> tuple[Any, float]:
    global _MR_MODEL, _MR_TAU
    if _MR_MODEL is None:
        _MR_MODEL = joblib.load(_MR_ART / "mr_lgbm.joblib")
        with (_MR_ART / "mr_threshold.json").open("r", encoding="utf-8") as fh:
            _MR_TAU = float(json.load(fh)["tau"])
        LOGGER.info("[MR] sub-model loaded | strict τ*=%.2f", _MR_TAU)
    return _MR_MODEL, float(_MR_TAU)


def mr_score_tickers(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Live MR panic score per ticker (leak-free oversold features).

    Loads an 80-bar OHLCV tail (≥ SMA50 history), runs build_mr_features,
    takes the LATEST row per ticker, and scores it with mr_lgbm. Any
    failure degrades to {} so the caller's primary flow is never broken.
    """
    tickers = sorted({t.upper().strip() for t in tickers if t})
    if not tickers:
        return {}
    try:
        model, tau = _load_mr()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[MR] model unavailable (%s) — MR features skipped.", exc)
        return {}
    try:
        win = Alpha360Generator()._load_live_stock_window(
            tickers=tickers, window_rows=80
        )
        pdf = win.to_pandas() if hasattr(win, "to_pandas") else win
        feat = build_mr_features(pdf)
        latest = (
            feat.sort_values(["ticker", "date"])
            .groupby("ticker", sort=False)
            .tail(1)
        )
        x = (
            latest[MR_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        proba = model.predict_proba(x)[:, 1]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[MR] scoring failed (%s) — MR features skipped.", exc)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for tk, p in zip(latest["ticker"].astype(str), proba, strict=False):
        out[tk.upper()] = {
            "prob": float(p),
            "fired": bool(p >= tau),
            "tau": float(tau),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# V3 TABULAR ENSEMBLE — drop-in replacement for the legacy V6 stacker
# ─────────────────────────────────────────────────────────────────────────────
# Loads the GOLDEN-config joblib bundle written by run_backtest.py and
# exposes the SAME 5-tuple contract as the legacy `predict_stacking_horizon`,
# so the four call sites in this file (daily_inference, inference_for_holdings,
# verify, rebalance_portfolio) need only swap the function name.

# DUAL-HORIZON ARTIFACT REGISTRY ─────────────────────────────────────────────
# Each horizon has its own GOLDEN bundle written by run_backtest.py
# (the artifact filename embeds the horizon to prevent cross-horizon overwrite).
# Loader is cached per horizon so the bot keeps a hot in-memory copy of each.
_V3_BUNDLE_PATHS: dict[int, Path] = {
    5:  Path("models/saved/v3_ensemble_5d.joblib"),
    20: Path("models/saved/v3_ensemble_20d.joblib"),
}
# Legacy fallback for the un-suffixed bundle from V3.0 runs.
_V3_BUNDLE_LEGACY_FALLBACK = Path("models/saved/v3_ensemble.joblib")
_V3_BOT_CACHE: dict[int, V3BotInference] = {}


def _load_v3_bot(horizon: int = 5) -> V3BotInference:
    """Lazy-load + cache the V3 GOLDEN bundle for `horizon` ∈ {5, 20}.

    Looks for `models/saved/v3_ensemble_{horizon}d.joblib`.  If absent AND
    the horizon-agnostic legacy `v3_ensemble.joblib` exists, falls back to it
    (so V3.0 single-horizon installs keep working).  Raises a clear error if
    neither file is present — the operator must run
    `python train_models.py --tb-horizon <h>` then `python run_backtest.py`
    to produce the missing bundle.
    """
    if horizon in _V3_BOT_CACHE:
        return _V3_BOT_CACHE[horizon]
    path = _V3_BUNDLE_PATHS.get(int(horizon))
    if path is None or not path.exists():
        if _V3_BUNDLE_LEGACY_FALLBACK.exists():
            LOGGER.warning(
                "V3 horizon=%d bundle not found at %s — falling back to "
                "horizon-agnostic legacy %s.  Run `python train_models.py "
                "--tb-horizon %d` then `python run_backtest.py` to produce the proper artifact.",
                horizon, path, _V3_BUNDLE_LEGACY_FALLBACK, horizon,
            )
            path = _V3_BUNDLE_LEGACY_FALLBACK
        else:
            raise FileNotFoundError(
                f"V3 horizon={horizon} artifact not found at {path}. "
                f"Run `python train_models.py --tb-horizon {horizon}` then "
                f"`python run_backtest.py` first."
            )
    bot = V3BotInference.from_artifact(path)

    # ── FEATURE-RECIPE TRIPWIRE ──────────────────────────────────────────────
    # The artifact's structural feature recipe MUST match the live pipeline's.
    # A mismatch means `build_features` changed shape (a new/removed/retuned
    # feature, window, or ordering) since this model was trained → its live
    # inputs no longer mean what the model learned (silent train/serve skew).
    # Refuse to serve a drifted model; demand a retrain.
    artifact_recipe = bot.metadata.get("feature_recipe_version")
    if artifact_recipe is None:
        # Backward-compat: artifacts trained before the stamp existed.  Assume
        # compatible — the operator is expected to retrain to stamp the version.
        LOGGER.warning(
            "V3 horizon=%d artifact has no feature_recipe_version (pre-tripwire "
            "build) — assuming compatible with pipeline %s.  Retrain to stamp it.",
            horizon, FEATURE_RECIPE_VERSION,
        )
    elif artifact_recipe != FEATURE_RECIPE_VERSION:
        raise RuntimeError(
            f"FEATURE-RECIPE MISMATCH (horizon={horizon}): the artifact was trained "
            f"with feature_recipe_version={artifact_recipe!r}, but the live pipeline "
            f"is {FEATURE_RECIPE_VERSION!r}.  build_features has changed shape since "
            f"this model was trained, so its live features no longer match what the "
            f"model expects (train/serve skew).  RETRAIN: `python train_models.py "
            f"--tb-horizon {horizon}` then `python run_backtest.py` to regenerate {path}."
        )

    _V3_BOT_CACHE[int(horizon)] = bot
    LOGGER.info("V3 ensemble loaded (horizon=%d, recipe=%s): %s",
                horizon, artifact_recipe or "unstamped", bot.card())
    return bot


# ── V4.0 sizing note: Kelly math lives in `src/bot/sizing.py` (suggested_weight).
#    run_trade_execution computes the half-Kelly size for EACH dispatched ticker
#    from its own P(UP) and shows it ON that ticker's card.  There is NO summary
#    table (the old format_kelly_section_html block was purged).


# Latest market_regime (0–7) per ticker, refreshed on every _compute_v3_features
# pass.  Read by run_trade_execution for regime-aware sizing + the Telegram card.
# (Decoupled from the prediction return-tuple so the 4 call sites stay untouched.)
_LATEST_REGIME_BY_TICKER: dict[str, int] = {}


def _compute_v3_features(latest_df: pd.DataFrame, feature_list: list[str],
                         frac_diff_d: float) -> pd.DataFrame:
    """Build today's live feature row via the SHARED training pipeline.

    Train/serve parity is structural: this calls
    `src.backtest.pipeline.build_features` — the EXACT recipe `train_models.py`
    uses — so FracDiff, the cross-sectional Gaussian-rank Z-scores, anti-FOMO
    over-extension, the alpha factors AND the 5 advanced statistical features are
    all computed identically.  We then project to `feature_list` (the iron-fist
    selection persisted in the artifact's `tabular_features`), replaying the
    train-time selection rather than hardcoding a feature set.

    Input:   pandas DataFrame with (ticker, date, open, high, low, close,
             volume) + per-ticker history for the rolling windows (the Alpha360
             live path provides 120 bars).
    Output:  pandas DataFrame indexed by ticker, columns = `feature_list` in the
             model's trained order, one row per ticker (latest decision-bar).
    """
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(latest_df.columns)
    if missing:
        raise ValueError(f"_compute_v3_features: missing OHLCV columns: {missing}")

    ohlcv = pl.from_pandas(latest_df[list(required)].copy()).sort(["ticker", "date"])
    # SAME recipe as training, with the EXACT frac_diff_d the artifact was trained
    # with (threaded down from the bundle metadata — never a library default).
    # The iron-fist SELECTION is replayed via feature_list.
    panel, all_features, _ = build_v3_feature_panel(
        ohlcv, V3FeatureConfig(frac_diff_d=frac_diff_d))

    unknown = [f for f in feature_list if f not in all_features]
    if unknown:
        raise ValueError(
            f"_compute_v3_features: the artifact requires features the live "
            f"pipeline does not produce: {unknown}. Train/serve recipe drift — "
            f"retrain against the current src/backtest/pipeline.py.")

    # Stash today's market_regime per ticker (the panel ALWAYS carries it, even if
    # a pre-regime artifact's feature_list does not) for regime-aware sizing + the
    # Telegram card.  One cheap tail(1) group-by on the already-built panel.
    if "market_regime" in panel.columns:
        _reg = (panel.sort(["ticker", "date"])
                     .group_by("ticker", maintain_order=True)
                     .tail(1)
                     .select(["ticker", "market_regime"]))
        _LATEST_REGIME_BY_TICKER.update({
            str(t): int(v)
            for t, v in zip(_reg["ticker"].to_list(), _reg["market_regime"].to_list())
            if v is not None
        })

    return (
        panel.sort(["ticker", "date"])
             .group_by("ticker", maintain_order=True)
             .tail(1)                                  # today's decision-bar per ticker
             .select(["ticker"] + list(feature_list))  # artifact's trained order
             .to_pandas()
             .set_index("ticker")
             .dropna()
    )


def predict_v3_horizon(latest_df: pd.DataFrame, horizon: int = 5) -> tuple[
    dict[str, list[float]], dict[str, Any], Any, list[str], dict[str, bool]
]:
    """DROP-IN replacement for `predict_stacking_horizon(latest_df, horizon)`.

    Returns the SAME 5-tuple shape — the four call sites in main.py and the
    downstream `_build_feature_explanation`, arbitrator, and meta-gate logic
    all keep working with no change:

        ( predictions_dict,   # {ticker: [p_down, p_flat, p_up]}
          thresholds,         # {'pnl_threshold_tau': ..., 'meta_labeler_enabled': False, ...}
          model_obj,          # V3BotInference (exposes .feature_importances_ + .feature_names_in_)
          selected_features,  # the artifact's iron-fist `tabular_features` (train/serve parity)
          meta_gate )         # {ticker: True if P(UP) >= up_threshold else False}

    The `horizon` parameter is accepted for API parity.  V3 was trained at the
    horizon stored in the bundle (T+5 for the current GOLDEN); other horizons
    return the SAME T+5 signal with a one-time warning so the arbitrator's
    5d-vs-20d compare degrades gracefully instead of crashing.
    """
    # Route to the horizon-specific bundle (5d ↔ v3_ensemble_5d.joblib, 20d ↔
    # v3_ensemble_20d.joblib).  _load_v3_bot is cached per horizon so dual-
    # horizon bot endpoints share their loaded models across calls.
    bot = _load_v3_bot(horizon)
    trained_h = int(bot.metadata.get("tb_horizon", horizon))
    if horizon != trained_h:
        LOGGER.warning(
            "predict_v3_horizon: bundle reports tb_horizon=%d but caller asked "
            "for %d.  Using the bundle as-is.", trained_h, horizon,
        )

    feature_list = list(bot.tabular_features)          # iron-fist selection from the artifact
    # RECIPE LOCK: conform to the artifact's training feature hyper-params, not
    # library defaults.  frac_diff_d is the only config-driven knob in
    # build_features; older artifacts without it fall back to the pipeline default.
    frac_diff_d = float(bot.metadata.get("frac_diff_d", V3FeatureConfig().frac_diff_d))
    feats = _compute_v3_features(latest_df, feature_list, frac_diff_d)
    if feats.empty:
        LOGGER.warning("predict_v3_horizon: V3 feature panel is empty after dropna.")
        return ({}, {"pnl_threshold_tau": bot.up_threshold}, bot, feature_list, {})

    proba_3 = bot.predict_proba_3class(feats)              # DataFrame indexed by ticker
    predictions_dict: dict[str, list[float]] = {
        str(t): [float(r["p_down"]), float(r["p_flat"]), float(r["p_up"])]
        for t, r in proba_3.iterrows()
    }
    thresholds: dict[str, Any] = {
        "pnl_threshold_tau": float(bot.up_threshold),
        "meta_labeler_enabled": False,
        "round_trip_cost": 0.008,                          # canonical VN cost (matches V6)
        "method": "v3_tabular_ensemble",
        "horizon_days": trained_h,
    }
    # V3 has no separate meta-labeler: the gate IS the up_threshold.
    meta_gate: dict[str, bool] = {
        str(t): bool(proba_3.loc[t, "p_up"] >= bot.up_threshold)
        for t in proba_3.index
    }
    return predictions_dict, thresholds, bot, feature_list, meta_gate


# ─────────────────────────────────────────────────────────────────────────────
# Legacy V6 stacker chain (load_stacking_artifacts / aligned_proba /
# predict_stacking_horizon) + _LATEST_5D_BREAKDOWN cache were PURGED in the
# V3.2 refactor.  All four call sites in this file (daily_inference,
# inference_for_holdings, verify_single_ticker, rebalance_portfolio) route
# through `predict_v3_horizon` above, which loads the per-horizon V3
# TabularEnsemble bundle (models/saved/v3_ensemble_{5,20}d.joblib).
# ─────────────────────────────────────────────────────────────────────────────


_REPORT_SEPARATOR = "\n\n══════════════════════════════\n\n"


def _build_combined_report(signal_data_list: list[dict]) -> str:
    """Concatenate per-ticker Telegram messages into one HTML chat report.

    Reuses `TelegramBot._build_message()` (already HTML-escapes every dynamic
    field) so the combined string is safe to send with `parse_mode=HTML`.

    Returns "" when the list is empty so callers can short-circuit on
    "no signals" with a single truthiness check.
    """
    if not signal_data_list:
        return ""
    parts = [TelegramBot._build_message(sd) for sd in signal_data_list]
    return _REPORT_SEPARATOR.join(parts)


# ---------------------------------------------------------------------------
# TD-05: RL prediction logging + T+5 outcome backfill
# ---------------------------------------------------------------------------
# Before this fix, the `rl_mistake_logs` table was being filled with rows
# whose `actual_t5_outcome` was a hardcoded `-0.05` — making the entire
# table useless for Phase-3 RL training. The corrected flow is two-phase:
#
#   T0:  INSERT row with actual_t5_outcome = NULL for every high-confidence
#        UP prediction. We don't know the outcome yet.
#   T+5: UPDATE every NULL row whose predicted_date is ≥5 days in the past,
#        looking up the actual close prices from stock_ohlcv and computing
#        (t5_close - t0_close) / t0_close.
#
# Both phases run inside `run_trade_execution` on every daily_inference
# call, so backfill happens automatically on the next session after a
# 5-day window elapses.

_RL_UP_CONFIDENCE_THRESHOLD: float = 0.6  # only log strong UP predictions
_RL_HORIZON_DAYS: int = 5                 # match the model's 5d horizon


def _log_rl_predictions(
    db: Any,
    predictions_5d: dict[str, list[float]],
    top_pos_features_text: str,
) -> int:
    """Phase 1 (T0): record every high-confidence UP prediction with NULL outcome.

    The actual outcome is filled in by `_backfill_rl_outcomes` once ≥5
    trading days have elapsed. Returns the number of rows inserted today.

    TD-50: each INSERT runs under `db._audit_lock` so concurrent writers
    (the bot's audit-log thread + this RL thread) serialize cleanly
    instead of relying on DuckDB's internal mutex alone.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    logged = 0
    for ticker, probs in predictions_5d.items():
        if not probs or len(probs) < 3:
            continue
        if probs[2] <= _RL_UP_CONFIDENCE_THRESHOLD:
            continue
        features_json = json.dumps(
            {
                "model": "stacking_gbdt_5d",
                "p_up": round(float(probs[2]), 4),
                "top_drivers": top_pos_features_text,
            },
            ensure_ascii=False,
        )
        with db._audit_lock:
            db.conn.execute(
                """
                INSERT INTO rl_mistake_logs
                (ticker, predicted_date, predicted_action, actual_t5_outcome, features_snapshot)
                VALUES (?, ?, 'BUY', NULL, ?)
                """,
                [ticker, today_str, features_json],
            )
        logged += 1
    return logged


def _backfill_rl_outcomes(db: Any) -> int:
    """Phase 2 (T+5+): fill in `actual_t5_outcome` for rows where predicted_date
    is ≥5 days in the past and outcome is still NULL.

    For each pending row, looks up:
        • t0_close = stock_ohlcv close on `predicted_date`
                     (or the most recent trading day on/before it — handles
                      weekends/holidays defensively)
        • t5_close = first available close from stock_ohlcv whose date is
                     ≥ predicted_date + 5 days

    Computes `(t5_close - t0_close) / t0_close` as the actual % return
    and UPDATEs the row. Rows with missing OHLCV (delisted, illiquid)
    stay NULL and will be retried on the next pipeline run.

    Returns the number of rows backfilled in this run.

    TD-50: the UPDATE write runs under `db._audit_lock` so concurrent
    writers serialize. The two SELECT lookups are read-only and don't
    need the lock — DuckDB's internal mutex handles read safety.
    """
    pending = db.conn.execute(
        f"""
        SELECT ticker, predicted_date, predicted_action
        FROM rl_mistake_logs
        WHERE actual_t5_outcome IS NULL
          AND predicted_date <= CURRENT_DATE - INTERVAL {_RL_HORIZON_DAYS} DAY
        """
    ).fetchall()

    backfilled = 0
    for ticker, predicted_date, action in pending:
        # Price lookups now hit the FRESH parquet vintage (the same source the
        # model trains/serves on) via price_lookup — the stale stock_ohlcv table
        # was retired. Semantics unchanged:
        #   t0 = close on/before predicted_date (walks back over weekends);
        #   t5 = first close on/after predicted_date + horizon.
        t0_close = price_lookup.close_on_or_before(ticker, predicted_date, conn=db.conn)
        t5_close = price_lookup.close_on_or_after(
            ticker, predicted_date + timedelta(days=_RL_HORIZON_DAYS), conn=db.conn
        )
        if t0_close is None or t5_close is None or t0_close <= 0:
            continue

        actual = (t5_close - t0_close) / t0_close
        # TD-50: UPDATE is the WRITE — lock it.
        with db._audit_lock:
            db.conn.execute(
                """
                UPDATE rl_mistake_logs
                SET actual_t5_outcome = ?
                WHERE ticker = ? AND predicted_date = ? AND predicted_action = ?
                  AND actual_t5_outcome IS NULL
                """,
                [actual, ticker, predicted_date, action],
            )
        backfilled += 1
    return backfilled


# VN30 constituents — STRICT live universe gate (avoid noisy mid-caps → false
# signals in weak markets). Update on the quarterly VN30 review.
_VN30_UNIVERSE: frozenset[str] = frozenset({
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
})


def daily_inference(
    window_rows: int = 120,
    max_candidates: int = 6,
    broadcast: bool = True,
    horizon: int = 5,
) -> str:
    """Daily trading path. No crawling. No full Alpha360 parquet load.

    Args:
        window_rows: per-ticker rows to load for live Alpha360 features.
        max_candidates: Top-N pool size sent to the arbitrator (default 6).
        broadcast: If True (default, cron path), push per-ticker alerts to
            every chat ID in `TELEGRAM_CHAT_ID` env. If False (bot on-demand
            path) suppress those pushes — caller is responsible for routing
            the returned report to its own chat. This prevents duplicate
            alerts when `/suggest_buy` triggers `daily_inference()` from the
            interactive bot.

    Returns:
        Combined HTML report of the dispatched Top-3 BUY signals (one block
        per ticker, separated by a horizontal rule). Suitable for posting to
        Telegram with `parse_mode=HTML`. Returns "" when no signals were
        produced (e.g. liquidity filter cleared the pool, or no live prices).
    """
    LOGGER.info("Starting dual-horizon daily inference. No crawling will run in this task.")
    total_start = time.perf_counter()

    with timed_step(f"Building live Alpha360 features from recent OHLCV/macro windows ({window_rows} rows/ticker)"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    LOGGER.info("Live feature frame loaded: %s rows x %s cols.", len(latest_df), len(latest_df.columns))
    if latest_df.empty:
        raise ValueError("Live feature frame is empty.")

    # DUAL-HORIZON: the user-selected `horizon` drives the primary signal +
    # the Kelly sizing block, while the OTHER horizon is fetched for the
    # arbitrator cross-check.  Variable names `_5d` / `_20d` are kept for
    # back-compat with the downstream arbitrator / sentiment code, but they
    # now mean "primary" / "secondary".
    stacking_predictions_5d, thr_5d, xgb_model_5d, selected_features_5d, meta_gate_5d = predict_v3_horizon(latest_df, int(horizon))
    # SECONDARY horizon = arbitrator cross-check ONLY; it must NEVER abort the
    # PRIMARY command.  If the OTHER brain's artifact is missing or version-
    # mismatched, degrade to an empty dict (evaluate_trades_batch falls back to
    # the primary probs per-ticker).  This is the fix for /suggest_buy5 dying
    # with "horizon=20 not found" when only the T+5 model was trained.
    _secondary_h = 20 if int(horizon) == 5 else 5
    try:
        stacking_predictions_20d, _, _, _, _ = predict_v3_horizon(latest_df, _secondary_h)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning(
            "Secondary horizon T+%d unavailable (%s) — running PRIMARY T+%d only; "
            "arbitrator loses its dual-horizon cross-check.",
            _secondary_h, exc.__class__.__name__, int(horizon),
        )
        stacking_predictions_20d = {}

    # ── V4.0 KELLY SIZING ─────────────────────────────────────────────────
    # Per-ticker half-Kelly size is computed from EACH dispatched ticker's OWN
    # P(UP) inside run_trade_execution (src/bot/sizing.suggested_weight) and
    # shown on its card.  No pre-filtered lookup map: the old map gated at
    # P>=0.50 + top-5, so dispatched names below that threshold rendered "N/A".

    sorted_preds = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=True)
    top_10_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in sorted_preds[:10]])
    LOGGER.info("[StackingGBDT] TOP 10 UP PROBS: %s", top_10_str)

    bottom_3_sorted = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=False)[:3]
    bottom_3_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in bottom_3_sorted])
    LOGGER.info("[StackingGBDT] BOTTOM 3 RISK: %s", bottom_3_str)

    # --- Universe Gate: STRICT VN30 (avoid noisy mid-caps → false signals) ---
    # Replaced the ADDV liquidity filter: only the 30 VN30 constituents reach the
    # model gate / arbitrator. Filter the predicted tickers strictly to this set.
    liquid_tickers: set[str] = {
        str(t) for t in stacking_predictions_5d if str(t).upper() in _VN30_UNIVERSE
    }
    LOGGER.info("[VN30Gate] %s / %s predicted tickers are in VN30.",
                len(liquid_tickers), len(stacking_predictions_5d))
    if not liquid_tickers:
        LOGGER.warning("[VN30Gate] no predicted ticker in VN30 — using all predictions as fallback.")
        liquid_tickers = set(stacking_predictions_5d.keys())

    # --- Universe = VN30 gate output ------------------------------------
    # (Old UniverseFilter / CONFIG.universe_filter.exclude_vn30 block DELETED —
    # it EXCLUDED VN30, which directly conflicts with the hardcoded VN30-only
    # gate above. The VN30 frozenset IS the universe now.)
    universe_tickers: set[str] = set(liquid_tickers)

    # Send Top-6 liquid tickers to arbitrator/sentiment layer.
    # 6 candidates → full LLM sentiment evaluation → final Top-3 selected by sentiment+quant score.
    #
    # TASK 3 GATE: a ticker only reaches the arbitrator/dispatch if BOTH the
    # primary model is bullish (P(UP) >= τ*) AND the meta-labeler predicts the
    # trade is profitable net of the 0.8% friction (P(profit) >= 0.5). This is
    # the single execution choke point — nothing un-approved can be traded.
    _ARBITRATOR_POOL = 6
    _gated_out = [
        t for t, _p in sorted(stacking_predictions_5d.items(),
                               key=lambda i: i[1][2], reverse=True)
        if t in universe_tickers and not meta_gate_5d.get(t, True)
    ]
    candidate_tickers = [
        ticker
        for ticker, _probs in sorted(
            stacking_predictions_5d.items(),
            key=lambda item: item[1][2],
            reverse=True,
        )
        if ticker in universe_tickers and meta_gate_5d.get(ticker, True)
    ][: min(max_candidates, _ARBITRATOR_POOL)]
    LOGGER.info(
        "[Brain] Meta-labeler gate: %s liquid tickers rejected (e.g. %s). "
        "Top-%s survivors → arbitrator pool: %s",
        len(_gated_out), _gated_out[:5], len(candidate_tickers), candidate_tickers,
    )

    # --- FALLBACK OBSERVABILITY MODE ------------------------------------
    # If NO ticker cleared the τ*/meta-labeler gates, the bot would return an
    # empty message — useless for monitoring. Instead, fall back to the Top-3
    # liquid-universe tickers by P(UP) PURELY FOR OBSERVABILITY. These are
    # NOT trade signals: run_trade_execution is bypassed entirely below, so
    # there is zero portfolio / RL / dispatch side effect. The Telegram
    # report is explicitly flagged (Vietnamese) as "do not trade".
    fallback_mode = False
    fallback_reasons: dict[str, str] = {}
    if not candidate_tickers:
        fallback_mode = True
        _tau5 = 0.45   # FORCED weak-market safe floor (ignore artifact's stale 0.50 tau)
        _ranked = [
            t for t, _p in sorted(
                stacking_predictions_5d.items(),
                key=lambda kv: kv[1][2], reverse=True,
            )
            if t in universe_tickers
        ]
        candidate_tickers = _ranked[:3]
        _floor_pct = _tau5 * 100.0
        for t in candidate_tickers:
            _pu = stacking_predictions_5d[t][2] * 100.0
            if _pu < _floor_pct:
                fallback_reasons[t] = (
                    f"Cửa tăng chỉ {_pu:.0f}% (dưới ngưỡng an toàn "
                    f"{_floor_pct:.0f}%) — không bõ công đánh đổi với rủi ro "
                    f"thị trường chung đang yếu."
                )
            elif not meta_gate_5d.get(t, True):
                fallback_reasons[t] = (
                    "Cửa tăng tạm ổn nhưng kỳ vọng lợi nhuận không đủ bù "
                    "chi phí và rủi ro — bộ lọc an toàn loại bỏ."
                )
            else:
                fallback_reasons[t] = (
                    f"Chỉ để theo dõi: thị trường yếu, cửa tăng {_pu:.0f}%."
                )
        LOGGER.warning(
            "[FallbackObservability] No gated candidates — MONITORING-ONLY "
            "Top-3 by P(UP): %s (tau*=%.2f). These will NOT be traded.",
            candidate_tickers, _tau5,
        )

    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}
    with timed_step("Evaluating candidates with dual-horizon arbitrator/sentiment layer"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, candidate_tickers)

    # Fallback returns the flagged Vietnamese observability report and
    # BYPASSES run_trade_execution — guaranteeing no trades are executed.
    if fallback_mode:
        if not candidate_tickers:
            return (
                "<b>[⚠️ THỊ TRƯỜNG YẾU]</b>\n<i>Không có mã nào trong vũ trụ "
                "giao dịch sau bộ lọc thanh khoản/Universe hôm nay.</i>"
            )
        # MR knife-catch scores for the monitored names → [🔪 BẮT ĐÁY] tag.
        mr_scores = mr_score_tickers(candidate_tickers)
        fb_prices = _get_live_exec_prices(latest_df, candidate_tickers)
        report_html = _build_fallback_observability_report_vi(
            candidate_tickers,
            stacking_predictions_5d,
            all_sentiments,
            fallback_reasons,
            mr_scores,
            live_prices=fb_prices,
        )
        LOGGER.warning(
            "[FallbackObservability] Returning monitoring-only report for %s "
            "— run_trade_execution SKIPPED (no trades).", candidate_tickers,
        )
        LOGGER.info(
            "Dual-horizon daily inference completed in %.2fs.",
            time.perf_counter() - total_start,
        )
        return report_html

    # --- Top-6 → Top-3 Sentiment Filter ---
    # Scope strictly to candidate_tickers (the 6 evaluated by arbitrator+LLM).
    # Sort primarily by sentiment_score DESC, secondarily by 5d quant probability DESC.
    # Only candidates that passed arbitration (decision == 2) are eligible for dispatch.
    _BUY_DECISION = 2
    evaluated_buys = [
        t for t in candidate_tickers
        if final_decisions.get(t) == _BUY_DECISION
    ]
    top_buy_signals = sorted(
        evaluated_buys,
        key=lambda t: (
            float(all_sentiments.get(t, {}).get("sentiment_score", 0.0)),  # primary  ↓ desc
            float(stacking_predictions_5d.get(t, [0, 0, 0])[2]),          # secondary ↓ desc
        ),
        reverse=True,
    )[:3]

    # Log full ranking for auditability
    _rank_str = " | ".join(
        f"{t} [sent={all_sentiments.get(t,{}).get('sentiment_score',0.0):+.2f}"
        f" quant={stacking_predictions_5d.get(t,[0,0,0])[2]*100:.1f}%]"
        for t in sorted(
            candidate_tickers,
            key=lambda t: (
                float(all_sentiments.get(t, {}).get("sentiment_score", 0.0)),
                float(stacking_predictions_5d.get(t, [0, 0, 0])[2]),
            ),
            reverse=True,
        )
    )
    LOGGER.info("[Brain] Sentiment-ranked pool (Top6): %s", _rank_str)
    LOGGER.info("[Brain] Top-3 Buy Signals after sentiment filter: %s", top_buy_signals)

    report_html = run_trade_execution(
        top_buy_signals=top_buy_signals,
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        stacking_predictions=horizon_predictions,
        latest_df=latest_df,
        xgb_model_5d=xgb_model_5d,
        selected_features_5d=selected_features_5d,
        horizon=int(horizon),
        broadcast=broadcast,
    )
    LOGGER.info("Dual-horizon daily inference completed in %.2fs.", time.perf_counter() - total_start)
    return report_html


def _smart_truncate(text: str, limit: int = 300) -> str:
    """Word-aware truncation that never splits a word in half.

    Operates on RAW text — callers must `html.escape()` the RESULT, never the
    other way round — so it is impossible to sever an HTML entity (the
    `html.escape(...)[:500]` anti-pattern cut `&amp;` → `&am`, which Telegram
    rejects with a parse error).  Returns `text` unchanged when within `limit`;
    otherwise trims to the last whole word and appends a single-glyph ellipsis.
    """
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    word_cut = cut.rsplit(" ", 1)[0].rstrip()   # back off to the last whole word
    return (word_cut if word_cut else cut.rstrip()) + "…"


def _build_fallback_observability_report_vi(
    fallback_tickers: list[str],
    stacking_predictions_5d: dict,
    all_sentiments: dict,
    fallback_reasons: dict,
    mr_scores: dict | None = None,
    live_prices: dict | None = None,
) -> str:
    """Vietnamese 'weak-market' observability report (HTML, the bot's
    Telegram parse mode).

    These tickers are MONITORING ONLY — not trade signals. The caller
    (`daily_inference`) returns this BEFORE `run_trade_execution`, so there
    is no portfolio / RL / dispatch side effect. The header and per-ticker
    flag make it impossible to mistake these for actionable BUYs.
    """
    out = [
        "<b>[⚠️ CẢNH BÁO: THỊ TRƯỜNG XẤU - KHÔNG CÓ ĐIỂM MUA AN TOÀN]</b>",
        "<i>⛔ KHÔNG GIAO DỊCH — danh sách dưới đây chỉ để theo dõi thị "
        "trường. Hôm nay không mã nào đủ tốt để vào lệnh.</i>",
        "",
    ]
    for i, t in enumerate(fallback_tickers, 1):
        p = stacking_predictions_5d.get(t, [0.0, 0.0, 0.0])
        p_dn, p_sd, p_up = p[0] * 100.0, p[1] * 100.0, p[2] * 100.0
        s = all_sentiments.get(t, {}) or {}
        try:
            score = float(s.get("sentiment_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        reason_vi = (
            s.get("reasoning_vi")
            or s.get("reasoning")
            or "Chưa có tin tức đáng chú ý."
        )
        why = fallback_reasons.get(
            t, "Cửa tăng quá thấp so với rủi ro thị trường chung."
        )
        # Trend gate rejected ALL of these (that's why we're in fallback),
        # so the only actionable tag here is whether the knife-catch
        # sub-model fired on extreme panic.
        mr = (mr_scores or {}).get(t) or {}
        tag = " [🔪 BẮT ĐÁY]" if mr.get("fired") else ""
        _px = (live_prices or {}).get(t)
        price_str = f"{_px:,.0f} VND" if _px else "N/A"
        out += [
            f"<b>{i}. {html.escape(t)}{tag}</b>",
            f"   • <b>Giá hiện tại:</b> {html.escape(price_str)}",
            f"   • <b>Đánh giá xu hướng (5 ngày tới):</b> "
            f"Cửa Tăng <b>{p_up:.1f}%</b> | Đi Ngang {p_sd:.1f}% | "
            f"Cửa Giảm {p_dn:.1f}%",
            f"   • <b>Trạng thái:</b> ❌ HỦY BỎ TÍN HIỆU"
            + ("  →  🔪 <b>nhưng MR phát hiện vùng bắt đáy!</b>"
               if mr.get("fired") else ""),
            f"   • <b>Lý do:</b> {html.escape(why)}",
            f"   • <b>Tin tức &amp; Tâm lý:</b> {html.escape(_smart_truncate(reason_vi, 800))}",
            f"   • {format_source_links(s.get('source_urls', []) or [])}",
            "",
        ]
    # Fix the footer-collision bug: guarantee exactly ONE blank line
    # (clean \n\n) between the last sentiment block and the footer.
    while out and out[-1] == "":
        out.pop()
    out.append("")
    out.append(
        "<i>Đây là chế độ theo dõi khi thị trường yếu — KHÔNG phải khuyến "
        "nghị MUA. Hệ thống sẽ tự động báo khi có điểm mua an toàn.</i>"
    )
    return "\n".join(out)


def run_trade_execution(
    top_buy_signals: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    latest_df: Any,
    xgb_model_5d: Any,
    selected_features_5d: list[str],
    horizon: int = 5,
    broadcast: bool = True,
) -> str:
    """Execute portfolio updates, RL outcome logging, and dispatch Telegram alerts.

    Args:
        stacking_predictions: dual-horizon dict {"5d": {...}, "20d": {...}} produced by the
            Stacking GBDT (XGBoost+LightGBM+CatBoost → logistic meta) model.
        horizon: primary horizon (5 or 20) — rendered as the "T+{h} Model"
            label at the top of each card. The half-Kelly position size is
            computed per-ticker from its own P(UP) (suggested_weight) below.
        broadcast: When False, the per-ticker push alert via
            `TelegramBot.send_signal_alert()` is skipped. The combined HTML
            report is still built and returned. Used by the interactive bot
            path (`/suggest_buy`) to avoid duplicating alerts.

    Returns:
        Combined HTML report of every signal dispatched (for chat-reply UX),
        or "" if no signals were dispatched.
    """
    LOGGER.info("Starting Trade Execution (Portfolio Manager)...")
    dispatched_signals: list[dict] = []
    try:
        manager = PortfolioManager()
        live_exec_prices = _get_live_exec_prices(latest_df, top_buy_signals)

        if not live_exec_prices:
            LOGGER.warning("No executable live prices available. Skipping portfolio price updates and alerts.")
            return ""

        with timed_step("Portfolio update/process_daily_trades"):
            manager.update_live_performance(live_exec_prices)
            manager.process_daily_trades(
                top_buy_signals=top_buy_signals,
                next_day_open_prices=live_exec_prices,
                predictions=final_decisions,
            )

        # TD-05: replaced the hardcoded `actual_t5_outcome = -0.05` stub with
        # a proper two-phase logger:
        #   (1) at T0 — INSERT a row per high-confidence UP prediction with
        #               actual_t5_outcome = NULL (we don't know the outcome yet).
        #   (2) at T+5+ — UPDATE any old NULL rows by looking up real close
        #                 prices from stock_ohlcv and computing the actual return.
        # Both phases run every daily_inference call, so backfill happens on the
        # next session after a 5-day window elapses for any given prediction.
        with timed_step("RL prediction logging (T0 INSERT + T+5 backfill UPDATE)"):
            db = manager.db
            predictions_5d = stacking_predictions.get("5d", stacking_predictions)
            top_pos_text, _ = _build_feature_explanation(
                xgb_model_5d, selected_features_5d, top_k=3
            )
            logged = _log_rl_predictions(db, predictions_5d, top_pos_text)
            backfilled = _backfill_rl_outcomes(db)
            LOGGER.info(
                "RL T0 logged=%s (UP prob > %.2f); T+5 backfilled=%s",
                logged, _RL_UP_CONFIDENCE_THRESHOLD, backfilled,
            )

        LOGGER.info("Dispatching Telegram Alerts...")
        bot = TelegramBot()
        sent = 0
        top_pos_features, top_neg_features = _build_feature_explanation(
            xgb_model_5d,
            selected_features_5d,
            top_k=3,
        )

        for ticker in top_buy_signals:
            exec_price = live_exec_prices.get(ticker)
            if exec_price is None:
                LOGGER.warning("Skipping Telegram alert for %s: no live market price.", ticker)
                continue

            sentiment_data = all_sentiments.get(ticker, {})
            # Cap raw URL list at 6 (>=5 sources); pass as list so the Telegram formatter loops explicitly
            source_urls: list[str] = (sentiment_data.get("source_urls", []) or [])[:6]
            _p5 = stacking_predictions.get("5d", {}).get(ticker, [0, 0, 0])
            confidence_5d = round(_p5[2] * 100, 2)
            _regime = _LATEST_REGIME_BY_TICKER.get(ticker)            # 0–7, or None pre-retrain
            LOGGER.info("[Alert] %s source_urls=%s regime=%s", ticker, source_urls, _regime)

            signal_data = {
                "action": "MUA",
                "ticker": ticker,
                "price": f"{exec_price:,.0f} VND",
                "horizon_label": f"T+{int(horizon)}",                  # card header → "T+5 Model"
                # Regime-aware half-Kelly: regime 0/7 → 0% (stand aside), 1/6 → ≤10% cap.
                "suggested_weight": suggested_weight(float(_p5[2]), market_regime=_regime),
                "market_regime": _regime,                              # int 0–7 or None
                "regime_label": regime_label_vi(_regime),              # VN label for the card
                # Plain-VN trend split for the new card
                "prob_up": round(_p5[2] * 100, 1),
                "prob_side": round(_p5[1] * 100, 1),
                "prob_down": round(_p5[0] * 100, 1),
                # Single integrated analytical paragraph — the card renders only this.
                "conclusion": sentiment_data.get(
                    "reasoning_vi", "Chưa có dữ liệu tin tức đáng kể."),
                "sentiment_score": sentiment_data.get("sentiment_score", 0.0),
                "sentiment_status": _format_sentiment_status(sentiment_data),
                "gemini_summary": sentiment_data.get("reasoning_vi", "Không có tin tức đáng kể."),
                "article_urls": source_urls,          # raw list → Telegram formatter loops this
                "confidence": confidence_5d,
                "top_pos_features": top_pos_features,
                "top_neg_features": top_neg_features,
            }

            if broadcast:
                bot.send_signal_alert(signal_data)
            dispatched_signals.append(signal_data)
            sent += 1
        LOGGER.info("Telegram alerts dispatched: %s (broadcast=%s)", sent, broadcast)

    except Exception:
        LOGGER.exception("Error during trade execution")
        return _build_combined_report(dispatched_signals)

    return _build_combined_report(dispatched_signals)


# ---------------------------------------------------------------------------
# /suggest_sell — on-demand inference for an arbitrary ticker list
# ---------------------------------------------------------------------------

_SELL_DECISION = 0  # arbitrator class label for DOWN / SELL


_MR_SELL_VETO = (
    "⚠️ <b>[CẢNH BÁO BÁN ĐÚNG ĐÁY: Mã này đang rơi vào vùng hoảng loạn "
    "tột độ, xác suất cao sẽ có nhịp hồi chữ V. Hạn chế bán tháo lúc "
    "này!]</b>"
)


def _build_sell_hold_report(
    holding_tickers: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    live_exec_prices: dict,
    missing_tickers: list[str] | None = None,
    mr_scores: dict | None = None,
) -> str:
    """Build the HTML BÁN/GIỮ digest for /suggest_sell.

    Every ticker passed in `holding_tickers` (that has a prediction) gets one
    block. `missing_tickers` lists holdings that could not be evaluated
    (delisted, illiquid, or absent from the live universe).
    """
    predictions_5d = stacking_predictions.get("5d", {})
    parts: list[str] = []

    for ticker in holding_tickers:
        if ticker not in predictions_5d:
            continue

        decision = final_decisions.get(ticker)
        sentiment = all_sentiments.get(ticker, {})
        sentiment_score = float(sentiment.get("sentiment_score", 0.0) or 0.0)
        sentiment_status = _format_sentiment_status(sentiment)
        reasoning = _smart_truncate(str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể.")), 400)
        confidence_5d = round(predictions_5d.get(ticker, [0, 0, 0])[2] * 100, 2)
        price = live_exec_prices.get(ticker)
        price_str = f"{price:,.0f} VND" if price else "N/A"

        if decision == _SELL_DECISION:
            verdict = "🔴 <b>NÊN BÁN</b>"
        elif decision == 2:
            verdict = "🟢 <b>GIỮ TIẾP (xu hướng còn tăng)</b>"
        else:
            verdict = "🟡 <b>GIỮ THẬN TRỌNG (đang đi ngang)</b>"

        # ── MR VETO ──────────────────────────────────────────────────
        # The trend model says SELL, but the knife-catch model fired:
        # the stock is in extreme capitulation with a high V-bounce
        # probability. Explicitly warn the user NOT to dump into the low.
        mr = (mr_scores or {}).get(ticker) or {}
        veto_line = ""
        if decision == _SELL_DECISION and mr.get("fired"):
            veto_line = f"{_MR_SELL_VETO}\n"

        # Plain-language target / trailing-stop from the standing risk rules.
        tp = CONFIG.trading.take_profit_pct   # e.g. +0.15
        sl = CONFIG.trading.stop_loss_pct     # e.g. -0.07
        if price:
            target_str = f"{price * (1.0 + tp):,.0f} VND (+{tp * 100:.0f}%)"
            stop_str = f"{price * (1.0 + sl):,.0f} VND ({sl * 100:.0f}%)"
        else:
            target_str = stop_str = "N/A"

        source_urls = (sentiment.get("source_urls", []) or [])[:6]
        if source_urls:
            url_lines = "\n".join(f"  • {html.escape(u)}" for u in source_urls)
        else:
            url_lines = "  • Không có tin tức đáng kể."

        block = (
            f"📌 <b>{html.escape(ticker)}</b> — giá hiện tại {html.escape(price_str)}\n"
            f"{veto_line}"
            f"• <b>Khuyến nghị:</b> {verdict}\n"
            f"• <b>Đánh giá xu hướng (5 ngày tới):</b> Cửa Tăng "
            f"<b>{confidence_5d}%</b>\n"
            f"• 🎯 <b>Mục tiêu chốt lời:</b> {target_str}\n"
            f"• 🛡️ <b>Ngưỡng cắt lỗ:</b> {stop_str}\n"
            f"• <b>Tin tức &amp; Tâm lý:</b> {html.escape(sentiment_status)} — "
            f"{html.escape(reasoning)}\n"
            f"• <b>Nguồn tham khảo:</b>\n{url_lines}"
        )
        parts.append(block)

    header = (
        f"💼 <b>[HỆ THỐNG] BÁN/GIỮ DANH MỤC</b>\n"
        f"📅 <b>Ngày:</b> {datetime.now().strftime('%d/%m/%Y')}\n"
        f"══════════════════════════════"
    )

    if not parts:
        body = "<i>Không có ticker nào trong danh mục được mô hình đánh giá.</i>"
    else:
        body = _REPORT_SEPARATOR.join(parts)

    footer = ""
    if missing_tickers:
        escaped_missing = html.escape(", ".join(missing_tickers))
        footer = (
            f"\n\n⚠️ <i>Không có dữ liệu live cho:</i> "
            f"<code>{escaped_missing}</code>"
        )

    return f"{header}\n\n{body}{footer}"


def inference_for_holdings(
    holding_tickers: list[str],
    window_rows: int = 120,
) -> str:
    """Run dual-horizon Stacking GBDT + arbitrator on the user's holdings only.

    Skips the liquidity gate and the Top-6/Top-3 funnel — every holding gets
    a recommendation. Used by the /suggest_sell bot command.

    Args:
        holding_tickers: tickers from the `portfolio` DuckDB table.
        window_rows: per-ticker rows for live Alpha360 features.

    Returns:
        HTML report (HTML-escaped) suitable for `parse_mode=HTML`. Empty
        string if no live features or no overlap with the live universe.
    """
    if not holding_tickers:
        return ""

    holding_tickers = sorted({t.upper().strip() for t in holding_tickers if t})
    LOGGER.info("Holdings inference for %s tickers: %s", len(holding_tickers), holding_tickers)
    total_start = time.perf_counter()

    with timed_step(f"Building live Alpha360 features ({window_rows} rows/ticker) for /suggest_sell"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    if latest_df.empty:
        LOGGER.warning("[/suggest_sell] live feature frame is empty.")
        return ""

    universe = set(latest_df["ticker"].astype(str).tolist())
    present = [t for t in holding_tickers if t in universe]
    missing = [t for t in holding_tickers if t not in universe]
    if missing:
        LOGGER.warning("[/suggest_sell] holdings absent from live universe: %s", missing)

    if not present:
        # Build an "empty" report that still warns about missing tickers.
        return _build_sell_hold_report(
            holding_tickers=[],
            final_decisions={},
            all_sentiments={},
            stacking_predictions={},
            live_exec_prices={},
            missing_tickers=missing,
        )

    # CROSS-SECTIONAL PARITY: predict over the FULL universe (do NOT slice to the
    # held set first, or the `_xsz` ranks degenerate); the predictions dict is
    # keyed by ticker, so the arbitrator / exec-price / MR steps below read off
    # `present`.  SECONDARY horizon (20d) is arbitrator cross-check only → non-fatal.
    stacking_predictions_5d, _, _, _, _ = predict_v3_horizon(latest_df, 5)
    try:
        stacking_predictions_20d, _, _, _, _ = predict_v3_horizon(latest_df, 20)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("[/suggest_sell] T+20 cross-check unavailable (%s) — using 5d only.",
                       exc.__class__.__name__)
        stacking_predictions_20d = {}
    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}

    with timed_step("Holdings arbitrator + sentiment scoring"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, present)

    live_exec_prices = _get_live_exec_prices(latest_df, present)

    # MR knife-catch scores → drives the "don't sell into the bottom" veto.
    with timed_step("Holdings MR (knife-catch) scoring"):
        mr_scores = mr_score_tickers(present)

    LOGGER.info("[/suggest_sell] completed in %.2fs.", time.perf_counter() - total_start)
    return _build_sell_hold_report(
        holding_tickers=present,
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        stacking_predictions=horizon_predictions,
        live_exec_prices=live_exec_prices,
        missing_tickers=missing,
        mr_scores=mr_scores,
    )


# ---------------------------------------------------------------------------
# /verify — single-ticker ad-hoc analysis (rumor / news verification)
# ---------------------------------------------------------------------------

# Class-label → display text mappings for the verify report.
_VERIFY_5D_PRED_LABELS: dict[int, str] = {
    0: "🔴 Giảm (DOWN)",
    1: "🟡 Đi ngang (SIDE)",
    2: "🟢 Tăng (UP)",
}
_VERIFY_20D_PRED_LABELS: dict[int, str] = {
    0: "Giảm",
    1: "Đi ngang",
    2: "Tăng",
}
_VERIFY_VERDICT_LABELS: dict[int, str] = {
    0: "🔴 <b>BÁN (SELL)</b>",
    1: "🟡 <b>GIỮ (HOLD)</b>",
    2: "🟢 <b>MUA / GIỮ (BUY/HOLD)</b>",
}


def _mr_state_line(mr_state: dict | None) -> str:
    """Plain-VN MR bottom-catch state for the /verify dual output."""
    if not mr_state:
        return "🔪 <b>Trạng thái Bắt đáy:</b> Không khả dụng"
    if mr_state.get("fired"):
        return (
            "🔪 <b>Trạng thái Bắt đáy:</b> "
            "🚨 <b>CẢNH BÁO HOẢNG LOẠN</b> — cổ phiếu đang ở vùng bán tháo "
            "cực đoan, xác suất cao có nhịp hồi chữ V."
        )
    return (
        "🔪 <b>Trạng thái Bắt đáy:</b> Chưa đạt "
        "(chưa rơi vào vùng hoảng loạn tột độ)"
    )


def _build_verify_report(
    ticker: str,
    decision: int | None,
    sentiment: dict,
    stacking_5d: list[float],
    stacking_20d: list[float],
    live_exec_price: float | None,
    mr_state: dict | None = None,
) -> str:
    """Build the HTML verification report for /verify.

    Every dynamic field is `html.escape`d. Structural tags (<b>, <code>, <i>)
    are constants. Output is safe to send with parse_mode=HTML.
    """
    # 5d distribution
    p_down, p_side, p_up = stacking_5d[0], stacking_5d[1], stacking_5d[2]
    pred_5d_idx = max(range(3), key=lambda i: stacking_5d[i])
    pred_5d_label = _VERIFY_5D_PRED_LABELS.get(pred_5d_idx, str(pred_5d_idx))
    confidence_5d = round(stacking_5d[pred_5d_idx] * 100, 2)

    # 20d distribution (compact)
    pred_20d_idx = max(range(3), key=lambda i: stacking_20d[i])
    pred_20d_label = _VERIFY_20D_PRED_LABELS.get(pred_20d_idx, str(pred_20d_idx))
    confidence_20d = round(stacking_20d[pred_20d_idx] * 100, 2)

    # Sentiment
    sent_score = float(sentiment.get("sentiment_score", 0.0) or 0.0)
    sent_status = _format_sentiment_status(sentiment)
    sent_reasoning = _smart_truncate(str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể.")), 600)

    # Final arbitrator verdict (decision integer 0/1/2)
    verdict_html = _VERIFY_VERDICT_LABELS.get(decision, "<i>(chưa có verdict)</i>")

    # Price (VN price normalization already applied by `_get_live_exec_prices`)
    price_str = f"{live_exec_price:,.0f} VND" if live_exec_price else "N/A"

    # Source URLs (already populated from ground-truth tracker in arbitrator)
    source_urls = (sentiment.get("source_urls", []) or [])[:3]
    if source_urls:
        url_lines = "\n".join(f"  - {html.escape(u)}" for u in source_urls)
    else:
        url_lines = "  Không có tin tức đáng kể"

    return (
        f"🔍 <b>[KIỂM ĐỊNH] {html.escape(ticker)}</b>\n"
        f"📅 <b>Ngày:</b> {datetime.now().strftime('%d/%m/%Y')}\n"
        f"══════════════════════════════\n\n"
        f"💵 <b>Giá hiện tại:</b> {html.escape(price_str)}\n\n"
        f"📊 <b>Đánh giá Xu hướng (5 ngày tới)</b>\n"
        f"• Cửa Tăng: <b>{p_up * 100:.1f}%</b> | Đi Ngang: {p_side * 100:.1f}% "
        f"| Cửa Giảm: {p_down * 100:.1f}%\n"
        f"• Nhận định 5 ngày: {pred_5d_label}\n"
        f"• Nhận định 20 ngày: {pred_20d_label} ({confidence_20d}%)\n\n"
        f"{_mr_state_line(mr_state)}\n\n"
        f"📰 <b>Tin tức &amp; Tâm lý</b>\n"
        f"• Đánh giá: {html.escape(sent_status)} (điểm {sent_score:+.2f})\n"
        f"• Phân tích: {html.escape(sent_reasoning)}\n\n"
        f"🎯 <b>Kết luận tổng hợp:</b> {verdict_html}\n\n"
        f"🔗 <b>Nguồn tham khảo:</b>\n{url_lines}"
    )


def verify_single_ticker(ticker: str, window_rows: int = 120) -> str:
    """Run ad-hoc 5d + 20d quant + LLM-sentiment verification for one ticker.

    Used by the /verify Telegram command for rumor / news fact-checks before
    a manual trade decision (e.g., "HPG announced dividends — should I buy?").

    Args:
        ticker: VN equity symbol (case-insensitive; coerced to upper).
        window_rows: per-ticker rows to load for live Alpha360 features.

    Returns:
        HTML report (HTML-escaped) suitable for `parse_mode=HTML`. Empty
        string only if the ticker name is itself empty. Liquidity / data-
        availability failures return a Vietnamese warning HTML message.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return ""

    LOGGER.info("[/verify] Single-ticker analysis: %s", ticker)
    total_start = time.perf_counter()

    # --- Step 1: Build features over the FULL active universe ---
    # CROSS-SECTIONAL PARITY: the alpha features are cross-sectional Gaussian-
    # rank Z-scores (`_xsz`).  Ranking a single ticker against itself collapses
    # every `_xsz` to 0 → garbage prediction.  So we build the WHOLE universe's
    # feature panel (exactly like daily_inference), let `_xsz` materialize over
    # the full cross-section, and slice to `ticker` only from the RESULT below.
    try:
        with timed_step(f"Building live universe OHLCV window for /verify {ticker}"):
            generator = Alpha360Generator()
            # RAW multi-row OHLCV window (full OHLCV + history) for the V3/V4
            # tabular pipeline — not the Alpha360 tail-1 lag row.
            live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
            latest_df = live_pl.to_pandas()
    except FileNotFoundError:
        LOGGER.warning("[/verify] No OHLCV parquet for %s.", ticker)
        return (
            f"⚠️ Mã <b>{html.escape(ticker)}</b> không đủ thanh khoản hoặc "
            f"không có dữ liệu để phân tích.\n"
            f"<i>(Không tìm thấy <code>data/ohlcv_{html.escape(ticker)}.parquet</code> — "
            f"có thể chưa được crawl hoặc đã hủy niêm yết.)</i>"
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] Live feature build failed for %s.", ticker)
        return (
            f"⚠️ Lỗi khi xây feature cho <b>{html.escape(ticker)}</b>: "
            f"<code>{html.escape(str(exc))}</code>"
        )

    if latest_df.empty or ticker not in set(latest_df["ticker"].astype(str)):
        return (
            f"⚠️ Mã <b>{html.escape(ticker)}</b> không đủ thanh khoản hoặc "
            f"không có dữ liệu để phân tích."
        )

    # Do NOT slice latest_df to `ticker` here — inference must see the full
    # universe so the cross-sectional `_xsz` ranks are non-degenerate.  The
    # predictions dict below is keyed by ticker, so we slice the RESULT instead.

    # --- Step 2: Stacking GBDT inference (5d primary + 20d optional cross-check) ---
    try:
        stacking_5d, _, _, _, _ = predict_v3_horizon(latest_df, 5)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] 5d inference failed for %s.", ticker)
        return (
            f"⚠️ Lỗi mô hình Stacking GBDT cho <b>{html.escape(ticker)}</b>: "
            f"<code>{html.escape(str(exc))}</code>"
        )
    # 20d is the secondary cross-check — optional; degrade gracefully if its
    # artifact is missing/mismatched so /verify still shows the 5d view.
    try:
        stacking_20d, _, _, _, _ = predict_v3_horizon(latest_df, 20)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning("[/verify] T+20 cross-check unavailable for %s (%s) — 5d only.",
                       ticker, exc.__class__.__name__)
        stacking_20d = {}

    if ticker not in stacking_5d:
        return (
            f"⚠️ Mô hình không thể dự đoán cho <b>{html.escape(ticker)}</b> "
            f"(thiếu feature hoặc lịch sử quá ngắn)."
        )

    horizon_predictions = {"5d": stacking_5d, "20d": stacking_20d}

    # --- Step 3: Arbitrator (news scrape + Gemini sentiment + decision) ---
    # `evaluate_trades_batch` accepts a candidate ticker list — passing a
    # singleton scopes the whole pipeline (GNews queries, LLM batch, decision)
    # to this one symbol. ~3-5s of news scrape + 1 Gemini call.
    try:
        with timed_step(f"Arbitrator + sentiment for /verify {ticker}"):
            final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, [ticker])
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] Arbitrator failed for %s.", ticker)
        # Fall through with empty sentiment so the user still sees the quant view.
        final_decisions, all_sentiments = {}, {}

    # --- Step 4: Live execution price (VN scaling already applied) ---
    live_exec_prices = _get_live_exec_prices(latest_df, [ticker])

    # --- Step 5: Mean-Reversion (knife-catch) state — parallel sub-model ---
    mr_state = mr_score_tickers([ticker]).get(ticker)

    LOGGER.info("[/verify] %s completed in %.2fs.", ticker, time.perf_counter() - total_start)
    return _build_verify_report(
        ticker=ticker,
        decision=final_decisions.get(ticker),
        sentiment=all_sentiments.get(ticker, {}),
        stacking_5d=list(stacking_5d.get(ticker, [0.33, 0.34, 0.33])),
        stacking_20d=list(stacking_20d.get(ticker, [0.33, 0.34, 0.33])),
        live_exec_price=live_exec_prices.get(ticker),
        mr_state=mr_state,
    )


# ---------------------------------------------------------------------------
# /rebalance — AI portfolio rebalancing advisor
# ---------------------------------------------------------------------------

_REBALANCE_PRED_LABELS: dict[int, str] = {0: "🔴 Giảm", 1: "🟡 Đi ngang", 2: "🟢 Tăng"}


def _build_rebalance_report(holdings_context: list[dict], advice: str) -> str:
    """Build Telegram-safe HTML for the /rebalance command output.

    Every dynamic field is html.escape'd. Structural tags are constants.
    """
    lines = []
    for h in holdings_context:
        ticker = html.escape(str(h.get("ticker", "?")))
        pnl_pct = float(h.get("pnl_pct", 0.0))
        pred_label = html.escape(str(h.get("pred_label", "N/A")))
        sign = "+" if pnl_pct >= 0 else ""
        icon = "🟢" if pnl_pct >= 0 else "🔴"
        lines.append(f"• {icon} <b>{ticker}</b>: {sign}{pnl_pct:.1f}% | {pred_label}")

    holdings_block = "\n".join(lines) if lines else "• Danh mục trống."

    return (
        "<b>⚖️ TƯ VẤN CƠ CẤU DANH MỤC</b>\n"
        "══════════════════════\n"
        f"<b>• Đang nắm giữ:</b>\n{holdings_block}\n\n"
        f"<b>📊 Đề xuất AI:</b>\n{html.escape(advice)}"
    )


def rebalance_portfolio(user_id: str, window_rows: int = 120) -> str:
    """Fetch live positions, run 5d model, scrape news, call Gemini for rebalance advice.

    Reads from the `portfolio` DuckDB table (multi-user, keyed by user_id).
    Returns an HTML report suitable for parse_mode=HTML, or "" for empty portfolio.
    """
    LOGGER.info("[/rebalance] Starting for user_id=%s", user_id)

    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415
        db = DuckDBEngine()
        rows = db.conn.execute(
            "SELECT ticker, price FROM portfolio WHERE user_id = ?",
            [user_id],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/rebalance] DB read failed for user_id=%s", user_id)
        raise

    if not rows:
        return ""

    # Deduplicate: if the same ticker was added multiple times, use the most recent price.
    entry_prices: dict[str, float] = {}
    for ticker_raw, price_raw in rows:
        t = str(ticker_raw).upper().strip()
        entry_prices[t] = float(price_raw or 0)
    held_tickers = sorted(entry_prices)

    LOGGER.info("[/rebalance] Holdings: %s", held_tickers)

    # CROSS-SECTIONAL PARITY: build the FULL active universe so `_xsz`
    # Gaussian-rank Z-scores span the whole cross-section, not just the
    # (possibly tiny) held set.  Predictions are sliced to held names afterwards.
    with timed_step("[/rebalance] Building live universe features"):
        generator = Alpha360Generator()
        # V3/V4 tabular path needs the RAW multi-row OHLCV window (full
        # open/high/low/close/volume + history) — NOT the Alpha360 tail-1 lag
        # row, which drops open/high/low and collapses the time series.
        live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    if latest_df.empty:
        LOGGER.warning("[/rebalance] live feature frame empty for %s", held_tickers)
        return ""

    present = [t for t in held_tickers if t in set(latest_df["ticker"].astype(str))]
    if not present:
        return ""

    # Predict over the FULL universe (correct cross-section), then read off the
    # held names — do NOT filter latest_df to `present` before inference.
    stacking_predictions_5d, _, _, _, _ = predict_v3_horizon(latest_df, 5)
    live_prices = _get_live_exec_prices(latest_df, present)

    holdings_context: list[dict] = []
    for ticker in present:
        entry = entry_prices.get(ticker, 0.0)
        current = live_prices.get(ticker, entry)
        pnl_pct = ((current - entry) / entry * 100.0) if entry > 0 else 0.0
        probs = stacking_predictions_5d.get(ticker, [0.33, 0.34, 0.33])
        pred_idx = int(np.argmax(probs))
        holdings_context.append({
            "ticker": ticker,
            "pnl_pct": pnl_pct,
            "pred_label": _REBALANCE_PRED_LABELS.get(pred_idx, "N/A"),
            "p_up": float(probs[2]),
        })

    raw_news = scrape_centralized_news(target_tickers=present)
    ticker_news_dict, _ = map_tickers_to_news(raw_news, present)
    advice = get_rebalance_advice(holdings_context, ticker_news_dict)

    return _build_rebalance_report(holdings_context, advice)


def full_pipeline(force_crawl: bool = False, days_back: int | None = None) -> None:
    """End-of-day pipeline: OHLCV crawl → LLM sentiment → inference.

    A single `--task full_pipeline` runs the full EOD sequence so the bot is
    primed for tomorrow's session:
      1. ingest the day's HOSE OHLCV (market-hour guarded; `force_crawl` bypasses;
         `days_back` limits to the last N days — e.g. 1 for previous-day-only);
      2. refresh daily LLM news sentiment;
      3. run the daily inference (T+5 Top-3 broadcast).

    Step 1 is the ONLY OHLCV ingestion entrypoint in the bot/CLI; the V4 backtest
    engine (train_models.py / run_backtest.py) reads the populated store and never
    crawls.
    """
    from src.crawlers.sentiment_crawler import update_daily_sentiment

    LOGGER.warning("Running full_pipeline (EOD): OHLCV crawl + sentiment refresh + inference.")

    # 1. EOD OHLCV ingestion (guarded by 15:00 ICT close unless --force-crawl).
    #    days_back=1 (previous-day) keeps the daily refresh incremental.
    crawl_hose(force_crawl=force_crawl, days_back=days_back)

    # 2. Daily LLM news sentiment.
    with timed_step("Fetching daily LLM sentiment"):
        sentiment_df = update_daily_sentiment(db_path="data/quant_v6_core.duckdb")
        LOGGER.info("Fetched %s sentiment records.", len(sentiment_df))

    # 3. Run the daily inference (T+5 Top-3 broadcast). The V4 serve path
    #    recomputes features from raw OHLCV via load_live_ohlcv_window; there is
    #    no Alpha360 feature-matrix build (that path was retired entirely).
    daily_inference()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant V6 orchestration CLI")
    parser.add_argument(
        "--task",
        default="daily_inference",
        choices=["daily_inference", "crawl_hose", "full_pipeline"],
        help="Task to run. daily_inference is the no-crawl live path; "
             "crawl_hose / full_pipeline are the EOD ingestion paths.",
    )
    parser.add_argument("--window-rows", type=int, default=120, help="Rows per ticker for live Alpha360 inference.")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=6,
        help=(
            "Pool size sent to the arbitrator (LLM sentiment evaluation). "
            "The final dispatched count is always the Top-3 of this pool, "
            "ranked by sentiment_score DESC then quant probability DESC. "
            "Default 6 matches the Top-6→Top-3 design."
        ),
    )
    parser.add_argument("--force-crawl", action="store_true",
                        help="Bypass the 15:00 ICT market-hour crawl guard (operator rebuild).")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Incremental crawl: fetch only the last N calendar days "
                             "(1 = previous day). Applies to --task crawl_hose / full_pipeline. "
                             "Omit for a full crawl from the 2016 start date.")
    return parser.parse_args()


def _send_crash_alert(task_name: str, exc: BaseException, tb_text: str) -> None:
    """Best-effort Telegram crash notification. Never raises.

    Used by the top-level `main()` wrapper so a single unhandled exception
    surfaces as a Telegram message before the process exits, instead of
    silently dying in a cron log.

    Args:
        task_name: the --task value (or whatever the operator passed).
        exc: the caught exception instance.
        tb_text: a pre-formatted traceback string (already captured from
            the except block — re-running `format_exc()` after another
            exception would corrupt it).
    """
    try:
        # Telegram caps at 4096 chars per message; reserve ~500 chars for
        # the header, headline, and HTML overhead. Keep the LAST 1500 chars
        # of the traceback because the deepest frame is usually the cause.
        tb_snippet = tb_text[-1500:] if tb_text else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:300]

        # TD-31: include the running code version so we can correlate a
        # crash to a specific deploy without timestamp archaeology.
        from src.utils.version import get_version  # noqa: PLC0415
        version = get_version()

        msg = (
            "🚨 <b>[SYSTEM CRASH] Pipeline Failed!</b>\n"
            f"📅 <b>Time:</b> {html.escape(ts)}\n"
            f"🔖 <b>Version:</b> <code>{html.escape(version)}</code>\n"
            f"⚙️ <b>Task:</b> <code>{html.escape(task_name)}</code>\n"
            f"❌ <b>Exception:</b> <code>{html.escape(exc_type)}</code>\n"
            f"📝 <b>Message:</b> <code>{html.escape(exc_msg)}</code>\n\n"
            f"<b>Traceback (last frames):</b>\n"
            f"<pre>{html.escape(tb_snippet)}</pre>"
        )

        bot = TelegramBot()
        bot.send_text_alert(msg, label=f"crash:{task_name}")
        LOGGER.info("Crash alert dispatched to Telegram chat IDs.")
    except Exception as alert_exc:  # noqa: BLE001
        # The alerter itself failed (token missing, network down, etc.).
        # Log but do NOT re-raise — we want the original exception to be
        # the one that propagates.
        LOGGER.exception("Crash alert dispatch itself failed: %s", alert_exc)


def main() -> None:
    """Entry point with global crash alerting (TD-09).

    Any unhandled exception inside a task body is caught here, formatted
    into a Telegram alert, then re-raised so the cron / shell sees a
    non-zero exit code. KeyboardInterrupt and SystemExit pass through
    unalerted (those are explicit operator actions / nested sys.exit calls).
    """
    # TD-26: install rotating file handler BEFORE anything else logs.
    # Idempotent — safe even if another import already configured logging.
    from src.utils.logging_utils import setup_rotating_logging  # noqa: PLC0415
    setup_rotating_logging()

    # TD-31: capture the running code version (git SHA / VERSION file fallback)
    # for log correlation. Memoized — first call resolves, the rest are O(1).
    from src.utils.version import get_version  # noqa: PLC0415
    version = get_version()

    args = parse_args()
    task_name = args.task or "unknown"

    LOGGER.info("=" * 70)
    LOGGER.info("Quant V6 starting | task=%s pid=%s version=%s", task_name, os.getpid(), version)
    LOGGER.info("=" * 70)

    try:
        if task_name == "crawl_hose":
            crawl_hose(force_crawl=args.force_crawl, days_back=args.days_back)
        elif task_name == "full_pipeline":
            full_pipeline(force_crawl=args.force_crawl, days_back=args.days_back)
        else:
            daily_inference(window_rows=args.window_rows, max_candidates=args.max_candidates)
    except (KeyboardInterrupt, SystemExit):
        # Operator-initiated stop or a nested sys.exit(): no crash alert.
        LOGGER.warning("Pipeline interrupted (KeyboardInterrupt/SystemExit). No crash alert sent.")
        raise
    except Exception as exc:  # noqa: BLE001
        # Catch ALL pipeline exceptions (API timeout, DuckDB lock, Gemini
        # failure, file-not-found, etc.) — alert THEN re-raise so cron sees
        # exit code != 0.
        tb_text = traceback.format_exc()
        LOGGER.error("=" * 70)
        LOGGER.error("Pipeline crashed during task=%s. Dispatching alert before exit.", task_name)
        LOGGER.error(tb_text)
        LOGGER.error("=" * 70)
        _send_crash_alert(task_name, exc, tb_text)
        raise

    LOGGER.info("Quant V6 task=%s completed cleanly.", task_name)


if __name__ == "__main__":
    main()