import argparse
import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from catboost import CatBoostClassifier

from src.features.alpha360_generator import Alpha360Generator
from src.models.quant_agent_arbitrator import evaluate_trades_batch
from src.trading.portfolio_manager import PortfolioManager
from src.utils.telegram_alerter import TelegramBot

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
MARKET_CLOSE = dt_time(14, 45)

# ---------------------------------------------------------------------------
# Human-readable feature name mapping for Alpha360 technical indicators
# ---------------------------------------------------------------------------
FEATURE_HUMAN_NAMES: dict[str, str] = {
    # --- Raw Alpha360 OHLCV base keys (used as lag-column base labels) ---
    "close": "Nền giá đóng cửa",
    "open": "Giá mở cửa",
    "high": "Mức đỉnh giá",
    "low": "Mức đáy giá",
    "vwap": "Giá trung bình gia quyền (VWAP)",
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
    now = datetime.now(VN_TZ)
    if force_crawl:
        LOGGER.warning("force_crawl=True. Market-hour crawl guard bypassed.")
        return True
    if now.time() < MARKET_CLOSE:
        LOGGER.warning("[WARNING] Market is still open. Skipping crawl phase to prevent unclosed candle data.")
        LOGGER.warning(
            "Crawl skipped before VN market close (%s ICT). Current time=%s.",
            MARKET_CLOSE.strftime("%H:%M"),
            now.strftime("%H:%M:%S %Z"),
        )
        return False
    return True


def crawl_hose(
    start_date: str = "2016-01-01",
    end_date: str | None = None,
    data_dir: str = "data",
    force_crawl: bool = False,
) -> None:
    """Overnight-safe full HOSE OHLCV crawl."""
    from src.data.crawlers import StockCrawler

    if not is_crawl_allowed(force_crawl=force_crawl):
        return

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


def build_alpha360() -> None:
    LOGGER.info("Starting offline Alpha360 rebuild.")
    with timed_step("Alpha360 full rebuild"):
        generator = Alpha360Generator()
        generator.run()


def load_stacking_artifacts(horizon: int) -> tuple[list[str], dict[str, Any], Any, Any, Any, CatBoostClassifier, Any]:
    artifact_dir = Path("models/stacking") / f"{horizon}d"
    required_artifacts = {
        "selected_features": artifact_dir / "selected_features.json",
        "scaler": artifact_dir / "scaler.joblib",
        "xgboost": artifact_dir / "xgboost_model.joblib",
        "lightgbm": artifact_dir / "lightgbm_model.joblib",
        "catboost": artifact_dir / "catboost_model.cbm",
        "meta_model": artifact_dir / "meta_model.joblib",
        "thresholds": artifact_dir / "quantile_thresholds.json",
    }
    missing = [str(path) for path in required_artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {horizon}d stacking artifacts: {missing}. Run train_stacking.py first.")

    LOGGER.info("Loading %sd stacking artifacts from %s", horizon, artifact_dir)
    with timed_step("Loading model artifacts"):
        with required_artifacts["selected_features"].open("r", encoding="utf-8") as f:
            selected_features = json.load(f)
        with required_artifacts["thresholds"].open("r", encoding="utf-8") as f:
            quantile_thresholds = json.load(f)

        scaler = joblib.load(required_artifacts["scaler"])
        xgb_model = joblib.load(required_artifacts["xgboost"])
        lgbm_model = joblib.load(required_artifacts["lightgbm"])
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(required_artifacts["catboost"]))
        meta_model = joblib.load(required_artifacts["meta_model"])

    LOGGER.info("Loaded %sd artifacts. selected_features=%s", horizon, len(selected_features))
    return selected_features, quantile_thresholds, scaler, xgb_model, lgbm_model, cat_model, meta_model


def aligned_proba(model, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = getattr(model, "classes_", np.array([0, 1, 2]))
    out = np.zeros((x.shape[0], 3), dtype=np.float32)
    for idx, cls in enumerate(classes):
        cls_int = int(cls)
        if cls_int in (0, 1, 2):
            out[:, cls_int] = probs[:, idx]
    denom = out.sum(axis=1, keepdims=True)
    return out / np.where(denom == 0.0, 1.0, denom)


def predict_stacking_horizon(latest_df, horizon: int) -> tuple[dict[str, list[float]], dict[str, Any], Any, list[str]]:
    """
    Run stacking model inference for a given horizon.
    
    Returns:
        predictions: {ticker: [p_down, p_sideways, p_up]}
        quantile_thresholds: {q33_return, q66_return}
        xgb_model: XGBoost base model (for feature importance)
        selected_features: list of feature names
    """
    selected_features, quantile_thresholds, scaler, xgb_model, lgbm_model, cat_model, meta_model = load_stacking_artifacts(horizon)

    missing_features = [c for c in selected_features if c not in latest_df.columns]
    if missing_features:
        raise ValueError(f"Missing selected features in live Alpha360 data for {horizon}d: {missing_features[:10]}")

    with timed_step(f"Preparing {horizon}d model input matrix"):
        x_raw = latest_df[selected_features].replace([np.inf, -np.inf], np.nan)
        x_raw = x_raw.fillna(x_raw.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float32)
        x_scaled = scaler.transform(x_raw).astype(np.float32)
    LOGGER.info("%sd model input shape=%s", horizon, x_scaled.shape)

    with timed_step(f"Running {horizon}d XGBoost/LightGBM/CatBoost base model inference"):
        base_meta = np.hstack([
            aligned_proba(xgb_model, x_scaled),
            aligned_proba(lgbm_model, x_scaled),
            aligned_proba(cat_model, x_scaled),
        ]).astype(np.float32)

    with timed_step(f"Running {horizon}d meta-model inference"):
        meta_probs = np.asarray(meta_model.predict_proba(base_meta), dtype=np.float32)

    predictions = {
        ticker: probs.tolist()
        for ticker, probs in zip(latest_df["ticker"].astype(str).tolist(), meta_probs, strict=False)
    }
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1][2], reverse=True)
    top_10_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in sorted_preds[:10]])
    LOGGER.info("[StackingGBDT %sd] TOP 10 UP PROBS: %s", horizon, top_10_str)
    LOGGER.info(
        "[StackingGBDT %sd] Quantile thresholds: q33=%.6f, q66=%.6f",
        horizon,
        quantile_thresholds["q33_return"],
        quantile_thresholds["q66_return"],
    )
    return predictions, quantile_thresholds, xgb_model, selected_features


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


def daily_inference(window_rows: int = 120, max_candidates: int = 6) -> str:
    """Daily trading path. No crawling. No full Alpha360 parquet load.

    Returns:
        Combined HTML report of the dispatched Top-3 BUY signals (one block
        per ticker, separated by a horizontal rule). Suitable for posting to
        Telegram with `parse_mode=HTML`. Returns "" when no signals were
        produced (e.g. liquidity filter cleared the pool, or no live prices).

        The on-cron alert path still pushes one message per ticker via
        `TelegramBot.send_signal_alert()` — this return value is purely an
        additional channel for chat-reply / on-demand UX.
    """
    LOGGER.info("Starting dual-horizon daily inference. No crawling will run in this task.")
    total_start = time.perf_counter()

    with timed_step(f"Building live Alpha360 features from recent OHLCV/macro windows ({window_rows} rows/ticker)"):
        generator = Alpha360Generator()
        live_pl = generator.build_live_features(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    LOGGER.info("Live feature frame loaded: %s rows x %s cols.", len(latest_df), len(latest_df.columns))
    if latest_df.empty:
        raise ValueError("Live feature frame is empty.")

    stacking_predictions_5d, _, xgb_model_5d, selected_features_5d = predict_stacking_horizon(latest_df, 5)
    stacking_predictions_20d, _, _, _ = predict_stacking_horizon(latest_df, 20)

    sorted_preds = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=True)
    top_10_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in sorted_preds[:10]])
    LOGGER.info("[StackingGBDT] TOP 10 UP PROBS: %s", top_10_str)

    bottom_3_sorted = sorted(stacking_predictions_5d.items(), key=lambda x: x[1][2], reverse=False)[:3]
    bottom_3_str = " | ".join([f"{t}: {p[2] * 100:.2f}%" for t, p in bottom_3_sorted])
    LOGGER.info("[StackingGBDT] BOTTOM 3 RISK: %s", bottom_3_str)

    # --- Liquidity Filter: ADDV >= 15,000,000,000 VND (20-day SMA of close*volume) ---
    # Raw close prices may be stored in thousands (e.g. 10.5 = 10,500 VND).
    # We multiply close*volume by 1000 when close < 1000 to restore actual VND turnover.
    _ADDV_THRESHOLD_VND = 15_000_000_000
    _ADDV_WINDOW = 20
    liquid_tickers: set[str] = set()
    if "close" in latest_df.columns and "volume" in latest_df.columns and "ticker" in latest_df.columns:
        for _ticker, _grp in latest_df.groupby("ticker"):
            _tail = _grp.tail(_ADDV_WINDOW)
            _close_raw = _tail["close"].astype(float)
            # Restore full-VND price if stored in thousands
            _scale = np.where(_close_raw < 1_000.0, 1_000.0, 1.0)
            _turnover = (_close_raw * _scale * _tail["volume"].astype(float))
            _addv = _turnover.mean()
            if np.isfinite(_addv) and _addv >= _ADDV_THRESHOLD_VND:
                liquid_tickers.add(str(_ticker))
        LOGGER.info(
            "[LiquidityFilter] %s / %s tickers pass ADDV >= %.0f VND threshold.",
            len(liquid_tickers), latest_df["ticker"].nunique(), _ADDV_THRESHOLD_VND,
        )
    else:
        LOGGER.warning("[LiquidityFilter] close/volume/ticker columns missing — skipping liquidity filter.")
        liquid_tickers = set(stacking_predictions_5d.keys())

    # Send Top-6 liquid tickers to arbitrator/sentiment layer.
    # 6 candidates → full LLM sentiment evaluation → final Top-3 selected by sentiment+quant score.
    _ARBITRATOR_POOL = 6
    candidate_tickers = [
        ticker
        for ticker, _probs in sorted(
            stacking_predictions_5d.items(),
            key=lambda item: item[1][2],
            reverse=True,
        )
        if ticker in liquid_tickers
    ][: min(max_candidates, _ARBITRATOR_POOL)]
    LOGGER.info("[Brain] Top-%s liquid candidates → arbitrator pool: %s", len(candidate_tickers), candidate_tickers)

    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}
    with timed_step("Evaluating candidates with dual-horizon arbitrator/sentiment layer"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, candidate_tickers)

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
    )
    LOGGER.info("Dual-horizon daily inference completed in %.2fs.", time.perf_counter() - total_start)
    return report_html


def run_trade_execution(
    top_buy_signals: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    latest_df: Any,
    xgb_model_5d: Any,
    selected_features_5d: list[str],
) -> str:
    """Execute portfolio updates, RL outcome logging, and dispatch Telegram alerts.

    Args:
        stacking_predictions: dual-horizon dict {"5d": {...}, "20d": {...}} produced by the
            Stacking GBDT (XGBoost+LightGBM+CatBoost → logistic meta) model.

    Returns:
        Combined HTML report of every signal dispatched (for chat-reply UX),
        or "" if no signals were dispatched. Per-ticker push alerts are still
        broadcast via `TelegramBot.send_signal_alert()` regardless.
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

        with timed_step("RL mistake logging"):
            db = manager.db
            today_str = datetime.now().strftime("%Y-%m-%d")
            logged = 0
            predictions_5d = stacking_predictions.get("5d", stacking_predictions)
            for ticker in predictions_5d.keys():
                if predictions_5d[ticker][2] > 0.6:
                    actual_t5_outcome = -0.05
                    if actual_t5_outcome < 0:
                        features_json = json.dumps(
                            {
                                "model": "stacking_gbdt_5d",
                                "top_drivers": _build_feature_explanation(
                                    xgb_model_5d,
                                    selected_features_5d,
                                    top_k=3,
                                )[0],
                            },
                            ensure_ascii=False,
                        )
                        db.conn.execute(
                            """
                            INSERT INTO rl_mistake_logs
                            (ticker, predicted_date, predicted_action, actual_t5_outcome, features_snapshot)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            [ticker, today_str, "BUY", actual_t5_outcome, features_json],
                        )
                        logged += 1
            LOGGER.info("RL mistakes logged: %s", logged)

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
            # Cap raw URL list at 3; pass as list so Telegram formatter loops explicitly
            source_urls: list[str] = (sentiment_data.get("source_urls", []) or [])[:3]
            confidence_5d = round(stacking_predictions.get("5d", {}).get(ticker, [0, 0, 0])[2] * 100, 2)
            LOGGER.info("[Alert] %s source_urls=%s", ticker, source_urls)

            signal_data = {
                "action": "MUA",
                "ticker": ticker,
                "price": f"{exec_price:,.0f} VND",
                "horizon": "5 ngày (5d)",
                "sentiment_score": sentiment_data.get("sentiment_score", 0.0),
                "sentiment_status": _format_sentiment_status(sentiment_data),
                "gemini_summary": sentiment_data.get("reasoning_vi", "Không có tin tức đáng kể."),
                "article_urls": source_urls,          # raw list → Telegram formatter loops this
                "model_class": "Stacking GBDT 5d: Tăng (UP)",
                "confidence": confidence_5d,
                "top_pos_features": top_pos_features,
                "top_neg_features": top_neg_features,
            }

            bot.send_signal_alert(signal_data)
            dispatched_signals.append(signal_data)
            sent += 1
        LOGGER.info("Telegram alerts dispatched: %s", sent)

    except Exception:
        LOGGER.exception("Error during trade execution")
        return _build_combined_report(dispatched_signals)

    return _build_combined_report(dispatched_signals)


def full_pipeline(force_crawl: bool = False) -> None:
    """Legacy full pipeline entrypoint kept explicit; not default daily trading path."""
    from src.crawlers.sentiment_crawler import update_daily_sentiment
    from src.data.crawlers import MacroCrawler, MacroProvider, StockCrawler
    from src.data.db_engine import DuckDBEngine

    LOGGER.warning("Running legacy full_pipeline. This performs crawl-if-allowed + full rebuild + inference.")

    if is_crawl_allowed(force_crawl=force_crawl):
        db_engine = DuckDBEngine()
        start_date = "2014-01-01"

        try:
            with timed_step(f"Fetching Macro Data since {start_date}"):
                macro_crawler = MacroCrawler()
                macro_df = macro_crawler.fetch_macro(start_date=start_date, file_path="data/macro_daily.parquet")

            if not macro_df.empty:
                LOGGER.info("Fetched %s macro records. Upserting to DuckDB.", len(macro_df))
                db_engine.upsert_dataframe(macro_df, "macro_daily")

            with timed_step(f"Fetching CPI and Deposit Rates since {start_date}"):
                macro_provider = MacroProvider()
                macro_raw_df = macro_provider.fetch_all()

            if not macro_raw_df.empty:
                LOGGER.info("Fetched %s long-format macro records. Upserting to DuckDB.", len(macro_raw_df))
                db_engine.upsert_dataframe(macro_raw_df, "macro_economic_raw")

            with timed_step("Fetching daily LLM sentiment"):
                sentiment_df = update_daily_sentiment(db_path="data/quant_v6_core.duckdb")
                LOGGER.info("Fetched %s sentiment records.", len(sentiment_df))

            stock_crawler = StockCrawler()
            hose_tickers = stock_crawler.get_hose_universe()

            if not hose_tickers:
                LOGGER.error("Could not discover HOSE universe. Check connection.")
                sys.exit(1)

            LOGGER.info("HOSE Universe Discovered: %s tickers.", len(hose_tickers))

            for idx, ticker in enumerate(hose_tickers, start=1):
                LOGGER.info("Ingesting HOSE %s/%s ticker=%s", idx, len(hose_tickers), ticker)
                stock_df = stock_crawler.fetch_ohlcv(
                    ticker,
                    start_date=start_date,
                    file_path=f"data/ohlcv_{ticker}.parquet",
                    sleep_before_request=True,
                )
                if not stock_df.empty:
                    db_engine.upsert_dataframe(stock_df, "stock_ohlcv")

            LOGGER.info("Final Verification of DuckDB State...")
            macro_count = db_engine.query("SELECT COUNT(*) FROM macro_daily").iloc[0, 0]
            macro_raw_count = db_engine.query("SELECT COUNT(*) FROM macro_economic_raw").iloc[0, 0]
            stock_count = db_engine.query("SELECT COUNT(*) FROM stock_ohlcv").iloc[0, 0]

            LOGGER.info("Total Macro Records (Daily): %s", macro_count)
            LOGGER.info("Total Macro Records (Raw Long): %s", macro_raw_count)
            LOGGER.info("Total Stock Records (OHLCV): %s", stock_count)

        finally:
            LOGGER.info("Closing DuckDB connection to release file lock.")
            db_engine.close()
    else:
        LOGGER.info("Crawl phase bypassed. Continuing with existing local data.")

    build_alpha360()
    daily_inference()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant V6 orchestration CLI")
    parser.add_argument(
        "--task",
        default="daily_inference",
        choices=["daily_inference", "build_alpha360", "crawl_hose", "full_pipeline"],
        help="Task to run. daily_inference is no-crawl live path.",
    )
    parser.add_argument("--start-date", default="2016-01-01", help="Crawler start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Crawler end date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--data-dir", default="data", help="Directory for ohlcv_<ticker>.parquet files.")
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
    parser.add_argument("--force-crawl", action="store_true", help="Bypass market-hour crawl guard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task == "crawl_hose":
        crawl_hose(
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
            force_crawl=args.force_crawl,
        )
        return

    if args.task == "build_alpha360":
        build_alpha360()
        return

    if args.task == "full_pipeline":
        full_pipeline(force_crawl=args.force_crawl)
        return

    daily_inference(window_rows=args.window_rows, max_candidates=args.max_candidates)


if __name__ == "__main__":
    main()