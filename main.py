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
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from catboost import CatBoostClassifier

from config.settings import CONFIG
from src.features.alpha360_generator import Alpha360Generator
from src.models.stacking_model.economic_metrics import (
    META_LABEL_FEATURE_NAMES,
    N_CLOSE_LAGS_FOR_META,
    meta_label_feature_matrix,
)
from src.models.quant_agent_arbitrator import (
    evaluate_trades_batch,
    get_rebalance_advice,
    map_tickers_to_news,
    scrape_centralized_news,
)
from src.trading.portfolio_manager import PortfolioManager
from src.utils.telegram_alerter import TelegramBot

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


def load_stacking_artifacts(horizon: int) -> tuple[list[str], dict[str, Any], Any, Any, CatBoostClassifier, Any, Any]:
    artifact_dir = Path("models/stacking") / f"{horizon}d"
    required_artifacts = {
        "selected_features": artifact_dir / "selected_features.json",
        "xgboost": artifact_dir / "xgboost_model.joblib",
        "lightgbm": artifact_dir / "lightgbm_model.joblib",
        "catboost": artifact_dir / "catboost_model.cbm",
        "meta_model": artifact_dir / "meta_model.joblib",
        "thresholds": artifact_dir / "quantile_thresholds.json",
    }
    missing = [str(path) for path in required_artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {horizon}d stacking artifacts: {missing}. Run train_stacking.py first.")

    # Task 3: the secondary meta-labeler is OPTIONAL. If absent (e.g. a model
    # trained before Task 3, or a degenerate fold), inference cleanly falls
    # back to the primary-only cost-aware gate — no crash.
    meta_labeler_path = artifact_dir / "meta_labeler.joblib"

    LOGGER.info("Loading %sd stacking artifacts from %s", horizon, artifact_dir)
    with timed_step("Loading model artifacts"):
        with required_artifacts["selected_features"].open("r", encoding="utf-8") as f:
            selected_features = json.load(f)
        with required_artifacts["thresholds"].open("r", encoding="utf-8") as f:
            quantile_thresholds = json.load(f)

        # StandardScaler removed from pipeline (train_stacking.py Flaw 3 fix).
        # Alpha360 features are already rolling Z-scores; no secondary scaling needed.
        xgb_model = joblib.load(required_artifacts["xgboost"])
        lgbm_model = joblib.load(required_artifacts["lightgbm"])
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(required_artifacts["catboost"]))
        meta_model = joblib.load(required_artifacts["meta_model"])
        meta_labeler = joblib.load(meta_labeler_path) if meta_labeler_path.exists() else None

    LOGGER.info(
        "Loaded %sd artifacts. selected_features=%s meta_labeler=%s",
        horizon, len(selected_features), "ON" if meta_labeler is not None else "OFF (primary-only)",
    )
    return selected_features, quantile_thresholds, xgb_model, lgbm_model, cat_model, meta_model, meta_labeler


# Task 2: most-recent 5d top-5 probability breakdown, populated by
# predict_stacking_horizon and surfaced in the Telegram empty-pool reply
# so the operator can see WHY the bot stayed quiet.
_LATEST_5D_BREAKDOWN: list[str] = []


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


def predict_stacking_horizon(latest_df, horizon: int) -> tuple[dict[str, list[float]], dict[str, Any], Any, list[str], dict[str, bool]]:
    """
    Run stacking model inference for a given horizon.

    Returns:
        predictions: {ticker: [p_down, p_sideways, p_up]}
        thresholds:  cost-aware metadata (pnl_threshold_tau, round_trip_cost…)
        xgb_model:   XGBoost base model (for feature importance)
        selected_features: list of feature names
        meta_gate:   {ticker: bool} — Task 3 combined decision. A ticker is
                     tradeable ONLY if the primary model is bullish
                     (P(UP) >= τ*) AND the meta-labeler says GO
                     (P(profit) >= 0.5). Falls back to the primary-only
                     cost-aware gate (or legacy all-pass) when no τ*/labeler.
    """
    (selected_features, quantile_thresholds, xgb_model, lgbm_model,
     cat_model, meta_model, meta_labeler) = load_stacking_artifacts(horizon)

    missing_features = [c for c in selected_features if c not in latest_df.columns]
    if missing_features:
        raise ValueError(f"Missing selected features in live Alpha360 data for {horizon}d: {missing_features[:10]}")

    with timed_step(f"Preparing {horizon}d model input matrix"):
        # No StandardScaler transform — features are already rolling Z-scores from Alpha360.
        x_raw = latest_df[selected_features].replace([np.inf, -np.inf], np.nan)
        x_input = x_raw.fillna(x_raw.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float32)
    LOGGER.info("%sd model input shape=%s", horizon, x_input.shape)

    with timed_step(f"Running {horizon}d XGBoost/LightGBM/CatBoost base model inference"):
        base_meta = np.hstack([
            aligned_proba(xgb_model, x_input),
            aligned_proba(lgbm_model, x_input),
            aligned_proba(cat_model, x_input),
        ]).astype(np.float32)

    with timed_step(f"Running {horizon}d meta-model inference"):
        # aligned_proba → guaranteed [P(DOWN), P(SIDE), P(UP)] column order,
        # identical to training's aligned_predict_proba (zero skew).
        meta_probs = aligned_proba(meta_model, base_meta)

    tickers = [str(t) for t in latest_df["ticker"].tolist()]
    predictions = {
        ticker: probs.tolist()
        for ticker, probs in zip(tickers, meta_probs, strict=False)
    }

    # ── TASK 3: combined meta-labeling gate ─────────────────────────────
    tau_star = float(quantile_thresholds.get("pnl_threshold_tau", 0.5))
    has_tau = "pnl_threshold_tau" in quantile_thresholds
    p_up_arr = meta_probs[:, 2]
    p_profit = None  # set only in the meta-labeled branch below

    if not has_tau:
        # Legacy artifact (pre-Task-2): preserve old behaviour — no gate.
        gate_arr = np.ones(len(tickers), dtype=bool)
        gate_mode = "legacy-all-pass"
    elif meta_labeler is None:
        # Task-2 primary-only cost-aware gate.
        gate_arr = p_up_arr >= tau_star
        gate_mode = f"primary-only τ*={tau_star:.2f}"
    else:
        lag_cols = [f"close_{i}" for i in range(N_CLOSE_LAGS_FOR_META)]
        miss = [c for c in lag_cols if c not in latest_df.columns]
        if miss:
            LOGGER.warning(
                "[MetaLabeler %sd] missing close lags %s — primary-only fallback.",
                horizon, miss[:3],
            )
            gate_arr = p_up_arr >= tau_star
            gate_mode = f"primary-only (no lags) τ*={tau_star:.2f}"
        else:
            close_lags = (
                latest_df[lag_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            x_meta = meta_label_feature_matrix(meta_probs, tau_star, close_lags)
            p_profit = np.asarray(
                meta_labeler.predict_proba(x_meta), dtype=np.float64
            )[:, 1]
            gate_arr = (p_up_arr >= tau_star) & (p_profit >= 0.5)
            gate_mode = f"meta-labeled τ*={tau_star:.2f} ∧ P(profit)≥0.5"

    meta_gate = {t: bool(g) for t, g in zip(tickers, gate_arr, strict=False)}

    # Per-ticker P(profit) map (meta-labeled mode only) for the diagnostic.
    pprofit_by_ticker: dict[str, float] = {}
    if p_profit is not None:
        pprofit_by_ticker = {
            t: float(pp) for t, pp in zip(tickers, p_profit, strict=False)
        }

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1][2], reverse=True)
    top_10_str = " | ".join(
        f"{t}:{p[2]*100:.1f}%{'✓' if meta_gate.get(t) else '✗'}"
        for t, p in sorted_preds[:10]
    )
    LOGGER.info("[StackingGBDT %sd] TOP 10 UP (✓=gate open): %s", horizon, top_10_str)
    LOGGER.info(
        "[StackingGBDT %sd] gate=%s | %s/%s tickers tradeable | round_trip_cost=%.4f",
        horizon, gate_mode, sum(meta_gate.values()), len(meta_gate),
        float(quantile_thresholds.get("round_trip_cost", 0.0)),
    )

    # ── TASK 2: per-candidate 3-class breakdown for the TOP 5 ──────────
    # Always logged regardless of gate pass, so the operator sees EXACTLY
    # why the bot is quiet. Cached for the Telegram empty-pool message.
    def _reason(tk: str, pu: float) -> str:
        if not has_tau:
            return "ACCEPTED (legacy, no gate)"
        if pu < tau_star:
            return f"REJECTED (P(UP)={pu*100:.1f}% < tau={tau_star:.2f})"
        if meta_labeler is not None:
            pp = pprofit_by_ticker.get(tk)
            if pp is not None and pp < 0.5:
                return (f"REJECTED (P(UP) ok, meta P(profit)="
                        f"{pp*100:.1f}% < 50%)")
        return f"ACCEPTED (P(UP)>=tau={tau_star:.2f}"+(
            f", P(profit)={pprofit_by_ticker.get(tk,0)*100:.1f}%)"
            if meta_labeler is not None else ")")

    breakdown_lines: list[str] = []
    for tk, pr in sorted_preds[:5]:
        p_dn, p_sd, p_u = pr[0], pr[1], pr[2]
        line = (f"{tk}: P(UP)={p_u*100:.1f}%, P(SIDE)={p_sd*100:.1f}%, "
                f"P(DN)={p_dn*100:.1f}% | Gate: {_reason(tk, p_u)}")
        breakdown_lines.append(line)
        LOGGER.info("[StackingGBDT %sd][TOP5] %s", horizon, line)
    if horizon == 5:
        _LATEST_5D_BREAKDOWN.clear()
        _LATEST_5D_BREAKDOWN.extend(breakdown_lines)

    return predictions, quantile_thresholds, xgb_model, selected_features, meta_gate


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
        # T0 close — use ≤ predicted_date so a weekend prediction still
        # finds the prior trading day's close. (Read — no lock needed.)
        t0_row = db.conn.execute(
            """
            SELECT close FROM stock_ohlcv
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
            """,
            [ticker, predicted_date],
        ).fetchone()
        # T+5 close — first available trading day at or after predicted_date + 5d.
        t5_row = db.conn.execute(
            f"""
            SELECT close FROM stock_ohlcv
            WHERE ticker = ? AND date >= ? + INTERVAL {_RL_HORIZON_DAYS} DAY
            ORDER BY date ASC LIMIT 1
            """,
            [ticker, predicted_date],
        ).fetchone()

        if not t0_row or not t5_row:
            continue
        t0_close, t5_close = t0_row[0], t5_row[0]
        if t0_close is None or t5_close is None or t0_close <= 0:
            continue

        actual = (float(t5_close) - float(t0_close)) / float(t0_close)
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


def daily_inference(
    window_rows: int = 120,
    max_candidates: int = 6,
    broadcast: bool = True,
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
        live_pl = generator.build_live_features(window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    LOGGER.info("Live feature frame loaded: %s rows x %s cols.", len(latest_df), len(latest_df.columns))
    if latest_df.empty:
        raise ValueError("Live feature frame is empty.")

    stacking_predictions_5d, thr_5d, xgb_model_5d, selected_features_5d, meta_gate_5d = predict_stacking_horizon(latest_df, 5)
    stacking_predictions_20d, _, _, _, _ = predict_stacking_horizon(latest_df, 20)

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

    # --- Universe Filter -------------------------------------------------
    # Segment exclusions applied AFTER liquidity, BEFORE the meta-labeler /
    # arbitrator. Knobs live in CONFIG.universe_filter (config/settings.json)
    # and default to a NO-OP, so live behaviour is unchanged until opted in.
    #
    # CAVEAT (no market-cap data in the pipeline): "penny / ultra-smallcap"
    # is approximated by a RAW-PRICE FLOOR (`min_price_vnd`), not a true
    # market cap. VN30 exclusion uses the published constituent list (exact).
    _uf = CONFIG.universe_filter
    universe_tickers: set[str] = set(liquid_tickers)
    if _uf.enabled and liquid_tickers:
        _vn30 = {t.upper() for t in _uf.vn30_tickers}
        _manual = {t.upper() for t in _uf.exclude_tickers}
        # Latest real-VND close per ticker (reuse the liquidity block's
        # thousands-scaling rule: price < 1000 ⇒ stored in thousands).
        _last_px: dict[str, float] = {}
        if {"close", "ticker"}.issubset(latest_df.columns):
            for _t, _g in latest_df.groupby("ticker"):
                _c = float(_g["close"].iloc[-1])
                _last_px[str(_t)] = _c * (1000.0 if _c < 1000.0 else 1.0)
        _excluded: dict[str, str] = {}
        for _t in liquid_tickers:
            tu = str(_t).upper()
            if _uf.exclude_vn30 and tu in _vn30:
                _excluded[_t] = "VN30"
            elif tu in _manual:
                _excluded[_t] = "manual-blacklist"
            elif (_uf.min_price_vnd > 0
                  and _last_px.get(_t, float("inf")) < _uf.min_price_vnd):
                _excluded[_t] = f"price<{_uf.min_price_vnd:.0f}VND(penny-proxy)"
        universe_tickers -= set(_excluded)
        LOGGER.info(
            "[UniverseFilter] excluded %s/%s (exclude_vn30=%s min_price_vnd=%.0f "
            "manual=%s). %s liquid→universe. e.g. %s",
            len(_excluded), len(liquid_tickers), _uf.exclude_vn30,
            _uf.min_price_vnd, len(_manual), len(universe_tickers),
            dict(list(_excluded.items())[:6]),
        )
    else:
        LOGGER.info(
            "[UniverseFilter] disabled — passing all %s liquid tickers through.",
            len(liquid_tickers),
        )

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
        _tau5 = float(thr_5d.get("pnl_threshold_tau", 0.5))
        _ranked = [
            t for t, _p in sorted(
                stacking_predictions_5d.items(),
                key=lambda kv: kv[1][2], reverse=True,
            )
            if t in universe_tickers
        ]
        candidate_tickers = _ranked[:3]
        for t in candidate_tickers:
            _pu = stacking_predictions_5d[t][2]
            if _pu < _tau5:
                fallback_reasons[t] = (
                    f"Bị loại: P(UP) {_pu * 100:.1f}% < tau={_tau5:.2f}"
                )
            elif not meta_gate_5d.get(t, True):
                fallback_reasons[t] = (
                    "Bị loại: Meta-Labeler P(lợi nhuận) < 50%"
                )
            else:
                fallback_reasons[t] = (
                    f"Theo dõi (thị trường yếu, P(UP) {_pu * 100:.1f}%)"
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
        report_html = _build_fallback_observability_report_vi(
            candidate_tickers,
            stacking_predictions_5d,
            all_sentiments,
            fallback_reasons,
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
        broadcast=broadcast,
    )
    LOGGER.info("Dual-horizon daily inference completed in %.2fs.", time.perf_counter() - total_start)
    return report_html


def _build_fallback_observability_report_vi(
    fallback_tickers: list[str],
    stacking_predictions_5d: dict,
    all_sentiments: dict,
    fallback_reasons: dict,
) -> str:
    """Vietnamese 'weak-market' observability report (HTML, the bot's
    Telegram parse mode).

    These tickers are MONITORING ONLY — not trade signals. The caller
    (`daily_inference`) returns this BEFORE `run_trade_execution`, so there
    is no portfolio / RL / dispatch side effect. The header and per-ticker
    flag make it impossible to mistake these for actionable BUYs.
    """
    out = [
        "<b>[⚠️ THỊ TRƯỜNG YẾU - KHÔNG CÓ TÍN HIỆU ĐẠT CHUẨN. "
        "HIỂN THỊ TOP 3 MÃ ĐỂ THEO DÕI]</b>",
        "<i>⛔ KHÔNG GIAO DỊCH các mã dưới đây — chỉ dùng để quan sát thị "
        "trường. Không mã nào vượt qua bộ lọc τ* + Meta-Labeler hôm nay.</i>",
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
            or "Không có phân tích sentiment khả dụng (tin tức trống/timeout)."
        )
        gate = fallback_reasons.get(t, "Bị loại: không đạt chuẩn")
        out += [
            f"<b>{i}. {html.escape(t)}</b>",
            f"   • Xác suất 5d: <code>P(TĂNG)={p_up:.1f}% | "
            f"P(ĐI NGANG)={p_sd:.1f}% | P(GIẢM)={p_dn:.1f}%</code>",
            f"   • Lý do bị loại: <code>{html.escape(gate)}</code>",
            f"   • Sentiment (LLM Gemini): <b>{score:+.2f}</b> — "
            f"{html.escape(str(reason_vi))[:600]}",
            "",
        ]
    out.append(
        "<i>Nguồn: Mô hình 5d (Stacking + Meta-Labeler, τ*=0.48) + Gemini "
        "sentiment. Chế độ Quan sát Dự phòng — KHÔNG phải khuyến nghị MUA.</i>"
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
    broadcast: bool = True,
) -> str:
    """Execute portfolio updates, RL outcome logging, and dispatch Telegram alerts.

    Args:
        stacking_predictions: dual-horizon dict {"5d": {...}, "20d": {...}} produced by the
            Stacking GBDT (XGBoost+LightGBM+CatBoost → logistic meta) model.
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


def _build_sell_hold_report(
    holding_tickers: list[str],
    final_decisions: dict,
    all_sentiments: dict,
    stacking_predictions: dict,
    live_exec_prices: dict,
    missing_tickers: list[str] | None = None,
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
        reasoning = str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể."))
        confidence_5d = round(predictions_5d.get(ticker, [0, 0, 0])[2] * 100, 2)
        price = live_exec_prices.get(ticker)
        price_str = f"{price:,.0f} VND" if price else "N/A"

        if decision == _SELL_DECISION:
            verdict = "🔴 <b>BÁN (SELL)</b>"
        elif decision == 2:
            verdict = "🟢 <b>GIỮ (HOLD - xu hướng tăng)</b>"
        else:
            verdict = "🟡 <b>GIỮ (HOLD - đi ngang)</b>"

        source_urls = (sentiment.get("source_urls", []) or [])[:3]
        if source_urls:
            url_lines = "\n".join(f"  - {html.escape(u)}" for u in source_urls)
        else:
            url_lines = "  Không có tin tức đáng kể"

        block = (
            f"📌 <b>{html.escape(ticker)}</b> @ {html.escape(price_str)}\n"
            f"• <b>Khuyến nghị:</b> {verdict}\n"
            f"• <b>Quant 5d UP confidence:</b> {confidence_5d}%\n"
            f"• <b>Tâm lý:</b> {html.escape(sentiment_status)} "
            f"(score={sentiment_score:+.2f})\n"
            f"• <b>Phân tích tin tức:</b> {html.escape(reasoning)}\n"
            f"• <b>Nguồn:</b>\n{url_lines}"
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
        live_pl = generator.build_live_features(window_rows=window_rows)
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

    latest_df = latest_df[latest_df["ticker"].astype(str).isin(present)].reset_index(drop=True)

    stacking_predictions_5d, _, _, _, _ = predict_stacking_horizon(latest_df, 5)
    stacking_predictions_20d, _, _, _, _ = predict_stacking_horizon(latest_df, 20)
    horizon_predictions = {"5d": stacking_predictions_5d, "20d": stacking_predictions_20d}

    with timed_step("Holdings arbitrator + sentiment scoring"):
        final_decisions, all_sentiments = evaluate_trades_batch(horizon_predictions, present)

    live_exec_prices = _get_live_exec_prices(latest_df, present)

    LOGGER.info("[/suggest_sell] completed in %.2fs.", time.perf_counter() - total_start)
    return _build_sell_hold_report(
        holding_tickers=present,
        final_decisions=final_decisions,
        all_sentiments=all_sentiments,
        stacking_predictions=horizon_predictions,
        live_exec_prices=live_exec_prices,
        missing_tickers=missing,
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


def _build_verify_report(
    ticker: str,
    decision: int | None,
    sentiment: dict,
    stacking_5d: list[float],
    stacking_20d: list[float],
    live_exec_price: float | None,
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
    sent_reasoning = str(sentiment.get("reasoning_vi", "Không có tin tức đáng kể."))

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
        f"📊 <b>[1] Định lượng (Stacking GBDT)</b>\n"
        f"• <b>Dự báo 5d:</b> {pred_5d_label}\n"
        f"• <b>Độ tin cậy 5d:</b> {confidence_5d}% "
        f"(UP={p_up * 100:.1f}%, SIDE={p_side * 100:.1f}%, DOWN={p_down * 100:.1f}%)\n"
        f"• <b>Dự báo 20d:</b> {pred_20d_label} ({confidence_20d}%)\n\n"
        f"📰 <b>[2] Sentiment (LLM)</b>\n"
        f"• <b>Tâm lý:</b> {html.escape(sent_status)} "
        f"(score={sent_score:+.2f})\n"
        f"• <b>Phân tích:</b> {html.escape(sent_reasoning)}\n\n"
        f"🎯 <b>[3] Verdict tổng hợp:</b> {verdict_html}\n\n"
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

    # --- Step 1: Liquidity / data availability check ---
    # `Alpha360Generator.build_live_features(tickers=[ticker])` filters the
    # parquet glob down to a single file — efficient single-ticker read.
    # If the parquet doesn't exist → FileNotFoundError → warning to user.
    try:
        with timed_step(f"Building live features for /verify {ticker}"):
            generator = Alpha360Generator()
            live_pl = generator.build_live_features(tickers=[ticker], window_rows=window_rows)
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

    # Defensive filter (build_live_features should already have filtered).
    latest_df = latest_df[latest_df["ticker"].astype(str) == ticker].reset_index(drop=True)

    # --- Step 2: Stacking GBDT inference (5d primary + 20d for arbitrator) ---
    try:
        stacking_5d, _, _, _, _ = predict_stacking_horizon(latest_df, 5)
        stacking_20d, _, _, _, _ = predict_stacking_horizon(latest_df, 20)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[/verify] Stacking inference failed for %s.", ticker)
        return (
            f"⚠️ Lỗi mô hình Stacking GBDT cho <b>{html.escape(ticker)}</b>: "
            f"<code>{html.escape(str(exc))}</code>"
        )

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

    LOGGER.info("[/verify] %s completed in %.2fs.", ticker, time.perf_counter() - total_start)
    return _build_verify_report(
        ticker=ticker,
        decision=final_decisions.get(ticker),
        sentiment=all_sentiments.get(ticker, {}),
        stacking_5d=list(stacking_5d.get(ticker, [0.33, 0.34, 0.33])),
        stacking_20d=list(stacking_20d.get(ticker, [0.33, 0.34, 0.33])),
        live_exec_price=live_exec_prices.get(ticker),
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

    with timed_step(f"[/rebalance] Building live Alpha360 features for {held_tickers}"):
        generator = Alpha360Generator()
        live_pl = generator.build_live_features(tickers=held_tickers, window_rows=window_rows)
        latest_df = live_pl.to_pandas()

    if latest_df.empty:
        LOGGER.warning("[/rebalance] live feature frame empty for %s", held_tickers)
        return ""

    present = [t for t in held_tickers if t in set(latest_df["ticker"].astype(str))]
    if not present:
        return ""

    latest_df = latest_df[latest_df["ticker"].astype(str).isin(present)].reset_index(drop=True)
    stacking_predictions_5d, _, _, _, _ = predict_stacking_horizon(latest_df, 5)
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
                # TD-12 circuit breaker: 45s hard cap per ticker.
                try:
                    stock_df = stock_crawler._fetch_ohlcv_with_timeout(
                        ticker=ticker,
                        start_date=start_date,
                        file_path=f"data/ohlcv_{ticker}.parquet",
                        sleep_before_request=True,
                    )
                except Exception as ticker_exc:  # noqa: BLE001
                    LOGGER.error("Per-ticker fetch wrapper crashed for %s: %s", ticker, ticker_exc)
                    continue
                if not stock_df.empty:
                    try:
                        db_engine.upsert_dataframe(stock_df, "stock_ohlcv")
                    except Exception as upsert_exc:  # noqa: BLE001
                        LOGGER.error("DuckDB upsert failed for %s: %s — continuing.", ticker, upsert_exc)

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
            crawl_hose(
                start_date=args.start_date,
                end_date=args.end_date,
                data_dir=args.data_dir,
                force_crawl=args.force_crawl,
            )
        elif task_name == "build_alpha360":
            build_alpha360()
        elif task_name == "full_pipeline":
            full_pipeline(force_crawl=args.force_crawl)
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