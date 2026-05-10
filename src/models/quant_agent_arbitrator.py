"""
Quant Agent Arbitrator

Extracts the core trading intelligence from the TradingAgents repository
and adapts it to work as a standalone module alongside the PyTorch LSTM model.
Integrates real LLM sentiment analysis via Gemini 2.5 Flash and RSS news fetching.

Refactored: async/aiohttp concurrency, hard timeouts, semaphore rate-limit, fault-tolerant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import aiohttp  # type: ignore[import-not-found]
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    print("WARNING: 'aiohttp' not found. pip install aiohttp")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]
    print("WARNING: 'beautifulsoup4' not found. pip install beautifulsoup4")

try:
    from gnews import GNews  # type: ignore[import-not-found]
except ImportError:
    GNews = None  # type: ignore[assignment,misc]
    print("WARNING: 'gnews' not found. pip install gnews")

try:
    from googlenewsdecoder import new_decoderv1  # type: ignore[import-not-found]
except ImportError:
    new_decoderv1 = None  # type: ignore[assignment]
    print("WARNING: 'googlenewsdecoder' not found. pip install googlenewsdecoder")

# New official Google GenAI SDK (replaces deprecated `google.generativeai`).
# Install: pip install google-genai
try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    print("WARNING: 'google-genai' not found. pip install google-genai")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config constants (can be moved to config/settings.py later)
# ---------------------------------------------------------------------------
NEWS_FETCH_TIMEOUT_SEC: float = 7.0
NEWS_MAX_CONCURRENT: int = 4
NEWS_MAX_PER_HOST: int = 1
NEWS_MAX_ARTICLES_PER_TICKER: int = 3
NEWS_DAYS_BACK: int = 3
NEWS_DOMAIN_JITTER_RANGE_SEC: tuple[float, float] = (0.5, 1.5)
NEWS_BINARY_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".mp3",
    ".mp4",
    ".avi",
)
NEWS_BINARY_CONTENT_TYPES: tuple[str, ...] = (
    "application/pdf",
    "application/msword",
    "application/vnd.",
    "application/octet-stream",
    "application/zip",
    "image/",
    "audio/",
    "video/",
)
NEWS_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
NEWS_HEADERS: dict[str, str] = {
    "User-Agent": NEWS_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# ---------------------------------------------------------------------------
# AGENT PROMPTS
# ---------------------------------------------------------------------------
NEWS_ANALYST_SYSTEM_PROMPT = """Bạn là TradingAgents-style News Analyst cho thị trường chứng khoán Việt Nam.

Bạn phân tích FULL ARTICLE BODY, không chỉ title. Tin tài chính Việt Nam thường clickbait: tiêu đề tích cực có thể chứa rủi ro sâu trong thân bài, hoặc ngược lại.

PHƯƠNG PHÁP (theo TradingAgents/TauricResearch):
1. Đọc toàn bộ thân bài đã trích xuất từ <p> tags.
2. Tách rõ catalyst tăng giá, risk/headwind, clickbait reversal.
3. Ưu tiên tin 24-48h gần nhất, nhưng không bỏ qua rủi ro sâu trong body.
4. Đánh giá tác động trực tiếp lên cổ phiếu/ticker, không đánh giá chung chung.

OUTPUT BẮT BUỘC:
- Trả về RAW JSON hợp lệ. Không markdown. Không ```json.
- Mỗi ticker phải có đúng các field sau:
{
  "FPT": {
    "catalyst": "Xúc tác: ...",
    "risk": "Rủi ro: ...",
    "sentiment_score": 0.35,
    "reasoning_vi": "Kết luận Tâm lý (Sentiment Score): +0.35. ...",
    "source_urls": ["https://..."]
  }
}

RÀNG BUỘC:
- catalyst: bắt đầu bằng "Xúc tác:"
- risk: bắt đầu bằng "Rủi ro:"
- reasoning_vi: bắt đầu bằng "Kết luận Tâm lý (Sentiment Score):"
- sentiment_score: float trong [-1.0, 1.0]
- Nếu body không đủ bằng chứng: sentiment_score gần 0.0, nói rõ thiếu bằng chứng.
"""

PORTFOLIO_MANAGER_CONTEXT = """As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.
Rating Scale:
- Buy: Strong conviction to enter or add to position
- Overweight: Favorable outlook, gradually increase exposure
- Hold: Maintain current position, no action needed
- Underweight: Reduce exposure, take partial profits
- Sell: Exit position or avoid entry"""


# ---------------------------------------------------------------------------
# ASYNC NEWS SCRAPING
# ---------------------------------------------------------------------------

def _is_binary_url(url: str) -> bool:
    """Fast URL-extension guard to avoid decoding PDFs/DOCX/binary blobs as UTF-8."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in NEWS_BINARY_EXTENSIONS)


def _is_binary_content_type(content_type: str) -> bool:
    """Content-Type guard before reading response body."""
    ctype = content_type.lower().split(";", 1)[0].strip()
    if not ctype:
        return False
    if ctype.startswith("text/"):
        return False
    if ctype in {"application/xhtml+xml", "application/xml"}:
        return False
    return any(ctype == blocked or ctype.startswith(blocked) for blocked in NEWS_BINARY_CONTENT_TYPES)


def _extract_article_body_from_html(html: str) -> str | None:
    """Extract article text from <p> tags in main content containers; never return raw HTML."""
    if BeautifulSoup is None:
        html = re.sub(r"(?is)<(script|style|noscript|nav|header|footer).*?>.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000] if len(text) > 200 else None

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    selectors = [
        "div.detail-content",
        "div.detail-content-body",
        "div.newscontent",
        "div.news-content",
        "div.article-content",
        "div.article__body",
        "div.contentdetail",
        "div#mainContent",
        "article",
        "main",
    ]
    containers = []
    for selector in selectors:
        containers.extend(soup.select(selector))

    search_roots = containers or [soup]
    paragraphs: list[str] = []
    for root in search_roots:
        for p_tag in root.find_all("p"):
            text = p_tag.get_text(" ", strip=True)
            if len(text) >= 20:
                paragraphs.append(text)

    deduped = list(dict.fromkeys(paragraphs))
    text = " ".join(deduped)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000] if len(text) > 200 else None


class AsyncNewsScraper:
    """Robust async VN finance scraper: stealth headers, binary skip, per-domain jitter, 7s timeout."""

    def __init__(
        self,
        timeout_sec: float = NEWS_FETCH_TIMEOUT_SEC,
        max_concurrent: int = NEWS_MAX_CONCURRENT,
        max_per_host: int = NEWS_MAX_PER_HOST,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec) if aiohttp is not None else None
        self.connector = (
            aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_per_host, ttl_dns_cache=300)
            if aiohttp is not None
            else None
        )

    async def fetch_article_text(self, session: Any, url: str) -> str | None:
        """Fetch HTML safely; skip PDF/DOCX/binary by URL + Content-Type; parse only article <p> text."""
        if not url or _is_binary_url(url):
            LOGGER.debug("Skip binary URL: %s", url)
            return None

        domain = urlparse(url).netloc.lower()
        lock = self.domain_locks[domain]

        async with self.semaphore:
            async with lock:
                await asyncio.sleep(random.uniform(*NEWS_DOMAIN_JITTER_RANGE_SEC))
                try:
                    async with asyncio.timeout(self.timeout_sec):
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status != 200:
                                LOGGER.debug("HTTP %s for %s", resp.status, url)
                                return None

                            content_type = resp.headers.get("Content-Type", "")
                            if _is_binary_content_type(content_type):
                                LOGGER.debug("Skip binary Content-Type=%s URL=%s", content_type, url)
                                return None

                            html = await resp.text(encoding="utf-8", errors="ignore")
                            return _extract_article_body_from_html(html)
                except asyncio.TimeoutError:
                    LOGGER.warning("Timeout fetching %s", url)
                except UnicodeError as exc:
                    LOGGER.warning("Decode error fetching %s: %s", url, exc)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Network/unexpected error fetching %s: %s", url, exc)
        return None

    async def decode_and_fetch(self, session: Any, article: dict[str, Any]) -> dict[str, Any] | None:
        """Decode Google News redirect, then fetch parsed article body."""
        url = article.get("url", "")
        title = article.get("title", "")
        if not url:
            return None

        real_url = url
        decoder = new_decoderv1
        if decoder is not None:
            try:
                loop = asyncio.get_running_loop()
                decoder_result = await loop.run_in_executor(None, lambda: decoder(url, interval=0.05))
                if decoder_result and decoder_result.get("status"):
                    real_url = decoder_result.get("decoded_url", url)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Decoder failed for %s: %s", url, exc)

        content = await self.fetch_article_text(session, real_url)
        if not content:
            return None

        return {"title": title, "url": real_url, "content": content}

    async def fetch_many(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch many articles with bounded concurrency; session-level Chrome headers."""
        if aiohttp is None:
            return []

        results: list[dict[str, Any]] = []
        async with aiohttp.ClientSession(timeout=self.timeout, connector=self.connector, headers=NEWS_HEADERS) as session:
            tasks = [self.decode_and_fetch(session, art) for art in articles]
            for coro in asyncio.as_completed(tasks):
                try:
                    item = await coro
                    if item:
                        results.append(item)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Task exception: %s", exc)
        return results


async def _scrape_news_async(days_back: int = NEWS_DAYS_BACK, target_tickers: list[str] | None = None) -> list[dict[str, Any]]:
    """Async scraper: GNews discovery + robust AsyncNewsScraper full-text extraction."""
    if GNews is None or aiohttp is None:
        LOGGER.error("GNews or aiohttp not installed; returning empty news.")
        return []
    if BeautifulSoup is None:
        LOGGER.warning("beautifulsoup4 missing; using weaker regex HTML fallback.")
        return []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    google_news = GNews(
        language="vi",
        country="VN",
        start_date=(start_date.year, start_date.month, start_date.day),
        end_date=(end_date.year, end_date.month, end_date.day),
        max_results=30,
    )

    # Strictly ticker-specific queries only — no general market queries to avoid noise/rate-limits.
    # Collect per-ticker article pools so we can cap the pre-fetch queue to 3 URLs/ticker.
    ticker_article_pool: dict[str, list[dict[str, Any]]] = {}
    if target_tickers:
        queries_built = [(ticker, f'"{ticker}" site:cafef.vn') for ticker in target_tickers] + \
                        [(ticker, f'"{ticker}" site:vietstock.vn') for ticker in target_tickers]
        LOGGER.info("[News Fetcher][DEBUG] Ticker-specific queries=%s for tickers=%s", len(queries_built), target_tickers)
        seen_urls: set[str] = set()
        for ticker, q in queries_built:
            try:
                query_articles = google_news.get_news(q)
                LOGGER.info("[News Fetcher][DEBUG] query=%r returned=%s", q, len(query_articles))
                pool = ticker_article_pool.setdefault(ticker, [])
                for art in query_articles:
                    u = art.get("url", "")
                    if not u or u in seen_urls:
                        continue
                    if len(pool) >= NEWS_MAX_ARTICLES_PER_TICKER:
                        break  # pre-fetch cap reached for this ticker
                    seen_urls.add(u)
                    pool.append(art)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("GNews query failed (%s): %s", q, exc)

    # Flatten capped pools into the final pre-fetch queue (max 3/ticker already enforced above)
    prefetch_queue: list[dict[str, Any]] = []
    for ticker, pool in ticker_article_pool.items():
        LOGGER.info("[News Fetcher][DEBUG] ticker=%s pre-fetch queue size=%s (cap=%s)", ticker, len(pool), NEWS_MAX_ARTICLES_PER_TICKER)
        prefetch_queue.extend(pool)

    LOGGER.info("[News Fetcher] Pre-fetch queue total=%s (max %s/ticker across %s tickers)", len(prefetch_queue), NEWS_MAX_ARTICLES_PER_TICKER, len(ticker_article_pool))

    if not prefetch_queue:
        return []

    scraper = AsyncNewsScraper(
        timeout_sec=NEWS_FETCH_TIMEOUT_SEC,
        max_concurrent=NEWS_MAX_CONCURRENT,
        max_per_host=NEWS_MAX_PER_HOST,
    )
    results = await scraper.fetch_many(prefetch_queue)

    LOGGER.info("[News Fetcher] Total articles scraped: %s", len(results))
    return results


def scrape_centralized_news(days_back: int = NEWS_DAYS_BACK, target_tickers: list[str] | None = None) -> list[dict[str, Any]]:
    """Sync wrapper for async scraper; safe to call from sync code."""
    try:
        return asyncio.run(_scrape_news_async(days_back, target_tickers))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("[News Fetcher] Centralized scraping failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# TICKER MAPPING
# ---------------------------------------------------------------------------

def map_tickers_to_news(
    news_items: list[dict[str, Any]],
    vn100_tickers: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map full-body articles to candidate tickers; cap at 3 articles/ticker.

    Returns:
        ticker_news_dict: ticker → list of formatted article strings (for LLM prompt)
        ticker_urls_dict: ticker → list of raw source URLs (ground-truth, not LLM-extracted)
    """
    ticker_news_dict: dict[str, list[str]] = {t: [] for t in vn100_tickers}
    ticker_urls_dict: dict[str, list[str]] = {t: [] for t in vn100_tickers}
    ticker_set = set(vn100_tickers)

    for item in news_items:
        text = f"{item.get('title', '')} {item.get('content', '')}"
        found = set(re.findall(r"\b[A-Z]{3}\b", text))
        matches = found & ticker_set
        LOGGER.info(
            "[News Mapper][DEBUG] url=%s title=%r extracted_tickers=%s matched=%s content_len=%s",
            item.get("url", ""),
            str(item.get("title", ""))[:120],
            sorted(found),
            sorted(matches),
            len(str(item.get("content", ""))),
        )
        for ticker in matches:
            if len(ticker_news_dict[ticker]) >= NEWS_MAX_ARTICLES_PER_TICKER:
                continue
            article_url = item.get("url", "")
            formatted = (
                f"Source URL: {article_url}\n"
                f"Title: {item.get('title', '')}\n"
                f"Full Article Body:\n{item.get('content', '')[:6000]}\n---\n"
            )
            ticker_news_dict[ticker].append(formatted)
            # Track raw URL independently — do NOT rely on LLM to re-extract it
            if article_url and article_url not in ticker_urls_dict[ticker]:
                ticker_urls_dict[ticker].append(article_url)

    for ticker, articles in ticker_news_dict.items():
        LOGGER.info("[News Mapper][DEBUG] ticker=%s articles_before_llm=%s urls_tracked=%s",
                    ticker, len(articles), len(ticker_urls_dict[ticker]))

    final_news = {t: news for t, news in ticker_news_dict.items() if news}
    final_urls = {t: urls for t, urls in ticker_urls_dict.items() if urls}
    LOGGER.info(
        "[News Fetcher] Tickers with news: %s; cap=%s articles/ticker",
        len(final_news),
        NEWS_MAX_ARTICLES_PER_TICKER,
    )
    return final_news, final_urls


# ---------------------------------------------------------------------------
# GEMINI SENTIMENT
# ---------------------------------------------------------------------------

def get_batch_sentiment_scores(ticker_news_dict: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Call Gemini API for batch sentiment; fault-tolerant with retries.

    Uses the new `google-genai` SDK (Client + client.models.generate_content).
    The legacy `google.generativeai` package is deprecated upstream.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None or genai_types is None:
        LOGGER.warning("[News Analyst] GEMINI_API_KEY not set or google-genai missing. Defaulting to Neutral.")
        return {
            t: {"sentiment_score": 0.0, "reasoning_vi": "Không có API Key", "source_urls": []}
            for t in ticker_news_dict
        }

    # New SDK: stateless Client; model name does not need the "models/" prefix.
    client = genai.Client(api_key=api_key)
    gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    LOGGER.info("[News Analyst][DEBUG] Initializing Gemini (google-genai) model=%s", gemini_model_name)

    user_prompt = (
        "Phân tích sentiment cho các ticker dưới đây dựa trên FULL ARTICLE BODY đã scrape từ <p> tags.\n"
        "Không suy luận từ title nếu body mâu thuẫn. Tìm catalyst, risk, clickbait reversal.\n\n"
    )
    for ticker, news_list in ticker_news_dict.items():
        LOGGER.info("[News Analyst][DEBUG] ticker=%s articles_sent_to_llm=%s", ticker, len(news_list))
        user_prompt += f"=== TICKER: {ticker} ===\n" + "".join(news_list[:NEWS_MAX_ARTICLES_PER_TICKER]) + "\n\n"
    user_prompt += (
        "Return strictly RAW JSON with fields: catalyst, risk, sentiment_score, reasoning_vi, source_urls. "
        "reasoning_vi MUST start with 'Kết luận Tâm lý (Sentiment Score):'."
    )

    generate_config = genai_types.GenerateContentConfig(
        system_instruction=NEWS_ANALYST_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )

    max_retries = 3
    delays = [2, 5, 10]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=gemini_model_name,
                contents=user_prompt,
                config=generate_config,
            )
            raw_response = (response.text or "") if hasattr(response, "text") else ""
            LOGGER.info("[News Analyst][DEBUG] Gemini raw response:\n%s", raw_response)
            try:
                result_json: dict[str, Any] = json.loads(raw_response)
            except json.JSONDecodeError:
                LOGGER.exception("[News Analyst][DEBUG] JSON parse failed. raw_response=%r", raw_response)
                cleaned = raw_response.strip()
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if not match:
                    raise
                LOGGER.info("[News Analyst][DEBUG] Retrying JSON parse after markdown/code-fence cleanup.")
                result_json = json.loads(match.group(0))
            for t in result_json:
                score = float(result_json[t].get("sentiment_score", 0.0))
                result_json[t]["sentiment_score"] = max(-1.0, min(1.0, score))
                catalyst = result_json[t].get("catalyst", "Xúc tác (Catalyst): Không có bằng chứng rõ ràng.")
                risk = result_json[t].get("risk", "Rủi ro (Risk): Không có rủi ro nổi bật trong body.")
                reasoning = result_json[t].get("reasoning_vi", "")
                if not str(reasoning).startswith("Kết luận Tâm lý (Sentiment Score):"):
                    reasoning = f"Kết luận Tâm lý (Sentiment Score): {result_json[t]['sentiment_score']:+.2f}. {reasoning}"
                result_json[t]["catalyst"] = catalyst
                result_json[t]["risk"] = risk
                result_json[t]["reasoning_vi"] = f"{catalyst}\n{risk}\n{reasoning}"
            LOGGER.info("[News Analyst] Batch evaluated %s tickers.", len(result_json))
            time.sleep(2.5)
            return result_json
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[News Analyst] Gemini attempt %s failed: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])

    LOGGER.error("[News Analyst] All retries exhausted. Defaulting to Neutral.")
    return {
        t: {"sentiment_score": 0.0, "reasoning_vi": "Lỗi gọi API", "source_urls": []}
        for t in ticker_news_dict
    }


# ---------------------------------------------------------------------------
# ARBITRATOR LOGIC
# ---------------------------------------------------------------------------

def make_final_decision(
    pred_5d_probs: list[float],
    sentiment_score: float,
    ticker: str = "UNKNOWN",
    pred_20d_probs: list[float] | None = None,
    log_detail: bool = False,
) -> int:
    """Dual-horizon deterministic veto system. Per-ticker logs only for selected/top candidates."""
    pred_5d = int(np.argmax(pred_5d_probs))
    pred_20d = int(np.argmax(pred_20d_probs)) if pred_20d_probs else pred_5d

    if pred_5d == 2 and sentiment_score < -0.5:
        if log_detail:
            LOGGER.info("[SAFETY OVERRIDE] 5d Buy rejected for %s (sentiment=%.2f)", ticker, sentiment_score)
        return 1

    if pred_5d == 2:
        if log_detail:
            LOGGER.info("[Arbitrator] %s: 5d UP => BUY / STRONG HOLD.", ticker)
        return 2

    if pred_5d == 1 and pred_20d == 2:
        if log_detail:
            LOGGER.info("[Arbitrator] %s: 5d SIDEWAYS + 20d UP => TREND ACTIVE - HOLD.", ticker)
        return 2

    if pred_5d == 0 and pred_20d == 0:
        if sentiment_score > 0.5:
            if log_detail:
                LOGGER.info("[Portfolio Manager] VETO: %s 5d/20d DOWN but sentiment=%.2f => HOLD.", ticker, sentiment_score)
            return 1
        if log_detail:
            LOGGER.info("[Arbitrator] %s: 5d DOWN + 20d DOWN => FULL EXIT.", ticker)
        return 0

    if pred_5d == 0 and sentiment_score > 0.5:
        if log_detail:
            LOGGER.info("[Portfolio Manager] VETO: %s 5d DOWN but sentiment=%.2f => HOLD.", ticker, sentiment_score)
        return 1

    return pred_5d


# ---------------------------------------------------------------------------
# PIPELINE TIE-IN
# ---------------------------------------------------------------------------

def evaluate_trades_batch(
    stacking_predictions: dict[str, Any],
    vn100_tickers: list[str],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """
    Main execution pipeline.

    Args:
        stacking_predictions: dual-horizon {"5d": {...}, "20d": {...}} produced by the
            Stacking GBDT (XGBoost+LightGBM+CatBoost → logistic meta) model.
            Falls back to a flat {ticker: probs} mapping for legacy callers.
        vn100_tickers: candidate tickers (the Top-N pool from the arbitrator gate).

    Returns:
        final_decisions: {ticker: 0|1|2}
        all_sentiments: {ticker: sentiment_data}
    """
    LOGGER.info("=" * 50)
    LOGGER.info("[Pipeline] Starting Batch Trade Evaluation...")

    predictions_5d = stacking_predictions.get("5d", stacking_predictions)
    target_tickers = [
        ticker
        for ticker, _probs in sorted(
            ((t, predictions_5d[t]) for t in vn100_tickers if t in predictions_5d),
            key=lambda item: item[1][2],
            reverse=True,
        )[:25]
    ]

    # Step A: Scrape & Map
    LOGGER.info("[Pipeline] Scraping Centralized News Pool (async) for target_tickers=%s", target_tickers)
    raw_news = scrape_centralized_news(target_tickers=target_tickers)
    ticker_news_dict, ticker_urls_dict = map_tickers_to_news(raw_news, vn100_tickers)

    # Step B: Batch Sentiment
    all_sentiments: dict[str, dict[str, Any]] = {}
    tickers_with_news = list(ticker_news_dict.keys())
    batch_size = 5

    for i in range(0, len(tickers_with_news), batch_size):
        batch_tickers = tickers_with_news[i : i + batch_size]
        batch_dict = {t: ticker_news_dict[t] for t in batch_tickers}
        LOGGER.info("[Pipeline] Sending Batch %s to Gemini (%s tickers)...", i // batch_size + 1, len(batch_tickers))
        batch_results = get_batch_sentiment_scores(batch_dict)
        # Guarantee source_urls are populated from ground-truth tracker regardless of LLM reliability
        for t, result in batch_results.items():
            if not result.get("source_urls"):
                result["source_urls"] = ticker_urls_dict.get(t, [])
                if result["source_urls"]:
                    LOGGER.info("[Pipeline] Patched source_urls for %s from ground-truth tracker (%s urls)", t, len(result["source_urls"]))
        all_sentiments.update(batch_results)

    # Step C: Arbitration
    final_decisions: dict[str, int] = {}
    predictions_20d = stacking_predictions.get("20d", {})

    log_tickers = {
        ticker
        for ticker, _probs in sorted(
            ((t, predictions_5d[t]) for t in vn100_tickers if t in predictions_5d),
            key=lambda item: item[1][2],
            reverse=True,
        )[:10]
    }
    LOGGER.info("[Arbitrator] Detailed per-ticker logs limited to Top %s candidates: %s", len(log_tickers), sorted(log_tickers))

    for ticker in predictions_5d:
        stacking_probs = list(predictions_5d[ticker])
        pred_20d_probs = list(predictions_20d.get(ticker, stacking_probs))

        if ticker in all_sentiments:
            sentiment_score = all_sentiments[ticker].get("sentiment_score", 0.0)
        else:
            sentiment_score = 0.0
            all_sentiments[ticker] = {
                "sentiment_score": 0.0,
                "reasoning_vi": "Không có tin tức đáng kể.",
                "source_urls": [],
            }
            stacking_probs[2] *= 0.95  # Activity penalty for tickers with no news coverage

        decision = make_final_decision(
            stacking_probs,
            sentiment_score,
            ticker,
            pred_20d_probs,
            log_detail=ticker in log_tickers,
        )
        final_decisions[ticker] = decision

    LOGGER.info("[Pipeline] Batch Evaluation Complete.")
    LOGGER.info("=" * 50)
    return final_decisions, all_sentiments


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_ticker = "VNM"
    test_probs_5d = [0.1, 0.2, 0.7]
    test_probs_20d = [0.1, 0.2, 0.7]
    decision = make_final_decision(test_probs_5d, 0.0, test_ticker, test_probs_20d)
    print(f"Decision={decision}")