import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from urllib.parse import urlparse

import duckdb  # type: ignore[import-untyped]
import pandas as pd

from config.settings import CONFIG
from src.data import price_lookup  # fresh-parquet liquidity ranking (stock_ohlcv retired)

LOGGER = logging.getLogger(__name__)

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None

try:
    from gnews import GNews  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    GNews = None

try:
    from googlenewsdecoder import new_decoderv1  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    new_decoderv1 = None

# New official Google GenAI SDK (replaces deprecated `google.generativeai`).
# Install: pip install google-genai
try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


# Probed live 12-06-2026. cafef.vn/*.rss returns 502 (edge-blocked) and
# vietstock.vn/rss/chung-khoan.rss returns 200 with 0 items — both dropped.
RSS_FEEDS = {
    "vietstock_stocks": "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "vietstock_corporate": "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "vietstock_macro": "https://vietstock.vn/761/kinh-te/vi-mo.rss",
    "vneconomy_stocks": "https://vneconomy.vn/chung-khoan.rss",
    "tinnhanhchungkhoan": "https://www.tinnhanhchungkhoan.vn/rss/chung-khoan-7.rss",
    "tnck_corporate": "https://www.tinnhanhchungkhoan.vn/rss/doanh-nghiep-37.rss",
}

DEFAULT_PROMPT = """
Bạn là Senior Vietnam Equity Sentiment Analyst trong hệ multi-agent kiểu TradingAgents.

Nhiệm vụ: đọc tin tức, tách tín hiệu đầu tư có thể giao dịch, chấm sentiment theo tác động kỳ vọng lên giá cổ phiếu/ngành/thị trường Việt Nam trong ngắn-trung hạn.

Bắt buộc phân tích theo khung:
1. Direction: tin hỗ trợ hay gây áp lực giá?
2. Materiality: tác động có đáng kể không, hay chỉ nhiễu?
3. Horizon: tác động tức thời, vài phiên, hay dài hơn?
4. Breadth: tác động riêng ticker, ngành, hay toàn thị trường?
5. Evidence: nêu chi tiết trong bài làm căn cứ.
6. Counterpoint: rủi ro diễn giải ngược/không chắc chắn.

Schema JSON hợp lệ:
{
  "sentiment_score": float -1..1,
  "magnitude": float 0..1,
  "horizon": "intraday|short_term|medium_term|unclear",
  "breadth": "ticker|sector|market|macro|unclear",
  "materiality": "low|medium|high",
  "reason": "tiếng Việt, <= 5 câu, có evidence + counterpoint"
}

Quy ước chấm:
- +0.7..+1.0: catalyst tích cực rõ, xác suất ảnh hưởng giá cao.
- +0.2..+0.6: tích cực vừa/gián tiếp.
- -0.2..-0.6: tiêu cực vừa/gián tiếp.
- -0.7..-1.0: rủi ro/catalyst tiêu cực rõ.
- Gần 0: không liên quan, đã phản ánh, thiếu bằng chứng, hoặc tác động hai chiều.
- magnitude đo độ mạnh/độ chắc của tác động, không phải hướng.
- Không phóng đại tiêu đề giật gân; ưu tiên dữ kiện định lượng, chính sách, KQKD, dòng tiền, thanh khoản, pháp lý.

Chỉ trả JSON, không markdown, không giải thích ngoài JSON.
"""


@dataclass
class NewsItem:
    date: date
    title: str
    url: str
    text: str
    ticker: str | None = None
    is_market_wide: bool = False


class SentimentCrawler:
    """Daily Vietnamese market/news sentiment crawler -> hist_sentiment_llm_labeled."""

    # Hard budget guard: Gemini scoring is paid per article — the crawler must
    # NEVER backfill deep history regardless of config/DB state. 3 days max.
    MAX_LOOKBACK_DAYS = 3

    def __init__(
        self,
        db_path: str | None = None,
        model_name: str | None = None,
    ):
        self.db_path = db_path or str(CONFIG.paths.duckdb_path)
        # Strip legacy "models/" prefix — new SDK accepts bare model name only.
        raw_model = model_name or CONFIG.sentiment.gemini_model
        self.model_name = raw_model.removeprefix("models/")
        self._client: Any | None = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and genai is not None and genai_types is not None:
            self._client = genai.Client(api_key=api_key)
            LOGGER.info("[SentimentCrawler] Gemini client (google-genai) initialised; model=%s", self.model_name)
        else:
            LOGGER.warning("[SentimentCrawler] GEMINI_API_KEY not set or google-genai missing — scoring disabled.")

    def update_daily_sentiment(self, lookback_days: int | None = None, max_tickers: int | None = None) -> pd.DataFrame:
        start_date = self._compute_start_date(lookback_days)
        end_date = datetime.now().date()
        tickers = self._active_tickers(limit=max_tickers or CONFIG.sentiment.max_tickers)

        LOGGER.info("[SentimentCrawler] Crawl window: %s..%s; tickers=%s", start_date, end_date, len(tickers))
        existing_urls = self._existing_urls()
        items = []
        items.extend(self._fetch_rss_items(start_date=start_date, end_date=end_date))
        items.extend(
            self._fetch_gnews_items(
                tickers=tickers, start_date=start_date, end_date=end_date,
                skip_urls=existing_urls,
            )
        )

        deduped = self._dedupe(items)
        new_items = [
            item for item in deduped
            if item.url and item.url not in existing_urls
            and not self._is_covered_warrant(item.ticker)
            and start_date <= item.date <= end_date
        ]

        if not new_items:
            LOGGER.info("[SentimentCrawler] No new sentiment articles found.")
            return pd.DataFrame()

        rows = [self._score_item(item) for item in new_items]
        df = pd.DataFrame(rows)
        self._append_rows(df)
        LOGGER.info("[SentimentCrawler] Appended %s sentiment rows to hist_sentiment_llm_labeled.", len(df))
        return df

    def _compute_start_date(self, lookback_days: int | None) -> date:
        today = datetime.now().date()
        if lookback_days is None:
            lookback_days = (
                CONFIG.sentiment.rss_lookback_monday_days
                if today.weekday() == 0
                else CONFIG.sentiment.rss_lookback_weekday_days
            )

        # Hard safeguard: clamp to [1, MAX_LOOKBACK_DAYS]. The old path took
        # min(MAX(date) in DB, fallback) which triggered deep historical
        # backfills (and Gemini bills) whenever the table had a stale gap.
        lookback_days = max(1, min(int(lookback_days), self.MAX_LOOKBACK_DAYS))
        return today - timedelta(days=lookback_days)

    @staticmethod
    def _is_covered_warrant(ticker: str | None) -> bool:
        # Standard HOSE covered-warrant format: 'C' + underlying + maturity code,
        # exactly 8 chars (e.g. CHPG2613, CVPB2210). No news value, never traded
        # by this system — scoring them burns Gemini budget for nothing.
        if not ticker:
            return False
        t = str(ticker).strip().upper()
        return len(t) == 8 and t.startswith("C")

    def _active_tickers(self, limit: int) -> list[str]:
        # Most-liquid names by recent volume, read from the FRESH parquet vintage
        # (stock_ohlcv was retired). price_lookup opens its own in-memory
        # connection, so no DuckDB *file* handle is taken here.
        raw = price_lookup.top_tickers_by_volume(limit=limit, lookback_days=10)
        tickers = [t for t in raw if not self._is_covered_warrant(t)]
        dropped = len(raw) - len(tickers)
        if dropped:
            LOGGER.info("[SentimentCrawler] Filtered %s covered warrants from ticker list.", dropped)
        return tickers

    def _fetch_rss_items(self, start_date: date, end_date: date) -> list[NewsItem]:
        if feedparser is None:
            LOGGER.warning("[SentimentCrawler] feedparser missing. Skipping RSS sentiment.")
            return []

        items = []
        for source, feed_url in RSS_FEEDS.items():
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries:
                    published = self._parse_entry_date(entry)
                    if published and not (start_date <= published <= end_date):
                        continue
                    title = self._clean_text(getattr(entry, "title", ""))
                    summary = self._clean_text(getattr(entry, "summary", ""))
                    url = getattr(entry, "link", "")
                    if title and url:
                        items.append(
                            NewsItem(
                                date=published or end_date,
                                title=title,
                                url=url,
                                text=f"{title}\n{summary}",
                                is_market_wide=True,
                            )
                        )
                if parsed.entries:
                    LOGGER.info("[SentimentCrawler] RSS %s: %s entries scanned.", source, len(parsed.entries))
                else:
                    LOGGER.warning("[SentimentCrawler] RSS %s returned 0 entries — feed may be dead/blocked: %s", source, feed_url)
            except Exception as exc:
                LOGGER.warning("[SentimentCrawler] RSS %s failed: %s", source, exc)
        return items

    # Wall-clock budget for the GNews loop. URL decoding sleeps ~1s/article and
    # full-article fetches add several seconds each — without a deadline the
    # loop can silently eat 5+ minutes of the EOD pipeline.
    GNEWS_BUDGET_SECONDS = 180

    def _fetch_gnews_items(
        self,
        tickers: Iterable[str],
        start_date: date,
        end_date: date,
        skip_urls: set[str] | None = None,
    ) -> list[NewsItem]:
        if GNews is None:
            LOGGER.warning("[SentimentCrawler] gnews missing. Skipping ticker GNews sentiment.")
            return []

        skip_urls = skip_urls or set()
        client = GNews(language="vi", country="VN", max_results=CONFIG.sentiment.gnews_max_results)
        items = []
        loop_start = time.monotonic()
        for ticker in tickers:
            elapsed = time.monotonic() - loop_start
            if elapsed > self.GNEWS_BUDGET_SECONDS:
                LOGGER.warning(
                    "[SentimentCrawler] GNews budget exhausted (%.0fs > %ss) — stopping at %s.",
                    elapsed, self.GNEWS_BUDGET_SECONDS, ticker,
                )
                break
            query = f'"{ticker}" chứng khoán OR cổ phiếu site:cafef.vn OR site:vietstock.vn'
            t0 = time.monotonic()
            try:
                results = client.get_news(query) or []
                kept = 0
                for result in results:
                    # Cheap filters FIRST — date and title — so we never pay the
                    # ~1s Google-URL decode or the full-article download for
                    # stale or already-scored entries.
                    published = self._parse_gnews_date(result.get("published date")) or end_date
                    if not (start_date <= published <= end_date):
                        continue
                    title = self._clean_text(result.get("title", ""))
                    if not title:
                        continue
                    url = self._decode_google_url(result.get("url", ""))
                    if not url or url in skip_urls:
                        continue
                    text = title
                    if hasattr(client, "get_full_article"):
                        try:
                            article = client.get_full_article(url)
                            if article and getattr(article, "text", None):
                                text = article.text[: CONFIG.sentiment.article_char_limit]
                        except Exception:
                            pass
                    items.append(NewsItem(date=published, title=title, url=url, text=text, ticker=ticker))
                    kept += 1
                LOGGER.info(
                    "[SentimentCrawler] GNews %s: kept %s/%s (%.1fs).",
                    ticker, kept, len(results), time.monotonic() - t0,
                )
                time.sleep(CONFIG.sentiment.gnews_sleep_seconds)
            except Exception as exc:
                LOGGER.warning("[SentimentCrawler] GNews %s failed: %s", ticker, exc)
        return items

    @staticmethod
    def _parse_entry_date(entry) -> date | None:
        for attr in ("published_parsed", "updated_parsed"):
            value = getattr(entry, attr, None)
            if value:
                return datetime(*value[:6]).date()
        return None

    @staticmethod
    def _parse_gnews_date(value) -> date | None:
        if not value:
            return None
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(value), fmt).date()
            except ValueError:
                continue
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None

    @staticmethod
    def _clean_text(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", str(value or ""))
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _decode_google_url(url: str) -> str:
        if not url:
            return ""
        if new_decoderv1 and "news.google.com" in urlparse(url).netloc:
            try:
                decoded = new_decoderv1(url, interval=1)
                if isinstance(decoded, dict) and decoded.get("status") and decoded.get("decoded_url"):
                    return decoded["decoded_url"]
            except Exception:
                return url
        return url

    @staticmethod
    def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
        seen = set()
        out = []
        for item in items:
            key = item.url.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _existing_urls(self) -> set[str]:
        try:
            # No `read_only=True` — DuckDB rejects mixed-config connections to
            # the same file in one process (see db_engine.py docstring).
            with duckdb.connect(self.db_path) as conn:
                rows = conn.execute("SELECT DISTINCT url FROM hist_sentiment_llm_labeled WHERE url IS NOT NULL").fetchall()
            return {str(r[0]) for r in rows}
        except Exception:
            return set()

    def _score_item(self, item: NewsItem) -> dict:
        score = {"sentiment_score": 0.0, "magnitude": 0.0, "reason": "Neutral fallback"}
        if self._client is not None and genai_types is not None:
            # DEFAULT_PROMPT is passed as system_instruction; user message contains only
            # the article data so the model can focus on content, not instructions.
            user_message = (
                f"TICKER: {item.ticker or 'MARKET_WIDE'}\n"
                f"DATE: {item.date.isoformat()}\n"
                f"TITLE: {item.title}\n"
                f"CONTENT: {item.text[: CONFIG.sentiment.article_char_limit]}"
            )
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=DEFAULT_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                raw = (response.text or "").strip()
                # response_mime_type="application/json" avoids markdown fences,
                # but keep the strip as a defensive fallback.
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                score = {
                    "sentiment_score": float(parsed.get("sentiment_score", parsed.get("score", 0.0))),
                    "magnitude": float(parsed.get("magnitude", 0.0)),
                    "reason": str(parsed.get("reason", ""))[:1000],
                }
            except Exception as exc:
                LOGGER.warning("[SentimentCrawler] Gemini scoring failed for %s: %s", item.url, exc)
                score["reason"] = f"Gemini fallback: {exc}"

        sentiment = max(-1.0, min(1.0, float(score["sentiment_score"])))
        magnitude = max(0.0, min(1.0, float(score["magnitude"])))
        # Market-wide RSS items carry no ticker — use the MARKET_WIDE sentinel
        # (same label the LLM prompt uses) so the NOT NULL filter in
        # _append_rows keeps them instead of dropping every RSS row.
        ticker = (item.ticker or "").strip().upper() or ("MARKET_WIDE" if item.is_market_wide else None)
        return {
            "date": pd.Timestamp(item.date),
            "ticker": ticker,
            "title": item.title[:1000],
            "sentiment_score": sentiment,
            "magnitude": magnitude,
            "reason": score["reason"],
            "url": item.url,
            "sentiment_nlp": sentiment,
            "impact_force": sentiment * magnitude,
            "is_market_wide": bool(item.is_market_wide),
        }

    def _append_rows(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.replace({"NaN": None, "nan": None})

        # Constraint guard: drop rows with NULL / NaN / empty ticker BEFORE the
        # insert — they violate the table constraint and crash the pipeline.
        before = len(df)
        ticker_str = df["ticker"].astype(str).str.strip()
        df = df[
            df["ticker"].notna()
            & (ticker_str != "")
            & (~ticker_str.str.lower().isin({"nan", "none", "null"}))
        ]
        if len(df) < before:
            LOGGER.warning(
                "[SentimentCrawler] Dropped %s rows with NULL/empty ticker before insert.",
                before - len(df),
            )
        if df.empty:
            return

        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hist_sentiment_llm_labeled (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    title VARCHAR,
                    sentiment_score DOUBLE,
                    magnitude DOUBLE,
                    reason VARCHAR,
                    url VARCHAR,
                    sentiment_nlp DOUBLE,
                    impact_force DOUBLE,
                    is_market_wide BOOLEAN
                )
                """
            )
            # Legacy tables predate the ticker column — BY NAME insert needs it.
            conn.execute("ALTER TABLE hist_sentiment_llm_labeled ADD COLUMN IF NOT EXISTS ticker VARCHAR")
            conn.execute("INSERT INTO hist_sentiment_llm_labeled BY NAME SELECT * FROM df")


def update_daily_sentiment(
    db_path: str | None = None,
    lookback_days: int | None = None,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    return SentimentCrawler(db_path=db_path).update_daily_sentiment(
        lookback_days=lookback_days,
        max_tickers=max_tickers,
    )


def fetch_latest_market_news(limit: int = 20) -> list[dict[str, Any]]:
    """Fetch the N most recent items from the configured Vietnamese RSS feeds.

    Used by the /news Telegram bot command. Independent of the LLM scoring
    path so it does NOT need a Gemini API key or DuckDB connection.

    Returns:
        list of {"title": str, "url": str, "source": str, "published": datetime}
        sorted by `published` DESC. Empty list if feedparser is unavailable
        or every feed errored.
    """
    if feedparser is None:
        LOGGER.warning("[News] feedparser missing — returning empty list.")
        return []

    items: list[dict[str, Any]] = []
    for source, feed_url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                title_raw = str(getattr(entry, "title", "") or "")
                title = re.sub(r"<[^>]+>", " ", title_raw)
                title = re.sub(r"\s+", " ", title).strip()
                url = str(getattr(entry, "link", "") or "")
                published_struct = (
                    getattr(entry, "published_parsed", None)
                    or getattr(entry, "updated_parsed", None)
                )
                published_dt = (
                    datetime(*published_struct[:6])
                    if published_struct
                    else datetime.min
                )
                if title and url:
                    items.append({
                        "title": title,
                        "url": url,
                        "source": source,
                        "published": published_dt,
                    })
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[News] RSS %s fetch failed: %s", source, exc)

    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]


if __name__ == "__main__":
    update_daily_sentiment()