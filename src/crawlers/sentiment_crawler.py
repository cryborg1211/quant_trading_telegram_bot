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
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import CONFIG
from src.data import price_lookup  # fresh-parquet liquidity ranking (stock_ohlcv retired)

LOGGER = logging.getLogger(__name__)


def _is_transient_gemini_error(exc: BaseException) -> bool:
    """True for transient Gemini server errors worth retrying (503 / UNAVAILABLE).

    The new google-genai SDK raises ``ServerError`` with a numeric ``code`` for
    5xx responses; older paths surface the status only in the message. Match
    both, and NEVER retry client-side faults (4xx, JSON parse, bad key) — those
    are permanent and would just burn the budget.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (502, 503, 504):
        return True
    text = str(exc).upper()
    return any(tok in text for tok in ("503", "502", "504", "UNAVAILABLE", "OVERLOADED"))

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


# Probed live 12-06-2026. cafef.vn native *.rss returns 502 for every path
# (origin broken behind their CDN) and vietstock.vn/rss/chung-khoan.rss is an
# empty 200 — both sources are restored through Google News RSS site-scoped
# proxies (when:2d keeps them inside the crawl window; links are
# news.google.com redirects, which still resolve to the article).
RSS_FEEDS = {
    "cafef_gnews": ("https://news.google.com/rss/search?"
                    "q=site%3Acafef.vn%20ch%E1%BB%A9ng%20kho%C3%A1n%20when%3A2d"
                    "&hl=vi&gl=VN&ceid=VN:vi"),
    "vietstock_gnews": ("https://news.google.com/rss/search?"
                        "q=site%3Avietstock.vn%20ch%E1%BB%A9ng%20kho%C3%A1n%20when%3A2d"
                        "&hl=vi&gl=VN&ceid=VN:vi"),
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
    # Per-feed cap: the Google News proxies return up to 100 items/feed; every
    # NEW item is one Gemini call. Feeds are newest-first, so the cap keeps the
    # freshest slice.
    RSS_MAX_ITEMS_PER_FEED = 30
    # Chunk size for BOTH the batched Gemini call (`_score_batch` — one call
    # per chunk instead of one per article) and the incremental DB append (an
    # interrupt loses at most ONE in-flight batch). 2026-07-13 incident: a
    # Ctrl+C 420s into per-article scoring billed ~7 min of calls and
    # persisted zero rows; 14-07 batching also cut a 300-article run from
    # 300 calls to 12.
    SCORE_APPEND_CHUNK = 25

    def __init__(
        self,
        db_path: str | None = None,
        model_name: str | None = None,
    ):
        self.db_path = db_path or str(CONFIG.paths.duckdb_path)
        # Model precedence: explicit arg → .env GEMINI_MODEL → config default.
        # The env pin governs BOTH this crawler and the arbitrator (parity with
        # quant_agent_arbitrator's os.environ.get("GEMINI_MODEL", ...)).
        # Floating aliases are banned here: "gemini-flash-latest" silently
        # became gemini-3.5-flash and caused the 2026-07-07 JSON-drift incident
        # plus intermittent 503s — pin GA models only.
        # Strip legacy "models/" prefix — new SDK accepts bare model name only.
        raw_model = model_name or os.getenv("GEMINI_MODEL") or CONFIG.sentiment.gemini_model
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

        # Score + persist in chunks: each chunk is ONE batched Gemini call
        # (`_score_batch`) and is INSERTed before the next chunk is scored, so
        # an interrupt keeps every fully-scored chunk (the `existing_urls` read
        # at the top makes the next run skip them). An exception mid-chunk
        # propagates exactly as before — no new swallowing.
        appended: list[pd.DataFrame] = []
        for start in range(0, len(new_items), self.SCORE_APPEND_CHUNK):
            chunk_items = new_items[start:start + self.SCORE_APPEND_CHUNK]
            chunk_df = pd.DataFrame(self._score_batch(chunk_items))
            self._append_rows(chunk_df)
            appended.append(chunk_df)
        df = pd.concat(appended, ignore_index=True)
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
                kept = 0
                for entry in parsed.entries:
                    if kept >= self.RSS_MAX_ITEMS_PER_FEED:
                        break
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
                        kept += 1
                if parsed.entries:
                    LOGGER.info("[SentimentCrawler] RSS %s: kept %s/%s entries.",
                                source, kept, len(parsed.entries))
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
        # URL dedupe first, then normalized-title dedupe: the Google News
        # proxy feeds and the native feeds carry the SAME article under
        # different URLs (news.google.com redirect vs real link) — without the
        # title pass each duplicate costs one extra Gemini call.  Google News
        # appends " - <Source>" to titles; strip it before normalizing.
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        out = []
        for item in items:
            key = item.url.strip().lower()
            if not key or key in seen_urls:
                continue
            title_norm = re.sub(r"\s+-\s+[^-]{2,40}$", "", item.title).strip().lower()
            title_norm = re.sub(r"\s+", " ", title_norm)
            if title_norm and title_norm in seen_titles:
                continue
            seen_urls.add(key)
            if title_norm:
                seen_titles.add(title_norm)
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

    @retry(
        reraise=True,
        retry=retry_if_exception(_is_transient_gemini_error),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
    )
    def _generate_content(self, user_message: str) -> str:
        """Single Gemini call, retried with exponential backoff on 503/UNAVAILABLE.

        Only the network call lives here so tenacity retries the TRANSIENT
        failure and nothing else — JSON parsing / clamping happen in the caller
        and a parse error is never retried. After 5 exhausted attempts the last
        transient error is re-raised (``reraise=True``) for the caller to fall
        back on.
        """
        config_kwargs: dict = dict(
            system_instruction=DEFAULT_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        )
        # COST: gemini-2.5-flash enables "thinking" by DEFAULT, and thinking
        # tokens bill at the OUTPUT rate (~8x input) — measured ~$1.5-2.0 per
        # full pipeline run (~300 scoring calls) before this was disabled
        # (2026-07-12). Deterministic JSON extraction needs none of it;
        # budget=0 turns it off (~85% cost cut). Guarded so an SDK/model
        # without ThinkingConfig support still works.
        if hasattr(genai_types, "ThinkingConfig"):
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=0
            )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        return (response.text or "").strip()

    @staticmethod
    def parse_gemini_json(raw: str) -> dict:
        """Parse a Gemini sentiment response, tolerating format drift.

        2026-07-07 incident: gemini-flash-latest began returning 200s whose
        bodies fail strict ``json.loads`` in two shapes — valid JSON followed
        by trailing content ("Extra data: line 9"), and objects broken mid-way
        by unescaped quotes inside the Vietnamese ``reason`` string
        ("Expecting ',' delimiter: line 7"). Ladder:
          1. strict parse (fences stripped),
          2. ``raw_decode`` of the FIRST object — survives trailing junk,
          3. regex salvage of the numeric fields — survives a broken reason
             string (scores precede it in the schema).
        Raises ``ValueError`` when nothing is salvageable.
        """
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        if start >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw[start:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        m_score = re.search(r'"(?:sentiment_)?score"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
        m_mag = re.search(r'"magnitude"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
        if m_score:
            return {
                "sentiment_score": float(m_score.group(1)),
                "magnitude": float(m_mag.group(1)) if m_mag else 0.0,
                "reason": "salvaged: malformed Gemini JSON (numeric fields only)",
            }
        raise ValueError("no parseable JSON object or salvageable score fields")

    def score_payload(
        self, *, ticker: str | None, dt: date, title: str, content: str
    ) -> dict:
        """Score one article via Gemini → clamped {sentiment_score, magnitude, reason}.

        Reusable scoring core shared by the live crawl (`_score_item`) and the
        503 backfill script. On persistent failure (transient retries exhausted
        OR a permanent error) returns a neutral payload whose ``reason`` is
        prefixed ``Gemini fallback:`` — the exact marker the backfill query
        targets.
        """
        score = {"sentiment_score": 0.0, "magnitude": 0.0, "reason": "Neutral fallback"}
        if self._client is not None and genai_types is not None:
            # DEFAULT_PROMPT is the system_instruction; the user message carries
            # only article data so the model focuses on content, not directions.
            user_message = (
                f"TICKER: {ticker or 'MARKET_WIDE'}\n"
                f"DATE: {dt.isoformat()}\n"
                f"TITLE: {title}\n"
                f"CONTENT: {content[: CONFIG.sentiment.article_char_limit]}"
            )
            try:
                raw = self._generate_content(user_message)
                parsed = self.parse_gemini_json(raw)
                score = {
                    "sentiment_score": float(parsed.get("sentiment_score", parsed.get("score", 0.0))),
                    "magnitude": float(parsed.get("magnitude", 0.0)),
                    "reason": str(parsed.get("reason", ""))[:1000],
                }
            except Exception as exc:
                LOGGER.warning("[SentimentCrawler] Gemini scoring failed (%s): %s", title[:60], exc)
                score["reason"] = f"Gemini fallback: {exc}"

        score["sentiment_score"] = max(-1.0, min(1.0, float(score["sentiment_score"])))
        score["magnitude"] = max(0.0, min(1.0, float(score["magnitude"])))
        return score

    def _item_row(self, item: NewsItem, sentiment: float, magnitude: float, reason: str) -> dict:
        """Assemble one `hist_sentiment_llm_labeled` row from an item + scores (pure)."""
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
            "reason": reason,
            "url": item.url,
            "sentiment_nlp": sentiment,
            "impact_force": sentiment * magnitude,
            "is_market_wide": bool(item.is_market_wide),
        }

    def _fallback_row(self, item: NewsItem, reason: str) -> dict:
        """Neutral row for a failed scoring — `reason` keeps the 'Gemini fallback:'
        marker when applicable so scripts/backfill_sentiment_503.py can re-score it."""
        return self._item_row(item, 0.0, 0.0, reason)

    def _score_item(self, item: NewsItem) -> dict:
        """Single-article scoring path (one Gemini call). Kept for parity with
        `score_payload` consumers (503 backfill); the live crawl loop uses the
        batched `_score_batch` instead."""
        score = self.score_payload(
            ticker=item.ticker,
            dt=item.date,
            title=item.title,
            content=item.text,
        )
        return self._item_row(
            item, score["sentiment_score"], score["magnitude"], score["reason"]
        )

    @staticmethod
    def _parse_batch_json(raw: str) -> dict[int, dict]:
        """Parse a batched scoring response into {article_id: obj}.

        Tolerates code fences, a top-level object wrapping the array (e.g.
        {"results": [...]}), and items missing "id" (positional fallback).
        Raises on anything that yields no usable array — the caller degrades
        the whole batch to neutral fallback rows.
        """
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = min((i for i in (raw.find("["), raw.find("{")) if i >= 0), default=-1)
        if start < 0:
            raise ValueError("no JSON payload in batch response")
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            raise ValueError("batch response is not a JSON array")
        out: dict[int, dict] = {}
        for pos, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("id", pos))
            except (TypeError, ValueError):
                idx = pos
            out[idx] = obj
        return out

    def _score_batch(self, items: list[NewsItem]) -> list[dict]:
        """Score a chunk of articles with ONE Gemini call.

        Cost surgery 14-07-26: per-article scoring re-sent the ~1k-token system
        prompt N times per run (the dominant input cost) and made N sequential
        calls (~60 min wall-clock, N× the 503 surface). Batching by
        SCORE_APPEND_CHUNK cuts a 300-article run from 300 calls to 12.

        Failure contract: any API/parse failure degrades the WHOLE batch (or
        the missing items) to neutral rows carrying the 'Gemini fallback:'
        reason marker — the exact marker scripts/backfill_sentiment_503.py
        targets, so individual rows stay heal-able without re-crawling.
        """
        if self._client is None or genai_types is None:
            return [self._fallback_row(it, "Neutral fallback") for it in items]

        blocks = []
        for i, it in enumerate(items):
            blocks.append(
                f"=== ARTICLE {i} ===\n"
                f"TICKER: {it.ticker or 'MARKET_WIDE'}\n"
                f"DATE: {it.date.isoformat()}\n"
                f"TITLE: {it.title}\n"
                f"CONTENT: {(it.text or '')[: CONFIG.sentiment.article_char_limit]}"
            )
        user_message = (
            "Chấm sentiment cho TỪNG bài viết dưới đây theo đúng khung phân tích.\n"
            "Trả về STRICT JSON ARRAY, mỗi phần tử một object:\n"
            '{"id": <số ARTICLE>, "sentiment_score": float -1..1, '
            '"magnitude": float 0..1, "reason": "tiếng Việt, <= 3 câu, có evidence"}\n'
            "KHÔNG thêm bất kỳ text nào ngoài JSON array.\n\n"
            + "\n\n".join(blocks)
        )

        try:
            raw = self._generate_content(user_message)
            parsed = self._parse_batch_json(raw)
        except Exception as exc:  # noqa: BLE001 — batch failure must not kill the crawl
            LOGGER.warning(
                "[SentimentCrawler] Batch scoring failed (%s items): %s", len(items), exc
            )
            return [self._fallback_row(it, f"Gemini fallback: batch {exc}") for it in items]

        rows: list[dict] = []
        for i, it in enumerate(items):
            obj = parsed.get(i)
            if obj is None:
                rows.append(self._fallback_row(it, "Gemini fallback: batch missing id"))
                continue
            try:
                sentiment = max(-1.0, min(1.0, float(obj.get("sentiment_score", obj.get("score", 0.0)))))
                magnitude = max(0.0, min(1.0, float(obj.get("magnitude", 0.0))))
            except (TypeError, ValueError):
                rows.append(self._fallback_row(it, "Gemini fallback: batch bad fields"))
                continue
            rows.append(self._item_row(it, sentiment, magnitude, str(obj.get("reason", ""))[:1000]))
        return rows

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

        # PK guard: the live table has PRIMARY KEY (ticker, date, title) and the
        # crawl window overlaps yesterday, so re-crawled articles violate it and
        # kill the whole EOD pipeline (crashed 16-07 + 17-07). Dedup within the
        # batch, then anti-join against rows already in the table.
        df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")

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
            conn.execute(
                """
                INSERT INTO hist_sentiment_llm_labeled BY NAME
                SELECT * FROM df d
                WHERE NOT EXISTS (
                    SELECT 1 FROM hist_sentiment_llm_labeled t
                    WHERE t.ticker = d.ticker
                      AND t.date   = d.date
                      AND t.title  = d.title
                )
                """
            )


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