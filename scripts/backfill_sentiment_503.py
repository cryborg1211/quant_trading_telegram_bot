"""Backfill sentiment rows corrupted by Gemini 503 fallbacks.

Targets rows in ``hist_sentiment_llm_labeled`` whose ``reason`` starts with
``Gemini fallback:`` — these were written when the Gemini API returned a
transient 503/UNAVAILABLE and the crawler defaulted the score to 0. It re-scores
each via the now tenacity-hardened ``SentimentCrawler.score_payload`` and UPDATEs
the row in place (keyed by ``url``).

IMPORTANT — no ``text_raw`` column exists. The table stores ``title`` + ``url``
only (full article text was never persisted). Re-scoring therefore uses the
stored ``title`` as content. For full-fidelity re-scoring you would re-fetch the
``url`` first; that network re-crawl is intentionally out of scope here.

Usage (dry-run by default — prints what WOULD change, writes nothing):
    python scripts/backfill_sentiment_503.py
    python scripts/backfill_sentiment_503.py --commit          # actually UPDATE
    python scripts/backfill_sentiment_503.py --commit --limit 10
"""

from __future__ import annotations
import time
import argparse
import logging
from typing import Any

import duckdb

from config.settings import CONFIG
from src.crawlers.sentiment_crawler import SentimentCrawler
from dotenv import load_dotenv
load_dotenv()

LOGGER = logging.getLogger(__name__)

_FALLBACK_PREFIX = "Gemini fallback%"
_TABLE = "hist_sentiment_llm_labeled"


def _fetch_corrupted(conn: Any, limit: int | None) -> list[tuple]:
    """Rows still carrying a Gemini-fallback reason: (url, title, ticker, date)."""
    sql = (
        f"SELECT url, title, ticker, date FROM {_TABLE} "
        "WHERE reason LIKE ? AND url IS NOT NULL "
        "ORDER BY date DESC"
    )
    params: list[Any] = [_FALLBACK_PREFIX]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def backfill_503(
    conn: Any,
    crawler: SentimentCrawler,
    *,
    limit: int | None = None,
    commit: bool = False,
) -> dict[str, int]:
    """Re-score every fallback row; UPDATE the recovered ones.

    Returns counts: ``scanned`` (rows examined), ``recovered`` (re-scored cleanly
    and UPDATEd when ``commit``), ``still_failing`` (Gemini still returned a
    fallback → left untouched for the next run). Pure enough to unit-test with a
    mock ``conn`` + mock ``crawler``.
    """
    rows = _fetch_corrupted(conn, limit)
    recovered = still_failing = 0

    for url, title, ticker, dt in rows:
        payload = crawler.score_payload(
            ticker=ticker, dt=dt, title=title or "", content=title or ""
        )
        # A reason still prefixed "Gemini fallback:" means the retry budget was
        # exhausted again — leave the row pending rather than overwrite a good
        # historical fallback with a fresh failed one.
        if str(payload["reason"]).startswith("Gemini fallback"):
            still_failing += 1
            LOGGER.info("[backfill] still failing, skipped: %s", url)
            continue

        recovered += 1
        sentiment = payload["sentiment_score"]
        magnitude = payload["magnitude"]
        if commit:
            conn.execute(
                f"UPDATE {_TABLE} SET sentiment_score = ?, magnitude = ?, "
                "reason = ?, sentiment_nlp = ?, impact_force = ? "
                "WHERE url = ? AND reason LIKE ?",
                [
                    sentiment, magnitude, payload["reason"],
                    sentiment, sentiment * magnitude, url, _FALLBACK_PREFIX,
                ],
            )
        LOGGER.info(
            "[backfill] %s %s -> score=%.2f mag=%.2f",
            "UPDATED" if commit else "WOULD UPDATE", url, sentiment, magnitude,
        )
    time.sleep(10)

    return {"scanned": len(rows), "recovered": recovered, "still_failing": still_failing}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(CONFIG.paths.duckdb_path), help="DuckDB path.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    parser.add_argument("--commit", action="store_true", help="Write UPDATEs (else dry-run).")
    args = parser.parse_args()

    crawler = SentimentCrawler(db_path=args.db)
    if crawler._client is None:
        raise SystemExit(
            "GEMINI_API_KEY not set / google-genai missing — cannot re-score. Aborting."
        )

    with duckdb.connect(args.db) as conn:
        stats = backfill_503(conn, crawler, limit=args.limit, commit=args.commit)

    mode = "COMMITTED" if args.commit else "DRY-RUN (no writes — pass --commit)"
    LOGGER.info(
        "[backfill] %s | scanned=%d recovered=%d still_failing=%d",
        mode, stats["scanned"], stats["recovered"], stats["still_failing"],
    )


if __name__ == "__main__":
    main()
