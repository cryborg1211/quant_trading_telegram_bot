"""PK-dedup guard for SentimentCrawler._append_rows (2026-07-19).

Incident: the live hist_sentiment_llm_labeled table has PRIMARY KEY
(ticker, date, title) and the crawl window overlaps yesterday, so re-crawled
articles (e.g. Vietstock warrant notices with identical titles) violated the
PK and crashed the whole EOD full_pipeline on 16-07 and 17-07 — no signals,
no EOD position report. _append_rows must dedup within the batch and
anti-join against rows already present.

Uses a real on-disk DuckDB (tmp_path) because the guard under test IS the
insert SQL.
"""
from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.crawlers.sentiment_crawler import SentimentCrawler


PK_TABLE_SQL = """
CREATE TABLE hist_sentiment_llm_labeled (
    date TIMESTAMP,
    ticker VARCHAR,
    title VARCHAR,
    sentiment_score DOUBLE,
    magnitude DOUBLE,
    reason VARCHAR,
    url VARCHAR,
    sentiment_nlp DOUBLE,
    impact_force DOUBLE,
    is_market_wide BOOLEAN,
    PRIMARY KEY (ticker, date, title)
)
"""


def _row(ticker: str = "MBB", title: str = "Chung quyen MBB", ts: str = "2026-07-15") -> dict:
    return {
        "date": pd.Timestamp(ts),
        "ticker": ticker,
        "title": title,
        "sentiment_score": 0.5,
        "magnitude": 0.3,
        "reason": "test",
        "url": f"https://news.example/{ticker}/{title}",
        "sentiment_nlp": 0.1,
        "impact_force": 0.2,
        "is_market_wide": False,
    }


@pytest.fixture()
def crawler(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    db = str(tmp_path / "sent.duckdb")
    with duckdb.connect(db) as conn:
        conn.execute(PK_TABLE_SQL)
    return SentimentCrawler(db_path=db)


def _count(db_path: str) -> int:
    with duckdb.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM hist_sentiment_llm_labeled").fetchone()[0]


def test_recrawled_rows_do_not_crash_pk(crawler):
    # First run inserts; second run re-crawls the same article — must be a
    # silent no-op, not a ConstraintException (the 16/17-07 crash).
    crawler._append_rows(pd.DataFrame([_row()]))
    crawler._append_rows(pd.DataFrame([_row()]))
    assert _count(crawler.db_path) == 1


def test_intra_batch_duplicates_collapse(crawler):
    df = pd.DataFrame([_row(), _row(), _row(ticker="ACB")])
    crawler._append_rows(df)
    assert _count(crawler.db_path) == 2


def test_existing_row_wins_new_score_discarded(crawler):
    first = _row()
    crawler._append_rows(pd.DataFrame([first]))
    changed = {**_row(), "sentiment_score": -0.9}
    crawler._append_rows(pd.DataFrame([changed]))
    with duckdb.connect(crawler.db_path) as conn:
        score = conn.execute(
            "SELECT sentiment_score FROM hist_sentiment_llm_labeled"
        ).fetchone()[0]
    assert score == pytest.approx(0.5)


def test_distinct_rows_still_insert(crawler):
    crawler._append_rows(pd.DataFrame([_row()]))
    crawler._append_rows(
        pd.DataFrame([_row(title="Other title"), _row(ticker="HPG")])
    )
    assert _count(crawler.db_path) == 3
