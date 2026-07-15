"""Cost-guard tests for the sentiment crawler (2026-07-14).

Two guards under test:
1. Model pin precedence — explicit arg → .env GEMINI_MODEL → config default
   (the floating "gemini-flash-latest" alias is banned; it silently became
   gemini-3.5-flash and caused the 2026-07-07 JSON-drift incident).
2. Chunked incremental append — scored rows are persisted every
   SCORE_APPEND_CHUNK articles so an interrupt keeps completed chunks
   (2026-07-13 incident: Ctrl+C lost ~7 minutes of paid Gemini calls).

No DB, no network: DB-facing and fetch methods are monkeypatched per test.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from config.settings import CONFIG
from src.crawlers.sentiment_crawler import NewsItem, SentimentCrawler


# ---------------------------------------------------------------------------
# Model pin precedence
# ---------------------------------------------------------------------------


def test_model_explicit_arg_beats_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "env-pinned-model")
    c = SentimentCrawler(model_name="models/explicit-model")
    assert c.model_name == "explicit-model"


def test_model_env_pin_beats_config_default(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-pin")
    c = SentimentCrawler()
    assert c.model_name == "gemini-env-pin"


def test_model_config_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    c = SentimentCrawler()
    assert c.model_name == CONFIG.sentiment.gemini_model.removeprefix("models/")


def test_config_default_is_ga_pinned_not_floating():
    # Guard against the floating alias sneaking back into the default.
    assert "latest" not in CONFIG.sentiment.gemini_model


# ---------------------------------------------------------------------------
# Chunked incremental append
# ---------------------------------------------------------------------------


def _make_items(n: int) -> list[NewsItem]:
    today = datetime.now().date()
    return [
        NewsItem(date=today, title=f"title {i}", url=f"https://news.example/{i}",
                 text="body", ticker="AAA")
        for i in range(n)
    ]


def _stubbed_crawler(monkeypatch, items: list[NewsItem]) -> SentimentCrawler:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    c = SentimentCrawler()
    monkeypatch.setattr(c, "_active_tickers", lambda limit: ["AAA"])
    monkeypatch.setattr(c, "_existing_urls", lambda: set())
    monkeypatch.setattr(c, "_fetch_rss_items", lambda **kw: [])
    monkeypatch.setattr(c, "_fetch_gnews_items", lambda **kw: items)
    return c


def test_chunked_append_flushes_every_chunk(monkeypatch):
    c = _stubbed_crawler(monkeypatch, _make_items(60))
    monkeypatch.setattr(
        c,
        "_score_batch",
        lambda items: [{"ticker": it.ticker, "url": it.url} for it in items],
    )
    append_sizes: list[int] = []
    monkeypatch.setattr(c, "_append_rows", lambda df: append_sizes.append(len(df)))

    out = c.update_daily_sentiment()

    assert append_sizes == [25, 25, 10]
    assert len(out) == 60


def test_interrupt_persists_completed_chunks(monkeypatch):
    """Ctrl+C mid-scoring must keep every fully-scored chunk (paid calls)."""
    c = _stubbed_crawler(monkeypatch, _make_items(60))
    calls = {"n": 0}

    def batch_scorer(items: list[NewsItem]) -> list[dict]:
        calls["n"] += 1
        if calls["n"] == 2:  # dies on the SECOND batch
            raise KeyboardInterrupt
        return [{"ticker": it.ticker, "url": it.url} for it in items]

    monkeypatch.setattr(c, "_score_batch", batch_scorer)
    append_sizes: list[int] = []
    monkeypatch.setattr(c, "_append_rows", lambda df: append_sizes.append(len(df)))

    with pytest.raises(KeyboardInterrupt):
        c.update_daily_sentiment()

    assert append_sizes == [25]  # first chunk persisted, second lost, third never ran


# ---------------------------------------------------------------------------
# Batched scoring (one Gemini call per chunk, 14-07 cost surgery)
# ---------------------------------------------------------------------------


def _bare_crawler(monkeypatch) -> SentimentCrawler:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return SentimentCrawler()


def test_parse_batch_json_happy_array():
    raw = '[{"id": 0, "sentiment_score": 0.5, "magnitude": 0.7, "reason": "ok"}, {"id": 1, "sentiment_score": -0.2}]'
    parsed = SentimentCrawler._parse_batch_json(raw)
    assert parsed[0]["sentiment_score"] == 0.5
    assert parsed[1]["sentiment_score"] == -0.2


def test_parse_batch_json_fenced_and_wrapped():
    raw = '```json\n{"results": [{"id": 0, "sentiment_score": 0.1}]}\n```'
    parsed = SentimentCrawler._parse_batch_json(raw)
    assert parsed[0]["sentiment_score"] == 0.1


def test_parse_batch_json_missing_id_falls_back_to_position():
    raw = '[{"sentiment_score": 0.3}, {"sentiment_score": 0.4}]'
    parsed = SentimentCrawler._parse_batch_json(raw)
    assert parsed[0]["sentiment_score"] == 0.3
    assert parsed[1]["sentiment_score"] == 0.4


def test_score_batch_total_failure_yields_backfillable_fallbacks(monkeypatch):
    """API death degrades to neutral rows carrying the 503-backfill marker."""
    c = _bare_crawler(monkeypatch)
    c._client = object()  # force the non-disabled path
    monkeypatch.setattr(
        c, "_generate_content", lambda msg: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    rows = c._score_batch(_make_items(3))
    assert len(rows) == 3
    assert all(r["sentiment_score"] == 0.0 for r in rows)
    assert all(r["reason"].startswith("Gemini fallback:") for r in rows)


def test_score_batch_missing_item_gets_individual_fallback(monkeypatch):
    c = _bare_crawler(monkeypatch)
    c._client = object()
    monkeypatch.setattr(
        c,
        "_generate_content",
        lambda msg: '[{"id": 0, "sentiment_score": 0.9, "magnitude": 0.5, "reason": "good"}]',
    )
    rows = c._score_batch(_make_items(2))
    assert rows[0]["sentiment_score"] == 0.9
    assert rows[1]["sentiment_score"] == 0.0
    assert rows[1]["reason"].startswith("Gemini fallback:")


def test_score_batch_clamps_out_of_range_scores(monkeypatch):
    c = _bare_crawler(monkeypatch)
    c._client = object()
    monkeypatch.setattr(
        c,
        "_generate_content",
        lambda msg: '[{"id": 0, "sentiment_score": 7.5, "magnitude": -3.0, "reason": "x"}]',
    )
    rows = c._score_batch(_make_items(1))
    assert rows[0]["sentiment_score"] == 1.0
    assert rows[0]["magnitude"] == 0.0


def test_empty_day_still_returns_empty_frame(monkeypatch):
    c = _stubbed_crawler(monkeypatch, [])
    appended: list[int] = []
    monkeypatch.setattr(c, "_append_rows", lambda df: appended.append(len(df)))

    out = c.update_daily_sentiment()

    assert isinstance(out, pd.DataFrame) and out.empty
    assert appended == []
