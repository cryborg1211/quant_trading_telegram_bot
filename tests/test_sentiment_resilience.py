"""Tests for Gemini 503 resilience + the fallback backfill.

No real Gemini, no real DuckDB. A fake client drives the tenacity retry paths;
a mock connection + mock crawler drive the backfill. Tenacity's real backoff
sleep is disabled so retries run instantly.
"""
from __future__ import annotations

import types
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.crawlers import sentiment_crawler as sc
from src.crawlers.sentiment_crawler import SentimentCrawler, _is_transient_gemini_error


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _ServerError(Exception):
    """Mimics google-genai ServerError: numeric code + message."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeClient:
    """generate_content fails `fail_times`, then returns `ok_text`."""

    def __init__(self, fail_times: int, exc: Exception, ok_text: str = "") -> None:
        self.calls = 0
        self._fail_times = fail_times
        self._exc = exc
        self._ok_text = ok_text
        self.models = self  # client.models.generate_content(...)

    def generate_content(self, **_kw):  # noqa: ANN003
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return _Resp(self._ok_text)


_OK_JSON = '{"sentiment_score": 0.6, "magnitude": 0.4, "reason": "tÃ­ch cá»±c"}'


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Disable tenacity's real sleep so retry tests run instantly."""
    original = SentimentCrawler._generate_content.retry.sleep
    SentimentCrawler._generate_content.retry.sleep = lambda *a, **k: None
    yield
    SentimentCrawler._generate_content.retry.sleep = original


@pytest.fixture(autouse=True)
def _fake_genai_types(monkeypatch):
    """Provide a stub genai_types so _generate_content can build its config."""
    stub = types.SimpleNamespace(GenerateContentConfig=lambda **k: None)
    monkeypatch.setattr(sc, "genai_types", stub)


def _crawler_with(client) -> SentimentCrawler:
    c = SentimentCrawler.__new__(SentimentCrawler)  # skip __init__ (no API key needed)
    c._client = client
    c.model_name = "gemini-flash-latest"
    return c


# --------------------------------------------------------------------------- #
# _is_transient_gemini_error
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "exc, expected",
    [
        (_ServerError("503 UNAVAILABLE", code=503), True),
        (_ServerError("model overloaded", code=503), True),
        (Exception("The service is currently UNAVAILABLE"), True),
        (Exception("502 Bad Gateway"), True),
        (ValueError("invalid json"), False),
        (_ServerError("400 INVALID_ARGUMENT", code=400), False),
    ],
)
def test_is_transient_classifier(exc, expected) -> None:
    assert _is_transient_gemini_error(exc) is expected


# --------------------------------------------------------------------------- #
# _generate_content retry behaviour
# --------------------------------------------------------------------------- #

def test_generate_content_retries_then_succeeds() -> None:
    client = _FakeClient(fail_times=2, exc=_ServerError("503 UNAVAILABLE", 503), ok_text=_OK_JSON)
    crawler = _crawler_with(client)
    out = crawler._generate_content("msg")
    assert out == _OK_JSON
    assert client.calls == 3  # 2 transient failures + 1 success


def test_generate_content_gives_up_after_five_attempts() -> None:
    client = _FakeClient(fail_times=99, exc=_ServerError("503 UNAVAILABLE", 503))
    crawler = _crawler_with(client)
    with pytest.raises(_ServerError):
        crawler._generate_content("msg")
    assert client.calls == 5  # stop_after_attempt(5)


def test_generate_content_does_not_retry_permanent_error() -> None:
    client = _FakeClient(fail_times=99, exc=_ServerError("400 INVALID_ARGUMENT", 400))
    crawler = _crawler_with(client)
    with pytest.raises(_ServerError):
        crawler._generate_content("msg")
    assert client.calls == 1  # 4xx is not retried


# --------------------------------------------------------------------------- #
# score_payload
# --------------------------------------------------------------------------- #

def test_score_payload_success_clamps() -> None:
    client = _FakeClient(0, _ServerError("x"), ok_text='{"sentiment_score": 1.5, "magnitude": 2.0, "reason": "r"}')
    crawler = _crawler_with(client)
    out = crawler.score_payload(ticker="HPG", dt=date(2026, 6, 1), title="t", content="c")
    assert out["sentiment_score"] == 1.0   # clamped to [-1, 1]
    assert out["magnitude"] == 1.0         # clamped to [0, 1]
    assert out["reason"] == "r"


def test_score_payload_persistent_503_falls_back() -> None:
    client = _FakeClient(fail_times=99, exc=_ServerError("503 UNAVAILABLE", 503))
    crawler = _crawler_with(client)
    out = crawler.score_payload(ticker="HPG", dt=date(2026, 6, 1), title="t", content="c")
    assert out["sentiment_score"] == 0.0
    assert out["reason"].startswith("Gemini fallback")
    assert client.calls == 5  # exhausted the retry budget before falling back


def test_score_payload_parse_error_not_retried() -> None:
    client = _FakeClient(0, _ServerError("x"), ok_text="NOT JSON")
    crawler = _crawler_with(client)
    out = crawler.score_payload(ticker="HPG", dt=date(2026, 6, 1), title="t", content="c")
    assert out["reason"].startswith("Gemini fallback")
    assert client.calls == 1  # bad JSON is permanent â†’ no retry


# --------------------------------------------------------------------------- #
# backfill_503
# --------------------------------------------------------------------------- #

def _mock_conn(rows: list[tuple]) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    return conn


def test_backfill_recovers_and_updates() -> None:
    from scripts.backfill_sentiment_503 import backfill_503

    rows = [
        ("http://a", "title a", "HPG", date(2026, 6, 1)),
        ("http://b", "title b", "SSI", date(2026, 6, 2)),
    ]
    conn = _mock_conn(rows)
    crawler = MagicMock()
    crawler.score_payload.return_value = {
        "sentiment_score": 0.5, "magnitude": 0.3, "reason": "ok",
    }
    stats = backfill_503(conn, crawler, commit=True)

    assert stats == {"scanned": 2, "recovered": 2, "still_failing": 0}
    updates = [c for c in conn.execute.call_args_list if "UPDATE" in c[0][0]]
    assert len(updates) == 2


def test_backfill_skips_still_failing_rows() -> None:
    from scripts.backfill_sentiment_503 import backfill_503

    conn = _mock_conn([("http://a", "title a", "HPG", date(2026, 6, 1))])
    crawler = MagicMock()
    crawler.score_payload.return_value = {
        "sentiment_score": 0.0, "magnitude": 0.0, "reason": "Gemini fallback: 503 again",
    }
    stats = backfill_503(conn, crawler, commit=True)

    assert stats == {"scanned": 1, "recovered": 0, "still_failing": 1}
    assert not [c for c in conn.execute.call_args_list if "UPDATE" in c[0][0]]


def test_backfill_dry_run_writes_nothing() -> None:
    from scripts.backfill_sentiment_503 import backfill_503

    conn = _mock_conn([("http://a", "title a", "HPG", date(2026, 6, 1))])
    crawler = MagicMock()
    crawler.score_payload.return_value = {
        "sentiment_score": 0.5, "magnitude": 0.3, "reason": "ok",
    }
    stats = backfill_503(conn, crawler, commit=False)

    assert stats["recovered"] == 1
    assert not [c for c in conn.execute.call_args_list if "UPDATE" in c[0][0]]


# --------------------------------------------------------------------------- #
# parse_gemini_json â€” 2026-07-07 format-drift incident coverage
# --------------------------------------------------------------------------- #

def test_parse_strict_json_still_works() -> None:
    out = SentimentCrawler.parse_gemini_json(
        '{"sentiment_score": 0.4, "magnitude": 0.2, "reason": "ok"}'
    )
    assert out["sentiment_score"] == 0.4


def test_parse_fenced_json() -> None:
    out = SentimentCrawler.parse_gemini_json(
        '```json\n{"sentiment_score": -0.3, "magnitude": 0.1, "reason": "x"}\n```'
    )
    assert out["sentiment_score"] == -0.3


def test_parse_trailing_extra_data() -> None:
    # "Extra data: line 9 column 1" shape: valid object then trailing junk.
    raw = '{"sentiment_score": 0.6, "magnitude": 0.5, "reason": "r"}\nNote: extra prose.'
    out = SentimentCrawler.parse_gemini_json(raw)
    assert out["sentiment_score"] == 0.6


def test_parse_two_objects_takes_first() -> None:
    raw = ('{"sentiment_score": 0.2, "magnitude": 0.1, "reason": "a"}\n'
           '{"sentiment_score": -0.9, "magnitude": 0.9, "reason": "b"}')
    out = SentimentCrawler.parse_gemini_json(raw)
    assert out["sentiment_score"] == 0.2


def test_parse_broken_reason_salvages_numeric_fields() -> None:
    # "Expecting ',' delimiter" shape: unescaped quote breaks the reason string.
    raw = ('{"sentiment_score": -0.5, "magnitude": 0.7, '
           '"reason": "cong ty "ABC" cong bo loi nhuan giam manh...')
    out = SentimentCrawler.parse_gemini_json(raw)
    assert out["sentiment_score"] == -0.5
    assert out["magnitude"] == 0.7
    assert out["reason"].startswith("salvaged:")


def test_parse_total_garbage_raises() -> None:
    with pytest.raises(ValueError):
        SentimentCrawler.parse_gemini_json("I cannot help with that request.")


def test_score_payload_falls_back_on_unparseable(monkeypatch) -> None:
    crawler = SentimentCrawler.__new__(SentimentCrawler)
    crawler._client = object()
    monkeypatch.setattr(sc, "genai_types", types.SimpleNamespace(), raising=False)
    crawler._generate_content = lambda msg: "no json here at all"
    out = crawler.score_payload(ticker="HPG", dt=date(2026, 7, 7), title="t", content="c")
    assert out["sentiment_score"] == 0.0
    assert out["reason"].startswith("Gemini fallback:")
