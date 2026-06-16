"""Sentiment-entry forward paper-log — capture + backfill unit tests.

Covers the two new pure-ish helpers in main.py:
    * `_log_sentiment_entry_paperlog` — writes the candidate cross-section.
    * `_backfill_paperlog_outcomes`   — fills realized T+3 / T+20 returns.

Both operate on a `db` object exposing `.conn` (DuckDB connection) and
`._audit_lock` (threading.Lock). The tests build a lightweight in-memory stand-
in for that object — a fresh `duckdb.connect()` with the paper-log DDL applied —
so no real DuckDBEngine singleton, no parquet shards, and no external services
are touched. Price lookups are monkeypatched to deterministic floats.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import duckdb
import pytest

import main


# The paper-log DDL mirrors DuckDBEngine._init_sentiment_paperlog_table exactly.
# Kept local so the test does not depend on a real engine init (which would
# create every other table + touch the data/ dir).
_PAPERLOG_DDL_SEQ = "CREATE SEQUENCE IF NOT EXISTS seq_sentiment_entry_id START 1"
_PAPERLOG_DDL_TABLE = """
    CREATE TABLE IF NOT EXISTS sentiment_entry_paperlog (
        id              INTEGER DEFAULT nextval('seq_sentiment_entry_id'),
        log_date        DATE    NOT NULL,
        ticker          VARCHAR NOT NULL,
        source          VARCHAR NOT NULL,
        p_down_5d       DOUBLE,
        p_side_5d       DOUBLE,
        p_up_5d         DOUBLE,
        decision_5d     INTEGER,
        p_down_20d      DOUBLE,
        p_side_20d      DOUBLE,
        p_up_20d        DOUBLE,
        final_decision  INTEGER,
        sentiment_score DOUBLE,
        entry_close     DOUBLE,
        ret_3d          DOUBLE,
        ret_20d         DOUBLE,
        outcome_filled  BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (id),
        UNIQUE (log_date, ticker, source)
    )
"""


@pytest.fixture()
def fake_db():
    """In-memory stand-in for the DuckDBEngine singleton.

    Exposes `.conn` + `._audit_lock`, the only two attributes the paper-log
    helpers touch. The connection is in-memory so it vanishes after the test.
    """
    conn = duckdb.connect()  # in-memory
    conn.execute(_PAPERLOG_DDL_SEQ)
    conn.execute(_PAPERLOG_DDL_TABLE)
    db = SimpleNamespace(conn=conn, _audit_lock=threading.Lock())
    yield db
    conn.close()


# --------------------------------------------------------------------------- #
# Test Group A — capture helper (_log_sentiment_entry_paperlog)
# --------------------------------------------------------------------------- #

_STACK_5D = {
    "HPG": [0.6, 0.3, 0.1],   # argmax → 0 (DOWN)
    "FPT": [0.1, 0.2, 0.7],   # argmax → 2 (UP)
    "VCB": [0.2, 0.6, 0.2],   # argmax → 1 (SIDE)
}
_STACK_20D = {
    "HPG": [0.5, 0.3, 0.2],
    "FPT": [0.2, 0.3, 0.5],
    "VCB": [0.3, 0.4, 0.3],
}
_FINAL = {"HPG": 0, "FPT": 2, "VCB": 1}
_SENT = {
    "HPG": {"sentiment_score": 0.85},
    "FPT": {"sentiment_score": 0.40},
    "VCB": {"sentiment_score": 0.10},
}


def _count(db) -> int:
    return int(db.conn.execute("SELECT COUNT(*) FROM sentiment_entry_paperlog").fetchone()[0])


def test_log_writes_full_crosssection(fake_db) -> None:
    n = main._log_sentiment_entry_paperlog(
        db=fake_db,
        candidate_tickers=["HPG", "FPT", "VCB"],
        stacking_5d=_STACK_5D,
        stacking_20d=_STACK_20D,
        final_decisions=_FINAL,
        all_sentiments=_SENT,
        source="daily",
    )
    assert n == 3
    assert _count(fake_db) == 3
    # Spot-check the DOWN argmax + sentiment landed correctly for the treatment name.
    row = fake_db.conn.execute(
        "SELECT decision_5d, sentiment_score, final_decision, source "
        "FROM sentiment_entry_paperlog WHERE ticker = 'HPG'"
    ).fetchone()
    assert row == (0, 0.85, 0, "daily")


def test_log_idempotent_same_day(fake_db) -> None:
    args = dict(
        candidate_tickers=["HPG", "FPT", "VCB"],
        stacking_5d=_STACK_5D,
        stacking_20d=_STACK_20D,
        final_decisions=_FINAL,
        all_sentiments=_SENT,
        source="daily",
    )
    main._log_sentiment_entry_paperlog(db=fake_db, **args)
    main._log_sentiment_entry_paperlog(db=fake_db, **args)  # same day, same tickers
    # UNIQUE(log_date, ticker, source) + INSERT OR IGNORE → no duplicates.
    assert _count(fake_db) == 3


def test_log_20d_none_stores_null(fake_db) -> None:
    main._log_sentiment_entry_paperlog(
        db=fake_db,
        candidate_tickers=["HPG", "FPT", "VCB"],
        stacking_5d=_STACK_5D,
        stacking_20d={},  # secondary horizon artifact missing
        final_decisions=_FINAL,
        all_sentiments=_SENT,
        source="daily",
    )
    nulls = int(
        fake_db.conn.execute(
            "SELECT COUNT(*) FROM sentiment_entry_paperlog "
            "WHERE p_down_20d IS NULL AND p_side_20d IS NULL AND p_up_20d IS NULL"
        ).fetchone()[0]
    )
    assert nulls == 3


def test_log_sentiment_score_none(fake_db) -> None:
    main._log_sentiment_entry_paperlog(
        db=fake_db,
        candidate_tickers=["HPG", "FPT", "VCB"],
        stacking_5d=_STACK_5D,
        stacking_20d=_STACK_20D,
        final_decisions=_FINAL,
        all_sentiments={},  # arbitrator unavailable
        source="daily",
    )
    nulls = int(
        fake_db.conn.execute(
            "SELECT COUNT(*) FROM sentiment_entry_paperlog WHERE sentiment_score IS NULL"
        ).fetchone()[0]
    )
    assert nulls == 3


def test_log_source_tagged_verify(fake_db) -> None:
    main._log_sentiment_entry_paperlog(
        db=fake_db,
        candidate_tickers=["HPG"],
        stacking_5d=_STACK_5D,
        stacking_20d=_STACK_20D,
        final_decisions=_FINAL,
        all_sentiments=_SENT,
        source="verify",
    )
    src = fake_db.conn.execute(
        "SELECT source FROM sentiment_entry_paperlog WHERE ticker = 'HPG'"
    ).fetchone()[0]
    assert src == "verify"


def test_log_skips_ticker_with_no_5d(fake_db) -> None:
    # MWG has no 5d prediction → must be skipped, not written.
    n = main._log_sentiment_entry_paperlog(
        db=fake_db,
        candidate_tickers=["HPG", "MWG"],
        stacking_5d=_STACK_5D,  # MWG absent
        stacking_20d=_STACK_20D,
        final_decisions=_FINAL,
        all_sentiments=_SENT,
        source="daily",
    )
    assert n == 1
    tickers = [
        r[0]
        for r in fake_db.conn.execute(
            "SELECT ticker FROM sentiment_entry_paperlog"
        ).fetchall()
    ]
    assert tickers == ["HPG"]


# --------------------------------------------------------------------------- #
# Test Group B — backfill helper (_backfill_paperlog_outcomes)
# --------------------------------------------------------------------------- #


def _insert_raw_row(db, ticker: str, log_date: date) -> None:
    """Insert a minimal unfilled paper-log row for backfill tests."""
    with db._audit_lock:
        db.conn.execute(
            """
            INSERT OR IGNORE INTO sentiment_entry_paperlog
            (log_date, ticker, source, p_down_5d, p_side_5d, p_up_5d, decision_5d)
            VALUES (?, ?, 'daily', 0.6, 0.3, 0.1, 0)
            """,
            [log_date.strftime("%Y-%m-%d"), ticker],
        )


def test_backfill_computes_ret_3d_and_ret_20d(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=25)  # matured (> 21 days)
    _insert_raw_row(fake_db, "HPG", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 100.0
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 110.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 1
    row = fake_db.conn.execute(
        "SELECT entry_close, ret_3d, ret_20d, outcome_filled "
        "FROM sentiment_entry_paperlog WHERE ticker = 'HPG'"
    ).fetchone()
    entry_close, ret_3d, ret_20d, filled = row
    assert entry_close == pytest.approx(100.0)
    assert ret_3d == pytest.approx((110.0 - 100.0) / 100.0)
    assert ret_20d == pytest.approx((110.0 - 100.0) / 100.0)
    assert filled is True


def test_backfill_skips_immature_rows(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=5)  # NOT matured (< 21 days)
    _insert_raw_row(fake_db, "FPT", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 100.0
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 110.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 0
    filled = fake_db.conn.execute(
        "SELECT outcome_filled FROM sentiment_entry_paperlog WHERE ticker = 'FPT'"
    ).fetchone()[0]
    assert filled is False


def test_backfill_handles_missing_parquet(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=25)
    _insert_raw_row(fake_db, "DELISTED", log_date)

    # T0 shard absent → close_on_or_before returns None → row left untouched.
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: None
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 110.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)  # must not crash
    assert n == 0
    filled = fake_db.conn.execute(
        "SELECT outcome_filled FROM sentiment_entry_paperlog WHERE ticker = 'DELISTED'"
    ).fetchone()[0]
    assert filled is False


def test_backfill_returns_count(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=25)
    _insert_raw_row(fake_db, "HPG", log_date)
    _insert_raw_row(fake_db, "FPT", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 50.0
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 55.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 2
    filled = int(
        fake_db.conn.execute(
            "SELECT COUNT(*) FROM sentiment_entry_paperlog WHERE outcome_filled = TRUE"
        ).fetchone()[0]
    )
    assert filled == 2
