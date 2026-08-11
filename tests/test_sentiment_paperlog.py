"""Sentiment-entry forward paper-log â€” capture + backfill unit tests.

Covers the two new pure-ish helpers in main.py:
    * `_log_sentiment_entry_paperlog` â€” writes the candidate cross-section.
    * `_backfill_paperlog_outcomes`   â€” fills realized T+3 / T+20 returns.

Both operate on a `db` object exposing `.conn` (DuckDB connection) and
`._audit_lock` (threading.Lock). The tests build a lightweight in-memory stand-
in for that object â€” a fresh `duckdb.connect()` with the paper-log DDL applied â€”
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
        id                   INTEGER DEFAULT nextval('seq_sentiment_entry_id'),
        log_date             DATE    NOT NULL,
        ticker               VARCHAR NOT NULL,
        source               VARCHAR NOT NULL,
        p_down_primary       DOUBLE,
        p_side_primary       DOUBLE,
        p_up_primary         DOUBLE,
        decision_primary     INTEGER,
        p_down_secondary     DOUBLE,
        p_side_secondary     DOUBLE,
        p_up_secondary       DOUBLE,
        primary_horizon_days INTEGER,
        liquid_at_log        BOOLEAN,
        final_decision       INTEGER,
        sentiment_score      DOUBLE,
        entry_close          DOUBLE,
        ret_3d               DOUBLE,
        ret_20d              DOUBLE,
        outcome_filled       BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (id),
        UNIQUE (log_date, ticker, source)
    )
"""
# NOTE: this DDL duplicates `db_engine._init_sentiment_paperlog_table` and must
# be kept in sync by hand — a column added there and not here fails only as an
# opaque DuckDB BinderException, which is how the 10-08-26 rename first showed
# up. `test_stub_schema_matches_production` below pins the two together.


def test_stub_schema_matches_production():
    """The stub DDL above must list the same columns as production.

    Guards the duplication: without this, adding a column to
    `db_engine._init_sentiment_paperlog_table` leaves these tests passing
    against a stale schema, and the mismatch surfaces later as an opaque
    BinderException from the writer.
    """
    import re

    from src.data import db_engine as dbe

    prod_sql = dbe.DuckDBEngine._init_sentiment_paperlog_table.__doc__ or ""
    del prod_sql  # the DDL lives in the body, not the docstring — parse the source

    import inspect
    body = inspect.getsource(dbe.DuckDBEngine._init_sentiment_paperlog_table)
    prod_block = body.split("CREATE TABLE IF NOT EXISTS sentiment_entry_paperlog", 1)[1]
    prod_block = prod_block.split("PRIMARY KEY", 1)[0]
    stub_block = _PAPERLOG_DDL_TABLE.split(
        "CREATE TABLE IF NOT EXISTS sentiment_entry_paperlog", 1)[1]
    stub_block = stub_block.split("PRIMARY KEY", 1)[0]

    def names(block: str) -> set[str]:
        out = set()
        for line in block.splitlines():
            m = re.match(r"\s*([a-z_]+)\s+(INTEGER|DOUBLE|VARCHAR|DATE|BOOLEAN)",
                         line)
            if m:
                out.add(m.group(1))
        return out

    prod, stub = names(prod_block), names(stub_block)
    assert prod, "failed to parse the production DDL — fix this test's parser"
    assert prod == stub, (
        f"paperlog schema drift — only in production: {sorted(prod - stub)}; "
        f"only in the test stub: {sorted(stub - prod)}")


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
# Test Group A â€” capture helper (_log_sentiment_entry_paperlog)
# --------------------------------------------------------------------------- #

_STACK_5D = {
    "HPG": [0.6, 0.3, 0.1],   # argmax â†’ 0 (DOWN)
    "FPT": [0.1, 0.2, 0.7],   # argmax â†’ 2 (UP)
    "VCB": [0.2, 0.6, 0.2],   # argmax â†’ 1 (SIDE)
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
        "SELECT decision_primary, sentiment_score, final_decision, source "
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
    # UNIQUE(log_date, ticker, source) + INSERT OR IGNORE â†’ no duplicates.
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
            "WHERE p_down_secondary IS NULL AND p_side_secondary IS NULL AND p_up_secondary IS NULL"
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
    # MWG has no 5d prediction â†’ must be skipped, not written.
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
# Test Group B â€” backfill helper (_backfill_paperlog_outcomes)
# --------------------------------------------------------------------------- #


def _insert_raw_row(db, ticker: str, log_date: date) -> None:
    """Insert a minimal unfilled paper-log row for backfill tests."""
    with db._audit_lock:
        db.conn.execute(
            """
            INSERT OR IGNORE INTO sentiment_entry_paperlog
            (log_date, ticker, source, p_down_primary, p_side_primary, p_up_primary, decision_primary)
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


def test_backfill_short_horizon_fills_progressively(fake_db, monkeypatch) -> None:
    # T+3 mature but T+20 NOT (6 days): ret_3d fills now, ret_20d stays NULL,
    # and the row stays pending until the long window matures.
    log_date = date.today() - timedelta(days=6)
    _insert_raw_row(fake_db, "FPT", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 100.0
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 110.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 1
    entry_close, ret_3d, ret_20d, filled = fake_db.conn.execute(
        "SELECT entry_close, ret_3d, ret_20d, outcome_filled "
        "FROM sentiment_entry_paperlog WHERE ticker = 'FPT'"
    ).fetchone()
    assert entry_close == pytest.approx(100.0)
    assert ret_3d == pytest.approx((110.0 - 100.0) / 100.0)
    assert ret_20d is None          # long horizon not yet mature
    assert filled is False          # stays pending until T+20


def test_backfill_skips_ultra_fresh_row(fake_db, monkeypatch) -> None:
    # Younger than the short-maturity gate â†’ not even scanned.
    log_date = date.today() - timedelta(days=2)
    _insert_raw_row(fake_db, "VCB", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 100.0
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 110.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 0
    ret_3d, filled = fake_db.conn.execute(
        "SELECT ret_3d, outcome_filled FROM sentiment_entry_paperlog WHERE ticker = 'VCB'"
    ).fetchone()
    assert ret_3d is None
    assert filled is False


def test_backfill_long_horizon_completes_partial_row(fake_db, monkeypatch) -> None:
    # A row already carrying ret_3d (filled on an earlier run) gets ret_20d and
    # flips terminal once the max horizon matures. entry_close is REUSED â€” the
    # T0 lookup must not run, proven by a poisoned close_on_or_before.
    log_date = date.today() - timedelta(days=25)
    with fake_db._audit_lock:
        fake_db.conn.execute(
            """
            INSERT INTO sentiment_entry_paperlog
            (log_date, ticker, source, p_down_primary, p_side_primary, p_up_primary, decision_primary,
             entry_close, ret_3d, outcome_filled)
            VALUES (?, 'HPG', 'daily', 0.6, 0.3, 0.1, 0, 100.0, 0.05, FALSE)
            """,
            [log_date.strftime("%Y-%m-%d")],
        )

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 999.0  # poisoned
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 120.0
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 1
    entry_close, ret_3d, ret_20d, filled = fake_db.conn.execute(
        "SELECT entry_close, ret_3d, ret_20d, outcome_filled "
        "FROM sentiment_entry_paperlog WHERE ticker = 'HPG'"
    ).fetchone()
    assert entry_close == pytest.approx(100.0)            # cached entry preserved
    assert ret_3d == pytest.approx(0.05)                  # short return untouched
    assert ret_20d == pytest.approx((120.0 - 100.0) / 100.0)
    assert filled is True                                 # terminal now


def test_backfill_handles_missing_parquet(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=25)
    _insert_raw_row(fake_db, "DELISTED", log_date)

    # T0 shard absent â†’ close_on_or_before returns None â†’ row left untouched.
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


# --------------------------------------------------------------------------- #
# Test Group C â€” corporate-action guard (2026-07-12, KLB Jul-2 stock dividend)
# --------------------------------------------------------------------------- #

from src.data.price_lookup import has_ca_gap  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ca_gap_by_default(monkeypatch):
    """Neutralize the CA guard for pre-existing tests: fake tickers would
    otherwise hit REAL parquet shards (HPG/FPT exist on disk) and make
    results date-dependent. CA tests override this per-test."""
    monkeypatch.setattr(main.price_lookup, "closes_between", lambda *a, **k: [])


def test_has_ca_gap_normal_path_false() -> None:
    assert has_ca_gap([100.0, 106.9, 100.1, 93.5]) is False  # inside Â±7% band


def test_has_ca_gap_klb_shape_true() -> None:
    # KLB 01-07 -> 02-07: 16.55 -> 12.78 = -22.8% single session.
    assert has_ca_gap([16.5, 16.55, 12.78, 12.85]) is True


def test_has_ca_gap_empty_and_single() -> None:
    assert has_ca_gap([]) is False
    assert has_ca_gap([100.0]) is False


def test_backfill_ca_gap_skips_short_fill(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=6)  # T+3 mature, T+20 not
    _insert_raw_row(fake_db, "KLB", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 16.6
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 12.85
    )
    monkeypatch.setattr(
        main.price_lookup, "closes_between",
        lambda *a, **k: [16.6, 16.55, 12.78, 12.85],  # CA gap inside window
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 0  # nothing written â€” short skipped, long not mature
    ret_3d, filled = fake_db.conn.execute(
        "SELECT ret_3d, outcome_filled FROM sentiment_entry_paperlog "
        "WHERE ticker = 'KLB'"
    ).fetchone()
    assert ret_3d is None
    assert filled is False  # stays pending until the long window settles it


def test_backfill_ca_gap_settles_long_with_null(fake_db, monkeypatch) -> None:
    log_date = date.today() - timedelta(days=25)  # both horizons mature
    _insert_raw_row(fake_db, "KLB", log_date)

    monkeypatch.setattr(
        main.price_lookup, "close_on_or_before", lambda t, d, conn=None: 16.6
    )
    monkeypatch.setattr(
        main.price_lookup, "close_on_or_after", lambda t, d, conn=None: 12.85
    )
    monkeypatch.setattr(
        main.price_lookup, "closes_between",
        lambda *a, **k: [16.6, 16.55, 12.78, 12.85],
    )

    n = main._backfill_paperlog_outcomes(fake_db)
    assert n == 1  # progress: the row settled (excluded), not retried forever
    ret_3d, ret_20d, filled = fake_db.conn.execute(
        "SELECT ret_3d, ret_20d, outcome_filled FROM sentiment_entry_paperlog "
        "WHERE ticker = 'KLB'"
    ).fetchone()
    assert ret_3d is None      # short window also CA-contaminated -> skipped
    assert ret_20d is None     # settled with NULL = permanently excluded
    assert filled is True
