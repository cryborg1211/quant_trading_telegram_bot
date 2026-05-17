#!/usr/bin/env python3
"""Backfill missing Macro + LLM-Sentiment context into the Quant V6 DuckDB.

WHY THIS EXISTS
--------------─
The production DB (`data/quant_v6_core.duckdb`) was recovered OHLCV-only -
`macro_daily` and `hist_sentiment_llm_labeled` are absent, so the current
retrain runs with those features zero-filled. This script repopulates both
so the NEXT retrain is fully featured.

It deliberately REUSES existing infra (no divergent schemas):
  • Macro     -> src.data.crawlers.MacroCrawler.fetch_macro  (yfinance live)
  • DB upsert -> src.data.db_engine.DuckDBEngine.upsert_dataframe
                (INSERT OR REPLACE BY NAME, resolves on PRIMARY KEY)
  • Sentiment -> google-genai Client, gemini-3.5-flash, EXACT call pattern
                copied from src.models.quant_agent_arbitrator

SCHEMA NOTES (important - read before editing)
----------------------------------------------
1. `macro_daily`  PK(date), columns:
   date, dxy_close, sp500_close, usd_vnd, interbank_on_rate, vnibor,
   inflation_yoy.  `DuckDBEngine()` creates/migrates this table on init.

2. `hist_sentiment_llm_labeled`: the sentiment crawler's CREATE TABLE has
   NO `ticker` column, but `Alpha360Generator._query_sentiment` does
   `SELECT ticker, ... GROUP BY ticker, date`. This script is therefore the
   AUTHORITATIVE creator of that table and writes the correct SUPERSET
   schema *including* `ticker`, with PK(ticker, date, title) so re-runs are
   idempotent (INSERT OR REPLACE, no duplicate blow-up). Alpha360's
   GROUP BY ticker,date still aggregates multiple headlines/day correctly.

USAGE
----─
    # Live macro via yfinance + Gemini-scored dummy sentiment:
    python backfill_context_data.py

    # Macro from a CSV you provide (yfinance often DNS-blocked from VN ISPs):
    python backfill_context_data.py --macro-csv my_macro.csv

    # Custom window:
    python backfill_context_data.py --start 2014-01-01 --end 2026-05-16

Both halves are independent: if one fails the other still runs, and the
script exits non-zero only if BOTH fail.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
from dotenv import load_dotenv

# CRITICAL: GEMINI_API_KEY lives in .env, not the shell environment.
# SentimentCrawler reads it via os.getenv at construction time, so without
# this the Gemini client is never created and every headline silently
# falls back to a neutral 0.0 score (useless as a feature). Load it before
# anything imports/constructs the crawler.
load_dotenv()

from src.data.db_engine import DuckDBEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("backfill_context")

DB_PATH = "data/quant_v6_core.duckdb"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2026-05-16"  # yfinance `end` is exclusive -> includes 2026-05-15

# macro_daily column contract (mirrors db_engine._MACRO_DAILY_COLUMNS).
MACRO_COLUMNS = [
    "date",
    "dxy_close",
    "sp500_close",
    "usd_vnd",
    "interbank_on_rate",
    "vnibor",
    "inflation_yoy",
]

SENTIMENT_TABLE = "hist_sentiment_llm_labeled"


# ════════════════════════════════════════════════════════════════════════
# 1. MACRO BACKFILL
# ════════════════════════════════════════════════════════════════════════
def _read_macro_csv(csv_path: str) -> pd.DataFrame:
    """Load a user-supplied macro CSV.

    Expected header (extra columns ignored; missing optional columns are
    created as NULL so the wide schema stays intact):

        date,dxy_close,sp500_close,usd_vnd,interbank_on_rate,vnibor,inflation_yoy
        2026-05-15,105.4,5210.3,25380,4.85,5.10,3.20
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"{csv_path}: required 'date' column missing.")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in MACRO_COLUMNS:
        if col not in df.columns:
            LOGGER.warning("CSV missing optional macro column '%s' -> NULL.", col)
            df[col] = None
    return df[MACRO_COLUMNS].sort_values("date").reset_index(drop=True)


def backfill_macro(
    engine: DuckDBEngine,
    start: str,
    end: str,
    macro_csv: str | None,
) -> bool:
    """Fetch (or load) macro indicators and upsert into `macro_daily`.

    Returns True on a non-empty upsert, False otherwise.
    """
    LOGGER.info("-- MACRO backfill | window %s -> %s --", start, end)

    if macro_csv:
        LOGGER.info("Loading macro from CSV: %s", macro_csv)
        try:
            macro_df = _read_macro_csv(macro_csv)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Macro CSV load failed: %s", exc)
            return False
    else:
        # Live path - reuse the production crawler (yfinance: DXY ^GSPC VND=X
        # + SBV interbank). Often DNS-blocked from VN ISPs (TD-25): on empty
        # we fall through to the CSV scaffold instructions below.
        try:
            from src.data.crawlers import MacroCrawler

            LOGGER.info("Fetching live macro via MacroCrawler (yfinance/SBV)...")
            macro_df = MacroCrawler().fetch_macro(
                start_date=start, end_date=end, file_path=None
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Live macro fetch crashed: %s", exc)
            macro_df = pd.DataFrame()

    if macro_df is None or macro_df.empty:
        LOGGER.warning(
            "No macro rows fetched. Provide a CSV and re-run:\n"
            "    python backfill_context_data.py --macro-csv my_macro.csv\n"
            "Expected header:\n    %s",
            ",".join(MACRO_COLUMNS),
        )
        return False

    # Keep only known columns; coerce date to python date for clean upsert.
    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.date
    keep = [c for c in MACRO_COLUMNS if c in macro_df.columns]
    macro_df = macro_df[keep]
    macro_df = macro_df[macro_df["date"] <= pd.to_datetime(end).date()]

    # `DuckDBEngine()` already CREATE/ALTERed macro_daily to the canonical
    # schema on init. upsert_dataframe -> INSERT OR REPLACE BY NAME on PK(date).
    engine.upsert_dataframe(macro_df, "macro_daily")
    LOGGER.info(
        "MACRO upserted: %s rows, %s ... %s",
        len(macro_df), macro_df["date"].min(), macro_df["date"].max(),
    )
    return True


# ════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT BACKFILL - REAL news, scored by Gemini 3.5 Flash
# ════════════════════════════════════════════════════════════════════════
# NO dummy headlines. This delegates to the production SentimentCrawler:
#   • _active_tickers()    -> top liquid tickers from stock_ohlcv
#   • _fetch_gnews_items() -> REAL per-ticker Vietnamese GNews headlines
#   • _fetch_rss_items()   -> REAL market-wide RSS headlines
#   • _score_item()        -> REAL Gemini-3.5-Flash scoring (its own
#                            battle-tested DEFAULT_PROMPT, JSON mode)
# We only add the `ticker` column the legacy DDL forgot and upsert
# idempotently on PK(ticker, date, title).


def _ensure_sentiment_table(engine: DuckDBEngine) -> None:
    """Create `hist_sentiment_llm_labeled` with the correct SUPERSET schema.

    Includes `ticker` (required by Alpha360._query_sentiment but ABSENT from
    the legacy crawler DDL) and a PK so backfills are idempotent.
    """
    engine.conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SENTIMENT_TABLE} (
            ticker          VARCHAR,
            date            TIMESTAMP,
            title           VARCHAR,
            sentiment_score DOUBLE,
            magnitude       DOUBLE,
            reason          VARCHAR,
            url             VARCHAR,
            sentiment_nlp   DOUBLE,
            impact_force    DOUBLE,
            is_market_wide  BOOLEAN,
            PRIMARY KEY (ticker, date, title)
        )
        """
    )
    # Migrate a pre-existing (legacy) table that lacks `ticker`.
    cols = {
        r[0]
        for r in engine.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{SENTIMENT_TABLE}'"
        ).fetchall()
    }
    if "ticker" not in cols:
        LOGGER.warning("Legacy %s lacks `ticker` -> ADD COLUMN.", SENTIMENT_TABLE)
        engine.conn.execute(
            f"ALTER TABLE {SENTIMENT_TABLE} ADD COLUMN ticker VARCHAR"
        )


def _fetch_and_score_real_news(
    sent_start: str,
    end: str,
    max_tickers: int,
) -> list[dict]:
    """Fetch REAL Vietnamese news for the window and score each via Gemini.

    Delegates entirely to the production ``SentimentCrawler`` so the news
    discovery (GNews per-ticker + RSS market-wide) and the LLM scoring
    (gemini-3.5-flash with its tuned DEFAULT_PROMPT) are identical to the
    live daily pipeline. We only attach the ``ticker`` column the legacy
    DDL omitted, and constrain to the [sent_start, end] window.

    Returns a list of upsert-ready row dicts (possibly empty if the network
    / Gemini is unavailable - the caller logs and reports honestly).
    """
    from src.crawlers.sentiment_crawler import SentimentCrawler

    start_d = pd.to_datetime(sent_start).date()
    end_d = pd.to_datetime(end).date()

    crawler = SentimentCrawler(db_path=DB_PATH)
    if crawler._client is None:
        LOGGER.warning(
            "SentimentCrawler has no Gemini client (GEMINI_API_KEY unset or "
            "google-genai missing). Real scoring unavailable."
        )

    tickers = crawler._active_tickers(limit=max_tickers)
    LOGGER.info(
        "Real-news fetch | window %s..%s | %s tickers (by liquidity)",
        start_d, end_d, len(tickers),
    )

    items: list = []
    try:
        items.extend(crawler._fetch_rss_items(start_date=start_d, end_date=end_d))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("RSS fetch failed: %s", exc)
    try:
        items.extend(
            crawler._fetch_gnews_items(
                tickers=tickers, start_date=start_d, end_date=end_d
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GNews fetch failed: %s", exc)

    items = crawler._dedupe(items)
    # Keep only items whose published date is inside the requested window.
    items = [it for it in items if start_d <= it.date <= end_d]
    LOGGER.info("Real news items in window after dedupe/filter: %s", len(items))

    rows: list[dict] = []
    for idx, item in enumerate(items, start=1):
        scored = crawler._score_item(item)  # REAL Gemini call (or neutral fallback)
        scored["ticker"] = (item.ticker or "MARKET").upper()
        rows.append(scored)
        if idx % 10 == 0:
            LOGGER.info("  scored %s/%s items...", idx, len(items))
    return rows


def backfill_sentiment(
    engine: DuckDBEngine,
    sent_start: str,
    end: str,
    max_tickers: int,
) -> bool:
    """Fetch REAL news for [sent_start, end], Gemini-score it, upsert."""
    LOGGER.info("-- SENTIMENT backfill | REAL news %s..%s --", sent_start, end)
    _ensure_sentiment_table(engine)

    rows = _fetch_and_score_real_news(sent_start, end, max_tickers)
    if not rows:
        LOGGER.warning(
            "No real news rows produced (network/Gemini unavailable or no "
            "headlines in window). Sentiment table left unchanged."
        )
        return False

    sent_df = pd.DataFrame(rows)
    # Normalise dtypes for a clean PK(ticker,date,title) upsert.
    sent_df["date"] = pd.to_datetime(sent_df["date"])
    sent_df["ticker"] = sent_df["ticker"].astype(str).str.upper()
    sent_df["title"] = sent_df["title"].astype(str).str.slice(0, 1000)
    sent_df = sent_df[sent_df["date"] <= pd.to_datetime(end)]
    # Drop intra-batch dup keys so INSERT OR REPLACE doesn't choke.
    sent_df = sent_df.drop_duplicates(subset=["ticker", "date", "title"], keep="last")
    if sent_df.empty:
        LOGGER.warning("All fetched news fell outside the window - nothing upserted.")
        return False

    engine.upsert_dataframe(sent_df, SENTIMENT_TABLE)
    LOGGER.info(
        "SENTIMENT upserted: %s rows | tickers=%s | mean_score=%.3f | "
        "date span %s..%s",
        len(sent_df),
        sorted(sent_df["ticker"].unique())[:15],
        float(sent_df["sentiment_score"].mean()),
        sent_df["date"].min(), sent_df["date"].max(),
    )
    return True


# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Macro + Sentiment context.")
    parser.add_argument("--start", default=DEFAULT_START, help="Macro start YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end YYYY-MM-DD (default 2026-05-15).")
    parser.add_argument("--sent-start", default="2026-05-09", help="Sentiment news window start YYYY-MM-DD.")
    parser.add_argument("--max-tickers", type=int, default=30, help="Top-N liquid tickers to fetch news for.")
    parser.add_argument("--macro-csv", default=None, help="Optional macro CSV (yfinance fallback).")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    args = parser.parse_args()

    LOGGER.info("Backfill start | db=%s", DB_PATH)
    engine = DuckDBEngine(db_path=DB_PATH)  # creates/migrates macro_daily

    macro_ok = sentiment_ok = False
    if not args.skip_macro:
        try:
            macro_ok = backfill_macro(engine, args.start, args.end, args.macro_csv)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Macro backfill crashed.")
    if not args.skip_sentiment:
        try:
            sentiment_ok = backfill_sentiment(
                engine, args.sent_start, args.end, args.max_tickers
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Sentiment backfill crashed.")

    LOGGER.info("DONE | macro=%s sentiment=%s", macro_ok, sentiment_ok)
    LOGGER.info(
        "Next: let the OHLCV retrain finish, then re-run the fully-featured "
        "retrain:\n    python main.py --task build_alpha360 && "
        "python -m src.models.stacking_model.train_stacking"
    )
    # Non-zero only if BOTH halves failed (and neither was skipped).
    if not macro_ok and not sentiment_ok and not (args.skip_macro and args.skip_sentiment):
        sys.exit(1)


if __name__ == "__main__":
    main()
