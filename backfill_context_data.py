#!/usr/bin/env python3
"""Backfill missing Macro + LLM-Sentiment context into the Quant V6 DuckDB.

WHY THIS EXISTS
───────────────
The production DB (`data/quant_v6_core.duckdb`) was recovered OHLCV-only —
`macro_daily` and `hist_sentiment_llm_labeled` are absent, so the current
retrain runs with those features zero-filled. This script repopulates both
so the NEXT retrain is fully featured.

It deliberately REUSES existing infra (no divergent schemas):
  • Macro     → src.data.crawlers.MacroCrawler.fetch_macro  (yfinance live)
  • DB upsert → src.data.db_engine.DuckDBEngine.upsert_dataframe
                (INSERT OR REPLACE BY NAME, resolves on PRIMARY KEY)
  • Sentiment → google-genai Client, gemini-3.5-flash, EXACT call pattern
                copied from src.models.quant_agent_arbitrator

SCHEMA NOTES (important — read before editing)
──────────────────────────────────────────────
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
─────
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
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

from src.data.db_engine import DuckDBEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("backfill_context")

DB_PATH = "data/quant_v6_core.duckdb"
DEFAULT_START = "2014-01-01"
DEFAULT_END = "2026-05-16"  # yfinance `end` is exclusive → includes 2026-05-15

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
            LOGGER.warning("CSV missing optional macro column '%s' → NULL.", col)
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
    LOGGER.info("── MACRO backfill | window %s → %s ──", start, end)

    if macro_csv:
        LOGGER.info("Loading macro from CSV: %s", macro_csv)
        try:
            macro_df = _read_macro_csv(macro_csv)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Macro CSV load failed: %s", exc)
            return False
    else:
        # Live path — reuse the production crawler (yfinance: DXY ^GSPC VND=X
        # + SBV interbank). Often DNS-blocked from VN ISPs (TD-25): on empty
        # we fall through to the CSV scaffold instructions below.
        try:
            from src.data.crawlers import MacroCrawler

            LOGGER.info("Fetching live macro via MacroCrawler (yfinance/SBV)…")
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
    # schema on init. upsert_dataframe → INSERT OR REPLACE BY NAME on PK(date).
    engine.upsert_dataframe(macro_df, "macro_daily")
    LOGGER.info(
        "MACRO upserted: %s rows, %s … %s",
        len(macro_df), macro_df["date"].min(), macro_df["date"].max(),
    )
    return True


# ════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT BACKFILL (Gemini 3.5 Flash)
# ════════════════════════════════════════════════════════════════════════
SENTIMENT_SYSTEM_PROMPT = (
    "You are a Vietnamese equities sentiment analyst. For each news item you "
    "are given (ticker, date, headline), output a sentiment_score in "
    "[-1.0, 1.0] (-1 very bearish, 0 neutral, +1 very bullish), a magnitude "
    "in [0.0, 1.0] (confidence/impact), a one-sentence Vietnamese reason, and "
    "is_market_wide (true if it affects the whole market, not just the "
    "ticker). Return STRICT RAW JSON: a list of objects with keys "
    "ticker, date, sentiment_score, magnitude, reason, is_market_wide."
)


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
        LOGGER.warning("Legacy %s lacks `ticker` → ADD COLUMN.", SENTIMENT_TABLE)
        engine.conn.execute(
            f"ALTER TABLE {SENTIMENT_TABLE} ADD COLUMN ticker VARCHAR"
        )


def _dummy_news_batch(end: str) -> list[dict[str, str]]:
    """Dummy headlines (the user asked for dummy strings).

    Replace this with a real RSS/scrape feed for a production backfill —
    everything downstream (Gemini scoring + upsert) stays identical.
    """
    end_d = pd.to_datetime(end).date()
    tickers = ["PLX", "HPG", "VCB", "SSI", "FPT"]
    batch: list[dict[str, str]] = []
    for i, tkr in enumerate(tickers):
        d = (end_d - timedelta(days=i)).isoformat()
        batch.append(
            {
                "ticker": tkr,
                "date": d,
                "title": f"[DUMMY] {tkr} reports quarterly update on {d}",
            }
        )
    return batch


def _gemini_score(news_batch: list[dict[str, str]]) -> list[dict] | None:
    """Score a batch via google-genai / gemini-3.5-flash.

    Mirrors src.models.quant_agent_arbitrator exactly (Client +
    client.models.generate_content + JSON-mode + code-fence cleanup +
    retry/backoff). Returns None if the SDK / API key is unavailable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        LOGGER.warning("google-genai not installed → neutral fallback.")
        return None
    if not api_key:
        LOGGER.warning("GEMINI_API_KEY unset → neutral fallback.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    LOGGER.info("Gemini sentiment model=%s items=%s", model_name, len(news_batch))

    prompt = "Score these Vietnamese-market news items:\n\n" + "\n".join(
        f"- ticker={n['ticker']} date={n['date']} headline={n['title']!r}"
        for n in news_batch
    ) + "\n\nReturn a JSON list as specified."

    cfg = genai_types.GenerateContentConfig(
        system_instruction=SENTIMENT_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )

    for attempt, delay in enumerate(((0, 2, 5)), start=1):
        if delay:
            time.sleep(delay)
        try:
            resp = client.models.generate_content(
                model=model_name, contents=prompt, config=cfg
            )
            raw = (resp.text or "") if hasattr(resp, "text") else ""
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
                if not m:
                    raise
                parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, list) else parsed.get("items", [])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Gemini attempt %s failed: %s", attempt, exc)
    LOGGER.error("Gemini scoring exhausted retries → neutral fallback.")
    return None


def backfill_sentiment(engine: DuckDBEngine, end: str) -> bool:
    """Score dummy headlines with Gemini and upsert into the sentiment table."""
    LOGGER.info("── SENTIMENT backfill | up to %s ──", end)
    _ensure_sentiment_table(engine)

    news_batch = _dummy_news_batch(end)
    scored = _gemini_score(news_batch)

    rows: list[dict] = []
    if scored:
        by_key = {(s.get("ticker"), str(s.get("date"))): s for s in scored}
        for n in news_batch:
            s = by_key.get((n["ticker"], n["date"]), {})
            score = max(-1.0, min(1.0, float(s.get("sentiment_score", 0.0))))
            mag = max(0.0, min(1.0, float(s.get("magnitude", 0.0))))
            rows.append(
                {
                    "ticker": n["ticker"],
                    "date": pd.to_datetime(n["date"]),
                    "title": n["title"],
                    "sentiment_score": score,
                    "magnitude": mag,
                    "reason": str(s.get("reason", "")),
                    "url": "",
                    "sentiment_nlp": score,          # mirror score (no separate NLP pass)
                    "impact_force": mag,
                    "is_market_wide": bool(s.get("is_market_wide", False)),
                }
            )
    else:
        # Neutral fallback so the table is POPULATED and the pipeline
        # self-heals (zero-filled features are equivalent to neutral).
        for n in news_batch:
            rows.append(
                {
                    "ticker": n["ticker"],
                    "date": pd.to_datetime(n["date"]),
                    "title": n["title"],
                    "sentiment_score": 0.0,
                    "magnitude": 0.0,
                    "reason": "neutral fallback (no Gemini)",
                    "url": "",
                    "sentiment_nlp": 0.0,
                    "impact_force": 0.0,
                    "is_market_wide": False,
                }
            )

    sent_df = pd.DataFrame(rows)
    sent_df = sent_df[sent_df["date"] <= pd.to_datetime(end)]
    if sent_df.empty:
        LOGGER.warning("No sentiment rows to upsert.")
        return False

    # PK(ticker,date,title) → INSERT OR REPLACE makes this idempotent.
    engine.upsert_dataframe(sent_df, SENTIMENT_TABLE)
    LOGGER.info(
        "SENTIMENT upserted: %s rows, tickers=%s",
        len(sent_df), sorted(sent_df["ticker"].unique()),
    )
    return True


# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Macro + Sentiment context.")
    parser.add_argument("--start", default=DEFAULT_START, help="Macro start YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end YYYY-MM-DD (default 2026-05-15).")
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
            sentiment_ok = backfill_sentiment(engine, args.end)
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
