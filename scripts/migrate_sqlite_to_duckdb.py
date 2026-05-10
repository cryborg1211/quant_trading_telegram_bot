from __future__ import annotations

import duckdb
from pathlib import Path

DUCKDB_PATH = Path("data/quant_v6_core.duckdb")
SQLITE_PATH = Path("old-data/master_quant_database.db")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def text_expr(col: str, alias: str | None = None) -> str:
    out = f"NULLIF(CAST({qident(col)} AS VARCHAR), '')"
    return f"{out} AS {qident(alias or col)}"


def numeric_expr(col: str, alias: str | None = None) -> str:
    out = (
        f"CASE WHEN CAST({qident(col)} AS VARCHAR) IN ('NaN','nan','NULL','') "
        f"THEN NULL ELSE TRY_CAST({qident(col)} AS DOUBLE) END"
    )
    return f"{out} AS {qident(alias or col)}"


def integer_expr(col: str, alias: str | None = None) -> str:
    out = (
        f"CASE WHEN CAST({qident(col)} AS VARCHAR) IN ('NaN','nan','NULL','') "
        f"THEN NULL ELSE TRY_CAST({qident(col)} AS BIGINT) END"
    )
    return f"{out} AS {qident(alias or col)}"


def date_expr(col: str, alias: str | None = None) -> str:
    out = (
        f"COALESCE("
        f"TRY_CAST({qident(col)} AS TIMESTAMP), "
        f"try_strptime({qident(col)}, '%a, %d %b %Y %H:%M:%S GMT')"
        f")"
    )
    return f"{out} AS {qident(alias or col)}"


def migrate_table(con: duckdb.DuckDBPyConnection, source: str, dest: str, select_exprs: list[str]) -> tuple[int, int]:
    source_count = con.execute(f"SELECT COUNT(*) FROM old_sqlite.{qident(source)}").fetchone()[0]
    select_sql = ",\n            ".join(select_exprs)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {qident(dest)} AS
        SELECT
            {select_sql}
        FROM old_sqlite.{qident(source)}
        """
    )
    dest_count = con.execute(f"SELECT COUNT(*) FROM {qident(dest)}").fetchone()[0]
    return source_count, dest_count


def main() -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB not found: {DUCKDB_PATH}")
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite not found: {SQLITE_PATH}")

    con = duckdb.connect(str(DUCKDB_PATH))
    attached = False

    try:
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")
        con.execute(f"ATTACH '{SQLITE_PATH.as_posix()}' AS old_sqlite (TYPE SQLITE);")
        attached = True

        migrations: list[tuple[str, str, list[str]]] = [
            (
                "UNIFIED_RESEARCH_MATRIX",
                "hist_unified_research_matrix",
                [
                    text_expr("ticker"),
                    date_expr("date"),
                    numeric_expr("open"),
                    numeric_expr("high"),
                    numeric_expr("low"),
                    numeric_expr("close"),
                    numeric_expr("volume"),
                    numeric_expr("pct"),
                    numeric_expr("vol_pct"),
                    numeric_expr("ratio_event"),
                    numeric_expr("adj_factor"),
                    numeric_expr("adj_close"),
                    numeric_expr("adj_open"),
                    numeric_expr("adj_high"),
                    numeric_expr("adj_low"),
                    numeric_expr("DXY_pct"),
                    numeric_expr("USDVND_pct"),
                    numeric_expr("SP500_pct"),
                    numeric_expr("Sentiment_Score_orig"),
                    integer_expr("Target_Buy"),
                    numeric_expr("Sentiment_NLP"),
                    numeric_expr("Magnitude_NLP"),
                    numeric_expr("Impact_Force"),
                    numeric_expr("rsi"),
                    numeric_expr("sma_50"),
                    numeric_expr("bb_width"),
                    numeric_expr("atr_pct"),
                    numeric_expr("vol_ma20"),
                    numeric_expr("ema_200"),
                    numeric_expr("dist_ema200"),
                    numeric_expr("MACD_hist"),
                    numeric_expr("proba_up"),
                    numeric_expr("IR_pct"),
                ],
            ),
            (
                "sentiment_LLM_labeled",
                "hist_sentiment_llm_labeled",
                [
                    date_expr("Date", "date"),
                    text_expr("Title", "title"),
                    numeric_expr("Sentiment_Score", "sentiment_score"),
                    numeric_expr("Magnitude", "magnitude"),
                    text_expr("Reason", "reason"),
                    text_expr("URL", "url"),
                    numeric_expr("Sentiment_NLP", "sentiment_nlp"),
                    numeric_expr("Impact_Force", "impact_force"),
                    "TRUE AS is_market_wide",
                ],
            ),
            (
                "macro_features_10y_macro_data",
                "hist_macro_features_10y",
                [
                    date_expr("Date", "date"),
                    numeric_expr("DXY_pct", "dxy_pct"),
                    numeric_expr("USDVND_pct", "usdvnd_pct"),
                    numeric_expr("SP500_pct", "sp500_pct"),
                    numeric_expr("Sentiment_Score", "sentiment_score"),
                ],
            ),
        ]

        summary: list[tuple[str, str, int, int, str]] = []
        for source, dest, exprs in migrations:
            source_count, dest_count = migrate_table(con, source, dest, exprs)
            status = "OK" if source_count == dest_count else "MISMATCH"
            summary.append((source, dest, source_count, dest_count, status))

        con.execute("DETACH old_sqlite;")
        attached = False

        print("MIGRATION_SUMMARY")
        print("source_table,destination_table,source_rows,destination_rows,status")
        for row in summary:
            print(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}")

        print("\nDUCKDB_VERIFICATION")
        for _, dest, _, _, _ in summary:
            count = con.execute(f"SELECT COUNT(*) FROM {qident(dest)}").fetchone()[0]
            min_max = con.execute(f"SELECT MIN(date), MAX(date) FROM {qident(dest)}").fetchone()
            print(f"{dest}: rows={count}, date_min={min_max[0]}, date_max={min_max[1]}")

        if any(row[4] != "OK" for row in summary):
            raise RuntimeError("One or more row-count checks failed.")

    finally:
        if attached:
            try:
                con.execute("DETACH old_sqlite;")
            except Exception:
                pass
        con.close()


if __name__ == "__main__":
    main()