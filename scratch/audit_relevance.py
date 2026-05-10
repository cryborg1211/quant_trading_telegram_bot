import sqlite3
import duckdb
from pathlib import Path

sqlite_path = Path("old-data/master_quant_database.db")
duck_path = Path("data/quant_v6_core.duckdb")

sqlite_tables = [
    "sentiment_LLM_labeled",
    "macro_features_10y_macro_data",
    "UNIFIED_RESEARCH_MATRIX",
    "clean_quant_matrix_backtest_ready_matrix",
    "hose_4years_enriched_matrix",
    "hose_4years_market_data",
]

print("RELEVANCE_STATS_SQLITE")
scon = sqlite3.connect(str(sqlite_path))
scur = scon.cursor()
for table in sqlite_tables:
    print(f"## {table}")
    cols = [r[1] for r in scur.execute(f'PRAGMA table_info("{table}")').fetchall()]
    date_col = "date" if "date" in cols else "Date" if "Date" in cols else None
    ticker_col = "ticker" if "ticker" in cols else None
    print("columns=", ",".join(cols))
    print("rows=", scur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    if date_col:
        print("date_range=", scur.execute(f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{table}"').fetchone())
    if ticker_col:
        print("tickers=", scur.execute(f'SELECT COUNT(DISTINCT "{ticker_col}") FROM "{table}"').fetchone()[0])
    nan_checks = []
    for c in cols:
        try:
            n = scur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{c}" IN (\'NaN\', \'nan\', \'NULL\', \'\')').fetchone()[0]
            if n:
                nan_checks.append((c, n))
        except Exception:
            pass
    print("string_nan_empty=", nan_checks)

print("\nDUCKDB_RANGE_STATS")
dcon = duckdb.connect(str(duck_path), read_only=True)
for table in ["stock_ohlcv", "macro_daily", "sentiment_score"]:
    print(f"## {table}")
    cols = [r[0] for r in dcon.execute(f'DESCRIBE "{table}"').fetchall()]
    print("columns=", ",".join(cols))
    print("rows=", dcon.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    if "date" in cols:
        print("date_range=", dcon.execute(f'SELECT MIN(date), MAX(date) FROM "{table}"').fetchone())
    if "ticker" in cols:
        print("tickers=", dcon.execute(f'SELECT COUNT(DISTINCT ticker) FROM "{table}"').fetchone()[0])
dcon.close()
scon.close()