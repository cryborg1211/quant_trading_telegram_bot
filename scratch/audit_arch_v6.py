from pathlib import Path

import duckdb
import polars as pl

data = Path("data")
ohlcv_files = sorted(data.glob("ohlcv_*.parquet"))
print("DATA_FILES")
print(f"ohlcv_parquet_count={len(ohlcv_files)}")
print(f"has_alpha360={Path('data/alpha360_features.parquet').exists()}")
print(f"has_macro_daily_parquet={Path('data/macro_daily.parquet').exists()}")
print(f"duckdb_exists={Path('data/quant_v6_core.duckdb').exists()}")

if Path("data/alpha360_features.parquet").exists():
    df = pl.scan_parquet("data/alpha360_features.parquet")
    schema = df.collect_schema()
    rows = df.select(pl.len()).collect().item()
    print(f"alpha360_rows={rows}")
    print(f"alpha360_cols={len(schema)}")

con = duckdb.connect("data/quant_v6_core.duckdb", read_only=True)
try:
    print("\nDUCKDB_TABLES")
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()
    for (table,) in rows:
        count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"{table},{count}")
finally:
    con.close()