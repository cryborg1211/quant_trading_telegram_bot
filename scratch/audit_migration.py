import sqlite3
from pathlib import Path

try:
    import duckdb
except Exception as exc:
    duckdb = None
    print(f"DUCKDB_IMPORT_ERROR: {exc}")

old = Path("old-data")
data = Path("data")

sqlite_files = [p for p in old.rglob("*") if p.suffix.lower() in (".db", ".sqlite")]
duck_files = [p for p in data.rglob("*") if p.suffix.lower() in (".duckdb", ".db")]

print("SQLITE_FILES")
for p in sqlite_files:
    print(str(p))

print("\nDUCKDB_FILES")
for p in duck_files:
    print(str(p))

print("\nSQLITE_SCHEMAS")
for p in sqlite_files:
    print(f"## {p}")
    con = sqlite3.connect(str(p))
    cur = con.cursor()
    tables = cur.execute(
        "SELECT name,type FROM sqlite_master "
        "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    for name, typ in tables:
        print(f"[{typ}] {name}")
        for col in cur.execute(f'PRAGMA table_info("{name}")').fetchall():
            print("  ", col)
        try:
            rows = cur.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            print("  rows=", rows)
        except Exception as exc:
            print("  rows_err=", exc)
    con.close()

print("\nDUCKDB_SCHEMAS")
if duckdb is None:
    print("DuckDB unavailable")
else:
    for p in duck_files:
        print(f"## {p}")
        con = duckdb.connect(str(p), read_only=True)
        try:
            tables = con.execute(
                "SELECT table_schema, table_name, table_type "
                "FROM information_schema.tables "
                "WHERE table_schema NOT IN ('pg_catalog','information_schema') "
                "ORDER BY table_schema, table_name"
            ).fetchall()
        except Exception as exc:
            print("tables_err=", exc)
            tables = []
        for schema, name, typ in tables:
            print(f"[{typ}] {schema}.{name}")
            try:
                for col in con.execute(f'DESCRIBE "{schema}"."{name}"').fetchall():
                    print("  ", col)
                rows = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{name}"').fetchone()[0]
                print("  rows=", rows)
            except Exception as exc:
                print("  desc_err=", exc)
        con.close()