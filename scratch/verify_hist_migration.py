import duckdb

con = duckdb.connect("data/quant_v6_core.duckdb", read_only=True)
try:
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'hist_%' ORDER BY table_name"
    ).fetchall()
    print("HIST_TABLES")
    for (table_name,) in tables:
        print(table_name)

    print("\nROW_COUNTS")
    for table in [
        "hist_unified_research_matrix",
        "hist_sentiment_llm_labeled",
        "hist_macro_features_10y",
    ]:
        row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"{table},{row[0]}")
finally:
    con.close()