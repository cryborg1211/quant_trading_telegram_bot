import logging
import os
from datetime import datetime
import duckdb
import pandas as pd
from typing import Optional, Union
import threading

LOGGER = logging.getLogger(__name__)


class DuckDBEngine:
    """Singleton engine for managing the DuckDB connection.

    ⚠ CRITICAL DuckDB CONSTRAINT ⚠
    ──────────────────────────────
    DuckDB refuses to open a second connection to the same database file
    inside the same process if the configurations differ. Production crash:

        Connection Error: Can't open a connection to same database file
        with a different configuration than existing connections.

    This means EVERY call to `duckdb.connect()` against `data/quant_v6_core.duckdb`
    in this codebase MUST use the same arguments. Concretely:
        • Do NOT pass `read_only=True` anywhere — the singleton holds
          the connection open as read-write (default).
        • Do NOT pass alternate `config={...}` dicts.
        • Prefer reusing `DuckDBEngine().conn` over opening side connections.

    Background read-only side connections (e.g. in alpha360_generator)
    have been stripped of their `read_only=True` flag for this reason.
    The thread-safety we still get from DuckDB's internal mutex is
    sufficient — DuckDB allows multiple READ cursors on a single
    connection concurrently.

    Init-time table creation runs inside the `_lock` so concurrent
    constructions don't race on `ALTER TABLE` / `CREATE SEQUENCE`.
    """

    _instance: Optional['DuckDBEngine'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure exactly one DuckDBEngine instance per process (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DuckDBEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "data/quant_v6_core.duckdb"):
        """Open the single DuckDB connection and run idempotent table init.

        The whole init body runs under `_lock` so concurrent first-time
        constructors from the bot's worker threads cannot race on
        `duckdb.connect` / `_init_tables` / `_init_portfolio_table`.

        Args:
            db_path (str): Path to the DuckDB database file. Only the FIRST
                constructor's `db_path` is honored — subsequent calls reuse
                the singleton and ignore this argument (DuckDB allows only
                one open file per singleton instance).
        """
        # Fast-path: skip if already initialized. Re-check inside the lock.
        if getattr(self, "initialized", False):
            return

        with type(self)._lock:
            # Re-check under the lock — another thread may have completed
            # initialization while we were waiting for the lock.
            if getattr(self, "initialized", False):
                return

            self.db_path = db_path
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # ⚠ DO NOT pass `read_only=` or `config=` here.
            # All other callers in the codebase MUST also use a bare
            # `duckdb.connect(path)` so the per-process config matches.
            self.conn = duckdb.connect(self.db_path)

            # Lock for serializing audit-log writes from async worker threads.
            # Created BEFORE `_init_tables` in case any future table-init step
            # wants to use it.
            self._audit_lock = threading.Lock()

            self._init_tables()
            self.initialized = True
            LOGGER.info("[DuckDB] Engine initialized: %s", self.db_path)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the singleton's connection.

        Preferred over `duckdb.connect(path, ...)` from outside this module —
        reusing this connection bypasses the config-mismatch crash entirely
        and shares DuckDB's internal locking with the rest of the codebase.
        """
        return self.conn

    def _init_tables(self):
        """Creates the core research tables if they do not exist."""
        # 1. stock_ohlcv
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_ohlcv (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                adj_close DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)

        # 2. macro_daily — wide-format daily macro features.
        # Schema-evolution safe: new columns (vnibor 1-month tenor, inflation_yoy)
        # are added in-place to existing tables on next bot/cron startup.
        self._init_macro_daily_table()

        # 3. sentiment_score
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_score (
                ticker VARCHAR,
                date DATE,
                sentiment_nlp DOUBLE,
                impact_force DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)

        # 4. macro_economic_raw (Qlib-style Long Format)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_economic_raw (
                date DATE,
                indicator_name VARCHAR,
                value DOUBLE,
                PRIMARY KEY (date, indicator_name)
            )
        """)
        
        # 5. live_positions (Active Portfolio Tracking)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS live_positions (
                telegram_id VARCHAR,
                ticker VARCHAR,
                quantity INTEGER,
                entry_price DOUBLE,
                entry_date DATE,
                PRIMARY KEY (telegram_id, ticker)
            )
        """)
        
        # 6. trade_history (Historical Logs for RL/Reporting)
        # Split into two execute() calls: DuckDB's behaviour for multi-statement
        # execute() is undocumented and version-sensitive, so we run them separately
        # to guarantee both statements actually fire.
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_trade_id START 1")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER DEFAULT nextval('seq_trade_id'),
                telegram_id VARCHAR,
                ticker VARCHAR,
                action VARCHAR,
                price DOUBLE,
                date DATE,
                pnl_percent DOUBLE,
                PRIMARY KEY (id)
            )
        """)
        
        # 7. rl_mistake_logs (Phase 3 Reinforcement Learning preparation)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rl_mistake_logs (
                ticker VARCHAR,
                predicted_date DATE,
                predicted_action VARCHAR,
                actual_t5_outcome DOUBLE,
                features_snapshot VARCHAR
            )
        """)

        # 8. portfolio (User-managed bot portfolio for /add, /suggest_sell, /remove)
        # Multi-user safe: every row is tagged with the Telegram user_id.
        # Migration: a previous deploy may have created this table WITHOUT
        # user_id. Detect and ALTER in place so existing rows survive.
        self._init_portfolio_table()

        # 9. audit_log (Per-user command audit trail for weekly/monthly review)
        # `self._audit_lock` was already created in `__init__` before this method
        # was called, so audit writes can begin as soon as the table exists.
        self._init_audit_log_table()

    # Columns the wide-format macro_daily table is expected to expose.
    # Adding to this list automatically migrates existing tables on next
    # DuckDBEngine() init via ALTER TABLE ADD COLUMN.
    _MACRO_DAILY_COLUMNS: tuple[tuple[str, str], ...] = (
        ("date", "DATE"),
        ("dxy_close", "DOUBLE"),
        ("sp500_close", "DOUBLE"),
        ("usd_vnd", "DOUBLE"),
        ("interbank_on_rate", "DOUBLE"),
        ("vnibor", "DOUBLE"),          # 1-month VN interbank rate (new)
        ("inflation_yoy", "DOUBLE"),   # VN CPI YoY % (new, monthly cadence)
    )

    def _init_macro_daily_table(self) -> None:
        """Create or migrate the `macro_daily` table to the latest column set."""
        table_exists = bool(
            self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'macro_daily'"
            ).fetchone()[0]
        )

        if not table_exists:
            cols_sql = ",\n                ".join(
                f"{name} {dtype}" for name, dtype in self._MACRO_DAILY_COLUMNS
            )
            self.conn.execute(f"""
                CREATE TABLE macro_daily (
                    {cols_sql},
                    PRIMARY KEY (date)
                )
            """)
            LOGGER.info("[DuckDB] Created `macro_daily` with %s columns.", len(self._MACRO_DAILY_COLUMNS))
            return

        existing_cols = {
            r[0]
            for r in self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'macro_daily'"
            ).fetchall()
        }

        for col_name, col_type in self._MACRO_DAILY_COLUMNS:
            if col_name not in existing_cols:
                LOGGER.info("[DuckDB] Migrating macro_daily: ADD COLUMN %s %s", col_name, col_type)
                self.conn.execute(f"ALTER TABLE macro_daily ADD COLUMN {col_name} {col_type}")

    def _init_audit_log_table(self) -> None:
        """Create the per-user command audit-trail table.

        Schema is append-only — no PRIMARY KEY constraint by design, since a
        single user can issue the same command twice within the same second
        (e.g., `/verify HPG` rapid double-tap), and we want both rows.

        Used for weekly / monthly performance reviews:
            • most-requested tickers per user
            • command usage frequency
            • timing of trades (`/add` timestamps) for backtest reconciliation
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                user_id   VARCHAR,
                command   VARCHAR,
                ticker    VARCHAR,
                details   VARCHAR,
                timestamp TIMESTAMP
            )
        """)

    def log_user_action(
        self,
        user_id: str,
        command: str,
        ticker: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Append one row to `audit_log`. Parameterized; safe against injection.

        Args:
            user_id: Telegram user_id (stringified — consistent with the
                portfolio table's `user_id` column).
            command: short slash-command label, e.g. "add", "remove",
                "suggest_buy", "suggest_sell", "verify". No leading slash.
            ticker: optional VN equity symbol relevant to the command.
            details: optional free-form string (cap at ~200 chars upstream
                so the column doesn't grow unboundedly). Use for parsed
                arguments like volume/price or portfolio_size.

        Never raises (best-effort logging) — a failure here must not block
        the user-facing command. The caller is responsible for catching the
        exception if it cares.
        """
        # Truncate defensively so a runaway caller can't bloat the table.
        safe_command = (command or "")[:64]
        safe_ticker = (ticker[:16] if ticker else None)
        safe_details = (details[:200] if details else None)

        with self._audit_lock:
            self.conn.execute(
                "INSERT INTO audit_log (user_id, command, ticker, details, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                [user_id, safe_command, safe_ticker, safe_details, datetime.now()],
            )

    def _init_portfolio_table(self) -> None:
        """Create or migrate the `portfolio` table to the multi-user schema."""
        # Does the table exist?
        table_exists = bool(
            self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'portfolio'"
            ).fetchone()[0]
        )

        if not table_exists:
            self.conn.execute("""
                CREATE TABLE portfolio (
                    user_id VARCHAR,
                    ticker VARCHAR,
                    volume INTEGER,
                    price DOUBLE,
                    added_at TIMESTAMP
                )
            """)
            LOGGER.info("[DuckDB] Created `portfolio` table (multi-user schema).")
            return

        # Table exists — check whether the user_id column is present.
        has_user_id = bool(
            self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name = 'portfolio' AND column_name = 'user_id'"
            ).fetchone()[0]
        )

        if not has_user_id:
            LOGGER.warning(
                "[DuckDB] Legacy single-user `portfolio` schema detected. "
                "Adding `user_id` column; existing rows tagged user_id='legacy' "
                "so they do not appear in any user's /suggest_sell query."
            )
            self.conn.execute("ALTER TABLE portfolio ADD COLUMN user_id VARCHAR")
            self.conn.execute(
                "UPDATE portfolio SET user_id = 'legacy' WHERE user_id IS NULL"
            )
            LOGGER.info("[DuckDB] Portfolio migration complete.")

    def upsert_dataframe(self, df: pd.DataFrame, table_name: str):
        """Performs an UPSERT (Insert or Replace) from a Pandas DataFrame.

        DuckDB's 'INSERT OR REPLACE' syntax automatically resolves conflicts
        based on the PRIMARY KEY constraints defined during table initialization.

        Args:
            df (pd.DataFrame): The source data to be inserted or updated.
            table_name (str): Target table name in the database.

        Raises:
            ValueError: If the table name is invalid or not initialized.
        """
        if df.empty:
            return

        # DuckDB can query Pandas DataFrames directly if they are in scope
        # We use 'INSERT OR REPLACE' for the Upsert behavior
        try:
            # Using 'BY NAME' ensures DuckDB maps DataFrame columns to table columns by their headers
            self.conn.execute(f"INSERT OR REPLACE INTO {table_name} BY NAME SELECT * FROM df")
        except Exception as e:
            cols = df.columns.tolist()
            raise RuntimeError(f"Failed to upsert data into {table_name}. DF Columns: {cols}. Error: {e}")

    def query(self, sql: str) -> pd.DataFrame:
        """Executes a SQL query and returns a Pandas DataFrame.

        Args:
            sql (str): SQL query string.

        Returns:
            pd.DataFrame: Result of the query.
        """
        return self.conn.execute(sql).df()

    def close(self):
        """Closes the database connection and resets the singleton state."""
        if hasattr(self, 'conn'):
            try:
                self.conn.close()
            except Exception:
                pass
            del self.conn
        
        # Reset initialization flag to allow re-initialization if needed
        if hasattr(self, 'initialized'):
            del self.initialized
            
        # Clear the singleton instance
        DuckDBEngine._instance = None
        print("DuckDB connection closed and engine state reset.")

    @classmethod
    def dispose(cls):
        """Class method to safely dispose of the singleton instance."""
        if cls._instance:
            cls._instance.close()

if __name__ == "__main__":
    # Internal validation of singleton and initialization
    engine = DuckDBEngine()
    print(f"Connected to: {engine.db_path}")
    
    # Check tables
    tables = engine.query("SHOW TABLES")
    print("Tables initialized:")
    print(tables)
