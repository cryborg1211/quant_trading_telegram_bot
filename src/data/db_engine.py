import os
import duckdb
import pandas as pd
from typing import Optional, Union
import threading

class DuckDBEngine:
    """Singleton engine for managing DuckDB connections and operations.
    
    This class ensures a single persistent connection to the DuckDB database
    and provides methods for table initialization and data upsertion.
    """
    
    _instance: Optional['DuckDBEngine'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensures singleton instance creation with thread safety."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DuckDBEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "data/quant_v6_core.duckdb"):
        """Initializes the DuckDB connection and sets up required tables.

        Args:
            db_path (str): Path to the DuckDB database file.
        """
        # Ensure __init__ only runs once for the singleton
        if not hasattr(self, 'initialized'):
            self.db_path = db_path
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Establish connection
            self.conn = duckdb.connect(self.db_path)
            self._init_tables()
            self.initialized = True

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

        # 2. macro_daily
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_daily (
                date DATE,
                dxy_close DOUBLE,
                sp500_close DOUBLE,
                usd_vnd DOUBLE,
                interbank_on_rate DOUBLE,
                PRIMARY KEY (date)
            )
        """)

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
