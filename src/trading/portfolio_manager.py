"""Portfolio manager – parameterized SQL, config-driven thresholds.

TD-33 (May 2026): unified portfolio storage on the `portfolio` table.
Previously `PortfolioManager` wrote to `live_positions` and the bot's `/add`
wrote to `portfolio` — two never-synced tables. Now BOTH paths share the
same `portfolio` table, distinguished by `user_id`:

    • Human users:    user_id = "<telegram_user_id>"  (set by bot handlers)
    • Automated cron: user_id = "cron"                (default for PortfolioManager)

The bot's `/suggest_sell` query already filters by `user_id`, so users only
see their own /add holdings — they cannot accidentally sell the cron's
automated positions.

Schema mapping live_positions → portfolio:
    telegram_id   → user_id     (VARCHAR)
    quantity      → volume      (INTEGER)
    entry_price   → price       (DOUBLE)
    entry_date    → added_at    (TIMESTAMP — was DATE)

Internally we alias the new columns back to their old names in `_query_positions`
so the iteration code in `update_live_performance` / `process_daily_trades`
keeps using the readable `entry_price` / `quantity` names without a deeper rewrite.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from src.data.db_engine import DuckDBEngine
from config.settings import CONFIG

LOGGER = logging.getLogger(__name__)

# Sentinel user_id for automated cron trades — distinguishes them from any
# real Telegram user_id (always numeric digits).
CRON_USER_ID: str = "cron"


class PortfolioManager:
    """
    Manages portfolio state using DuckDB.
    Enforces risk management (Stop-Loss / Take-Profit) and capital allocation.

    All SQL queries use parameterized placeholders (?) to prevent injection.
    All thresholds are sourced from CONFIG.trading.

    Backwards-compat note: the legacy `live_positions` table is no longer
    written to. If a deploy is rolling back to pre-TD-33 code, any orphaned
    rows in `live_positions` remain readable but stale — restore from
    `data/backups/` to recover historical positions.
    """

    def __init__(self, user_id: str | None = None) -> None:
        # Backwards-compat: accept `telegram_id=` callers via the param name
        # change. Default to CRON_USER_ID for the automated cron path.
        self.user_id = user_id or CRON_USER_ID
        self.db = DuckDBEngine()
        self._stop_loss = CONFIG.trading.stop_loss_pct        # -0.07
        self._take_profit = CONFIG.trading.take_profit_pct    # +0.15
        self._fee_rate = CONFIG.trading.fee_rate              # 0.002
        self._alloc = CONFIG.trading.virtual_allocation_per_ticker  # 10_000_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_positions(self, cols: str | None = None) -> pd.DataFrame:
        """Fetch this user's portfolio rows from the unified `portfolio` table.

        Columns are aliased to the legacy names (`quantity`, `entry_price`,
        `entry_date`) so the iteration logic in `update_live_performance` /
        `process_daily_trades` continues to use the readable column names
        without per-row renaming.

        The `cols` argument is accepted for back-compat with the old
        signature but currently ignored — we always project the same alias
        set since the consumer code references specific column names.
        """
        del cols  # legacy parameter, intentionally unused
        return self.db.conn.execute(
            """
            SELECT
                ticker,
                volume        AS quantity,
                price         AS entry_price,
                CAST(added_at AS DATE) AS entry_date
            FROM portfolio
            WHERE user_id = ?
            """,
            [self.user_id],
        ).df()

    def _execute_sell(
        self,
        ticker: str,
        exec_price: float,
        qty: int,
        pnl_pct: float,
        date_str: str,
        reason: str,
        report_lines: list[str],
    ) -> None:
        """Parameterized SELL: remove from `portfolio`, log to `trade_history`."""
        self.db.conn.execute(
            "DELETE FROM portfolio WHERE user_id = ? AND ticker = ?",
            [self.user_id, ticker],
        )
        # trade_history.telegram_id is a legacy column name (VARCHAR) — we
        # continue writing the (now-generalized) user_id into it. No schema
        # change to keep the rollback path simple.
        self.db.conn.execute(
            """INSERT INTO trade_history
               (telegram_id, ticker, action, price, date, pnl_percent)
               VALUES (?, ?, 'SELL', ?, ?, ?)""",
            [self.user_id, ticker, exec_price, date_str, pnl_pct],
        )
        icon = "🟢" if pnl_pct > 0 else "🔴"
        report_lines.append(
            f"{icon} SOLD {qty:,} {ticker} @ {exec_price:,.0f}"
            f" | PnL: {pnl_pct * 100:+.1f}% | Reason: {reason}"
        )

    def _execute_buy(
        self,
        ticker: str,
        exec_price: float,
        date_str: str,
        report_lines: list[str],
    ) -> None:
        """Parameterized BUY: insert into `portfolio` and `trade_history`."""
        qty = int(self._alloc // exec_price)
        qty = max(100, (qty // 100) * 100)

        # Cron trades land with user_id='cron' by default — invisible to
        # users' `/suggest_sell` queries which filter by their telegram_id.
        self.db.conn.execute(
            """INSERT INTO portfolio
               (user_id, ticker, volume, price, added_at)
               VALUES (?, ?, ?, ?, ?)""",
            [self.user_id, ticker, qty, exec_price, datetime.now()],
        )
        self.db.conn.execute(
            """INSERT INTO trade_history
               (telegram_id, ticker, action, price, date, pnl_percent)
               VALUES (?, ?, 'BUY', ?, ?, 0.0)""",
            [self.user_id, ticker, exec_price, date_str],
        )
        report_lines.append(
            f"🔵 SIGNAL EXECUTED: BOUGHT {qty:,} {ticker} @ {exec_price:,.0f} VND"
            f" [user_id={self.user_id}]"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_live_performance(self, current_market_data: Mapping[str, float | int]) -> list[str]:
        """
        Calculates floating PnL for active positions.
        Triggers a SELL if Hard Stop-Loss or Take-Profit threshold is hit.

        Args:
            current_market_data: Mapping of ticker → current price, e.g. {"FPT": 136000}.

        Returns:
            List of human-readable report lines for sold positions.
        """
        report_lines: list[str] = []
        date_str = datetime.now().strftime("%Y-%m-%d")

        positions_df = self._query_positions()
        if positions_df.empty:
            return report_lines

        for _, row in positions_df.iterrows():
            ticker: str = row["ticker"]
            if ticker not in current_market_data:
                continue

            entry_price = float(row["entry_price"])
            qty = int(row["quantity"])
            current_price = float(current_market_data[ticker])
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct <= self._stop_loss:
                reason = f"Stop-Loss hit ({pnl_pct * 100:.1f}%)"
            elif pnl_pct >= self._take_profit:
                reason = f"Take-Profit hit ({pnl_pct * 100:.1f}%)"
            else:
                continue

            self._execute_sell(ticker, current_price, qty, pnl_pct, date_str, reason, report_lines)

        return report_lines

    def process_daily_trades(
        self,
        top_buy_signals: list[str],
        next_day_open_prices: Mapping[str, float | int],
        predictions: Mapping[str, int] | None = None,
    ) -> str:
        """
        Executes daily trading logic based on buy signals.

        Args:
            top_buy_signals: Tickers to buy.
            next_day_open_prices: Ticker → T+1 open price.
            predictions: Optional ticker → class label (0=DOWN, 1=SIDE, 2=UP).

        Returns:
            Formatted daily trading report string.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_lines: list[str] = [f"\n--- 📝 DAILY TRADING REPORT ({date_str}) ---"]

        positions_df = self._query_positions("ticker, quantity, entry_price")
        current_holdings: set[str] = (
            set(positions_df["ticker"].tolist()) if not positions_df.empty else set()
        )

        # --- SELL LOGIC (model-driven) ---
        if not positions_df.empty and predictions:
            for _, row in positions_df.iterrows():
                ticker: str = row["ticker"]
                if ticker in next_day_open_prices and predictions.get(ticker) == 0:
                    exec_price = float(next_day_open_prices[ticker])
                    entry_price = float(row["entry_price"])
                    pnl_pct = (exec_price - entry_price) / entry_price
                    self._execute_sell(
                        ticker, exec_price, int(row["quantity"]),
                        pnl_pct, date_str, "Model predicted DOWN (Class 0)", report_lines,
                    )
                    current_holdings.discard(ticker)

        # --- BUY LOGIC ---
        valid_buys = [
            t for t in top_buy_signals
            if t in next_day_open_prices and t not in current_holdings
        ]
        for ticker in valid_buys:
            self._execute_buy(ticker, float(next_day_open_prices[ticker]), date_str, report_lines)

        if len(report_lines) == 1:
            report_lines.append("No trades executed today.")

        report_lines.append("-" * 50)
        report_str = "\n".join(report_lines)
        LOGGER.info("Daily trading report:\n%s", report_str)
        return report_str