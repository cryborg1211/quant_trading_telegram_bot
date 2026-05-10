"""Portfolio manager – parameterized SQL, config-driven thresholds."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from config.settings import CONFIG
from src.data.db_engine import DuckDBEngine

LOGGER = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolio state using DuckDB.
    Enforces risk management (Stop-Loss / Take-Profit) and capital allocation.

    All SQL queries use parameterized placeholders (?) to prevent injection.
    All thresholds are sourced from CONFIG.trading.
    """

    def __init__(self, telegram_id: str | None = None) -> None:
        self.telegram_id = telegram_id or CONFIG.trading.default_telegram_id
        self.db = DuckDBEngine()
        self._stop_loss = CONFIG.trading.stop_loss_pct        # -0.07
        self._take_profit = CONFIG.trading.take_profit_pct    # +0.15
        self._fee_rate = CONFIG.trading.fee_rate              # 0.002
        self._alloc = CONFIG.trading.virtual_allocation_per_ticker  # 10_000_000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_positions(self, cols: str = "*") -> pd.DataFrame:
        """Fetch live positions for this user using a parameterized query."""
        return self.db.conn.execute(
            f"SELECT {cols} FROM live_positions WHERE telegram_id = ?",
            [self.telegram_id],
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
        """Parameterized SELL: remove from live_positions, log to trade_history."""
        self.db.conn.execute(
            "DELETE FROM live_positions WHERE telegram_id = ? AND ticker = ?",
            [self.telegram_id, ticker],
        )
        self.db.conn.execute(
            """INSERT INTO trade_history
               (telegram_id, ticker, action, price, date, pnl_percent)
               VALUES (?, ?, 'SELL', ?, ?, ?)""",
            [self.telegram_id, ticker, exec_price, date_str, pnl_pct],
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
        """Parameterized BUY: insert into live_positions and trade_history."""
        qty = int(self._alloc // exec_price)
        qty = max(100, (qty // 100) * 100)

        self.db.conn.execute(
            """INSERT INTO live_positions
               (telegram_id, ticker, quantity, entry_price, entry_date)
               VALUES (?, ?, ?, ?, ?)""",
            [self.telegram_id, ticker, qty, exec_price, date_str],
        )
        self.db.conn.execute(
            """INSERT INTO trade_history
               (telegram_id, ticker, action, price, date, pnl_percent)
               VALUES (?, ?, 'BUY', ?, ?, 0.0)""",
            [self.telegram_id, ticker, exec_price, date_str],
        )
        report_lines.append(
            f"🔵 SIGNAL EXECUTED: BOUGHT {qty:,} {ticker} @ {exec_price:,.0f} VND"
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