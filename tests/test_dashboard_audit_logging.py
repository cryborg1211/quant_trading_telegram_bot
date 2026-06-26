"""Audit-trail logging tests for the dashboard data-feed fix.

The post-mortem Audit tab (`run_post_mortem`) can only grade tickers that exist
in the `audit_log` table. Before this fix the dashboard wrote audit rows on a
rare path only (the Telegram-push button), and `/add` wrote NO audit row at all
— so the NET-PnL branch of the evaluator was dead for the local user.

These tests pin the revived contract:
    * `portfolio_add` writes an `audit_log` "add" row after a successful insert.
    * That audit write is BEST-EFFORT — a logging failure must never bubble up
      and fail the portfolio write itself.

`DuckDBEngine` is mocked at its source module so the lazy import inside
`portfolio_add` resolves to the patched class. No real DuckDB / parquet needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _mock_engine() -> MagicMock:
    """A DuckDBEngine stub whose duplicate-check COUNT returns 0 (no dupe)."""
    engine = MagicMock(name="DuckDBEngine_instance")
    engine.conn.execute.return_value.fetchone.return_value = (0,)
    return engine


def test_portfolio_add_logs_audit_row():
    """A successful /add writes an audit_log 'add' row with the upper ticker."""
    from dashboard.utils import headless

    engine = _mock_engine()
    with patch("src.data.db_engine.DuckDBEngine", return_value=engine):
        headless.portfolio_add("local", "hpg", 100, 25000.0)

    engine.log_user_action.assert_called_once()
    args, kwargs = engine.log_user_action.call_args
    assert args[0] == "local"
    assert args[1] == "add"
    assert args[2] == "HPG"  # ticker normalised to upper
    # details carries the parsed volume/price for backtest reconciliation.
    details = kwargs.get("details") or (args[3] if len(args) > 3 else "")
    assert "vol=100" in details
    assert "price=25000.0" in details


def test_portfolio_add_audit_failure_does_not_raise():
    """An audit-logging failure is swallowed — the /add still succeeds."""
    from dashboard.utils import headless

    engine = _mock_engine()
    engine.log_user_action.side_effect = RuntimeError("db down")
    with patch("src.data.db_engine.DuckDBEngine", return_value=engine):
        # Must NOT raise despite the audit write blowing up.
        headless.portfolio_add("local", "ssi", 50, 12000.0)

    # The portfolio INSERT was still attempted (COUNT check + INSERT >= 2 calls).
    assert engine.conn.execute.call_count >= 2
    engine.log_user_action.assert_called_once()
