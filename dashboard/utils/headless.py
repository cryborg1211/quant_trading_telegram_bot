"""Headless inference + portfolio wrappers (P2).

Thin wrappers that call the serve-path functions in ``main.py`` / ``src`` with
``broadcast=False`` and ``persist=False`` so the dashboard renders real signals
WITHOUT triggering Telegram sends or mutating the cron portfolio book (P0
GOTCHA 1). Portfolio add/remove/list go straight to the ``portfolio`` DuckDB
table (there is no add/remove API on ``PortfolioManager`` — the bot does the
same raw SQL).

IMPORTANT — lazy imports: every heavy serve module (``main``, ``src.*``) is
imported INSIDE the function body, never at module top. This keeps the dashboard
package importable (for ``py_compile`` / import-checks) without the full ML
stack, defers cold-start cost, and avoids circular imports.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

# Repo root = three levels up from this file (dashboard/utils/headless.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_JSON_PATH = _REPO_ROOT / "config" / "settings.json"

# Serialises all portfolio writes within the dashboard process. DuckDB's own WAL
# handles cross-process writers (cron); this lock only orders dashboard threads.
_audit_lock = threading.Lock()

# Default local user namespace. Kept separate from cron ("cron") and the bot
# (Telegram numeric id) so dashboard portfolio + audit rows never collide.
_DEFAULT_USER_ID = "local"


def _read_user_id() -> str:
    """Return the dashboard user_id from settings.json, else ``"local"``.

    Reads the optional ``dashboard_user_id`` key (top-level or under
    ``trading``). Any read/parse failure or missing key falls back silently to
    the default — a new install must never crash on this.
    """
    try:
        if not _SETTINGS_JSON_PATH.exists():
            return _DEFAULT_USER_ID
        data = json.loads(_SETTINGS_JSON_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _DEFAULT_USER_ID
        # Accept either a top-level key or a trading-scoped override.
        candidate = data.get("dashboard_user_id")
        if candidate is None and isinstance(data.get("trading"), dict):
            candidate = data["trading"].get("dashboard_user_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return _DEFAULT_USER_ID
    except (json.JSONDecodeError, OSError, ValueError):
        return _DEFAULT_USER_ID


# Resolved once at import; Settings-tab edits require a dashboard restart to
# apply (CONFIG and this constant are both built once — consistent with P0
# GOTCHA 3 / the Settings-tab restart caption).
LOCAL_USER_ID: str = _read_user_id()


def _pnl_ratio(entry_price: float, current_close: float | None) -> float | None:
    """Fractional PnL ``(current - entry) / entry``; ``None`` if no close.

    Applies the VN price-scale convention: parquet OHLCV is stored in thousands
    of VND. The ``portfolio`` table stores absolute-VND entry prices (the bot's
    /add takes whole-VND prices), so a ``current_close`` below 1000 is treated
    as thousands-VND and scaled up before the ratio is computed.
    """
    if current_close is None:
        return None
    if current_close < 1000:
        current_close = current_close * 1000
    if entry_price == 0:
        return None
    return (current_close - entry_price) / entry_price


# --------------------------------------------------------------------------- #
# Inference wrappers (preview-safe — broadcast=False, persist=False)
# --------------------------------------------------------------------------- #

def daily_inference_headless(horizon: int) -> tuple[str, list[dict]]:
    """Read-only buy-signal inference for the MUA tab.

    Returns ``(report_html, dispatched_signals)``. ``persist=False`` skips every
    DuckDB write (portfolio / RL / paperlog); ``broadcast=False`` suppresses the
    Telegram push. The caller still gets the full structured signal list.
    """
    from main import daily_inference  # noqa: PLC0415 — lazy heavy import

    return daily_inference(broadcast=False, persist=False, horizon=int(horizon))


def inference_for_holdings_headless(tickers: list[str]) -> str:
    """Sell/hold + rebalance inference for the BÁN tab.

    Wraps ``main.inference_for_holdings`` (HEADLESS-OK — no DB write, no
    Telegram). Returns the SELL/HOLD report HTML (empty string if no holdings).
    """
    from main import inference_for_holdings  # noqa: PLC0415 — lazy heavy import

    if not tickers:
        return ""
    return inference_for_holdings(tickers, window_rows=120)


def verify_single_ticker_headless(ticker: str) -> str:
    """Single-ticker dual-horizon check for the Verify tab.

    Wraps ``main.verify_single_ticker`` (HEADLESS-OK — only side-effect is an
    optional paperlog write that is itself try/except-wrapped). Returns the
    result HTML.
    """
    from main import verify_single_ticker  # noqa: PLC0415 — lazy heavy import

    return verify_single_ticker(ticker, window_rows=120)


# --------------------------------------------------------------------------- #
# Portfolio raw-SQL helpers (mirror the bot's /add, /remove, list)
# --------------------------------------------------------------------------- #

def _parse_price(value: object) -> float:
    """Coerce a stored price to float, tolerating legacy formatted strings.

    The ``portfolio`` table is shared with the bot era, where some rows stored
    the entry price as display TEXT (e.g. ``'47,800 VND'`` / ``'47.800 ₫'``)
    rather than a number. Strip thousands separators + any currency/letters so
    both clean floats and formatted strings parse to a usable number; returns
    0.0 on anything unparseable so a single bad row never crashes the GIỮ tab.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    import re  # noqa: PLC0415 — local; stays out of the hot import path

    # Drop thousands commas first, then keep only digits / dot / minus.
    cleaned = re.sub(r"[^0-9.\-]", "", str(value).replace(",", ""))
    if cleaned in ("", "-", ".", "-."):
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def portfolio_list(user_id: str) -> list[dict]:
    """Return this user's holdings as row dicts (empty list on any failure)."""
    try:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415 — lazy heavy import

        rows = DuckDBEngine().conn.execute(
            "SELECT ticker, volume, price, added_at FROM portfolio WHERE user_id = ? "
            "ORDER BY added_at",
            [user_id],
        ).fetchall()
    except Exception:  # noqa: BLE001 — degrade to empty list, never crash a tab
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).warning(
            "portfolio_list(%s) failed", user_id, exc_info=True
        )
        return []
    return [
        {"ticker": r[0], "volume": r[1], "price": _parse_price(r[2]), "added_at": r[3]}
        for r in rows
    ]


def portfolio_add(user_id: str, ticker: str, volume: int, price: float) -> None:
    """Insert one holding row for this user (raises ValueError on duplicate)."""
    from datetime import datetime  # noqa: PLC0415
    from src.data.db_engine import DuckDBEngine  # noqa: PLC0415 — lazy heavy import

    ticker = ticker.strip().upper()
    with _audit_lock:
        conn = DuckDBEngine().conn
        # No DB-level UNIQUE constraint exists on (user_id, ticker) — mirror the
        # bot's table — so enforce uniqueness explicitly to match the plan
        # contract (one row per ticker per user from the dashboard).
        existing = conn.execute(
            "SELECT COUNT(*) FROM portfolio WHERE user_id = ? AND ticker = ?",
            [user_id, ticker],
        ).fetchone()
        if existing and int(existing[0]) > 0:
            raise ValueError(f"{ticker} đã có trong danh mục.")
        conn.execute(
            "INSERT INTO portfolio (user_id, ticker, volume, price, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [user_id, ticker, int(volume), float(price), datetime.now()],
        )


def portfolio_remove(user_id: str, ticker: str) -> None:
    """Delete every row matching (user_id, ticker) for this user."""
    from src.data.db_engine import DuckDBEngine  # noqa: PLC0415 — lazy heavy import

    ticker = ticker.strip().upper()
    with _audit_lock:
        DuckDBEngine().conn.execute(
            "DELETE FROM portfolio WHERE user_id = ? AND ticker = ?",
            [user_id, ticker],
        )
