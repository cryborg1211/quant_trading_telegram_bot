"""Dispatched-signal ledger — closes the tranche entry/exit loop on the serve path.

The backtest's tranche book holds every cohort exactly ``hold_days`` TRADING
sessions, then liquidates at ATC. The bot dispatches the entries but (before
this module) never alerted the exits, so live behavior silently diverged from
the simulated strategy after the first horizon elapsed.

Flow:
    1. ``record_dispatch``  — called by ``run_trade_execution`` after a broadcast
       dispatch; one OPEN row per (ticker, dispatch_date).
    2. ``check_exits_due``  — called daily by ``full_pipeline``; an OPEN row is
       due once ``hold_days`` trading sessions (per the fresh parquet calendar,
       NOT calendar days) have elapsed since dispatch.
    3. ``mark_closed``      — flips rows to CLOSED after the exit alert is sent.

Storage: ``dispatched_signals`` table in the core DuckDB file. All writes go
through short-lived connections (same convention as the sentiment crawler — no
mixed-config handles to the same file).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import duckdb  # type: ignore[import-untyped]

from config.settings import CONFIG
from src.data import price_lookup

LOGGER = logging.getLogger(__name__)

TABLE = "dispatched_signals"

# Cancelled-signal regret table (brand new — no legacy rows). Captures every
# ticker the daily screen REJECTED ("❌ HỦY BỎ TÍN HIỆU"): weak-market fallback
# or post-arbitration no-buy. Read back by the EOD regret report to show "what
# would have happened if bought anyway" — HYPOTHETICAL, never a recommendation.
CANCELLED_TABLE = "cancelled_signals"

# VN round-trip transaction cost (~0.15%/leg fee+tax, buy+sell) deducted from the
# gross price move so evaluate_signal_pnl reports a NET, cost-aware return. Kept
# in sync with the sibling constant audit_evaluator._VN_ROUND_TRIP_COST_PCT
# (intentionally duplicated — the two modules use different connection
# conventions, see the EOD position-report plan Decision 4).
_VN_ROUND_TRIP_COST_PCT: float = 0.30


def _connect(db_path: str | None) -> Any:
    return duckdb.connect(db_path or str(CONFIG.paths.duckdb_path))


def _opt_float(x: Any) -> float | None:
    """Coerce an optional numeric to float, preserving None (→ SQL NULL)."""
    return float(x) if x is not None else None


def ensure_table(conn: Any) -> None:
    """Create the ledger table, and add `is_paper` to a pre-existing one.

    `is_paper` marks rows that were RECONSTRUCTED rather than really dispatched
    (see scripts/backfill_dispatched_signals_from_paperlog.py). It exists because
    `open_tickers` feeds the open-cohort dedup veto in `_select_candidates`: an
    OPEN paper row would silently block a REAL dispatch of that ticker for weeks,
    which inverts the guard's purpose — it is there to stop re-buying something
    you already HOLD. Discriminating on `weight == 0.0` instead would be unsafe,
    since `record_dispatch` also writes 0.0 when `suggested_weight` is missing.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            ticker        VARCHAR NOT NULL,
            dispatch_date DATE    NOT NULL,
            horizon       INTEGER,
            hold_days     INTEGER NOT NULL,
            weight        DOUBLE,
            status        VARCHAR DEFAULT 'OPEN',
            closed_date   DATE,
            dispatched_at TIMESTAMP DEFAULT current_timestamp,
            is_paper      BOOLEAN DEFAULT FALSE
        )
        """
    )
    # Idempotent migration for ledgers created before is_paper existed. Their
    # rows are all real dispatches, so FALSE is the correct backfill value.
    try:
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{TABLE}'"
            ).fetchall()
        }
        if "is_paper" not in cols:
            conn.execute(
                f"ALTER TABLE {TABLE} ADD COLUMN is_paper BOOLEAN DEFAULT FALSE")
            conn.execute(f"UPDATE {TABLE} SET is_paper = FALSE WHERE is_paper IS NULL")
            LOGGER.info("[SignalLedger] added is_paper (existing rows = real dispatches).")
    except Exception:  # noqa: BLE001 — never block the dispatch path
        LOGGER.warning("[SignalLedger] is_paper migration skipped.", exc_info=True)


def record_dispatch(
    signals: list[dict],
    strategy: dict | None,
    horizon: int,
    db_path: str | None = None,
    today: date | None = None,
    is_paper: bool = False,
) -> int:
    """Record OPEN ledger rows for a broadcast tranche dispatch.

    No-op for legacy half-Kelly artifacts (no tranche strategy → no fixed
    exit horizon to track). Idempotent per (ticker, dispatch_date, horizon) so a
    pipeline re-run on the same day cannot double-book a cohort — the horizon is
    part of the dedup key so a T+5 tracking row and a T+20 tranche row for the
    same ticker+day both persist independently.

    `is_paper=True` marks the row as TRACKED-BUT-NOT-HELD, which keeps it out of
    `open_tickers()` and therefore out of the open-cohort dedup veto.

    WHY THAT MATTERS (27-08-26). `/suggest` is a preview command, yet it recorded
    real cohorts, so a second tap 17 minutes later contradicted the first:

        08:18  dedup skips GVR,VRE,PNJ  -> 1 survivor  -> VJC dispatched
        08:35  dedup skips GVR,VRE,VJC,PNJ -> 0 survivors -> "THỊ TRƯỜNG XẤU"

    The meta-gate rejected the identical 46 names both times; the model had not
    changed its mind. VJC simply moved from "candidate" to "already held" because
    the FIRST tap booked it. The second report then told the operator no name was
    good enough, while GVR 44.1% / VRE 43.0% / VJC 42.4% had all cleared the 0.42
    gate — they were excluded for being held, not for being weak. Left alone this
    also consumes the day's candidates before the 15:30 cron ever runs.

    Returns the number of rows actually inserted.
    """
    if not strategy or strategy.get("mode") != "tranche":
        return 0
    hold_days = int(strategy.get("hold_days") or 0)
    if hold_days <= 0 or not signals:
        return 0

    today = today or datetime.now().date()
    rows = [
        (str(s["ticker"]).upper(), today, int(horizon), hold_days,
         float(s.get("suggested_weight") or 0.0), bool(is_paper))
        for s in signals
        if s.get("ticker")
    ]
    if not rows:
        return 0

    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            existing = {
                (str(r[0]).upper(), int(r[1]) if r[1] is not None else None)
                for r in conn.execute(
                    f"SELECT ticker, horizon FROM {TABLE} WHERE dispatch_date = ?", [today]
                ).fetchall()
            }
            rows = [r for r in rows if (r[0], r[2]) not in existing]
            if rows:
                conn.executemany(
                    f"INSERT INTO {TABLE} "
                    "(ticker, dispatch_date, horizon, hold_days, weight, is_paper) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
    except Exception:  # noqa: BLE001 — ledger must never kill the dispatch path
        LOGGER.exception("[SignalLedger] record_dispatch failed.")
        return 0

    LOGGER.info("[SignalLedger] Recorded %s %s signals (hold=%s sessions).",
                len(rows), "TRACKED (paper, not held)" if is_paper else "OPEN",
                hold_days)
    return len(rows)


def list_open(db_path: str | None = None, today: date | None = None,
              include_paper: bool = False) -> list[dict]:
    """Every REAL OPEN signal with its elapsed trading-session count.

    Sessions are counted on the fresh parquet calendar (NOT calendar days),
    mirroring the backtest engine's day indexing. Sorted oldest-first.

    `is_paper` rows are EXCLUDED by default. They are reconstructions of what
    the model once suggested (see
    scripts/backfill_dispatched_signals_from_paperlog.py), not positions — and
    every caller of this function is operator-facing: the EOD position report,
    the `/exits` command, the dashboard GIU tab, and (via `check_exits_due`) the
    tranche EXIT ALERTS. Including them shipped a 118-line report of holdings
    the user never had, and would have gone on to fire SELL alerts for them.
    Pass `include_paper=True` only for research reads.
    """
    paper_pred = "" if include_paper else "AND COALESCE(is_paper, FALSE) = FALSE"
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            open_rows = conn.execute(
                f"SELECT ticker, dispatch_date, horizon, hold_days, weight "
                f"FROM {TABLE} WHERE status = 'OPEN' {paper_pred} "
                f"ORDER BY dispatch_date, ticker"
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] list_open read failed.")
        return []

    if not open_rows:
        return []

    min_dispatch = min(r[1] for r in open_rows)
    sessions = price_lookup.trading_dates_after(min_dispatch)
    today = today or datetime.now().date()

    out: list[dict] = []
    for ticker, d0, horizon, hold_days, weight in open_rows:
        elapsed = sum(1 for s in sessions if d0 < s <= today)
        out.append({
            "ticker": str(ticker),
            "dispatch_date": d0,
            "horizon": int(horizon) if horizon is not None else None,
            "hold_days": int(hold_days),
            "weight": float(weight or 0.0),
            "sessions_elapsed": elapsed,
            "sessions_remaining": max(0, int(hold_days) - elapsed),
        })
    return out


def open_tickers(db_path: str | None = None) -> set[str]:
    """Uppercased tickers with ANY REAL status='OPEN' row, across all horizons.

    Feeds the dispatch open-cohort dedup (July-2026 incident: BSR re-dispatched
    4 consecutive days into a falling knife). Cheap single-column read — no
    session counting. Never raises: an unreadable ledger returns an empty set
    (candidate selection must not die because dedup data is unavailable).

    `is_paper` rows are EXCLUDED. The guard's purpose is "do not re-buy a name
    you already hold", and a reconstructed row is not a holding — counting them
    would veto real dispatches (measured: 2 liquid names, PVD and VHM, would
    have been blocked for weeks by the 10-08-26 backfill).
    """
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            rows = conn.execute(
                f"SELECT DISTINCT ticker FROM {TABLE} "
                "WHERE status = 'OPEN' AND COALESCE(is_paper, FALSE) = FALSE"
            ).fetchall()
        return {str(r[0]).upper() for r in rows}
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] open_tickers read failed — dedup skipped.")
        return set()


def check_exits_due(db_path: str | None = None, today: date | None = None) -> list[dict]:
    """OPEN signals whose hold horizon has elapsed in TRADING sessions.

    A signal dispatched on D with hold_days=H is due once the fresh parquet
    calendar contains >= H trading dates strictly after D (mirrors the
    backtest engine, which exits the cohort at the close of session D+H).
    """
    return [r for r in list_open(db_path, today) if r["sessions_elapsed"] >= r["hold_days"]]


def mark_closed(
    due: list[dict],
    db_path: str | None = None,
    today: date | None = None,
) -> int:
    """Flip the given (ticker, dispatch_date) rows to CLOSED."""
    if not due:
        return 0
    today = today or datetime.now().date()
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            for d in due:
                # Horizon-scoped close: a due T+5 row and a still-open T+20 row can
                # coexist for the same (ticker, dispatch_date). Without the horizon
                # predicate, closing the due T+5 row would also flip the live T+20
                # row to CLOSED. `IS NOT DISTINCT FROM` is DuckDB's NULL-safe
                # equality (matches a legacy NULL-horizon row to a NULL parameter).
                conn.execute(
                    f"UPDATE {TABLE} SET status = 'CLOSED', closed_date = ? "
                    "WHERE ticker = ? AND dispatch_date = ? AND status = 'OPEN' "
                    "AND horizon IS NOT DISTINCT FROM ?",
                    [today, d["ticker"], d["dispatch_date"], d.get("horizon")],
                )
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] mark_closed failed.")
        return 0
    return len(due)


def _evaluate_pnl(
    ticker: str,
    entry_date: date,
    hold_days: int,
    today: date,
    cost_pct: float,
) -> dict:
    """Shared open/matured PnL% core for one signal row, ``cost_pct``-adjusted.

    Callers pass ALREADY-NORMALIZED ``ticker``/``today``/``hold_days``. Two thin
    public wrappers differ ONLY in ``cost_pct``:
      • ``evaluate_signal_pnl`` → ``_VN_ROUND_TRIP_COST_PCT`` (NET — a real
        tracked dispatch that pays a round-trip fee).
      • ``evaluate_regret_pnl`` → ``0.0`` (GROSS — a hypothetical "if bought"
        figure; nothing was traded, so no cost was ever paid).

    Branching (mirrors ``audit_evaluator._evaluate_dispatched_signal``):
      • t0 = close at-or-before ``entry_date``.
      • Once ``hold_days`` TRADING sessions have elapsed (per the fresh parquet
        calendar, counted <= ``today``), exit = the ``hold_days``-th session's
        close and ``matured=True``; otherwise exit = the latest close and
        ``matured=False`` (provisional, still inside the hold window).
      • pct = (t_exit - t0) / t0 * 100 - cost_pct.

    Corporate-action guard + AUTO-ADJUSTMENT (reuses the already-shipped
    ``price_lookup.closes_between`` / ``price_lookup.has_ca_gap`` pair — the SAME
    guard wired into ``main._backfill_paperlog_outcomes`` and
    ``portfolio_guard.evaluate_position``, default 10% threshold — plus the
    sibling ``price_lookup.derive_ca_adjustment_factor`` for the correction
    magnitude): when the close PATH across the return window contains a >10%
    single-session jump (HOSE caps a session at ±7%, so that means a split /
    stock dividend / rights issue, not price action) ``gap_flag=True`` and ``pct``
    is AUTO-ADJUSTED by rebasing T0 onto the post-event price scale
    (``t0_eff = t0 * adjustment_factor``), so the returned ``pct`` is the
    economically correct number rather than a phantom −40%-class move. Both the
    detection and the magnitude read the SAME ``closes`` list and SAME default
    threshold, so they can never disagree. ``gap_flag=True`` now means "this
    row's ``pct`` was auto-adjusted," NOT "unreliable, do not show." A hold window
    spanning multiple corporate actions composes their factors as a cumulative
    product. Defensive: a non-positive rebased basis (corrupted parquet) falls
    back to the raw t0.

    Never raises: returns ``{"ticker", "error"}`` on missing price history or a
    non-positive t0 (gap/adjustment logic skipped on the error path — nothing to
    compute), else ``{"ticker", "pct", "matured", "t0", "t_exit", "gap_flag",
    "adjustment_factor"}`` (``adjustment_factor`` is the applied factor when a gap
    fired, else ``None``). ``t0``/``t_exit`` stay the RAW, as-quoted closes for
    audit transparency (the caller can reconstruct ``t0 * adjustment_factor`` if
    it ever needs the rebased entry basis).
    """
    try:
        t0 = price_lookup.close_on_or_before(ticker, entry_date)
        sessions = [s for s in price_lookup.trading_dates_after(entry_date) if s <= today]
        if hold_days > 0 and len(sessions) >= hold_days:
            exit_date = sessions[hold_days - 1]
            t_exit = price_lookup.close_on_or_before(ticker, exit_date)
            matured = True
        else:
            exit_date = None
            t_exit = price_lookup.latest_close(ticker)
            matured = False
    except Exception as exc:  # noqa: BLE001 — evaluation must never kill the report
        LOGGER.warning("[SignalLedger] _evaluate_pnl(%s) price fetch failed: %s",
                       ticker, exc)
        return {"ticker": ticker, "error": f"price lookup: {exc}"}

    if t0 is None or t_exit is None:
        return {"ticker": ticker, "error": "thiếu giá lịch sử trong DB"}
    if t0 <= 0:
        return {"ticker": ticker, "error": f"giá T0 không hợp lệ ({t0})"}

    # Corporate-action gap guard + auto-adjustment — reuse existing gap-detection
    # infra (has_ca_gap), plus derive_ca_adjustment_factor for the correction
    # magnitude. Both are called against the SAME `closes` list and the SAME
    # default 10% threshold so gap_flag and adjustment_factor can never disagree
    # about which sessions counted as a corporate-action reset (plan Decision 2/3).
    window_end = exit_date if matured else today
    closes = price_lookup.closes_between(ticker, entry_date, window_end)
    gap_flag = price_lookup.has_ca_gap(closes)
    adjustment_factor = (
        price_lookup.derive_ca_adjustment_factor(closes) if gap_flag else None
    )

    # Rebase T0 onto the post-event price scale (standard back-adjustment
    # technique — see the plan's Architecture Decision 2 for the full derivation
    # + worked example). Defensive fallback to the raw t0 if the rebased value is
    # non-positive (only reachable with corrupted parquet data).
    t0_eff = t0
    if gap_flag and adjustment_factor and t0 * adjustment_factor > 0:
        t0_eff = t0 * adjustment_factor
    pct = (t_exit - t0_eff) / t0_eff * 100.0 - cost_pct
    return {"ticker": ticker, "pct": pct, "matured": matured,
            "t0": t0, "t_exit": t_exit, "gap_flag": gap_flag,
            "adjustment_factor": adjustment_factor}


def evaluate_signal_pnl(
    ticker: str,
    dispatch_date: date,
    hold_days: int,
    today: date | None = None,
) -> dict:
    """NET-of-round-trip-cost PnL% for one ledger row, open or matured.

    Thin NET wrapper over the shared ``_evaluate_pnl`` core (``cost_pct =
    _VN_ROUND_TRIP_COST_PCT``). See ``_evaluate_pnl`` for the full open/matured
    branching + corporate-action gap guard (intentionally duplicated from
    ``audit_evaluator._evaluate_dispatched_signal`` per the EOD position-report
    plan Decision 4 — this module uses its own default price_lookup connection).

    Signature UNCHANGED. The success-path return dict now ALWAYS carries a
    ``gap_flag: bool`` key (True when the close path across the return window
    contains a >10% single-session jump — a corporate action, not price action)
    and a companion ``adjustment_factor: float | None`` key. When ``gap_flag`` is
    True the ``pct`` is AUTO-ADJUSTED: T0 is rebased by the observed gap ratio so
    the number is the economically correct return, not a phantom. Motivating live
    bug: PVD 2026-07-09→07-10 fell 33.3→19.47 VND overnight on a 66.9%
    stock-dividend ex-rights event; the pre-retrofit code booked that as a fake
    −41.3% "loss", and the interim "detect and hide" code hid it behind a warning.
    This retrofit AUTOMATICALLY CORRECTS it to a NET pct ≈ −0.30% (the round-trip
    cost only — the underlying move nets to flat once the stock-dividend shares
    are accounted for), with ``adjustment_factor ≈ 0.585``. ``adjustment_factor``
    is ``None`` on every non-gap row.

    Never raises: returns ``{"ticker", "error"}`` on missing price history or a
    non-positive t0, else ``{"ticker", "pct", "matured", "t0", "t_exit",
    "gap_flag", "adjustment_factor"}``.
    """
    ticker = str(ticker).upper().strip()
    today = today or datetime.now().date()
    hold_days = int(hold_days or 0)
    return _evaluate_pnl(ticker, dispatch_date, hold_days, today, _VN_ROUND_TRIP_COST_PCT)


def list_closed_since(
    lookback_days: int, today: date | None = None, db_path: str | None = None,
    include_paper: bool = False,
) -> list[dict]:
    """Every REAL signal CLOSED within the trailing ``lookback_days``-day window.

    `is_paper` rows are EXCLUDED by default, for the same reason as in
    ``list_open``: the EOD position report is operator-facing and reconstructed
    suggestions are not closed positions.

    Window is inclusive on both ends: ``(today - lookback_days) <= closed_date
    <= today``. Widened from the original exact-``today`` filter so recently
    closed suggestions (incl. historically backfilled rows) surface for a few
    days rather than only on their close date.

    Same dict shape as ``list_open()`` rows minus ``sessions_elapsed`` /
    ``sessions_remaining`` (meaningless for a closed row). Sorted oldest-first
    then by ticker. Never raises — a read failure degrades to ``[]``.
    """
    if today is None:
        today = datetime.now().date()
    start = today - timedelta(days=lookback_days)
    paper_pred = "" if include_paper else "AND COALESCE(is_paper, FALSE) = FALSE"
    try:
        with _connect(db_path) as conn:
            ensure_table(conn)
            rows = conn.execute(
                f"SELECT ticker, dispatch_date, horizon, hold_days, weight "
                f"FROM {TABLE} WHERE status = 'CLOSED' {paper_pred} "
                "AND closed_date >= ? AND closed_date <= ? "
                "ORDER BY dispatch_date, ticker",
                [start, today],
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] list_closed_since read failed.")
        return []

    return [
        {
            "ticker": str(r[0]),
            "dispatch_date": r[1],
            "horizon": int(r[2]) if r[2] is not None else None,
            "hold_days": int(r[3]),
            "weight": float(r[4] or 0.0),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Cancelled-signal regret ledger (brand-new table + read/eval surface)
# ---------------------------------------------------------------------------

def ensure_cancelled_table(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CANCELLED_TABLE} (
            screen_date  DATE      NOT NULL,
            ticker       VARCHAR   NOT NULL,
            horizon      INTEGER   NOT NULL,
            hold_days    INTEGER   NOT NULL,
            p_down       DOUBLE,
            p_side       DOUBLE,
            p_up         DOUBLE,
            reason       VARCHAR,
            screened_at  TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def record_cancelled(
    candidates: list[dict],
    horizon: int,
    hold_days: int,
    db_path: str | None = None,
    today: date | None = None,
) -> int:
    """Record one cancelled-signal row per rejected candidate. Never raises.

    Called from BOTH daily_inference capture points (weak-market fallback +
    post-arbitration no-buy monitor), regardless of broadcast/persist — so manual
    /suggest_buy taps are captured too (mirrors the suggest_buy_ledger_tracking
    precedent's gating philosophy, Decision 5).

    No ``mode == tranche`` gate (unlike record_dispatch — a cancelled signal has
    no artifact-strategy concept; ``hold_days`` is passed directly by the caller
    via main._cancelled_hold_days). Idempotent per (ticker, horizon) scoped to
    ``screen_date = today`` — a repeated same-day tap of the same rejected ticker
    is a no-op (Decision 6, matching dispatched_signals' own dedup convention).

    Each ``candidates[i]`` dict shape: {"ticker", "p_down", "p_side", "p_up",
    "reason"}. Returns the number of rows actually inserted.
    """
    if not candidates:
        return 0
    today = today or datetime.now().date()
    horizon = int(horizon)
    hold_days = int(hold_days)
    rows = [
        (today, str(c["ticker"]).upper(), horizon, hold_days,
         _opt_float(c.get("p_down")), _opt_float(c.get("p_side")),
         _opt_float(c.get("p_up")), str(c.get("reason") or ""))
        for c in candidates
        if c.get("ticker")
    ]
    if not rows:
        return 0

    try:
        with _connect(db_path) as conn:
            ensure_cancelled_table(conn)
            existing = {
                (str(r[0]).upper(), int(r[1]))
                for r in conn.execute(
                    f"SELECT ticker, horizon FROM {CANCELLED_TABLE} "
                    "WHERE screen_date = ?", [today]
                ).fetchall()
            }
            rows = [r for r in rows if (r[1], r[2]) not in existing]
            if rows:
                conn.executemany(
                    f"INSERT INTO {CANCELLED_TABLE} "
                    "(screen_date, ticker, horizon, hold_days, p_down, p_side, "
                    "p_up, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
    except Exception:  # noqa: BLE001 — capture must never kill daily_inference
        LOGGER.exception("[SignalLedger] record_cancelled failed.")
        return 0

    LOGGER.info("[SignalLedger] Recorded %s cancelled signals (horizon=%s, hold=%s).",
                len(rows), horizon, hold_days)
    return len(rows)


def list_cancelled_since(
    lookback_days: int, today: date | None = None, db_path: str | None = None
) -> list[dict]:
    """Every cancelled signal within the trailing ``lookback_days``-day window.

    Window inclusive on both ends: (today - lookback_days) <= screen_date <=
    today (same pattern as list_closed_since). Each row also carries
    ``sessions_elapsed`` / ``sessions_remaining`` computed the SAME way as
    list_open (fresh parquet trading calendar, NOT calendar days), so the regret
    report can partition still-tracking vs matured. Sorted by screen_date then
    ticker. Never raises — a read failure degrades to [].
    """
    if today is None:
        today = datetime.now().date()
    start = today - timedelta(days=lookback_days)
    try:
        with _connect(db_path) as conn:
            ensure_cancelled_table(conn)
            rows = conn.execute(
                f"SELECT screen_date, ticker, horizon, hold_days, "
                "p_down, p_side, p_up, reason "
                f"FROM {CANCELLED_TABLE} "
                "WHERE screen_date >= ? AND screen_date <= ? "
                "ORDER BY screen_date, ticker",
                [start, today],
            ).fetchall()
    except Exception:  # noqa: BLE001
        LOGGER.exception("[SignalLedger] list_cancelled_since read failed.")
        return []

    if not rows:
        return []

    min_screen = min(r[0] for r in rows)
    sessions = price_lookup.trading_dates_after(min_screen)

    out: list[dict] = []
    for screen_date, ticker, horizon, hold_days, p_down, p_side, p_up, reason in rows:
        elapsed = sum(1 for s in sessions if screen_date < s <= today)
        out.append({
            "screen_date": screen_date,
            "ticker": str(ticker),
            "horizon": int(horizon),
            "hold_days": int(hold_days),
            "p_down": _opt_float(p_down),
            "p_side": _opt_float(p_side),
            "p_up": _opt_float(p_up),
            "reason": str(reason) if reason is not None else "",
            "sessions_elapsed": elapsed,
            "sessions_remaining": max(0, int(hold_days) - elapsed),
        })
    return out


def evaluate_regret_pnl(
    ticker: str,
    screen_date: date,
    hold_days: int,
    today: date | None = None,
) -> dict:
    """GROSS (cost-free) hypothetical PnL% for one cancelled signal, open/matured.

    Thin GROSS wrapper over the shared ``_evaluate_pnl`` core with
    ``cost_pct=0.0`` — a DELIBERATE deviation from evaluate_signal_pnl's NET
    convention (Decision 3): nothing was actually traded — no round-trip
    transaction cost is deducted; the caller/report must render this as a gross,
    hypothetical figure ("gộp, chưa trừ phí"), never a literal executable net
    return.

    Shares _evaluate_pnl's corporate-action gap guard + auto-adjustment, so a
    cancelled ticker that later had a split / stock dividend has its GROSS ``pct``
    auto-adjusted (T0 rebased by the observed gap ratio) exactly the same way a
    tracked dispatch is — the returned number is trusted, not a phantom.
    ``gap_flag=True`` means "auto-adjusted," and ``adjustment_factor`` carries the
    applied factor (``None`` when no gap fired). Never raises: returns
    {"ticker", "error"} on failure, else {"ticker", "pct", "matured", "t0",
    "t_exit", "gap_flag", "adjustment_factor"}.
    """
    ticker = str(ticker).upper().strip()
    today = today or datetime.now().date()
    hold_days = int(hold_days or 0)
    return _evaluate_pnl(ticker, screen_date, hold_days, today, 0.0)
