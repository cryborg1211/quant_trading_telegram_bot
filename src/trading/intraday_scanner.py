"""Intraday attack scanner — MONITORING-ONLY provisional-bar rescore.

Rescores the live model on PROVISIONAL intraday bars (spliced in memory from an
SSI iBoard snapshot) during HOSE trading hours, and emits a compact "top movers"
card when something notable moves before the 15:30 EOD crawl. This gives the
operator earlier visibility so they can *manually* act before the close — it
does NOT alter the automated system's trading behavior in any way.

HARD CONTRACT (do not relax without returning to PLAN)
──────────────────────────────────────────────────────
  • NO PERSISTENCE. This module writes NOTHING to `data/ohlcv_*.parquet`,
    `data/quant_v6_core.duckdb` (sentiment_entry_paperlog / portfolio /
    trade_history / signal_ledger / audit tables), or any other on-disk store.
    The provisional-bar splice is in-memory only, per scan.
  • NO ARBITRATOR / NO GEMINI. The scan loop never invokes the sentiment
    arbitrator or any Gemini call. It is a pure price-model rescore + alert.
  • NO TRADE SIGNAL. Every card carries a visible "provisional intraday — NOT a
    trade signal" tag and contains no BUY wording.
  • MONITORING-ONLY STATE. The only mutable state this module allows is the
    in-memory last-scan `ScannerState`, reset daily. No new DuckDB tables, no
    new parquet writes.

PURITY LAYERING (mirrors serve_universe.py / regime_policy.py precedent)
───────────────────────────────────────────────────────────────────────
  • The transform/detect/format functions (`snapshot_row_to_provisional_bar`,
    `splice_provisional_bar`, `splice_all`, `is_trading_window`, `detect_events`,
    `build_scan_card`) are PURE: no I/O, no CONFIG import, deterministic on their
    inputs (`is_trading_window` takes `now` as a parameter, never calls
    `datetime.now()` itself).
  • `fetch_snapshot` and `run_scan` are the I/O-bearing orchestration layer.
    They deliberately import NO `telegram`/`asyncio` — the bot module owns those
    and calls `run_scan` off the event loop via `asyncio.to_thread`, so this
    module stays independently unit-testable without a live PTB Application.

SSI PRICE-SCALE CONVENTION
──────────────────────────
SSI iBoard reports absolute VND. This repo's on-disk parquet convention is
THOUSANDS of VND (see vn_price_scale_convention). Every price field
(open/high/low/close) is divided by `price_unit_divisor` (default 1000.0) so a
spliced provisional bar is dimensionally consistent with the OHLCV tail it is
appended to. Volume (`nmTotalTradedQty`) is passed through unscaled — the
parquet stores raw share counts.
"""
from __future__ import annotations

import gc
import logging
import sys
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

LOGGER = logging.getLogger("quant.intraday_scanner")

# HOSE trading sessions (ICT): morning 09:15–11:30, afternoon 13:00–14:45. The
# lunch gap (11:30–13:00) and everything outside the two windows is a no-op.
_MORNING_START = (9, 15)
_MORNING_END = (11, 30)
_AFTERNOON_START = (13, 0)
_AFTERNOON_END = (14, 45)

# The two horizons the scanner rescores: T+20 primary, T+5 short.
HORIZONS: tuple[int, ...] = (20, 5)

# Output column contract of Alpha360Generator.load_live_ohlcv_window — the
# schema every spliced frame must preserve.
_OHLCV_COLUMNS: tuple[str, ...] = ("ticker", "date", "open", "high", "low", "close", "volume")


@dataclass
class ScannerState:
    """In-memory, single-process last-scan state. Reset daily.

    `previous`: {horizon: {ticker: p_up}} from the prior in-window scan.
    `prev_top3`: {horizon: [ticker, ...]} top-3 in-universe by P(UP) prior scan.
    `session_date`: the calendar date the current `previous`/`prev_top3` belong
        to; when a scan observes a newer date, the state is treated as a fresh
        baseline (decision 4: first scan of the day). None = never scanned.

    NOT persisted anywhere. A bot restart mid-session loses this, which the plan
    explicitly accepts (next scan = fresh baseline "session opened" card).
    """

    previous: dict[int, dict[str, float]] | None = None
    prev_top3: dict[int, list[str]] | None = None
    session_date: date_cls | None = None
    last_card: str | None = field(default=None)


def is_trading_window(now: datetime, tz: str = "Asia/Ho_Chi_Minh") -> bool:
    """True iff `now` falls inside a HOSE trading session (Mon–Fri, ICT).

    Pure: takes `now` as a parameter and never calls `datetime.now()`. A naive
    `now` is interpreted in `tz`; an aware `now` is converted to `tz` first, so
    the caller can pass either a local or a UTC-aware clock safely.

    Sessions (ICT): morning 09:15–11:30 (inclusive of both bounds), afternoon
    13:00–14:45 (inclusive of both bounds). The 11:30–13:00 lunch gap, all
    pre-09:15 / post-14:45 time, and Saturday/Sunday return False.
    """
    zone = ZoneInfo(tz)
    local = now.astimezone(zone) if now.tzinfo is not None else now.replace(tzinfo=zone)
    if local.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    minutes = local.hour * 60 + local.minute
    in_morning = _to_min(_MORNING_START) <= minutes <= _to_min(_MORNING_END)
    in_afternoon = _to_min(_AFTERNOON_START) <= minutes <= _to_min(_AFTERNOON_END)
    return in_morning or in_afternoon


def _to_min(hm: tuple[int, int]) -> int:
    return hm[0] * 60 + hm[1]


def snapshot_row_to_provisional_bar(
    item: dict[str, Any], price_unit_divisor: float = 1000.0
) -> dict[str, Any] | None:
    """Map one SSI snapshot row to a provisional OHLCV bar dict, or None.

    Returns `{ticker, date, open, high, low, close, volume}` with the four price
    fields divided by `price_unit_divisor` (absolute VND → thousands of VND) and
    volume passed through unscaled. `date` is parsed from the snapshot's own
    `tradingDate` (YYYYMMDD string), never wall-clock — mirrors
    foreign_flow_crawler's "trust the exchange's own session date" convention.

    Returns None (never raises) for a malformed row — missing `stockSymbol`, a
    `tradingDate` that is not exactly 8 digits, or a price/volume field that
    cannot be coerced to float. This mirrors crawl_today's per-row
    skip-not-fail pattern: one bad row must not sink the whole snapshot.
    """
    symbol = item.get("stockSymbol")
    if not symbol:
        return None
    trading_date_raw = str(item.get("tradingDate") or "")
    if len(trading_date_raw) != 8 or not trading_date_raw.isdigit():
        return None
    try:
        row_date = date_cls(
            int(trading_date_raw[0:4]),
            int(trading_date_raw[4:6]),
            int(trading_date_raw[6:8]),
        )
    except ValueError:
        return None  # e.g. month=13 / day=32 — malformed, skip this row only

    try:
        open_ = _scaled(item.get("openPrice"), price_unit_divisor)
        high = _scaled(item.get("highest"), price_unit_divisor)
        low = _scaled(item.get("lowest"), price_unit_divisor)
        close = _scaled(item.get("matchedPrice"), price_unit_divisor)
        volume = float(item.get("nmTotalTradedQty") or 0.0)
    except (TypeError, ValueError):
        return None

    if close is None:  # a bar with no matched price is unusable for a rescore
        return None

    return {
        "ticker": str(symbol).upper(),
        "date": row_date,
        "open": open_ if open_ is not None else close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": volume,
    }


def _scaled(value: Any, divisor: float) -> float | None:
    """Coerce `value` to float and divide by `divisor`; None passes through."""
    if value is None:
        return None
    return float(value) / divisor


def splice_provisional_bar(tail_pl: pl.DataFrame, provisional: dict[str, Any]) -> pl.DataFrame:
    """Replace-if-same-date-else-append ONE ticker's provisional bar (pure).

    Never mutates `tail_pl` — returns a NEW frame. Preserves the
    `_OHLCV_COLUMNS` schema/dtypes that `load_live_ohlcv_window` produces:
      • REPLACE today's row when the EOD crawl for that date already landed in
        the tail (tail's max date == provisional date).
      • APPEND otherwise (today's session hasn't settled into parquet yet).

    The provisional dict is coerced to the tail's own column dtypes before the
    concat so `date` stays `pl.Date` and prices stay the tail's float dtype.
    """
    prov_row = _provisional_to_frame(provisional, tail_pl.schema)
    if tail_pl.is_empty():
        return prov_row

    prov_date = provisional["date"]
    max_date = tail_pl.get_column("date").max()
    if max_date == prov_date:
        # Drop the existing same-date row(s), then append the provisional one.
        kept = tail_pl.filter(pl.col("date") != prov_date)
        return pl.concat([kept, prov_row], how="vertical").sort(["ticker", "date"])
    return pl.concat([tail_pl, prov_row], how="vertical").sort(["ticker", "date"])


def _provisional_to_frame(provisional: dict[str, Any], schema: dict[str, Any]) -> pl.DataFrame:
    """Build a single-row frame from a provisional dict, cast to `schema`."""
    row = {c: provisional[c] for c in _OHLCV_COLUMNS}
    frame = pl.DataFrame([row])
    # Cast each column to the tail's own dtype so the vertical concat is exact
    # (date -> pl.Date, prices/volume -> whatever the parquet tail carries).
    casts = [pl.col(c).cast(schema[c]) for c in _OHLCV_COLUMNS if c in schema]
    return frame.with_columns(casts).select(list(_OHLCV_COLUMNS))


def splice_all(tails_pl: pl.DataFrame, snapshot_items: list[dict[str, Any]]) -> pl.DataFrame:
    """Splice each ticker's provisional bar onto its tail (pure, cross-section).

    Tickers WITHOUT a matching snapshot row pass through untouched (not dropped).
    Tickers with a snapshot row get `splice_provisional_bar` applied to their
    own tail. Malformed snapshot rows (None from
    `snapshot_row_to_provisional_bar`) are skipped. Never mutates the input.
    """
    if tails_pl.is_empty():
        return tails_pl

    provisionals: dict[str, dict[str, Any]] = {}
    for item in snapshot_items:
        bar = snapshot_row_to_provisional_bar(item)
        if bar is not None:
            provisionals[bar["ticker"]] = bar  # last snapshot row per ticker wins

    tail_tickers = tails_pl.get_column("ticker").unique().to_list()
    out_frames: list[pl.DataFrame] = []
    for ticker in tail_tickers:
        one_tail = tails_pl.filter(pl.col("ticker") == ticker)
        bar = provisionals.get(str(ticker))
        if bar is None:
            out_frames.append(one_tail)  # pass-through, no snapshot match
        else:
            out_frames.append(splice_provisional_bar(one_tail, bar))

    return pl.concat(out_frames, how="vertical").sort(["ticker", "date"])


def _top3(scores: dict[str, float]) -> list[str]:
    """Top-3 tickers by P(UP) desc, ties broken alphabetically (deterministic)."""
    return [
        t for t, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]


def detect_events(
    current: dict[int, dict[str, float]],
    previous: dict[int, dict[str, float]] | None,
    thresholds: dict[int, float],
    delta_pp: float,
    top3_by_horizon: dict[int, list[str]],
    prev_top3_by_horizon: dict[int, list[str]] | None,
) -> list[dict[str, Any]]:
    """Return the list of alert events for this scan (pure).

    Event kinds (per Frozen Design Decision 4):
      • "baseline"        — `previous is None`: first scan of the day. Returns a
        SINGLE baseline marker (caller sends the opening card) rather than
        flooding a "new_entrant" for every current top-3 name.
      • "new_entrant"     — a ticker newly in the in-universe top-3 for a horizon
        vs the previous scan.
      • "threshold_cross" — a ticker's P(UP) crossed its horizon's τ in either
        direction since the previous scan.
      • "delta_move"      — |ΔP(UP)| >= `delta_pp` vs the previous scan for a
        CURRENT top-3 name.

    Each event dict: {"ticker", "horizon", "kind", "p_up", "delta"}. When
    nothing crosses any threshold, returns []. `current`/`previous` are
    {horizon: {ticker: p_up}} maps already restricted to the in-universe set.
    """
    if previous is None:
        return [{"ticker": None, "horizon": None, "kind": "baseline", "p_up": None, "delta": 0.0}]

    events: list[dict[str, Any]] = []
    prev_top3 = prev_top3_by_horizon or {}

    for horizon, cur_scores in current.items():
        tau = thresholds.get(horizon)
        prev_scores = previous.get(horizon, {})
        cur_top3 = top3_by_horizon.get(horizon, [])
        prev_top3_h = prev_top3.get(horizon, [])

        # (a) new entrant into the in-universe top-3 vs the previous scan.
        for ticker in cur_top3:
            if ticker not in prev_top3_h:
                events.append({
                    "ticker": ticker, "horizon": horizon, "kind": "new_entrant",
                    "p_up": cur_scores.get(ticker), "delta": _delta(cur_scores, prev_scores, ticker),
                })

        # (b) τ cross in either direction since the previous scan.
        if tau is not None:
            for ticker, p_now in cur_scores.items():
                p_prev = prev_scores.get(ticker)
                if p_prev is None:
                    continue
                crossed_up = p_prev < tau <= p_now
                crossed_down = p_now < tau <= p_prev
                if crossed_up or crossed_down:
                    events.append({
                        "ticker": ticker, "horizon": horizon, "kind": "threshold_cross",
                        "p_up": p_now, "delta": p_now - p_prev,
                    })

        # (c) |ΔP(UP)| >= delta_pp for a CURRENT top-3 name.
        for ticker in cur_top3:
            p_now = cur_scores.get(ticker)
            p_prev = prev_scores.get(ticker)
            if p_now is None or p_prev is None:
                continue
            if abs(p_now - p_prev) >= delta_pp:
                events.append({
                    "ticker": ticker, "horizon": horizon, "kind": "delta_move",
                    "p_up": p_now, "delta": p_now - p_prev,
                })

    return events


def _delta(cur: dict[str, float], prev: dict[str, float], ticker: str) -> float:
    """P(UP) change vs previous scan, 0.0 when the ticker had no prior score."""
    p_now = cur.get(ticker)
    p_prev = prev.get(ticker)
    if p_now is None or p_prev is None:
        return 0.0
    return p_now - p_prev


# Mandatory provisional tag (no BUY wording anywhere in the card).
_PROVISIONAL_TAG = "⚠️ TẠM THỜI TRONG PHIÊN — KHÔNG PHẢI TÍN HIỆU GIAO DỊCH"
_HORIZON_LABEL = {20: "T+20", 5: "T+5"}


def build_scan_card(
    events: list[dict[str, Any]],
    scores_now: dict[int, dict[str, float]],
    scores_prev_eod: dict[int, dict[str, float]] | None,
    prices: dict[str, dict[str, float]],
) -> str:
    """Render the compact HTML scan card (pure).

    Always carries the `_PROVISIONAL_TAG`. Per horizon (T+20 then T+5), lists
    top-3 in-universe names by P(UP): ticker, P(UP) now, Δ vs previous scan,
    Δ vs yesterday's EOD (if available), last price, % vs refPrice. Contains no
    BUY wording. Kept well under the 4096-char Telegram limit (3 names × 2
    horizons is short); if it ever exceeds a defensive 3500-char cap it is
    truncated with an ellipsis marker rather than risking a Telegram reject.

    `events` drives the header line only (which kinds fired). `scores_now` is
    {horizon: {ticker: p_up}}; `scores_prev_eod` is the same shape from the
    prior EOD run (or None). `prices` is {ticker: {"last": float, "pct": float}}.
    """
    is_baseline = any(e.get("kind") == "baseline" for e in events)
    lines: list[str] = []
    if is_baseline:
        lines.append("📡 <b>PHIÊN MỞ — QUÉT TRONG PHIÊN</b>")
    else:
        kinds = sorted({str(e.get("kind")) for e in events})
        lines.append(f"📡 <b>QUÉT TRONG PHIÊN</b> — <i>{html_escape(', '.join(kinds))}</i>")
    lines.append(f"<i>{_PROVISIONAL_TAG}</i>")

    for horizon in HORIZONS:
        cur = scores_now.get(horizon, {})
        if not cur:
            continue
        prev_eod = (scores_prev_eod or {}).get(horizon, {})
        top3 = _top3(cur)
        lines.append(f"\n<b>{_HORIZON_LABEL.get(horizon, f'H{horizon}')}</b>")
        for ticker in top3:
            p_now = cur[ticker]
            price_info = prices.get(ticker, {})
            last = price_info.get("last")
            pct = price_info.get("pct")
            eod = prev_eod.get(ticker)
            d_eod = f" | ΔEOD {(p_now - eod) * 100:+.1f}pp" if eod is not None else ""
            price_txt = f" | {last:,.2f}k" if isinstance(last, (int, float)) else ""
            pct_txt = f" ({pct:+.1f}%)" if isinstance(pct, (int, float)) else ""
            lines.append(
                f"• <b>{html_escape(ticker)}</b>: P(UP) {p_now * 100:.1f}%{d_eod}{price_txt}{pct_txt}"
            )

    card = "\n".join(lines)
    if len(card) > 3500:  # defensive cap — should never trigger for top-3×2
        card = card[:3480] + "\n… (rút gọn)"
    return card


def html_escape(text: str) -> str:
    """Minimal HTML-escape for the ticker/kind text placed in the card."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ─────────────────────────────────────────────────────────────────────────────
# I/O + orchestration layer. Imports NO telegram/asyncio (see module docstring).
# ─────────────────────────────────────────────────────────────────────────────


def fetch_snapshot(exchange: str = "HOSE") -> list[dict[str, Any]]:
    """Thin wrapper over the ONE audited SSI iBoard bulk snapshot call.

    Reuses `foreign_flow_crawler.fetch_ssi_hose_snapshot` (the single tenacity
    retry policy — NOT duplicated here). Degrades to `[]` on any failure
    (exhausted retries, malformed response) so `run_scan` treats it identically
    to an empty snapshot and no-ops that cycle instead of raising.
    """
    from src.data.foreign_flow_crawler import fetch_ssi_hose_snapshot  # noqa: PLC0415

    try:
        return fetch_ssi_hose_snapshot(exchange)
    except Exception as exc:  # noqa: BLE001 — exhausted retries / malformed response
        LOGGER.warning("[intraday_scanner] SSI snapshot fetch failed: %s — no-op this cycle.", exc)
        return []


@dataclass
class ScanResult:
    """Return of `run_scan`: the updated state plus an optional card to send."""

    state: ScannerState
    card: str | None = None


def run_scan(
    state: ScannerState,
    now: datetime,
    *,
    tz: str = "Asia/Ho_Chi_Minh",
    delta_pp: float = 0.02,
    window_rows: int = 120,
) -> ScanResult:
    """One scan cycle (I/O-bearing orchestration).

    Steps (see the plan's Data Flow section):
      1. is_trading_window(now)? no → return state unchanged, card=None.
      2. fetch_snapshot(); empty/failed → return state unchanged, card=None.
      3. load the 120-row OHLCV tails (READ-ONLY) → splice provisional bars.
      4. rescore both horizons via the cached serve bots.
      5. resolve the ADV top-N universe; restrict both horizons' scores to it.
      6. detect events; if any, build a card. Update state (in-memory) with the
         current scores/top3 and reset to a fresh baseline when the calendar
         date changes.

    Never raises past this function: every failure branch logs and returns the
    state unchanged with card=None. This is the degrade-not-crash contract — a
    scan-cycle failure must never propagate into the bot's job-queue surface.

    Imports of `main` and `Alpha360Generator` are lazy (inside the function) so
    this module stays importable/unit-testable without loading the heavy serve
    stack or a model artifact.
    """
    if not is_trading_window(now, tz):
        LOGGER.debug("[intraday_scanner] outside trading window (%s) — no-op.", now)
        return ScanResult(state=state, card=None)

    snapshot = fetch_snapshot()
    if not snapshot:
        LOGGER.info("[intraday_scanner] empty SSI snapshot (holiday/off-hours?) — no-op.")
        return ScanResult(state=state, card=None)

    try:
        tails = _load_tails(window_rows)
    except Exception as exc:  # noqa: BLE001 — no parquet shards / read failure
        LOGGER.error("[intraday_scanner] OHLCV tail load failed: %s — skip cycle.", exc)
        return ScanResult(state=state, card=None)

    try:
        spliced = splice_all(tails, snapshot)
    except Exception as exc:  # noqa: BLE001 — defensive; splice is pure but guard anyway
        LOGGER.error("[intraday_scanner] provisional splice failed: %s — skip cycle.", exc)
        return ScanResult(state=state, card=None)

    try:
        scores_now, universe = _rescore_all_horizons(spliced)
    except Exception as exc:  # noqa: BLE001 — model missing / recipe skew / feature error
        LOGGER.error(
            "[intraday_scanner] rescore failed: %s — skip cycle. "
            "If persistent, check the serve model artifacts (train_models.py).",
            exc,
        )
        return ScanResult(state=state, card=None)

    if not any(scores_now.get(h) for h in HORIZONS):
        LOGGER.info("[intraday_scanner] no in-universe scores this cycle — no-op.")
        return ScanResult(state=state, card=None)

    # Daily reset: a new calendar date invalidates the prior in-memory baseline.
    scan_date = now.date()
    previous = state.previous
    prev_top3 = state.prev_top3
    if state.session_date is not None and state.session_date != scan_date:
        previous, prev_top3 = None, None

    thresholds = _thresholds()
    top3 = {h: _top3(scores_now.get(h, {})) for h in HORIZONS}
    events = detect_events(
        current=scores_now,
        previous=previous,
        thresholds=thresholds,
        delta_pp=delta_pp,
        top3_by_horizon=top3,
        prev_top3_by_horizon=prev_top3,
    )

    card: str | None = None
    if events:
        prices = _extract_prices(snapshot, universe)
        card = build_scan_card(events, scores_now, scores_prev_eod=None, prices=prices)

    new_state = ScannerState(
        previous=scores_now,
        prev_top3=top3,
        session_date=scan_date,
        last_card=card if card is not None else state.last_card,
    )
    # Memory hygiene (2026-07-12): each cycle transiently allocates ~1 GB of
    # pandas/numpy intermediates (full-panel feature build × 2 horizons)
    # inside the LONG-LIVED bot process; without an explicit release the
    # process RSS ratchets between scans and starves low-RAM laptops. The
    # retained state above is tiny (score dicts) — everything else is
    # droppable right now.
    del tails, spliced, snapshot, scores_now
    _release_memory()
    return ScanResult(state=new_state, card=card)


def _release_memory() -> None:
    """Force-release scan transients: full GC, then (Windows) trim the
    process working set so the freed pages actually return to the OS instead
    of sitting in the allocator. Best-effort — never raises."""
    gc.collect()
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(
                handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0
            )
        except Exception:  # noqa: BLE001 — trimming is opportunistic
            LOGGER.debug("[intraday_scanner] working-set trim unavailable.")


def _load_tails(window_rows: int) -> pl.DataFrame:
    """Load the live OHLCV tails (READ-ONLY). Lazy import of the heavy loader."""
    from src.features.alpha360_generator import Alpha360Generator  # noqa: PLC0415

    return Alpha360Generator().load_live_ohlcv_window(window_rows=window_rows)


def _rescore_all_horizons(
    spliced: pl.DataFrame,
) -> tuple[dict[int, dict[str, float]], frozenset[str]]:
    """Rescore both horizons on the spliced panel, restricted to the ADV universe.

    Reuses main's cached serve bots (`_load_v3_bot`), the shared feature builder
    (`_compute_v3_features`), and the SAME ADV top-N universe gate
    (`_resolve_candidate_universe`) daily_inference uses. Returns
    ({horizon: {ticker: p_up}} restricted to the universe, universe frozenset).
    """
    import main  # noqa: PLC0415

    universe = main._resolve_candidate_universe(spliced)
    latest_df = spliced.to_pandas()

    scores: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        bot = main._load_v3_bot(horizon)
        feature_list = list(bot.tabular_features)
        frac_diff_d = float(bot.metadata.get("frac_diff_d", 0.4))
        feats = main._compute_v3_features(latest_df, feature_list, frac_diff_d)
        if feats.empty:
            scores[horizon] = {}
            continue
        p_up = bot.predict_proba(feats)
        scores[horizon] = {
            str(t): float(p) for t, p in p_up.items() if str(t) in universe
        }
    return scores, universe


def _thresholds() -> dict[int, float]:
    """Per-horizon τ: T+20 = main.SAFE_BUY_THRESHOLD, T+5 = the 5d bot's live gate.

    T+5's τ is read LIVE from the loaded artifact's `up_threshold` (never
    hardcoded) so a future retrain can't silently desync the alert threshold
    from the model's actual gate.
    """
    import main  # noqa: PLC0415

    tau = {20: float(main.SAFE_BUY_THRESHOLD)}
    try:
        tau[5] = float(main._load_v3_bot(5).up_threshold)
    except Exception as exc:  # noqa: BLE001 — degrade: skip the 5d τ-cross detection
        LOGGER.warning("[intraday_scanner] could not read T+5 up_threshold: %s", exc)
    return tau


def _extract_prices(
    snapshot: list[dict[str, Any]], universe: frozenset[str]
) -> dict[str, dict[str, float]]:
    """Build {ticker: {"last": close_thousands, "pct": % vs refPrice}} for the card.

    Only for in-universe tickers. Prices scaled to thousands (matching the card's
    "k" unit); % change is (matchedPrice - refPrice) / refPrice in absolute VND
    (a ratio, unit-independent). Malformed rows are skipped silently.
    """
    out: dict[str, dict[str, float]] = {}
    for item in snapshot:
        symbol = item.get("stockSymbol")
        if not symbol or str(symbol).upper() not in universe:
            continue
        matched = item.get("matchedPrice")
        ref = item.get("refPrice")
        if matched is None:
            continue
        try:
            last_k = float(matched) / 1000.0
        except (TypeError, ValueError):
            continue
        entry: dict[str, float] = {"last": last_k}
        try:
            if ref is not None and float(ref) != 0.0:
                entry["pct"] = (float(matched) - float(ref)) / float(ref) * 100.0
        except (TypeError, ValueError):
            pass
        out[str(symbol).upper()] = entry
    return out


__all__ = [
    "ScannerState",
    "ScanResult",
    "is_trading_window",
    "snapshot_row_to_provisional_bar",
    "splice_provisional_bar",
    "splice_all",
    "detect_events",
    "build_scan_card",
    "fetch_snapshot",
    "run_scan",
    "HORIZONS",
]
