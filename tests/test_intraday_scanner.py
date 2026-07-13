"""Tests for src/trading/intraday_scanner.py.

Pure-function tests + degrade-path tests. NO live HTTP, NO live PTB Application,
NO apscheduler dependency — every test passes in the bare dev environment (the
scanner's runtime apscheduler dep lives only in the bot's conda env). All SSI
fetch paths and the heavy serve stack are monkeypatched.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.trading import intraday_scanner as scanner


# ── is_trading_window ────────────────────────────────────────────────────────

def _ict(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    """Naive datetime; is_trading_window interprets naive as ICT."""
    return datetime(y, m, d, hh, mm)


def test_is_trading_window_inside_morning() -> None:
    # 2026-07-06 is a Monday.
    assert scanner.is_trading_window(_ict(2026, 7, 6, 9, 15)) is True   # open bound
    assert scanner.is_trading_window(_ict(2026, 7, 6, 10, 30)) is True
    assert scanner.is_trading_window(_ict(2026, 7, 6, 11, 30)) is True  # close bound


def test_is_trading_window_inside_afternoon() -> None:
    assert scanner.is_trading_window(_ict(2026, 7, 6, 13, 0)) is True   # open bound
    assert scanner.is_trading_window(_ict(2026, 7, 6, 14, 0)) is True
    assert scanner.is_trading_window(_ict(2026, 7, 6, 14, 45)) is True  # close bound


def test_is_trading_window_before_open() -> None:
    assert scanner.is_trading_window(_ict(2026, 7, 6, 9, 14)) is False
    assert scanner.is_trading_window(_ict(2026, 7, 6, 8, 0)) is False


def test_is_trading_window_lunch_gap() -> None:
    assert scanner.is_trading_window(_ict(2026, 7, 6, 11, 31)) is False
    assert scanner.is_trading_window(_ict(2026, 7, 6, 12, 0)) is False
    assert scanner.is_trading_window(_ict(2026, 7, 6, 12, 59)) is False


def test_is_trading_window_after_close() -> None:
    assert scanner.is_trading_window(_ict(2026, 7, 6, 14, 46)) is False
    assert scanner.is_trading_window(_ict(2026, 7, 6, 16, 0)) is False


def test_is_trading_window_weekend() -> None:
    # 2026-07-04 = Saturday, 2026-07-05 = Sunday — always False even mid-session.
    assert scanner.is_trading_window(_ict(2026, 7, 4, 10, 0)) is False
    assert scanner.is_trading_window(_ict(2026, 7, 5, 14, 0)) is False


# ── snapshot_row_to_provisional_bar ──────────────────────────────────────────

def _ssi_row(**over) -> dict:
    base = {
        "stockSymbol": "SSI",
        "tradingDate": "20260706",
        "openPrice": 22_000.0,
        "highest": 23_000.0,
        "lowest": 21_500.0,
        "matchedPrice": 22_600.0,
        "nmTotalTradedQty": 1_234_500.0,
        "refPrice": 22_000.0,
    }
    base.update(over)
    return base


def test_snapshot_row_scales_prices_by_1000() -> None:
    bar = scanner.snapshot_row_to_provisional_bar(_ssi_row())
    assert bar is not None
    assert bar["ticker"] == "SSI"
    assert bar["date"] == date(2026, 7, 6)
    assert bar["open"] == 22.0
    assert bar["high"] == 23.0
    assert bar["low"] == 21.5
    assert bar["close"] == 22.6
    # volume is passed through UNSCALED (raw share count).
    assert bar["volume"] == 1_234_500.0


def test_snapshot_row_custom_divisor() -> None:
    bar = scanner.snapshot_row_to_provisional_bar(_ssi_row(), price_unit_divisor=1.0)
    assert bar is not None
    assert bar["close"] == 22_600.0  # no scaling


def test_snapshot_row_missing_symbol_returns_none() -> None:
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(stockSymbol=None)) is None
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(stockSymbol="")) is None


def test_snapshot_row_malformed_date_returns_none() -> None:
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(tradingDate="2026-07-06")) is None  # wrong format
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(tradingDate="2026070")) is None     # 7 chars
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(tradingDate="20261306")) is None     # month 13
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(tradingDate="abcdefgh")) is None     # non-digit


def test_snapshot_row_missing_matched_price_returns_none() -> None:
    assert scanner.snapshot_row_to_provisional_bar(_ssi_row(matchedPrice=None)) is None


def test_snapshot_row_missing_ohl_fall_back_to_close() -> None:
    bar = scanner.snapshot_row_to_provisional_bar(
        _ssi_row(openPrice=None, highest=None, lowest=None)
    )
    assert bar is not None
    assert bar["open"] == bar["close"] == 22.6
    assert bar["high"] == bar["low"] == 22.6


# ── splice_provisional_bar ───────────────────────────────────────────────────

def _tail(ticker: str, dates: list[date], closes: list[float]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": dates,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        },
        schema={
            "ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64,
            "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _prov(ticker: str, d: date, close: float) -> dict:
    return {
        "ticker": ticker, "date": d, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 5000.0,
    }


def test_splice_append_when_new_date() -> None:
    tail = _tail("SSI", [date(2026, 7, 3), date(2026, 7, 4)], [22.0, 22.3])
    out = scanner.splice_provisional_bar(tail, _prov("SSI", date(2026, 7, 6), 22.9))
    assert out.height == 3
    assert out.get_column("date").max() == date(2026, 7, 6)
    assert out.filter(pl.col("date") == date(2026, 7, 6)).row(0, named=True)["close"] == 22.9


def test_splice_replace_when_same_date() -> None:
    tail = _tail("SSI", [date(2026, 7, 3), date(2026, 7, 6)], [22.0, 22.3])
    out = scanner.splice_provisional_bar(tail, _prov("SSI", date(2026, 7, 6), 22.9))
    # Height unchanged (replaced, not appended).
    assert out.height == 2
    today = out.filter(pl.col("date") == date(2026, 7, 6))
    assert today.height == 1
    assert today.row(0, named=True)["close"] == 22.9  # provisional value won


def test_splice_preserves_schema_and_dtypes() -> None:
    tail = _tail("SSI", [date(2026, 7, 3)], [22.0])
    out = scanner.splice_provisional_bar(tail, _prov("SSI", date(2026, 7, 6), 22.9))
    assert out.columns == list(scanner._OHLCV_COLUMNS)
    assert out.schema == tail.schema


def test_splice_does_not_mutate_input() -> None:
    tail = _tail("SSI", [date(2026, 7, 3), date(2026, 7, 4)], [22.0, 22.3])
    before = tail.clone()
    _ = scanner.splice_provisional_bar(tail, _prov("SSI", date(2026, 7, 6), 22.9))
    assert tail.equals(before)  # original frame unchanged


# ── splice_all ───────────────────────────────────────────────────────────────

def test_splice_all_passthrough_for_unmatched_tickers() -> None:
    tails = pl.concat([
        _tail("SSI", [date(2026, 7, 3), date(2026, 7, 4)], [22.0, 22.3]),
        _tail("VJC", [date(2026, 7, 3), date(2026, 7, 4)], [90.0, 91.0]),
    ])
    # Only SSI has a snapshot row; VJC must pass through untouched.
    snapshot = [_ssi_row(stockSymbol="SSI", matchedPrice=22_900.0)]
    out = scanner.splice_all(tails, snapshot)
    vjc = out.filter(pl.col("ticker") == "VJC")
    assert vjc.height == 2  # untouched, no provisional appended
    ssi = out.filter(pl.col("ticker") == "SSI")
    assert ssi.height == 3  # SSI got its provisional bar
    assert ssi.get_column("date").max() == date(2026, 7, 6)


def test_splice_all_multiple_matches() -> None:
    tails = pl.concat([
        _tail("SSI", [date(2026, 7, 4)], [22.0]),
        _tail("VJC", [date(2026, 7, 4)], [90.0]),
    ])
    snapshot = [
        _ssi_row(stockSymbol="SSI", matchedPrice=22_900.0),
        _ssi_row(stockSymbol="VJC", matchedPrice=91_500.0),
    ]
    out = scanner.splice_all(tails, snapshot)
    assert out.filter(pl.col("ticker") == "SSI").height == 2
    assert out.filter(pl.col("ticker") == "VJC").height == 2
    vjc_today = out.filter((pl.col("ticker") == "VJC") & (pl.col("date") == date(2026, 7, 6)))
    assert vjc_today.row(0, named=True)["close"] == 91.5


def test_splice_all_skips_malformed_snapshot_rows() -> None:
    tails = _tail("SSI", [date(2026, 7, 4)], [22.0])
    snapshot = [_ssi_row(stockSymbol="SSI", tradingDate="bad")]  # malformed → skipped
    out = scanner.splice_all(tails, snapshot)
    assert out.height == 1  # no provisional appended (row was malformed)


def test_splice_all_empty_tails_passthrough() -> None:
    empty = _tail("SSI", [], []).clear()
    out = scanner.splice_all(empty, [_ssi_row()])
    assert out.is_empty()


# ── detect_events ────────────────────────────────────────────────────────────

_TAU = {20: 0.45, 5: 0.40}


def test_detect_events_first_scan_baseline_only() -> None:
    current = {20: {"AAA": 0.6, "BBB": 0.5, "CCC": 0.4}, 5: {"AAA": 0.5}}
    top3 = {20: ["AAA", "BBB", "CCC"], 5: ["AAA"]}
    events = scanner.detect_events(current, None, _TAU, 0.02, top3, None)
    assert len(events) == 1
    assert events[0]["kind"] == "baseline"
    # NO false new_entrant flood.
    assert all(e["kind"] != "new_entrant" for e in events)


def test_detect_events_threshold_cross_up() -> None:
    previous = {20: {"AAA": 0.40}}
    current = {20: {"AAA": 0.50}}  # crossed 0.45 upward
    top3 = {20: ["AAA"]}
    events = scanner.detect_events(current, previous, _TAU, 0.02, top3, {20: ["AAA"]})
    kinds = {e["kind"] for e in events if e["ticker"] == "AAA"}
    assert "threshold_cross" in kinds


def test_detect_events_threshold_cross_down() -> None:
    previous = {20: {"AAA": 0.50}}
    current = {20: {"AAA": 0.40}}  # crossed 0.45 downward
    top3 = {20: ["AAA"]}
    events = scanner.detect_events(current, previous, _TAU, 0.02, top3, {20: ["AAA"]})
    assert any(e["kind"] == "threshold_cross" for e in events)


def test_detect_events_delta_move() -> None:
    previous = {20: {"AAA": 0.30}}
    current = {20: {"AAA": 0.33}}  # +3pp >= 2pp threshold, AAA is top-3
    top3 = {20: ["AAA"]}
    events = scanner.detect_events(current, previous, _TAU, 0.02, top3, {20: ["AAA"]})
    assert any(e["kind"] == "delta_move" for e in events)


def test_detect_events_new_entrant() -> None:
    previous = {20: {"AAA": 0.6, "BBB": 0.5, "CCC": 0.4, "DDD": 0.1}}
    current = {20: {"AAA": 0.6, "BBB": 0.5, "DDD": 0.55, "CCC": 0.4}}
    top3_now = {20: ["AAA", "DDD", "BBB"]}   # DDD newly entered top-3
    prev_top3 = {20: ["AAA", "BBB", "CCC"]}
    events = scanner.detect_events(current, previous, _TAU, 0.02, top3_now, prev_top3)
    entrants = {e["ticker"] for e in events if e["kind"] == "new_entrant"}
    assert "DDD" in entrants


def test_detect_events_no_event_when_nothing_crosses() -> None:
    previous = {20: {"AAA": 0.30, "BBB": 0.20}}
    current = {20: {"AAA": 0.305, "BBB": 0.205}}  # <2pp move, no τ cross, same top3
    top3 = {20: ["AAA", "BBB"]}
    events = scanner.detect_events(current, previous, _TAU, 0.02, top3, {20: ["AAA", "BBB"]})
    assert events == []


# ── build_scan_card ──────────────────────────────────────────────────────────

def test_build_scan_card_contains_provisional_tag() -> None:
    events = [{"ticker": None, "horizon": None, "kind": "baseline", "p_up": None, "delta": 0.0}]
    scores = {20: {"AAA": 0.6, "BBB": 0.5, "CCC": 0.4}, 5: {"AAA": 0.42}}
    prices = {"AAA": {"last": 22.6, "pct": 2.7}}
    card = scanner.build_scan_card(events, scores, None, prices)
    assert scanner._PROVISIONAL_TAG in card
    assert "TÍN HIỆU GIAO DỊCH" in card


def test_build_scan_card_no_buy_wording() -> None:
    events = [{"ticker": "AAA", "horizon": 20, "kind": "delta_move", "p_up": 0.6, "delta": 0.03}]
    scores = {20: {"AAA": 0.6, "BBB": 0.5}, 5: {"AAA": 0.5}}
    card = scanner.build_scan_card(events, scores, None, {})
    lowered = card.lower()
    assert "buy" not in lowered
    assert "mua" not in lowered  # Vietnamese "buy"


def test_build_scan_card_under_char_budget() -> None:
    # Even with many names, top-3×2 card must stay well under Telegram's 4096.
    scores = {
        20: {f"T{i:02d}": 0.5 - i * 0.01 for i in range(30)},
        5: {f"T{i:02d}": 0.4 - i * 0.01 for i in range(30)},
    }
    events = [{"ticker": "T00", "horizon": 20, "kind": "new_entrant", "p_up": 0.5, "delta": 0.0}]
    card = scanner.build_scan_card(events, scores, None, {})
    assert len(card) < 3500


def test_build_scan_card_shows_only_top3_per_horizon() -> None:
    scores = {20: {"AAA": 0.6, "BBB": 0.5, "CCC": 0.4, "DDD": 0.3}, 5: {}}
    events = [{"ticker": "AAA", "horizon": 20, "kind": "delta_move", "p_up": 0.6, "delta": 0.03}]
    card = scanner.build_scan_card(events, scores, None, {})
    assert "AAA" in card and "BBB" in card and "CCC" in card
    assert "DDD" not in card  # 4th name excluded


# ── fetch_snapshot / run_scan degrade paths (monkeypatched, no live HTTP) ─────

def test_fetch_snapshot_degrades_to_empty_on_exception(monkeypatch) -> None:
    from src.data import foreign_flow_crawler as ffc

    def _boom(exchange: str = "HOSE"):
        raise RuntimeError("network down")

    monkeypatch.setattr(ffc, "fetch_ssi_hose_snapshot", _boom)
    assert scanner.fetch_snapshot() == []  # degraded, no raise


def test_fetch_snapshot_returns_payload(monkeypatch) -> None:
    from src.data import foreign_flow_crawler as ffc

    monkeypatch.setattr(ffc, "fetch_ssi_hose_snapshot", lambda exchange="HOSE": [_ssi_row()])
    out = scanner.fetch_snapshot()
    assert len(out) == 1
    assert out[0]["stockSymbol"] == "SSI"


def test_run_scan_outside_window_is_noop() -> None:
    state = scanner.ScannerState()
    result = scanner.run_scan(state, _ict(2026, 7, 6, 8, 0))  # before open
    assert result.card is None
    assert result.state is state  # unchanged


def test_run_scan_empty_snapshot_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [])
    state = scanner.ScannerState()
    result = scanner.run_scan(state, _ict(2026, 7, 6, 10, 0))  # in window
    assert result.card is None
    assert result.state is state


def test_run_scan_tail_load_failure_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [_ssi_row()])

    def _boom(window_rows):
        raise FileNotFoundError("no parquet shards")

    monkeypatch.setattr(scanner, "_load_tails", _boom)
    state = scanner.ScannerState()
    result = scanner.run_scan(state, _ict(2026, 7, 6, 10, 0))
    assert result.card is None
    assert result.state is state


def test_run_scan_rescore_failure_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [_ssi_row()])
    monkeypatch.setattr(
        scanner, "_load_tails",
        lambda window_rows: _tail("SSI", [date(2026, 7, 4)], [22.0]),
    )

    def _boom(spliced):
        raise RuntimeError("model artifact missing")

    monkeypatch.setattr(scanner, "_rescore_all_horizons", _boom)
    state = scanner.ScannerState()
    result = scanner.run_scan(state, _ict(2026, 7, 6, 10, 0))
    assert result.card is None
    assert result.state is state


def test_run_scan_baseline_card_on_first_scan(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [_ssi_row(stockSymbol="AAA")])
    monkeypatch.setattr(
        scanner, "_load_tails",
        lambda window_rows: _tail("AAA", [date(2026, 7, 4)], [22.0]),
    )
    monkeypatch.setattr(
        scanner, "_rescore_all_horizons",
        lambda spliced: ({20: {"AAA": 0.6}, 5: {"AAA": 0.42}}, frozenset({"AAA"})),
    )
    monkeypatch.setattr(scanner, "_thresholds", lambda: {20: 0.45, 5: 0.40})

    state = scanner.ScannerState()
    result = scanner.run_scan(state, _ict(2026, 7, 6, 10, 0))
    assert result.card is not None
    assert scanner._PROVISIONAL_TAG in result.card
    # State now carries the current scores + top3 + session date.
    assert result.state.previous == {20: {"AAA": 0.6}, 5: {"AAA": 0.42}}
    assert result.state.session_date == date(2026, 7, 6)


def test_run_scan_no_events_second_scan_no_card(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [_ssi_row(stockSymbol="AAA")])
    monkeypatch.setattr(
        scanner, "_load_tails",
        lambda window_rows: _tail("AAA", [date(2026, 7, 4)], [22.0]),
    )
    monkeypatch.setattr(
        scanner, "_rescore_all_horizons",
        lambda spliced: ({20: {"AAA": 0.60}, 5: {"AAA": 0.42}}, frozenset({"AAA"})),
    )
    monkeypatch.setattr(scanner, "_thresholds", lambda: {20: 0.45, 5: 0.40})

    # Prior state = same scores, same date → no event, no card on this scan.
    prior = scanner.ScannerState(
        previous={20: {"AAA": 0.60}, 5: {"AAA": 0.42}},
        prev_top3={20: ["AAA"], 5: ["AAA"]},
        session_date=date(2026, 7, 6),
    )
    result = scanner.run_scan(prior, _ict(2026, 7, 6, 10, 30))
    assert result.card is None
    # State still refreshed to the current scores.
    assert result.state.session_date == date(2026, 7, 6)


def test_run_scan_new_calendar_date_resets_to_baseline(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "fetch_snapshot", lambda exchange="HOSE": [_ssi_row(stockSymbol="AAA")])
    monkeypatch.setattr(
        scanner, "_load_tails",
        lambda window_rows: _tail("AAA", [date(2026, 7, 6)], [22.0]),
    )
    monkeypatch.setattr(
        scanner, "_rescore_all_horizons",
        lambda spliced: ({20: {"AAA": 0.60}, 5: {"AAA": 0.42}}, frozenset({"AAA"})),
    )
    monkeypatch.setattr(scanner, "_thresholds", lambda: {20: 0.45, 5: 0.40})

    # Prior state from a PREVIOUS day → the new date should reset to baseline.
    prior = scanner.ScannerState(
        previous={20: {"AAA": 0.10}, 5: {"AAA": 0.10}},
        prev_top3={20: ["AAA"], 5: ["AAA"]},
        session_date=date(2026, 7, 3),  # older date
    )
    result = scanner.run_scan(prior, _ict(2026, 7, 6, 10, 0))
    assert result.card is not None  # baseline card, not a spurious huge-delta alert
    assert "PHIÊN MỞ" in result.card
    assert result.state.session_date == date(2026, 7, 6)


def test_release_memory_never_raises() -> None:
    """Hygiene helper is best-effort by contract - must run cleanly anywhere."""
    from src.trading.intraday_scanner import _release_memory
    _release_memory()
    _release_memory()  # idempotent
