"""Tests for the Portfolio Guard EOD protective alert layer.

Part A — normalize_entry_price_vnd + evaluate_position (pure)
Part B — build_guard_alert_card (pure)
Part C — load_guard_positions (I/O, temp-file DuckDB, cron exclusion)
Part D — TelegramBot.send_text_to_chat / _send_to_one (single-recipient send)
Part E — main._run_guard_for_users / main.notify_portfolio_guard (orchestration)

Pure-function tests need no DB/stubs. The loader test seeds a temp-file DuckDB
(a fresh :memory: connect would be isolated per-call, so a file is required).
The orchestration tests monkeypatch the heavy serve stack (predict_v3_horizon,
Alpha360Generator, evaluate_trades_batch, mr_score_tickers, price_lookup,
TelegramBot) — mirroring tests/test_intraday_scanner.py's run_scan style.
"""
from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

from src.reports.builders import _MR_SELL_VETO
from src.trading import portfolio_guard as pg

_KW = {"stop_loss_pct": -0.07, "take_profit_pct": 0.15, "trailing_pct": 0.08}


def _kinds(triggers: list[dict]) -> set[str]:
    return {t["kind"] for t in triggers}


# ─────────────────────────────────────────────────────────────────────────────
# Part A.1 — normalize_entry_price_vnd
# ─────────────────────────────────────────────────────────────────────────────


def test_normalize_thousands_scale() -> None:
    # 32.5 is only economically sane as 32,500 VND (bot /add example convention).
    assert pg.normalize_entry_price_vnd(32.5) == 32500.0


def test_normalize_already_absolute() -> None:
    assert pg.normalize_entry_price_vnd(47800.0) == 47800.0


def test_normalize_exact_boundary_unchanged() -> None:
    # Rule is strictly `< 1000`, so 1000.0 stays 1000.0 (NOT multiplied).
    assert pg.normalize_entry_price_vnd(1000.0) == 1000.0
    # Just below the boundary still scales.
    assert pg.normalize_entry_price_vnd(999.99) == pytest.approx(999_990.0)


def test_normalize_zero_negative_none() -> None:
    assert pg.normalize_entry_price_vnd(0.0) == 0.0
    assert pg.normalize_entry_price_vnd(-5.0) == 0.0
    assert pg.normalize_entry_price_vnd(None) == 0.0
    assert pg.normalize_entry_price_vnd("garbage") == 0.0  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Part A.2 — evaluate_position triggers
# ─────────────────────────────────────────────────────────────────────────────


def _pos(price: float = 100.0) -> dict:
    return {"entry_price_raw": price, "ticker": "AAA"}


def test_empty_series_degrades_to_empty() -> None:
    assert pg.evaluate_position(_pos(), [], None, None, None, **_KW) == []


def test_nonpositive_entry_degrades_to_empty() -> None:
    assert pg.evaluate_position(_pos(0.0), [100000.0, 90000.0], None, None, None, **_KW) == []


def test_hard_stop_fires_confident() -> None:
    # entry 100 (thousands) -> 100000 abs; today 92000 -> pnl -8% <= -7%.
    triggers = pg.evaluate_position(
        _pos(100.0), [100000.0, 98000.0, 95000.0, 92000.0], None, None, None, **_KW
    )
    assert "hard_stop" in _kinds(triggers)
    hard = next(t for t in triggers if t["kind"] == "hard_stop")
    assert "CẮT LỖ" in hard["message_vi"]
    assert "-8.0%" in hard["message_vi"]
    assert "-7%" in hard["message_vi"]
    assert hard["ca_gap_downgraded"] is False


def test_hard_stop_not_fires_when_shallow() -> None:
    # pnl -5% > -7% → no hard stop. Trailing pct large so it doesn't fire either.
    triggers = pg.evaluate_position(
        _pos(100.0), [100000.0, 95000.0], None, None, None,
        stop_loss_pct=-0.07, take_profit_pct=0.15, trailing_pct=0.50,
    )
    assert "hard_stop" not in _kinds(triggers)


def test_take_profit_fires() -> None:
    # Monotonic rise to +16% → take_profit, no trailing (today == peak).
    triggers = pg.evaluate_position(
        _pos(100.0), [100000.0, 108000.0, 116000.0], None, None, None, **_KW
    )
    assert "take_profit" in _kinds(triggers)
    tp = next(t for t in triggers if t["kind"] == "take_profit")
    assert "CHỐT LỜI" in tp["message_vi"]
    assert "+16.0%" in tp["message_vi"]


def test_trailing_stop_fires() -> None:
    # peak 108000, today 99000 → drawdown 8.33% >= 8%; pnl -1% (no hard stop).
    triggers = pg.evaluate_position(
        _pos(100.0), [100000.0, 108000.0, 99000.0], None, None, None, **_KW
    )
    assert "trailing_stop" in _kinds(triggers)
    tr = next(t for t in triggers if t["kind"] == "trailing_stop")
    assert "TRAILING STOP" in tr["message_vi"]
    assert "108,000 VND" in tr["message_vi"]  # peak rendered in absolute VND
    assert tr["ca_gap_downgraded"] is False


def test_model_flip_only_5d() -> None:
    flat = [100000.0, 100000.0]
    triggers = pg.evaluate_position(
        _pos(100.0), flat, [0.6, 0.2, 0.2], [0.1, 0.2, 0.7], None, **_KW
    )
    assert "model_flip" in _kinds(triggers)
    flip = next(t for t in triggers if t["kind"] == "model_flip")
    assert "T+5" in flip["message_vi"]
    assert "T+20" not in flip["message_vi"]
    assert "P(Tăng)=20%" in flip["message_vi"]


def test_model_flip_only_20d() -> None:
    flat = [100000.0, 100000.0]
    triggers = pg.evaluate_position(
        _pos(100.0), flat, [0.1, 0.2, 0.7], [0.55, 0.25, 0.20], None, **_KW
    )
    flip = next(t for t in triggers if t["kind"] == "model_flip")
    assert "T+20" in flip["message_vi"]
    assert "T+5" not in flip["message_vi"]


def test_model_flip_both_horizons_or_rule() -> None:
    flat = [100000.0, 100000.0]
    triggers = pg.evaluate_position(
        _pos(100.0), flat, [0.6, 0.2, 0.2], [0.5, 0.1, 0.4], None, **_KW
    )
    flip = next(t for t in triggers if t["kind"] == "model_flip")
    assert "T+5" in flip["message_vi"] and "T+20" in flip["message_vi"]


def test_model_flip_not_fired_when_up() -> None:
    flat = [100000.0, 100000.0]
    triggers = pg.evaluate_position(
        _pos(100.0), flat, [0.1, 0.2, 0.7], [0.2, 0.2, 0.6], None, **_KW
    )
    assert "model_flip" not in _kinds(triggers)


def test_regime_warning_fires_no_trade() -> None:
    flat = [100000.0, 100000.0]
    for regime in (0, 7):
        triggers = pg.evaluate_position(_pos(100.0), flat, None, None, regime, **_KW)
        assert "regime_warning" in _kinds(triggers)
        rw = next(t for t in triggers if t["kind"] == "regime_warning")
        assert f"Regime {regime}" in rw["message_vi"]
        assert "CẢNH BÁO PHA THỊ TRƯỜNG" in rw["message_vi"]


def test_regime_warning_not_fired_normal_regime() -> None:
    flat = [100000.0, 100000.0]
    triggers = pg.evaluate_position(_pos(100.0), flat, None, None, 3, **_KW)
    assert "regime_warning" not in _kinds(triggers)


def test_ca_gap_downgrades_hard_stop_and_trailing() -> None:
    # A -40% single-session jump = corporate action, not price action.
    closes = [100000.0, 60000.0]
    triggers = pg.evaluate_position(_pos(100.0), closes, None, None, None, **_KW)
    hard = next(t for t in triggers if t["kind"] == "hard_stop")
    assert hard["ca_gap_downgraded"] is True
    assert "hành động doanh nghiệp" in hard["message_vi"]
    assert "CẮT LỖ" not in hard["message_vi"]  # confident wording suppressed
    trail = next(t for t in triggers if t["kind"] == "trailing_stop")
    assert trail["ca_gap_downgraded"] is True
    assert "hành động doanh nghiệp" in trail["message_vi"]


def test_ca_gap_does_not_downgrade_take_profit() -> None:
    # +40% CA jump: take-profit fires with CONFIDENT wording (approved scope).
    closes = [100000.0, 140000.0]
    triggers = pg.evaluate_position(_pos(100.0), closes, None, None, None, **_KW)
    tp = next(t for t in triggers if t["kind"] == "take_profit")
    assert tp["ca_gap_downgraded"] is False
    assert "CHỐT LỜI" in tp["message_vi"]


def test_trigger_order_is_deterministic() -> None:
    # Force hard_stop + trailing + model_flip + regime together; check order.
    closes = [100000.0, 108000.0, 90000.0]  # pnl -10%, drawdown ~16.7%
    triggers = pg.evaluate_position(
        _pos(100.0), closes, [0.6, 0.2, 0.2], None, 0, **_KW
    )
    order = [t["kind"] for t in triggers]
    assert order == ["hard_stop", "trailing_stop", "model_flip", "regime_warning"]


# ─────────────────────────────────────────────────────────────────────────────
# Part B — build_guard_alert_card
# ─────────────────────────────────────────────────────────────────────────────


def _lot(ticker="AAA", entry=date(2026, 7, 1), triggers=None, pnl=-0.08) -> dict:
    return {
        "ticker": ticker,
        "entry_date": entry,
        "entry_price_norm": 100000.0,
        "current_price": 92000.0,
        "pnl_pct": pnl,
        "volume": 1000,
        "triggers": triggers if triggers is not None else [
            {"kind": "hard_stop", "message_vi": "🔴 <b>CẮT LỖ</b>: PnL -8.0%", "ca_gap_downgraded": False}
        ],
    }


def test_card_has_header_and_footer() -> None:
    card = pg.build_guard_alert_card([_lot()], None, {})
    assert "CẢNH BÁO DANH MỤC" in card
    assert "══" in card
    assert "quyết định cuối" in card  # footer disclaimer
    assert "<b>AAA</b>" in card


def test_card_mr_veto_included_for_sell_leaning() -> None:
    card = pg.build_guard_alert_card([_lot()], None, {"AAA": {"fired": True}})
    assert _MR_SELL_VETO in card


def test_card_mr_veto_omitted_for_take_profit_only() -> None:
    tp_lot = _lot(triggers=[
        {"kind": "take_profit", "message_vi": "🟢 <b>CHỐT LỜI</b>", "ca_gap_downgraded": False}
    ], pnl=0.16)
    card = pg.build_guard_alert_card([tp_lot], None, {"AAA": {"fired": True}})
    assert _MR_SELL_VETO not in card


def test_card_mr_veto_omitted_when_not_fired() -> None:
    card = pg.build_guard_alert_card([_lot()], None, {"AAA": {"fired": False}})
    assert _MR_SELL_VETO not in card


def test_card_char_safety_with_many_positions() -> None:
    lots = [_lot(ticker=f"T{i:03d}", entry=date(2026, 7, 1)) for i in range(200)]
    card = pg.build_guard_alert_card(lots, None, {})
    assert len(card) <= 4096
    assert "vị thế khác" in card  # overflow notice present


def test_card_multi_lot_same_ticker_distinct_by_date() -> None:
    lot_a = _lot(ticker="HPG", entry=date(2026, 6, 1))
    lot_b = _lot(ticker="HPG", entry=date(2026, 7, 10))
    card = pg.build_guard_alert_card([lot_a, lot_b], None, {})
    assert "01/06/2026" in card
    assert "10/07/2026" in card


def test_card_quant_only_when_no_enrichment() -> None:
    card = pg.build_guard_alert_card([_lot()], None, {})
    assert "Trọng tài tin tức" not in card


def test_card_renders_enrichment_line() -> None:
    enrichment = ({"AAA": 0}, {"AAA": {"sentiment_score": -0.4, "reasoning_vi": "Tin xấu."}})
    card = pg.build_guard_alert_card([_lot()], enrichment, {})
    assert "Trọng tài tin tức" in card
    assert "BÁN/THOÁT" in card
    assert "Tin xấu." in card


# ─────────────────────────────────────────────────────────────────────────────
# Part C — load_guard_positions (temp-file DuckDB, cron exclusion)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def seeded_db(tmp_path):
    """Temp-file DuckDB seeded with 1 cron row + 2 real-user rows."""
    dbfile = tmp_path / "guard_test.duckdb"
    conn = duckdb.connect(str(dbfile))
    conn.execute(
        "CREATE TABLE portfolio (user_id VARCHAR, ticker VARCHAR, "
        "volume INTEGER, price DOUBLE, added_at TIMESTAMP)"
    )
    conn.executemany(
        "INSERT INTO portfolio VALUES (?, ?, ?, ?, ?)",
        [
            ("cron", "FPT", 500, 136.0, datetime(2026, 7, 1, 9, 0)),
            ("111", "HPG", 1000, 28.5, datetime(2026, 7, 2, 10, 0)),
            ("222", "VHM", 300, 41.2, datetime(2026, 7, 3, 11, 0)),
        ],
    )
    conn.close()
    return str(dbfile)


def test_load_excludes_cron_all_users(seeded_db) -> None:
    rows = pg.load_guard_positions(db_path=seeded_db)
    users = {r["user_id"] for r in rows}
    assert users == {"111", "222"}
    assert "cron" not in users
    assert len(rows) == 2


def test_load_single_user_filter_excludes_cron(seeded_db) -> None:
    rows = pg.load_guard_positions(db_path=seeded_db, user_id="111")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "111"
    assert rows[0]["ticker"] == "HPG"
    assert isinstance(rows[0]["entry_date"], date)


def test_load_single_user_cron_returns_empty(seeded_db) -> None:
    # Asking for cron explicitly still excludes it (user_id != cron AND = cron).
    assert pg.load_guard_positions(db_path=seeded_db, user_id="cron") == []


def test_load_missing_table_degrades_to_empty(tmp_path) -> None:
    empty = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty)).close()  # valid DB, no portfolio table
    assert pg.load_guard_positions(db_path=str(empty)) == []


# ─────────────────────────────────────────────────────────────────────────────
# Part D — TelegramBot.send_text_to_chat / _send_to_one
# ─────────────────────────────────────────────────────────────────────────────


def test_send_text_to_chat_single_recipient(monkeypatch) -> None:
    from src.utils import telegram_alerter as ta

    posted: list[dict] = []

    class _Resp:
        status_code = 200
        text = "ok"

    monkeypatch.setattr(ta.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(ta.requests, "post", lambda url, json, timeout: (posted.append(json), _Resp())[1])

    bot = ta.TelegramBot()
    bot.bot_token = "REAL-TOKEN"  # leave mock mode so it actually "posts"
    bot.chat_id_list = ["1", "2", "3"]  # must NOT be iterated

    bot.send_text_to_chat("999", "<b>hi</b>", "portfolio_guard")

    assert len(posted) == 1
    assert posted[0]["chat_id"] == "999"
    assert posted[0]["text"] == "<b>hi</b>"
    assert posted[0]["parse_mode"] == "HTML"


def test_send_text_to_chat_mock_mode_no_post(monkeypatch) -> None:
    from src.utils import telegram_alerter as ta

    calls: list = []
    monkeypatch.setattr(ta.requests, "post", lambda *a, **k: calls.append(1))

    bot = ta.TelegramBot()
    bot.bot_token = "YOUR_BOT_TOKEN"  # mock mode
    bot.send_text_to_chat("999", "<b>hi</b>", "portfolio_guard")

    assert calls == []  # mock mode logs, never posts


# ─────────────────────────────────────────────────────────────────────────────
# Part E — main._run_guard_for_users / main.notify_portfolio_guard
# ─────────────────────────────────────────────────────────────────────────────


class _FakePl:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class _FakeGen:
    """Stand-in for Alpha360Generator — returns a non-empty OHLCV frame."""

    def load_live_ohlcv_window(self, window_rows: int = 120) -> _FakePl:
        return _FakePl(pd.DataFrame({"ticker": ["AAA"], "date": [date(2026, 7, 13)],
                                     "close": [92.0]}))


def _patch_serve(monkeypatch, closes_map, preds=None, batch=None, mr=None):
    """Patch the whole serve stack _run_guard_for_users touches."""
    import main

    preds = preds if preds is not None else {"AAA": [0.2, 0.3, 0.5]}
    monkeypatch.setattr(main, "Alpha360Generator", _FakeGen)
    monkeypatch.setattr(main, "predict_v3_horizon",
                        lambda df, h: (dict(preds), {}, None, [], {}))
    monkeypatch.setattr(main.price_lookup, "closes_between",
                        lambda t, s, e, conn=None: list(closes_map.get(t, [])))
    monkeypatch.setattr(main, "mr_score_tickers", mr if mr is not None else (lambda ts: {}))
    if batch is not None:
        monkeypatch.setattr(main, "evaluate_trades_batch", batch)


def test_run_guard_event_only_no_trigger(monkeypatch) -> None:
    import main
    # Flat price → no trigger at all.
    _patch_serve(monkeypatch, {"AAA": [100.0, 100.0]})
    out = main._run_guard_for_users({"111": [
        {"ticker": "AAA", "price": 100.0, "volume": 100, "entry_date": date(2026, 7, 1)}
    ]}, today=date(2026, 7, 13))
    assert out == {}


def test_run_guard_hard_stop_builds_card(monkeypatch) -> None:
    import main
    _patch_serve(monkeypatch, {"AAA": [100.0, 92.0]},
                 batch=lambda preds, tickers: ({}, {}))
    out = main._run_guard_for_users({"111": [
        {"ticker": "AAA", "price": 100.0, "volume": 100, "entry_date": date(2026, 7, 1)}
    ]}, today=date(2026, 7, 13))
    assert "111" in out
    assert "CẮT LỖ" in out["111"]


def test_run_guard_gemini_exception_falls_back(monkeypatch) -> None:
    import main

    def _boom(preds, tickers):
        raise RuntimeError("Gemini 503")

    _patch_serve(monkeypatch, {"AAA": [100.0, 92.0]}, batch=_boom)
    monkeypatch.setattr(main.CONFIG.trading, "portfolio_guard_llm_enabled", True)
    out = main._run_guard_for_users({"111": [
        {"ticker": "AAA", "price": 100.0, "volume": 100, "entry_date": date(2026, 7, 1)}
    ]}, today=date(2026, 7, 13))
    # Card still built (quant-only) despite the arbitrator raising.
    assert "111" in out
    assert "CẮT LỖ" in out["111"]


def test_run_guard_llm_disabled_skips_batch(monkeypatch) -> None:
    import main
    called: list = []

    def _batch(preds, tickers):
        called.append(1)
        return ({}, {})

    _patch_serve(monkeypatch, {"AAA": [100.0, 92.0]}, batch=_batch)
    monkeypatch.setattr(main.CONFIG.trading, "portfolio_guard_llm_enabled", False)
    out = main._run_guard_for_users({"111": [
        {"ticker": "AAA", "price": 100.0, "volume": 100, "entry_date": date(2026, 7, 1)}
    ]}, today=date(2026, 7, 13))
    assert called == []            # evaluate_trades_batch NEVER invoked
    assert "111" in out            # quant-only card still built


def test_notify_disabled_no_db_read(monkeypatch) -> None:
    import main
    reads: list = []
    monkeypatch.setattr(main.CONFIG.trading, "portfolio_guard_enabled", False)
    monkeypatch.setattr(main.portfolio_guard, "load_guard_positions",
                        lambda *a, **k: reads.append(1) or [])
    assert main.notify_portfolio_guard() == 0
    assert reads == []             # short-circuit before any DB read


def test_notify_end_to_end_routes_per_user(monkeypatch) -> None:
    import main
    sent: list = []

    class _FakeBot:
        def send_text_to_chat(self, chat_id, html_text, label="alert"):
            sent.append((chat_id, label))

    positions = [
        {"user_id": "111", "ticker": "AAA", "price": 100.0, "volume": 100, "entry_date": date(2026, 7, 1)},
        {"user_id": "222", "ticker": "AAA", "price": 100.0, "volume": 50, "entry_date": date(2026, 7, 2)},
    ]
    monkeypatch.setattr(main.CONFIG.trading, "portfolio_guard_enabled", True)
    monkeypatch.setattr(main.portfolio_guard, "load_guard_positions", lambda *a, **k: positions)
    monkeypatch.setattr(main, "TelegramBot", _FakeBot)
    _patch_serve(monkeypatch, {"AAA": [100.0, 92.0]}, batch=lambda p, t: ({}, {}))

    count = main.notify_portfolio_guard()
    assert count == 2
    assert sorted(sent) == [("111", "portfolio_guard"), ("222", "portfolio_guard")]


def test_notify_no_positions_returns_zero(monkeypatch) -> None:
    import main
    monkeypatch.setattr(main.CONFIG.trading, "portfolio_guard_enabled", True)
    monkeypatch.setattr(main.portfolio_guard, "load_guard_positions", lambda *a, **k: [])
    assert main.notify_portfolio_guard() == 0
