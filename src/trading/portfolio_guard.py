"""Portfolio Guard — EOD, ALERT-ONLY protective scan for human holdings.

Evaluates every non-cron user's `/add`-tracked positions against five
deterministic Stage-1 triggers (hard stop-loss, take-profit, trailing-stop,
model-flip-to-SELL, NO_TRADE regime warning) and renders a per-user Vietnamese
alert card. Stage 1 decides WHETHER an alert fires; the LLM/arbitrator enrichment
(Stage 2, orchestrated in `main.py`) can only add context, never suppress.

HARD CONTRACT (do not relax without returning to PLAN)
──────────────────────────────────────────────────────
  • THIS MODULE MUST NEVER IMPORT `main` — not even lazily. All ML / arbitrator /
    Telegram orchestration is performed by `main.py`'s own functions, which
    import THIS module (one-directional dependency graph → zero circular-import
    risk for the 12 test files that `import main`).
  • ALERT-ONLY. ZERO persistence: no parquet write, no DuckDB table write, no
    `signal_ledger` / `sentiment_entry_paperlog` write, no auto-sell of any user
    position. The one I/O function (`load_guard_positions`) is READ-ONLY.
  • Gemini spend is bounded by the caller (`main._run_guard_for_users`): at most
    one arbitrator batch per EOD run, only when a trigger fired. This module
    itself makes NO network / LLM call. THIS STILL HOLDS after the 13-08-26
    corporate-action work: declared CA events are fetched and cached by
    `src/data/corporate_actions.py` from `main.py`'s EOD path and handed to
    `evaluate_position` as a plain parameter, exactly like `prediction_5d` /
    `regime`. This module only ever consumes them through a PURE resolver.

CORPORATE-ACTION CORRECTION (13-08-26)
──────────────────────────────────────
The 17-07-26 shield detected a corporate-action price gap and downgraded the
hard-stop / trailing-stop WORDING, but never computed the corrected number, so
the alert still fired. VHM's 100% stock dividend (exright 2026-08-06) therefore
produced a live hard-stop + trailing-stop alert at PnL -52.92% / drawdown 53.07%
when the true figures were -5.84% / 6.87% — below BOTH thresholds.

`resolve_lot_prices` now fixes the INPUTS (entry basis and the whole pre-event
close segment, so the trailing-stop peak is rebased too) and the five existing
triggers then run unchanged against the corrected numbers. Nothing is
"suppressed": a lot that is genuinely down past the stop AND paid a stock
dividend still fires. Only the phantom disappears.

The declared factor is used ONLY when three independent conditions agree (a
declared priceable event, an observed price gap, and factor agreement within
tolerance) and no unpriceable rights issue sits in the window — see
`corporate_actions.resolve_adjustment`'s tier A. Every other case falls back to
the pre-existing behaviour.

PURITY LAYERING (mirrors intraday_scanner.py / regime_policy.py precedent)
─────────────────────────────────────────────────────────────────────────
  • `normalize_entry_price_vnd`, `evaluate_position`, `build_guard_alert_card`
    are PURE: no I/O, no network, deterministic on their inputs (the card
    header reads the wall clock only for its display date, matching
    `main.notify_tranche_exits`; nothing else touches the clock).
  • `load_guard_positions` is the ONLY I/O-bearing function — a single READ-ONLY
    DuckDB `SELECT`, degrades to `[]` on any failure (mirrors
    `signal_ledger.list_open`).

VN PRICE-SCALE CONVENTION (the highest-risk defect this feature guards against)
───────────────────────────────────────────────────────────────────────────────
`portfolio.price` (a user's `/add` entry price) has a GENUINELY ambiguous unit:
the bot's own `/add` example (`/add VNE 1000 32.5`) implies THOUSANDS of VND,
while `dashboard/utils/headless.py::_pnl_ratio` assumes absolute VND. This module
resolves it with `normalize_entry_price_vnd`, reusing the SAME `1_000.0` magnitude
heuristic already canonical in `main._VN_PRICE_SCALE_THRESHOLD` (main.py:89) and
`dashboard/utils/headless.py::_pnl_ratio` (headless.py:65-79). `closes_between`
output is UNCONDITIONALLY thousands-of-VND by contract, so the caller multiplies
it by 1000.0 unconditionally to reach absolute VND — no per-value heuristic there.
"""
from __future__ import annotations

import html
import logging
from datetime import date

import duckdb

from config.settings import CONFIG
from src.data import corporate_actions, price_lookup
from src.features.market_regime import regime_label_vi
from src.reports.builders import _MR_SELL_VETO
from src.trading.portfolio_manager import CRON_USER_ID
from src.trading.regime_policy import NO_TRADE_REGIMES

LOGGER = logging.getLogger("quant.portfolio_guard")

# VN stocks are quoted in thousands (e.g. 32.5 = 32,500 VND). A stored entry
# price below this threshold is therefore a thousands-scale value that must be
# multiplied by 1000 to reach absolute VND. IDENTICAL rule + threshold as
# main._VN_PRICE_SCALE_THRESHOLD (main.py:89) and headless._pnl_ratio
# (headless.py:65-79) — a deliberate, cross-referenced convention, not a new
# arbitrary number. Rule is strictly `< threshold` (1000.0 stays 1000.0).
_ENTRY_PRICE_SCALE_THRESHOLD_VND: float = 1_000.0

# Arbitrator decision encoding (shared with quant_agent_arbitrator / accuracy_audit).
_DECISION_LABELS_VI: dict[int, str] = {0: "BÁN/THOÁT", 1: "GIỮ", 2: "MUA"}

# Trigger kinds whose lines carry the mean-reversion "don't sell the bottom"
# caution when the MR sub-model has fired (never take-profit / regime-warning).
_SELL_LEANING_KINDS: frozenset[str] = frozenset({"hard_stop", "trailing_stop", "model_flip"})

# Card header separator — matches main._build_exits_report / notify_tranche_exits style.
_CARD_SEPARATOR: str = "══════════════════════════════"
_CARD_HEADER_TITLE: str = "🛡️ <b>CẢNH BÁO DANH MỤC</b>"
_CARD_FOOTER: str = (
    "<i>Đây là cảnh báo tự động dựa trên quy tắc + mô hình, KHÔNG phải lệnh "
    "giao dịch. Hệ thống KHÔNG tự động bán bất kỳ vị thế nào — quyết định cuối "
    "cùng luôn thuộc về bạn.</i>"
)
# On-demand /guard replies.
GUARD_NO_TRIGGERS_MESSAGE: str = "✅ Danh mục ổn định — không có cảnh báo nào hôm nay."

# Telegram hard limit is 4096; keep a defensive soft budget matching
# notify_tranche_exits.
_CARD_SOFT_LIMIT: int = 3800


# ─────────────────────────────────────────────────────────────────────────────
# PURE LAYER — price scaling, trigger evaluation, card rendering.
# ─────────────────────────────────────────────────────────────────────────────


def normalize_entry_price_vnd(raw_price: float) -> float:
    """Normalize a stored `portfolio.price` to ABSOLUTE VND (pure).

    Applies the canonical VN price-scale heuristic (see module docstring): a
    value `< _ENTRY_PRICE_SCALE_THRESHOLD_VND` (1000.0) is thousands-scale and is
    multiplied by 1000; a value `>= 1000.0` is already absolute VND and passes
    through unchanged. `None`, non-numeric, or `<= 0` inputs degrade to `0.0`
    (the caller treats a `0.0` entry price as "insufficient data", skipping the
    lot rather than dividing by zero).

    Precedent (identical threshold + rule): main._VN_PRICE_SCALE_THRESHOLD
    (main.py:89), dashboard/utils/headless.py::_pnl_ratio (headless.py:65-79).
    """
    if raw_price is None:
        return 0.0
    try:
        val = float(raw_price)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0.0:
        return 0.0
    if val < _ENTRY_PRICE_SCALE_THRESHOLD_VND:
        return val * 1000.0
    return val


def _argmax_is_sell(prediction: list[float] | None) -> bool:
    """True iff the 3-class `[p_down, p_flat, p_up]` argmax is class 0 (SELL/DOWN).

    Defensive: `None` / malformed (len < 3) predictions return False (no flip).
    Ties resolve to the lowest index (numpy-argmax semantics) — a p_down tie
    counts as a flip, the conservative choice for a protective feature.
    """
    if prediction is None or len(prediction) < 3:
        return False
    try:
        vals = [float(p) for p in prediction[:3]]
    except (TypeError, ValueError):
        return False
    return max(range(3), key=lambda i: vals[i]) == 0


def resolve_lot_prices(
    position: dict,
    closes_since_entry_abs: list[float],
    ca_events: list[dict] | None = None,
    *,
    factor_tolerance: float | None = None,
) -> dict:
    """Corporate-action-aware price basis for ONE lot (pure).

    Single source of truth for a lot's entry basis, peak, PnL and drawdown, so
    the trigger logic and the card's own header line can never disagree about
    what the position is worth.

    Args:
        position: the lot dict; reads `entry_price_raw` (falls back to `price`).
        closes_since_entry_abs: ordered daily closes in ABSOLUTE VND from entry
            through today.
        ca_events: declared events for this ticker inside the holding window
            (from `corporate_actions.load_events`). `None`/`[]` ⇒ the resolver
            can only reach tiers B/C, i.e. exactly the pre-13-08-26 behaviour.
        factor_tolerance: overrides `CONFIG.trading.ca_event_factor_tolerance`.

    Returns `{"valid", "entry_effective", "closes_effective", "today_close",
    "peak", "pnl_pct", "drawdown_pct", "ca_tier", "ca_factor", "ca_label_vi",
    "ca_adjusted"}`. `valid` is False (and every number 0.0) when the close
    series is empty or the normalized entry price is non-positive — the same
    two "insufficient data" conditions `evaluate_position` already short-circuits
    on.

    ONLY tier A ("declared, confident") rebases anything. Tier D — a declared
    event with NO observed gap — deliberately changes nothing, because the
    parquet is sometimes ALREADY back-adjusted and adjusting again would corrupt
    it (see `corporate_actions`' module docstring).
    """
    empty = {
        "valid": False, "entry_effective": 0.0, "closes_effective": [],
        "today_close": 0.0, "peak": 0.0, "pnl_pct": 0.0, "drawdown_pct": 0.0,
        "ca_tier": "C", "ca_factor": 1.0, "ca_label_vi": None, "ca_adjusted": False,
    }
    if not closes_since_entry_abs:
        return dict(empty)
    entry_abs = normalize_entry_price_vnd(
        position.get("entry_price_raw", position.get("price"))
    )
    if entry_abs <= 0.0:
        return dict(empty)

    closes = [float(c) for c in closes_since_entry_abs]
    tol = (CONFIG.trading.ca_event_factor_tolerance
           if factor_tolerance is None else factor_tolerance)
    adj = corporate_actions.resolve_adjustment(
        closes, list(ca_events or []), factor_tolerance=tol
    )

    ca_adjusted = adj["tier"] == "A"
    if ca_adjusted:
        closes_eff = corporate_actions.back_adjust_closes(closes, adj["gap_factors"])
        entry_eff = entry_abs * float(adj["factor"])
        # Defensive: a corrupted factor must never produce a zero/negative basis.
        if entry_eff <= 0.0:
            closes_eff, entry_eff, ca_adjusted = closes, entry_abs, False
    else:
        closes_eff, entry_eff = closes, entry_abs

    today_close = closes_eff[-1]
    peak = max(closes_eff)
    return {
        "valid": True,
        "entry_effective": entry_eff,
        "closes_effective": closes_eff,
        "today_close": today_close,
        "peak": peak,
        "pnl_pct": (today_close - entry_eff) / entry_eff,
        "drawdown_pct": (peak - today_close) / peak if peak > 0.0 else 0.0,
        "ca_tier": adj["tier"],
        "ca_factor": float(adj["factor"]),
        "ca_label_vi": adj["label_vi"] if ca_adjusted else None,
        "ca_adjusted": ca_adjusted,
    }


def evaluate_position(
    position: dict,
    closes_since_entry_abs: list[float],
    prediction_5d: list[float] | None,
    prediction_20d: list[float] | None,
    regime: int | None,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    trailing_pct: float,
    ca_events: list[dict] | None = None,
) -> list[dict]:
    """Evaluate the five Stage-1 triggers for ONE lot (pure).

    Args:
        position: the lot dict; reads `entry_price_raw` (falls back to `price`).
        closes_since_entry_abs: ordered daily closes in ABSOLUTE VND from entry
            date through today (the caller has already scaled the thousands-VND
            `closes_between` output by 1000.0). Empty ⇒ "insufficient data".
        prediction_5d / prediction_20d: `[p_down, p_flat, p_up]` for this ticker
            at each horizon, or None when the model produced no row for it.
        regime: latest cached `market_regime` int (0-7) for this ticker, or None.
        stop_loss_pct / take_profit_pct / trailing_pct: ratios (e.g. -0.07 /
            0.15 / 0.08); rendered as percentages in the message copy.
        ca_events: declared corporate-action events for this ticker inside the
            holding window (from `corporate_actions.load_events`). Defaults to
            `None`, which reproduces the pre-13-08-26 behaviour EXACTLY — every
            existing call site and test is therefore unaffected.

    Returns a list of `{"kind", "message_vi", "ca_gap_downgraded", "ca_adjusted",
    "ca_label_vi"}` dicts in the fixed order hard_stop, take_profit,
    trailing_stop, model_flip, regime_warning. Degrades to `[]` immediately when
    the close series is empty or the normalized entry price is non-positive.

    Two mutually exclusive corporate-action paths:
      • AUTO-CORRECTED (tier A — declared event + observed gap + factor
        agreement): the numbers below are already right, so the wording is
        CONFIDENT and carries a line naming the event. `ca_adjusted=True`.
      • SHIELDED (tier B — a gap with no confident declared match): unchanged
        17-07-26 behaviour — the hard-stop and trailing-stop wording is
        downgraded to a disclaimer (take-profit shares the pnl exposure but is
        left confident, exactly as approved) and the trigger STILL fires.
    """
    prices = resolve_lot_prices(position, closes_since_entry_abs, ca_events)
    if not prices["valid"]:
        return []

    peak_abs = prices["peak"]
    pnl_pct = prices["pnl_pct"]
    drawdown_pct = prices["drawdown_pct"]
    ca_adjusted = prices["ca_adjusted"]
    ca_label = prices["ca_label_vi"]
    # Only the UNCORRECTED path needs the disclaimer. Once the numbers are
    # rebased on a declared ratio they are trusted and shown, not hedged.
    ca_gap = (not ca_adjusted) and price_lookup.has_ca_gap(closes_since_entry_abs)
    ca_note = (
        f"\n<i>ℹ️ Đã tự động điều chỉnh giá theo {html.escape(ca_label)} — "
        f"số liệu trên là PnL thực tế sau điều chỉnh.</i>"
        if ca_adjusted and ca_label else ""
    )

    def _mark(kind: str, msg: str, downgraded: bool) -> dict:
        return {"kind": kind, "message_vi": msg, "ca_gap_downgraded": downgraded,
                "ca_adjusted": ca_adjusted, "ca_label_vi": ca_label}

    triggers: list[dict] = []

    # 1. Hard stop-loss (CA-gap shielded).
    if pnl_pct <= stop_loss_pct:
        if ca_gap:
            msg = (
                f"⚠️ PnL {pnl_pct * 100:+.1f}% — có thể do hành động doanh nghiệp "
                f"(chia cổ tức/cổ phiếu thưởng/phát hành thêm), KHÔNG chắc là lỗ "
                f"thật. Vui lòng tự kiểm tra giá điều chỉnh — đây KHÔNG phải tín "
                f"hiệu cắt lỗ chắc chắn."
            )
        else:
            msg = (
                f"🔴 <b>CẮT LỖ</b>: PnL hiện tại {pnl_pct * 100:+.1f}% "
                f"(đã vượt ngưỡng cắt lỗ {stop_loss_pct * 100:.0f}%){ca_note}"
            )
        triggers.append(_mark("hard_stop", msg, ca_gap))

    # 2. Take-profit (info; NOT CA-gap shielded, per approved scope).
    if pnl_pct >= take_profit_pct:
        msg = (
            f"🟢 <b>CHỐT LỜI</b>: PnL hiện tại {pnl_pct * 100:+.1f}% "
            f"(đã vượt ngưỡng chốt lời {take_profit_pct * 100:.0f}%){ca_note}"
        )
        triggers.append(_mark("take_profit", msg, False))

    # 3. Trailing stop (CA-gap shielded).
    if peak_abs > 0.0 and drawdown_pct >= trailing_pct:
        if ca_gap:
            msg = (
                f"⚠️ Giảm {drawdown_pct * 100:.1f}% từ đỉnh — có thể do hành động "
                f"doanh nghiệp, KHÔNG chắc là lỗ thật. Vui lòng tự kiểm tra — đây "
                f"KHÔNG phải tín hiệu cắt lỗ chắc chắn."
            )
        else:
            msg = (
                f"🟠 <b>TRAILING STOP</b>: giá đã giảm {drawdown_pct * 100:.1f}% "
                f"từ đỉnh {peak_abs:,.0f} VND kể từ khi mua "
                f"(ngưỡng {trailing_pct * 100:.0f}%){ca_note}"
            )
        triggers.append(_mark("trailing_stop", msg, ca_gap))

    # 4. Model flip — OR across T+5 / T+20; one line per flipped horizon.
    flip_lines: list[str] = []
    if _argmax_is_sell(prediction_5d):
        flip_lines.append(
            f"🔻 <b>MÔ HÌNH ĐỔI TÍN HIỆU</b>: T+5 → BÁN "
            f"(P(Tăng)={float(prediction_5d[2]) * 100:.0f}%)"
        )
    if _argmax_is_sell(prediction_20d):
        flip_lines.append(
            f"🔻 <b>MÔ HÌNH ĐỔI TÍN HIỆU</b>: T+20 → BÁN "
            f"(P(Tăng)={float(prediction_20d[2]) * 100:.0f}%)"
        )
    if flip_lines:
        triggers.append(_mark("model_flip", "\n".join(flip_lines), False))

    # 5. Regime warning — NO_TRADE regimes {0, 7}.
    if regime is not None and int(regime) in NO_TRADE_REGIMES:
        msg = (
            f"🌐 <b>CẢNH BÁO PHA THỊ TRƯỜNG</b>: đang ở pha "
            f"{regime_label_vi(regime)} (Regime {int(regime)}) — rủi ro hệ thống "
            f"cao, hạn chế mở/giữ vị thế mới."
        )
        triggers.append(_mark("regime_warning", msg, False))

    return triggers


def _entry_date_str(entry_date: object) -> str:
    """Render a lot's entry date as dd/mm/yyyy; tolerant of date or str input."""
    if hasattr(entry_date, "strftime"):
        return entry_date.strftime("%d/%m/%Y")  # type: ignore[attr-defined]
    return str(entry_date)


def _enrichment_line(
    ticker: str, enrichment: tuple[dict, dict] | None
) -> str | None:
    """Build the optional arbitrator/sentiment context line for one ticker.

    Uses the SAME "Trọng tài tin tức" label the buy card already uses
    (telegram_alerter.py). Returns None when Stage 2 did not run or produced no
    data for this ticker (the "quant-only" card path).
    """
    if not enrichment:
        return None
    final_decisions, all_sentiments = enrichment
    decision = final_decisions.get(ticker) if final_decisions else None
    sentiment = all_sentiments.get(ticker, {}) if all_sentiments else {}
    if decision is None and not sentiment:
        return None
    parts: list[str] = ["🧠 <b>Trọng tài tin tức:</b>"]
    if decision is not None:
        parts.append(html.escape(_DECISION_LABELS_VI.get(int(decision), "N/A")))
    try:
        score = float(sentiment.get("sentiment_score"))
        parts.append(f"(tâm lý {score:+.2f})")
    except (TypeError, ValueError):
        pass
    line = " ".join(parts)
    reasoning = str(sentiment.get("reasoning_vi", "")).strip()
    if reasoning:
        if len(reasoning) > 300:
            reasoning = reasoning[:297] + "…"
        line += f"\n<i>{html.escape(reasoning)}</i>"
    return line


def _render_lot_block(
    lot: dict, enrichment: tuple[dict, dict] | None, mr_scores: dict
) -> str:
    """Render one fired-trigger lot into an HTML block (pure)."""
    ticker = str(lot.get("ticker", "N/A")).upper()
    volume = lot.get("volume")
    entry_norm = float(lot.get("entry_price_norm", 0.0) or 0.0)
    current = float(lot.get("current_price", 0.0) or 0.0)
    pnl_pct = float(lot.get("pnl_pct", 0.0) or 0.0)
    triggers: list[dict] = list(lot.get("triggers", []))

    try:
        vol_str = f"{int(volume):,} cp" if volume is not None else "—"
    except (TypeError, ValueError):
        vol_str = "—"

    lines: list[str] = [
        f"<b>{html.escape(ticker)}</b> — {vol_str} @ {entry_norm:,.0f} VND "
        f"(vào {_entry_date_str(lot.get('entry_date'))})",
        f"Giá hiện tại: {current:,.0f} VND | PnL: {pnl_pct * 100:+.1f}%",
    ]
    for trig in triggers:
        lines.append(str(trig.get("message_vi", "")))

    # MR knife-catch caution — once per lot, only for SELL-leaning triggers.
    mr_fired = bool(mr_scores.get(ticker, {}).get("fired")) if mr_scores else False
    has_sell_leaning = any(t.get("kind") in _SELL_LEANING_KINDS for t in triggers)
    if mr_fired and has_sell_leaning:
        lines.append(_MR_SELL_VETO)

    enrich = _enrichment_line(ticker, enrichment)
    if enrich:
        lines.append(enrich)

    return "\n".join(lines)


def build_guard_alert_card(
    ticker_lots: list[dict],
    enrichment: tuple[dict, dict] | None,
    mr_scores: dict,
) -> str:
    """Render ONE user's guard alert card (pure).

    Args:
        ticker_lots: this user's fired-trigger lots, each
            `{"ticker", "entry_date", "entry_price_norm", "current_price",
              "pnl_pct", "volume", "triggers": [...]}`.
        enrichment: `(final_decisions, all_sentiments)` from the single Stage-2
            arbitrator batch, or `None`/empty when Stage 2 didn't run
            (quant-only card).
        mr_scores: `{ticker: {"fired": bool, ...}}` from `mr_score_tickers`.

    Header/footer/trigger copy is the exact contract from the plan's Functional
    Requirements table; `_MR_SELL_VETO` is appended verbatim per the stated rule.
    Applies the same 3800-char soft-truncate-with-overflow-notice pattern as
    `main.notify_tranche_exits`, dropping whole lot blocks (never mid-block) so
    the card always delivers as valid HTML under the 4096-char Telegram limit.
    """
    header = f"{_CARD_HEADER_TITLE}\n{date.today().strftime('%d/%m/%Y')}\n{_CARD_SEPARATOR}"
    blocks = [_render_lot_block(lot, enrichment, mr_scores) for lot in ticker_lots]

    # Block-based soft truncation (mirrors notify_tranche_exits' line-based one).
    budget = _CARD_SOFT_LIMIT - len(header) - len(_CARD_FOOTER) - 60  # overflow head-room
    kept: list[str] = []
    running = 0
    for blk in blocks:
        if running + len(blk) + 2 > budget:
            break
        kept.append(blk)
        running += len(blk) + 2
    dropped = len(blocks) - len(kept)

    parts = [header, *kept]
    if dropped > 0:
        parts.append(f"... và <b>{dropped}</b> vị thế khác (rút gọn).")
    parts.append(_CARD_FOOTER)
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# I/O LAYER — a single READ-ONLY DuckDB SELECT. Never raises (degrades to []).
# ─────────────────────────────────────────────────────────────────────────────


def load_guard_positions(
    db_path: str | None = None, user_id: str | None = None
) -> list[dict]:
    """Load every NON-CRON portfolio lot (READ-ONLY). Never raises.

    Returns one dict per row: `{"user_id", "ticker", "volume", "price",
    "entry_date"}` (`entry_date` cast to a DATE). The automated cron book
    (`user_id == CRON_USER_ID`) is ALWAYS excluded — this feature protects human
    `/add` holdings only. Pass `user_id` to scope to a single user (the /guard
    on-demand path). Degrades to `[]` on any exception (missing DB, locked file,
    absent table), mirroring `signal_ledger.list_open`.
    """
    query = (
        "SELECT user_id, ticker, volume, price, CAST(added_at AS DATE) AS entry_date "
        "FROM portfolio WHERE user_id != ?"
    )
    params: list[object] = [CRON_USER_ID]
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " ORDER BY user_id, ticker, added_at"

    try:
        with duckdb.connect(db_path or str(CONFIG.paths.duckdb_path)) as conn:
            rows = conn.execute(query, params).fetchall()
    except Exception as exc:  # noqa: BLE001 — missing/locked DB, absent table
        LOGGER.warning("[portfolio_guard] load_guard_positions failed: %s — degrading to [].", exc)
        return []

    return [
        {
            "user_id": str(r[0]),
            "ticker": str(r[1]).upper() if r[1] is not None else "",
            "volume": r[2],
            "price": float(r[3]) if r[3] is not None else 0.0,
            "entry_date": r[4],
        }
        for r in rows
        if r and r[1] is not None
    ]


__all__ = [
    "normalize_entry_price_vnd",
    "resolve_lot_prices",
    "evaluate_position",
    "build_guard_alert_card",
    "load_guard_positions",
    "GUARD_NO_TRIGGERS_MESSAGE",
]
