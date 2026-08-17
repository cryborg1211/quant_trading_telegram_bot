"""Declared corporate-action events from vnstock's `Company.events()`, cached.

WHY THIS EXISTS
───────────────
`price_lookup.has_ca_gap` / `derive_ca_adjustment_factor` infer a corporate
action from the PRICE SERIES ALONE (any >10% single-session move). That is cheap
and needs no network, but it is structurally blind in four ways:

  1. Sub-threshold events are invisible. A stock dividend of <=11.1% produces a
     gap of >=0.9 and never trips the 10% detector (HPG's declared 10% stock
     dividend, exright 2026-05-25, is exactly this shape).
  2. A corporate action cannot be told apart from a genuine limit move. Scanning
     every shard since 2026-01-01 for moves in the 7-10% band returns a list
     dominated by EXACT +-7.00% limit sessions (PIT 7.00->7.49, GVR
     30.00->32.10, MCH 140.00->149.80). Lowering the threshold to catch (1)
     starts "correcting" real losses into fake small ones.
  3. The observed ratio absorbs genuine ex-date trading into the "correction"
     (VHM: observed 0.503922 vs declared exactly 0.500000).
  4. It cannot NAME the event, so the alert can only say "might be a corporate
     action, go check yourself".

Motivating incident (13-08-26): the EOD portfolio guard alerted a hard stop +
trailing stop on the live VHM holding at PnL -52.92% / drawdown 53.07%. VHM had
paid a 100% stock dividend (exright 2026-08-06); the real numbers were -5.84% /
6.87%, and NEITHER threshold was breached.

LIVE API FINDINGS (measured 13-08-26 — do not re-derive these from docs)
────────────────────────────────────────────────────────────────────────
  * `Company(source='VCI', symbol=T).events()` works; 0.37-1.09s per call.
    `source='KBS'` returns an EMPTY (0,0) frame — there is NO fallback source.
  * An unknown symbol RAISES `ValueError`; an EMPTY symbol string returns 50
    rows of garbage. Validate the ticker before calling.
  * Every response is capped at 50 rows, so only the ~50 most recent events per
    ticker are reachable. Fine for guard windows (weeks), not for deep history.
  * All date columns arrive as `str` in "2026-08-06T00:00:00" form;
    `exercise_ratio` / `value_per_share` are float64.

  * `exercise_ratio` IS POLYSEMOUS — the VHM case does NOT generalize:
        event_code ISS -> new shares per existing share (VHM 1.0 = 100%)
        event_code DIV -> cash per share / 10,000 (par). Confirmed on every row
                          with a populated value_per_share: 450/0.045,
                          6000/0.6, 1000/0.1, 500/0.05 == exactly 10000.
    Within ISS the `event_title_vi` prefix carries the real taxonomy and only
    TWO sub-types are proportional entitlements with a computable price factor
    (stock dividend, bonus shares). Rights issues ARE price-adjusting but are
    UNPRICEABLE here: they need the subscription price, and `value_per_share` is
    NaN on every ISS row. `exercise_ratio == 0.0` appears on many older ISS rows
    ("declared, ratio unknown") and must NOT become 1/(1+0) == 1.0.

  * DECLARED DATES DO NOT ALIGN WITH THE PARQUET. Across 21 matched gaps since
    2026-04-01, (gap_date - exright_date) ranges -6..0 CALENDAR DAYS and is
    never positive (KLB -1, PVD -4, PET -6). Matching on date equality would
    fail ~70% of real cases. Factor agreement, by contrast, is tight (median
    |err| ~3%). ==> MATCH ON FACTOR AGREEMENT INSIDE A DATE WINDOW, NOT ON DATE.
    A useful side effect: gap POSITIONS come from the ordered close list itself
    (exactly as `derive_ca_adjustment_factor` already does), so this module
    needs no per-bar dates and `price_lookup.py` needs no change at all.

  * THE PARQUET IS SOMETIMES ALREADY BACK-ADJUSTED. VHM 2026-08-06 (0.5039) and
    MBB 2026-08-11 (0.8392) sit RAW, but VCB 2025-03-12 (1.0302), KLB
    2025-09-24 (1.0130), MBB 2025-08-13 (1.0609) and HPG 2025-06-26 (1.0058)
    show no gap at all. Applying a declared factor UNCONDITIONALLY would corrupt
    every already-adjusted series ==> the correction is gated on an OBSERVED
    gap, never on the declaration alone (tier D below).

FOUR-TIER RESOLUTION (`resolve_adjustment`)
───────────────────────────────────────────
  A  DECLARED   observed gap AND a priceable declared bundle matching it within
                `factor_tolerance` AND no unpriceable RIGHTS event in the window
                -> use the DECLARED factor, name the event.
  B  OBSERVED   observed gap, no confident declared match -> self-referential
                factor (identical to today's behaviour).
  C  NONE       no gap, no declared event -> 1.0.
  D  ADJUSTED   declared event but NO observed gap -> 1.0, change nothing.
                (the double-adjustment guard)

The RIGHTS VETO is not theoretical: MBB 2026-08-11 bundles a 15% stock dividend
with a 10% rights issue. Stock-dividend-only theory gives 0.8696 against an
observed 0.8392 — a -3.49% error that would PASS a 10% tolerance and be promoted
to "confident" while silently omitting the rights leg.

PURITY LAYERING (mirrors portfolio_guard.py / intraday_scanner.py)
──────────────────────────────────────────────────────────────────
`classify_event`, `theoretical_price_factor`, `bundle_factor`,
`resolve_adjustment` and `back_adjust_closes` are PURE. `ensure_tables`,
`load_events`, `tickers_needing_refresh` and `refresh_events` are the ONLY
I/O-bearing functions; every one degrades to []/0 rather than raising, so a
vnstock outage can never break the EOD pipeline.

`refresh_events` is the only function that touches the network. It is called
from `main.py`'s EOD path — NEVER from `portfolio_guard.py`, whose hard contract
states it makes no network call.
"""
from __future__ import annotations

import logging
import math
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import duckdb  # type: ignore[import-untyped]

from config.settings import CONFIG

LOGGER = logging.getLogger("quant.corporate_actions")

# Module-level import so tests can monkeypatch `corporate_actions.Company`
# (same optional-import shape as sentiment_crawler's gnews/genai handling).
try:  # pragma: no cover - exercised implicitly
    from vnstock import Company  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    Company = None  # type: ignore[assignment]

EVENTS_TABLE = "corporate_action_events"
FETCH_LOG_TABLE = "corporate_action_fetch_log"

# vnstock's `Company` accepts only these; KBS was measured to return an empty
# frame, so VCI is effectively the sole source.
_SOURCE = "VCI"

# Same ticker shape price_lookup validates before path interpolation. An empty
# symbol makes events() return 50 rows of unrelated garbage, so this is a
# correctness guard, not just hygiene.
_TICKER_RE = re.compile(r"[A-Z0-9]{1,12}")

# ── Event taxonomy ──────────────────────────────────────────────────────────
KIND_STOCK_DIVIDEND = "stock_dividend"
KIND_BONUS = "bonus"
KIND_RIGHTS = "rights"
KIND_ESOP = "esop"
KIND_PRIVATE = "private"
KIND_CONVERTIBLE = "convertible"
KIND_CASH_DIVIDEND = "cash_dividend"
KIND_OTHER = "other"

# Kinds that scale the quoted price by a computable factor.
PRICEABLE_KINDS: frozenset[str] = frozenset({KIND_STOCK_DIVIDEND, KIND_BONUS})

# Title fragments, checked in this order. Vietnamese diacritics are matched
# exactly as vnstock returns them.
_TITLE_KINDS: tuple[tuple[str, str], ...] = (
    ("Trả Cổ tức bằng Cổ phiếu", KIND_STOCK_DIVIDEND),
    ("Cổ phiếu thưởng", KIND_BONUS),
    ("Quyền mua CP cho Cổ đông hiện hữu", KIND_RIGHTS),
    ("Phát hành cho CBCNV", KIND_ESOP),
    ("Phát hành riêng lẻ", KIND_PRIVATE),
    ("Chuyển từ trái phiếu chuyển đổi", KIND_CONVERTIBLE),
    ("Phát hành để sáp nhập", KIND_OTHER),
)

# Declared events are matched to observed gaps inside this window around
# `exright_date`. Derived from measurement (offset -6..0 days across 21 matched
# gaps), with margin on both sides. Deliberately module constants, NOT config:
# these are empirical properties of the data source, not user preferences.
_MATCH_WINDOW_DAYS_BEFORE = 10
_MATCH_WINDOW_DAYS_AFTER = 3

# VN par value — the denominator DIV rows normalise cash dividends against.
# Retained for documentation/analysis; cash dividends are out of scope for the
# price-factor path (income, not a scale change).
PAR_VALUE_VND = 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# PURE LAYER — taxonomy, factors, tiered resolution, back-adjustment.
# ─────────────────────────────────────────────────────────────────────────────


def classify_event(event_code: str | None, event_title_vi: str | None) -> str:
    """Map a raw (`event_code`, `event_title_vi`) pair onto a `KIND_*` constant.

    `DIV` is always a cash dividend regardless of title. Only `ISS` rows carry
    the share-issuance taxonomy; every other code (DDIND/DDINS/DDRP insider
    trading, AGME/EGME meetings, AIS/NLIS/SUSP/MOVE/OTHE) is `KIND_OTHER` —
    451 of 727 sampled rows are insider-trading noise.
    """
    code = str(event_code or "").strip().upper()
    if code == "DIV":
        return KIND_CASH_DIVIDEND
    if code != "ISS":
        return KIND_OTHER
    title = str(event_title_vi or "")
    for fragment, kind in _TITLE_KINDS:
        if fragment in title:
            return kind
    return KIND_OTHER


def theoretical_price_factor(kind: str, exercise_ratio: float | None) -> float | None:
    """Price factor `1/(1+r)` for a priceable entitlement, else `None`.

    `r` is new shares per existing share. Returns `None` for every
    non-priceable kind (rights need a subscription price that the payload does
    not carry; ESOP / private placement / convertible conversion create no
    proportional entitlement) and for a missing, NaN or non-positive ratio —
    `exercise_ratio == 0.0` is a "declared, ratio unknown" marker on many older
    ISS rows and must NOT collapse to `1/(1+0) == 1.0`.
    """
    if kind not in PRICEABLE_KINDS:
        return None
    if exercise_ratio is None:
        return None
    try:
        r = float(exercise_ratio)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(r) or r <= 0.0:
        return None
    return 1.0 / (1.0 + r)


def _event_kind(event: dict) -> str:
    """Prefer a pre-computed `kind` (cache rows carry one); else classify."""
    kind = event.get("kind")
    if kind:
        return str(kind)
    return classify_event(event.get("event_code"), event.get("event_title_vi"))


def has_unpriceable_rights(events: list[dict]) -> bool:
    """True when any event is a rights issue with a real (positive) ratio.

    Such an event moves the price by an amount this module cannot compute, so
    any window containing one is disqualified from tier A (see the MBB
    2026-08-11 case in the module docstring).
    """
    for ev in events or []:
        if _event_kind(ev) != KIND_RIGHTS:
            continue
        try:
            r = float(ev.get("exercise_ratio") or 0.0)
        except (TypeError, ValueError):
            r = 0.0
        if math.isfinite(r) and r > 0.0:
            return True
    return False


def bundle_factor(events: list[dict]) -> float | None:
    """Combined price factor for entitlements sharing one ex-right date.

    Vietnamese issuers routinely bundle several entitlements onto a single
    ex-date (GEX 2026-05-05 = 20% bonus + 25% stock dividend; POW 2025-12-10 =
    4% stock dividend + 15% bonus + 12% rights), so the ratios ADD before the
    reciprocal: `1 / (1 + sum(r_i))`.

    Returns `None` when the bundle contains an unpriceable rights issue (the
    computable legs alone would understate the real move) or when no priceable
    leg is present. Non-price-adjusting kinds (ESOP, private placement,
    convertible conversion, cash dividend, other) are simply ignored — they do
    not move the quoted price.
    """
    if not events:
        return None
    if has_unpriceable_rights(events):
        return None
    total = 0.0
    found = False
    for ev in events:
        kind = _event_kind(ev)
        if kind not in PRICEABLE_KINDS:
            continue
        try:
            r = float(ev.get("exercise_ratio") or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r) or r <= 0.0:
            continue
        total += r
        found = True
    if not found:
        return None
    return 1.0 / (1.0 + total)


def find_gaps(closes: list[float], max_session_move: float = 0.10) -> list[tuple[int, float]]:
    """`(index_of_post_gap_bar, ratio)` for every corporate-action-sized move.

    Identical detection rule and `prev > 0` guard as `price_lookup.has_ca_gap` /
    `derive_ca_adjustment_factor`, so the three can never disagree about which
    sessions counted as a reset.
    """
    out: list[tuple[int, float]] = []
    for i, (prev, cur) in enumerate(zip(closes, closes[1:])):
        if prev > 0 and abs(cur / prev - 1.0) > max_session_move:
            out.append((i + 1, cur / prev))
    return out


def _event_label_vi(event: dict) -> str:
    """Short Vietnamese description: title (minus the boilerplate prefix) + GDKHQ."""
    title = str(event.get("event_title_vi") or "").strip()
    for prefix in ("Phát hành cổ phiếu - ",):
        if title.startswith(prefix):
            title = title[len(prefix):]
    xd = event.get("exright_date")
    if isinstance(xd, datetime):
        xd = xd.date()
    if isinstance(xd, date):
        return f"{title} (GDKHQ {xd.strftime('%d/%m/%Y')})" if title else \
            f"GDKHQ {xd.strftime('%d/%m/%Y')}"
    return title or "hành động doanh nghiệp"


def _group_by_exright(events: list[dict]) -> list[tuple[Any, list[dict]]]:
    """Group events by `exright_date`, oldest first. Undated events are dropped
    (an entitlement that cannot be located in time cannot be matched)."""
    buckets: dict[Any, list[dict]] = {}
    for ev in events or []:
        xd = ev.get("exright_date")
        if isinstance(xd, datetime):
            xd = xd.date()
        if not isinstance(xd, date):
            continue
        buckets.setdefault(xd, []).append(ev)
    return sorted(buckets.items(), key=lambda kv: kv[0])


def resolve_adjustment(
    closes: list[float],
    events: list[dict],
    *,
    max_session_move: float = 0.10,
    factor_tolerance: float = 0.10,
) -> dict:
    """Resolve the corporate-action correction for one close series (pure).

    Args:
        closes: ordered closes across the window, in a CONSISTENT unit.
        events: declared events already filtered to the window (see
            `load_events`). Pass `[]` to get the pure self-referential answer.
        max_session_move: gap-detection threshold, shared with `has_ca_gap`.
        factor_tolerance: max `abs(observed/declared - 1)` for a declared bundle
            to be trusted over the observed ratio.

    Returns `{"tier", "factor", "gap_factors", "label_vi", "matched",
    "gap_ratios"}`:
        tier         "A" declared | "B" observed | "C" none | "D" already adjusted
        factor       TOTAL factor to apply to the entry basis (1.0 for C/D)
        gap_factors  `[(index, factor)]` per gap, for `back_adjust_closes`
        label_vi     Vietnamese event description on tier A, else None
        matched      the declared events backing a tier-A verdict

    Tier A requires THREE independent conditions to agree — a declared priceable
    bundle, an observed gap, and factor agreement within tolerance — plus the
    absence of any unpriceable rights issue in the window. Anything less falls
    back to tier B, which reproduces today's behaviour exactly.
    """
    gaps = find_gaps(closes, max_session_move)
    priceable_declared = [e for e in (events or []) if _event_kind(e) in PRICEABLE_KINDS]

    if not gaps:
        # No observed gap. If something WAS declared, the series is already
        # back-adjusted (VCB/KLB/MBB/HPG shapes) — adjusting again would corrupt it.
        tier = "D" if priceable_declared else "C"
        return {"tier": tier, "factor": 1.0, "gap_factors": [], "label_vi": None,
                "matched": [], "gap_ratios": gaps}

    self_factor = 1.0
    for _, ratio in gaps:
        self_factor *= ratio
    fallback = {"tier": "B", "factor": self_factor,
                "gap_factors": list(gaps), "label_vi": None,
                "matched": [], "gap_ratios": gaps}

    if not priceable_declared or has_unpriceable_rights(events):
        # No declared basis, or a rights leg this module cannot price (MBB
        # 2026-08-11) — the observed ratio is the only trustworthy number.
        return fallback

    # Pair each observed gap with a declared bundle by FACTOR AGREEMENT, not by
    # date (measured offsets run -6..0 days). Each bundle is consumed at most once.
    groups = _group_by_exright(events)
    used: set[int] = set()
    matched: list[dict] = []
    gap_factors: list[tuple[int, float]] = []
    for idx, ratio in gaps:
        best: tuple[float, int, float] | None = None
        for gi, (_, bundle) in enumerate(groups):
            if gi in used:
                continue
            declared = bundle_factor(bundle)
            if declared is None or declared <= 0.0:
                continue
            err = abs(ratio / declared - 1.0)
            if err <= factor_tolerance and (best is None or err < best[0]):
                best = (err, gi, declared)
        if best is None:
            return fallback  # any unexplained gap disqualifies the whole window
        _, gi, declared = best
        used.add(gi)
        gap_factors.append((idx, declared))
        matched.extend(groups[gi][1])

    total = 1.0
    for _, f in gap_factors:
        total *= f
    labels = [_event_label_vi(e) for e in matched if _event_kind(e) in PRICEABLE_KINDS]
    return {"tier": "A", "factor": total, "gap_factors": gap_factors,
            "label_vi": "; ".join(dict.fromkeys(labels)) or None,
            "matched": matched, "gap_ratios": gaps}


def back_adjust_closes(
    closes: list[float], gap_factors: list[tuple[int, float]]
) -> list[float]:
    """Rebase pre-event closes onto the post-event price scale (pure).

    Standard back-adjustment: a close at index `i` is multiplied by the product
    of the factors of every event occurring AFTER it, so the whole series ends
    up on today's scale. Adjusting only the entry price would leave the
    trailing-stop PEAK on the old scale and keep reporting a ~53% drawdown.

    `gap_factors` is `[(post_gap_index, factor)]` as returned by
    `resolve_adjustment`. An empty list returns the input unchanged.
    """
    if not gap_factors:
        return list(closes)
    out = list(closes)
    # Walk gaps latest -> earliest, accumulating. Each pass rewrites the prefix
    # strictly BEFORE that gap with the running product, so a close ends up
    # scaled by exactly the events that follow it.
    running = 1.0
    for idx, factor in sorted(gap_factors, key=lambda t: t[0], reverse=True):
        running *= factor
        for i in range(0, min(idx, len(out))):
            out[i] = closes[i] * running
    return out


# ─────────────────────────────────────────────────────────────────────────────
# I/O LAYER — DuckDB cache + the single network call. Never raises.
# ─────────────────────────────────────────────────────────────────────────────


def _connect(db_path: str | None) -> Any:
    return duckdb.connect(db_path or str(CONFIG.paths.duckdb_path))


def ensure_tables(conn: Any) -> None:
    """Create the event cache + fetch log if absent (idempotent).

    The fetch log exists so "fetched, legitimately zero events" is
    distinguishable from "never fetched" — without it a quiet ticker would be
    re-fetched on every single EOD run.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EVENTS_TABLE} (
            ticker          VARCHAR NOT NULL,
            event_id        VARCHAR,
            event_code      VARCHAR,
            category        VARCHAR,
            event_title_vi  VARCHAR,
            exercise_ratio  DOUBLE,
            value_per_share DOUBLE,
            exright_date    DATE,
            record_date     DATE,
            public_date     DATE,
            kind            VARCHAR,
            price_factor    DOUBLE,
            fetched_at      TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FETCH_LOG_TABLE} (
            ticker          VARCHAR NOT NULL,
            last_fetched_at TIMESTAMP,
            event_count     INTEGER,
            status          VARCHAR
        )
        """
    )


def _valid_ticker(ticker: Any) -> str | None:
    t = str(ticker or "").upper().strip()
    return t if _TICKER_RE.fullmatch(t) else None


def _parse_date(value: Any) -> date | None:
    """vnstock hands back '2026-08-06T00:00:00' strings, NaN floats or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_events(
    ticker: str,
    start: Any,
    end: Any,
    db_path: str | None = None,
) -> list[dict]:
    """Cached events whose `exright_date` falls in the matching window (READ-ONLY).

    The window is `[start - 10d, end + 3d]` — declared ex-right dates sit up to
    6 days AFTER the observed parquet gap (see the module docstring), so an
    exact-range filter would miss the very events it is meant to find. Rows with
    a NULL `exright_date` are excluded: an entitlement that cannot be located in
    time can neither be matched nor contaminate a window.

    Never raises — a missing table / unreadable DB degrades to `[]`, which
    resolves to tier B (today's behaviour).
    """
    t = _valid_ticker(ticker)
    if t is None:
        return []
    start_d, end_d = _parse_date(start), _parse_date(end)
    if start_d is None or end_d is None:
        return []
    lo = start_d - timedelta(days=_MATCH_WINDOW_DAYS_BEFORE)
    hi = end_d + timedelta(days=_MATCH_WINDOW_DAYS_AFTER)
    try:
        with _connect(db_path) as conn:
            ensure_tables(conn)
            rows = conn.execute(
                f"SELECT ticker, event_id, event_code, category, event_title_vi, "
                f"exercise_ratio, value_per_share, exright_date, record_date, "
                f"public_date, kind, price_factor FROM {EVENTS_TABLE} "
                "WHERE ticker = ? AND exright_date IS NOT NULL "
                "AND exright_date >= ? AND exright_date <= ? "
                "ORDER BY exright_date",
                [t, lo, hi],
            ).fetchall()
    except Exception:  # noqa: BLE001 — cache read must never break the guard
        LOGGER.warning("[CA] load_events(%s) failed — degrading to [].", t, exc_info=True)
        return []
    return [
        {
            "ticker": r[0], "event_id": r[1], "event_code": r[2], "category": r[3],
            "event_title_vi": r[4], "exercise_ratio": r[5], "value_per_share": r[6],
            "exright_date": r[7], "record_date": r[8], "public_date": r[9],
            "kind": r[10], "price_factor": r[11],
        }
        for r in rows
    ]


def tickers_needing_refresh(
    tickers: list[str],
    db_path: str | None = None,
    refresh_days: int | None = None,
    limit: int | None = None,
) -> list[str]:
    """Subset of `tickers` never fetched or last fetched > `refresh_days` ago.

    Bounded by `limit` (default `ca_event_max_refresh_per_run`) so a cold cache
    cannot turn the EOD run into a long throttled crawl. Never raises: on a read
    failure it returns the capped input, i.e. it errs toward refreshing.
    """
    days = CONFIG.trading.ca_event_refresh_days if refresh_days is None else refresh_days
    cap = CONFIG.trading.ca_event_max_refresh_per_run if limit is None else limit
    wanted = [t for t in (_valid_ticker(x) for x in tickers) if t]
    wanted = sorted(dict.fromkeys(wanted))
    if not wanted:
        return []
    cutoff = datetime.now() - timedelta(days=max(0, int(days)))
    try:
        with _connect(db_path) as conn:
            ensure_tables(conn)
            fresh = {
                str(r[0]).upper()
                for r in conn.execute(
                    f"SELECT ticker FROM {FETCH_LOG_TABLE} "
                    "WHERE last_fetched_at IS NOT NULL AND last_fetched_at >= ?",
                    [cutoff],
                ).fetchall()
            }
    except Exception:  # noqa: BLE001
        LOGGER.warning("[CA] tickers_needing_refresh read failed — refreshing all.",
                       exc_info=True)
        fresh = set()
    stale = [t for t in wanted if t not in fresh]
    return stale[: max(0, int(cap))]


def _rows_from_events_frame(ticker: str, frame: Any) -> list[tuple]:
    """Map a vnstock `events()` DataFrame onto `corporate_action_events` rows."""
    rows: list[tuple] = []
    for record in frame.to_dict("records"):
        code = record.get("event_code")
        title = record.get("event_title_vi")
        kind = classify_event(code, title)
        ratio = _parse_float(record.get("exercise_ratio"))
        rows.append((
            ticker,
            None if record.get("id") is None else str(record.get("id")),
            None if code is None else str(code),
            None if record.get("category") is None else str(record.get("category")),
            None if title is None else str(title),
            ratio,
            _parse_float(record.get("value_per_share")),
            _parse_date(record.get("exright_date")),
            _parse_date(record.get("record_date")),
            _parse_date(record.get("public_date")),
            kind,
            theoretical_price_factor(kind, ratio),
        ))
    return rows


def _write_fetch_log(conn: Any, ticker: str, count: int, status: str) -> None:
    conn.execute(f"DELETE FROM {FETCH_LOG_TABLE} WHERE ticker = ?", [ticker])
    conn.execute(
        f"INSERT INTO {FETCH_LOG_TABLE} (ticker, last_fetched_at, event_count, status) "
        "VALUES (?, ?, ?, ?)",
        [ticker, datetime.now(), int(count), status],
    )


def refresh_events(tickers: list[str], db_path: str | None = None) -> int:
    """Fetch + cache declared events for `tickers`. Returns rows written.

    The ONLY network-touching function in this module. Fully short-circuits when
    `corporate_action_events_enabled` is False. Per ticker:

      * the symbol is validated first (an empty symbol returns 50 unrelated rows);
      * calls are paced by `CONFIG.crawler.throttle_min_interval_seconds`, the
        same guest-safe cadence the OHLCV crawler uses;
      * a NON-EMPTY payload REPLACES that ticker's cached rows (delete-then-
        insert). vnstock restates events — many rows carry a NULL `exright_date`
        that is filled in later — the payload is authoritative, and it is capped
        at 50 rows per ticker, so replacement is simpler and more correct than an
        anti-join;
      * an EMPTY payload or an exception leaves existing rows INTACT and only
        records the outcome in the fetch log.

    One ticker failing never aborts the rest, and the function never raises.
    """
    if not CONFIG.trading.corporate_action_events_enabled:
        LOGGER.info("[CA] corporate_action_events_enabled=False — refresh skipped.")
        return 0
    if Company is None:
        LOGGER.warning("[CA] vnstock Company unavailable — refresh skipped.")
        return 0

    wanted = [t for t in (_valid_ticker(x) for x in tickers) if t]
    wanted = sorted(dict.fromkeys(wanted))
    if not wanted:
        return 0

    pace = float(getattr(CONFIG.crawler, "throttle_min_interval_seconds", 4.25) or 0.0)
    written = 0
    try:
        with _connect(db_path) as conn:
            ensure_tables(conn)
            for i, ticker in enumerate(wanted):
                if i > 0 and pace > 0:
                    time.sleep(pace)
                try:
                    frame = Company(source=_SOURCE, symbol=ticker).events()
                except Exception as exc:  # noqa: BLE001 — bad symbol, network, rate limit
                    LOGGER.warning("[CA] %s events() failed: %s", ticker, exc)
                    _write_fetch_log(conn, ticker, 0, "error")
                    continue
                if frame is None or getattr(frame, "empty", True):
                    LOGGER.info("[CA] %s returned no events — cache left intact.", ticker)
                    _write_fetch_log(conn, ticker, 0, "empty")
                    continue
                rows = _rows_from_events_frame(ticker, frame)
                if not rows:
                    _write_fetch_log(conn, ticker, 0, "empty")
                    continue
                conn.execute(f"DELETE FROM {EVENTS_TABLE} WHERE ticker = ?", [ticker])
                conn.executemany(
                    f"INSERT INTO {EVENTS_TABLE} (ticker, event_id, event_code, category, "
                    "event_title_vi, exercise_ratio, value_per_share, exright_date, "
                    "record_date, public_date, kind, price_factor) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                _write_fetch_log(conn, ticker, len(rows), "ok")
                written += len(rows)
                LOGGER.info("[CA] %s: cached %s events.", ticker, len(rows))
    except Exception:  # noqa: BLE001 — cache write must never break the EOD pipeline
        LOGGER.exception("[CA] refresh_events failed.")
        return written
    return written


__all__ = [
    "EVENTS_TABLE",
    "FETCH_LOG_TABLE",
    "KIND_STOCK_DIVIDEND",
    "KIND_BONUS",
    "KIND_RIGHTS",
    "KIND_ESOP",
    "KIND_PRIVATE",
    "KIND_CONVERTIBLE",
    "KIND_CASH_DIVIDEND",
    "KIND_OTHER",
    "PRICEABLE_KINDS",
    "classify_event",
    "theoretical_price_factor",
    "bundle_factor",
    "has_unpriceable_rights",
    "find_gaps",
    "resolve_adjustment",
    "back_adjust_closes",
    "ensure_tables",
    "load_events",
    "tickers_needing_refresh",
    "refresh_events",
]
