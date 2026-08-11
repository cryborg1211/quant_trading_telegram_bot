"""Model accuracy auditor â€” confusion-matrix analytics over the paperlog.

READ-SIDE ONLY. This module adds the one thing the existing audit stack does
not have: it classifies the forward paper-logged predictions in
``sentiment_entry_paperlog`` into a BUY-vs-defensive confusion matrix
(TP / FP / TN / FN) and reports BUY precision + defensive recall.

It deliberately introduces NO new table and patches NO serve path. Capture and
settlement already exist:
    * capture   â€” ``main._log_sentiment_entry_paperlog`` (daily + /verify)
    * settlement â€” ``main._backfill_paperlog_outcomes`` (progressive, T+20 terminal)
A row is "settled" once its terminal ``ret_20d`` is filled
(``outcome_filled = TRUE``); ``ret_20d`` is a fraction ``(t20 - t0) / t0``.

Decision encoding (argmax over [DOWN, SIDE, UP], shared by ``decision_primary`` and
the arbitrated ``final_decision``): ``0 = SELL/EXIT``, ``1 = HOLD``, ``2 = BUY``.
"""

from __future__ import annotations

from typing import Any

# Decision integer encoding (matches the arbitrator + paperlog argmax).
_SELL, _HOLD, _BUY = 0, 1, 2
_DECISION_LABEL = {_BUY: "BUY", _HOLD: "HOLD", _SELL: "SELL"}

# Confusion-matrix tag â†’ status emoji. Correct bullish call = green, correct
# defensive call = blue, any wrong call (bad buy OR missed upside) = red.
_TAG_EMOJI = {"TP": "ðŸŸ¢", "TN": "ðŸ”µ", "FP": "ðŸ”´", "FN": "ðŸ”´"}

_DEFAULT_TABLE_ROWS = 15


def decision_label(decision: int | None) -> str:
    """Human label for a decision int (``?`` when missing / unknown)."""
    return _DECISION_LABEL.get(decision, "?") if decision is not None else "?"


def classify_outcome(decision: int | None, ret: float | None) -> str | None:
    """Classify one settled prediction into TP / FP / TN / FN.

    BUY is the positive class; SELL and HOLD are the defensive (negative) class.

    * BUY  & ret > 0  â†’ ``TP``  (acted, price rose)
    * BUY  & ret <= 0 â†’ ``FP``  (acted, price did not rise)
    * SELL/HOLD & ret <= 0 â†’ ``TN``  (stayed out, price fell/flat â€” correct)
    * SELL/HOLD & ret > 0  â†’ ``FN``  (stayed out, price rose â€” missed)

    Returns ``None`` when the row is ungradable (no decision or no return).
    """
    if decision is None or ret is None:
        return None
    if decision == _BUY:
        return "TP" if ret > 0 else "FP"
    # SELL or HOLD = defensive class.
    return "FN" if ret > 0 else "TN"


def fetch_settled_predictions(
    db: Any, lookback_days: int | None = None
) -> list[dict[str, Any]]:
    """Settled paperlog rows (terminal T+20 return present), newest first.

    ``db`` must expose ``.conn`` (DuckDB). Prefers the arbitrated
    ``final_decision``; falls back to the raw ``decision_primary`` argmax when the
    arbitrated value is NULL. Degrades to ``[]`` on any read failure so an
    audit can never crash the caller.
    """
    sql = (
        "SELECT log_date, ticker, "
        "COALESCE(final_decision, decision_primary) AS decision, "
        "entry_close, ret_20d "
        "FROM sentiment_entry_paperlog "
        "WHERE outcome_filled = TRUE AND ret_20d IS NOT NULL "
    )
    params: list[Any] = []
    if lookback_days is not None:
        sql += "AND log_date >= (CURRENT_DATE - CAST(? AS INTEGER)) "
        params.append(int(lookback_days))
    sql += "ORDER BY log_date DESC, ticker"

    try:
        rows = db.conn.execute(sql, params).fetchall()
    except Exception:  # noqa: BLE001 â€” missing table / cold DB â†’ empty audit.
        return []

    return [
        {
            "log_date": r[0],
            "ticker": str(r[1]).upper(),
            "decision": int(r[2]) if r[2] is not None else None,
            "entry_close": r[3],
            "ret": float(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]


def summarize_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate settled rows into the confusion matrix + precision/recall.

    BUY precision   = TP / (TP + FP)   â€” of acted-on buys, how many rose.
    Defensive recall = TN / (TN + FN)  â€” of true non-winners, how many we avoided.
    Both are ``None`` when their denominator is zero (shown as "â€”" upstream).
    """
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for row in rows:
        tag = classify_outcome(row.get("decision"), row.get("ret"))
        if tag is not None:
            counts[tag] += 1

    total = sum(counts.values())
    buy_den = counts["TP"] + counts["FP"]
    def_den = counts["TN"] + counts["FN"]
    return {
        "counts": counts,
        "total_settled": total,
        "buy_precision": (counts["TP"] / buy_den) if buy_den else None,
        "defensive_recall": (counts["TN"] / def_den) if def_den else None,
    }


def _fmt_pct(value: float | None) -> str:
    """Percent with one decimal, or ``â€”`` for an undefined ratio."""
    return f"{value * 100:.1f}%" if value is not None else "â€”"


def build_accuracy_report(
    db: Any | None = None,
    last_n: int = _DEFAULT_TABLE_ROWS,
    lookback_days: int | None = None,
) -> str:
    """Render the HTML accuracy digest (header stats + last-N settled table).

    Returns ``""`` when no settled predictions exist so the caller can show a
    clean empty-state instead of an empty table. ``db`` defaults to the
    ``DuckDBEngine`` singleton (lazy import) so callers can stay thin.
    """
    if db is None:
        from src.data.db_engine import DuckDBEngine  # noqa: PLC0415 â€” lazy heavy import

        db = DuckDBEngine()

    rows = fetch_settled_predictions(db, lookback_days=lookback_days)
    if not rows:
        return ""

    summary = summarize_accuracy(rows)
    c = summary["counts"]

    header = (
        "ðŸ“Š <b>Háº¬U KIá»‚M Äá»˜ CHÃNH XÃC MÃ” HÃŒNH</b>\n"
        f"<b>Tá»•ng Ä‘Ã£ chá»‘t:</b> {summary['total_settled']}  Â·  "
        f"<b>BUY Precision:</b> {_fmt_pct(summary['buy_precision'])}  Â·  "
        f"<b>Defensive Recall:</b> {_fmt_pct(summary['defensive_recall'])}\n"
        f"<i>ðŸŸ¢ TP {c['TP']} Â· ðŸ”´ FP {c['FP']} Â· ðŸ”µ TN {c['TN']} Â· ðŸ”´ FN {c['FN']}</i>"
    )

    # Fixed-width table inside <pre> so columns align in Telegram HTML.
    lines = [f"{'MÃ£':<6}{'NgÃ y':<12}{'Lá»‡nh':<6}{'R%':>8}  KQ"]
    for row in rows[:last_n]:
        tag = classify_outcome(row["decision"], row["ret"])
        emoji = _TAG_EMOJI.get(tag, "")
        ret_pct = f"{row['ret'] * 100:+.1f}" if row["ret"] is not None else "â€”"
        lines.append(
            f"{row['ticker']:<6}{str(row['log_date']):<12}"
            f"{decision_label(row['decision']):<6}{ret_pct:>8}  {emoji} {tag}"
        )

    table = "<pre>" + "\n".join(lines) + "</pre>"
    return f"{header}\n{table}"
