"""`/verify` disqualifier block (12-08-26).

WHY
───
`/verify`'s verdict is T+5-DOMINANT: `verify_single_ticker` passes SHORT_HORIZON
as the primary and `make_final_decision` is asymmetric (`pred_5d == 2` returns
BUY regardless of the other horizon). So the card could print
`Kết luận tổng hợp: MUA` on the exact configuration measured at -1.57% mean 20d
(T+5=UP with T+20=DOWN) against T+20=UP's +1.19% — the documented July
side-door — with nothing on it saying so. The dispatch card gained that warning
in the 10-08 redesign; `/verify` uses a separate builder and did not.
"""
from __future__ import annotations

from src.reports.builders import _build_verify_report

_SENT = {"sentiment_score": 0.35, "reasoning_vi": "Tin trung tinh.",
         "source_urls": []}
_T5_UP = [0.20, 0.22, 0.58]
_T20_DOWN = [0.45, 0.24, 0.31]
_T20_UP = [0.20, 0.20, 0.60]


def _card(**over) -> str:
    kw = dict(ticker="HPG", decision=2, sentiment=_SENT,
              stacking_5d=list(_T5_UP), stacking_20d=list(_T20_DOWN),
              live_exec_price=22100.0, mr_state=None,
              tau_5d=0.44, tau_20d=0.46,
              room_exhausted=False, market_regime=6)
    kw.update(over)
    return _build_verify_report(**kw)


def test_t5_only_fires_the_side_door_warning():
    card = _card()          # p5 0.58 >= 0.44, p20 0.31 < 0.46
    assert "CẢNH BÁO" in card
    assert "Chỉ qua cổng T+5" in card


def test_both_gates_open_is_clean():
    card = _card(stacking_20d=list(_T20_UP), market_regime=3)
    assert "CẢNH BÁO" not in card


def test_room_exhausted_fires():
    card = _card(stacking_20d=list(_T20_UP), market_regime=3, room_exhausted=True)
    assert "Hết room ngoại" in card


def test_no_trade_regime_fires():
    card = _card(stacking_20d=list(_T20_UP), market_regime=7)
    assert "KHÔNG giao dịch" in card


def test_all_three_can_fire_together():
    card = _card(room_exhausted=True, market_regime=7)
    assert card.count("•") >= 3
    for frag in ("Hết room ngoại", "Chỉ qua cổng T+5", "KHÔNG giao dịch"):
        assert frag in card


def test_missing_taus_invent_no_warning():
    """A gate verdict needs BOTH a probability and its threshold.

    `predict_v3_horizon` may fail for the secondary horizon, and guessing a
    warning from a half-known state would cry wolf on a clean ticker.
    """
    card = _card(tau_5d=None, tau_20d=None, room_exhausted=None,
                 market_regime=None)
    assert "CẢNH BÁO" not in card


def test_unavailable_t20_artifact_does_not_fire_the_side_door():
    """The secondary horizon can be missing entirely.

    `verify_single_ticker` then passes the flat [0.33, 0.34, 0.33] placeholder
    and `tau_20d=None`. That state is "unknown", not "gate failed", and warning
    on it would fire on every ticker whenever the T+20 artifact is unloadable.
    """
    card = _card(stacking_20d=[0.33, 0.34, 0.33], tau_20d=None)
    assert "Chỉ qua cổng T+5" not in card


def test_warning_sits_above_the_verdict_it_qualifies():
    card = _card()
    assert card.index("CẢNH BÁO") < card.index("Kết luận tổng hợp")


def test_card_stays_under_the_telegram_limit_with_all_warnings():
    card = _card(room_exhausted=True, market_regime=7)
    assert len(card) <= 4096


def test_html_tags_stay_balanced_with_the_warning_block():
    card = _card(room_exhausted=True, market_regime=7)
    for tag in ("b", "i", "code"):
        assert card.count(f"<{tag}>") == card.count(f"</{tag}>")


def test_clean_card_is_byte_identical_to_the_pre_warning_layout():
    """No warnings => no structural change.

    The block must be additive: a clean ticker's card should look exactly as it
    did before, so the new args cannot regress the common case.
    """
    with_args = _card(stacking_20d=list(_T20_UP), market_regime=3)
    without = _build_verify_report(
        ticker="HPG", decision=2, sentiment=_SENT,
        stacking_5d=list(_T5_UP), stacking_20d=list(_T20_UP),
        live_exec_price=22100.0, mr_state=None)
    assert with_args == without
