"""Unit tests for the event layer (rescue bull-bypass + bear veto).

`build_event_overrides` is the PURE core of the live arbitration in
`daily_inference` (rescue + veto). It decides which tickers get a forced
EVENT-DRIVEN 5%-NAV probe or a 0%-NAV bad-news block — i.e. it directly drives
real-money sizing. These tests lock the four cases + the snippet wiring.
"""
from __future__ import annotations

import main
from main import (
    build_event_overrides,
    SAFE_BUY_THRESHOLD, EVENT_MIN_P_UP, EVENT_BULL_SENTIMENT, EVENT_BEAR_SENTIMENT,
    _EVENT_CAP,
)

UNIVERSE = {"AAA", "BBB", "CCC", "DDD"}


def _preds(**kv):
    # ticker -> [p_down, p_flat, p_up]
    return {t: [0.0, 0.0, p] for t, p in kv.items()}


def test_bull_rescue_adds_event_override_at_cap():
    preds = _preds(AAA=0.43)                       # 0.42 ≤ 0.43 < 0.45
    sents = {"AAA": {"sentiment_score": 0.72, "reasoning_vi": "hợp đồng lớn ký kết"}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=[])
    assert rescued == ["AAA"]
    assert ov["AAA"]["weight"] == _EVENT_CAP        # forced 5% NAV
    assert "EVENT-DRIVEN" in ov["AAA"]["status"]
    assert "Bắt tin:" in ov["AAA"]["ly_do"] and "hợp đồng" in ov["AAA"]["ly_do"]


def test_bull_rescue_needs_strong_sentiment():
    preds = _preds(AAA=0.43)
    sents = {"AAA": {"sentiment_score": EVENT_BULL_SENTIMENT - 0.01}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=[])
    assert ov == {} and rescued == []               # below 0.60 → no rescue


def test_no_rescue_when_pup_below_floor():
    preds = _preds(AAA=EVENT_MIN_P_UP - 0.01)        # < 0.42
    sents = {"AAA": {"sentiment_score": 0.90}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=[])
    assert ov == {} and rescued == []


def test_no_rescue_when_already_held_or_out_of_universe():
    preds = _preds(AAA=0.43, ZZZ=0.43)               # ZZZ not in universe
    sents = {"AAA": {"sentiment_score": 0.9}, "ZZZ": {"sentiment_score": 0.9}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=["AAA"])
    assert ov == {} and rescued == []                # AAA held, ZZZ out


def test_bear_veto_hard_blocks_strong_signal():
    preds = _preds(AAA=0.52)                          # passed the 0.45 gate
    sents = {"AAA": {"sentiment_score": EVENT_BEAR_SENTIMENT - 0.05}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=["AAA"])
    assert rescued == []
    assert ov["AAA"]["weight"] == 0.0                # hard block
    assert "TIN X" in ov["AAA"]["status"]            # HỦY BỎ (TIN XẤU)


def test_normal_signal_no_override():
    preds = _preds(AAA=0.52)
    sents = {"AAA": {"sentiment_score": 0.10}}        # neutral news
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=["AAA"])
    assert ov == {} and rescued == []                # standard pass → untouched


def test_rescue_and_veto_coexist():
    preds = _preds(AAA=0.52, BBB=0.43)               # AAA technical, BBB rescue candidate
    sents = {"AAA": {"sentiment_score": -0.7}, "BBB": {"sentiment_score": 0.8}}
    ov, rescued = build_event_overrides(preds, sents, UNIVERSE, top_buy_signals=["AAA"])
    assert rescued == ["BBB"]
    assert ov["BBB"]["weight"] == _EVENT_CAP          # rescued
    assert ov["AAA"]["weight"] == 0.0                 # vetoed
