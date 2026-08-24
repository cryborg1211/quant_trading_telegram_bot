"""The fallback card must not blame a gate the name actually passed (13-08-26).

WHAT WENT WRONG LIVE
────────────────────
The 13-08 retrain dropped the T+20 gate 0.46 -> 0.43. VHM then scored 43.4%,
CLEARED the gate for the first time in weeks, and was parked by ADMISSION
HYSTERESIS (streak 1 of 2 required). The operator card said:

    Cửa tăng chỉ 43% (dưới ngưỡng an toàn 45%)

Wrong twice over: 45% was a hardcoded literal the system no longer used, and the
name had PASSED the real 43% gate. The explanation contradicted what the code did.
"""
from __future__ import annotations

from unittest.mock import patch

import main
from main import _select_candidates

_UNIVERSE = frozenset({"VHM", "VCK", "DXG"})
# p_up chosen against a 0.43 gate: VHM clears it, the other two do not.
_PREDS = {
    "VHM": [0.558, 0.009, 0.434],
    "VCK": [0.570, 0.009, 0.421],
    "DXG": [0.570, 0.010, 0.421],
}
_GATE = {"VHM": True, "VCK": False, "DXG": False}


def _run(tau, *, streaks, min_days=2, hyst=True):
    with patch.object(main.candidate_hysteresis, "read_streaks", return_value=streaks), \
         patch.object(main.candidate_hysteresis, "update_streaks"), \
         patch.object(main.signal_ledger, "open_tickers", return_value=set()), \
         patch.object(main.CONFIG.trading, "hysteresis_enabled", hyst), \
         patch.object(main.CONFIG.trading, "hysteresis_min_qualify_days", min_days), \
         patch.object(main.CONFIG.trading, "dispatch_open_cohort_dedup_enabled", False):
        return _select_candidates(_PREDS, _GATE, _UNIVERSE, 6,
                                  persist=False, tau=tau)


def test_hysteresis_held_name_is_not_told_it_failed_the_gate():
    _c, _u, fallback, reasons = _run(0.43, streaks={})
    assert fallback, "no survivors -> fallback expected"
    assert "ĐÃ QUA cổng" in reasons["VHM"]
    assert "dưới ngưỡng" not in reasons["VHM"]


def test_held_name_reason_states_the_streak_progress():
    _c, _u, _f, reasons = _run(0.43, streaks={})
    assert "1/2" in reasons["VHM"]


def test_a_partial_streak_reports_the_right_remaining_count():
    _c, _u, _f, reasons = _run(0.43, streaks={"VHM": 1}, min_days=3)
    assert "2/3" in reasons["VHM"]


def test_genuinely_below_gate_names_keep_the_original_reason():
    _c, _u, _f, reasons = _run(0.43, streaks={})
    for t in ("VCK", "DXG"):
        assert "dưới ngưỡng an toàn 43.0%" in reasons[t]
        assert "ĐÃ QUA" not in reasons[t]


def test_the_quoted_threshold_tracks_the_live_tau():
    """The whole bug: a pinned 0.45 while the artifact said 0.43."""
    _c, _u, _f, r43 = _run(0.43, streaks={})
    assert "43.0%" in r43["VCK"]
    _c, _u, _f, r46 = _run(0.46, streaks={})
    assert "46.0%" in r46["VCK"]
    assert "45.0%" not in r46["VCK"]


def test_missing_tau_falls_back_to_the_old_literal_not_a_guess():
    _c, _u, _f, reasons = _run(None, streaks={})
    assert "45.0%" in reasons["VCK"]


def test_a_cleared_name_with_a_full_streak_is_admitted_not_explained():
    cands, _u, fallback, reasons = _run(0.43, streaks={"VHM": 1})
    assert cands == ["VHM"]
    assert not fallback
    assert reasons == {}


def test_hysteresis_disabled_admits_the_cleared_name():
    cands, _u, fallback, _r = _run(0.43, streaks={}, hyst=False)
    assert cands == ["VHM"]
    assert not fallback


def test_a_near_miss_never_prints_the_same_number_on_both_sides():
    """The 25-08 report: "Cửa tăng chỉ 43% (dưới ngưỡng an toàn 43%)".

    Both numbers were `:.0f`, so a name just under the gate rendered as
    "43% < 43%" - text that reads like a broken gate. Real rows on 24-08:
    MBB 0.4275, GMD 0.4263, GAS 0.4252 all printed exactly that. The DECISION was
    correct (all three sit below 0.43); only the wording lied.

    Latent until the gate moved: at 0.45/0.46 the gate sat above the dense part of
    the p_up distribution so the rounded pair rarely collided. At 0.42/0.43 it
    sits inside it.
    """
    preds = {
        "MBB": [0.562, 0.010, 0.4275],   # the real 24-08 values
        "GMD": [0.563, 0.010, 0.4263],
        "GAS": [0.564, 0.010, 0.4252],
    }
    gate = {t: False for t in preds}
    with patch.object(main.candidate_hysteresis, "read_streaks", return_value={}), \
         patch.object(main.candidate_hysteresis, "update_streaks"), \
         patch.object(main.signal_ledger, "open_tickers", return_value=set()), \
         patch.object(main.CONFIG.trading, "hysteresis_enabled", True), \
         patch.object(main.CONFIG.trading, "hysteresis_min_qualify_days", 2), \
         patch.object(main.CONFIG.trading, "dispatch_open_cohort_dedup_enabled", False):
        _c, _u, fallback, reasons = main._select_candidates(
            preds, gate, frozenset(preds), 6, persist=False, tau=0.43)

    assert fallback
    for t, p in (("MBB", 42.8), ("GMD", 42.6), ("GAS", 42.5)):
        r = reasons[t]
        assert f"chỉ {p}%" in r, f"{t}: expected one-decimal probability, got: {r}"
        assert "an toàn 43.0%" in r
        # The whole point: the two figures must never render identically.
        assert f"chỉ {p}% (dưới ngưỡng an toàn {p}%)" not in r


def test_a_cleared_name_just_above_the_gate_also_reads_unambiguously():
    """Same collision on the ADMITTED side: 43.0% vs a 43% gate."""
    preds = {"AAA": [0.559, 0.010, 0.4305]}
    with patch.object(main.candidate_hysteresis, "read_streaks", return_value={}), \
         patch.object(main.candidate_hysteresis, "update_streaks"), \
         patch.object(main.signal_ledger, "open_tickers", return_value=set()), \
         patch.object(main.CONFIG.trading, "hysteresis_enabled", True), \
         patch.object(main.CONFIG.trading, "hysteresis_min_qualify_days", 2), \
         patch.object(main.CONFIG.trading, "dispatch_open_cohort_dedup_enabled", False):
        _c, _u, _f, reasons = main._select_candidates(
            preds, {"AAA": True}, frozenset(preds), 6, persist=False, tau=0.43)
    r = reasons["AAA"]
    assert "43.05% ĐÃ QUA cổng 43.00%" in r, r
