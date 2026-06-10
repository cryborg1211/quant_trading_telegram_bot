"""Unit tests for _select_candidates() — pure VN30 gate + meta-gate + top-N sort."""
from __future__ import annotations

import pytest

from main import _select_candidates, _VN30_UNIVERSE


# ── helpers ──────────────────────────────────────────────────────────────────

_VN30 = _VN30_UNIVERSE  # shorthand for the real frozenset

def _preds(**kv: float) -> dict[str, list[float]]:
    """Build predictions dict: ticker -> [p_down, p_flat, p_up]."""
    return {t: [0.0, 0.0, p] for t, p in kv.items()}


def _gate_all_true(tickers: list[str]) -> dict[str, bool]:
    return {t: True for t in tickers}


def _gate_all_false(tickers: list[str]) -> dict[str, bool]:
    return {t: False for t in tickers}


# ── tests ────────────────────────────────────────────────────────────────────


def test_vn30_filter_keeps_only_universe_members():
    preds = _preds(VCB=0.7, BID=0.6, AAAA=0.8)  # AAAA not in VN30
    gate = _gate_all_true(["VCB", "BID", "AAAA"])
    candidates, universe, fb, reasons = _select_candidates(preds, gate, _VN30, 6)
    assert "AAAA" not in candidates
    assert "AAAA" not in universe
    assert set(candidates) <= {"VCB", "BID"}


def test_meta_gate_rejects_unprofitable_tickers():
    preds = _preds(VCB=0.7, BID=0.6, VHM=0.5)
    gate = {"VCB": True, "BID": False, "VHM": True}
    candidates, _, fb, _ = _select_candidates(preds, gate, _VN30, 6)
    assert "BID" not in candidates
    assert fb is False


def test_top_n_cap_limits_candidates():
    # 10 VN30 tickers, all gated True, max_candidates=3
    tickers = list(_VN30)[:10]
    preds = {t: [0.0, 0.0, 0.5 + i * 0.01] for i, t in enumerate(tickers)}
    gate = _gate_all_true(tickers)
    candidates, _, fb, _ = _select_candidates(preds, gate, _VN30, 3)
    assert len(candidates) == 3
    assert fb is False


def test_candidates_sorted_by_p_up_desc():
    preds = _preds(VCB=0.7, BID=0.5, VHM=0.6)
    gate = _gate_all_true(["VCB", "BID", "VHM"])
    candidates, _, _, _ = _select_candidates(preds, gate, _VN30, 6)
    assert candidates == ["VCB", "VHM", "BID"]


def test_fallback_mode_triggered_when_no_candidates():
    preds = _preds(VCB=0.3, BID=0.2, VHM=0.1)
    gate = _gate_all_false(["VCB", "BID", "VHM"])
    candidates, _, fb, reasons = _select_candidates(preds, gate, _VN30, 6)
    assert fb is True
    assert len(candidates) > 0


def test_fallback_mode_selects_top3_by_p_up():
    tickers = ["VCB", "BID", "VHM", "FPT", "MBB"]
    preds = {t: [0.0, 0.0, 0.3 + i * 0.01] for i, t in enumerate(tickers)}
    gate = _gate_all_false(tickers)
    candidates, _, fb, _ = _select_candidates(preds, gate, _VN30, 6)
    assert fb is True
    assert len(candidates) == 3
    # Top-3 by P(UP) desc: MBB=0.34, FPT=0.33, VHM=0.32
    assert candidates == ["MBB", "FPT", "VHM"]


def test_fallback_reasons_include_low_p_up_reason():
    preds = _preds(VCB=0.30)
    gate = _gate_all_false(["VCB"])
    _, _, fb, reasons = _select_candidates(preds, gate, _VN30, 6)
    assert fb is True
    assert "VCB" in reasons
    assert "ngưỡng an toàn" in reasons["VCB"]


def test_fallback_reasons_include_meta_gate_reason():
    # Ticker with P(UP) >= tau (0.45) but meta_gate=False
    preds = _preds(VCB=0.48)
    gate = {"VCB": False}
    _, _, fb, reasons = _select_candidates(preds, gate, _VN30, 6)
    assert fb is True
    assert "VCB" in reasons
    assert "bộ lọc" in reasons["VCB"]


def test_no_fallback_when_candidates_exist():
    preds = _preds(VCB=0.7, BID=0.6)
    gate = _gate_all_true(["VCB", "BID"])
    candidates, _, fb, reasons = _select_candidates(preds, gate, _VN30, 6)
    assert fb is False
    assert reasons == {}
    assert len(candidates) == 2


def test_vn30_universe_empty_fallback_uses_all_predictions():
    preds = _preds(AAAA=0.8, BBBB=0.7)
    gate = _gate_all_true(["AAAA", "BBBB"])
    candidates, universe, fb, _ = _select_candidates(preds, gate, _VN30, 6)
    # Neither AAAA nor BBBB is in VN30, so liquid_tickers falls back to all
    assert "AAAA" in universe
    assert "BBBB" in universe
    assert fb is False
    assert len(candidates) == 2
