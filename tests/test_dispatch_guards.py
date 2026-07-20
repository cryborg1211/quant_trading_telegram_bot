"""July-2026 dispatch guards: open-cohort dedup + arbitrator sector cap.

Incident: 12 of 13 July dispatches were red — BSR re-dispatched 4 consecutive
days while its cohort bled (no per-name dedup), and 8 of 13 picks were one
PVN energy/fertilizer complex (no sector cap). Guards live in
`main._select_candidates` (real pool only — the monitoring-only fallback
branch stays unfiltered), backed by `signal_ledger.open_tickers()` and
`src/trading/sector_map.py`.
"""
from __future__ import annotations

import duckdb
import pytest

import main
from config.settings import CONFIG
from src.trading import sector_map, signal_ledger


# ---------------------------------------------------------------------------
# sector_map (pure)
# ---------------------------------------------------------------------------


def test_sector_of_known_and_unknown():
    assert sector_map.sector_of("BSR") == "OIL_GAS"
    assert sector_map.sector_of("dpm") == "OIL_GAS"  # PVN fertilizer, same complex
    assert sector_map.sector_of("VCB") == "BANKS"
    assert sector_map.sector_of("ZZZ") == sector_map.OTHER


def test_july_cluster_maps_to_one_sector():
    # The exact July-2026 loss cluster must share ONE sector so the cap bites.
    assert {sector_map.sector_of(t) for t in ["BSR", "PVD", "GAS", "DPM", "DCM"]} == {"OIL_GAS"}


def test_apply_sector_cap_keeps_rank_order_and_trims():
    ranked = ["BSR", "PVD", "GAS", "VCB", "DPM", "HPG"]
    kept, trimmed = sector_map.apply_sector_cap(ranked, 2)
    assert kept == ["BSR", "PVD", "VCB", "HPG"]
    assert trimmed == ["GAS", "DPM"]


def test_apply_sector_cap_other_uncapped():
    ranked = ["AAA1", "AAA2", "AAA3"]  # all unmapped → OTHER
    kept, trimmed = sector_map.apply_sector_cap(ranked, 1)
    assert kept == ranked
    assert trimmed == []


def test_apply_sector_cap_zero_disables():
    ranked = ["BSR", "PVD", "GAS"]
    kept, trimmed = sector_map.apply_sector_cap(ranked, 0)
    assert kept == ranked
    assert trimmed == []


# ---------------------------------------------------------------------------
# signal_ledger.open_tickers (real temp DuckDB)
# ---------------------------------------------------------------------------


def test_open_tickers_reads_only_open_rows(tmp_path):
    db = str(tmp_path / "ledger.duckdb")
    with duckdb.connect(db) as conn:
        signal_ledger.ensure_table(conn)
        conn.execute(
            f"INSERT INTO {signal_ledger.TABLE} "
            "(ticker, dispatch_date, horizon, hold_days, weight, status) VALUES "
            "('BSR', DATE '2026-07-14', 5, 5, 0.05, 'OPEN'), "
            "('bsr', DATE '2026-07-15', 5, 5, 0.05, 'OPEN'), "
            "('HDB', DATE '2026-07-07', 5, 5, 0.05, 'CLOSED')"
        )
    assert signal_ledger.open_tickers(db_path=db) == {"BSR"}


def test_open_tickers_never_raises(tmp_path):
    assert signal_ledger.open_tickers(db_path=str(tmp_path / "missing" / "x.duckdb")) == set()


# ---------------------------------------------------------------------------
# _select_candidates integration (monkeypatched ledger + config)
# ---------------------------------------------------------------------------

_UNIVERSE = frozenset({"BSR", "PVD", "GAS", "DPM", "VCB", "HPG", "FPT", "MWG"})


def _preds(*tickers: str) -> dict[str, list[float]]:
    # Descending P(UP) in argument order.
    n = len(tickers)
    return {t: [0.1, 0.1, 0.9 - i * 0.05] for i, t in enumerate(tickers)}


def _gate_all(preds: dict) -> dict[str, bool]:
    return {t: True for t in preds}


@pytest.fixture(autouse=True)
def _guard_defaults(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "dispatch_open_cohort_dedup_enabled", True)
    monkeypatch.setattr(CONFIG.trading, "arbitrator_sector_cap", 2)


def test_select_candidates_skips_open_cohorts(monkeypatch):
    preds = _preds("BSR", "VCB", "HPG", "FPT")
    monkeypatch.setattr(main.signal_ledger, "open_tickers", lambda: {"BSR"})
    cands, _, fallback, _ = main._select_candidates(preds, _gate_all(preds), _UNIVERSE, 6)
    assert not fallback
    assert "BSR" not in cands
    assert cands == ["VCB", "HPG", "FPT"]


def test_select_candidates_sector_cap_promotes_next_best(monkeypatch):
    preds = _preds("BSR", "PVD", "GAS", "DPM", "VCB", "HPG")
    monkeypatch.setattr(main.signal_ledger, "open_tickers", lambda: set())
    cands, _, fallback, _ = main._select_candidates(preds, _gate_all(preds), _UNIVERSE, 6)
    assert not fallback
    # OIL_GAS capped at 2 (BSR, PVD) — GAS/DPM trimmed, later sectors promoted.
    assert cands == ["BSR", "PVD", "VCB", "HPG"]


def test_select_candidates_dedup_kill_switch(monkeypatch):
    preds = _preds("BSR", "VCB")
    monkeypatch.setattr(CONFIG.trading, "dispatch_open_cohort_dedup_enabled", False)
    monkeypatch.setattr(
        main.signal_ledger, "open_tickers",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    cands, _, _, _ = main._select_candidates(preds, _gate_all(preds), _UNIVERSE, 6)
    assert cands == ["BSR", "VCB"]


def test_select_candidates_fallback_branch_unfiltered(monkeypatch):
    # All names gated out → monitoring-only fallback must show the raw Top-3
    # (including open-cohort/sector-capped names) — observability, not trading.
    preds = _preds("BSR", "PVD", "GAS")
    gate = {t: False for t in preds}
    monkeypatch.setattr(main.signal_ledger, "open_tickers", lambda: {"BSR"})
    cands, _, fallback, reasons = main._select_candidates(preds, gate, _UNIVERSE, 6)
    assert fallback
    assert cands == ["BSR", "PVD", "GAS"]
    assert set(reasons) == {"BSR", "PVD", "GAS"}
