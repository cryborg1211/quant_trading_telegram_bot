"""Static HOSE ticker→sector map + pure sector-cap filter for candidate pools.

Born from the July-2026 loss cluster: 8 of 13 dispatched signals were one
correlated PVN energy/fertilizer complex (BSR×4, GAS×2, PVD, DPM, DCM) —
sentiment arbitration kept picking oil&gas headlines with no sector awareness.

Design notes:
    * OIL_GAS deliberately includes the PetroVietnam fertilizer subsidiaries
      (DPM, DCM) and gas-fired power (POW, NT2): same conglomerate, same
      gas-price driver — the whole July cluster maps to ONE sector here.
    * Coverage targets the liquid ADV-top-50 serve universe. Unmapped tickers
      fall to OTHER, which `apply_sector_cap` never caps (a mixed bucket is
      not a correlation cluster). Extend the map, don't cap OTHER.
    * Pure module: no config, no DB, no imports from main — mirrors the
      regime_policy.py single-source-of-truth pattern.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

OTHER = "OTHER"

_SECTOR_TICKERS: dict[str, frozenset[str]] = {
    "BANKS": frozenset({
        "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "STB", "HDB", "TPB",
        "SHB", "EIB", "MSB", "LPB", "VIB", "OCB", "NAB", "SSB",
    }),
    "SECURITIES": frozenset({
        "SSI", "VND", "VCI", "HCM", "VIX", "SHS", "MBS", "FTS", "BSI", "ORS",
        "VDS", "CTS", "AGR", "TCX",
    }),
    "REAL_ESTATE": frozenset({
        "VIC", "VHM", "VRE", "NVL", "KDH", "DXG", "PDR", "DIG", "NLG", "CEO",
        "HDC", "HDG", "KBC", "SZC", "TCH", "SCR", "ITA", "DXS",
    }),
    # PVN complex: upstream/midstream/downstream + gas-input fertilizer +
    # gas-fired power. One gas-price driver ⇒ one sector for capping purposes.
    "OIL_GAS": frozenset({
        "GAS", "PLX", "BSR", "PVD", "PVS", "PVT", "PVC", "PVB", "CNG", "PGD",
        "DPM", "DCM", "POW", "NT2",
    }),
    "STEEL_MATERIALS": frozenset({
        "HPG", "HSG", "NKG", "SMC", "TLH", "VGS",
    }),
    "CONSUMER": frozenset({
        "VNM", "MSN", "MWG", "PNJ", "SAB", "FRT", "DGW", "KDC", "MCH",
    }),
    "TECH_TELCO": frozenset({
        "FPT", "CMG", "VGI", "CTR", "ELC",
    }),
    "TRANSPORT_LOGISTICS": frozenset({
        "HVN", "VJC", "GMD", "VTP", "HAH", "VSC", "VOS",
    }),
    "CHEMICALS": frozenset({
        "DGC", "CSV", "LAS", "BFC",
    }),
    "CONSTRUCTION_UTILITIES": frozenset({
        "VCG", "HHV", "CII", "LCG", "FCN", "CTD", "REE", "PC1", "GEG", "GEX",
        "VGC",
    }),
    "AGRI_AQUA": frozenset({
        "HAG", "HNG", "VHC", "ANV", "DBC", "PAN",
    }),
    "INSURANCE_FINANCE": frozenset({
        "BVH", "BIC", "BMI", "EVF",
    }),
}

_TICKER_TO_SECTOR: dict[str, str] = {
    t: sector for sector, tickers in _SECTOR_TICKERS.items() for t in tickers
}


def sector_of(ticker: str) -> str:
    """Sector name for `ticker`, OTHER when unmapped."""
    return _TICKER_TO_SECTOR.get(str(ticker).upper(), OTHER)


def apply_sector_cap(
    ranked_tickers: list[str], cap: int
) -> tuple[list[str], list[str]]:
    """Keep at most `cap` names per sector, preserving rank order.

    OTHER is never capped (mixed bucket ≠ correlation cluster). `cap <= 0`
    disables capping entirely. Returns (kept, trimmed) — trimmed keeps rank
    order too, for the caller's log line.
    """
    if cap <= 0:
        return list(ranked_tickers), []
    kept: list[str] = []
    trimmed: list[str] = []
    counts: dict[str, int] = {}
    for ticker in ranked_tickers:
        sector = sector_of(ticker)
        if sector == OTHER:
            kept.append(ticker)
            continue
        if counts.get(sector, 0) >= cap:
            trimmed.append(ticker)
            continue
        counts[sector] = counts.get(sector, 0) + 1
        kept.append(ticker)
    return kept, trimmed
