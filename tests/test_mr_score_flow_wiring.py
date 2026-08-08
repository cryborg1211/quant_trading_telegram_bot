"""mr_score_tickers' flow-divergence wiring (08-08-26).

Rules under test:
  * flow_context.live_flow_divergence() is called ONCE PER FIRED TICKER
    (unlike breadth, which is one market-wide reading reused for every
    ticker) — it's a per-ticker live fetch, so a name that didn't fire
    must never trigger a call.
  * Only called when the config kill-switch is on.
  * Every returned ticker dict carries "flow_divergence" (possibly None).
  * A live_flow_divergence failure for one ticker never breaks scoring
    for the others (fail-open contract).

Heavy dependencies (_load_mr, Alpha360Generator, build_mr_features) are
mocked — this is wiring-logic coverage, not a model/feature integration
test. Mirrors test_mr_score_breadth_wiring.py's structure.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import main
from config.settings import CONFIG
from src.features.mr_features import MR_FEATURE_COLUMNS


def _mr_frame(tickers: list[str]) -> pd.DataFrame:
    row = {c: 0.0 for c in MR_FEATURE_COLUMNS}
    return pd.DataFrame([
        {"ticker": t, "date": "2026-08-08", **row} for t in tickers
    ])


@pytest.fixture(autouse=True)
def _mr_model_cache_reset():
    main._MR_MODEL = None
    main._MR_TAU = None
    yield
    main._MR_MODEL = None
    main._MR_TAU = None


@pytest.fixture(autouse=True)
def _breadth_context_off(monkeypatch):
    # Isolate flow-divergence wiring from the unrelated breadth leg.
    monkeypatch.setattr(CONFIG.trading, "mr_breadth_context_enabled", False)


def _patched(probs: np.ndarray, tickers: list[str]):
    model = MagicMock()
    model.predict_proba.return_value = np.column_stack([1 - probs, probs])
    gen = MagicMock()
    gen._load_live_stock_window.return_value = MagicMock(
        to_pandas=lambda: pd.DataFrame({"ticker": tickers, "date": ["2026-08-08"] * len(tickers)})
    )
    return patch.multiple(
        main,
        _load_mr=MagicMock(return_value=(model, 0.96)),
        Alpha360Generator=MagicMock(return_value=gen),
        build_mr_features=MagicMock(side_effect=lambda pdf: _mr_frame(tickers)),
    )


def test_no_fires_never_calls_flow_divergence(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_flow_divergence_enabled", True)
    with _patched(np.array([0.10, 0.20]), ["AAA", "BBB"]), \
         patch.object(main.flow_context, "live_flow_divergence") as m_flow:
        out = main.mr_score_tickers(["AAA", "BBB"])
    m_flow.assert_not_called()
    assert all(v["flow_divergence"] is None for v in out.values())


def test_one_fire_calls_flow_divergence_only_for_that_ticker(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_flow_divergence_enabled", True)
    with _patched(np.array([0.99, 0.10, 0.05]), ["AAA", "BBB", "CCC"]), \
         patch.object(main.flow_context, "live_flow_divergence",
                      return_value={"divergence": True, "flow_net_scaled_adv20": 1.2}) as m_flow:
        out = main.mr_score_tickers(["AAA", "BBB", "CCC"])
    m_flow.assert_called_once_with("AAA")  # NOT called for BBB/CCC (never fired)
    assert out["AAA"]["flow_divergence"] is True
    assert out["BBB"]["flow_divergence"] is None
    assert out["CCC"]["flow_divergence"] is None


def test_multiple_fires_call_flow_divergence_per_ticker(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_flow_divergence_enabled", True)
    with _patched(np.array([0.99, 0.98]), ["AAA", "BBB"]), \
         patch.object(main.flow_context, "live_flow_divergence",
                      side_effect=lambda t: {"divergence": t == "AAA", "flow_net_scaled_adv20": 0.5}) as m_flow:
        out = main.mr_score_tickers(["AAA", "BBB"])
    assert m_flow.call_count == 2
    assert out["AAA"]["flow_divergence"] is True
    assert out["BBB"]["flow_divergence"] is False


def test_flow_context_unavailable_degrades_to_none(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_flow_divergence_enabled", True)
    with _patched(np.array([0.99]), ["AAA"]), \
         patch.object(main.flow_context, "live_flow_divergence", return_value=None):
        out = main.mr_score_tickers(["AAA"])
    assert out["AAA"]["flow_divergence"] is None
    assert out["AAA"]["fired"] is True  # fire decision unaffected by flow-context failure


def test_kill_switch_skips_flow_divergence_even_on_fire(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_flow_divergence_enabled", False)
    with _patched(np.array([0.99]), ["AAA"]), \
         patch.object(main.flow_context, "live_flow_divergence") as m_flow:
        out = main.mr_score_tickers(["AAA"])
    m_flow.assert_not_called()
    assert out["AAA"]["flow_divergence"] is None
    assert out["AAA"]["fired"] is True  # kill-switch only disables the annotation
