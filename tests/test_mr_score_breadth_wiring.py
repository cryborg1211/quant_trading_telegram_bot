"""mr_score_tickers' breadth-context wiring (22-07-26).

Rules under test:
  * breadth.live_breadth_inflection() is called AT MOST once per
    mr_score_tickers() call (never per-ticker) — it loads a full OHLCV
    panel, so per-ticker calls would be wasteful.
  * It is ONLY called when at least one ticker actually fired (the
    annotation is only ever displayed alongside a fire) AND the config
    kill-switch is on.
  * Every returned ticker dict carries "breadth_favorable" (possibly None).

Heavy dependencies (_load_mr, Alpha360Generator, build_mr_features) are
mocked — this is wiring-logic coverage, not a model/feature integration test.
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
        {"ticker": t, "date": "2026-07-20", **row} for t in tickers
    ])


@pytest.fixture(autouse=True)
def _mr_model_cache_reset():
    main._MR_MODEL = None
    main._MR_TAU = None
    yield
    main._MR_MODEL = None
    main._MR_TAU = None


def _patched(probs: np.ndarray, tickers: list[str]):
    model = MagicMock()
    model.predict_proba.return_value = np.column_stack([1 - probs, probs])
    gen = MagicMock()
    gen._load_live_stock_window.return_value = MagicMock(
        to_pandas=lambda: pd.DataFrame({"ticker": tickers, "date": ["2026-07-20"] * len(tickers)})
    )
    return patch.multiple(
        main,
        _load_mr=MagicMock(return_value=(model, 0.96)),
        Alpha360Generator=MagicMock(return_value=gen),
        build_mr_features=MagicMock(side_effect=lambda pdf: _mr_frame(tickers)),
    )


def test_no_fires_never_calls_breadth(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_breadth_context_enabled", True)
    with _patched(np.array([0.10, 0.20]), ["AAA", "BBB"]), \
         patch.object(main.breadth, "live_breadth_inflection") as m_breadth:
        out = main.mr_score_tickers(["AAA", "BBB"])
    m_breadth.assert_not_called()
    assert all(v["breadth_favorable"] is None for v in out.values())
    assert all(v["fired"] is False for v in out.values())


def test_one_fire_calls_breadth_exactly_once_for_all_tickers(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_breadth_context_enabled", True)
    with _patched(np.array([0.99, 0.10, 0.05]), ["AAA", "BBB", "CCC"]), \
         patch.object(main.breadth, "live_breadth_inflection",
                      return_value={"breadth": 0.2, "breadth_delta": 0.05, "favorable": True}) as m_breadth:
        out = main.mr_score_tickers(["AAA", "BBB", "CCC"])
    m_breadth.assert_called_once()
    # Broadcast to EVERY ticker, not just the one that fired.
    assert out["AAA"]["breadth_favorable"] is True
    assert out["BBB"]["breadth_favorable"] is True
    assert out["CCC"]["breadth_favorable"] is True
    assert out["AAA"]["fired"] is True
    assert out["BBB"]["fired"] is False


def test_breadth_context_unavailable_degrades_to_none(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_breadth_context_enabled", True)
    with _patched(np.array([0.99]), ["AAA"]), \
         patch.object(main.breadth, "live_breadth_inflection", return_value=None):
        out = main.mr_score_tickers(["AAA"])
    assert out["AAA"]["breadth_favorable"] is None
    assert out["AAA"]["fired"] is True  # fire decision unaffected by breadth failure


def test_kill_switch_skips_breadth_even_on_fire(monkeypatch):
    monkeypatch.setattr(CONFIG.trading, "mr_breadth_context_enabled", False)
    with _patched(np.array([0.99]), ["AAA"]), \
         patch.object(main.breadth, "live_breadth_inflection") as m_breadth:
        out = main.mr_score_tickers(["AAA"])
    m_breadth.assert_not_called()
    assert out["AAA"]["breadth_favorable"] is None
    assert out["AAA"]["fired"] is True  # kill-switch only disables the annotation
