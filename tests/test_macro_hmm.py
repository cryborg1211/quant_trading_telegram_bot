"""Tests for macro-blended regime HMM (Macro-Integration P2).

Covers `build_regime_observation` (price-only fallback vs macro-widened) and the
N-D generalization of `MacroRiskHMM` / `train_macro_risk_hmm` — including that the
legacy 1-D price-only path is preserved and that the Bull state is still defined
by the market-breadth dim (column 0), with macro dims only sharpening the split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("hmmlearn")  # the HMM is the point of this module

from src.models.macro_risk_hmm import (  # noqa: E402
    build_regime_observation,
    train_macro_risk_hmm,
)


def _panel(n: int = 40, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pl.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(1)
    rows = []
    for t in tickers:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        rows.extend({"ticker": t, "date": d, "close": float(c)} for d, c in zip(idx, close))
    return pl.DataFrame(rows)


def _macro_parquet(tmp_path, dates) -> str:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "sp500": 4000.0, "dxy": 100.0, "usdvnd": 24000.0,
        "sp500_ret": rng.normal(0, 0.01, len(dates)),
        "dxy_ret": rng.normal(0, 0.005, len(dates)),
        "usdvnd_ret": rng.normal(0, 0.002, len(dates)),
    })
    p = tmp_path / "macro_daily.parquet"
    df.to_parquet(p, index=False)
    return str(p)


def _two_regime_market(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    half = n // 2
    vals = np.concatenate([
        rng.normal(-0.012, 0.008, half),     # bear regime
        rng.normal(0.012, 0.008, n - half),  # bull regime
    ])
    return pd.Series(vals, index=idx, name="market_ret")


def _two_regime_obs(n: int = 300, seed: int = 0) -> pd.DataFrame:
    s = _two_regime_market(n, seed)
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "market_ret": s,
        "sp500_ret": s.to_numpy() * 0.7 + rng.normal(0, 0.004, n),   # risk-on co-moves
        "dxy_ret": -s.to_numpy() * 0.4 + rng.normal(0, 0.004, n),    # USD risk-off
    }, index=s.index)


# --------------------------------------------------------------------------- #
# build_regime_observation
# --------------------------------------------------------------------------- #
class TestBuildRegimeObservation:
    def test_returns_series_when_macro_off(self) -> None:
        out = build_regime_observation(_panel(), use_macro=False)
        assert isinstance(out, pd.Series)

    def test_falls_back_to_series_when_parquet_absent(self, tmp_path) -> None:
        out = build_regime_observation(
            _panel(), use_macro=True, macro_parquet=tmp_path / "nope.parquet"
        )
        assert isinstance(out, pd.Series)

    def test_widens_to_dataframe_with_macro(self, tmp_path) -> None:
        panel = _panel()
        dates = panel["date"].unique().to_list()
        p = _macro_parquet(tmp_path, dates)
        out = build_regime_observation(panel, use_macro=True, macro_parquet=p)
        assert isinstance(out, pd.DataFrame)
        assert out.columns[0] == "market_ret"        # market dim stays first
        assert {"sp500_ret", "dxy_ret", "usdvnd_ret"}.issubset(out.columns)
        assert not out.isna().any().any()            # dense emission matrix


# --------------------------------------------------------------------------- #
# MacroRiskHMM — 1-D backward compat + N-D
# --------------------------------------------------------------------------- #
class TestMacroRiskHMM:
    def test_1d_backward_compat(self) -> None:
        s = _two_regime_market()
        hmm = train_macro_risk_hmm(s)
        assert hmm.bull_state in (0, 1)
        pb = hmm.p_bull_series(s, filtered=False)
        assert pb.between(0.0, 1.0).all()
        # Bull regime (second half) carries higher P(Bull) than the bear half.
        assert pb.iloc[len(pb) // 2:].mean() > pb.iloc[: len(pb) // 2].mean()

    def test_nd_trains_and_predicts(self) -> None:
        df = _two_regime_obs()
        hmm = train_macro_risk_hmm(df)
        assert hmm.bull_state in (0, 1)
        pb = hmm.p_bull_series(df, filtered=False)
        assert len(pb) == len(df)
        assert pb.between(0.0, 1.0).all()

    def test_nd_bull_state_tracks_market_dim(self) -> None:
        # With macro dims present, Bull is still the high-market-return regime.
        df = _two_regime_obs()
        hmm = train_macro_risk_hmm(df)
        pb = hmm.p_bull_series(df, filtered=False)
        assert pb.iloc[len(pb) // 2:].mean() > pb.iloc[: len(pb) // 2].mean()

    def test_p_bull_latest_scalar(self) -> None:
        df = _two_regime_obs()
        hmm = train_macro_risk_hmm(df)
        val = hmm.p_bull_latest(df)
        assert 0.0 <= val <= 1.0
