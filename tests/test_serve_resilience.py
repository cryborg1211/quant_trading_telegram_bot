"""Serve-path resilience — a missing SECONDARY horizon must NOT abort a command.

Guards the regression where /suggest_buy5 died with "horizon=20 not found"
because daily_inference / verify load the other horizon for the arbitrator
cross-check.  Imports `main` (heavy ML stack) lazily and skips if it can't be
imported in a bare environment (e.g. CI without config/.env).
"""
import datetime

import pytest


def _import_main():
    try:
        import main
        return main
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"main not importable in this env ({type(exc).__name__}: {exc})")


def test_missing_t20_does_not_crash_verify(monkeypatch):
    import polars as pl
    main = _import_main()

    # Only the T+5 brain is trained; the T+20 artifact is missing.
    def fake_predict(latest_df, horizon):
        if int(horizon) == 20:
            raise FileNotFoundError("v3_ensemble_20d.joblib not found")
        return ({"AAA": [0.2, 0.3, 0.5]}, {"pnl_threshold_tau": 0.5}, None, [], {"AAA": True})

    monkeypatch.setattr(main, "predict_v3_horizon", fake_predict)
    monkeypatch.setattr(main, "evaluate_trades_batch", lambda hp, cands: ({}, {}))
    monkeypatch.setattr(main, "mr_score_tickers", lambda tks: {})
    monkeypatch.setattr(
        main.Alpha360Generator, "load_live_ohlcv_window",
        lambda self, tickers=None, window_rows=120: pl.DataFrame({
            "ticker": ["AAA"], "date": [datetime.date(2024, 1, 2)],
            "open": [10.0], "high": [10.5], "low": [9.8], "close": [10.2], "volume": [100000],
        }),
    )

    out = main.verify_single_ticker("AAA")
    assert isinstance(out, str) and "AAA" in out
    # Degraded gracefully to the 5d view — did NOT surface a model error / abort.
    assert "Lỗi mô hình" not in out


def test_load_v3_bot_passes_horizon_through(monkeypatch):
    """predict_v3_horizon(df, H) must load the H-d artifact, not a hardcoded one."""
    main = _import_main()
    seen = {}

    class _FakeBot:
        metadata = {"feature_recipe_version": "v1.0", "tb_horizon": 5}
        tabular_features = ["close_fd_xsz"]
        up_threshold = 0.5

        def card(self):
            return "fake"

    def fake_loader(h):
        seen["h"] = h
        return _FakeBot()

    monkeypatch.setattr(main, "_load_v3_bot", fake_loader)
    monkeypatch.setattr(main, "_compute_v3_features",
                        lambda df, feats, fdd: __import__("pandas").DataFrame())
    # empty features → early return, but _load_v3_bot must have been called with H
    main.predict_v3_horizon(__import__("pandas").DataFrame(), horizon=20)
    assert seen.get("h") == 20
