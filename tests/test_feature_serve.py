"""Live feature-serving path — full OHLCV + multi-row window → V4 features.

Guards the regression where the serve path fed a tail-1, open/high/low-dropped
frame to build_features, so `_compute_v3_features` raised
'missing OHLCV columns' (and would have produced an empty panel even if it
hadn't).  Requires live parquets + the ML stack; skips gracefully otherwise.
"""
import glob

import pytest


def _need_parquets(n: int = 10) -> list[str]:
    files = sorted(glob.glob("data/ohlcv_*.parquet"))
    if len(files) < n:
        pytest.skip(f"needs >= {n} live parquets for a non-degenerate cross-section "
                    f"(found {len(files)})")
    return files


def test_live_window_has_full_ohlc_and_stat_features():
    files = _need_parquets()
    try:
        import polars as pl
        from src.features.alpha360_generator import Alpha360Generator
        import main
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"serve stack not importable ({type(exc).__name__}: {exc})")

    tickers = [f.split("ohlcv_")[1].rsplit(".", 1)[0].upper() for f in files][:30]
    win = Alpha360Generator().load_live_ohlcv_window(tickers=tickers, window_rows=160)

    # (1) RAW multi-row OHLCV window — full suite, not tail-1.
    need = {"ticker", "date", "open", "high", "low", "close", "volume"}
    assert need.issubset(set(win.columns)), f"missing {need - set(win.columns)}"
    assert win.height > win["ticker"].n_unique(), "expected a multi-row window, got ~tail(1)"

    # (2) The open/high/low-dependent statistical features now materialize live.
    feats = ["close_fd_xsz", "hl_range_ratio_xsz", "gap_risk_xsz"]
    out = main._compute_v3_features(win.to_pandas(), feats, 0.4)
    assert list(out.columns) == feats
    assert len(out) > 0, "empty feature panel — window lacked history/columns"
    import numpy as np
    assert np.isfinite(out.to_numpy()).all(), "NaN/inf in OHLC-derived stat features"
