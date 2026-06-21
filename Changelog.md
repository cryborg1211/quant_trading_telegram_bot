# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## 2026-06-21

### Local Dashboard — P2 live-render gate closed
- Added `tests/test_dashboard_app_smoke.py`: deterministic Streamlit `AppTest`
  boot smoke for `dashboard/app.py`. Verifies all six tabs
  (MUA / GIỮ / BÁN / Verify / Audit / Settings) render with no uncaught
  exception and no per-tab error boundary, plus a one-holding GIỮ render path.
  Heavy seams (`daily_inference` / `run_post_mortem` / `portfolio_list` /
  `price_lookup`) are stubbed at the tab use-sites, so the smoke needs no
  models, parquet, DuckDB, Gemini, or Telegram.
- Installed `streamlit` (1.58.0) per `requirements_dashboard.txt`; confirmed a
  real `streamlit run --server.headless` boot is clean (health `ok`, no
  traceback).
- Test suite: 249 → **251 passed**.
- Updated `process/features/local-dashboard/HANDOFF.md`: P2 gate marked CLOSED;
  NEXT now points to P3 (launcher).
