# Foreign-Flow Data via SSI FastConnect — Backlog Plan

**Date**: 24-07-26 (unblocked + backfill launched 07-08-26)
**Status**: 🟡 IN PROGRESS — credentials obtained, adapter built+live-verified, full-universe backfill running (kill-proof background job, launched 07-08-26 22:14, ~5h projected). Not yet done: quality comparison, EDA rerun, everything after.
**Origin**: user asked to research "SSI's new API"; found FastConnect Data (`DailyStockPrice` endpoint) exposes historical, date-range-queryable foreign buy/sell volume + foreign room — the exact gap that stalled the 01-07-26 foreign-flow research (see project memory: "SSI per-ticker: only 1 day exists, ADF/correlation both empty (n=0)").

## Precondition (external, not code) — ✅ DONE 07-08-26

1. ~~User opens an SSI trading account~~ — done.
2. ~~User registers for FastConnect Data API~~ — done. Credentials were briefly pasted into chat (ConsumerID/Secret/PublicKey/PrivateKey) — flagged as exposed, user rotated via iBoard's API Service screen, new ConsumerID/ConsumerSecret placed directly in `.env` (`Consumer_Key`/`ConsumerSecret_Key`) going forward, never re-pasted.

## What already exists (confirmed by reading the code this session — zero rework needed)

- `src/data/foreign_flow_crawler.py` — `data/foreign_flow_daily.parquet`, schema is source-agnostic (`date, ticker, foreign_buy_val, foreign_sell_val, foreign_net_val, foreign_buy_vol, foreign_sell_vol, foreign_remain_room_vol, prop_*, source, fetched_at`). Merge/dedup (idempotent on `(date, ticker)`, fresh wins) doesn't care which adapter filled a row.
- `src/features/flow_features.py` — `build_flow_features` only needs `ticker, date, close, volume, foreign_net_val` (+ optional `prop_net_val`). Fully reusable once fed real history.
- `scripts/eda_flow_features.py` — leakage check (`fetched_at` vs ATC close), ADF, lead-lag correlation. General; a backfill's `fetched_at` (today) is always safely after any historical `date`'s close, so it trivially passes the leak check.
- Current adapter (`fetch_ssi_hose_snapshot` → unofficial `iboard-query.ssi.com.vn`) stays as the live/today path — no need to replace it, FastConnect is additive for HISTORY.
- `src/data/etf_flow_proxy_crawler.py` (weak ETF proxy fallback) becomes unnecessary once real history lands — not broken, just superseded. No urgency to remove.

## What's new work

1. ~~**New backfill adapter**~~ — ✅ DONE 07-08-26 (`868d45b`, `59b1882`). FastConnect token flow (ConsumerID+ConsumerSecret only, no PrivateKey/RSA needed for Data), `DailyStockPrice` GET, mapped into `_SCHEMA` (+6 new columns: block-deal + aggressor-side trade fields, captured free from the same response). Live-verified: real auth, real historical rows, two undocumented API constraints found (30-calendar-day max range per call, real history depth starts ~2020-02 not further back) and handled (`_chunk_date_range`). 27 tests, all HTTP mocked.
2. **Full-universe backfill** — 🟡 RUNNING (launched 07-08-26 22:14, kill-proof background, ~5h projected for 359 tickers × 2020-01-01→today). `scripts/backfill_ssi_foreign_flow.py` / `run_ssi_foreign_flow_backfill.ps1`.
3. **Compare data QUALITY between SOURCE 1 (iBoard live snapshot) and SOURCE 2 (FastConnect history)** — explicit user ask (07-08-26), do this once the backfill completes. Both sources now have overlapping `(date, ticker)` rows for recent dates (SOURCE 1 from the daily production cron, SOURCE 2 from the backfill). Check: do `foreign_buy_val`/`foreign_sell_val`/`foreign_net_val` agree on the same (date, ticker) between the two sources, or diverge? If they diverge, which one is more trustworthy (SOURCE 2 is the OFFICIAL documented API vs SOURCE 1's reverse-engineered endpoint — prior is SOURCE 2 wins, but verify, don't assume). This should inform whether SOURCE 1's daily cron stays in production going forward or gets replaced by a daily FastConnect call.
4. **Re-run `scripts/eda_flow_features.py` for real** — actual multi-year series instead of n=0. Decide for real (not a placeholder) whether foreign flow / knife-catch divergence is additive. Also worth a first look at whether the NEW block-deal/aggressor-trade fields (item 1) show any signal, even though they weren't the original ask.
5. **If EDA clears a bar**: walk-forward backtest with `flow_features` wired into `pipeline.build_features` (recipe-version bump, full retrain gate — same discipline as every other feature-recipe change this session).
6. **Full system re-pass** (explicit user ask, not just the new feature): re-test the whole system once flow data is live, not just the isolated feature.
7. **Full documentation deep-dive** (explicit user ask): go through every doc in `process/context/` — not just router files — and update anything stale, not just what this feature touches. Write up future-work / next-steps at that point.

## Trigger to reactivate this plan

Backfill (item 2) finishing is the next trigger — resume at item 3 (quality comparison), not from scratch.

## Notes

- Registration/renewal is a recurring 3-month operational chore, not one-time — worth a calendar reminder once live, or the crawler adapter should fail loudly (not silently degrade) on an auth error so a lapsed key gets noticed fast.
- Sources for the API research: [FastConnect API - SSI](https://guide.ssi.com.vn/ssi-products), [Danh sách các API | FastConnect Data](https://guide.ssi.com.vn/ssi-products/tieng-viet/fastconnect-data/danh-sach-cac-api), [Đăng ký dịch vụ](https://guide.ssi.com.vn/ssi-products/tieng-viet/dang-ky-dich-vu), [Mở tài khoản trực tuyến](https://iboard.ssi.com.vn/open-account/).
