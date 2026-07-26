# Foreign-Flow Data via SSI FastConnect — Backlog Plan

**Date**: 24-07-26
**Status**: 🔒 BLOCKED — user has no SSI trading account / FastConnect API access yet
**Origin**: user asked to research "SSI's new API"; found FastConnect Data (`DailyStockPrice` endpoint) exposes historical, date-range-queryable foreign buy/sell volume + foreign room — the exact gap that stalled the 01-07-26 foreign-flow research (see project memory: "SSI per-ticker: only 1 day exists, ADF/correlation both empty (n=0)").

## Precondition (external, not code)

1. User opens an SSI trading account — **fully online eKYC, ~3 min, free**, no branch visit. Needs chip-embedded national ID + phone camera + OTP.
2. User registers for FastConnect Data API on top of that account — **NOT self-serve**: branch visit or mailed documents to SSI, then SSI emails an approval link, then ConsumerID/ConsumerSecret/PrivateKey are generated via iBoard's API Service screen. **FastConnect Data itself is free**, but needs renewal every 3 months (hotline/email call).
3. Neither step has happened yet as of this session. Nothing below starts until the user has real credentials.

## What already exists (confirmed by reading the code this session — zero rework needed)

- `src/data/foreign_flow_crawler.py` — `data/foreign_flow_daily.parquet`, schema is source-agnostic (`date, ticker, foreign_buy_val, foreign_sell_val, foreign_net_val, foreign_buy_vol, foreign_sell_vol, foreign_remain_room_vol, prop_*, source, fetched_at`). Merge/dedup (idempotent on `(date, ticker)`, fresh wins) doesn't care which adapter filled a row.
- `src/features/flow_features.py` — `build_flow_features` only needs `ticker, date, close, volume, foreign_net_val` (+ optional `prop_net_val`). Fully reusable once fed real history.
- `scripts/eda_flow_features.py` — leakage check (`fetched_at` vs ATC close), ADF, lead-lag correlation. General; a backfill's `fetched_at` (today) is always safely after any historical `date`'s close, so it trivially passes the leak check.
- Current adapter (`fetch_ssi_hose_snapshot` → unofficial `iboard-query.ssi.com.vn`) stays as the live/today path — no need to replace it, FastConnect is additive for HISTORY.
- `src/data/etf_flow_proxy_crawler.py` (weak ETF proxy fallback) becomes unnecessary once real history lands — not broken, just superseded. No urgency to remove.

## What's new work, once unblocked

1. **New backfill adapter** (`src/data/foreign_flow_crawler.py` or a sibling module): implement the FastConnect token flow (`AccessToken` POST with ConsumerID/ConsumerSecret) + signed `DailyStockPrice` GET, paginated (`pageIndex`/`pageSize`), mapped into the EXISTING `_SCHEMA`, tagged `source="ssi_fastconnect_history"`. Needs real ConsumerID/Secret to test against — cannot be built/verified blind.
2. **Re-run `scripts/eda_flow_features.py` for real** — actual multi-year series instead of n=0. Decide for real (not a placeholder) whether foreign flow / knife-catch divergence is additive.
3. **If EDA clears a bar**: walk-forward backtest with `flow_features` wired into `pipeline.build_features` (recipe-version bump, full retrain gate — same discipline as every other feature-recipe change this session).
4. **Full system re-pass** (explicit user ask, not just the new feature): re-test the whole system once flow data is live, not just the isolated feature.
5. **Full documentation deep-dive** (explicit user ask): go through every doc in `process/context/` — not just router files — and update anything stale, not just what this feature touches. Write up future-work / next-steps at that point.

## Trigger to reactivate this plan

Either: (a) the user says foreign-flow / SSI credentials are ready, or (b) a future session finds real rows in `data/foreign_flow_daily.parquet` with `source != "ssi_iboard_live_snapshot"`. Either signal means: start at item 1 above, not from scratch.

## Notes

- Registration/renewal is a recurring 3-month operational chore, not one-time — worth a calendar reminder once live, or the crawler adapter should fail loudly (not silently degrade) on an auth error so a lapsed key gets noticed fast.
- Sources for the API research: [FastConnect API - SSI](https://guide.ssi.com.vn/ssi-products), [Danh sách các API | FastConnect Data](https://guide.ssi.com.vn/ssi-products/tieng-viet/fastconnect-data/danh-sach-cac-api), [Đăng ký dịch vụ](https://guide.ssi.com.vn/ssi-products/tieng-viet/dang-ky-dich-vu), [Mở tài khoản trực tuyến](https://iboard.ssi.com.vn/open-account/).
