# Session Handoff — 12-07-26 (written 13-07 AM)

**Driving events:** full database audit after a failed pipeline run → found the
paperlog still starved and Gemini both format-drifting AND silently costing
~$2/run in thinking tokens. Day spent on cost surgery, data repairs, and the
scanner's laptop-OOM pressure.

- **Branch:** `main`
- **HEAD at handoff:** `00613f1` — **pushed to origin** (GitHub backup current)
- **Tests:** 609 green (`pytest -q`), RC=0
- **Working tree:** clean

---

## 1. Paperlog reality check — STILL starved, item-1 slips to ~late August

Hard facts from the DB audit:
- `source='daily'` rows exist for only THREE dates ever (Jun 19/25/30 × 356);
  `outcome_filled` = 0 everywhere. No daily row since Jun 30.
- Root cause chain: the old box's scheduler only ever ran `crawl_hose`;
  full_pipeline runs were manual; the new box had NO task at all until
  07-07; since then every full_pipeline attempt died to Ctrl+C
  (07-07 16:28, 12-07 10:24 — task result 0xC000013A = console kill).
- `QuantV6-FullPipeline` scheduled task (15:30 Mon–Fri, conda stock,
  `main.py --task full_pipeline`) is registered and Ready on this box.
  **13-07 15:30 is the next shot. Box on. Do not Ctrl+C the window.**
- Item-1 rank-sleeve verdict (n≥60): realistically **~late August** (3
  sleeve rows/day once dailies flow + 21-day settlement).

**New escape hatch:** `python main.py --task inference_only` (commit
`c9a76b6`) = daily inference + paperlog row + tranche exit alerts on
already-crawled data. Skips the 25-min crawl AND the per-article sentiment
pass. Use it when you've already run `crawl_hose` manually — no more reason
to kill the long pipeline.

## 2. Gemini cost surgery — $2/run → ~$0.30/run

- **The leak:** gemini-2.5-flash enables *thinking* by default; thinking
  tokens bill at the OUTPUT rate (~8× input). ~300 scoring calls/run ×
  1-3k invisible tokens ≈ $1.50-2.00/run. Fixed in `1d76c97`:
  `thinking_budget=0` on per-article scoring (deterministic JSON extraction
  needs none). Arbitrator batch keeps thinking (1-2 calls/day, ~$0.05).
- **Format drift (07-07 incident):** Gemini started returning 200s that fail
  strict `json.loads` (trailing prose; reason string broken by unescaped
  quotes). Crawler hardened with `parse_gemini_json` ladder (`0e3f631`);
  arbitrator batch parse got the same `raw_decode` first-object fallback
  (`_parse_llm_json`, `c9a76b6`).
- **Sentiment backfill DONE:** 176/176 fallback rows recovered (Jun 18-19 +
  May 503 damage) — `hist_sentiment_llm_labeled` has ZERO fallback rows for
  the first time since June. Total spend ≈ $1.20 (of which ~$1 was the
  old thinking-enabled process — killed and rerun post-fix at ~$0.10).
  Script got `--since` (incident isolation) + repo-root bootstrap (`014f4d7`).
- **Cost map now:** full_pipeline ≈ $0.30/run (~$6-7/mo daily);
  inference_only ≈ $0.02/run; scanner $0; backfills pennies.

## 3. Corporate-action guard on paperlog returns (`a95ae1b`)

Parquet closes are UNADJUSTED → KLB's 02-07 ~30% stock dividend booked a
phantom −22.6% `ret_3d`. Now: any single-session close-to-close move >|10%|
(HOSE band ±7%) inside a return window = CA-contaminated → short fill
skipped; matured long window settles `ret_20d=NULL` + `outcome_filled=TRUE`
(permanently excluded from sleeve/audit math, no infinite retry). New
`price_lookup.closes_between` + `has_ca_gap` (pure). KLB's phantom row
repaired to NULL. KLB Jun 19/25/30 rows will settle as excluded at maturity.

## 4. Data repairs (one-offs, done)

- `trade_history`: 6 pre-fix phantom `telegram_id='cron'` BUY rows purged
  (the whole table was only these) — backup at
  `backups/trade_history_cron_backup_20260712.csv`.
- `foreign_flow_daily.parquet`: dropped the leaky 01-07 partial-session rows
  (1627→1221); Jul 7/9/10 settled EOD snapshots kept.
- `macro_daily.parquet`: refreshed through 10-07 (3339 rows). Was "stale
  since Jun 22" simply because nothing ever invoked the crawl — auto-heals
  daily once full_pipeline actually completes (step 1b).

## 5. Scanner OOM pressure (`00613f1`) — RESTART BOT TO ACTIVATE

User observed laptop near-OOM during scan sessions. Cause: each 15-min scan
transiently allocates ~1GB (full-panel feature build × 2 horizons) inside the
long-lived bot process; freed memory squats in the allocator → RSS ratchets;
box measured at 15.4GB total / **2.1GB free**. Fix: post-scan `del` of panel
locals + `gc.collect()` + Windows working-set trim so pages return to the OS.

**Answer to "will it shut down my laptop": no** — worst case is thrash →
Windows kills the bot → supervisor relaunches (memory resets). Sawtooth,
not shutdown. But:
1. **Restart the bot** — it runs pre-fix code (started 12-07 12:06).
2. If still tight: `"intraday_scan_interval_min": 30` in settings.json.
3. **Suspicious: 6 python processes at 12-07 12:xx, two fat (228+331MB,
   started a minute apart) — possible DOUBLE BOT.** Check consoles; one
   supervisor instance only (Telegram getUpdates conflict + double RAM).

## 6. Plan hygiene (`ef9ad88`)

`active/` now holds ONLY `intraday-attack-scanner_PLAN_06-07-26.md` (Gate 5
live smoke pending — first window with the memory fix = 13-07 09:15).
Archived: sentiment-entry-paperlog (1068 daily rows exist = its own archival
condition), telegram-per-ticker-dispatch (implemented, 20 tests green).
Backlogged: notification-attribution-risk-tier (code-complete; nav-tier-cap
A/B deferred indefinitely). `settings.json` `intraday_scanner_enabled=true`
is now committed steady state.

## 7. Monday 13-07 checklist

1. **09:15** — scanner Gate-5 window (restart bot first for the memory fix).
2. **15:30** — `QuantV6-FullPipeline` fires: first complete EOD pipeline on
   this box. Costs ~$0.30 now. First daily paperlog row since Jun 30 +
   outcome backfill (Jun 19 rows settle, KLB excluded by the CA guard).
3. After it completes: `SELECT log_date, COUNT(*) FROM sentiment_entry_paperlog
   WHERE source='daily' GROUP BY 1` should show 13-07 × ~356, and Jun 19
   rows `outcome_filled=TRUE`.
4. Check double-bot (see §5.3).

## 8. Standing backlog (unchanged)

Foreign-flow real data subscription (1-2 months out; swap point =
full_pipeline step 1c). Leadership-features retry needs a NEW
pre-registration (de-saturated grid, 2 features). Graduated-attack
arbitration change stays GATED on the item-1 sleeve verdict (~late Aug).
main.py God-module decomposition (chronic). All paper until DSR ≥ 0.95.

## Commit trail (this session)

```
00613f1 fix(scanner): release scan transients - stop RSS ratchet on low-RAM boxes
ef9ad88 chore(process): plan audit + arm scanner flag
a95ae1b fix(paperlog): corporate-action guard on outcome backfill + data repairs
1d76c97 fix(cost): disable Gemini thinking on per-article sentiment scoring
c9a76b6 feat(ops): inference_only task + arbitrator JSON drift tolerance
014f4d7 fix(scripts): repo-root sys.path bootstrap in backfill_sentiment_503
0e3f631 fix(sentiment): tolerate Gemini JSON format drift + --since backfill filter  [07-07 evening]
```
All pushed to origin.
