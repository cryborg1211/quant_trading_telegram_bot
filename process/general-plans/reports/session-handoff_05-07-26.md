# Session Handoff — 05-07-26

**Driving question:** "Why does my model always alert *defend* when the last
month was a good uptrend?" → root-caused, A/B-tested, and acted on. Serve
universe widened to match the validated backtest; bot-uptime supervisor added.

- **Branch:** `main`
- **HEAD at handoff:** `62e9767`
- **Tests:** 546 green (`pytest -q`), RC=0
- **Working tree:** clean after this handoff + `run_bot_forever.ps1` land (see §6, §8)

---

## 1. The June "all-defend" question — answered

**Root cause (evidence, not guess):** serve admission is an *absolute* per-name
gate — `meta_gate = {ticker: P(UP) >= up_threshold (0.45)}`. Every logged June
`daily_inference` run had zero names clear it → fallback "MONITORING-ONLY /
THỊ TRƯỜNG YẾU" card, zero tradable BUYs all month. The model's cross-sectional
*ranking* was fine (it ranked SSB #1; SSB finished June **+16.2%**) — the gate,
not the ranking, blocked everything.

**But the model was mostly right.** June breadth was actually negative:
331-ticker equal-weight median **−2.5%**, VN30-ish mean **−0.78%**. The VNINDEX
"rally" was narrow bank leadership. Even the unrestricted tranche backtest lost
**−1.26%** in its June-2026 row. Defensiveness in a negative-breadth month is
correct behavior, not a bug.

**Not culprits (confirmed, left untouched):** GARCH-HMM brake (benign, P(Bull)
≈0.975), sentiment arbitrator (only softens SELL→HOLD, never blocks a fresh
BUY), regime sizing.

---

## 2. Serve-admission A/B — verdict: NO rule change

Plan: `serve-admission-tranche-ab_PLAN_04-07-26.md` ·
Report: `serve-admission-ab-result_04-07-26.md` · Commit: `91fe7da`

Added an opt-in `absolute_gate` admission mode to the tranche engine (default
`cross_sectional`, byte-identical) + `zero_candidate_days` diagnostic. Ran a
**pre-committed** 7-config × 4-seed grid (DSR n_trials=28) on the frozen T+20
checkpoint.

| | Sharpe | MaxDD | Zero-cand days |
|---|---|---|---|
| Baseline (cross-sectional) | +0.629 | −13.00% | 0 |
| Serve-mirror @0.45/N5 | +0.592 | −13.47% | 8/911 (0.9%) |
| Best gate @0.35/N5 | +0.644 | −13.00% | 0 — fails DSR 0.22 / PBO 85.6% |

**Verdict (mechanical, per the plan's pre-committed rule):** absolute gate is
NOT measurably costly OOS — best gate config +0.015 Sharpe, no DD payback,
0.9% blocked days, winner statistically ungated. **June files as an acceptable
false-defensive. No serve admission-rule change warranted.**

**Caveat recorded:** the backtest mirrors the gate but NOT serve's VN30-only
universe, so June's live 100%-fallback regime didn't reproduce (0.9% here).
Gate × universe interaction untested — which motivated §3.

---

## 3. Serve universe alignment — SHIPPED & LIVE

Plan: `serve-universe-adv-alignment_PLAN_05-07-26.md` · Commits: `717c756`
(impl, kill-switched), `810b10b` (flip to live after smoke)

**What & why:** all validated backtest evidence runs a *dynamic top-50
trailing-20d ADV* universe (`WalkForwardConfig.liquid_top_n=50`), but serve
alone screened to the static hardcoded `_VN30_UNIVERSE` — serve was deviating
from its own validated config. Static-list rot measured: 6/30 VN30 names had
fallen out of the current ADV top-50 (incl. SSB); 6 of June's 10 best names sat
outside VN30 (MSB +12.3%, PVD +8.5%, TCX +8.4%, …).

**Delivered:**
- `src/trading/serve_universe.py::liquid_universe()` — pure, Polars-native,
  deliberate NO `shift(1)` (live snapshot, not a historical walk — documented).
- `main._resolve_candidate_universe()` — degrade precedence: empty ADV →
  WARNING+VN30, exception → ERROR+VN30, invalid mode → warning+VN30. Never
  crashes `daily_inference`. `_select_candidates` signature UNCHANGED.
- Config: `CONFIG.trading.serve_universe_mode` (`adv_top_n` | `vn30`),
  `serve_liquid_top_n=50`, `serve_adv_window=20`. `[VN30Gate]` → `[UniverseGate]`.
- 11 new tests; VN30 frozenset kept as permanent fallback.

**NOT literal VNX50 index** — dynamic ADV-top-50 by design (static index lists
rot the same way VN30 did; ADV self-maintains and matches the backtest exactly).
A `"vnx50"` official-constituent mode is a cheap future option if ever wanted;
recommended against as default.

**Smoke PASSED** (user-run, 05-07 12:25 ICT, both modes on identical data):
- vn30: `mode=vn30 size=30`, fallback Top-3 `[SSI, VJC, HDB]`
- adv_top_n: `mode=adv_top_n size=50`, fallback Top-3 `[SSI, VJC, MSB]` (MSB =
  non-VN30 name entering) — pools differ, no exceptions. Kill-switch flipped to
  `adv_top_n` = **LIVE**.

---

## 4. Sentiment "high score but always rejected" — answered

**By design, sentiment is a brake, not a gas pedal.** Chain is one-way:
`ensemble P(UP) → τ-gate → arbitrator/sentiment → dispatch`. Names die at the
gate *before* sentiment is consulted. Inside the arbitrator, good news can only
soften a model SELL→HOLD — it can never *create* a BUY the models didn't give.
Built one-armed because VN news sentiment has no point-in-time history →
un-backtestable → gets no capital authority until forward evidence earns it.

**Paperlog early read (05-07, `paperlog-early-read.py`):** 1039/1086 rows have
T+3 returns; **T+20 all still pending** (first maturities ~Jul 7).

| Bucket | n | avg T+3 | % positive |
|---|---|---|---|
| Sentiment ≥0.7 | **5** | +2.77% | 60% |
| Sentiment <0.7 | 1025 | −0.02% | 39% |
| Model said SELL | 995 | −0.01% | 39% |

The +2.77% is a hint but **n=5 = anecdote, not evidence** (deep-dive Q5 warned
exactly this). Model's SELL-labeled names went nowhere at T+3 → its June
defensiveness is validated on the short horizon. **Re-run mid-July** when T+20
fills and hi-sentiment n grows.

---

## 5. persist-gate fix (background task, landed)

Commit `dd2174d` — `daily_inference(persist=False)` previously opened the ledger
DuckDB read-write (via `PortfolioManager()`) *before* the `if persist:` guard,
so any preview/smoke needed the exclusive lock and crashed while `run_bot.py`
held it. Now `persist=False` never touches the ledger DB. (Fix spun off as a
background session from this one and committed itself.)

---

## 6. Bot uptime supervisor — NEW, uncommitted

`run_bot_forever.ps1` (repo root). Windows equivalent of the Linux
`deploy/quant-v6-bot.service` (`Restart=on-failure`) — relaunches `run_bot.py`
whenever it exits, 10s backoff. Motivation: log shows recurring
`NetworkError: httpx.ConnectError: getaddrinfo failed` (DNS/wifi blips); once
`run_polling` throws an unhandled one the process just ends with nothing to
restart it.

Ruled out as the "1h death" cause (with evidence): Windows sleep (Never on AC),
Task Scheduler (none), code self-timeout / job-queue stop (none). Exact "1h"
number unconfirmed — 83 manual restarts over 2 months pollute the cadence — but
the fix survives any exit cause.

**Run:** `conda activate stock; powershell -ExecutionPolicy Bypass -File run_bot_forever.ps1`
(instead of `python run_bot.py`). ONE instance only or Telegram getUpdates
Conflict.

---

## 7a. Foreign-flow crawler — 5xx fix landed, rest PARKED

Commit `62e9767` — `foreign_flow_crawler` now retries transient 5xx (new
`_is_retryable_5xx` predicate; docstring had promised it, code never did →
a 502/503 killed the whole day's crawl, a permanent hole since this source has
no backfill). 4xx still propagates. +8 tests (first coverage on this module).

**User decision:** crawler is otherwise fine as-is — will subscribe to a better
foreign-flow data source later rather than harden the free SSI-iBoard line.
So the following were deliberately NOT done: evening second-crawl, zero-row
alert, tz-aware timestamps (M2), partial-write guard (M3).

**Data reality (05-07 check):** `foreign_flow_daily.parquet` has exactly 1 day
(2026-07-01), captured 14:30 = pre-15:00 buffer = partial-session/leaky. Nothing
since (step 1c only wired this session; bot ran old code). **Backfill is
impossible** — SSI iBoard is a live snapshot with no date parameter, and every
historical API (vnstock/TCBS/Fireant/VNDirect/CafeF) is dead (module docstring).
When the better source arrives, the swap point is clean: `update_foreign_flow_daily()`
in `full_pipeline` step 1c is the only wiring; `build_flow_features` is still
collection-only (not in the recipe), so it drops in without touching serve.

## 7. Open items / next session

1. **Restart `run_bot.py`** (preferably via `run_bot_forever.ps1`) — the live
   process is still running pre-session code. Restart loads: new ADV universe,
   phantom-trade fix, horizon-label fix, attribution card, persist-gate fix.
2. **Commit `run_bot_forever.ps1`** — orchestrator left it uncommitted pending
   user OK.
3. **Archive completed plans (UPDATE PROCESS):** `serve-admission-tranche-ab`
   and `serve-universe-adv-alignment` are both verification-complete →
   `completed/`. `notification-attribution-risk-tier` stays active (its own A/B
   still pending). Also stale in active: `sentiment-entry-paperlog`,
   `telegram-per-ticker-dispatch` (shipped weeks ago). Run `vc-audit-plans`.
4. **Mid-July: re-run `analyze_sentiment_paperlog.py`** — T+20 rows mature ~Jul
   7; decide if the hi-sentiment edge survives real sample size.
5. **The real "predict better in uptrends" lever** is feature work, not gating:
   momentum / flow / breadth features → recipe bump + retrain. Foreign-flow
   collection is already accumulating for exactly this. Choppy-default-penalty
   direction-conditioning is a secondary regime-overlay lever.
6. **Housekeeping:** `macro_daily.parquet` stale since Jun 22 (fix macro crawl);
   possible pre-fix phantom `user_id='cron'` rows in `trade_history` to purge
   (deep-dive item 2); optional `hardening main()` so `run_polling` retries
   transient network errors at the source (supervisor covers it for now).
7. **Foreign-flow: subscribe to a real data source** (user-deferred) — replaces
   the free SSI-iBoard line at `full_pipeline` step 1c. Drop the lone leaky
   07-01 partial row when starting clean accumulation. See §7a.

## Commit trail addendum (post-handoff-v1)

```
62e9767 fix(data): retry 5xx in foreign_flow_crawler to match its own contract
```

---

## Commit trail (this session)

```
dd2174d fix(serve): persist=False no longer opens ledger DuckDB      [bg task]
810b10b chore(serve): flip serve universe to adv_top_n after smoke pass
717c756 feat(serve): dynamic top-50 ADV candidate universe (staged, kill-switched)
91fe7da feat(backtest): opt-in absolute-gate admission A/B + verdict report
a8ca188 docs(process): deep-dive + retrain reports; risk-tier + serve-admission plans
12f4166 chore(models): 02-07 T+5/T+20 retrain artifacts; ignore catboost_info
d18ae22 feat(dashboard): TV-style volume + foreign-flow panes, oversight row
bf4cae2 feat(trading): discrete risk-tier NAV cap (backtest, off-default) + card attribution
66fc83d feat(data): foreign/prop flow data line (SSI iBoard + ETF proxy)
38b357c fix(reports): real horizon label on fallback card + empty post-arbitration book
8ed8d4d fix(bot): gate /suggest_buy* preview runs out of the shared trade ledger
```
(Plus `dd2174d`'s parent chain started from the dirty tree committed as
`8ed8d4d..a8ca188` at session start.)
