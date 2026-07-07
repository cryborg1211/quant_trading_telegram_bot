# Session Handoff — 06-07-26

**Driving question:** picking up from 05-07's "always defend" investigation —
debated whether the arbitrator's argmax gate should be removed, landed on a
graduated-attack direction gated on live evidence; then built the eyes needed
to act on intraday moves instead of waiting for the 15:30 EOD crawl.

- **Branch:** `main`
- **HEAD at handoff:** `a71b370`
- **Tests:** 591 green (`pytest -q`), RC=0
- **Working tree:** clean
- **Bot status:** running (started 08:37 ICT this session) on **pre-scanner
  code** — needs a restart to pick up everything below.

---

## 1. τ=0.42 sandbox question — answered, no action

User asked: this morning's T+5 live run had 24/50 names survive admission but
plateau ~42% P(UP), below τ=0.45 (that's actually the T+20 gate — T+5 serves at
0.40/0.35). Ran an isolated sandbox sweep to check.

**Log:** `logs/backtest_t5_tau042_20260706.log` (`--no-save`, no artifacts
touched). **Report:** `process/general-plans/reports/t5-tau042-sandbox_06-07-26.md`

| up_thr | sig_thr | mean Sharpe | mean DD | total predUP | mean UP prec |
|---|---|---|---|---|---|
| 0.50 | 0.45 | +0.592 | −13.47% | 15,676 | 0.5582 |
| 0.45 | 0.40 | +0.629 | −13.00% | 218,361 | 0.4648 |
| **0.42** | **0.37** | **+0.644** | **−13.00%** | 518,921 | 0.4538 |
| 0.40 | 0.35 | +0.644 | −13.00% | 718,281 | 0.4468 |
| 0.35 | 0.30 | +0.644 | −13.00% | 1,039,723 | 0.4346 |

**Finding:** 0.42/0.40/0.35 rows are byte-identical per-seed — saturation
starts at 0.42, the τ floor stops binding there (top-5 selection binds
instead). τ=0.42 catches neither edge nor garbage; it's the same book as
incumbent GOLDEN (0.40). **No serve change.**

**Side finding (more interesting):** the 0.50 row is the first *backtest-level
proxy* evidence against the arbitrator's argmax kill — precision goes up
(0.558 vs 0.447) but Sharpe, Net PnL, and DD all get worse, and participation
drops ~46x. Feeds directly into §2.

---

## 2. Graduated-attack direction — decided, not yet actioned

Long debate (verbatim saved to orchestrator memory, `graduated-attack-decision.md`)
on whether to rip out the arbitrator's argmax kill (`P(UP)<~50% → 5d/20d DOWN
→ FULL EXIT`). Verdict: **"good armor, bad architecture."**

**Three sins of the killer:** (1) unvalidated — the 911-day GOLDEN evidence
belongs to an engine with NO argmax kill; (2) it silently dominates every
upstream gate — effective BUY threshold is `max(τ_admission, ~0.50)`, making
last week's whole universe/admission fix cycle moot for BUY *emission*; (3)
miscalibrated to the label geometry — triple-barrier pt=3σ makes P(UP)>0.5
structurally rare (T+20 best-in-357 today = 46.65%), so the BUY channel is
near-dead by construction. June 2026 had zero BUYs all month for exactly this
reason.

**Chosen direction (not yet built):** BUY eligibility from rank (pool member +
cleared τ_admission + top-k), argmax demoted to bear-veto-only. Intensity
throttled by already-validated overlays already sitting in the codebase idle
(regime policy, GARCH-HMM brake, risk-tier NAV caps). Candidate new guard:
sector cap in top-k (June's top-3 was all banks).

**Gate before touching code:** item-1 rank-sleeve counterfactual
(`scripts/analyze_rank_sleeve.py`, frozen criteria, `attack-narrow-market-preregistration_05-07-26.md`).
FAILURE → direction dead, ranking has no forward edge. SUCCESS → new
pre-registration for the arbitrator change, one shot. **First T+20 outcomes
settle after tonight's 15:30 cron; n≥60 verdict threshold ~mid-July.** Do NOT
touch the arbitrator before that verdict lands.

---

## 3. Intraday attack scanner — SHIPPED, kill-switch OFF, Gate 5 pending

Plan: `process/general-plans/active/intraday-attack-scanner_PLAN_06-07-26.md`
(**stays active** — this is the only open plan blocking on real-world
verification, not code). Commit: `82c9b62`.

**What it does:** every 10–30 min (default 15) during HOSE hours
(09:15–11:30 / 13:00–14:45 ICT, Mon–Fri), one SSI iBoard bulk snapshot call
(~407 tickers, same endpoint `foreign_flow_crawler` already uses) → builds a
provisional daily bar per ticker (`open/high/low/close` = snapshot fields,
÷1000 for the VND-thousands convention, `volume` = cumulative session qty) →
splices that bar **in-memory only** onto each ticker's 120-row parquet tail →
rebuilds features → rescores both T+5 and T+20 → applies the same
ADV-top-50 universe gate as serve → sends an event-only Telegram card (new
top-3 entrant / τ crossing / |Δ P(UP)| ≥ 2pp) to **both** chats (ADMIN +
USER, per explicit "scanner for both" instruction). First scan of the day
always sends one baseline card.

**Hard constraints (verified in the diff, not just claimed):** zero writes to
parquet / DuckDB / `sentiment_entry_paperlog` — this protects the item-1
experiment above from contamination — and no arbitrator/Gemini call in the
loop. Purely a human-facing "look here" signal, no dispatch.

**Config:** `TradingConfig.intraday_scanner_enabled` (default **False**),
`intraday_scan_interval_min` (default 15, clamped 10–30), `intraday_alert_delta_pp`
(default 0.02). New dependency: `requirements.txt` now pins
`python-telegram-bot[job-queue]==22.7` — the bot's conda `stock` runtime
already has apscheduler 3.11.2 (proven live in this morning's log), the
dev/test env intentionally does not (all 39 new tests are pure-function, no
live HTTP, pass without it).

**Gate 5 (live market-open smoke) is the only thing left** — needs the bot
restarted with the kill-switch flipped on, during a real trading window.

**To arm it:**
1. `config/settings.json` → `"intraday_scanner_enabled": true`
2. Restart: Ctrl+C the current `run_bot_forever.ps1` supervisor, then
   `conda activate stock; powershell -ExecutionPolicy Bypass -File run_bot_forever.ps1`
3. First live window: **tomorrow (07-07) 09:15 ICT.** Watch for
   `[intraday_scanner]` log lines; both chats should get a baseline card at
   the first in-window tick.
4. Rollback: flip the switch back to `false` + restart. No data/DB side
   effects either way (it's a pure reader).

---

## 4. Rider fix

`/suggest_buy20` hold-period label corrected 30→20 phiên — T+20 means 20
*trading* days, the help text and `BotCommand` description both said 30
(copy-paste artifact from the old recipe). Same commit `82c9b62`.

---

## 5. Process cleanup (commit `a71b370`)

Archived to `completed/`: `leadership-features-recipe_PLAN_05-07-26.md` (FAIL
verdict was already applied 05/06-07, just hadn't moved), `serve-universe-adv-alignment_PLAN_05-07-26.md`
(rollout+smoke done, was already live), `serve-admission-tranche-ab_PLAN_04-07-26.md`
(verdict applied 04-07). Context docs refreshed: `all-context.md` +
`tests/all-tests.md` had stale counts (473/238 tests, 21 files) — corrected
to 591/47.

**Still active, unconfirmed either way** (a future `vc-audit-plans` pass
should look at these): `notification-attribution-risk-tier_PLAN_02-07-26.md`,
`sentiment-entry-paperlog_PLAN_16-06-26.md`, `telegram-per-ticker-dispatch_PLAN_18-06-26.md`.
None of these were touched this session — carried over from 05-07's list,
still not resolved.

---

## 6. Open items / next session

1. **Restart the bot** — it's running pre-scanner code from this morning
   (08:37 ICT launch). Restart to load the scanner (§3) and the phiên fix
   (§4). Flip the kill-switch first if you want Gate 5 to run tomorrow
   morning.
2. **Item-1 sleeve verdict is the gating event for everything in §2.**
   `python scripts/analyze_rank_sleeve.py` (bot stopped — exclusive DuckDB
   lock) still prints INSUFFICIENT_DATA today; first settled T+20 rows
   appear after tonight's 15:30 cron backfill. n≥60 threshold ~mid-July —
   check back periodically, don't touch the arbitrator before then.
3. **`vc-audit-plans`** on the 3 stale-active plans in §5 — none confirmed
   done or blocked this session, just inherited from last time.
4. **Housekeeping carried over from 05-07, still untouched:**
   `macro_daily.parquet` stale since Jun 22; possible pre-fix phantom
   `user_id='cron'` rows in `trade_history`; foreign-flow real subscription
   (user-deferred, 1–2 months out).
5. **Leadership-features retry** needs a NEW pre-registration if ever
   revisited — de-saturated sweep grid + smaller feature set (2 features,
   not 4) per the verdict report's own recommendation. Not scheduled.
6. **Dev environment note:** git-bash was non-functional for parts of this
   session (`unexpected EOF` parser error, unrelated to any command content)
   — PowerShell was the working fallback both times it happened. If it
   recurs, don't debug the command; switch tool.

---

## Commit trail (this session)

```
a71b370 docs(process): 06-07 closeout - archive 3 plans, sandbox report, context refresh
82c9b62 feat(scanner): intraday attack scanner - monitoring-only, kill-switch OFF
432f0f4 feat(analysis): item-1 rank-sleeve counterfactual evaluator (frozen criteria)  [prior session, carried]
```
