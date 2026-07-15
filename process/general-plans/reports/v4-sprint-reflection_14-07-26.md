# V4.0 Post-Sprint Reflection Report
**Period:** June 1 – July 14, 2026 · **Author:** Principal Quant AI / Lead Architect · **Status:** paper-trading (DSR gate unbroken)

---

## 1. The Graveyard (What we killed and why)

| Hypothesis | Verdict | Evidence | Lesson |
|---|---|---|---|
| **Leadership features** | KILLED — PBO 18.9% > 10% gate, despite raw Sharpe lift | `backtest_leadership_20260705.log`; reverted to GOLDEN | Raw Sharpe improvements are cheap; PBO-survivable ones are not. The gate did its job. |
| **Absolute admission gate** (τ floor replacing cross-sectional top-N) | REJECTED — no MaxDD improvement OOS | A/B `91fe7da` (June defend-bias program) | The June "always defend" failure was a *market-breadth* fact (narrow bank rally), not a gate defect. Caveat on record: VN30-universe × gate interaction untested. |
| **Macro features in the GBM** | KILLED — MaxDD −13.0%→−18.5%, PBO 42.7%→87% | Macro-integration P4 A/B (`8fc6dd7`) | Market-level features are constant across the cross-section → ~zero ranking lift, pure overfit surface. Predicted by design, confirmed by A/B. Macro belongs in the regime overlay (kept), never the ranker. |
| **Per-tranche PT/SL sigma barriers** | FALSIFIED — PT+SL −5.7%, SL-only +29.2%, no-barrier +45.5% | Tranche barrier A/B (June) | The edge is right-tail skew at ~50% WR. Any position-level stop amputates the tail that pays for everything. DD control must be portfolio-level (sizing, exposure), never position-level. |
| **T+5 as primary horizon** | SHELVED — retrain PBO 43.1% FAIL (T+20: 3.0% PASS) | 07-02 retrain sweep | Same recipe, shorter horizon → 14× worse overfit signature. T+20 stays primary. |
| **Grid rebalance mode** | STRUCTURALLY UNFIT | ~45 correlated entry dates → market beta dominates a +2%/trip edge | Backtest architecture is a hypothesis too. Tranche staggering is the strategy, not a detail. |

**Meta-lesson:** every kill above was bought with a pre-committed gate (PBO ≤ 10%, DSR ≥ 0.95, A/B vs GOLDEN). Zero reverts shipped to serve. The graveyard is the system working.

**Standing verdict:** nothing yet passes DSR (best: 0.45 after regime sizing, vs 0.95 hurdle). The system is a *paper* system until that changes. No live capital.

## 2. The Arsenal (What actually works)

- **Tranche book (AFML staggered cohorts, H=30):** +45.5% net / 3.9y OOS baseline; per-day edge peaks at H=30. The load-bearing architecture.
- **Regime-conditional sizing (SHIPPED, serve default-ON):** MaxDD −23.3%→−16.9%, Sharpe +0.73→+0.88, DSR 0.35→0.45. NO_TRADE {0,7} → cash; PENALTY {1,6} → 0.5×. Single source of truth `regime_policy.py` imported by both backtest and serve — the parity pattern, not a copy-paste.
- **Serve universe ADV alignment (SHIPPED):** dynamic trailing-20d ADV top-50 replaced the rotting static VN30 list. Train/serve parity extends to *universe definition* — a stale universe is silent skew. (Residual debt: dead `universe_filter` key still lies in settings.json.)
- **Cross-sectional parity discipline:** every serve path (`/verify`, `/suggest_sell`, portfolio guard) predicts over the FULL universe before slicing — `_xsz` ranks degenerate otherwise. Now an enforced house rule.
- **Exit parity (signal ledger):** dispatched cohorts tracked in trading sessions, exits alerted at D+H — live behavior no longer silently diverges from the simulated book after the first horizon.
- **HMM overlays (kept):** macro-HMM P(Bull) risk-awareness (OOS bull-days 89%→57%); GARCH-HMM exposure brake = OOS loss-mitigation KEEP on T+5.
- **Intraday attack scanner (LIVE as of 14-07):** in-memory, monitoring-only, zero look-ahead. 30-min feature-rebuild cadence observed running through HOSE hours on 14-07 (kill-switch flipped ON; Gate 5 smoke effectively in progress). *[Corrected 14-07 PM — an earlier draft said SHIPPED-DARK.]*
- **GARCH-HMM exposure brake (LIVE in serve — corrected 14-07 PM):** `src/bot/garch_brake.py` → `_dispatch_signals` weight multiplier, default ON, floor 0.2, fail-open 1.0, `garch_scalar` stamped per signal. Proven reactive: floored 0.20 on 01-07 (late-June vol spike). Known blind spot: read ~0.99 through the 07-07→13-07 low-vol grind-down — GARCH sees vol shocks, not drift. Open: shipped floor 0.2 vs sweep-optimal 0.1 undocumented; sweep seeds 2/3 incomplete.
- **Portfolio guard (SHIPPED 14-07):** the VHM −6.2%/HDB −2.9% silent-bleed incident → 5 deterministic EOD triggers + CA shield + event-only per-user DM. Canonicalized the entry-price scale rule (`normalize_entry_price_vnd`, <1000 ⇒ ×1000) that three code sites disagreed on. First live sweep: tonight's cron.
- **Gemini cost surgery ($2.00 → ~$0.30/run expected):** thinking-off on extraction (12-07) + GA model pin all config layers + interrupt-safe chunked persistence (14-07). Reasoning sites keep thinking BY DESIGN.
- **Test discipline:** 228 → 660 green over the sprint; hub-node coverage added for the 5 highest-degree functions.

## 3. Operational Debt & Realities

**Paperlog starvation — the expensive one.** `sentiment_entry_paperlog` went weeks without `source='daily'` rows. Root cause chain: EOD `full_pipeline` rarely *completed* — manual morning runs, Ctrl+C kills (motivated by the very Gemini cost bug we hadn't fixed), laptop-sleep job deaths, one 3-hour silent no-op from a PowerShell `$args` shadowing bug. Consequence: the item-1 sleeve analysis (prerequisite for the graduated-attack decision) slipped to ~late August — **we burned calendar time, the one resource A/B tests cannot refund.** The forward paper-log is our ONLY point-in-time sentiment store (sentiment is un-backtestable here); starving it starves the entire research program downstream.
Fixes now in place: scheduler repair (07-07), `--task inference_only` (≈$0.02 EOD path: inference + exits + paperlog row, no 25-min crawl — decouples research data from crawl fragility), chunked scoring persistence (14-07: interrupts keep paid rows), and cost fix removing the *reason* operators kill runs.

**Gemini surgery reflection.** Three stacked cost/reliability failures, all invisible in code review: (1) thinking tokens ON by default billing at output rate (~8× — $1.7 of every $2 run); (2) a floating `gemini-flash-latest` alias that silently re-resolved to `gemini-3.5-flash` and caused the 07-07 JSON-drift incident — while the `.env` pin only governed *one of four* call sites, and settings.json held the actual live knob; (3) burn-then-discard: one terminal append meant an interrupt paid for everything and persisted nothing. Institutional rules extracted: **pin GA models everywhere; one precedence chain (arg → env → config) for any paid-API knob; persist incrementally; a $0.0005 usage-metadata probe beats a week of billing forensics.**

**Other realities:** self-caught prod-data overwrite (synthetic fixture → live `ohlcv_AAA.parquet`; recovered via re-crawl; rule: fixtures live in scratchpad, no exceptions). Verify-before-trust for background launches (the `$args` incident). Plan-doc drift (intraday plan says PLANNED; context says live). Current worktree: 11+ uncommitted files across 3 logical groups — commit hygiene owed. ~~`macro_daily.parquet` stale since Jun 22~~ *[corrected 14-07 PM: probe shows fresh through 13-07 (3341 rows) — staleness was a symptom of never-completing pipelines, auto-heals with completed runs].*

## 4. Actionable Roadmap (strictly prioritized)

**P0 — Ops trust (this week). Everything else is gated on runs that complete.**
1. Confirm tonight's 15:30 ICT cron end-to-end: cost ≤ $0.40, `source='daily'` paperlog row lands, portfolio-guard sweep behaves (fires or stays correctly silent), tranche exits alert. This simultaneously closes the guard plan's gate and produces the first clean cost datapoint.
2. Commit split (3 logical groups) + reconcile intraday-scanner plan status; decide Gate 5: run the 09:15 smoke this week or consciously park the feature.

**P1 — Data accumulation (calendar-gated; zero code, maximum discipline).**
- Let the paperlog mature to **n ≥ 60 settled T+20 rows** (ETA ~late Aug at current fill rate). **The Arbitrator Gate is untouchable until then** — analysis pre-registered (threshold 0.7 fixed at log time), no peeking, no mid-flight parameter edits.
- Keep SSI foreign-flow daily accumulation running (source is today-only; history only exists if we collect it). Flow features stay dormant until ~40+ sessions exist.

**P2 — Statistical rigor upgrade.**
- Extend `PurgedKFold` (already train-side) to **CPCV** for backtest evaluation; report PBO from CPCV paths rather than seed resamples.
- PBO protocol fix: sweep `hold_days` {20,30,40}, never signal threshold (top-N book saturates below τ≈0.41 → configs become clones → PBO uninformative).
- DSR ≥ 0.95 remains the live-capital gate. Expect continued failure; that is information, not obstruction.

**P3 — Cross-sectional layer additions (each behind pre-registered CPCV + PBO ≤ 10% + GOLDEN A/B).**
- Foreign-flow divergence features (net-flow/ADV20, knife-catch divergence) once P1 data exists.
- Leadership features retry *only* under the P2 protocol — the idea isn't dead, the validation was.
- Choppy-regime penalty conditioning (the remaining lever from the June defend-bias post-mortem).

**Non-goals (explicit):** position-level stops in the tranche book (falsified), GBM macro features (killed), graduated-attack architecture before the item-1 verdict, any live capital before DSR ≥ 0.95.

---
*Filed 14-07-26. Companion artifacts: `portfolio-guard_PLAN_13-07-26.md` (ACTIVE), `macro-ab-result_23-06-26.md`, `doc/audit_paperlog_temporal_flaw.md`.*
