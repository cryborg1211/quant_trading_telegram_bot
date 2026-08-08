# Future Work — System State Snapshot & Next Steps

**Date**: 08-08-26
**Trigger**: full system re-pass + doc deep-dive, requested once real foreign-flow data landed (see `process/general-plans/backlog/foreign-flow-fastconnect-integration_PLAN_24-07-26.md`)
**Scope**: this is a synthesis/report, not a plan — it points at what's next, doesn't execute it.

---

## 1. Where the system actually stands

- **Paper-only, still.** Every backtest configuration tested to date — across dozens of admission/sizing/feature experiments spanning weeks — fails the Deflated Sharpe Ratio gate (`p_dsr < 0.95`). The 22-07-26 DSR calibration diagnostic (`scripts/analyze_dsr_calibration.py`) confirmed this is a **robust** failure, not a small-sample or inflated-trial-count artifact: only 0.4% of 2000 block-bootstrap resamples pass, even though the sweep grid is genuinely diverse (not near-clone configs). **This is the single most important fact for prioritizing future work**: sizing, admission, and portfolio-construction knobs have been thoroughly explored and are not going to close this gap. The edge itself needs to be structurally bigger — via a real new signal, not portfolio-level engineering.
- **Serve path is stable and well-guarded.** T+20 primary / T+5 verify, GARCH-HMM+drift+breadth exposure brakes, portfolio guard, admission hysteresis, sector caps, open-cohort dedup, promote-gate on the weekly auto-retrain. These guards were all built in direct response to the July-2026 loss post-mortem and are load-bearing — do not weaken them without re-running the A/Bs that justified them.
- **894 tests, full suite green** as of this session's close.

## 2. Signal-search track record (what's been tried)

Rejected (do not re-attempt without a materially different angle):
- Regime-gating MR-LGBM, RSI-direction slicing MR-LGBM (redundant with existing oversold features)
- LGBMRanker/lambdarank admission objective (2x drawdown, PBO 66.7%)
- Confluence (T+5 ∧ T+20 agreement) — but this run **did** surface a real, actionable finding: T+20's gate does essentially all the quality-filtering work; T+5-only conviction is statistical noise. If T+5 side-door dispatches are ever reconsidered, require the T+20 gate too.
- Prob-scaled cohort weights, rank_breadth admission, vol-scaled burst divisor (all A/B'd, all lost to their simpler baselines)
- Fast-attack/impulse momentum-ignition sub-model (OOF precision 0.520, misses both the 0.60 target and MR-LGBM's own bar)
- **Foreign flow, as a linear predictor** (this session) — real EDA, real statistical power (n>500K), correlation ~0.001-0.002 everywhere tested. Closed.

Shipped and working:
- Burst-budget divisor (`tranche_budget_days=10`, opt-in) — ~3x PnL over the calendar-budget baseline at DD still inside the production comfort band.
- Regime-conditional sizing, GARCH-HMM/drift/breadth exposure brakes, admission hysteresis, sector caps, promote-gate.

**Promising but statistically unconfirmed** (same "thin holdout" problem in both cases — worth revisiting if/when there's more data, not worth re-testing on the same static window):
- Breadth inflection (low breadth + rising delta) on MR-LGBM knife-catch timing — wired as a **display-only annotation**, never gates the trigger.
- MR context features (volume-exhaustion + sector-relative oversold) — clears the 0.60 OOF bar for the first time (0.642) but holdout collapses to n=5 fires. **Not wired into serve.**

## 3. Concrete next steps, ranked

1. **`flow_knife_catch_divergence` conditional test** (cheap, narrow, uses data already in hand). The just-closed foreign-flow EDA tested the blunt/unconditional hypothesis only. `flow_features.py` already has a z-scored, price-drop-conditioned divergence feature built but never separately tested. If foreign flow has any signal at all, this conditional framing is a more plausible place to find it than the raw correlation that was just rejected.
2. **SOURCE 1 ticker-coverage bug** (`src/data/foreign_flow_crawler.py::crawl_today`, called daily from `main.py`'s `full_pipeline`). Found this session: the live snapshot has been writing only ~51 tickers/day, zero overlap with the 359-ticker research universe, since ~07-01. Root cause unknown — worth a focused debug session (check whether `_fetch_ssi_hose_snapshot`'s raw response shape silently drifted, or a parsing/filter bug in `crawl_today`). Low urgency (doesn't block anything, SOURCE 2/FastConnect now has full coverage) but a real, silent production defect.
3. **Block-deal + aggressor-trade fields** (`block_deal_val/vol`, `buy_trade_count/vol`, `sell_trade_count/sell_trade_vol` — captured free in the FastConnect backfill, 08-08-26, never validated as features). Two genuinely new signal families not yet tested: negotiated/off-book block trades (institutional accumulation proxy) and aggressor-side order-flow imbalance (a different flavor of momentum than anything currently in the feature set). Worth an EDA pass using the exact same `eda_flow_features.py` scaffold, now that it's fixed and proven against real data.
4. **`audit-engine-picks` dashboard plan** (`process/features/local-dashboard/active/audit-engine-picks_PLAN_26-06-26.md`) — genuinely still pending, never executed, sitting since 26-06-26 awaiting an EXECUTE decision. May be partially superseded by the later `/audit_accuracy` bot command (`src/utils/accuracy_audit.py`, 29-06-26) which covers similar ground via a different surface (Telegram command vs. dashboard tab) — worth a quick scope check before deciding to execute, archive-as-superseded, or drop.
5. **Dashboard P3 (launcher)** — the only unstarted phase of the local-dashboard umbrella; P0-P2 are done and shipped (their plan files were stale — corrected this session).
6. **Structural**: the DSR gate itself (§1) is the real constraint. Nothing above is likely to single-handedly clear it — any of these, if they pan out, should be evaluated primarily on whether they move the DSR/PBO numbers, not just Sharpe/PnL in isolation, given how many Sharpe-positive-but-DSR-failing configs have already been found and set aside.

## 4. Documentation hygiene fixed this session

- `doc/SYSTEM_DESIGN.md` — was describing a fully superseded pre-V4.0 architecture (Alpha360 stacking) with no warning; a future session could have read it as current-state truth. Added a SUPERSEDED banner pointing to `all-context.md`.
- `process/general-plans/active/intraday-attack-scanner_PLAN_06-07-26.md` — header said "PLANNED"; code has been shipped and tested (542 lines, 40/40 tests) since 06-07-26. Corrected.
- `process/general-plans/active/auto-ca-price-adjustment_PLAN_17-07-26.md` — self-declared "ready for UPDATE PROCESS" but was still sitting in `active/`. Moved to `completed/`.
- `process/features/local-dashboard/active/{p0,p1,p2}_PLAN_19-06-26.md` — all three said "PLANNED"/"ready to execute" despite being shipped since 21-06-26. Corrected and moved to a newly-created `process/features/local-dashboard/completed/`.
- `process/context/all-context.md` / `process/context/tests/all-tests.md` — synced with the full 4-idea batch + SSI FastConnect work (repo tree, test counts, research narrative, Scan Metadata) earlier this session; re-verified accurate as of this pass.
- `process/context/planning/all-planning.md` — checked, accurate, no changes needed (it's a stable process-convention doc, not project-state).

## 5. What this pass deliberately did NOT do

- Did not re-run or re-validate the weekly auto-retrain (`quant_weekly_retrain`, scheduled Saturdays 09:00) — that's on its own schedule and gated by the promote-gate; nothing here should interact with it.
- Did not touch the pre-existing dirty model artifacts (`models/mr/*`, `models/saved/*`) sitting uncommitted since before this session started — still not this session's to resolve, flagged to the user twice already.
- Did not execute `audit-engine-picks_PLAN` or start dashboard P3 — both are real pending decisions, not silently resolved either way.
- Did not attempt a full line-by-line audit of every plan file's internal checklist (many have 20-40 line items) — fixed the header/status line, which is what routing and discovery actually read, rather than hand-checking every box.
