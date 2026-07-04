# Deep-Dive Project Feedback — Quant Engine V4.0

- **Date:** 2026-07-02
- **Scope:** Whole repo — architecture, code quality, quant methodology, tests, ops, repo hygiene, uncommitted work-in-progress
- **Evidence base:** code-review-graph (2,364 nodes / 22,943 edges), full pytest run (all green, exit 0, ~482 tests), direct source reads, git history, dependency audit

---

## 1. Executive Summary

This is a **genuinely well-engineered solo quant project** — top-decile for a one-person codebase. The methodology discipline (DSR gate before real money, purged k-fold, triple-barrier labels, forward paper-log instead of un-backtestable sentiment history, cost model with VN market microstructure) is what most retail quant projects fatally lack. The documentation-of-uncertainty culture ("SOURCE HONESTY", "be honest about the gap" docstrings) is rare and valuable.

The biggest risks right now are **not in the code — they are operational**:

1. A week of working-tree changes (including a production bug fix and a retrained model artifact) sits **uncommitted on `main`**, undeployable and one `git checkout` away from loss.
2. The serve path currently runs a **freshly retrained T+20 artifact next to a week-stale T+5 artifact** (vintage mismatch between the primary signal and the `/verify` confirmation signal).
3. The new foreign-flow data line depends on a **single unofficial API with no history endpoint** — every missed daily crawl is a permanent hole in the dataset.

Scorecard (1–10, calibrated against professional solo-quant systems):

| Dimension | Score | One-liner |
|---|---|---|
| Quant methodology | 9 | DSR gating, leak paranoia, forward paper-logs — institutional habits |
| Code quality | 8 | Typed, documented, pure-function style; a few god-modules remain |
| Test culture | 8 | ~482 green tests, leak-regression tests, parity tests; new flow modules at 0 coverage |
| Architecture | 7 | Clean src/ layering; main.py still a 2,109-line orchestrator-god |
| Repo hygiene | 5 | 110 MiB pack, binary artifacts in history, debris dirs, junk commit message |
| Ops robustness | 6 | Best-effort isolation is good; single-shot crawls + uncommitted deploys are the gaps |

---

## 2. What Is Genuinely Good (keep doing this)

- **Statistical discipline.** DSR < 0.95 → stays paper-only. Almost nobody enforces this on themselves. The regime-sizing A/B (MaxDD −23.3% → −16.9%) was shipped *and still correctly kept paper-only* because DSR said so.
- **Leak paranoia as a first-class feature.** `flow_features.py` documents its look-ahead contract in the header; `eda_flow_features.py::check_no_leakage` audits `fetched_at` vs ATC close; `mr_features` has pytest-lifted leak regression tests; the paperlog backfill was redesigned after the temporal-flaw audit. This is the single most important habit in the repo.
- **Source honesty docstrings.** `foreign_flow_crawler.py` lines 8–34 record *every dead-end tried live* (vnstock NotImplementedError, TCBS 404, Fireant 401, CafeF param rejection) with dates. Six months from now this saves you from re-testing the same dead APIs. Same for the ETF proxy's "WHAT THIS ACTUALLY MEASURES" section admitting it's activity, not flow decomposition.
- **Degrade-not-crash pipeline contract.** Every `full_pipeline` step (macro, foreign-flow, sentiment) is individually try/except-wrapped best-effort; one feed outage never kills the EOD run. `_fetch_one` per-symbol isolation in the ETF proxy matches.
- **Requirements file as documentation.** `requirements.txt` explains *why* each pin exists ("ALL THREE required — tabular_ensemble silently drops a missing base learner and trains a weaker stack"). That comment alone prevents a subtle production degradation.
- **Feature recipe hash gate.** `FEATURE_RECIPE_VERSION` computed from schema hash, checked at artifact load. Prevents the classic silent train/serve skew failure.
- **Bot access control.** Split-ID allowlist (`_role_for`, telegram_bot.py:519), mutation commands admin-only, ID2 activity shadow-copied to admin, per-user rate limiting, no secrets in repo (verified by scan), `.env` properly ignored.

---

## 3. Findings — Prioritized

### 🔴 HIGH

#### H1. Uncommitted production-relevant work on `main`
Working tree holds: the `/suggest_buy*` phantom-trade fix ([telegram_bot.py:657](src/utils/telegram_bot.py:657), `persist=False`), the `full_pipeline` step-1c wiring ([main.py:1893](main.py:1893)), a retrained `v3_ensemble_20d.joblib`, and 4 untracked new modules (foreign-flow crawler, ETF proxy, flow features, EDA script). Consequences:
- The phantom-trade fix is **live-affecting** (every manual `/suggest_buy` tap writes fake trades into the shared cron ledger until deployed) yet cannot be deployed because it isn't committed.
- One careless `git checkout -- .` or clean loses a full session of work including a model retrain.
- **Action:** commit now, in logical chunks (fix / pipeline wiring / new data line / artifacts). This is the single highest-value 15 minutes available.

#### H2. T+5 / T+20 artifact vintage mismatch on the serve path
`v3_ensemble_20d.joblib` retrained **Jul 1 14:58**; `v3_ensemble_5d.joblib` still dated **Jun 24** (T+5 retrain was interrupted). Both pass the recipe-hash gate (same schema), so nothing will *error* — but `/verify`'s short-horizon confirmation now runs on a model trained on ~1 week less data than the primary signal it's supposed to confirm. Silent, gate-invisible skew.
- **Action:** finish the resume path already documented in memory: `train_models.py --tb-horizon 5` → `run_backtest.py --mode tranche --hold-days 30`. Until then, treat `/verify` short-horizon output with reduced confidence.
- **Structural suggestion:** store the training-data end-date inside each artifact and have `_load_v3_bot` log a warning when sibling horizons diverge by more than N sessions. The recipe gate catches schema skew; nothing catches *vintage* skew.

#### H3. Foreign-flow data line: single fragile source, forward-only, permanent gaps
`foreign_flow_crawler.py` is honest that SSI iBoard is a live-snapshot-only source. Follow-through implications not yet mitigated:
- **A missed cron day is an unrecoverable hole forever** (no backfill path exists, by the source's nature). One VPS reboot at 15:30 costs you a data point permanently, and the dataset you're accumulating for the Phase-2 EDA becomes gap-riddled precisely when sample size is what you need.
- Unofficial endpoint = zero SLA; schema drift or a Cloudflare rule change kills the line silently (it degrades to no-op by design — good for the pipeline, bad for noticing).
- **Actions:** (a) add a second crawl attempt later in the evening (e.g. 20:00 ICT) — idempotent merge already makes this free; (b) alert (Telegram) when a trading day ends with zero new rows; (c) log a weekly row-count/expected-count ratio so drift is visible.

### 🟡 MEDIUM

#### M1. Retry contract bug: 5xx is *not* retried, contradicting the module docstring
[foreign_flow_crawler.py:36](src/data/foreign_flow_crawler.py:36) claims "`_fetch_with_retry` retries transient errors only (timeout, connection, **5xx**)". But `_TRANSIENT_EXC` (lines 95–100) contains only connection/timeout types, and `r.raise_for_status()` (line 121) raises `requests.exceptions.HTTPError` — not in the tuple, therefore **never retried**. A transient 502/503 from SSI fails the whole day's crawl on the first try (and per H3, that gap is permanent). Also: the function name in the docstring (`_fetch_with_retry`) doesn't exist — the decorated function is `_fetch_ssi_hose_snapshot`.
- **Fix:** either add an `HTTPError`-with-status≥500 predicate to the retry (tenacity `retry_if_exception` with a status check), or correct the docstring to match reality. Given H3, retrying 5xx is the right call, not the doc change.

#### M2. Naive wall-clock timestamps assume an ICT host
`crawl_today` gates on `datetime.now()` ([foreign_flow_crawler.py:147](src/data/foreign_flow_crawler.py:147)) and stores naive `fetched_at`. Deployment target is a VPS (systemd/cron). If that box runs UTC — the default for most VPS images — the 15:00 safe-crawl gate and the EDA leak audit (`check_no_leakage`, which compares naive `fetched_at` against naive `date+14:45`) are both silently wrong by 7 hours. The leak check would then *flag correct data as leaked* or vice versa.
- **Fix:** use `zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")` explicitly for the gate and store `fetched_at` as tz-aware (or store UTC + document the convention). This is cheap now, expensive after months of ambiguous rows accumulate.

#### M3. Safe-crawl gate warns but still writes partial-session rows
If the crawler runs at 14:30, it logs a warning **and then persists the partial snapshot anyway**. The merge policy ("fresh wins on same (date,ticker)") repairs it only if a second post-close run happens the same day; otherwise the partial row survives as permanent poison, detectable only if someone runs the EDA script. You already discovered this failure mode live (the 406/406 flagged-rows incident documented at lines 66–72).
- **Fix options:** refuse to write before the safe hour (return early), or write with `source='ssi_iboard_partial'` so downstream consumers can filter. Silent-with-warning is the worst of the three.

#### M4. Zero test coverage on the entire new flow line
Four new modules (`foreign_flow_crawler`, `etf_flow_proxy_crawler`, `flow_features`, `eda_flow_features`) — no test files. This is the same repo that keeps 482 tests elsewhere. `build_flow_features` in particular is pure, Polars-native, and trivially testable (contract errors, null-prop degradation, rolling-window correctness, cross-ticker isolation via `.over`) — exactly the kind of leak-sensitive feature code that got regression tests in `mr_features`. The idempotent-merge logic in `update_foreign_flow_daily` (fresh-wins dedup, corrupt-parquet rebuild) is also cheap to test with tmp-path parquets.
- **Suggested first tests:** (1) leak test mirroring `mr_features`' pattern — mutate future rows, assert features at t unchanged; (2) `.over("ticker")` isolation — two tickers, assert no bleed; (3) merge idempotency — same day crawled twice, later `fetched_at` wins.

#### M5. God-modules persist despite Phase-1 decomposition
- [main.py](main.py) — 2,109 lines, 28 top-level functions, spanning CLI parsing, crawling, model loading, inference, paperlog, dispatch, trade execution, crash alerting. `daily_inference` is back up to 174 lines (was decomposed to 169; it's regrowing).
- [telegram_bot.py](src/utils/telegram_bot.py) — 1,704 lines: auth, rate limiting, HTML rendering, command handlers, dispatch.
- [quant_agent_arbitrator.py](src/models/quant_agent_arbitrator.py) — 1,073 lines mixing prompt construction, API calls, veto logic.
The graph confirms the cost: `daily_inference` degree 84, `build_application` degree 78 (and **untested** per the graph's coverage analysis). This is documented debt ("V4.1 Structural Debt" marked complete, but main.py wasn't in its scope). Not urgent — but every new feature (paperlog, event overrides, regime sizing, step 1c) lands in main.py by default, and the file's growth rate is the real warning sign.
- **Suggestion:** adopt a rule — *new pipeline steps get their own module and main.py only gets the call site* (step 1c already followed this pattern; make it the norm). Consider `src/pipeline/` for the orchestration stages next time main.py needs surgery.

#### M6. Repo hygiene: 110 MiB pack and growing ~17 MB per retrain
`git count-objects`: **109.72 MiB**. Committed binaries: `v3_training_checkpoint.joblib` (13.4 MB), two ensembles (~3.4 MB each), plus `models/mr/backup_20260602/` (an entire backup dir tracked in git — that's what `backups/` and gitignore rules were supposed to prevent). Every retrain commit re-adds the full blobs; history never shrinks. Additional debris:
- `catboost_info/` (CatBoost training scratch) untracked and not gitignored — will get accidentally committed eventually.
- `real_backtest/result.json` (509 lines of run output) committed in `24be6d5` — run artifacts belong in `logs/` or reports, not source history.
- **Actions:** (a) add `catboost_info/` to .gitignore now; (b) decide the artifact policy: either git-lfs for `models/**/*.joblib`, or untrack them and keep a `models/MANIFEST.md` (hash + train date + recipe hash) in git with blobs stored outside; (c) untrack `models/mr/backup_20260602/`; (d) don't commit `result.json`-style run outputs.

#### M7. Dependency pinning gaps
- `statsmodels` used by `eda_flow_features.py` (ADF test) — unpinned, present only as a transitive dep. The script itself flags this (good) but the pin was never added. One `pip install -U scikit-learn` away from breaking.
- `requirements_dashboard.txt` uses `>=` ranges (`streamlit>=1.39`, `plotly>=5.18`) while the main file is strictly `==`-pinned with a documented regeneration procedure. Streamlit minor releases break APIs regularly; the dashboard is the most UI-fragile component and has the loosest pins.
- `pytest` (and any dev tooling) isn't captured anywhere — fine for a solo machine, but a `requirements-dev.txt` makes environment rebuilds deterministic.

#### M8. Commit message hygiene
`24be6d5` — *"idk nhma đại đại đi"* — actually contains a **user-facing fix** (three T+3→T+5 label corrections in bot help text) plus an unrelated 509-line data blob. The repo's own history shows the standard you normally hold (`fix: align event-override bear sentiment boundary...`). Junk messages on mixed-content commits are exactly what makes `git log` archaeology fail six months later, and this project demonstrably relies on archaeology (memory files, handoff docs, audit trails).

### 🟢 LOW

- **L1. Active-plan archive debt.** `sentiment-entry-paperlog_PLAN_16-06-26.md` and `telegram-per-ticker-dispatch_PLAN_18-06-26.md` sit in `general-plans/active/` though the work shipped weeks ago; dashboard P0–P2 plans likewise remain in `features/local-dashboard/active/` with P0–P2 done. Resume-flow tooling (and any agent following the routing rules) will keep rediscovering them as "in flight". Run the `vc-audit-plans` flow.
- **L2. `doc/SYSTEM_DESIGN.md`** is self-admittedly "partially stale" while `process/context/all-context.md` is excellent and current. Two sources of truth, one rotting. Either delete SYSTEM_DESIGN.md with a pointer, or demote it explicitly to historical.
- **L3. Graph pollution from harness duplication.** `.agents/skills/` and `.claude/skills/` are byte-duplicated (by design, for Codex parity), but they dominate the code graph's hub/large-function lists (4 of the top 15 hubs are skill scripts counted twice). If the graph tool supports exclusion globs, exclude both dirs — it will make hub/blast-radius queries about *your actual system* much sharper.
- **L4. Untested hotspots (graph-confirmed):** `run_backtest.main` (degree 97 — though `_build_wf_config`/`run_oos` are tested, the CLI arg-plumbing itself isn't), `build_application` (degree 78), `dashboard/tabs/giu.render` (degree 72). Streamlit render functions are painful to test — fine to skip — but a smoke test that `build_application` constructs with all handlers registered would catch handler-registration regressions cheaply.
- **L5. `update_foreign_flow_daily` line 234:** the empty-fetch branch does `pl.read_parquet(path).height` outside any try — a corrupt parquet raises here even though the merge branch handles the same corruption gracefully. Caught by `full_pipeline`'s wrapper, so cosmetic, but inconsistent with the module's own degrade philosophy.
- **L6. Asymmetric null handling in `crawl_today`:** `foreign_net_val` treats a missing buy (with present sell) as 0, while `foreign_buy_val` stays NULL. Defensible, but the net column silently embeds an assumption the raw columns don't. A one-line docstring note would do.
- **L7. Two ATC constants.** Crawler gates at 15:00 (`_SAFE_CRAWL_HOUR`), EDA flags before 14:45 (`_ATC_CLOSE_*`). Intentional (buffer > audit threshold) but defined independently in two files with no cross-reference — a future edit to one invites drift. Worth a shared constant or at least reciprocal comments.

---

## 4. Quant-Methodology Review

### Strengths already noted; the substantive concerns:

**Q1. Overlay stack is approaching attribution opacity.** The serve decision now passes through: tabular ensemble → statistical gates → regime-conditional sizing → macro-HMM overlay → GARCH-HMM exposure brake → sentiment arbitrator soft overlay → hard bear veto → event overrides. That's **seven conditioning layers** on one signal. Each was individually validated (good), but interactions weren't — e.g., regime sizing and the GARCH brake both respond to volatility regimes and can double-penalize the same state, and the bear veto + event overrides both intercept dispatch. When live performance eventually diverges from backtest, decomposing *which layer* caused it will be near-impossible without infrastructure.
- **Recommendation:** log a per-decision attribution record (base score, then the multiplier/veto applied by each layer) into DuckDB alongside the paperlog. Cheap to add at dispatch time, impossible to reconstruct later. This turns "the system did X" into "the ensemble said 0.74, regime penalized ×0.5, brake floored at 0.2, arbitrator vetoed" — auditable per trade.

**Q2. GARCH-HMM brake: "KEEP" on a loss-mitigation-only result deserves a sunset condition.** OOS verdict was keep-but-still-negative-Sharpe (loss mitigation). A component that costs complexity (arch + hmmlearn training, persistence caps, serialization, tests) and *reduces* risk without adding return is only worth keeping if the drawdown protection shows up when it matters. Define now, while you're objective, the condition under which it gets deleted (e.g., "if after N months of paper regimes it never fires during a real drawdown, or fires mostly in up-moves, remove it"). Components kept "because they might help in a crash" tend to become permanent unfalsifiable residents.

**Q3. Flow-feature thresholds are placeholders — the code says so; hold that line.** `_DIVERGENCE_PRICE_DROP_PCT = -1%` / `_FLOW_ZSCORE = 1.5` ([flow_features.py:56](src/features/flow_features.py:56)) are uncalibrated. The plan (EDA → lead-lag study → walk-forward → recipe bump) is correct. The one trap to pre-commit against: **do not tune these two thresholds on the same window you then walk-forward on.** With only forward-accumulated flow data (weeks, not years), the temptation to iterate thresholds against the full available history will be strong, and the sample is too small to survive it. Decide the maximum number of threshold configurations you'll test — DSR's trial-count input — *before* looking.

**Q4. ETF proxy: pre-register the kill criterion.** The module honestly labels itself a weak activity proxy (volume, not signed flow; market-level, not per-ticker). Weak features with a plausible story are the ones that sneak into recipes and add noise. Suggestion: write down now what lead-lag correlation magnitude/stability constitutes "in", and delete the module if it fails — the SOURCE HONESTY culture applied to features, not just APIs.

**Q5. Paperlog-derived precision/recall: mind the sample size.** `/audit_accuracy`'s BUY-precision / defensive-recall framing is sound, but with settled (T+20-matured) rows accumulating at a handful per day since mid-June, confusion-matrix cells are still single-to-low-double digits. A 🟢/🔴 display invites overreacting to what is statistically noise. Consider showing Wilson intervals (or at least n per cell) in the report so the display carries its own uncertainty.

**Q6. Vintage skew is a general blind spot (see H2).** The system has excellent *schema* skew protection and zero *data-freshness* skew protection across sibling artifacts. One structural fix covers all future horizons.

---

## 5. Ops & Deployment

- **Deploy gap:** the phantom-trade fix exists only in the working tree (H1). Whatever the VPS deploy mechanism is (git pull, presumably), it can't receive the fix until committed. Until then every manual `/suggest_buy5|20` invocation on production continues writing fake trades into the cron ledger — and those poison rows may already exist from before the fix. **Check `trade_history` for `user_id='cron'` rows timestamped at interactive hours and purge them**, or the audit analytics built on that ledger inherit the contamination.
- **Cron single-shot fragility:** both macro and foreign-flow refreshes get exactly one attempt per day at 15:30. For macro (yfinance, backfillable) that's fine; for foreign-flow (H3) it's not. Cheap fix: a second `update_foreign_flow_daily()` invocation from a 20:00 cron — the idempotent merge makes it a no-op when the first succeeded.
- **No heartbeat on data accumulation:** the flow parquet grows silently or fails silently. A weekly one-line Telegram digest ("foreign_flow: 5/5 sessions captured this week, 402 tickers/day avg") converts silent failure into a glanceable signal. Same pattern would have caught the still-unconfirmed `source='daily'` paperlog row noted in the context docs.
- **`models/saved/backups/` gitignore vs `models/mr/backup_20260602/` tracked** — the backup convention is inconsistently applied across model families (see M6).

---

## 6. Process / Harness Observations

- The RIPER-5 + context-router setup is elaborate and — unusually — actually maintained (`all-context.md` matches reality nearly everywhere I checked; the one stale line found in the 07-01 audit was already fixed). The context docs were more accurate than `doc/SYSTEM_DESIGN.md`; see L2.
- Plan archive discipline lags execution discipline (L1) — the one part of the process loop that's consistently skipped is UPDATE-PROCESS/archival. Given how much this repo leans on "resume from plan" flows, stale actives are the most likely future confusion source.
- Memory/handoff quality is exceptional (the self-caught prod-data incident, the interrupted-retrain resume command). This is why H1 is recoverable at all.

---

## 7. Recommended Action Order

| # | Action | Effort | Pays for |
|---|---|---|---|
| 1 | Commit the working tree in logical chunks (fix / wiring / new modules / artifacts) | 15 min | H1, M8 |
| 2 | Purge phantom `user_id='cron'` rows written by pre-fix `/suggest_buy*`; deploy the fix | 30 min | Ledger integrity |
| 3 | Finish T+5 retrain (`train_models.py --tb-horizon 5` → tranche backtest) | 1 run | H2 |
| 4 | Fix 5xx-retry gap + tz-aware timestamps + no-write-before-safe-hour in foreign_flow_crawler | 1 hr | M1, M2, M3 |
| 5 | `.gitignore` catboost_info/; untrack `models/mr/backup_20260602/`; decide joblib policy (LFS vs manifest) | 1 hr | M6 |
| 6 | Tests for `build_flow_features` (leak, ticker isolation) + merge idempotency | 2 hr | M4 |
| 7 | Pin `statsmodels`; `==`-pin dashboard deps | 10 min | M7 |
| 8 | Add per-decision overlay attribution logging at dispatch | 2–3 hr | Q1 — biggest long-term analytical payoff |
| 9 | Second daily flow-crawl attempt + weekly capture digest | 1 hr | H3 |
| 10 | Archive completed plans (`vc-audit-plans`) | 20 min | L1 |

Items 1–3 before the next trading session if possible; the rest within the week.

---

## 8. Closing Assessment

The project's core loop — hypothesize → validate with proper statistics → ship paper-only until DSR clears → keep honest records of what failed — is the right loop, executed with unusual rigor. The failure modes that remain are the ones rigor-at-the-component-level doesn't catch: uncommitted state, cross-artifact freshness, stacked-overlay interactions, and data lines that can't be backfilled. All four are addressable with process, not cleverness, and none require touching the model math.

The single thought to carry forward: **the system's edge now depends as much on its data-collection reliability (forward-only paperlog, forward-only flow line) as on its models.** Those forward-accumulating datasets are the only truly irreplaceable assets in this repo — the models can be retrained, but a missed day of flow data or a corrupted paperlog cannot be re-observed. Protect them with the same paranoia currently reserved for feature leakage.
