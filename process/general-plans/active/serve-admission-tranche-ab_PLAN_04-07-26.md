# Serve-Mirror Admission A/B (backtest evidence for the June "all-defend" gate)

- **Date:** 04-07-26
- **Type:** SIMPLE
- **Status:** NOT STARTED
- **Scope of this plan:** Task A only (backtest engine change + A/B run + evidence
  writeup). Task B (serve-side promotion of a floor/top-N admission rule) is
  DOCUMENTED here as a decision rule but explicitly OUT of this plan's
  implementation scope — it requires a NEW plan after this A/B's verdict.

## Background (research already complete — do not redo)

Problem reported by the user: every Telegram alert in June-2026 said "MONITORING-ONLY /
THỊ TRƯỜNG YẾU" (defend) while VNINDEX rallied on narrow bank leadership.

Root-cause chain, confirmed from `logs/quant_v6.log` + OHLCV parquets + direct
source read of `main.py` during this planning session (corrects one detail in
the original task brief — see **Correction to task brief** below):

1. **Serve admission is an absolute per-name gate.** `predict_v3_horizon`
   (`main.py:497`) builds `meta_gate = {ticker: P(UP) >= bot.up_threshold}`
   where `bot.up_threshold` is the artifact-frozen GOLDEN threshold (currently
   T+20 GOLDEN = 0.45, per the 02-07-26 retrain). `_select_candidates`
   (`main.py:864`) then filters `predictions` by VN30 membership AND
   `meta_gate.get(ticker, True)`, keeps the top `min(max_candidates, 6)` by
   P(UP), and — **when that filtered pool is empty** — falls into its OWN
   fallback branch (`main.py:907-943`): ranks the top-3 VN30 tickers by raw
   P(UP) regardless of gate, tags each with a Vietnamese "not worth the
   trade-off" reason string, sets `fallback_mode=True`, and returns them for a
   **MONITORING-ONLY observability card** (`daily_inference` `main.py:1075-1102`
   short-circuits and never calls `run_trade_execution`). This is the June
   "defend" card the user saw — every logged June run had zero names pass the
   absolute meta-gate.
2. **The model's cross-sectional ranking was directionally correct even while
   the absolute gate blocked everyone.** Fallback Top-3 by raw P(UP): Jun 16 =
   [SSB, VIB, MBB], Jun 18 = [SSB, VIB, BID]. SSB finished June +16.2% — the
   best VN30 name for the month. The admission gate, not the ranking, is what
   suppressed the trade.
3. **Market context was genuinely mixed, not purely bullish** — 331-ticker
   equal-weight median return −2.5%, VN30-ish mean −0.78% in June; the rally
   was narrow bank leadership. Some defensiveness during a negative-breadth
   month may be correct behavior. The A/B in this plan must MEASURE that
   trade-off, not assume the gate is pure lost alpha.
4. **The validated tranche backtest already runs cross-sectional (non-absolute)
   admission and does not exhibit serve's zero-candidate-day dynamic.**
   `_tranche_day` (`src/backtest/walk_forward.py:732`) ranks by P(UP), takes
   top `cfg.max_positions` above `cfg.signal_threshold` (a much lower floor —
   GOLDEN T+20 `signal_threshold=0.40`, 5pp below `up_threshold=0.45`), and
   deploys `NAV/hold_days` into whoever clears that floor every trading day.
   GOLDEN T+20 (retrained 02-07-26): Net +32.55%, Sharpe +0.689, MaxDD
   −13.16%, DSR p=0.3146 (FAIL), PBO=3.0% (PASS) — see
   `process/general-plans/reports/retrain_t5_t20_result_02-07-26.md`. The
   full threshold sweep shows the 0.40/0.35 row is near-identical to GOLDEN
   in Net/Sharpe/DD — the top-5/day book saturates once the floor is low
   enough, so absolute thresholds above ~0.40 add little measurable value IN
   THAT BACKTEST'S EXISTING DESIGN (which never runs the higher, serve-style
   absolute floor as a standalone admission rule with the top-N slot count
   held fixed — that comparison is this plan's whole point).
5. **NOT culprits — do not touch:** GARCH-HMM exposure brake (benign this
   period, P(Bull)≈0.975 → no material scale-down); sentiment arbitrator
   (only softens SELL→HOLD, never blocks a BUY that already passed the
   meta-gate); regime-conditional sizing / discrete NAV-tier cap (separate
   overlays — see **Deliberately out of scope** below).

### Correction to task brief

The task brief describes serve's behavior as "meta_gate = {ticker: P(UP) >=
up_threshold}... EVERY logged June daily_inference run had zero VN30 names
pass -> fallback_mode -> MONITORING-ONLY defensive card. Zero tradable BUYs
the whole month." That is accurate for the ADMISSION step, but the brief's
phrasing implies serve returns nothing at all. In fact `_select_candidates`
already has its own top-3 "MONITORING-ONLY" fallback with per-ticker Vietnamese
reason strings, and `daily_inference` has a SECOND fallback layer
(`main.py:1161-1192`, "post-arbitration empty book") that also surfaces a
top-3 monitoring card if the arbitrator itself empties an otherwise-non-empty
candidate pool. Both fallback paths already exist and already work as
designed — this plan does not touch them. What this plan quantifies is
whether REPLACING the admission step's absolute meta-gate with a floor +
top-N cross-sectional rule (so names like June's SSB get admitted to
`run_trade_execution` instead of only into a monitoring card) would have been
net-better, using the tranche backtest as the measurement instrument.

## Goal

**Task A (this plan's execute scope):** Add an opt-in "serve-mirror admission"
mode to the tranche backtest engine so the tranche book can run EITHER its
existing cross-sectional top-N-above-floor admission (unchanged default) OR an
absolute-gate admission that mirrors serve's `meta_gate` mechanics (deploy
nothing on a day where zero names clear an absolute floor, cap the candidate
pool at 6 to mirror `_ARBITRATOR_POOL`), on identical data/models/costs. Run a
pre-committed grid of configs and report the comparison, including new
diagnostics (zero-candidate-day count, average deployed exposure, monthly
return series) that isolate narrow-leadership months structurally similar to
June-2026.

**Task B (documented, NOT implemented in this plan):** A decision rule for
whether/how to change `main._select_candidates`'s admission behavior, gated on
Task A's evidence. See **Task B — Decision Rule (out of scope for EXECUTE)**
below.

## Touchpoints

| File | Change |
|---|---|
| `src/backtest/walk_forward.py` | NEW `WalkForwardConfig` fields: `admission_mode: str = "cross_sectional"` (values: `"cross_sectional"` \| `"absolute_gate"`), `admission_floor: float = 0.45`, `admission_pool_cap: int = 6`. `_tranche_day` step-3 admission logic branches on `admission_mode`: unchanged top-N-above-`signal_threshold` when `"cross_sectional"`; when `"absolute_gate"`, filter today's ranked candidates to `p_up >= admission_floor` FIRST, cap the survivor list at `admission_pool_cap`, THEN take the existing top `max_positions` of THAT capped/filtered list (empty survivor list -> zero orders that day, budget stays cash — mirrors serve's zero-candidate-day fallback-to-cash economics, not its Vietnamese-text monitoring card, which is a display-layer concern out of scope for a backtest). New per-day diagnostic counters (`zero_candidate_days`, `daily_deployed_exposure` list) collected on `WalkForwardResult`/`DailyRecord` or returned as an auxiliary dict — see **Diagnostics contract** below for the exact shape decision EXECUTE must follow. |
| `run_backtest.py` | New CLI flags `--admission-mode {cross_sectional,absolute_gate}` (default `cross_sectional`), `--admission-floor FLOAT` (default 0.45), `--admission-pool-cap INT` (default 6). Threaded `_cli -> main -> run_oos -> _build_wf_config` mirroring the existing `--regime-sizing` / `--nav-tier-cap` pattern (see `run_backtest.py:764-822` for the exact plumbing shape to copy: CLI arg -> `overrides`/explicit kwarg -> `main(...)` param -> `run_oos(...)` param -> `_build_wf_config(...)` param -> `WalkForwardConfig(...)` field). |
| `tests/test_admission_ab.py` | NEW test file. Default-off byte-identical test (`admission_mode="cross_sectional"` unchanged vs. today's tranche behavior on a small synthetic panel), absolute-gate zero-candidate-day behavior (a day where all P(UP) fall below `admission_floor` deploys nothing, cash carries), pool-cap enforcement (more than `admission_pool_cap` survivors still caps at `admission_pool_cap` before the `max_positions` slice), and a boundary test at exactly `admission_floor` (inclusive, `>=`, matching serve's `meta_gate` `>=` semantics). |
| `scripts/ab_serve_admission.py` | NEW — thin script that runs the pre-committed 7-config grid (see **Experimental design**) against the SAME frozen T+20 checkpoint (`models/saved/v3_training_checkpoint.joblib`), reusing `run_oos`/`_build_wf_config` from `run_backtest.py` directly (no CLI re-invocation per config — import and call in-process so the expensive dataset re-materialization step happens exactly once, matching the existing `seed_inference_caches` per-seed-not-per-config reuse pattern already in `run_backtest.py:382-437`). Emits a single markdown-formatted comparison table to stdout AND writes it to `process/general-plans/reports/serve-admission-ab-result_[run-date].md`. |

## Deliberately out of scope (do not touch)

- `src/trading/risk_tier.py`, `WalkForwardConfig.use_nav_tier_cap`,
  `--nav-tier-cap` — belongs to the sibling
  `notification-attribution-risk-tier_PLAN_02-07-26.md` (CODE-COMPLETE,
  awaiting its own A/B run). This plan's new fields are independent and must
  not interact: `admission_mode`/`admission_floor`/`admission_pool_cap` gate
  WHICH NAMES enter a tranche; `use_nav_tier_cap` gates HOW MUCH TOTAL NAV a
  tranche may deploy. Both default off/unchanged; EXECUTE must confirm they
  compose cleanly (an `absolute_gate` day with zero candidates has
  `budget` never computed against the tier cap — no interaction to test
  beyond "does the code path still run without crashing when both flags are
  True", which is a nice-to-have combinatorial test, not a required one for
  this plan's verdict).
- `src/trading/regime_policy.py`, `WalkForwardConfig.use_regime_sizing`,
  `--regime-sizing` — validated, shipped, unrelated (per-name sizing
  modulation, not admission).
- `pipeline.py`, `FEATURE_RECIPE_VERSION`, any retrain — recipe
  `v2-sha8:53b5bd85` and the freshly-retrained artifacts
  (`models/saved/v3_ensemble_20d.joblib`, `v3_ensemble_5d.joblib`,
  `v3_training_checkpoint.joblib`, all uncommitted from the 02-07-26 retrain)
  must not be touched or re-written by this plan's work. `scripts/
  ab_serve_admission.py` reads the checkpoint read-only via `run_oos`'s normal
  path; it must call with `save_bot_payload=False` equivalent (i.e. do not
  call `_persist_bot_payload` at all — do not go through `run_backtest.main`,
  call `run_oos` directly).
- `main.py::_select_candidates` / `daily_inference` — Task B only, not this
  plan. Zero production code changes to the serve path in this plan.
- Sentiment arbitrator, GARCH-HMM brake — confirmed not culprits (see
  Background item 5); no changes.

## Guardrails

1. **NO feature-recipe change, NO retrain.** `pipeline.py` untouched. No
   writes to `models/saved/` (script must not call `_persist_bot_payload` or
   any path that writes a `.joblib`).
2. **Engine change is OFF-by-default / behavior-preserving at the default.**
   `admission_mode: str = "cross_sectional"` must reproduce the CURRENT
   `_tranche_day` admission logic byte-for-byte (verified by a default-off
   regression test comparing engine output before/after this change on a
   fixed synthetic panel + fixed seed).
3. **CLI flag threading mirrors the existing `--regime-sizing` /
   `--nav-tier-cap` pattern exactly** — same six-hop plumbing
   (arg -> variable -> `main()` kwarg -> `run_oos()` kwarg ->
   `_build_wf_config()` kwarg -> `WalkForwardConfig` field), same default
   values baked in at every hop (not just the CLI parser), same docstring
   style noting "A/B experiment — default off" where applicable.
4. **Tests run via the bare pytest runner** (`process/context/tests/
   all-tests.md`) — `pytest -q`, no conda-stock environment, no need to stop
   `run_bot.py` (pytest does not touch the live DuckDB lock).
5. **Precondition — commit the current dirty tree in logical chunks BEFORE
   this work starts.** Confirmed via `git status --porcelain`: `main.py`,
   `run_backtest.py`, and `src/backtest/walk_forward.py` (this plan's exact
   touchpoints) already carry uncommitted deltas from the in-flight
   `notification-attribution-risk-tier` work (`use_nav_tier_cap` +
   `--nav-tier-cap`), plus unrelated dashboard/flow-crawler/paperlog work is
   also dirty. Layering a second uncommitted feature onto files that already
   have one uncommitted feature compounds diff size and makes rollback of
   EITHER feature independently much harder. EXECUTE must not start editing
   `walk_forward.py`/`run_backtest.py`/`main.py` until the existing dirty
   state is committed in coherent chunks (suggested split: risk-tier work as
   one commit per its own plan's touchpoint list; dashboard/flow-crawler/
   paperlog work as separate commit(s) by topic — recommend routing this
   specific step through `vc-git-manager` rather than a manual single
   `git add -A`).
6. **Resource contention is a scheduling note, not a code concern.** If GPU/CPU
   is busy with training when the A/B grid runs, the grid will simply run
   slower (tranche mode scores daily — first threshold/config pays the full
   oracle-inference cost; reuse the per-seed inference-cache pattern within
   the script to avoid re-scoring the SAME day across configs that share a
   seed and only differ in `admission_*` fields — the oracle call itself is
   admission-mode-independent, matching the existing `inference_cache`
   contract in `walk_forward.py:299-321`).
7. **DSR trial-count discipline.** Per the deep-dive feedback report's Q3 (
   `process/general-plans/reports/deep-dive-feedback_02-07-26.md`), the total
   config count run in an A/B is itself a DSR input (more configs tried =
   higher multiplicity penalty on the deflated Sharpe of whichever config
   looks best). This plan PRE-COMMITS the exact grid (below) before any run —
   no ad-hoc "let's also try 0.42" additions once results start coming in.

## Experimental design — pre-committed grid (7 configs, do not expand)

All configs run on the SAME frozen T+20 checkpoint
(`models/saved/v3_training_checkpoint.joblib`, GOLDEN base params:
`up_threshold`/`admission_floor` reference = 0.45, `hold_days=30`,
`max_positions` from `RunConfig` current default), same 4 seeds already in the
checkpoint (matching the retrain report's methodology), same corporate
actions, same P(Bull) series, same cost model.

| # | Label | `admission_mode` | `admission_floor` | top-N (`max_positions`) | `admission_pool_cap` | Purpose |
|---|---|---|---|---|---|---|
| 1 | Baseline (incumbent) | `cross_sectional` | n/a (`signal_threshold=0.40` unchanged) | 5 (GOLDEN default) | n/a | Control — unmodified GOLDEN tranche path |
| 2 | Serve-mirror @0.45/N5 | `absolute_gate` | 0.45 | 5 | 6 | Direct mirror of serve's current `up_threshold=0.45` gate at the GOLDEN top-N |
| 3 | Serve-mirror @0.45/N3 | `absolute_gate` | 0.45 | 3 | 6 | Mirrors serve's actual dispatched Top-3 slot count (`top_buy_signals[:3]`, `main.py:1120`) |
| 4 | Floor sweep @0.40/N3 | `absolute_gate` | 0.40 | 3 | 6 | Lower floor at serve's Top-3 slot count |
| 5 | Floor sweep @0.40/N5 | `absolute_gate` | 0.40 | 5 | 6 | Lower floor at GOLDEN's top-5 slot count |
| 6 | Floor sweep @0.35/N3 | `absolute_gate` | 0.35 | 3 | 6 | Floor near the sweep's low end, Top-3 slots |
| 7 | Floor sweep @0.35/N5 | `absolute_gate` | 0.35 | 5 | 6 | Floor near the sweep's low end, top-5 slots |

7 configs x 4 seeds = 28 equity curves total. This IS the multiplicity fed
into any DSR computed on the "best" config in the writeup (n_trials=28, or
n_trials=7 if reporting per-config-not-per-seed DSR — EXECUTE must state which
convention it used and be consistent with `run_backtest.py`'s existing
`n_trials_total = len(sweep_thresholds) * max(1, len(seeds))` convention,
i.e. use 28).

**Rejected from the grid (explicitly, so no ambiguity for EXECUTE):** N=10,
floor=0.50, floor=0.42 (rescue-band-only), any `admission_pool_cap` other
than 6. If Task A's evidence is inconclusive on this grid, expanding the grid
requires a NEW plan revision with the same pre-commit discipline, not an
ad-hoc addition mid-run.

## Metrics per config

Per the 4-seed mean, matching `run_backtest.py`'s existing `sweep_results`
aggregation style.

Existing (reuse `equity_metrics` / `_teardown_report` math, do not reinvent):
- Mean Net PnL (VND), mean annualized Sharpe, mean Max Drawdown
- DSR p-value (n_trials=28) and PBO (CSCV) computed on the winning config's
  per-seed equity curves, same as the existing teardown

NEW diagnostics (this plan's actual contribution beyond the existing sweep
machinery):
- **Zero-candidate-day count** — number of trading days (post-`start_trading_date`)
  where the admission filter produced an empty survivor list (only meaningful
  for `absolute_gate` configs; always 0 for `cross_sectional` baseline since
  it has no absolute floor — confirm this is actually true given
  `signal_threshold=0.40` could ALSO empty on a very weak day; report the
  baseline's count too rather than assuming zero).
- **Average deployed exposure** — mean of `gross_exposure` (already a
  `DailyRecord` field) across the OOS window, so a lower number for
  `absolute_gate` configs is visible as "more cash on the sideline" not just
  inferred from PnL.
- **Monthly return series** — reuse `monthly_net_sharpe`-style monthly
  grouping (already exists in `run_backtest.py:158-167`, generalize to
  monthly NET RETURN not just monthly Sharpe) so a June-2026-pattern month
  (negative breadth, narrow leadership) can be isolated in the OOS window and
  each config's behavior in that specific month-type compared directly.
  Note: the OOS window is historical backtest data, not literally
  June-2026 (which is live/current) — "June-2026-pattern month" means
  identifying analogous months in the OOS sample by the same signature
  (negative median breadth + narrow-leadership concentration), not
  literally June-2026 itself.

## Diagnostics contract (EXECUTE must follow exactly — avoids ambiguity)

Extend `WalkForwardResult` with one new field (not a separate return value —
keep the existing single-result-object contract every other diagnostic
already uses):
```
zero_candidate_days: int
```
and extend the existing per-day loop in `_tranche_day` (or a thin wrapper
around it) to increment a counter whenever the admission-filtered survivor
list used for that day's buy decision is empty AND `admission_mode ==
"absolute_gate"` (cross_sectional's existing "no picks" path already returns
early with `n_orders=n_fills=n_rej=0`; reuse that same early-return branch,
just add the counter increment before the `return` when the mode is
`absolute_gate`). `gross_exposure` per day is ALREADY on `DailyRecord` — no
new field needed there; the "average deployed exposure" metric is a
post-hoc `eq["gross_exposure"].mean()` computed in the A/B script, not a new
engine field. Monthly return series is likewise post-hoc in the A/B script
using the existing `equity_curve` DataFrame — do not add engine-side monthly
aggregation.

## Exact run commands (for EXECUTE and for the eventual report to cite)

```
python scripts/ab_serve_admission.py
```
(no CLI args — the 7-config grid is pre-committed IN the script per the table
above, not passed at the command line, so there is no way to accidentally run
an unplanned 8th config). The script must log, for each config, the same
per-seed line format `run_backtest.py` already emits
(`thr=... seed=... NetPnL=... Sharpe=... DD=... predUP=... prec=...`
equivalent, substituting the config label for `thr=`), then print the
comparison table, then write the markdown report.

Manual single-config smoke test during development (not the full A/B):
```
python run_backtest.py --mode tranche --hold-days 30 --admission-mode absolute_gate --admission-floor 0.45 --no-save
```

## Verification evidence checklist

- [ ] `pytest -q tests/test_admission_ab.py` — new tests green (default-off
      byte-identical, zero-candidate-day cash carry, pool-cap enforcement,
      floor boundary inclusive)
- [ ] `pytest -q` — full suite still green (baseline ~505 per the sibling
      risk-tier plan's last known count; confirm the exact count at EXECUTE
      time since more uncommitted work may have landed tests since)
- [ ] Default-off regression: `admission_mode="cross_sectional"` run vs. a
      pre-change engine run on the same fixed synthetic panel/seed produces
      identical equity curve (not just similar — exact match on `nav` per
      day) — this is the byte-identical guardrail, prove it explicitly, not
      just by code inspection
- [ ] `scripts/ab_serve_admission.py` completes all 7 configs x 4 seeds
      without any config raising an exception (a failed config in the sweep
      loop already logs a warning and continues per the existing
      `run_backtest.py` pattern — carry that resilience into the new script)
- [ ] Comparison table + markdown report written to
      `process/general-plans/reports/serve-admission-ab-result_[run-date].md`
      with all 7 rows, both existing and new-diagnostic columns populated
- [ ] Zero-candidate-day count and average-deployed-exposure numbers are
      visibly different between `cross_sectional` and `absolute_gate` configs
      (if they are NOT visibly different, that itself is a finding to report,
      not a bug to chase)

## Task B — Decision Rule (out of scope for EXECUTE)

Documented for the follow-up plan, not implemented here.

If Task A's evidence shows the `absolute_gate` configs (rows 2-7) underperform
the `cross_sectional` baseline (row 1) on BOTH of:
- mean Sharpe delta worse by more than 0.05, AND
- mean Max Drawdown NOT meaningfully better (i.e. the absolute gate is not
  earning its lost-return back via lower risk)

...while ALSO showing a materially nonzero zero-candidate-day count (e.g. more
than ~5% of OOS trading days), then the evidence supports a follow-up plan
that:
1. Replaces `_select_candidates`'s hard `meta_gate.get(ticker, True)` filter
   with a floor + top-N admission rule (parameters to be chosen from whichever
   grid row performed best, or a fresh sweep if none of the 7 pre-committed
   rows is a clear winner) — implemented behind a NEW
   `CONFIG.trading` kill-switch, default OFF until independently reviewed.
2. Keeps the EXISTING fallback observability card behavior (top-3 monitoring
   card with Vietnamese reason strings) for days that are genuinely weak
   below the new (lower, cross-sectional) floor — i.e. Task B is about
   moving WHERE the line is drawn and how many names can pass it, not about
   removing the "market is weak, here's what we're watching" UX.
3. Requires its OWN plan (new RESEARCH pass confirming the serve-path
   `_select_candidates` call sites and downstream `run_trade_execution`
   wiring haven't drifted since this plan's research, since Task B is a
   PRODUCTION serve-path change with real capital-allocation consequences,
   unlike this plan's backtest-only Task A).

If the evidence instead shows the `absolute_gate` configs are close to or
better than baseline on risk-adjusted terms, the conclusion is that serve's
current defensiveness, while it happened to block June's SSB winner, is not
measurably costly on average across the OOS sample — and Task B should NOT
proceed; the June episode gets filed as an acceptable false-defensive instance
within an overall-sound gate, and no serve-path change is warranted from this
evidence alone.

## Deferred / follow-ups

- Task B implementation (see decision rule above) — separate plan, gated on
  this plan's A/B verdict.
- `signal_ledger` attribution of WHICH admission mode produced a given trade
  — not needed for this plan's backtest-only scope; would matter if/when
  Task B ships to serve.
- Combinatorial test of `admission_mode="absolute_gate"` +
  `use_nav_tier_cap=True` together — nice-to-have, not required for this
  plan's verdict (see Deliberately out of scope).
- Analogous absolute-gate A/B for the T+5 horizon — T+5's retrain already
  fails PBO (43.1%) independent of admission mode (see
  `retrain_t5_t20_result_02-07-26.md`), so T+5 evidence would be read with
  extra skepticism regardless; T+20 is the more informative horizon to spend
  this A/B's compute on first.

## Resume and execution handoff

- **Primary plan file to execute:** THIS file
  (`process/general-plans/active/serve-admission-tranche-ab_PLAN_04-07-26.md`).
- **Precondition before EXECUTE touches any file:** commit the current dirty
  tree (see Guardrail 5). Do not start editing `walk_forward.py`,
  `run_backtest.py`, or creating the new files until that commit lands.
- **Sibling in-flight plan to be aware of (do not merge scope):**
  `process/general-plans/active/notification-attribution-risk-tier_PLAN_02-07-26.md`
  — CODE-COMPLETE, same two hot files, independent flag, its own A/B still
  pending. Both plans' A/B runs may eventually want to be sequenced (not
  run simultaneously) to avoid GPU/CPU contention and to keep each A/B's
  report attributable to exactly one code change.
- **Execution order inside this plan:**
  1. Commit precondition (Guardrail 5).
  2. `src/trading/regime_policy.py`/`risk_tier.py` — read-only reference, no
     edits.
  3. `src/backtest/walk_forward.py` — add the three new `WalkForwardConfig`
     fields + `_tranche_day` admission branching + `zero_candidate_days`
     counter.
  4. `run_backtest.py` — thread the three new CLI flags through the six-hop
     pattern.
  5. `tests/test_admission_ab.py` — write and run the 4 required test cases
     BEFORE writing the A/B script (prove the engine change is correct in
     isolation first).
  6. `scripts/ab_serve_admission.py` — write the pre-committed 7-config grid
     runner.
  7. Run the grid, produce the report, run the full verification checklist.
- **Open questions needing user decision (surfaced now, not deferred
  silently):**
  1. Confirm the **Correction to task brief** section's reading of serve's
     actual fallback behavior (two nested monitoring-card fallbacks already
     exist) matches the user's understanding of the bug — if the user
     intended something different by "zero tradable BUYs," clarify before
     EXECUTE, since it changes what "serve-mirror" should structurally copy.
  2. Confirm the 7-config grid (floor x top-N combinations) is the right
     shape — in particular whether `admission_pool_cap=6` should also vary
     (this plan holds it fixed at serve's actual `_ARBITRATOR_POOL` value to
     keep the grid at exactly 7 rows; varying it would require expanding the
     grid, which Guardrail 7 says needs a plan revision, not an ad-hoc add).
  3. Confirm `n_trials=28` (7 configs x 4 seeds) vs. `n_trials=7`
     (per-config, using each config's best seed) is the intended DSR
     convention for the eventual report — this plan recommends 28 for
     consistency with `run_backtest.py`'s existing convention but flags it
     as a judgment call worth a second look given the deep-dive feedback's
     Q3 concern about trial-count inflation.
