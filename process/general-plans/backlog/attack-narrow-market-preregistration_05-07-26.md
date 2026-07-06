# Pre-registration: "Attack Narrow-Leadership Markets" — items (1) and (2)

- **Date frozen:** 05-07-26
- **Status:** BACKLOG — criteria frozen BEFORE outcome data exists, per repo
  DSR/trial-count discipline. Do not edit thresholds after data arrives.
- **Context:** June-2026 "green shell, red inside" episode. Admission A/B
  (`serve-admission-ab-result_04-07-26.md`) verdict: absolute gate NOT costly
  on average; even the unrestricted engine lost −1.26% in June-2026. Any
  "attack" capability must come from new signal (item 3, active) or a new
  conditional mechanism (item 2), gated by forward evidence (item 1).
- **Ordering constraint:** item 3 (feature retrain, separate active plan)
  SHIPS FIRST — item 2's sleeve conditions on model ranks; running its A/B on
  a superseded checkpoint is wasted compute. Item 1 accrues by itself.

---

## Item 1 — Paperlog rank-sleeve counterfactual (free, running now)

**Question:** does top-3-by-P(UP), ignoring the absolute gate, make money at
T+20 in the current regime?

**Data:** `sentiment_entry_paperlog` — logs the full cross-section daily since
16-06-26. The daily monitoring-only Top-3 IS the paper sleeve. T+20 outcomes
mature from ~07-07-26.

**Frozen evaluation (run when ready, not before):**
- Sleeve = each day's top-3 tickers by `p_up_20d` (source='daily' rows only),
  equal-weight, T+20 horizon (`ret_20d`).
- Minimum sample: **n ≥ 60 sleeve-name-days** (≈ 20 trading days × 3) with
  `outcome_filled=TRUE` before ANY conclusion is drawn.
- SUCCESS (sleeve idea graduates to item 2's A/B): sleeve mean `ret_20d` > 0
  AND sleeve mean > the same-period equal-weight cross-section mean (i.e.
  ranking adds value over the tape, not just beta).
- FAILURE: either condition misses → item 2 is CANCELLED, not retuned.
- One evaluation per month-of-data, max 3 evaluations total before a
  permanent verdict (multiplicity cap).

**How to run:** extend `scripts/analyze_sentiment_paperlog.py` or a one-off
read-only query (see `paperlog-early-read.py` pattern from the 05-07 session).

## Item 2 — Breadth-conditional rank sleeve (backtest A/B; GATED on item 1)

**Mechanism (new — NOT the rejected gate-replacement):** keep the absolute
gate and the existing book untouched; add a SMALL side-sleeve that activates
only when the market is in the June-signature state:
- breadth negative: % of universe above SMA50 < 40% (exact cut frozen here),
- leadership concentrated: cap-weight index return − equal-weight return > 0
  over trailing 20d.
When active: deploy 0.25× the normal tranche budget into top-3 by rank
(no absolute floor), same hold/cost model, hard cap.

**Pre-committed grid (do not expand):** sleeve size ∈ {0.25×} (ONE value),
breadth cut ∈ {40%} (ONE value), top-N ∈ {3} (ONE value) = **1 config + 1
baseline**, 4 seeds. n_trials for DSR = 2×4 = 8 combined with baseline
convention at run time. If inconclusive → revise via a NEW pre-registration,
never mid-run.

**Verdict rule:** sleeve arm must improve Net PnL AND not worsen MaxDD by
more than 1pp AND pass PBO ≤ 10%. DSR reported; paper-only regardless until
DSR ≥ 0.95 (house rule).

**Precondition:** item 3's retrained checkpoint (new GOLDEN) is the model.
Implementation shape: `WalkForwardConfig` opt-in flag, OFF default,
byte-identical default test — mirrors `use_regime_sizing` / `admission_mode`
precedents in `src/backtest/walk_forward.py`.

## Explicitly rejected (do not re-open without new mechanism)

- Lowering the absolute gate τ to 0.40/0.35 outright — tested 04-07-26,
  winner failed DSR (p=0.22) and PBO (85.6%).
- Raw continuous market-level feature columns in the tabular stack — repeats
  the killed GBM-macro experiment (worse DD + PBO; `use_macro_features`
  default OFF stands).
