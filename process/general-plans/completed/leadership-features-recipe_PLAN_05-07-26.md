# Leadership-Features Recipe Bump — Plan

**Date:** 05-07-26
**Complexity:** COMPLEX (feature engineering + full retrain + statistical evaluation gates, one-shot trial discipline)
**Status:** EXECUTED 05/06-07-26 — verdict **FAIL** (PBO 18.9% > 10% gate; Sharpe/DD gates passed). Code reverted, 02-07 GOLDEN artifacts restored. See `process/general-plans/reports/leadership-features-verdict_05-07-26.md`. No retry under this plan — new pre-registration required (de-saturated grid / smaller feature set recommended). READY TO ARCHIVE.
**Sequencing:** This is "item 3" from `process/general-plans/backlog/attack-narrow-market-preregistration_05-07-26.md`. Item 2 (breadth-conditional rank sleeve) is GATED on this plan's checkpoint (PASS or FAIL) — item 2 must not run against the current (superseded) checkpoint.

---

## 1. Problem Statement (context, do not re-derive)

June-2026 "green shell, red inside": VNINDEX rallied on a narrow set of bank
names (SSB +16.2%, VIB, MBB) while the 331-ticker cross-sectional median
return was ≈ −2.5%. The model's own rank ordering was correct — SSB ranked
#1 by `p_up_20d` — but the absolute admission gate (τ) never opened because
triple-barrier labels are ABSOLUTE-return: in a negative-breadth tape the
entire cross-sectional P(UP) surface compresses below τ, so even a correctly
ranked leader cannot clear the bar. `serve-admission-ab-result_04-07-26.md`
confirmed lowering τ is not a fix (fails DSR/PBO) and the absolute gate is
not, on average, costly. The approved fix direction is **per-ticker
leadership / momentum-persistence features** so a genuine leader's own
P(UP) rises above τ on the strength of its own persistence signal, not a
市場-wide adjustment.

**Explicitly excluded from this plan:** foreign/prop capital-flow features
(data subscription arrives in 1-2 months — future recipe bump, separate
plan). Do not substitute flow proxies here.

## 2. Binding Architecture Constraints (verified this session)

1. **All 14 current model features are cross-sectionally Gaussian-rank
   Z-scored (`_xsz`) per date** via `add_cross_sectional_features`
   (`src/data/tensor_builder.py:243`). A market-wide scalar (e.g. raw
   breadth %) is CONSTANT across tickers on a given date — cross-sectional
   ranking of a constant collapses to a degenerate/undefined rank (ties →
   thin-cross-section fallback = 0.0 for all rows), destroying the signal.
   Market-level information may therefore enter ONLY as:
   (a) a categorical Int8 bucket via the `CATEGORICAL_FEATURES`
       forced-survive passthrough (the `market_regime` precedent,
       `src/features/market_regime.py`), which bypasses xsz entirely and is
       declared categorical to the GBMs at fit time, OR
   (b) a genuinely per-ticker feature that varies across the cross-section
       (so xsz ranking is meaningful).
2. **GBM-macro kill precedent (BINDING, do not repeat):** raw continuous
   market-level columns (`sp500_ret`, `dxy_ret`, `usdvnd_ret`) were A/B'd on
   this exact stack via `RunConfig.use_macro_features` and KILLED — worse
   MaxDD + worse PBO; default stays OFF (`process/features/macro-integration/`).
   This plan MUST NOT re-introduce raw continuous market-level columns as
   GBM features. If a market-context signal is wanted, it must be the
   `market_regime`-style discrete/categorical route, not a continuous join.
3. **The existing candidate pool already competes for exactly top-3 of 5
   slots.** `select_features` (`src/backtest/pipeline.py:625`) runs Step A
   (collinearity filter, |r|>0.65 vs originals → drop), Step B (mutual
   information ranking), Step C (top-K=3 cap) over the CURRENT 5
   candidates: `amihud_liquidity_xsz`, `realized_skewness_20d_xsz`,
   `vol_of_vol_20d_xsz`, `hl_range_ratio_xsz`, `gap_risk_xsz`
   (`src/data/tensor_builder.py:553-632`, wired in
   `src/backtest/pipeline.py:473-494`). Adding new candidates into this SAME
   pool changes selection dynamics (dilutes existing survivors,
   competes for MI ranking) — this is a load-bearing decision the plan
   makes explicitly in Section 4, not an incidental side effect.

## 3. What This Plan Must Settle (frozen before any training — trial-count discipline)

Per repo convention (`attack-narrow-market-preregistration_05-07-26.md`,
DSR/PBO discipline): criteria are frozen BEFORE outcome data exists. No
iterating feature combinations by re-running retrains. One committed
feature set → one retrain per horizon → one verdict. If it fails, a NEW
pre-registration is required to try a different set (not a revision of
this one).

## 4. Frozen Feature Set (FINAL — do not expand during EXECUTE)

**Recommendation: 4 new continuous per-ticker features, added as a NEW,
SEPARATE candidate sub-pool (own top-K), not merged into the existing
5-candidate iron-fist pool.** Rationale in 4.4.

### 4.1 Feature formulas (Polars expressions, leak-safe: every window ends at t, grouped `.over("ticker")`)

All four are computed from RAW OHLCV only (mirrors `mr_features.py` /
`market_regime.py` leak discipline — no post-hoc data, no `.shift(-n)`).

1. **`updown_ratio_20`** — up-day ratio over trailing 20 bars (momentum
   persistence, distinct from magnitude-based `mom20`):
   ```
   daily_ret = close / close.shift(1).over("ticker") - 1.0
   up_flag   = (daily_ret > 0).cast(Int8)
   updown_ratio_20 = up_flag.rolling_mean(window_size=20, min_samples=20).over("ticker")
   ```
   Range [0,1]. A persistent leader (SSB-style grind-up) has a high ratio
   even when `mom20` (pure price change) is muted by volatility.

2. **`efficiency_ratio_20`** — Kaufman efficiency ratio (net displacement /
   path length) — already partially precedented in `market_regime.py`'s
   internal `_er` (there computed over 14 bars, NOT exposed as a model
   feature). This is a NEW exposed continuous candidate at 20 bars to
   match the model's horizon:
   ```
   net_move  = (close - close.shift(20).over("ticker")).abs()
   path_len  = (close - close.shift(1).over("ticker")).abs()
                 .rolling_sum(window_size=20, min_samples=20).over("ticker")
   efficiency_ratio_20 = net_move / (path_len + 1e-9)
   ```
   Range [0,1]. High = clean directional trend (leadership); low = choppy/
   noisy path covering the same net distance.

3. **`dist_from_252high`** — distance from the trailing 252-day (52-week)
   high, signed so a fresh high = 0 and further below = more negative:
   ```
   roll_high_252 = close.rolling_max(window_size=252, min_samples=252).over("ticker")
   dist_from_252high = (close - roll_high_252) / (roll_high_252 + 1e-9)
   ```
   Range (-inf, 0]. A true leader making new highs sits at/near 0; laggards
   sit deeply negative. NOTE: 252-bar warm-up is longer than every existing
   feature's window (current max is `overext_20`/`rs_20`'s 20 bars) — see
   Section 4.3 for the row-loss implication that must be measured, not
   assumed negligible.

4. **`rs_rank_stability_10`** — rolling stability (inverse volatility) of
   the ticker's OWN cross-sectional `rs_20` rank over the trailing 10 days.
   This is the one feature in the set that depends on an already-computed
   xsz column (`rs_20_xsz`) rather than raw OHLCV — computed AFTER
   `add_alpha_factors` in the pipeline ordering (Section 5):
   ```
   rs_rank_stability_10 = -1.0 * rs_20_xsz.rolling_std(window_size=10, min_samples=10).over("ticker")
   ```
   Sign convention: LESS volatile (more stable) cross-sectional rank →
   HIGHER (less negative) value. A name whose relative-strength rank is
   consistently near the top for 10 days (SSB-style persistent leadership)
   has low rank volatility → high `rs_rank_stability_10`. A name that spikes
   in and out of the top ranks has high rank volatility → low value.
   This is itself a raw (non-`_xsz`) signal fed forward for its own
   cross-sectional Z-score in Section 4.2 (double-ranking is intentional:
   first rank per-date for `rs_20_xsz`, then rank the STABILITY of that
   rank — these measure different things and are not redundant by
   construction, but Step A collinearity in selection will catch it if
   they end up correlated in practice).

### 4.2 xsz vs categorical

All 4 are **continuous, cross-sectionally Z-scored** (`_xsz` suffix) —
each varies per-ticker per-date, so xsz ranking is meaningful (satisfies
constraint 2.1(b)). None are categorical. Final column names entering
`FEATURE_SCHEMA`:
- `updown_ratio_20_xsz`
- `efficiency_ratio_20_xsz`
- `dist_from_252high_xsz`
- `rs_rank_stability_10_xsz`

### 4.3 Rejected from the set (with reasoning, so EXECUTE does not re-litigate)

- **MA-alignment flag** (from the candidate family list in the routing
  prompt) — REJECTED. `market_regime`'s Strong-Trend bucket (regime 3)
  already encodes `ma_up`/`ma_dn` alignment structurally
  (`market_regime.py:156-158`); adding a redundant per-ticker MA-alignment
  feature is likely to fail Step A collinearity against `market_regime`
  or against `mom20_xsz`/`overext_20_xsz`, wasting a trial slot for near-
  zero incremental information.
- **Market-wide breadth categorical bucket** (e.g. Int8 {0: broad-down, 1:
  narrow, 2: broad-up}) — REJECTED for this first shot. See Section 4.4
  for the full reasoning (this is one of the two explicit open questions
  the routing prompt asked to be surfaced, and the recommendation is to
  decline it now).
- **252-day feature count** — capped at ONE (`dist_from_252high`) rather
  than also adding a 252-day efficiency ratio or similar, specifically to
  bound the row-loss / warm-up cost (Section 4.5) to a single long-window
  feature.

### 4.4 Open Question 1 — breadth categorical bucket: RECOMMENDATION = OUT of this first shot

Arguments for including it (steelman): `market_regime` proves the
categorical-bucket route is architecturally safe (constraint 2.1(a)); a
breadth regime {broad-down, narrow, broad-up} directly encodes the exact
June-2026 diagnosis (negative breadth + concentrated leadership) that
per-ticker features can only proxy indirectly.

Arguments against (why this plan recommends declining it now):
1. **It duplicates the GBM-macro kill's failure mode at one remove.** The
   killed macro features were raw market-level continuous columns; a
   breadth bucket is market-level information too, just discretized. The
   repo's own precedent shows market-level signal on this stack has a
   documented failure history (worse DD + PBO) even when the *directional*
   hypothesis (macro conditions matter) was reasonable. A categorical
   encoding reduces but does not eliminate the risk that the GBM stack
   overfits to a handful of historical breadth regimes.
2. **Trial-count discipline.** Item 2 in the sibling pre-registration
   ALREADY proposes a breadth-conditional MECHANISM (a sizing sleeve, not
   a model feature) gated on breadth<40% + concentration>0 over trailing
   20d — using nearly the identical breadth signal this plan would encode
   as a feature. Building it into BOTH the feature recipe (this plan) and
   the sizing sleeve (item 2) at the same time makes it impossible to
   attribute any future performance change to one mechanism vs the other,
   and doubles the trial surface before either is validated once.
3. **`market_regime`'s bucket is PER-TICKER structural context** (each
   ticker gets its own regime label from its own OHLCV), not a market-wide
   scalar broadcast to every name on a date. A breadth bucket would be the
   first FEATURE_SCHEMA column that is identical across the entire
   cross-section on a given day — a genuinely new architecture pattern,
   not a repeat of an existing precedent, and deserves its own isolated
   A/B rather than bundling into this leadership-feature trial.

**Verdict: ship the 4 per-ticker features only in this shot.** If item 1's
paperlog counterfactual (running now, matures ~07-07-26) and this plan's
verdict both look promising, propose the breadth bucket as ITS OWN
follow-up pre-registration — never silently added mid-plan.

### 4.5 Row-loss / warm-up cost (must be measured in EXECUTE, not assumed)

`dist_from_252high` requires 252 bars of history per ticker before it is
non-null, versus the current maximum window in `FEATURE_SCHEMA` (20 bars
for `overext_20`/`rs_20`/`amihud`/`vol_of_vol`). `build_features` drops
any row with a null in the continuous pool
(`src/backtest/pipeline.py:525`, `df.drop_nulls(subset=continuous)`). This
means every ticker loses its first ~232 additional trading days (252 − 20)
of history to warm-up versus the current recipe, and any ticker with under
252 days of total history (already filtered somewhat by `min_history=120`
in `RunConfig`, but 120 < 252) is dropped ENTIRELY from the panel, not just
truncated. EXECUTE must log and report the before/after row count and
ticker count in the leak/unit-test step (Section 6, Step 3) so this cost is
visible before committing to the full retrain — this is a candidate
abort point if the row loss is severe (see Section 8 abort criteria).

## 5. Pipeline Wiring (exact touchpoints)

### 5.1 New module: `src/features/leadership_features.py`

Mirrors `src/features/mr_features.py`'s docstring discipline (WHY /
LOOK-AHEAD SAFETY / OUTPUT CONTRACT sections) and `market_regime.py`'s
pure-Polars vectorized style (this feature set operates on the SAME Polars
panel already flowing through `build_features`, so Polars — not the
pandas style of `mr_features.py` — is the correct implementation
language; `mr_features.py` is pandas because it serves a separate MR
sub-model pipeline, not this one).

Public function signature:
```
def add_leadership_features(
    df: pl.DataFrame,
    *,
    updown_window: int = 20,
    efficiency_window: int = 20,
    high_window: int = 252,
    rank_stability_window: int = 10,
    cross_sectional: bool = True,
    clip_z: float = 4.0,
) -> pl.DataFrame:
```//
Returns `df` + the 4 raw columns (`updown_ratio_20`,
`efficiency_ratio_20`, `dist_from_252high`, `rs_rank_stability_10`) and,
when `cross_sectional=True`, their `_xsz` counterparts via
`add_cross_sectional_features` (imported from
`src.data.tensor_builder`, same helper the existing alpha factors use —
do not reimplement the Gaussian-rank transform).

`rs_rank_stability_10` REQUIRES `rs_20_xsz` to already exist on `df` as an
input column — the function must raise `ValueError` if `rs_20_xsz` is
missing, mirroring the `_REQUIRED` column-check pattern in
`mr_features.py:101-103`.

Module-level constants (mirrors `mr_features.py:51-66` /
`market_regime.py` pattern):
```
LEADERSHIP_FEATURE_COLUMNS: list[str] = [
    "updown_ratio_20", "efficiency_ratio_20",
    "dist_from_252high", "rs_rank_stability_10",
]
```
No standalone schema-hash constant needed here (unlike `mr_features.py`'s
`MR_SCHEMA_HASH`) — this module feeds `pipeline.py`'s `FEATURE_SCHEMA`,
which already owns `FEATURE_RECIPE_VERSION` computation. Do not create a
second, redundant hash.

### 5.2 `src/backtest/pipeline.py` changes

1. **Import**: add `from src.features.leadership_features import
   add_leadership_features` near the existing feature imports
   (`pipeline.py:40-48`).
2. **`build_features()` wiring** (`pipeline.py:442-533`): call
   `add_leadership_features(df, cross_sectional=True)` AFTER
   `add_alpha_factors` (so `rs_20_xsz` exists for
   `rs_rank_stability_10`) and BEFORE `add_advanced_statistical_features`
   (ordering is arbitrary between these two but must be documented in the
   call site comment — the plan fixes it as: FracDiff → mom20 → xsz →
   overextension → alpha factors → **leadership features (NEW)** →
   advanced statistical → regime).
3. **New candidate sub-pool, NOT merged into the existing 5**
   (Section 2.3 concern). Introduce a SECOND named candidate list:
   ```
   leadership_candidate_features = [
       "updown_ratio_20_xsz", "efficiency_ratio_20_xsz",
       "dist_from_252high_xsz", "rs_rank_stability_10_xsz",
   ]
   ```
   These are appended to `all_features` (continuous pool, before
   `CATEGORICAL_FEATURES`) exactly like the existing `candidate_features`
   is today (`pipeline.py:505`: `continuous = original_features +
   candidate_features` → becomes `continuous = original_features +
   candidate_features + leadership_candidate_features`).
4. **`FEATURE_SCHEMA` update** (`pipeline.py:63-79`): append the 4 new
   `("<name>_xsz", "Float32")` entries BEFORE the trailing
   `("market_regime", "Int8")` entry (continuous-then-categorical order is
   load-bearing per the existing comment at line 61-62). This changes
   `FEATURE_RECIPE_VERSION`'s computed hash automatically (it is derived
   from `FEATURE_SCHEMA` via `compute_feature_schema_hash`,
   `pipeline.py:207-209`) — no manual version bump needed, confirming the
   binding convention in `process/context/all-context.md`.
5. **`select_features()` call site in `train_models.py:116-124`**: this
   is the decision point for how the two candidate pools interact.
   **Chosen approach (must be implemented exactly, no ad-lib):** widen
   the existing single `select_features` call's `candidate_features`
   argument to the CONCATENATION of the old 5 + new 4 = 9 candidates, but
   **raise `top_k` from 3 to 4** (not left at 3, and not raised further).
   Rationale: leaving `top_k=3` while adding 4 new competitors would make
   it structurally impossible for even one new leadership feature to
   survive without evicting an existing survivor, silently re-litigating
   the ALREADY-VALIDATED existing candidate pool. Raising to exactly 4
   (not 9, not "all survive") preserves the iron-fist discipline's intent
   (cap the final model at a small, MI-ranked pool) while giving the new
   family a fair, non-zero chance without unbounded pool growth. This is
   the single knob change EXECUTE is allowed to make in `train_models.py`;
   `corr_threshold=0.65` stays unchanged.
6. **`materialize_dataset()`** (`pipeline.py:774-777`): `original_features`
   is derived as `all_features` minus `candidate_features` minus
   `CATEGORICAL_FEATURES` — this must also exclude the new
   `leadership_candidate_features`, i.e. the exclusion list becomes
   `candidate_features + leadership_candidate_features +
   CATEGORICAL_FEATURES`. Read this exact line before editing; do not
   assume — confirm the excluded-sets union is correct so leadership
   candidates don't leak into `original_features` (which always survive,
   defeating the point of putting them through selection).

### 5.3 Files touched (complete list)

- `src/features/leadership_features.py` — NEW
- `src/backtest/pipeline.py` — `FEATURE_SCHEMA`, `build_features()`,
  `materialize_dataset()` (exclusion-set fix)
- `train_models.py` — `select_features()` call site (`candidate_features`
  concat + `top_k=3→4`)
- `tests/test_leadership_features.py` — NEW (Section 6)
- `run_backtest.py` — NO code change expected (reads `tabular_features`
  from the checkpoint; confirm no hardcoded feature-count assumption
  exists — grep for `len(tabular_features)` usages defensively before
  declaring this a no-op)
- `models/saved/*.joblib` — regenerated ONLY at the final swap step
  (Section 9), never touched before the verdict

## 6. Evaluation Protocol (one-shot, no iteration)

**Step 1 — Implement.** Write `leadership_features.py` + wire per Section
5. No training yet.

**Step 2 — Leak regression tests** (`tests/test_leadership_features.py`,
mirrors `mr_features.py`'s `if __name__ == "__main__"` smoke-test pattern
translated into pytest, and `test_market_regime.py`'s structure):
- `test_updown_ratio_range` — output ∈ [0,1] (or null pre-warm-up), matches
  a hand-computed synthetic up/down sequence.
- `test_efficiency_ratio_range` — output ∈ [0,1] on a synthetic straight-
  line trend (expect ≈1.0) vs a synthetic sawtooth of equal net
  displacement (expect ≪1.0).
- `test_dist_from_252high_nonpositive` — output ≤ 0 everywhere it is
  non-null; == 0 exactly at the bar making a new 252-day high.
- `test_rs_rank_stability_requires_rs20` — raises `ValueError` if
  `rs_20_xsz` is absent from the input frame.
- `test_leak_free_leadership_features` — THE critical test, same pattern
  as `mr_features.py:225-232` / `market_regime.py:220-230`: build features
  on a base synthetic panel, then mutate ONLY future bars (e.g. bars
  beyond index 60 of an 80-bar series) and rebuild; assert all 4 raw + 4
  `_xsz` columns are byte-identical for every bar BEFORE the mutation
  point. This must pass before any retrain is run.
- `test_row_loss_252_warmup` — on a multi-ticker synthetic panel with
  mixed history lengths (some <252 bars, some >252), assert the
  documented behavior from Section 4.5: tickers with <252 bars produce
  either an entirely-null `dist_from_252high` column (dropped downstream)
  or are otherwise correctly excluded — this test's job is to CONFIRM the
  claimed row-loss mechanics, not to change them.

**Step 3 — Measure real-panel impact BEFORE training.** Run
`materialize_dataset()` (or an equivalent standalone script) against the
real OHLCV panel with the new recipe and log: total row count before vs.
after this change, ticker count before vs. after, and confirm this
against the abort criteria in Section 8. Do NOT proceed to Step 4 until
this number is reported and reviewed (the plan does not pre-approve
proceeding if row loss is severe — see Section 8).

**Step 4 — ONE retrain, T+20 only, first.** `python train_models.py
--tb-horizon 20` (all other flags default — 4 seeds, same
`train_frac=0.70`, same `frac_diff_d=0.4`). Do not retrain T+5 in this
same shot — see Section 7 for the explicit T+5 recommendation.

**Step 5 — GOLDEN threshold sweep.** `python run_backtest.py --mode
tranche --hold-days 30` against the new T+20 checkpoint (existing
machinery, `run_backtest.py:311` `main()`, unchanged sweep grid — do not
add new threshold candidates, reuse `DEFAULT_SWEEP_THRESHOLDS`).

**Step 6 — June-2026-signature monthly slice.** Extract the June-2026
row from the GOLDEN's `monthly_cols`/`monthly_net_sharpe` output
(`run_backtest.py:172-181`, already computed per-seed inside the sweep
loop at `run_backtest.py:434`) and report it explicitly alongside the
full-OOS headline metrics — this is the number that most directly answers
"did this fix narrow-leadership months," even though the full verdict
(Section 7) is NOT conditioned on this slice alone (n=1 month is not a
statistically meaningful sample; it is a diagnostic, not a pass/fail
criterion).

**Step 7 — Teardown.** Compare against the frozen T+20 GOLDEN baseline
from `process/general-plans/reports/retrain_t5_t20_result_02-07-26.md`
(Sharpe +0.689, MaxDD −13.16%, DSR p=0.3146, PBO 3.0%) per the verdict
rule in Section 7. Write the result to
`process/general-plans/reports/leadership-features-verdict_<run-date>.md`
(same reporting convention as the 02-07-26 retrain report).

## 7. Verdict Criteria (FROZEN — do not renegotiate after seeing results)

**PASS** requires ALL of:
1. New-recipe T+20 mean Sharpe (across 4 seeds, GOLDEN threshold) >
   +0.689 (current GOLDEN baseline).
2. New-recipe T+20 MaxDD not worse than −13.16% by more than 1 percentage
   point, i.e. MaxDD ≥ −14.16%.
3. New-recipe T+20 PBO (CSCV) ≤ 10%.

**Any single miss → FAIL.** On FAIL: `git revert` the pipeline/module
changes (Section 5), leave `models/saved/*.joblib` untouched (still the
02-07-26 GOLDEN artifacts), and write the verdict report documenting
which criterion failed and by how much. A FAIL here means: do NOT
silently try a different feature subset in a follow-up commit under this
same plan — open a NEW pre-registration per repo trial-count discipline
(same rule the sibling backlog item states explicitly).

**DSR is reported, not gated** — consistent with existing practice (T+20
currently fails DSR at p=0.31 and is still "the GOLDEN" for paper-trading
purposes; the house rule is DSR≥0.95 before LIVE capital, not before
paper-only iteration). Regardless of PASS/FAIL on the 3 gates above, this
recipe stays PAPER-ONLY exactly like every current artifact — this plan
does not change the paper/live gate.

**June-signature monthly slice** (Section 6, Step 6) is reported for
narrative diagnosis only and is NOT part of the PASS/FAIL gate (n=1
month, not statistically robust — see Step 6 rationale).

## 8. Abort Criteria (mid-execution, before the retrain commits GPU time)

If Step 3 (Section 6) shows:
- more than 15% of the current row count is dropped due to the 252-day
  warm-up, OR
- more than 10% of currently-tradeable tickers are dropped entirely
  (insufficient total history for `dist_from_252high` to ever populate),

then STOP before Step 4 (the retrain) and report back to the user with
the exact numbers. Do not unilaterally shrink the `high_window` from 252
to something smaller to route around this — that would be silently
re-litigating the frozen Section 4 feature set. The correct response to
tripping this abort criterion is: report the finding, and let the user
decide whether to accept the row loss, drop `dist_from_252high` from the
set (which requires re-freezing Section 4, i.e. a plan amendment/re-
approval, NOT a unilateral EXECUTE decision), or proceed anyway.

## 9. Open Question 2 — T+5 in this shot? RECOMMENDATION = NO, T+20 ONLY

Argument for including T+5 in the same shot: symmetry with the 02-07-26
retrain report, which always retrained both horizons together, and
avoids a second retrain cycle later if T+20 passes and T+5 is wanted too.

Argument against (this plan's recommendation): T+5's PBO is ALREADY at
43.1% (FAIL, `retrain_t5_t20_result_02-07-26.md`) — the WORST overfit
reading of either horizon on record, and per that same report's own
caveat, "saturated threshold sweeps mechanically inflate PBO" for T+5
specifically (referencing `tranche_sweep_validation_12-06-26.md`), meaning
T+5's selection-robustness signal is already suspect independent of any
feature change. Adding 4 new candidate features to an already
overfitting-prone horizon, in the SAME trial as the more important T+20
test, muddies attribution: if T+5 also improves, it is unclear whether
that reflects real signal or is riding the same saturation artifact
already flagged. Retraining T+5 doubles the GPU wall-clock
(~35min → ~70min per the environment notes) for a horizon whose baseline
statistical footing is already the weaker of the two.

**Decision: T+20 only in this plan's execution.** If T+20 PASSES the
Section 7 verdict, T+5 becomes a natural, cheap follow-up (same recipe,
already-implemented feature module — just `train_models.py --tb-horizon
5` + `run_backtest.py`) but that follow-up is a SEPARATE decision made
after seeing the T+20 verdict, not pre-committed here.

## 10. Serve Safety Sequencing

1. Serve (`main.daily_inference`, the Telegram bot, the dashboard)
   continues running on the CURRENT `models/saved/v3_ensemble_20d.joblib`
   / `v3_ensemble_5d.joblib` / `v3_training_checkpoint.joblib`
   (02-07-26 GOLDEN artifacts) for the entire duration of this plan's
   Steps 1-7.
2. The `FEATURE_RECIPE_VERSION` schema-hash gate
   (`main._load_v3_bot`, per `process/context/all-context.md`'s binding
   convention) makes any accidental mismatch between a new-recipe
   checkpoint and the old-recipe serve path a HARD, LOUD error — this is
   the existing safety net, not something this plan needs to add.
3. `models/saved/*.joblib` files are overwritten ONLY by
   `run_backtest.py`'s `_persist_bot_payload` step (Section 6, Step 5) —
   this happens automatically as part of the retrain+backtest run
   (`save_bot_payload=True` is the CLI default), which means the NEW
   artifacts land on disk as soon as Step 5 completes, BEFORE the verdict
   is evaluated (Step 7). This is acceptable ONLY because:
   - the previous artifact is backed up before being overwritten
     (`run_backtest.py:518`, "Back up the EXISTING artifact before
     overwriting"), and
   - the schema-hash gate prevents the live bot from silently switching
     onto a mismatched recipe without an explicit code deploy — there is
     no `main.py`/serve-path code change in this plan, so nothing in the
     live serve path picks up the new artifact automatically.
   EXECUTE must nonetheless explicitly confirm the backup fired
   (check `backups/` for the pre-overwrite `.joblib`) as part of the Step
   5 report, since this is the closest this plan comes to a live-artifact
   mutation before the verdict is final.
4. **On FAIL** (Section 7): the new-recipe artifacts sit in
   `models/saved/` but the serve path never reads them because no code
   change routes serve to a "new" artifact path — serve always loads
   `v3_ensemble_20d.joblib` by convention. Restore the pre-overwrite
   backup from `backups/` to `models/saved/v3_ensemble_20d.joblib` (and
   the checkpoint) so the filesystem state matches the pipeline-code
   revert from Section 7, avoiding any drift between "code says old
   recipe" and "artifact on disk is new recipe."
5. **On PASS**: the artifact swap is already effectively done (Step 3
   above) — no further "swap" action needed, but EXECUTE must get
   explicit user approval before considering the new checkpoint the
   production-referenced GOLDEN (i.e. before updating
   `process/context/all-context.md`'s Current Features table or any
   other doc that names the active baseline).

## 11. Runtime / Environment Notes

- RTX 3050 4GB VRAM → sequential (not parallel) seed×model GBM fits;
  historically ~35 minutes for train+backtest per horizon
  (`retrain_t5_t20_result_02-07-26.md`: 2088.0s / 2007.4s wall-clock).
  This plan commits to ONE such cycle (T+20 only, Section 9).
- Test runner: bare `pytest` (not the `stock` conda env) per
  `process/context/tests/all-tests.md` conventions — confirm this at
  EXECUTE time by reading that file's quick-start if not already loaded.
- `run_bot.py` may be running live (DuckDB lock on
  `data/quant_v6_core.duckdb`) — nothing in Steps 1-4 of this plan touches
  the live DuckDB write path; `load_ohlcv` is parquet-first read-only
  (`pipeline.py:268-334`). No coordination with the live bot process is
  required for this plan's execution.
- `models/saved/` gets NEW artifacts only at Step 5 (Section 6) — not
  before.

## 12. Implementation Checklist (atomic, ordered)

1. Create `src/features/leadership_features.py` with
   `add_leadership_features()` implementing the 4 formulas in Section 4.1,
   the `LEADERSHIP_FEATURE_COLUMNS` constant, the `rs_20_xsz`
   presence-check `ValueError`, and a module docstring mirroring
   `mr_features.py`'s WHY / LOOK-AHEAD SAFETY / OUTPUT CONTRACT structure.
2. Create `tests/test_leadership_features.py` implementing all 6 tests
   listed in Section 6 Step 2 (range checks ×3, the `ValueError` guard,
   the leak-free regression test, and the row-loss confirmation test).
   Run `pytest tests/test_leadership_features.py -v` and confirm all pass
   before touching `pipeline.py`.
3. Edit `src/backtest/pipeline.py`:
   a. Add the import for `add_leadership_features`.
   b. Insert the `add_leadership_features(df, cross_sectional=True)` call
      in `build_features()` immediately after the `add_alpha_factors`
      call and before `add_advanced_statistical_features`.
   c. Define `leadership_candidate_features` list (the 4 `_xsz` names).
   d. Update the `continuous = ...` assembly line to include
      `leadership_candidate_features`.
   e. Append the 4 new `("<name>_xsz", "Float32")` tuples to
      `FEATURE_SCHEMA`, immediately before the `("market_regime", "Int8")`
      entry.
   f. Confirm (read, do not assume) and fix the `original_features`
      exclusion-set computation in `materialize_dataset()` so it also
      excludes `leadership_candidate_features`.
   g. Confirm the `all_features == expected_schema` assertion
      (`pipeline.py:517-521`) still holds given the new column order.
4. Edit `train_models.py`'s `select_features()` call site: concatenate
   `ds.candidate_features` with the new `leadership_candidate_features`
   list (import or thread it through from `pipeline.py`'s
   `build_features`/`materialize_dataset` return values — confirm the
   cleanest way to expose this list to `train_models.py` without breaking
   the `Dataset` dataclass contract; adding a new `Dataset` field may be
   necessary — decide and document at EXECUTE time, this plan does not
   pre-specify the dataclass field name), and change `top_k=3` to
   `top_k=4`. Leave `corr_threshold=0.65` unchanged.
5. Run the FULL existing test suite (`pytest`) to confirm no regression
   from the `FEATURE_SCHEMA` / `Dataset` shape change — pay particular
   attention to `tests/test_run_backtest_wiring.py`,
   `tests/test_tabular_ensemble.py`, and `tests/test_triple_barrier.py`
   (hub-node coverage per `process/context/all-context.md`) since they
   exercise the exact dataclasses touched here.
6. Run Section 6 Step 3: materialize the dataset against the real OHLCV
   panel (via a short standalone script or an interactive `python -c`
   invoking `materialize_dataset(RunConfig(tb_horizon=20))`) and log
   before/after row count and ticker count. Compare against the Section 8
   abort thresholds (15% row loss, 10% ticker loss). STOP and report if
   tripped; otherwise proceed.
7. Run `python train_models.py --tb-horizon 20` (T+20 only, per Section
   9). Monitor for the expected ~35min wall-clock; confirm GPU engagement
   via `nvidia-smi` the same way the 02-07-26 retrain confirmed it.
8. Run `python run_backtest.py --mode tranche --hold-days 30` against the
   new T+20 checkpoint. Capture the full sweep table, GOLDEN teardown
   metrics (Sharpe, MaxDD, DSR, PBO), and the June-2026 monthly-slice
   Sharpe from `monthly_cols`.
9. Confirm the pre-overwrite backup of the prior `v3_ensemble_20d.joblib`
   / `v3_training_checkpoint.joblib` exists in `backups/` (Section 10,
   point 3).
10. Apply the Section 7 verdict rule mechanically against the captured
    metrics vs. the 02-07-26 baseline. Write
    `process/general-plans/reports/leadership-features-verdict_<run-date>.md`
    with the full sweep table, teardown metrics, June-slice number, and
    the PASS/FAIL determination with reasoning.
11. **On FAIL**: `git revert` the commits from steps 1-4 above (or a
    single squashed revert if committed as one unit); restore
    `models/saved/v3_ensemble_20d.joblib` +
    `models/saved/v3_training_checkpoint.joblib` from the Step 9 backup.
    Report the reverted state and the verdict reasoning to the user.
    Stop — do not propose a modified feature set in the same plan.
12. **On PASS**: report to the user and explicitly ask for approval
    before (a) treating the new checkpoint as the referenced GOLDEN in
    any context doc, and (b) considering the T+5 follow-up retrain
    (Section 9) or unblocking item 2's sleeve A/B (the sibling backlog
    plan's stated precondition).

## 13. Touchpoints

- `src/features/leadership_features.py` (new)
- `src/backtest/pipeline.py` (`FEATURE_SCHEMA`, `build_features`,
  `materialize_dataset`, possibly `Dataset` dataclass field)
- `train_models.py` (`select_features` call site)
- `tests/test_leadership_features.py` (new)
- `models/saved/v3_ensemble_20d.joblib`,
  `models/saved/v3_training_checkpoint.joblib`,
  `models/saved/prob_distribution.png` (regenerated by Step 8, backed up
  automatically)
- `process/general-plans/reports/leadership-features-verdict_<run-date>.md`
  (new, written at Step 10)

## 14. Public Contracts

- `add_leadership_features(df, **kwargs) -> pl.DataFrame` — new public
  function, consumed only by `pipeline.py::build_features`.
- `LEADERSHIP_FEATURE_COLUMNS: list[str]` — new module constant.
- `FEATURE_SCHEMA` — extended (4 new tuples); `FEATURE_RECIPE_VERSION`'s
  computed value changes as a DIRECT, EXPECTED consequence (this is the
  hard gate the binding convention describes — any consumer pinning the
  old hash string will correctly fail loudly).
- `select_features(..., top_k=4)` call-site change in `train_models.py`
  — the `select_features` function signature itself (`pipeline.py:625`)
  is NOT changed, only the value passed at the call site.
- Possible `Dataset` dataclass field addition (Section 12, Step 4) — if
  added, must be additive (new optional/defaulted field), not a rename or
  removal of any existing `Dataset` field, to avoid breaking
  `run_backtest.py`'s consumption of the same dataclass.

## 15. Blast Radius

- **Direct:** `src/backtest/pipeline.py` is THE shared dataset-
  construction module for both `train_models.py` and `run_backtest.py`
  (per its own module docstring) — a `FEATURE_SCHEMA` change here is felt
  by both entry points, by design and by necessity (train/serve parity).
- **Indirect via schema-hash gate:** any artifact trained under the new
  recipe becomes incompatible with the OLD serve code's expected hash
  (this is the intended safety property, not a bug) — confirms no
  accidental cross-recipe artifact swap can occur silently.
- **Test suite:** `tests/test_run_backtest_wiring.py`,
  `tests/test_tabular_ensemble.py`, `tests/test_triple_barrier.py`
  (hub-node coverage, per `process/context/all-context.md`) are the tests
  most likely to need a fixture update if they hardcode feature counts or
  `FEATURE_SCHEMA` length assumptions — audit these specifically during
  Step 5 of the checklist, not just "run pytest and see."
- **Not touched:** `src/bot/bot_inference.py`, `main.py` serve dispatch,
  `dashboard/`, the Telegram bot — none of these have hardcoded feature
  lists (they consume whatever `tabular_features` the checkpoint declares
  at load time), so this plan does not expect to touch them. Confirm this
  assumption is still true by grepping for any hardcoded reference to the
  9-feature or 14-column names during EXECUTE, but do not pre-emptively
  edit these files.
- **Not touched:** item 2's sleeve mechanism (sibling backlog plan) — its
  own separate plan, gated on this one's checkpoint, not edited here.

## 16. Verification Evidence

- `pytest tests/test_leadership_features.py -v` — all new tests green,
  captured before Step 3 of the checklist proceeds.
- Full `pytest` run post-wiring (Section 12, Step 5) — zero regressions,
  captured before the retrain (Step 7) proceeds.
- Row/ticker count log from Section 6 Step 3 / checklist Step 6 —
  captured and compared against Section 8 thresholds before the retrain.
- `nvidia-smi` GPU-engagement snapshot during the Step 7 retrain (same
  evidence bar as `retrain_t5_t20_result_02-07-26.md`).
- Full sweep table + GOLDEN teardown metrics + June-2026 monthly-slice
  Sharpe from Step 8, captured verbatim into the verdict report (Step
  10) — no paraphrasing of numbers, copy the actual sweep/teardown output.
- Backup-file existence check (Step 9) before any verdict is finalized.

## 17. Resume and Execution Handoff

- **Selected plan file for EXECUTE:**
  `process/general-plans/active/leadership-features-recipe_PLAN_05-07-26.md`
  (this file — the only plan file for this work; no legacy phase-file
  siblings exist for this effort).
- **Preconditions before EXECUTE starts:** none outside this plan itself
  — this is a self-contained unit of work. It does NOT need to wait on
  item 1's paperlog maturation (that item is independent, "accrues by
  itself" per the sibling backlog doc) or on item 2 (item 2 waits on
  THIS plan, not the reverse).
- **What EXECUTE must NOT do:** expand the Section 4 feature list, change
  `top_k` beyond 3→4, retrain T+5 without a separate explicit go-ahead
  after the T+20 verdict, or route around the Section 8 abort criteria by
  silently shrinking `high_window`.
- **What happens after this plan's verdict, regardless of PASS/FAIL:**
  report back to the user with the closeout packet (verdict, what was
  verified, what remains). On PASS, the natural next actions are (a) the
  optional T+5 follow-up, and (b) unblocking item 2's sleeve A/B in the
  sibling backlog plan — neither is auto-started.

---

## Open Questions Summary (as requested — do not resolve silently)

1. **Final feature list + count**: RECOMMENDED = 4 features
   (`updown_ratio_20_xsz`, `efficiency_ratio_20_xsz`,
   `dist_from_252high_xsz`, `rs_rank_stability_10_xsz`), added as a
   SEPARATE candidate sub-pool with `top_k` raised from 3→4 (not merged
   into the existing 5-candidate pool at unchanged `top_k=3`). See
   Section 4 for full formulas and Section 4.3 for what was rejected and
   why.
2. **Breadth categorical bucket — in or out?**: RECOMMENDED = OUT of this
   first shot. Full reasoning in Section 4.4 (duplicates item 2's
   mechanism, repeats the GBM-macro market-level risk pattern at one
   remove, deserves its own isolated pre-registration if wanted later).
3. **T+5 in the same shot?**: RECOMMENDED = NO, T+20 only. Full reasoning
   in Section 9 (T+5's PBO is already the worse of the two horizons and
   already flagged as possibly saturation-inflated; conflating a feature
   trial with an already-shaky horizon muddies attribution and doubles
   GPU time for the less important test).
