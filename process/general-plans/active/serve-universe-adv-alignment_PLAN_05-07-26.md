# Serve Universe / ADV Alignment (replace static VN30 gate with dynamic top-N ADV)

- **Date:** 05-07-26
- **Type:** SIMPLE
- **Status:** NOT STARTED
- **Scope:** Serve-path only (`main.py::_select_candidates` + a new small
  candidate-universe builder module + config). Does NOT touch the absolute
  meta-gate admission rule, the arbitrator pool cap, dispatch top-3, tranche
  weight math, the sentiment paper-log, or the feature recipe. No retrain.

## Background (research complete this session — cite, do not redo)

1. **Serve/backtest universe mismatch.** All validated backtest evidence
   (GOLDEN T+20 Net +32.55%, Sharpe +0.689, 911 OOS days; today's
   `serve-admission-tranche-ab_PLAN_04-07-26.md` A/B) runs the tranche engine
   with a DYNAMIC top-50 trailing-20d ADV universe gate:
   `WalkForwardConfig.liquid_top_n=50`, `adv_window=20`,
   `_apply_liquidity_filter` (`src/backtest/walk_forward.py:543-569`) ranks
   `adv20 = (close×volume) rolling-mean(20).shift(1)` WITHIN each date and
   keeps the top 50 (leak-free — shift(1) means the gate for day D uses only
   data through D-1). Serve alone narrows candidates to the STATIC hardcoded
   `_VN30_UNIVERSE` frozenset (`main.py:857-861`, 30 names). Serve deviates
   from the exact universe definition its own validated backtest evidence
   was produced under.
2. **Static-list rot (measured 04-07-26 from live parquets).** 6 of 30 VN30
   names are NOT currently in the top-50 ADV book (BCM, BVH, CTG, POW, SAB,
   and June's best performer SSB is borderline/rotating); 2 VN30 names never
   appear in predictions at all. Serve logs confirm narrow pass-through
   (`[VN30Gate] 28/356 in VN30`). The frozenset's own code comment
   ("Update on the quarterly VN30 review") has not been honored since it was
   written.
3. **June-2026 measurement.** Of the 10 best June performers within the
   top-50 ADV book, 6 were OUTSIDE VN30 (MSB +12.3%, PVD +8.5%, TCX +8.4%,
   HCM +5.2%, VND +4.8%, VCK +4.6%). Incremental-26 names' June mean return
   was −2.77% (worse than VN30's −1.35%) — this is not itself a reason to
   include them: a top-N ranking strategy harvests the right tail of a wider
   distribution, not the mean, and the validated tranche backtest's positive
   results already include these names in its universe.
4. **Honesty constraint (must appear in this plan's own framing, not just
   here).** This change would NOT have unblocked June's "all-defend" episode
   by itself — SSB was already VN30-eligible all month; the absolute
   probability gate was what blocked it (see the sibling
   `serve-admission-tranche-ab_PLAN_04-07-26.md` A/B, whose verdict was
   REJECTED changing that gate — evidence showed the gate is not measurably
   costly OOS). This plan's motivation is serve/backtest configuration
   parity, a self-maintaining universe (no more manual quarterly-review
   debt), and a wider funnel feeding the SAME unchanged admission gate and
   arbitrator — NOT a fix for June specifically.
5. **User decision already made:** expand serve's universe using the
   engine's own dynamic top-N trailing-ADV definition (mirrors
   `_apply_liquidity_filter`), NOT literal VNX50 index membership (static
   lists rot exactly as VN30's did — see #2).
6. **Sibling plan boundary (must not cross).** The 04-07-26 admission A/B's
   verdict explicitly says: "Serve's current defensiveness... is not
   measurably costly on average... no serve-path change is warranted from
   this evidence alone" — referring to the absolute meta-gate/admission
   RULE. This plan does not touch that rule, `_ARBITRATOR_POOL=6`, dispatch
   top-3, or tranche weight math. It only widens the pool of tickers that
   are ELIGIBLE to compete for that gate and that top-3 slot.

## Goal

Replace `_select_candidates`'s static `_VN30_UNIVERSE` frozenset filter with
a dynamic top-N trailing-ADV universe builder that mirrors the backtest
engine's `_apply_liquidity_filter` definition (N=50, window=20 trading days,
`adv = mean(close × volume, window)` ranked within the current snapshot),
config-gated with a safety fallback to the existing static frozenset.

## Design decisions (settled — no ambiguity left for EXECUTE)

### 1. Where ADV is computed

**Decision: new pure module, computed from the ALREADY-LOADED live panel —
no new DB query.**

`daily_inference` (`main.py:1017-1023`) already loads the exact frame
needed before `_select_candidates` is ever called:

```
live_pl = generator.load_live_ohlcv_window(window_rows=window_rows)   # Polars, ticker/date/open/high/low/close/volume
latest_df = live_pl.to_pandas()
```

`window_rows` defaults to 120 (`daily_inference(window_rows: int = 120, ...)`),
comfortably exceeding the 20-day ADV window with margin for tickers that
have short histories.

New module: `src/trading/serve_universe.py` (mirrors the existing
`src/trading/regime_policy.py` / `src/trading/risk_tier.py` precedent of
one small pure-function module per serve-path policy concern). Polars-native
per repo convention.

```
def liquid_universe(
    ohlcv_pl: pl.DataFrame,     # ticker, date, open, high, low, close, volume
    top_n: int = 50,
    adv_window: int = 20,
    min_valid_names: int = 5,
) -> frozenset[str]:
```

Semantics:
- Group by `ticker`, sort by `date`, compute
  `dvol = close * volume`, then `adv = dvol.rolling(adv_window).mean()`
  over each ticker's OWN trailing window (Polars `.rolling_mean()` /
  `over("ticker")` window function — no cross-ticker leakage).
- Take EACH ticker's LATEST available `adv` value (the last row of its
  per-ticker series in the input panel — since `latest_df`'s last row per
  ticker IS the most recently completed trading session, per
  `_compute_v3_features`'s existing "predictions are the latest decision
  bar" contract already used for the model features).
- **No `.shift(1)` here — documented deliberately, not an oversight.** The
  backtest's `shift(1)` exists because `_prepare` walks a HISTORICAL
  calendar day-by-day and must not let day D's own volume leak into day D's
  liquidity gate. Serve has no such concern: it is computing ONE live
  snapshot as of the most recent completed session, not iterating forward
  through history. Using that session's own volume in its own trailing mean
  is standard current-ADV, not a leak. State this explicitly in the module
  docstring so a future reader does not "fix" it into a shift and silently
  lag the live universe by one day.
- Rank tickers by `adv` descending; keep the top `top_n`. Ties broken by
  ticker name (deterministic, matches `rank(method="first")` intent from
  the engine without needing pandas' exact tie-break — determinism, not
  exact algorithmic parity, is the requirement here).
- If fewer than `min_valid_names` tickers have a non-null `adv` (insufficient
  history — e.g. thin `window_rows` or many delisted/new tickers), return an
  **empty frozenset** so the caller's degrade path (see #3) takes over. Do
  not silently fall back to "all tickers" inside this function — that
  decision belongs to the caller, which also has to decide between "all
  predictions" and "the static VN30 set" as the safety net.
- Function must never raise on ordinary bad input (missing columns → raise
  ValueError early and loudly, since that indicates a real upstream schema
  break, not a data-sparsity degrade case — same distinction the engine's
  own `_prepare` makes for `panel missing columns`).

### 2. Config surface

New `TradingConfig` fields in `config/settings.py` (`@dataclass TradingConfig`,
alongside `regime_sizing_enabled` / `garch_brake_enabled` precedent style —
docstring comment above each field, settings.json kill-switch):

```python
serve_universe_mode: str = "adv_top_n"     # "adv_top_n" | "vn30"
serve_liquid_top_n: int = 50
serve_adv_window: int = 20
```

- `serve_universe_mode="adv_top_n"` (**default ON**): use
  `serve_universe.liquid_universe(...)` with `serve_liquid_top_n` /
  `serve_adv_window`.
- `serve_universe_mode="vn30"`: use the existing static `_VN30_UNIVERSE`
  frozenset unchanged (kill-switch value — also the degrade target, see #3).
- Any other string value → treat as invalid config, log a warning, and use
  `"vn30"` (fail toward the OLD, already-proven-safe behavior, not toward an
  unvalidated new one — same fail-safe direction as `garch_brake_enabled`'s
  "any failure → full exposure" precedent, adapted to "any config failure →
  old universe").

`config/settings.json` gets the three new keys added to the existing
`"trading"` block (do NOT resurrect the dead `"universe_filter"` block at
the bottom of settings.json — that block is confirmed dead config per
`settings.py:118-119`'s own comment and must be left alone or removed in a
SEPARATE cleanup, not conflated with this change).

**Default-on rationale (flagged for explicit user review — see Open
Questions):** the evidence for the ADV-top-50 universe IS the entire
existing validated backtest (GOLDEN T+20, the 04-07 admission A/B, all of
it ran under `liquid_top_n=50`). Serve currently runs a DIFFERENT, smaller,
staler universe than what was validated — shipping default-OFF would mean
serve keeps running on effectively unvalidated-by-omission territory
indefinitely unless someone remembers to flip the switch. This mirrors the
`regime_sizing_enabled` precedent (default ON, settings.json kill-switch)
more than the `use_nav_tier_cap` / admission-A/B precedent (default OFF,
because THOSE were genuinely new/unvalidated mechanics being tested). This
change is not new mechanics — it is aligning serve to an already-validated
existing mechanic. That said, this is a real production behavior change
(more tickers become tradeable) and the user should explicitly confirm
default-ON before EXECUTE, not just accept it silently.

### 3. Degrade path

`_select_candidates` gains a new parameter (or the universe is resolved by
its caller and passed in — see Touchpoints) that replaces the single
hardcoded `vn30_universe: frozenset[str]` argument with a resolved
"candidate universe" computed BEFORE the call, using this precedence:

1. `serve_universe_mode == "adv_top_n"` → call `liquid_universe(...)`.
   - If it returns a NON-empty frozenset → use it.
   - If it returns EMPTY (insufficient ADV history) → log a WARNING
     (`[UniverseGate] ADV universe empty (insufficient history) — falling
     back to static VN30 list.`) and fall back to `_VN30_UNIVERSE`.
   - If `liquid_universe(...)` itself raises (schema break, unexpected
     exception) → catch at the call site, log an ERROR with the exception,
     fall back to `_VN30_UNIVERSE`. Serve must never crash `daily_inference`
     because of this new code path — matches repo's degrade-not-crash
     culture (GARCH brake fail-open precedent, secondary-horizon
     `FileNotFoundError`/`RuntimeError` catch precedent at
     `main.py:1043-1049`).
2. `serve_universe_mode == "vn30"` → use `_VN30_UNIVERSE` directly (no ADV
   computation attempted at all).
3. Any other/invalid mode string → same as `"vn30"`, plus the invalid-value
   warning from #2 above.

`_VN30_UNIVERSE` frozenset stays in `main.py` verbatim — it is the
permanent safety net, not deleted or deprecated. `_select_candidates`'s
existing internal "no predicted ticker in the universe → use all
predictions as fallback" branch (`main.py:881-883`) is UNCHANGED and now
sits one layer further inside (after the universe-resolution degrade above
already ran).

### 4. Sentiment coverage (`CONFIG.sentiment.max_tickers`)

**Decision: leave `max_tickers=30` unchanged. Do not bump it.**

Justification (verified this session, not assumed):
- `max_tickers=30` governs `SentimentCrawler._active_tickers` /
  `update_daily_sentiment` (`src/crawlers/sentiment_crawler.py:159-227`),
  which runs as an EARLIER, SEPARATE step in `full_pipeline`
  (`main.py:1945-1979`, before `daily_inference` is called), pre-crawling
  RSS/news into a persisted sentiment table by `top_tickers_by_volume`
  (raw SUM(volume) ranking over the DuckDB parquet glob — a DIFFERENT
  definition than the new `adv_top_n` $-ADV ranking; not something this
  plan unifies, since it is out of scope).
- `evaluate_trades_batch` (`src/models/quant_agent_arbitrator.py:970-1050`)
  is what actually consumes sentiment for the (now possibly-wider-sourced,
  but still capped at 6) candidate pool. When a candidate ticker has NO
  pre-crawled sentiment row, it already degrades gracefully: neutral
  `sentiment_score=0.0`, a small activity penalty (`stacking_probs[2] *=
  0.95`), and a Vietnamese "no significant news" reasoning string — it does
  NOT block, skip, or crash on missing coverage (`quant_agent_arbitrator.py:
  1042-1050`).
- The candidate pool arbitration cap (`_ARBITRATOR_POOL = 6`,
  `main.py:874`) is unchanged by this plan — Gemini call volume per run is
  identical regardless of universe size, since at most 6 tickers are ever
  evaluated by the arbitrator on any given day.
- Net effect of leaving `max_tickers=30` alone: a wider pool of 50
  ADV-eligible names competes for the top-6 arbitrator slots by P(UP); if a
  winning name happens to lack pre-crawled news, it gets a neutral score
  instead of a a richer sentiment read — already-existing, already-tested
  behavior, not a new failure mode this plan introduces.

If this degrade is later judged insufficient (e.g. observed in production
that ADV-top-50 names are frequently sentiment-blind), bumping
`max_tickers` to 50 is a trivial one-line follow-up — not bundled here to
keep this plan's blast radius to exactly the universe filter.

### 5. What does NOT change (explicit non-goals)

- The absolute meta-gate admission rule (`meta_gate.get(ticker, True)`
  filter inside `_select_candidates`) — sibling A/B verdict: keep as-is.
- `_ARBITRATOR_POOL = 6`, dispatch top-3 (`main.py:1120` slice), tranche
  cohort weight math (`1/(hold_days×n_picks)`).
- `sentiment_entry_paperlog` — already logs the full cross-section
  regardless of universe size; no schema or write-path change.
- `FEATURE_RECIPE_VERSION` / any retrain — this is a candidate-pool filter
  widening applied AFTER predictions are already computed on the full
  355-ish-ticker panel; no feature engineering changes, no model artifact
  changes.
- `models/saved/*.joblib` — untouched.
- `src/backtest/walk_forward.py` / `run_backtest.py` — the backtest engine's
  OWN `_apply_liquidity_filter` is the thing being MIRRORED, not touched.
  Zero changes to backtest code in this plan.
- `CONFIG.sentiment.max_tickers` — explicitly left at 30 (#4 above).

### 6. Naming / observability (production serve change — must be loud)

Enumerate every place the OLD static-VN30 framing leaks into logs/text and
update or generalize it:

| Location | Current text | Change |
|---|---|---|
| `main.py:855-856` (comment above frozenset) | "VN30 constituents — STRICT live universe gate... Update on the quarterly VN30 review." | Keep frozenset + comment as-is (it is now clearly labeled as the FALLBACK/kill-switch universe, not the primary one) — add one sentence noting it is now the `serve_universe_mode="vn30"` fallback target. |
| `main.py:879` `[VN30Gate]` log line | `"[VN30Gate] %s / %s predicted tickers are in VN30."` | Generalize to `[UniverseGate]` and log the ACTIVE mode + resolved universe size, e.g. `"[UniverseGate] mode=%s universe_size=%d | %s / %s predicted tickers in universe."` |
| `main.py:882` warning | `"[VN30Gate] no predicted ticker in VN30..."` | Generalize wording to reference "the resolved universe" not literally "VN30" (still fires correctly under `vn30` mode too). |
| `main.py:1078-1079` fallback report text | `"...vũ trụ giao dịch sau bộ lọc thanh khoản/Universe hôm nay."` | Already generic ("universe" in Vietnamese, not VN30-specific) — leave unchanged, confirm no edit needed. |
| NEW log line (add) | n/a | On every `daily_inference` run, log the resolved mode, universe size, and (at INFO or DEBUG) the top ~10 members by ADV, e.g. `"[UniverseGate] resolved mode=adv_top_n size=50 top10=[...]"` — this is a production serve change; a human must be able to see from the log alone which universe was used and why, without reading code. |

### 7. Tests

- **Update** `tests/test_select_candidates.py` — it imports `_VN30_UNIVERSE`
  directly and calls `_select_candidates(preds, gate, _VN30, N)` with the
  frozenset as a positional arg. If `_select_candidates`'s signature changes
  (see Touchpoints — decision: keep `_select_candidates` itself
  universe-agnostic, i.e. it still just takes a `candidate_universe:
  frozenset[str]` param; universe RESOLUTION happens one layer up in
  `daily_inference` before the call). Under that design, `_VN30_UNIVERSE`
  keeps its meaning as "a specific frozenset the function is a pure filter
  over" and **none of the 11 existing tests need behavioral changes** — they
  keep passing a frozenset positionally, exactly as today. Import stays
  valid since the frozenset itself is not deleted. State this explicitly so
  EXECUTE does not over-engineer a signature change that isn't needed.
- **NEW** `tests/test_serve_universe.py` for `src/trading/serve_universe.py`:
  - `test_top_n_correctness` — synthetic 10-ticker panel, hand-computed ADV
    ranks, assert the exact top-N set returned for `top_n < 10`.
  - `test_adv_window_respects_trailing_window` — a ticker with a volume
    spike OUTSIDE the trailing window must not affect its ranked ADV (only
    the last `adv_window` rows count).
  - `test_no_shift_uses_latest_session` — mutating the LAST row's volume for
    a ticker changes that ticker's computed ADV / rank (proves the
    deliberate no-`shift(1)` design from decision #1 — the opposite
    assertion from what a leak-check test would assert on the backtest
    side; docstring must explain why this is correct here, referencing the
    live-single-snapshot rationale, not a copy-paste of the backtest's
    leak-check test).
  - `test_insufficient_history_returns_empty` — fewer than `min_valid_names`
    tickers with valid ADV → returns `frozenset()`, not a partial set and
    not an exception.
  - `test_missing_columns_raises` — malformed input (missing `volume` etc.)
    raises `ValueError` (loud schema break, not a silent degrade).
  - `test_ticker_isolation` — one ticker's ADV computation must be
    unaffected by another ticker's volume/price values (no cross-ticker
    leakage through a mis-grouped rolling window).
- **NEW** (small, in `tests/test_select_candidates.py` or a new file —
  EXECUTE's call which) covering the degrade wiring at the
  `daily_inference`-adjacent layer:
  - `test_universe_mode_vn30_bypasses_adv_computation` — mode="vn30" never
    calls `liquid_universe`.
  - `test_universe_mode_adv_degrades_to_vn30_on_empty` — `liquid_universe`
    returns empty → resolved universe falls back to `_VN30_UNIVERSE`.
  - `test_universe_mode_adv_degrades_to_vn30_on_exception` — `liquid_universe`
    raises → caught, falls back to `_VN30_UNIVERSE`, no crash.
  - `test_invalid_mode_string_falls_back_to_vn30` — e.g.
    `serve_universe_mode="bogus"` → behaves like `"vn30"` plus a warning.

Runner: bare pytest (`pytest -q`), per
`process/context/tests/all-tests.md` — no conda-stock env, does not touch
the live DuckDB lock (`run_bot.py` may keep running throughout).

## Touchpoints

| File | Change |
|---|---|
| `src/trading/serve_universe.py` | **NEW.** `liquid_universe(ohlcv_pl: pl.DataFrame, top_n: int = 50, adv_window: int = 20, min_valid_names: int = 5) -> frozenset[str]`. Polars-native. Pure function, no I/O, no config import (caller passes resolved params) — matches `regime_policy.py`/`risk_tier.py` precedent of policy modules being pure and config-agnostic. |
| `config/settings.py` | `TradingConfig`: add `serve_universe_mode: str = "adv_top_n"`, `serve_liquid_top_n: int = 50`, `serve_adv_window: int = 20`, each with a docstring comment in the existing style (kill-switch note, default rationale, fail-safe direction). |
| `config/settings.json` | Add the three new keys under the existing `"trading"` block. Do NOT touch the dead `"universe_filter"` block. |
| `main.py` | (a) Keep `_VN30_UNIVERSE` frozenset verbatim, add one clarifying sentence to its comment (now documented as the fallback/kill-switch universe). (b) New small helper (e.g. `_resolve_candidate_universe(latest_df_pl_or_pd, cfg) -> frozenset[str]`) implementing the precedence/degrade logic from Design Decision #3, calling `serve_universe.liquid_universe(...)` when mode is `adv_top_n`. (c) `daily_inference` calls this new resolver ONCE (using the already-loaded `live_pl`/`latest_df`) and passes the RESULT into the existing `_select_candidates(stacking_predictions_5d, meta_gate_5d, <resolved_universe>, max_candidates)` call at line 1065-1067 — `_select_candidates`'s own signature and internals are UNCHANGED. (d) Update the `[VN30Gate]` log lines to `[UniverseGate]` + mode/size per Design Decision #6, plus one new resolved-universe summary log line. |
| `tests/test_select_candidates.py` | No behavioral changes required (see Design Decision #7) — confirm all 11 existing tests still pass unmodified after the `main.py` changes (they exercise `_select_candidates` directly with a frozenset, which still works identically). |
| `tests/test_serve_universe.py` | **NEW** — 6 tests per Design Decision #7. |
| `tests/test_universe_resolution.py` (or appended to `test_select_candidates.py`) | **NEW** — 4 degrade-path tests per Design Decision #7. |

## Guardrails

1. **No feature-recipe change, no retrain.** `pipeline.py`,
   `FEATURE_RECIPE_VERSION`, `models/saved/*.joblib` untouched.
2. **`_select_candidates`'s own signature and internal logic are UNCHANGED.**
   It remains a pure filter over whatever frozenset it is given — universe
   RESOLUTION is a new, separate concern living in a new helper +
   `serve_universe.py`, not folded into `_select_candidates` itself. This
   keeps the 11 existing tests stable and the blast radius contained.
3. **Never crash `daily_inference` because of this change.** Every new
   failure mode (empty ADV set, exception inside `liquid_universe`, invalid
   config string) must degrade to the existing `_VN30_UNIVERSE` fallback
   with a logged warning/error, never propagate an unhandled exception.
4. **This is a live production serve-path change with real capital-allocation
   consequences** (more tickers become tradeable candidates on any given
   day). Do not let it reach the cron path (`full_pipeline` at 15:30 ICT)
   without the staged verification below completing first.
5. **`run_bot.py` is currently running and holds the DuckDB file lock.**
   Nothing in this plan's implementation needs the live DB — `serve_universe.py`
   is a pure function over an in-memory Polars frame already loaded by
   `daily_inference`, and pytest uses in-memory DuckDB stubs. EXECUTE should
   not need to stop `run_bot.py` for implementation or unit testing. Manual
   smoke runs (see Verification) that call `daily_inference(persist=False,
   broadcast=False)` should also avoid DB writes by construction of those
   kwargs — confirm this at EXECUTE time by reading what `persist=False`
   actually gates before running the smoke test live.
6. **Do not touch the dead `universe_filter` settings.json block** in this
   plan — it is confirmed-dead config (per `settings.py:118-119`'s own
   comment); resurrecting or removing it is a separate, unrelated cleanup.
7. **Working tree must be clean before EXECUTE starts** (confirmed clean at
   `91fe7da` as of this plan's writing) — this plan should be the only new
   artifact until EXECUTE begins.

## Verification evidence checklist

- [ ] `pytest -q tests/test_select_candidates.py` — all 11 existing tests
      still green, unmodified (proves the pure-filter function truly didn't
      need a signature change).
- [ ] `pytest -q tests/test_serve_universe.py` — all 6 new tests green
      (top-N correctness, trailing-window respect, no-shift/latest-session
      behavior, insufficient-history empty-set, missing-column raise,
      ticker isolation).
- [ ] `pytest -q tests/test_universe_resolution.py` (or wherever the 4
      degrade tests land) — vn30-mode bypass, empty-set degrade,
      exception degrade, invalid-mode degrade all green.
- [ ] `pytest -q` — full suite green (confirm exact count at EXECUTE time;
      baseline noted elsewhere in-flight as ~481-526 depending on which
      sibling plans have landed — get the live number, don't assume).
- [ ] **Side-by-side manual smoke run (required before any live cron
      pickup):** run `daily_inference(broadcast=False, persist=False)` (or
      the dashboard/bot on-demand equivalent) TWICE against the same day's
      data — once with `serve_universe_mode="vn30"` (old behavior) and once
      with `serve_universe_mode="adv_top_n"` (new default) — and diff the
      resulting `candidate_tickers` lists side by side. Confirm: (a) the
      new run's candidate pool is a superset-or-different-composition, not
      identical (proves the change is actually taking effect); (b) no
      exception in either run; (c) the new `[UniverseGate]` log line
      appears with the expected mode + size in both runs.
- [ ] Only AFTER the smoke run passes and the user has reviewed its output:
      the config default (`serve_universe_mode="adv_top_n"`) is allowed to
      reach the next scheduled cron `full_pipeline` run. If the smoke run
      surfaces anything surprising, keep the default at `"vn30"` in
      `settings.json` until resolved (kill-switch exists precisely for
      this).

## Deferred / follow-ups

- Unifying `price_lookup.top_tickers_by_volume`'s SUM(volume) ranking with
  the new $-ADV definition — they now measure liquidity two different ways
  in two different subsystems (sentiment pre-crawl vs. serve universe);
  not a contradiction that needs resolving now, but worth a future note if
  it ever causes confusion.
- Bumping `CONFIG.sentiment.max_tickers` above 30 if production observation
  shows ADV-top-50 names are frequently sentiment-blind (see Design
  Decision #4) — trivial one-line follow-up, not bundled here.
- A quarterly/periodic audit of how often the ADV-top-50 set drifts
  relative to VN30 membership, to build confidence the dynamic universe is
  behaving as expected in production — not required for this plan's
  verdict, but a natural companion to `/audit_accuracy`-style observability
  already in the repo.
- Task B from the sibling admission A/B plan (absolute-gate rule change) —
  explicitly NOT this plan, tracked separately, gated on its own evidence.

## Resume and execution handoff

- **Primary plan file to execute:** THIS file
  (`process/general-plans/active/serve-universe-adv-alignment_PLAN_05-07-26.md`).
- **No precondition commit needed** — working tree confirmed clean at
  `91fe7da`. `main.py`, `config/settings.py`, `config/settings.json` are
  NOT currently dirty from any sibling in-flight plan (unlike the admission
  A/B plan's precondition situation) — safe to start directly.
- **Sibling plans to be aware of (do not merge scope):**
  - `serve-admission-tranche-ab_PLAN_04-07-26.md` (COMPLETE, verdict:
    REJECTED changing the admission rule) — this plan must not re-open
    that question; it only widens the pool feeding the SAME unchanged gate.
  - `notification-attribution-risk-tier_PLAN_02-07-26.md` (CODE-COMPLETE,
    own A/B pending) — independent NAV-cap concern, no file overlap with
    this plan's touchpoints.
- **Execution order inside this plan:**
  1. `src/trading/serve_universe.py` — write `liquid_universe(...)` +ust its
     docstring explaining the deliberate no-`shift(1)` design.
  2. `tests/test_serve_universe.py` — write and run the 6 tests BEFORE
     wiring it into `main.py` (prove the builder is correct in isolation).
  3. `config/settings.py` — add the three `TradingConfig` fields.
  4. `config/settings.json` — add the three keys to the `"trading"` block.
  5. `main.py` — add `_resolve_candidate_universe(...)` helper, wire it into
     `daily_inference` before the existing `_select_candidates(...)` call,
     update `[VN30Gate]` → `[UniverseGate]` log lines.
  6. `tests/test_universe_resolution.py` (or equivalent) — write and run
     the 4 degrade-path tests.
  7. `pytest -q tests/test_select_candidates.py` — confirm zero regressions
     on the untouched existing tests.
  8. `pytest -q` — full suite green.
  9. Manual side-by-side smoke run (`vn30` vs `adv_top_n`) per Verification
     checklist — review output with the user before considering this
     production-ready.
- **Open questions needing user decision (surfaced now, not deferred
  silently):**
  1. **Default ON vs OFF.** This plan documents `serve_universe_mode:
     str = "adv_top_n"` as the DEFAULT (ships ON), reasoning that the
     validated backtest evidence already assumes this universe and serve
     has been quietly running on a smaller/staler one. Confirm this is the
     intended risk posture before EXECUTE — if the user prefers a more
     cautious rollout, flip the default to `"vn30"` in the plan (one-line
     change) and treat this as an opt-in feature for now, promoting to
     default-on after the smoke run + a few days of live observation.
  2. **`serve_liquid_top_n=50` / `serve_adv_window=20` exact values** — this
     plan copies the backtest's exact validated numbers
     (`WalkForwardConfig.liquid_top_n=50`, `adv_window=20`). Confirm no
     different value is preferred for serve specifically (e.g. a tighter
     top-30 for a more conservative live rollout) before EXECUTE.
  3. **`CONFIG.sentiment.max_tickers` left at 30** — confirm this is
     acceptable given the degrade-gracefully behavior described in Design
     Decision #4, rather than bumping it defensively to 50 alongside this
     change.
  4. **Tie-break rule in `liquid_universe`** (ticker-name alphabetical, for
     determinism) is NOT algorithmically identical to the backtest's
     `Series.rank(method="first")` (insertion-order-based) tie-break. This
     is deliberate (serve doesn't need byte-identical tie-breaking with the
     backtest, only the same TOP-N SET most days when ties are rare) but
     flagging it explicitly in case exact parity is actually wanted.
