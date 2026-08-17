# Declared Corporate-Action Event Detection (vnstock `Company.events()`)

- **Status:** ACTIVE — awaiting `ENTER EXECUTE MODE`
- **Shape:** COMPLEX
- **Created:** 13-08-26
- **Owner plan path:** `process/general-plans/active/vnstock-ca-event-detection_PLAN_13-08-26.md`
- **Predecessor (completed):** `process/general-plans/completed/auto-ca-price-adjustment_PLAN_17-07-26.md`
- **Related active plan:** `process/general-plans/active/portfolio-guard_PLAN_13-07-26.md`

---

## 1. Problem

On 2026-08-13 the EOD portfolio guard fired a **hard-stop + trailing-stop** alert for the live
`/add` holding VHM (user `1818282405`, 1000 shares @ `152.5` entered 2026-07-09):

```
RAW : entry=152,500  today=71,800  peak=153,000  pnl=-52.92%  dd=53.07%
```

VHM paid a **100% stock dividend** with `exright_date` 2026-08-06 — the share count doubled and
the quoted price halved. The user lost ~6%, not ~53%.

The existing shield behaved correctly but incompletely. `price_lookup.has_ca_gap` detected the
`153,000 → 77,100` step and `portfolio_guard.evaluate_position` downgraded the wording to
"có thể do hành động doanh nghiệp … KHÔNG chắc là lỗ thật". It did **not** compute the corrected
number, because `derive_ca_adjustment_factor` — shipped 17-07-26 and already consumed by
`signal_ledger._evaluate_pnl` — was explicitly excluded from `portfolio_guard.py` by that plan
("read-only precedent reference only, zero edits").

So the alert still **fired at all**, when with correction neither threshold is breached:

| basis | factor | entry_eff | PnL | rebased peak | drawdown | hard stop (−7%) | trailing (8%) |
|---|---|---|---|---|---|---|---|
| raw | — | 152,500 | **−52.92%** | 153,000 | **53.07%** | FIRES | FIRES |
| self-referential | 0.503922 | 76,848 | −6.57% | 77,100 | 6.87% | no | no |
| **declared (1/(1+1.0))** | **0.500000** | **76,250** | **−5.84%** | **77,100** | **6.87%** | **no** | **no** |

*(All figures reproduced live from `data/ohlcv_VHM.parquet` + the live `portfolio` table on
13-08-26; the raw row matches the alert the user received to 0.01pp.)*

### Why declared data, and not just wiring in the existing self-referential factor

The self-referential factor is derived purely from the price ratio. That is cheap and needs no
network, but it is structurally blind in four ways, all measured on this repo's own data:

1. **It cannot see sub-threshold events.** `has_ca_gap` triggers at >10%. A stock dividend of
   ≤11.1% produces a gap of ≥0.9 and is invisible. HPG's declared 10% stock dividend
   (`exright_date` 2026-05-25) is exactly this shape.
2. **It cannot distinguish a corporate action from a genuine limit move.** A scan of every shard
   since 2026-01-01 for moves in the 7–10% band returns a long list dominated by **exact ±7.00%
   limit sessions** (PIT `7.00→7.49`, GVR `30.00→32.10`, MCH `140.00→149.80`, …). Any attempt to
   lower the threshold to catch (1) starts "correcting" real losses into fake small ones. Declared
   data breaks the tie; price alone cannot.
3. **It absorbs genuine ex-date price movement into the "correction".** Observed 0.503922 vs
   declared 0.500000 — the 0.78% difference is real trading, and the self-referential factor
   silently deletes it from the PnL.
4. **It cannot name the event.** The user gets "might be a corporate action, go check yourself"
   instead of "VHM trả cổ tức bằng cổ phiếu 100%, ngày GDKHQ 06/08/2026".

The problem also is not rare or shrinking. **47 raw >10% single-session gaps exist across the
shards since 2026-01-01** (MBB 2026-08-11 `0.8392`, VHM 2026-08-06 `0.5039`, TRA 2026-07-17
`0.4765`, PVD 2026-07-10 `0.5847`, PET 2026-07-03 `0.6697`, …).

---

## 2. RESEARCH — live API findings

All findings below were produced by running the real API on 13-08-26, not from documentation.

### 2.1 Call surface

```python
from vnstock import Company
Company(source='VCI', symbol='VHM').events()   # -> pandas DataFrame, 22 columns
```

- **`source='VCI'` is the only usable source.** `source='KBS'` returns an empty `(0, 0)` DataFrame
  for both VHM and HPG. There is **no fallback source** — plan accordingly.
- An unknown symbol raises `ValueError: Invalid symbol. Your symbol format is not recognized!`.
- An **empty** symbol string returns 50 rows of garbage → the ticker must be validated before the
  call (reuse `price_lookup._TICKER_RE`'s `[A-Z0-9]{1,12}` shape).
- Every response is capped at **50 rows** (VCB/HPG/SSI/MBB/… all return exactly 50; VHM 41, KLB 36).
  This is a page cap, so only the ~50 most recent events per ticker are reachable. Acceptable —
  guard windows are weeks, not years — but must be documented.
- Latency measured across 15 tickers: **0.37–1.09s** (one 3.13s cold start). vnstock already wraps
  the call in its own tenacity retry.
- All date columns come back as **`str`** in `2026-08-06T00:00:00` form. `exercise_ratio` and
  `value_per_share` are `float64`.

### 2.2 `exercise_ratio` is NOT uniform across event types — confirmed

The VHM case does **not** generalize. Measured across 15 tickers / 727 events:

| `event_code` | meaning of `exercise_ratio` | example |
|---|---|---|
| `ISS` | new shares per existing share | VHM `1.0` = 100% stock dividend |
| `DIV` | **cash per share ÷ 10,000 (par value)** | VCB `0.045` with `value_per_share=450.0` |

Par-value normalization confirmed on every `DIV` row with a populated `value_per_share`:
`450/0.045`, `6000/0.6`, `1000/0.1`, `500/0.05` — all exactly `10000`.

Within `ISS`, the `event_title_vi` prefix carries the real taxonomy, and only **two** sub-types are
proportional entitlements to existing holders with a computable price factor:

| `event_title_vi` contains | kind | price-adjusting? | factor |
|---|---|---|---|
| `Trả Cổ tức bằng Cổ phiếu` | stock dividend | **YES** | `1/(1+r)` |
| `Cổ phiếu thưởng` | bonus shares | **YES** | `1/(1+r)` |
| `Quyền mua CP cho Cổ đông hiện hữu` | rights issue | yes, but **UNPRICEABLE** — needs the subscription price, which is **not in the payload** (`value_per_share` is NaN on every `ISS` row) | — |
| `Phát hành riêng lẻ` | private placement | no (third party, no holder entitlement) | — |
| `Phát hành cho CBCNV` | ESOP | no | — |
| `Chuyển từ trái phiếu chuyển đổi` | convertible conversion | no | — |
| `Phát hành để sáp nhập` | merger issuance | no | — |
| `DIV` (any) | cash dividend | different math (income, not a scale change) — **out of scope** | — |

Non-`ISS`/`DIV` codes (`DDIND`/`DDINS`/`DDRP` insider trading, `AGME`/`EGME` meetings, `AIS`
listing changes, `SUSP`, `NLIS`, `MOVE`, `OTHE`) are never price-adjusting. `DDIND`/`DDINS`/`DDRP`
alone are 451 of the 727 rows — the table is mostly noise for this purpose.

Also observed: `exercise_ratio == 0.0` appears on many older `ISS` rows (MSN, VCB, VND, SHB) —
a "declared but ratio unknown" state that must be treated as unpriceable, not as factor `1/(1+0)=1`.

### 2.3 Declared ratios match observed gaps — but dates do NOT align

26 of the 45 raw >10% gaps since 2026-04-01 were matched against live `events()` data:

```
TICK  GAPDATE      ratio | declared                          exright      theor    err%   dOFF
MBB   2026-08-11  0.8392 | stock div 15% (+ rights 10%)      2026-08-11  0.8696   -3.49     +0
VHM   2026-08-06  0.5039 | stock div 100%                    2026-08-06  0.5000   +0.78     +0
KLB   2026-07-02  0.7722 | bonus 29.5%                       2026-07-03  0.7722   +0.00     -1
VTP   2026-07-10  0.8521 | stock div 17.4%                   2026-07-14  0.8521   -0.00     -4
PVD   2026-07-10  0.5847 | bonus 66.9%                       2026-07-14  0.5992   -2.42     -4
PET   2026-07-03  0.6697 | bonus 40.0%                       2026-07-09  0.7143   -6.24     -6
AAN   2026-06-24  0.7043 | bonus 23.0%                       2026-06-29  0.8130  -13.37     -5
… 5 of 26 had no priceable declared event within ±20 days (VDP, HCM, VVS, TYA, EVE)
```

Two decisive calibration facts:

- **`dOFF` (gap date − declared `exright_date`) ranges `-6 … 0` calendar days and is never
  positive.** The parquet gap lands on or *before* the declared ex-right date. Matching on exact
  date equality would fail on ~70% of real cases.
- **Factor agreement is tight where it matches at all**: median `|err|` ≈ 3%, worst-case 13.37%
  (AAN). Several of the larger errors (VCG −6.11%, BAF −8.20%, PET −6.24%) are consistent with a
  *second* entitlement bundled on the same ex-date that the single-event match ignored.

→ **Design consequence: match on FACTOR AGREEMENT within a date window, not on date equality.**
The measurement inverts the intuitive design.

### 2.4 The parquet is *sometimes* already back-adjusted — double-adjustment is a live hazard

Probed against the shards:

| ticker | declared event | observed gap at ex-date | state |
|---|---|---|---|
| VHM | stock div 100%, 2026-08-06 | `0.5039` | **raw** |
| MBB | stock div 15% + rights 10%, 2026-08-11 | `0.8392` | **raw** |
| VCB | stock div 49.5%, 2025-03-12 | `1.0302` | already adjusted |
| KLB | stock div 60%, 2025-09-24 | `1.0130` | already adjusted |
| MBB | stock div 32%, 2025-08-13 | `1.0609` | already adjusted |
| HPG | stock div 20%, 2025-06-26 | `1.0058` | already adjusted |

Recent events sit raw; older ones have been restated. Applying a declared factor unconditionally
would **corrupt** every already-adjusted series. The correction must therefore be gated on an
**observed** gap, never on the declaration alone.

### 2.5 Repo conventions this must follow

- Crawler/never-raise/kill-switch shape: `src/data/market_breadth_crawler.py`,
  `src/data/fastconnect_ohlcv.py`, `src/data/crawlers.py`.
- DuckDB table + `CREATE TABLE IF NOT EXISTS` + dedup-before-insert:
  `src/crawlers/sentiment_crawler.py::_append_rows`, `src/trading/signal_ledger.py::ensure_table`.
- Short-lived `duckdb.connect(...)` per operation, degrade to `[]`/`0` on failure:
  `signal_ledger.list_open`, `portfolio_guard.load_guard_positions`.
- Pure/IO layering inside one module: `portfolio_guard.py`, `intraday_scanner.py`.
- Per-leg fail-open-to-identity: `src/bot/garch_brake.py`.
- Test style: pure functions tested directly (`tests/test_price_lookup_ca_adjustment.py`);
  I/O tested against `tmp_path` DuckDB; heavy serve stack monkeypatched
  (`tests/test_portfolio_guard.py::_patch_serve`). **Zero real network in tests.**

---

## 3. INNOVATE — decisions

### Decision 1 — Where the network call lives: a cached crawler, NOT the alert path

**Chosen:** a `refresh` function called from `main.py`'s EOD path, writing a DuckDB cache; the
guard reads only the cache.

Rejected: calling `Company.events()` inside `portfolio_guard.evaluate_position`. That function is
called once per lot per EOD run and is documented **PURE** — its module docstring states "This
module itself makes NO network / LLM call." vnstock guest throttling is already a documented
bottleneck in this repo (the ~20 req/min cap is what forced the entire FastConnect prefetch
rewrite). Putting a network call in a per-lot loop would violate both the purity contract and the
throttle discipline.

Rejected: refreshing the full ~359-ticker universe. CA lookups only matter for names someone
actually holds.

**Scope:** `non-cron portfolio tickers ∪ open dispatched_signals tickers`. Today that is
**2 tickers** (`HDB`, `VHM`; zero open real signals). Bounded by
`ca_event_max_refresh_per_run` (default 20).

**Cadence:** refresh a ticker only when its cache row is missing or older than
`ca_event_refresh_days` (default **7**). Justified by the data: VHM's event had
`public_date` 2026-07-24 vs `exright_date` 2026-08-06 — **13 days of notice**. A 7-day refresh is
comfortably ahead of any event.

### Decision 2 — Four-tier resolution, gated on an OBSERVED gap

| tier | condition | factor used | wording |
|---|---|---|---|
| **A — DECLARED** | observed gap present **AND** ≥1 priceable declared event in window **AND** no unpriceable rights event in the same window **AND** `abs(observed/theoretical − 1) ≤ ca_event_factor_tolerance` | **declared** `1/(1+Σr)` | confident, names the event |
| **B — OBSERVED** | observed gap present, no confident declared match | self-referential (existing) | existing disclaimer, unchanged |
| **C — NONE** | no observed gap, no declared event | `1.0` | unchanged |
| **D — ALREADY ADJUSTED** | declared event in window but **no** observed gap | `1.0` (do nothing) | unchanged |

Tier D is the double-adjustment guard from §2.4. Tier B preserves today's behavior exactly, so
the change is strictly additive.

The **rights-contamination rule** in tier A is not theoretical — MBB 2026-08-11 bundles a 15%
stock dividend with a 10% rights issue. Stock-dividend-only theory gives `0.8696` vs observed
`0.8392`, a −3.49% error that would *pass* a 10% tolerance and be wrongly promoted to "confident"
while silently omitting the rights leg. Any unpriceable rights event in the window forces tier B.

`ca_event_factor_tolerance` default **0.10**: accepts every matched case in §2.3 except AAN
(−13.37%), which correctly falls back to tier B. Same magnitude as the existing `has_ca_gap`
threshold, so the two numbers stay conceptually paired.

Same-ex-date **bundling**: multiple priceable events sharing an `exright_date` compose as
`1/(1 + Σ rᵢ)` (GEX 2026-05-05 = bonus 20% + stock div 25%; POW 2025-12-10 = 4% + 15% + rights).

### Decision 3 — Match on factor agreement inside a date window, not on date equality

Directly forced by §2.3's `dOFF ∈ [-6, 0]`. Events are filtered to
`[window_start − 10d, window_end + 3d]` (module constants, not config — these are empirical
properties of the data source, not user preferences), then paired to observed gap ratios by
factor agreement.

**Consequence: no change to `price_lookup.py` is needed.** Gap positions come from the ordered
close list itself, exactly as `derive_ca_adjustment_factor` already does; no per-bar dates are
required. This keeps a shipped, load-bearing module untouched.

### Decision 4 — Correct the INPUTS, then evaluate triggers normally (auto-correct)

**Chosen:** on tier A, back-adjust the pre-event closes and the entry price, then run the five
existing triggers unchanged against the corrected numbers.

Rejected: keep raw numbers and merely quote the real ratio in a disclaimer. Consistent with the
17-07 precedent ("`gap_flag=True` now means *this row's pct was auto-adjusted*, NOT *unreliable,
do not show*") and with `signal_ledger._evaluate_pnl`, which already auto-corrects for `/report`.
It would be incoherent for `/report` to show VHM at ≈ −6% while the guard alarms at −52.9%.

This framing also avoids special-casing suppression: a trigger does not get "cancelled". The
input basis is fixed and the threshold comparison is re-run. **If the corrected PnL is still
≤ −7%, the alert still fires** — a genuine loss on a stock that also paid a stock dividend is
still reported. Only the phantom disappears.

Back-adjustment is applied to the **whole pre-event segment**, not just the entry, so the
trailing-stop peak is rebased too (`peak 153,000 → 77,100`). Correcting only the entry would
leave the drawdown at 53%.

**Flagged as the plan's principal decision point.** It changes whether an alert fires, which is a
safety-relevant behavior change. Mitigations: three independent conditions must agree (declared
event + observed gap + factor match within 10%), the rights-contamination veto, tier B's
conservative fallback, and a master kill-switch.

### Decision 5 — Kill-switch: yes

Unlike the 17-07 plan (which added zero new I/O and deliberately shipped no kill-switch), this
introduces a **new external network dependency**. `corporate_action_events_enabled` (default
**True**) fully short-circuits refresh *and* consumption; OFF reproduces today's behavior exactly.
Consistent with `fastconnect_ohlcv_enabled`, `portfolio_guard_enabled`, `garch_brake_enabled`.

### Decision 6 — `signal_ledger` consumption is OUT of scope for this pass

`_evaluate_pnl` already auto-corrects via the self-referential factor and is working. Wiring
declared data there would change historical reported PnL across `/report`, `/exits`, the EOD
position report and the dashboard GIỮ tab — a materially larger verification surface for a
marginal accuracy gain (0.78pp on the VHM case).

The resolver is nevertheless designed **source-agnostic and reusable** so this is a small,
self-contained follow-up. Recorded in §9.

### Decision 7 — Cash dividends (`DIV`) out of scope

A cash dividend is income, not a price-scale change; correcting it needs a different formula
(subtract `value_per_share` from the entry basis) and only matters for total-return accounting,
which this system does not do anywhere today. VHM's 6,000 VND cash dividend (2026-06-29) produced
a `0.9634` step — below the gap threshold, invisible to the guard, and not a false alarm. Not the
reported problem. Deliberately excluded to keep this pass small.

### Decision 8 — Module placement and storage

`src/data/corporate_actions.py`: reference market data derived from vnstock, sitting beside
`price_lookup.py` (which it complements) and the other `src/data/*` ingestion modules. Pure and
I/O layers are separated **inside** the one module, following `portfolio_guard.py`'s documented
precedent, rather than split across two files — this stays DRY and the module is small.

DuckDB (core DB) rather than parquet: this is keyed reference data queried by `(ticker, date)`,
matching `hist_sentiment_llm_labeled` / `dispatched_signals`. The parquet crawlers
(`market_breadth`, `foreign_flow`) store per-date time series feeding features — a different shape.

Restatement policy: **delete-then-insert per ticker** on a successful non-empty fetch. vnstock
restates events (many rows carry a NaN `exright_date` that is filled in later), the payload is
authoritative, and it is ≤50 rows per ticker. Simpler and more correct than an anti-join.

---

## 4. Architecture

```
EOD pipeline (main.py)
  └─ refresh_corporate_actions()          [NEW, network, config-gated, never raises]
        ├─ tickers = portfolio(non-cron) ∪ dispatched_signals(OPEN, real)
        ├─ filter to stale (> ca_event_refresh_days) , cap at ca_event_max_refresh_per_run
        ├─ per ticker: throttle → Company(source='VCI', symbol=t).events()
        └─ classify → delete-then-insert into corporate_action_events (+ fetch log)

  └─ notify_portfolio_guard() → _run_guard_for_users()
        ├─ closes = price_lookup.closes_between(...)          [unchanged]
        ├─ ca_events = corporate_actions.load_events(t, entry, today)   [NEW, cache read only]
        ├─ prices = portfolio_guard.resolve_lot_prices(pos, closes_abs, ca_events)   [NEW, PURE]
        │     └─ corporate_actions.resolve_adjustment(closes, events)   [PURE, tiers A-D]
        └─ portfolio_guard.evaluate_position(..., ca_events=ca_events)  [PURE, same call]
```

`portfolio_guard.py` gains **no** network call and **no** new I/O — it receives already-loaded
events as a parameter, exactly like `prediction_5d` / `regime` today. Its hard contract holds.

### New DuckDB tables

```sql
CREATE TABLE IF NOT EXISTS corporate_action_events (
    ticker          VARCHAR NOT NULL,
    event_id        VARCHAR,          -- vnstock `id`
    event_code      VARCHAR,          -- ISS / DIV / AGME / ...
    category        VARCHAR,          -- DIVIDEND / OTHER / ...
    event_title_vi  VARCHAR,
    exercise_ratio  DOUBLE,
    value_per_share DOUBLE,
    exright_date    DATE,
    record_date     DATE,
    public_date     DATE,
    kind            VARCHAR,          -- derived, see §2.2 taxonomy
    price_factor    DOUBLE,           -- derived 1/(1+r); NULL when unpriceable
    fetched_at      TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS corporate_action_fetch_log (
    ticker          VARCHAR NOT NULL,
    last_fetched_at TIMESTAMP,
    event_count     INTEGER,
    status          VARCHAR           -- 'ok' | 'empty' | 'error'
);
```

The fetch log exists so "fetched, legitimately zero events" is distinguishable from "never
fetched" — without it a quiet ticker is re-fetched on every single run.

---

## 5. Public contracts

### `src/data/corporate_actions.py` (NEW)

```python
# ── PURE ────────────────────────────────────────────────────────────────
KIND_STOCK_DIVIDEND, KIND_BONUS, KIND_RIGHTS, KIND_ESOP, \
KIND_PRIVATE, KIND_CONVERTIBLE, KIND_CASH_DIVIDEND, KIND_OTHER: str

def classify_event(event_code: str | None, event_title_vi: str | None) -> str
def theoretical_price_factor(kind: str, exercise_ratio: float | None) -> float | None
def bundle_factor(events: list[dict]) -> float | None      # 1/(1 + sum r) over priceable
def resolve_adjustment(
    closes: list[float],
    events: list[dict],
    *,
    max_session_move: float = 0.10,
    factor_tolerance: float = 0.10,
) -> dict          # {"tier","factor","label_vi","matched","gap_ratios"}
def back_adjust_closes(closes: list[float], factor: float,
                       max_session_move: float = 0.10) -> list[float]

# ── I/O (never raises) ──────────────────────────────────────────────────
def ensure_tables(conn) -> None
def load_events(ticker: str, start, end, db_path: str | None = None) -> list[dict]
def tickers_needing_refresh(tickers, db_path=None, refresh_days=..., limit=...) -> list[str]
def refresh_events(tickers: list[str], db_path: str | None = None) -> int
```

### `src/trading/portfolio_guard.py` (MODIFIED — additive only)

```python
def resolve_lot_prices(                                  # NEW, PURE
    position: dict,
    closes_since_entry_abs: list[float],
    ca_events: list[dict] | None = None,
) -> dict    # {"entry_effective","closes_effective","today_close","peak",
             #  "pnl_pct","drawdown_pct","ca_tier","ca_label_vi","ca_factor"}

def evaluate_position(..., ca_events: list[dict] | None = None) -> list[dict]
#   default None  ⇒  byte-identical to today's behavior
#   trigger dicts gain "ca_adjusted": bool and "ca_label_vi": str | None
#   existing "kind" / "message_vi" / "ca_gap_downgraded" keys PRESERVED
```

### `config/settings.py::TradingConfig` (4 new knobs)

```python
corporate_action_events_enabled: bool = True   # master kill-switch (refresh + consumption)
ca_event_refresh_days: int = 7
ca_event_max_refresh_per_run: int = 20
ca_event_factor_tolerance: float = 0.10
```

---

## 6. Touchpoints

| # | File | Change |
|---|---|---|
| 1 | `src/data/corporate_actions.py` | **NEW** — taxonomy, factors, tiered resolver, cache I/O |
| 2 | `config/settings.py` | +4 `TradingConfig` knobs (documented block, repo comment style) |
| 3 | `src/trading/portfolio_guard.py` | +`resolve_lot_prices`; `evaluate_position` gains optional `ca_events`; hard-stop + trailing-stop branches use corrected basis on tier A; module docstring updated |
| 4 | `main.py` | +`refresh_corporate_actions()`; call it in `full_pipeline` + `inference_only` before `notify_portfolio_guard()`; `_run_guard_for_users` loads `ca_events` and uses `resolve_lot_prices` for the displayed `pnl_pct` |
| 5 | `tests/test_corporate_actions.py` | **NEW** — taxonomy / factors / tiers / cache I/O |
| 6 | `tests/test_portfolio_guard.py` | +VHM regression block (raw fires 2, corrected fires 0) |
| 7 | `process/context/all-context.md` | new paragraph documenting the module, taxonomy and tiers |

**Explicitly NOT touched:** `src/data/price_lookup.py` (Decision 3), `src/trading/signal_ledger.py`
(Decision 6), `src/data/crawlers.py`, the feature pipeline, any model artifact. No feature-recipe
change → **no retrain**.

---

## 7. Blast radius

| Surface | Risk | Control |
|---|---|---|
| EOD `full_pipeline` | a hung vnstock call delays the pipeline | per-ticker throttle + `refresh_events` never raises; caps at `ca_event_max_refresh_per_run` (≤20, realistically 2) |
| Portfolio guard alerts | a wrong tier-A match suppresses a real stop-loss | three independent conditions must agree; rights-contamination veto; tier B fallback; kill-switch |
| `/guard` on-demand | same path | shares `_run_guard_for_users`, same controls |
| Core DuckDB | two new tables | additive `CREATE TABLE IF NOT EXISTS`; no existing table read or written |
| Existing tests | `evaluate_position` signature | new param is keyword-only with `None` default → all current call sites and tests unchanged |
| Serve / backtest / models | none | no feature, label, threshold or artifact touched |
| Network | new vnstock dependency | config-gated; failure degrades to the cache, then to tier B |

**Adjacent risk discovered, deliberately out of scope:** the same 47 unadjusted gaps also feed
`src/backtest/pipeline.py::build_features` through the shards, so every raw CA gap injects a
phantom −50%-class return observation into the momentum/return features for that ticker. That is a
larger, separate concern than the guard alert reported here and must not be folded into this pass.
Recorded in §9.

---

## 8. Implementation checklist

1. Create `src/data/corporate_actions.py` with the module docstring recording the §2 live findings
   (VCI-only, 50-row cap, `exercise_ratio` polysemy, `dOFF ∈ [-6,0]`, partial pre-adjustment).
2. Implement the pure layer: `classify_event`, `theoretical_price_factor`, `bundle_factor`.
3. Implement `resolve_adjustment` (tiers A–D, factor-agreement matching, rights-contamination
   veto) and `back_adjust_closes`.
4. Implement the I/O layer: `ensure_tables`, `load_events`, `tickers_needing_refresh`,
   `refresh_events` (ticker validation, throttle, delete-then-insert, fetch log, never raises).
5. Add the 4 `TradingConfig` knobs in `config/settings.py` with the repo's documented-comment style.
6. Add `portfolio_guard.resolve_lot_prices`; refactor `evaluate_position` to call it; thread the
   optional `ca_events` param; add `ca_adjusted` / `ca_label_vi` to trigger dicts; update the
   module docstring (its "no network" contract still holds — say so explicitly).
7. Add `main.refresh_corporate_actions()`; wire into `full_pipeline` + `inference_only`; update
   `_run_guard_for_users` to load events and take its displayed `pnl_pct` from `resolve_lot_prices`.
8. Write `tests/test_corporate_actions.py` (see §8.1).
9. Extend `tests/test_portfolio_guard.py` with the VHM regression (§8.2).
10. Run `pytest -q` — full suite must stay green (894 baseline + new).
11. Live read-only dry run: refresh the 2 real tickers, print the resolved tier/factor/label for
    the live VHM lot, confirm zero triggers.
12. Update `process/context/all-context.md`.

### 8.1 `tests/test_corporate_actions.py`

Conventions: pure functions tested directly; `tmp_path` DuckDB for I/O; `Company` monkeypatched.
**Zero real network, zero real parquet.**

- **Taxonomy** — all 8 real `event_title_vi` shapes from §2.2 map to the right `kind`; `DIV` →
  cash dividend; `DDINS`/`AGME`/`AIS` → other.
- **Factors** — `1.0 → 0.5`, `0.295 → 0.7722`, `0.10 → 0.9091`; rights/ESOP/private/convertible →
  `None`; `exercise_ratio` of `0.0` and `None` → `None` (not `1.0`).
- **Bundling** — GEX-shape (0.20 + 0.25) → `1/1.45`; mixed priceable+unpriceable → `None`.
- **Tier A** — VHM shape: closes with a `0.5039` gap + declared `ratio=1.0` → tier `A`,
  factor `0.5`, non-empty `label_vi`.
- **Tier A rejected on tolerance** — AAN shape (observed `0.7043`, declared 23% → `0.8130`,
  err −13.37%) → falls back to tier `B`.
- **Tier A vetoed by rights** — MBB shape (stock div 15% + rights 10% same window) → tier `B`
  even though the numbers would otherwise pass.
- **Tier B** — gap present, empty event list → self-referential factor, matches
  `derive_ca_adjustment_factor` exactly.
- **Tier C** — no gap, no events → factor `1.0`.
- **Tier D** — declared event present but closes have no gap (VCB 2025 shape) → factor `1.0`,
  nothing adjusted. *(the double-adjustment guard)*
- **`back_adjust_closes`** — pre-gap segment scaled, post-gap untouched, peak recomputed;
  multi-gap composition.
- **I/O** — `refresh_events` with a stubbed `Company` writes N rows + an `ok` log row; a second
  run with a restated payload replaces (not duplicates) the ticker's rows; an empty DataFrame
  writes an `empty` log row and leaves existing rows intact; a raising `Company` writes an
  `error` log row, leaves the cache untouched and returns `0`; an invalid ticker is skipped
  without a call.
- **`load_events`** — window filter on `exright_date`; missing table → `[]`.
- **`tickers_needing_refresh`** — fresh rows excluded, stale/never-fetched included, `limit`
  respected.
- **Kill-switch** — `corporate_action_events_enabled=False` → `refresh_events` performs no call.

### 8.2 `tests/test_portfolio_guard.py` additions

- `resolve_lot_prices` on the **real VHM series** (entry `152.5`, 26 closes with the
  `153,000 → 77,100` step) returns `pnl_pct ≈ -0.0584`, `drawdown_pct ≈ 0.0687`, `ca_tier == "A"`.
- `evaluate_position` with `ca_events=None` on that series → `{"hard_stop", "trailing_stop"}`
  **(today's behavior, must not regress)**.
- `evaluate_position` with the declared VHM event → **empty trigger list**.
- A lot that is genuinely down 20% *and* had a stock dividend → hard stop **still fires**, with
  `ca_adjusted=True` and confident (non-disclaimer) wording.
- Tier B path still produces the existing `ca_gap_downgraded=True` disclaimer text.
- `_run_guard_for_users` with the kill-switch OFF makes no `corporate_actions` call and produces
  byte-identical cards to today.

---

## 9. Verification evidence

| Claim | How verified | Status |
|---|---|---|
| VHM alert numbers | live parquet + live `portfolio` table replay: `-52.92%` / `53.07%` | **DONE (research)** |
| Corrected numbers clear both thresholds | `-5.84%` / `6.87%` vs `-7%` / `8%` | **DONE (research)** |
| `Company(source='VCI').events()` works | 15 tickers, 727 rows, 0.37–1.09s | **DONE (research)** |
| `source='KBS'` unusable | returns `(0,0)` for VHM + HPG | **DONE (research)** |
| `exercise_ratio` polysemy | `DIV` par-normalized (`450/0.045 = 10000` ×4 cases); `ISS` = share ratio | **DONE (research)** |
| Date offset `[-6, 0]` | 21 matched gaps since 2026-04-01 | **DONE (research)** |
| Partial pre-adjustment is real | 6-case table §2.4 | **DONE (research)** |
| Full suite green | `pytest -q` | pending EXECUTE |
| Live dry run: 2 tickers refresh, VHM resolves tier A, 0 triggers | manual read-only run | pending EXECUTE |
| First production EOD run confirms no regression | 15:30 ICT cron | pending post-EXECUTE (plan stays ACTIVE until then, same precedent as the intraday scanner and portfolio guard plans) |

### Deferred follow-ups (do NOT expand this plan)

1. Wire the declared factor into `signal_ledger._evaluate_pnl` (Decision 6).
2. Cash-dividend total-return adjustment (Decision 7).
3. **Feature-pipeline contamination** from the same 47 raw gaps (§7) — the largest of the three.

---

## 10. Resume and execution handoff

- **Selected plan:** `process/general-plans/active/vnstock-ca-event-detection_PLAN_13-08-26.md`
- **Resume point:** §8 step 1 (nothing implemented yet — RESEARCH + INNOVATE + PLAN complete).
- **Gate:** requires explicit `ENTER EXECUTE MODE`.
- **Pre-flight:** `git status` is already dirty with unrelated model artifacts
  (`models/mr/*`, `models/saved/*`, `AGENTS.md`) — do **not** stage those; this work touches only
  the §6 files.
- **Verification command:** `pytest -q`
- **Rollback:** set `"corporate_action_events_enabled": false` in `config/settings.json` and
  restart — restores exact current behavior without a code revert.
- **Scratch artifacts** (research probes, safe to delete):
  `C:\Users\caokh\AppData\Local\Temp\claude\C--Users-caokh-Desktop-vscode-stock-price-v3\13752334-b343-48cc-90b1-72c2c97248aa\scratchpad\probe*.py`
