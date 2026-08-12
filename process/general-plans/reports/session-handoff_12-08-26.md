# Session handoff — 2026-08-12

Two-day session (10–12 Aug). Read this before touching the serve path or citing
any backtest number.

**One-line state:** `main` @ `c9eff85`, 1131 tests green, worktree clean. The EOD
pipeline runs clean in ~11 min (was 29.5). The system dispatches almost nothing,
and we now know exactly why.

---

## 1. The single most important finding

**The validated OOS numbers did not describe the deployed system.** Measured over
the full 920-day OOS window:

> The configuration production actually runs retains **1.7% of the validated
> PnL** (87.6M of 5.19B VND) and **0.11% of the bets** — **five trades in 920
> days**.

Everything below is either a cause of that or a consequence.

### 1a. Root cause: a classification threshold was adopted as an admission rule

The GOLDEN artifact stores two numbers:

| field | value | what it actually is |
|---|---|---|
| `up_threshold` | 0.46 | **metric only** in the backtest — feeds UP-precision / confusion matrix |
| `signal_threshold` | 0.41 | the number the **engine traded on** (`sig_thr = thr - 0.05`) |

`predict_v3_horizon`'s `meta_gate` reads `bot.up_threshold`, so **serve admits on
0.46 while the sweep optimised 0.41**. Not a 5pp slip — a category error.

### 1b. Every defensive layer costs Sharpe — 8 of 8 cells

`scripts/analyze_defensive_layers_ab.py`, single seed, 920 OOS days:

| threshold | layers | NetPnL | Sharpe | MaxDD | buys |
|---|---|---|---|---|---|
| 0.41 (validated) | none | +5.19B | +0.600 | −31.00% | 4555 |
| 0.41 | brake | +4.18B | +0.557 | −29.17% | 4555 |
| 0.41 | filters | +2.08B | +0.412 | −21.86% | 1248 |
| 0.41 | all four | +1.79B | +0.393 | −22.48% | 1248 |
| 0.46 (serve) | none | +644M | **+0.620** | −4.44% | 46 |
| 0.46 | brake | +444M | +0.480 | −4.19% | 46 |
| 0.46 | filters | +97.6M | +0.355 | −0.91% | 5 |
| **0.46 + all four = SERVE TODAY** | | **+87.6M** | **+0.340** | −0.87% | **5** |

Best Sharpe of all eight is **0.46 with NO layers**, so the τ-gate is the only
defensive layer that pays for itself. The four added later — sector cap,
open-cohort dedup, hysteresis, 3-leg brake, each added after a real loss, **none
ever measured** — trade Sharpe for drawdown roughly one-for-one.

### 1c. Gate level sweep

`scripts/analyze_gate_level_sweep.py`, 18 arms. With all four layers on:

| gate | Sharpe | MaxDD | buys | NetPnL |
|---|---|---|---|---|
| 0.42 | 0.548 | −16.18% | 913 | +2.40B |
| **0.43** | **0.633** | **−10.75%** | **532** | **+2.05B** |
| **0.44** | **0.693** | **−7.64%** | 151 | +1.15B |
| 0.45 | 0.200 | −2.55% | 21 | +99M |
| 0.46 (serve) | 0.340 | −0.87% | 5 | +87.6M |

**Do NOT pick a level off this table.** The Sharpe curve is non-monotonic (0.44
peak → 0.45 collapse → 0.46 bounce), the dips sit on 21–273-trade samples, it is
ONE seed, and eyeballing a winner from 18 arms is precisely the threshold-mining
DSR exists to punish. The defensible reading is only: **0.46 is on the wrong side
of the peak.**

### 1c-bis. The production sweep disagrees with 1c on levels — and on the ranking

A 2-arm smoke test of the real production path (`run_backtest.py --serve-parity
--sweep-thresholds 0.44,0.43 --no-save`, 4 seeds each) landed **nowhere near** the
1c table:

| thr | mean Sharpe | per-seed Sharpe | mean NetPnL | DD range |
|---|---|---|---|---|
| 0.44 | +0.411 | 0.21 / 0.43 / 0.50 / 0.51 | +1.22B | −12.57…−13.85% |
| **0.43** | **+0.469** | 0.41 / 0.49 / 0.44 / 0.53 | **+1.56B** | −12.54…−15.16% |

Two things to take from this, in order of importance:

1. **The 0.43-vs-0.44 ranking FLIPS.** 1c says 0.44 (0.693) beats 0.43 (0.633);
   the production sweep says 0.43 (0.469) beats 0.44 (0.411).
2. **Seed variance swamps the threshold effect.** At thr=0.44 alone, Sharpe spans
   **0.21 → 0.51** (range 0.30) — roughly **5× the 0.058 mean gap** between the two
   thresholds. Four seeds cannot rank 0.43 against 0.44. Anything the Saturday
   retrain picks inside 0.41–0.44 is a coin flip within noise.

**Why they disagree — they are not measuring the same model.**
`analyze_gate_level_sweep.py` loads the **frozen GOLDEN ensemble** off
`v3_ensemble_20d.joblib` (the stale 2026-07-17 artifact) and varies only the gate.
`run_backtest.py` **retrains 4 seeds** and averages. So 1c is the gate response of
one specific soon-to-be-replaced model; the sweep is the gate response averaged
over fresh ones. Budget is not the difference — neither sets
`tranche_budget_days`, so both run the nav/30 calendar budget.

**Never quote 1c's absolute Sharpe or DD as a production expectation.** Its
directional finding (0.46 sits past the peak) survives — it is the only claim both
harnesses support.

### 1d. Two things that looked like divergences but are not

- **`admission_mode` was never a variable.** `absolute_gate` and
  `cross_sectional` are byte-identical at every level, because
  `admission_pool_cap` (6) ≥ `max_positions` (5) so the pool cap never binds.
- **Ranking-by-sentiment and top-3-vs-top-5 are vacuous.** Random-ranking arms
  across 3 seeds and a top-3 arm came back byte-identical to baseline: under the
  τ-gate the admitted set never exceeds 3 names (46 buys / ~46 gate-open days ≈
  1/day), so there is nothing to reorder or truncate. **This changes if the gate
  is ever loosened.**

---

## 2. What to watch first: Saturday 2026-08-15, 09:00

`quant_weekly_retrain` runs `scripts/retrain_all.ps1`, which now calls
`run_backtest.py --serve-parity --sweep-thresholds 0.46,0.45,0.44,0.43,0.42,0.41`.

This is the first retrain that:
- optimises the threshold **under production conditions** (gate-offset 0 + all
  four defensive layers), so the stored `up_threshold` finally *is* the optimised
  value, and
- runs under the **repaired promote-gate**, which had rejected four consecutive
  retrains and froze the live T+20 artifact at 2026-07-17.

If it promotes, gate starvation (serve p90 0.423 vs τ 0.46) and the threshold
conflation resolve together. **Check the log and `metadata.sweep_conditions` of
the new artifact.**

**Judge it on the right thing.** Per 1c-bis, the 4-seed Sharpe spread at a *single*
threshold (0.21→0.51) is ~5× the gap between adjacent thresholds, so **which** level
inside 0.41–0.44 wins is noise. The result that matters is only whether the picked
level is **below 0.45** — i.e. off the starved side of the curve — and whether the
gate lets it promote at all. Do not read the winning `up_threshold` as a tuned
optimum, and do not re-run the sweep hoping for a "better" level: that is the
threshold-mining the DSR penalty exists to catch, and the sweep is already the
`n_trials` count feeding it.

Runtime note, so nothing looks broken: the **first** threshold in a sweep costs
~30 min and every one after costs ~90 s. Per-seed inference caches are populated on
the first threshold and reused (`run_backtest.py:481-486`) — P(UP) is
threshold-independent, so the GBM scoring is paid once per seed, not per arm.

---

## 3. Commits this session (14)

| commit | what |
|---|---|
| `fa9a2e9` | FastConnect concurrent OHLCV prefetch — crawl 25.4 → 6.4 min |
| `ded6e4c` | dtype int64/float64 canonicalisation (crashed the EOD run) |
| `855d8fc` | same dtype fix in `backtest.pipeline.load_ohlcv` — the reader the first fix missed |
| `7df162e` | paperlog horizon columns un-swapped → `*_primary`/`*_secondary` + `primary_horizon_days`; `check_drift` fixed; `GATE STARVED` check added |
| `08effa2` | promote-gate: stop comparing MaxDD across nested OOS windows |
| `dd177cd` | backfill T5/T20 mapping un-swapped; `dispatched_signals` audited clean |
| `76de404` | `liquid_at_log` on the paperlog |
| `ee8a8e7` | `is_paper` on the ledger so reconstructed rows cannot veto real dispatches |
| `94441af` | confluence bucket labels by true horizon; stripped 10 BOMs |
| `4ba6378` | argmax admission mode + A/B |
| `289b76c` | argmax REJECTED — 0 buys in 920 days |
| `418acca` | brake `drift_raw` — distinguishes "idle" from "failed open" |
| `ec8e28d` | market-breadth feed (DailyIndex, 7200 rows 2016→now) |
| `e3f7765` | limit-down EDA UNTESTABLE — `floors` is a ~1.5-year series |
| `6060b31` | serve-stack A/B — argmax gate is an off switch |
| `4081965` | **arbitrator becomes a VETO at the entry gate** |
| `6949524` | paper rows hidden from operator reads; dual-horizon pair merged |
| `6e93828` | four defensive layers encoded into the engine, opt-in |
| `c9eff85` | `--serve-parity` sweep + `sweep_conditions` provenance stamp |

---

## 4. Open items, ranked

### 4.1 Event-rescue path — the last unmeasured money path

`main.build_event_overrides`:

```python
SAFE_BUY_THRESHOLD   = 0.45
EVENT_MIN_P_UP       = 0.42    # rescue band: 0.42 <= P(UP) < 0.45
EVENT_BULL_SENTIMENT = 0.60
_EVENT_CAP           = 0.05    # 5% NAV

if _ov:                        # event override present
    _w = float(_ov["weight"])  # 5% NAV, ~7x the tranche per-name weight
else:                          # <-- THIS WHOLE BRANCH IS SKIPPED
    ...NO_TRADE skip / PENALTY x0.5 / exposure_scalar...
```

A name the τ-gate **rejected** is force-dispatched at 5% NAV on sentiment alone,
skipping regime sizing, the PENALTY multiplier and the entire 3-leg brake. The
band `[0.42, 0.45)` is exactly where P(UP) measured **anti-informative** (Platt
slope −0.274; bin 0.4–0.5 realised 29.3% vs bin 0.2–0.3's 45.9%).

Not backtestable — sentiment has no point-in-time history. **Deferred by user
decision until the paperlog has enough settled rows.** No evidence it caused the
July losses (their weights are fully explained by `1/(hold_days × n_picks)`).

### 4.2 `/verify` prints a T+5-dominant verdict with nothing qualifying it

`_build_verify_report` has no `warning_lines` and no score line, and prints
`Kết luận tổng hợp: {verdict}` where the verdict is **T+5-dominant**
(`verify_single_ticker` passes `SHORT_HORIZON` as primary, and
`make_final_decision` is asymmetric — `pred_5d == 2` → BUY regardless). T+5=UP
with T+20=DOWN measured **−1.57%** vs T+20=UP's **+1.19%**.

**RESOLVED — deliberately left as-is.** The warning block was built (`2632845`,
11 tests) and then **reverted at the operator's request** (`258ba41`) because it
made the card ugly. Reverted whole, not just the render, so no dead plumbing
remains.

Accepted rationale: `/verify` is a **lookup tool, not a dispatch path** — nothing
is bought because a card rendered. The same three disqualifiers are still shown
where they gate money, on the dispatch card (`_build_signal_card`). Do not
re-add this to `/verify` without asking; it was removed on purpose, not missed.

### 4.3 `ceilings` / `floors` need ~3 more years

The DailyIndex feed is real and worth keeping — **A/D is usable from 2020**
(~6.5 years). But `ceilings`/`floors` only start 2025, and SSI sends a literal
`0` (not null) for eras it does not carry, so the backfill looks complete while
being structurally empty. Inside a chrono split the flag degenerates into a
**time dummy** (train 1.9% floors≥1 vs holdout 86%).

### 4.4 Lower priority

- FastConnect foreign-flow universe (359) is missing liquid names: `CTG`, `POW`,
  `CII`, `CMG`, `CTR`, `CSV`, `PC1`. Only matters if that feed is used again
  (foreign flow was REJECTED, corr ~0.001).
- `prop_*` (tự doanh) columns have **never held a value** from either source, and
  FastConnect has no prop data at all — all 5 working endpoints dumped, 8 guessed
  endpoint names 404. Documented placeholder, not a bug.

---

## 5. Conclusions RETRACTED this session — do not cite these

Five of my own claims were withdrawn after checking provenance or slicing by
liquidity. A fresh session must not pick them back up:

| retracted claim | why it was wrong |
|---|---|
| "the edge lives in the P(UP) top decile, exactly where τ sits" | d10 `+0.56%` was driven by **untradeable** names; liquid-only d10 is **−1.84%**, and d1 is the best decile |
| "BUY calls beat baseline by +1.95pp" | 43 of 44 settled BUY rows are **not** in the ADV top-50 |
| "τ was calibrated on a different population" | `run_backtest` already sweeps with `liquid_top_n=50`; the real cause is distribution drift against a **frozen** artifact |
| "ranking by sentiment discards the measured edge" | measured **vacuous** — the admitted set never exceeds 3 names, so there is no ordering choice to make |
| "floors captures real capitulation events (nonzero 809/7200)" | the 809 are concentrated in the last ~1.5 years, not spread over the decade |

Also **UNVERIFIED pending re-run** (derived from by-name reads of the
swapped paperlog columns, now fixed): `analyze_rank_sleeve.py`. The confluence
and calibration analyses **were** re-run and CONFIRMED in direction.

---

## 6. Commands

Start the bot (own window, Ctrl+C stops supervisor + bot; only ONE instance may
poll the token):

```bash
powershell -ExecutionPolicy Bypass -File C:\Users\caokh\Desktop\vscode\stock_price_v3\run_bot_forever.ps1
```

Check it is not already running:

```bash
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_bot' }
```

Serve-parity sweep by hand (no save):

```bash
python run_backtest.py --mode tranche --hold-days 30 --serve-parity --sweep-thresholds 0.44,0.43 --no-save
```

Drift / gate-starvation monitor:

```bash
python scripts/check_drift.py
```

Inference only (no crawl, no sentiment — safe to re-run, costs one arbitrator
Gemini call):

```bash
python main.py --task inference_only
```

---

## 7. Environment gotchas that cost time this session

- **Never edit source files via PowerShell `Get-Content -Raw` → `Set-Content`.**
  Even with `-Encoding utf8` the round-trip mangles non-ASCII (em-dash →
  mojibake) and adds BOMs. It corrupted 216 lines of `main.py` in one command and
  left BOMs in 10 files. Use the Edit tool. This repo is full of Vietnamese
  strings — almost every file is exposed.
- **`git commit -m @'...'@` breaks on literal `"` and `%`** — the message splits
  into bogus pathspecs and the commit fails. Write bodies without those chars.
- Bot/pipeline supervisors die with their PowerShell window. The 15:30 cron
  missed 2026-08-06 and 08-07 simply because the laptop was off — not a bug.
- After any bulk text operation, check `git diff` for mojibake before committing.

---

## 8. The habit to carry forward

Five retractions in one session, all the same shape: **reading numbers without
checking where the columns came from, or without slicing to the tradeable
universe.** Two related lessons already cost a production crash each:

- A schema fix must cover **every reader**, not the one that crashed
  (`alpha360_generator` fixed, `load_ohlcv` still broken → the meta-controller
  silently read full exposure).
- The same applies to filters: `open_tickers()` was fixed for `is_paper` but
  `list_open` / `list_closed_since` were not, so 118 phantom positions reached
  the user's Telegram and SELL alerts for them were next.

Before reporting a finding: verify column provenance, filter to tradeable names,
and grep every consumer of whatever you changed.
