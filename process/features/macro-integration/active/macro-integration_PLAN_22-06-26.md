# Macro Integration (A/B) — Plan

**Plan type:** COMPLEX (4 sequential phases)
**Feature folder:** `process/features/macro-integration/`
**Created:** 2026-06-22
**Status:** READY FOR EXECUTE (phase-by-phase)
**Branch:** `fix/techdebt-paperlog` (or a new `feat/macro-integration` — confirm at EXECUTE)

---

## Goal

Add a live **macro crawler** (DXY / USD-VND / SP500) and wire it in two places:
1. **HMM regime overlay** (the safe win) — give the macro-regime classifier real
   macro inputs instead of the price-only proxy it uses today.
2. **GBM stack as a flagged variant** — behind `RunConfig.use_macro_features`
   (default OFF), so we can **A/B** whether market-level features help the
   cross-sectional ranker BEFORE committing the expensive full retrain.

Locked decisions (2026-06-22):
- **Source: `yfinance`** (already installed 1.2.2). vnstock is VN-only — cannot
  serve DXY/SP500/USD-VND. Symbols: `^GSPC` (SP500), `DX-Y.NYB` (DXY), `VND=X`
  (USD/VND).
- **Price-macros only** — no market-sentiment series (provenance/cost risk;
  per-ticker sentiment already lives in the serve path).

Design context (why this is structured as an A/B, see `src/backtest/pipeline.py:415`):
V4 deliberately removed macro from the GBM — macro is **market-level** (identical
across tickers each day) so it carries ~0 cross-sectional **ranking** signal. The
HMM overlay is the architecturally-correct home for market-level signal. P3/P4
test the GBM hypothesis empirically rather than assuming.

---

## Touchpoints

| File | Phase | Change |
|---|---|---|
| `src/data/macro_crawler.py` (NEW) | P1 | yfinance fetch → `data/macro_daily.parquet` |
| `main.py` (`full_pipeline`, `parse_args`) | P1 | new `--task crawl_macro`; fold into EOD pipeline |
| `requirements.txt` | P1 | pin `yfinance` (already installed; make explicit) |
| `config/settings.py` | P1/P3 | `macro_parquet` path already exists; add `use_macro_features: bool = False`, `use_macro_in_hmm: bool = True` |
| `src/models/macro_risk_hmm.py` | P2 | `build_market_proxy_returns` / `train_macro_risk_hmm` optionally blend macro returns |
| `src/backtest/pipeline.py` (`build_features`, `FEATURE_SCHEMA`) | P3 | flag-gated macro merge; conditional schema so the recipe hash changes ONLY when flag ON |
| `train_models.py` / `run_backtest.py` | P4 | A/B: retrain + backtest both variants |
| `tests/test_macro_crawler.py` (NEW) | P1 | mock yfinance |
| `tests/test_macro_hmm.py` (NEW) | P2 | macro-blended HMM wiring |
| `process/context/all-context.md` | P1 | add `macro-integration` to Current features |

---

## Phase 1 — Macro crawler (no retrain)

**New: `src/data/macro_crawler.py`**
- `fetch_macro_history(start="2015-01-01") -> pd.DataFrame` — yfinance pull of the
  3 symbols; outer-join on date; columns `date, sp500, dxy, usdvnd`. Forward-fill
  small gaps (US holidays vs VN calendar); compute `*_ret` (pct_change).
- `update_macro_daily(parquet_path, days_back=None)` — full backfill (None) or
  incremental (last N days); idempotent upsert keyed on `date`; writes
  `data/macro_daily.parquet` (revive the retired-but-configured path —
  `CONFIG.paths.macro_parquet`).
- VN price-scale: N/A (macro series are absolute index/FX levels; only `*_ret`
  ratios feed models).
- Resilience: any symbol fetch failure → log + skip that column (NULL), never
  crash the EOD pipeline (mirror `sentiment_crawler` guard pattern).

**`main.py`**: add `--task crawl_macro` → `update_macro_daily(...)`; add a step 1b
in `full_pipeline` (after OHLCV crawl) calling `update_macro_daily(days_back=5)`.

**Tests** (`tests/test_macro_crawler.py`): monkeypatch `yfinance.Ticker.history`
to deterministic frames; assert schema, ret math, gap-fill, idempotent upsert,
single-symbol-failure degradation.

**Verify (PowerShell):** `python -m pytest tests/test_macro_crawler.py -v`;
manual `python main.py --task crawl_macro` → inspect `data/macro_daily.parquet`.

**Gate:** parquet built with 3 series back to 2015; suite green.

---

## Phase 2 — HMM overlay enrichment (HMM retrain only; the safe win)

- Extend `build_market_proxy_returns` to optionally append macro return columns
  (sp500_ret, dxy_ret, usdvnd_ret) as additional HMM observation dims, gated by
  `CONFIG.*.use_macro_in_hmm` (default ON).
- `train_macro_risk_hmm` consumes the widened observation matrix; keep 2-state
  Gaussian; re-fit. Falls back to price-only if `macro_daily.parquet` absent.
- **Leakage guard:** macro merged as-of `date` (no forward fill from the future);
  HMM is trained in-sample < cutoff exactly as today.

**Tests** (`tests/test_macro_hmm.py`): macro-blended proxy has the right shape;
flag OFF → identical to price-only; absent parquet → graceful price-only.

**Verify:** retrain HMM (cheap), confirm regime series still sane (state dwell
times, P(Bull) range); `python -m pytest tests/test_macro_hmm.py -v`.

**Gate:** HMM trains with macro dims; regime sanity preserved; suite green.
*This phase can ship independently — it does NOT touch the GBM recipe.*

---

## Phase 3 — GBM macro-feature variant (flagged, default OFF, no retrain yet)

- `config/settings.py` / `RunConfig`: `use_macro_features: bool = False`.
- `build_features`: when ON, left-join `macro_daily` on `date` and append
  `{sp500_ret, dxy_ret, usdvnd_ret}` to the continuous pool (NOT cross-sectional
  Z — they are market-level by nature; document this).
- **Recipe-hash discipline (critical):** build `FEATURE_SCHEMA` CONDITIONALLY on
  `use_macro_features` so `compute_feature_schema_hash(...)` yields:
  - flag OFF → **same hash as today** (`v2-sha8:53b5bd85`) → no retrain forced,
    serve-path unaffected.
  - flag ON → **new hash** → retrain required (this is the A/B variant).
  - RESEARCH sub-step first: confirm exactly how `FEATURE_SCHEMA` +
    `compute_feature_schema_hash` are assembled so the conditional is correct
    and the OFF-path hash is byte-identical to current.

**Tests:** flag OFF → `FEATURE_RECIPE_VERSION` unchanged + `build_features`
columns unchanged; flag ON → macro cols present + hash differs.

**Gate:** flag OFF is a provable no-op (hash identical); flag ON adds 3 cols.

---

## Phase 4 — A/B + decision (the ONE expensive full retrain)

- Train two checkpoints: baseline (flag OFF, == current) and macro-GBM (flag ON).
  `python train_models.py` per variant (4 seeds each; ~mins/seed).
- Backtest both via `run_backtest.py` (tranche/H=30 default), regime-sizing as
  configured.
- **Decision criteria** (report to `process/features/macro-integration/reports/`):
  compare Net, Sharpe, MaxDD, **DSR**, turnover. Keep GBM macro ONLY if it
  improves the deflated metric without inflating DD; else KILL the flag (leave
  default OFF) and keep macro solely in the HMM overlay (P2).
- Whichever wins, the HMM-overlay macro (P2) stays.

**Gate:** A/B report written; explicit keep/kill decision; if kill, recipe hash
returns to baseline (flag OFF default → no serve-path disruption).

---

## Sequencing & cost

P1 → P2 (ship the safe win) → P3 → P4. The expensive full retrain is isolated to
**P4** and gated on the A/B. P1+P2 deliver value with no GBM retrain.

## Env / verification reminders
- PowerShell only (git-bash broken). Runner: `python -m pytest` (bare Python311).
- Orchestrator runs pytest + commits (subagents can't verify here).
- yfinance hits the network — tests MUST mock it; only the manual crawl step
  touches Yahoo.

## Open items to confirm at EXECUTE
- Branch: stay on `fix/techdebt-paperlog` or cut `feat/macro-integration`?
- Macro gap-fill policy across US/VN trading-calendar mismatch (ffill ≤ N days?).
- HMM observation weighting once macro dims are added (equal vs scaled).
