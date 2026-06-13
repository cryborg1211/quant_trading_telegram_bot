# Session Handoff — 13-06-26

Context dump for a fresh session. Everything below is committed unless flagged
**UNCOMMITTED**.

## TL;DR — what the system does now

| Surface | Model / behavior |
|---|---|
| Daily EOD broadcast + `/suggest_buy20` | **T+20 tranche** book (sig_thr 0.43, hold 30 sessions), cohort sizing NAV/(H×picks) |
| `/verify <ticker>` | **T+3** primary view + **T+20** cross-check |
| `/suggest_sell`, `/rebalance` | T+3 short view |
| `/exits` | open tranche cohorts: entered date, % NAV, sessions elapsed/remaining, DUE flag |
| `full_pipeline` step 4 | auto exit alerts when a cohort's hold horizon elapses, then marks CLOSED |
| Sentiment crawler | 3-day max lookback, covered-warrant filter, NULL-ticker drop, 180s GNews budget, live RSS (cafef/vietstock via GNews proxy + vietstock category feeds + vneconomy + tnck) |

Phase: **paper-trading / signal only**, no live order routing.

## This session's commits (oldest→newest)

- `961a42e` fix(crawler): budget(3d) + covered-warrant filter + NULL-ticker drop guards
- `c9ececb` perf(crawler): 180s GNews loop budget, early date/title filters, progress logs
- `4a35d53` feat(serve): tranche exit ledger + live RSS swap + T+3 verify-only short horizon
- `82310cf` feat(bot+crawler): cafef/vietstock restored via GNews RSS proxy + `/exits` command + bot copy fitted to tranche model

(Pre-session baseline tip: `a039250` — barrier-falsification report.)

## Models (models/saved/)

All three carry recipe `v2-sha8:53b5bd85` (matches live pipeline — load-clean).
Metrics are from the tranche/H=30 evaluator, 4 seeds, ~905 OOS days (3.9y).

| Artifact | Role | thr (up/sig) | seed-best Net | Sharpe | MaxDD |
|---|---|---|---|---|---|
| `v3_ensemble_20d.joblib` | **PRIMARY** dispatch | 0.48 / 0.43 | +44.2% | 0.65 | −27.4% |
| `v3_ensemble_3d.joblib` | /verify short | 0.50 / 0.45 | +46.2% | 0.73 | −23.3% |
| `v3_ensemble_5d.joblib` | retired (still loadable) | 0.50 / 0.45 | +34.7% | 0.55 | — |

- DSR FAILS on all (p≈0.31–0.34 vs 0.95 hurdle) — OOS window too short at this Sharpe to rule out luck. Drawdown is the binding constraint, not edge.
- **Side finding:** T+3 labels rank *better* than T+20 through the 30-session book (+46.2%/0.73/−23.3% > T+20). Worth a future sweep, but T+20 stays primary per the user's call this session.
- Position-level PT/SL barriers are **FALSIFIED** (edge is right-tail skew) — `tranche_pt_sigma`/`tranche_sl_sigma` MUST stay off.

## UNCOMMITTED working-tree state (decide in new session)

```
 M models/saved/v3_ensemble_20d.joblib      # freshly retrained this session
 M models/saved/v3_ensemble_5d.joblib
 M models/saved/v3_training_checkpoint.joblib
 M models/saved/prob_distribution.png
?? models/saved/v3_ensemble_3d.joblib       # NEW T+3 artifact (not yet tracked)
?? models/saved/backups/*.joblib            # auto-backups
 M .claude/hooks/.logs/hook-log.jsonl       # noise
```

**Decision needed:** these 3–13 MB binaries are tracked in git and churn every
retrain. Earlier audit recommended `git rm --cached models/saved/*.joblib` +
gitignore (+ gitignore `.claude/hooks/.logs/` and `scratch/`). Not done yet —
the new 3d artifact is currently untracked, so `/verify` works locally but a
fresh clone would lack it. Either commit it or move to the gitignore plan.

## Open follow-ups (none blocking)

1. **Hygiene commit** — gitignore + `git rm --cached` binaries, hook logs, scratch/; archive stale `process/features/v4-1-structural-debt/active/phase2-feature-schema-hashing_PLAN_10-06-26.md` (schema-hashing already shipped — it caught the drift that forced this session's retrain).
2. **Portfolio-level DD control** — the highest-value lever (vol-scaled tranche budgets, gross-exposure cap, regime-conditional sizing). Position stops are dead; DD −27% is what fails DSR.
3. **hold_days sweep (20/30/40)** — for a *meaningful* PBO; threshold axis saturates below sig ~0.41 (configs become clones → PBO uninformative).
4. **T+3-as-primary experiment** — given it beat T+20 through the book.
5. Structural debt Phase 3 — test coverage on hub nodes (`run_backtest.main`, `triple_barrier_pipeline`, `TabularEnsemble.fit`, `VNCostModel.simulate`).

## How to run

```powershell
python run_bot.py                      # Telegram bot (polling)
python main.py --task full_pipeline    # EOD: crawl → sentiment → inference → exit alerts
python main.py --task daily_inference  # no-crawl live path
# retrain a horizon (after any build_features change):
python train_models.py --tb-horizon 20
python run_backtest.py --sweep-thresholds 0.50,0.48,0.45   # persists v3_ensemble_20d.joblib
```

Tests: **210 green** as of `82310cf` (`python -m pytest tests/ -q`).

## Hard gotchas (do not relearn the hard way)

- **VN price scale:** parquet OHLCV is in *thousands* of VND. Scale to absolute VND (`WalkForwardConfig.price_unit_vnd=1000`) before any `VNCostModel` tick/band/qty math, or you get 5–100% phantom per-fill costs.
- **Feature recipe gate:** `FEATURE_RECIPE_VERSION` in `src/backtest/pipeline.py`. Serve refuses to load a model whose stamped recipe ≠ live. Any feature change ⇒ bump ⇒ full retrain of all horizons.
- **Train/serve timing:** engine scores D−1 features for D's ATC fill (1-bar lag); verified 100% score parity with train path on that basis.
- **Grid mode is structurally broken** for this signal (44 correlated entry dates → market beta dominates). Tranche is the only fit construction.
- **code-review-graph post-commit hook** throws a cosmetic cp1252 UnicodeEncodeError on Windows — commits still succeed; ignore it.

## Key files this session

- `src/trading/signal_ledger.py` — `dispatched_signals` table; `record_dispatch` / `list_open` / `check_exits_due` / `mark_closed`
- `src/crawlers/sentiment_crawler.py` — budget/warrant/RSS guards
- `src/data/price_lookup.py` — `trading_dates_after()` (session calendar)
- `main.py` — `notify_tranche_exits()`, T+20 primary defaults, `SHORT_HORIZON`
- `src/utils/telegram_bot.py` — `/exits`, `_build_exits_report`, T+20-only buy menu
- `src/reports/builders.py` — `SHORT_HORIZON_DAYS=3`, "N ngày" copy
