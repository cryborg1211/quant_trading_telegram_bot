# SESSION HANDOFF — 2026-06-24 (Quant Engine V4)

## STATE NOW
- Branch **`main`** @ `cba1254`. `feat/macro-integration` merged in (same tip).
- UNCOMMITTED (tracked, M): `models/saved/v3_ensemble_{5d,20d}.joblib` + `prob_distribution.png` = **retrained 24/06**. Deleted (D, not mine, leave): `AUDIT_REPORT/CONTEXT_DUMP/GEMINI/QODER.md`.
- Graph stale (built on feat/macro, now main) → `code-review-graph build` to rebuild.

## DONE THIS SESSION (committed to main)
- **Macro-Integration program COMPLETE** (`62447b1`→`cba1254`):
  - P1 `src/data/macro_crawler.py` → `data/macro_daily.parquet` (yfinance `^GSPC`/`DX-Y.NYB`/`VND=X`; vnstock CANT do global macro). `--task crawl_macro` + `full_pipeline` step1b.
  - P2 macro→regime HMM N-D (`src/models/macro_risk_hmm.py::build_regime_observation`; `use_macro_in_hmm` default ON). **KEPT** (more risk-aware; serve-safe = HMM is backtest-only, main.py serve doesnt apply p_bull).
  - P3 macro→GBM features flagged `use_macro_features` default **OFF**; `effective_recipe_version(False)` BYTE-IDENTICAL to baseline `v2-sha8:53b5bd85`.
  - P4 A/B → **KILL GBM-macro** (DD −13→−18.5%, PBO 43→87%, gain = seed noise, DSR no improve). report `process/features/macro-integration/reports/macro-ab-result_23-06-26.md`.
- **Paperlog temporal flaw FIXED** (`d6652e8`): progressive backfill, short scan gate 4d (`_PAPERLOG_SHORT_MATURE_DAYS`), T+3 no longer starves to 21; `tests/test_sentiment_paperlog.py` 12. audit `doc/audit_paperlog_temporal_flaw.md`.
- **Structural-Debt program CLOSED OUT** (`d9cbd51`): Phase1-3 done, +95 hub tests. ⚠ Phase3 commit `194525f` was ORPHANED mid-session (reset) — VERIFY the 4 test files actually on main.
- **Dashboard P2 live-render gate CLOSED** (`0491db6`): `tests/test_dashboard_app_smoke.py` AppTest, streamlit 1.58.
- **stock_price_old/ untracked** (`1a939e5`): 1.5GB legacy, `git rm --cached` (on disk, gitignored). `.git` history still holds blobs → `git filter-repo` for real reclaim.
- **Serve models RETRAINED 24/06** (T+5 + T+20): recipe `v2-sha8:53b5bd85`, macro-GBM OFF + macro-HMM ON, 4 seeds. **PAPER-ONLY** (DSR fail: T20 SR0.73 / T5 SR0.66 < hurdle 0.948; PBO T20 36% T5 75%). Old → `models/saved/backups/`.

## NEXT / OPEN
- **RESTART `run_bot.py`** → load new models (caches at load; live bot on OLD until restart).
- Commit retrained models? (tracked binaries M) — user call.
- `v3_ensemble_3d.joblib` (13-06, old T+3) unused (serve=T5) → deletable.
- Scratch `models/saved/ab_{baseline,macro}.joblib` (~26MB A/B) → deletable.
- VERIFY orphaned `194525f` hub tests on main: `tests/test_{vn_cost_model,triple_barrier,tabular_ensemble,run_backtest_wiring}.py`.

## ENV GOTCHAS (CRITICAL)
- git-bash BROKEN → **PowerShell only**.
- python = `C:\Users\caokh\AppData\Local\Programs\Python\Python311\python.exe` (polars/ML+pytest). conda env has NO polars. **nested `powershell -File` grabs WRONG conda python** → always explicit path.
- long bg jobs **DIE on laptop sleep** (Balanced) → wake-lock `SetThreadExecutionState([uint32]2147483649)` + keep plugged/awake.
- code-review-graph post-commit hook → cp1252 `UnicodeEncodeError` = **COSMETIC, commit succeeds**.
- subagents only Bash (broken) → CANT run python/train; **ORCHESTRATOR runs+commits**.
- prefix heavy runs: `$env:PYTHONIOENCODING="utf-8"`.

## KEY CMDS
- tests: `python -m pytest -q`
- retrain H+save serve: `python train_models.py --tb-horizon H --out models/saved/ckpt.joblib` then `python run_backtest.py --checkpoint models/saved/ckpt.joblib` → writes `v3_ensemble_Hd.joblib` (backs up old). `--use-macro-features`=macro arm; `--no-save --sweep-thresholds X`=pure eval.
- macro crawl: `python main.py --task crawl_macro`
- live EOD (broadcasts!): `python main.py --task full_pipeline --force-crawl --days-back 1`
