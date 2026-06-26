# SESSION HANDOFF — 2026-06-26 (Quant Engine V4)

## STATE NOW
- Branch **`main`** @ `ff15746`, **pushed to origin** (github.com/cryborg1211/quant_trading_telegram_bot).
- Working tree: **UNCOMMITTED** = `scripts/benchmark_defense_layers.py` (new, import-verified, NOT yet committed — running from working tree).
- **Benchmark `btb0vjrxf` DIED** (laptop power-off again). 3/7 arms cached + durable in
  `process/features/macro-integration/reports/defense_benchmark_s0_h5.json`. **Resume to finish.**
- Multi-seed brake sweep ABANDONED: seed-1 has 2/9 cells on disk (`garch_hmm_brake_sweep_s1_h5.csv`), seeds 2–3 not started. Killed on purpose (laptop too flaky for 3.5hr).

## DONE THIS SESSION (committed + pushed)
GARCH-HMM regime overlay — full arc, all on main:
- `f5f7d37` core overlay (log-vol space, z-score, degenerate rejection).
- `f9c7e53` **persistence guard** (cap α+β ≤ 0.96, project + preserve σ²_unc → escapes IGARCH trap) + **linear `exposure_scaler` = clip(P(Bull), 0.2, 1.0)** (replaces binary cash-out).
- `8b11aa2` **resumable sweep + constant-exposure control arm**. KEY RESULT (seed 0, T+5, 915d bear OOS): **TIMING ADDS** — median timing_α=+0.22, 9/9 cells beat their matched flat-leverage control. Flat-leverage Sharpe (−0.359) == baseline (−0.356) → constant leverage is Sharpe-neutral → the brake's −0.356→−0.108 Sharpe gain is GENUINE REGIME TIMING, not de-leverage. (MaxDD gain IS mostly de-leverage; still negative absolute → timed loss-mitigation, NOT alpha.)
- `ff15746` per-seed grid CSV (multi-seed safe).
- `836cbc7` dashboard **dark-premium theme** (`dashboard/theme.py` + `.streamlit/config.toml`).
- `cf9a83a` **GIỮ-tab crash fix** — `_parse_price` in `headless.py` tolerates legacy bot-era TEXT prices (`'47,800 VND'`). +13 tests.
- README.md (GitHub) + Changelog + all-context all updated earlier.
- Test suite: **408 passed**.

## ⚠️ GARCH-HMM IS NOT WIRED TO SERVE
`garch_hmm` appears in only 6 files: the model, trainer, tests, 3 scripts. **ZERO refs in `main.py` / `dashboard/` / `src/bot/` / `src/trading/`.** It is pure research. Using it live needs real integration (load weights, apply scaler in `main._dispatch_signals`, config flag + kill-switch, AND resolve triple-regime compounding with the already-ON `regime_policy`). Recommendation: do NOT default-on. Single-seed, bear-OOS, loss-mitigation only, system is paper-only anyway.

## BENCHMARK — partial result (3/7 arms, resume to finish)
`scripts/benchmark_defense_layers.py` — head-to-head of defensive overlays on ONE walk-forward (seed 0), + flat-leverage TIMING control per soft arm. Pure evaluation → pick ONE fixed winner (NOT a dynamic selector = lookahead).
Cached so far:
- `baseline`        Sharpe −0.364  MaxDD −54.99%  Ret −38.5%
- `regime_policy`   Sharpe **−0.290**  MaxDD −38.8%  Ret −22.2%  ← best so far (= current serve default)
- `macro_hmm`       Sharpe −0.779  ← **HURTS** (worse than baseline; macro 2-state HMM is bad here)
- pending: `garch_hmm`, `macro+garch_min`, `regime+garch`, `all_min` + controls.
Note: in the floor-sweep, garch_hmm@floor0.2 was Sharpe −0.137 (beats regime_policy −0.290) — so garch may win once it finishes. CONFIRM by resuming.
**Resume:** `python scripts/benchmark_defense_layers.py --floor 0.2 --seed-idx 0` (skips the 3 cached arms, runs the rest, prints ranked table + BEST).

## NEXT
1. **Resume the benchmark** (one command above, ~50min left). Get the ranked table → decide the single defense layer (or min-combine) to deploy as a FIXED default.
2. **Commit `benchmark_defense_layers.py`** once the run confirms end-to-end (+ its result json).
3. THEN (user's stated next task): **audit the audit system.**

## AUDIT SYSTEM (user auditing next — pointers)
- `src/utils/audit_evaluator.py` → `run_post_mortem` — trade audit / post-mortem evaluation.
- `dashboard/tabs/audit.py` → `_cached_postmortem(uid, days)` — Audit tab (HTML via st.markdown).
- Serve namespace: `LOCAL_USER_ID = "local"` (dashboard) vs bot/cron ids — audit rows are user-id scoped; bot-era rows under telegram id won't show under "local".
- Start by mapping what `run_post_mortem` actually computes vs what the Audit tab renders.

## ENV GOTCHAS (CRITICAL — cost the whole session)
- **LAPTOP DIES CONSTANTLY** (power-off + RAM exhaustion from other tasks). Every long run died ≥once. ALL heavy scripts are now **resumable** (per-cell/per-arm cache) — relaunch the same command to continue. Keep plugged + awake + close RAM-heavy apps.
- git-bash BROKEN → **PowerShell only** for git/python/pytest. Bash tool fails on quotes.
- python = `C:\Users\caokh\AppData\Local\Programs\Python\Python311\python.exe` (has polars/ML/pytest). conda has NO polars. **Always explicit path.**
- Prefix heavy runs: `$env:PYTHONIOENCODING="utf-8"`.
- code-review-graph post-commit hook → cp1252 `UnicodeEncodeError` = **COSMETIC, commit succeeds.**
- Disable sleep before long runs: `powercfg /change standby-timeout-ac 0; powercfg /change standby-timeout-dc 0`.
- Subagents only get (broken) Bash → can't run python. **Orchestrator runs + commits.**

## KEY CMDS
- tests: `python -m pytest -q` (408 pass)
- resume benchmark: `python scripts/benchmark_defense_layers.py --floor 0.2 --seed-idx 0`
- resume brake sweep (any seed): `python scripts/sweep_garch_hmm_brake.py --floors 0.1,0.2,0.3 --caps 0.94,0.96,0.98 --seed-idx N`
- A/B one config: `python scripts/validate_garch_hmm_brake.py --min-exposure 0.2 --max-exposure 1.0`
- train garch overlay: `python train_macro_regime.py --n-states 3`
