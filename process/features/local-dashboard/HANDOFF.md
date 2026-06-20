# Local Dashboard — Handoff (resume next session)

Last: 2026-06-19. HEAD `75bfad5` (main).

## Program
Local Streamlit dashboard, packaged for ANOTHER user's laptop. Action-split nav:
MUA / GIỮ / BÁN / Verify / Audit / Settings. Full local compute. User imports OWN
Gemini key. Telegram **send-only** (shared token, no polling — multi-poller = 409).
Launcher builds venv on first run (NO PyInstaller). `setup.exe` = Inno Setup installer.

## Phase state
- **P0** reuse audit ✓ → `process/features/local-dashboard/reports/p0-reuse-audit-findings_19-06-26.md`
- **P1** UI skeleton ✓ → `dashboard/` package, 6 tabs (was stub)
- **P2** logic linking ✓ (code-verified, suite 249 green, py_compile clean) — persist gate + tabs wired live
- **P2 GATE OPEN:** live `streamlit run` smoke NOT done (streamlit not installed)
- **P3** launcher / **P4** installer / **P5** hardening — NOT planned (author after P2 smoke)

Plans: `process/features/local-dashboard/active/` — umbrella, p0, p1, p2 PLANs.

## Locked decisions
- **persist gate**: `run_trade_execution`/`daily_inference` `persist=True` default (cron unchanged); dashboard calls `daily_inference(broadcast=False, persist=False)` → preview, ZERO DB writes.
- **LOCAL_USER_ID = "local"** (fresh namespace; bot-era audit rows under telegram-id won't appear; overridable via settings.json `dashboard_user_id`).
- MUA/GIỮ = structured data; Verify/BÁN/Audit = HTML via `st.markdown(unsafe_allow_html=True)`.
- Threading: `dashboard/utils/thread_runner.run_in_thread` (ThreadPoolExecutor + session_state + spinner).

## NEXT (do first)
1. **Live smoke**: `pip install -r requirements_dashboard.txt` → `streamlit run dashboard/app.py`. Verify: 6 tabs render; GIỮ add/remove round-trip; MUA preview leaves `portfolio` `user_id='cron'` row count unchanged; Verify push needs `.env` keys.
2. Then author + execute **P3** (`run_dashboard.bat` launcher + first-run venv bootstrap).

## ENV GOTCHAS (cost time last session)
- **git-bash BROKEN this box** (profile ~line 180 unterminated quote) → EVERY Bash-tool call fails. Use **PowerShell** for git/pytest/python.
- **pytest runner = bare `pytest`** (Python311 on PATH, has deps). conda env `stock` has NO pytest. streamlit NOT installed.
- **Subagents (execute/git-manager) only have Bash → cannot verify/commit.** Orchestrator must verify + commit via PowerShell.
- code-review-graph post-commit hook throws cp1252 `UnicodeEncodeError` — cosmetic, commit succeeds.

## Other session work (committed earlier, separate from dashboard)
- Telegram per-ticker dispatch + tag-safe splitter + safe URLs (`5546c16`,`200f83a`).
- `/suggest_buy5` restored; horizon-aware short-horizon hold display + sizing (`disp_hold`, weight `1/(disp_hold*picks)` cap 0.20).
- **LIVE BOT (`run_bot.py`) runs OLD code → restart to pick up telegram fix + /suggest_buy5.**

## Session commits
`5546c16` `200f83a` `74546ef` `fa98def` `a634ad1` `40bb482` `75bfad5`.
Working tree: pre-existing `AGENTS.md`/`CLAUDE.md`/`.claude/*`/`.cursor` etc uncommitted — NOT this session's, leave.
