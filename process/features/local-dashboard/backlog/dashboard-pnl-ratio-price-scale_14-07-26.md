# Dashboard `_pnl_ratio` Price-Scale Latent Bug

- **Date flagged:** 14-07-26 (discovered 13-07-26 during the Portfolio Guard PLAN investigation)
- **Priority:** LOW-MEDIUM — no confirmed production incident yet (single-user local dashboard), but a real misfire would be silently wrong (no crash, no alert) if it ever triggers
- **Status:** BACKLOG — confirmed latent-bug candidate, not yet fixed
- **Source:** `process/general-plans/active/portfolio-guard_PLAN_13-07-26.md`, "Critical Investigation Finding: Price-Scale Ambiguity" section
- **Note:** a session-level background-task chip already existed for this bug in the originating agent session, but chips do not persist across app restarts — this file is the durable, repo-tracked record.

## Problem

`portfolio.price` (a human user's `/add`-inserted entry price, `DOUBLE` column, `src/data/db_engine.py:287-293`) has two conflicting scale assumptions live in the codebase simultaneously:

1. `src/utils/telegram_bot.py::add_portfolio_command` does a bare `float(raw_price)` cast — no scale normalization. Its own `_BOT_COMMANDS` help example is `/add VNE 1000 32.5`, which is only economically sane as **thousands-of-VND** (32,500 VND) — no HOSE stock trades at an absolute 32.5 VND.
2. `dashboard/utils/headless.py::_pnl_ratio` explicitly asserts the **opposite** in its own docstring — "the `portfolio` table stores absolute-VND entry prices" — and only rescales the *current-close* side (`if current_close < 1000: current_close *= 1000`), trusting `entry_price` unconditionally as already-absolute.

If a user ever followed the bot's own documented `/add` example literally (entering a thousands-scale price like `32.5`), the dashboard's GIỮ tab computes a wildly wrong PnL ratio: `(32500 − 32.5) / 32.5 ≈ +99,900%`.

## Root cause

Two independent code paths (`telegram_bot.py` and `dashboard/utils/headless.py`) were written at different times with different unstated assumptions about `portfolio.price`'s unit, and neither validates the unit at insert time. No existing test catches the mismatch because each path's tests only exercise its own assumption.

## Evidence / precedent

`main.py` already has a named canonical fix for exactly this class of ambiguity: `main._VN_PRICE_SCALE_THRESHOLD = 1_000.0` (`main.py:89`), used by `_get_live_exec_prices`. The Portfolio Guard feature (13-07-26, shipped 14-07-26) adopted the identical rule for its own read of `portfolio.price` — see `src/trading/portfolio_guard.py::normalize_entry_price_vnd` — but deliberately did NOT touch `dashboard/utils/headless.py` (scope discipline; the portfolio-guard plan explicitly flagged this as out-of-scope rather than silently fixing or silently ignoring it).

## Fix options

1. **Reuse `src.trading.portfolio_guard.normalize_entry_price_vnd` (now shipped) directly on `headless.py::_pnl_ratio`'s `entry_price` side**, instead of duplicating the `1_000.0` threshold rule inline a third time. Cheapest fix; consistent with the now-twice-precedented heuristic; inherits the same rare-penny-stock-under-1000-VND edge case already accepted elsewhere in the codebase. `dashboard/` importing from `src.trading` would be a new cross-package import direction for `dashboard/` — worth a quick check that nothing in this repo's conventions constrains that (nothing found so far), before landing.
2. **Enforce/validate thousands-VND at input time** in both `add_portfolio_command` and `dashboard/utils/headless.py::portfolio_add`, so `portfolio.price` has one guaranteed unit going forward — fixes the root cause, not just the read side; touches two live write paths.
3. **Store an explicit unit tag** alongside `portfolio.price` (schema change) — most robust, highest effort; likely overkill for a single-user local dashboard.

No fix applied yet. Recommend option 1 as the minimal, precedent-consistent fix; option 2 as a more durable follow-up if `/add` input volume grows.

## Impact if unfixed

Only affects the Streamlit dashboard's GIỮ tab PnL display (`dashboard/utils/headless.py::_pnl_ratio`) for a user who entered a thousands-scale `/add` price. Portfolio Guard and all Telegram-bot-side PnL math are unaffected — they use the corrected `normalize_entry_price_vnd` / `main._VN_PRICE_SCALE_THRESHOLD` paths respectively. Not confirmed to have actually misfired for any real user row — flagged proactively, not reactively.
