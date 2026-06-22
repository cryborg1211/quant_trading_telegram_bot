# Audit: Paperlog Temporal Backfill Flaw

Date: 2026-06-22
Scope: `main.py::_backfill_paperlog_outcomes` and related tests

**Status: RESOLVED 2026-06-22.** Progressive per-horizon backfill implemented in
`main.py::_backfill_paperlog_outcomes` (short scan gate `_PAPERLOG_SHORT_MATURE_DAYS=4`,
decoupled `ret_3d`/`ret_20d` writes, `outcome_filled` flips only at T+20). Tests updated
in `tests/test_sentiment_paperlog.py` (12 tests, +2 progressive). See §4 for the executed plan.

## 1/ Root Cause

The temporal starvation is caused by two tightly coupled behaviors:

1. SQL pending-row gate is tied to the **maximum horizon**
- In `main.py`, `_PAPERLOG_MATURE_DAYS` is hardcoded to `21`.
- Pending rows are fetched only when:
  - `outcome_filled = FALSE`
  - `log_date <= CURRENT_DATE - INTERVAL 21 DAY`
- This means short-horizon outcomes (T+3 / T+5 style) are not even scanned until day 21.

2. Python write path updates all horizons in one atomic pass
- The loop computes both `ret_3d` and `ret_20d` in the same iteration.
- A single UPDATE sets `entry_close`, `ret_3d`, `ret_20d`, and `outcome_filled = TRUE` together.
- There is no independent lifecycle for short-horizon maturity vs max-horizon maturity.

Net effect: the function is designed as "all-at-once at max horizon" rather than "progressive per horizon".

## 2/ Blast Radius (Exact Files/Lines)

### Affected implementation
- `main.py:653`
  - `_PAPERLOG_MATURE_DAYS: int = 21`
- `main.py:770-771`
  - Pending SQL filter:
    - `WHERE outcome_filled = FALSE`
    - `AND log_date <= CURRENT_DATE - INTERVAL {_PAPERLOG_MATURE_DAYS} DAY`
- `main.py:786-787`
  - Coupled return calculations in same loop:
    - `ret_3d = ...`
    - `ret_20d = ...`
- `main.py:793-796`
  - Single UPDATE writes both returns and flips `outcome_filled = TRUE`.

### Test suite encoding current behavior
- `tests/test_sentiment_paperlog.py:223`
  - Mature case explicitly assumes `> 21 days`.
- `tests/test_sentiment_paperlog.py:247`
  - Immature case explicitly assumes `< 21 days` and expects skip.
- `tests/test_sentiment_paperlog.py:236-243`
  - Validation pattern assumes both `ret_3d` and `ret_20d` are produced together in one backfill pass.

## 3/ Symptom

Why T+3 / T+5 outcomes look missing/starved downstream:

- Rows younger than 21 days never enter the pending set, even though short horizons are already mature.
- Therefore short-horizon metrics remain NULL/absent during the entire day-3..day-20 window.
- Any dashboard/report/query reading realized short-horizon performance sees delayed, sparse, or apparently missing outcomes.
- Data appears to arrive in jumps around max-horizon maturity, not continuously as short horizons mature.

## 4/ Execution Plan — IMPLEMENTED 2026-06-22

Steps below were executed (1–3 + 5 in code, 4 in tests):

1. Broaden pending scan window for progressive processing
- Replace 21-day pending gate with a short maturity threshold for earliest horizon availability.
- Suggested: scan rows with `log_date <= CURRENT_DATE - INTERVAL 4 DAY` (calendar buffer for T+3 maturity on trading calendars).
- Keep `outcome_filled = FALSE` as coarse pending selector.

2. Decouple horizon computations and writes
- Compute/write `ret_3d` (or short horizon) independently of `ret_20d`.
- If short horizon is mature and currently NULL:
  - update `entry_close` (if missing) + `ret_3d` only.
- If long horizon is mature and currently NULL:
  - update `ret_20d` only.

3. Flip completion flag only at max-horizon completeness
- Set `outcome_filled = TRUE` only when max horizon is mature and its target metric is successfully populated (e.g., `ret_20d` non-NULL, or explicit terminal condition).
- Keep row pending (`outcome_filled = FALSE`) while only partial horizons are available.

4. Add/adjust tests for progressive semantics
- Add test: day-5 row receives `ret_3d` while `ret_20d` remains NULL and `outcome_filled` stays FALSE.
- Add test: later run (post day-21) fills `ret_20d` and flips `outcome_filled` TRUE.
- Retain missing-parquet safety behavior (row retried later, no crash).

5. Optional observability hardening
- Add counters/logging for:
  - scanned pending rows
  - short-only fills
  - long-horizon fills
  - completion flips
- This helps verify starvation is resolved in production.
