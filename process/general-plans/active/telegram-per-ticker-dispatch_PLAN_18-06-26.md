# telegram-per-ticker-dispatch_PLAN_18-06-26.md

**Date:** 2026-06-18
**Status:** ACTIVE
**Goal:** Fix raw-HTML-tag rendering and parse errors on the interactive Telegram reply path by sending one message per ticker, hardening the splitter fallback, and applying consistent URL escaping in two builders.
**Complexity:** MEDIUM — multi-file serve-path change (telegram_bot.py + builders.py) in a known fragile area (HTML mode, 4096-char limit); no model/trading logic change; 4 call sites to repoint.

---

## 1. Problem Statement

Three distinct bugs share the same root cause (combined HTML string + brittle splitter):

| Bug | User-visible symptom | Root cause |
|---|---|---|
| Raw HTML tags in reply | `<a href="...">` rendered verbatim | `_split_html_report` hard-slices at char 4000 mid-`<a href=...>` (telegram_bot.py:168-170) → Telegram `BadRequest` → fallback retry at line 254-255 re-sends without `parse_mode`, showing tags |
| Single oversized block truncates HTML | Broken `<b>` in message | Same hard slice at telegram_bot.py:169 |
| Non-clickable URLs in verify + sell/hold reports | Plain-text URLs in `_build_verify_report` and `_build_sell_hold_report` | Both use `f"  - {html.escape(u)}"` (builders.py:499) / `f"  • {html.escape(u)}"` (builders.py:404) instead of `format_source_links` |

---

## 2. Touchpoints Table

### Group A — Reply-path per-ticker sends (telegram_bot.py)

| Location | Current behavior | Change |
|---|---|---|
| `telegram_bot.py:147-175` `_split_html_report` | Called at 4 sites to chunk a combined string | Remains for defense-in-depth; hardened (see Group B). Also: the 4 call sites below stop feeding it a combined string — they call the new `_send_per_ticker_reports` helper instead |
| `telegram_bot.py:512` `_suggest_buy_dispatch` | `await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))` | Replace with `await _send_per_ticker_reports(update, wait_msg, signal_data_list)` — requires `daily_inference(broadcast=False)` to also return the raw `signal_data_list` (see section 2.1 below) |
| `telegram_bot.py:775` `suggest_sell_command` | same pattern | Replace with `await _send_per_ticker_reports(update, wait_msg, signal_data_list)` — requires `inference_for_holdings` to return `signal_data_list` (see section 2.1) |
| `telegram_bot.py:918` `verify_command` | same pattern | `/verify` always returns a single ticker; still use `_send_or_reply_chunks(update, wait_msg, [report_html])` — no per-ticker helper needed; benefit comes from URL escaping fix in `_build_verify_report` |
| `telegram_bot.py:1098` `_run_audit_command` | same pattern | `/audit_weekly` and `/audit_monthly` return post-mortem prose (not per-ticker signal cards) — keep `_split_html_report` + `_send_or_reply_chunks` with the hardened splitter; no per-ticker change needed |
| `telegram_bot.py:1182` `rebalance_command` | `await _send_or_reply_chunks(update, wait_msg, [report_html])` | Already passes a single-element list; no bundling issue; keep as-is |
| NEW: `telegram_bot.py` (new function ~line 193-area) | does not exist | Add `async def _send_per_ticker_reports(update, wait_msg, signal_data_list, ...)` — see design spec in section 3 |

#### 2.1 Return-value changes required in main.py callers

`daily_inference(broadcast=False)` currently returns `_build_combined_report(dispatched_signals)` (main.py:1448, 1450). The per-ticker helper in the bot needs the raw list, not the already-joined string.

**Change:** `daily_inference` returns `list[dict]` when `broadcast=False`. The bot side checks `isinstance(result, list)` and feeds it to `_send_per_ticker_reports`. The cron path (`broadcast=True`) discards the return value (main.py:1798) — no change needed there.

**Alternative (lower blast radius):** Return a `tuple[str, list[dict]]` where `str` is the legacy combined report (kept for backward compat) and `list[dict]` is the raw signals. The bot unpacks the tuple.

**Decision:** Use the tuple return. This is safer — any other consumer calling `daily_inference(broadcast=False)` that expects a `str` will break loudly rather than silently. The tuple approach lets us add `# type: ignore` at one unpacking site rather than auditing every consumer.

Similarly, `inference_for_holdings` (main.py:1457-onwards) returns an HTML string. It must also return the raw `list[dict]` used to build `_build_sell_hold_report`. Apply the same tuple return pattern: `tuple[str, list[dict]]`.

### Group B — Splitter hardening + fallback escape (telegram_bot.py)

| Location | Current behavior | Change |
|---|---|---|
| `telegram_bot.py:168-170` hard-slice branch | `for i in range(0, len(block), max_len): chunks.append(block[i:i+max_len])` | Replace with tag-safe split: scan backward from `block[max_len-1]` to find a newline boundary that is NOT inside `<...>`. If no safe boundary exists within the block, slice at the last whitespace before `max_len` that is outside a tag. As last resort, accept the hard cut but strip back to outside the last open `<` |
| `telegram_bot.py:253-255` `BadRequest` fallback | `await update.message.reply_text(chunk)` (no parse_mode, shows raw tags) | Change to `await update.message.reply_text(html.escape(chunk))` — degrades to readable escaped text, not raw HTML tags. Note: `html.escape` is already imported at line 27 |

**Tag-safe split algorithm (exact spec for executor):**

```
def _safe_split_block(block: str, max_len: int) -> list[str]:
    """Split a single oversized HTML block without cutting inside a tag.

    Strategy:
    1. Walk backward from position max_len-1.
    2. Track open-tag depth: if we are inside '<...>', keep walking back.
    3. Stop at the first newline character that is outside any tag.
    4. If no newline found within the block up to max_len, stop at the
       last whitespace outside a tag.
    5. If no whitespace found, stop at the last position BEFORE any '<'
       that would be split.
    6. Whatever boundary is found, recurse on the remainder.
    Returns a list of safe chunks all <= max_len.
    """
```

The existing 11 tests in `test_telegram_split.py` must still pass (all cover non-HTML plain-char blocks). The new HTML-validity test (step 6 in the test plan) will test the hardened path.

### Group C — URL escaping (src/reports/builders.py)

| Location | Current behavior | Change |
|---|---|---|
| `builders.py:402-404` `_build_sell_hold_report` url_lines block | `"\n".join(f"  • {html.escape(u)}" for u in source_urls)` — plain-text, non-clickable | Replace with `format_source_links(source_urls)` (already imported at builders.py:17). Remove the 3-line `if source_urls / else` guard — `format_source_links` handles empty list gracefully (returns "Nguồn tham khảo: chưa có liên kết."). Remove the `url_lines` variable. Update the f-string at builders.py:418 from `{url_lines}` to the inline `format_source_links(...)` call |
| `builders.py:497-499` `_build_verify_report` url_lines block | `"\n".join(f"  - {html.escape(u)}" for u in source_urls)` — same issue | Same fix: replace with `format_source_links(source_urls)`. The result renders as a single-line "Nguồn tham khảo: `<a href=...>Domain</a> · ...`" — consistent with the buy card. Update builders.py:518 from `{url_lines}` to `{format_source_links(source_urls)}` |
| `builders.py:326` hardcoded `5 ngày tới` | `f"   • <b>Đánh giá xu hướng (5 ngày tới):</b>"` | Bonus cleanup: replace literal `5` with `{SHORT_HORIZON_DAYS}` (already defined at builders.py:129). Not a bug, but noted in context as "fix pending in the Telegram-work effort." Include in this implementation step |

### Group D — Optional: notify_tranche_exits size guard (main.py)

| Location | Current behavior | Change |
|---|---|---|
| `main.py:1820-1832` `notify_tranche_exits` | Builds `"\n".join(lines)` → `send_text_alert(...)`. No size guard. Caller HTML-escapes ticker fields (line 1829). | Bounded in practice (one line per due ticker, typically ≤ 10 tickers). Risk: a very large tranche book could push >4096. Add a lightweight size check: if `len(msg) > 3800`, truncate the lines list and append an overflow notice. Keep it in scope — it is a one-liner guard, low risk |

---

## 3. Shared Helper Design — `_send_per_ticker_reports`

New async function in `telegram_bot.py`, placed just before `_send_or_reply_chunks` (around line 193).

**Signature:**
```python
async def _send_per_ticker_reports(
    update: Update,
    wait_msg: Any,
    signal_data_list: list[dict],
    *,
    disable_preview: bool = True,
) -> None:
```

**Behavior:**
1. If `signal_data_list` is empty: edit `wait_msg` to empty-results message and return.
2. Build a header message: `f"📊 {len(signal_data_list)} tín hiệu — chi tiết từng mã bên dưới."` — edit `wait_msg` in-place with parse_mode=HTML.
3. For each `sd` in `signal_data_list`: call `TelegramBot._build_message(sd)` to get the per-ticker HTML card (~600-900 chars, well below 4096), then send via `update.message.reply_text(card, parse_mode=ParseMode.HTML, disable_web_page_preview=disable_preview)`. Wrap each send in `try/except BadRequest` — on failure, retry with `html.escape(card)` (no parse_mode), same as the hardened fallback pattern.
4. Honor the oversight mirror: mirror each card to ADMIN_CHAT_ID if `_role_for(update) == "user"` (same condition as in `_send_or_reply_chunks:209`).
5. No rate-limit sleep here — PTB handles flood control via its own queue; the existing 0.5s sleep in `TelegramBot._dispatch` is only on the sync `requests.post` path, which is not used here.

**Import requirement:** `TelegramBot` is already importable from `src.utils.telegram_alerter` (currently imported in builders.py:17). It needs to be available in `telegram_bot.py` as well. Currently it is NOT imported in `telegram_bot.py` (the file uses PTB's `Application`, not the alerter `TelegramBot`). The executor must add: `from src.utils.telegram_alerter import TelegramBot as AlerterBot` (alias to avoid collision with PTB internals) or reference it directly. Check existing imports at telegram_bot.py:23-51 — `TelegramBot` is not present. Use `AlerterBot` alias.

---

## 4. Blast Radius

**In scope (changed):**
- `src/utils/telegram_bot.py` — `_split_html_report` (hardened), `_send_or_reply_chunks` fallback (line 254-255), new `_send_per_ticker_reports` helper, 2 of 4 `_send_or_reply_chunks` call sites (lines 512, 775)
- `src/reports/builders.py` — `_build_sell_hold_report` url_lines block (lines 402-404, 418), `_build_verify_report` url_lines block (lines 497-499, 518), `_build_fallback_observability_report_vi` line 326 (bonus)
- `main.py` — `run_trade_execution` return type (lines 1448, 1450 → tuple), `inference_for_holdings` return type (similar pattern), `notify_tranche_exits` size guard (line 1832)

**Explicitly NOT changed:**
- `TelegramBot._build_message` (telegram_alerter.py:116-212) — already correct; format_source_links already called at line 195
- `TelegramBot._dispatch` (telegram_alerter.py:218-241) — cron push path; already per-ticker; no bundling
- All model/ML/backtest/trading-decision code
- `_split_html_report` logic for the separator-split path (lines 157-174 non-hardslice branch) — keep unchanged
- `/news`, `/exits`, `/audit_weekly`, `/audit_monthly`, `/rebalance` send paths — no per-ticker signal_data involved
- `_chunk_lines` (telegram_bot.py:178-191) — used only by `/news`; unrelated

**Confirmed callers of the 4 `_split_html_report` sites:**
- Line 512: `_suggest_buy_dispatch` → called by `suggest_buy20_command` (line 516)
- Line 775: `suggest_sell_command` (standalone handler)
- Line 918: `verify_command` (standalone handler) — KEEP `_split_html_report`; single ticker, benefit is from builder fix
- Line 1098: `_run_audit_command` → called by `audit_weekly_command` (line 1119) and `audit_monthly_command` (line 1124) — KEEP `_split_html_report`; prose report, not signal cards

**Hub node impact:** `build_application` (telegram_bot.py, degree 72) is NOT changed — only handler implementations are changed. `daily_inference` (main.py, degree 84) has its return type changed when `broadcast=False`. Audit all callers of `daily_inference` before implementing.

---

## 5. Implementation Steps (ordered, each independently testable)

### Step 1 — URL escaping in 2 builders (builders.py)
**Files:** `src/reports/builders.py`
**Actions:**
1a. `_build_sell_hold_report` (line 402-418): remove the `if source_urls / else / url_lines` block (lines 402-406); replace `{url_lines}` at line 418 with `{format_source_links(source_urls)}`. The `source_urls` variable at line 402 stays; just remove the local formatter block.
1b. `_build_verify_report` (line 497-499): remove `url_lines` block (lines 497-501); replace `{url_lines}` at line 518 with `{format_source_links(source_urls)}`.
1c. (Bonus) `_build_fallback_observability_report_vi` (line 326): change `(5 ngày tới)` to `({SHORT_HORIZON_DAYS} ngày tới)`.
**Verify:** `pytest -q tests/test_main_logic.py tests/test_cards.py` — all existing tests pass; manually inspect generated HTML contains `<a href=` for URLs.

### Step 2 — Splitter tag-safe hardening + fallback escape (telegram_bot.py)
**Files:** `src/utils/telegram_bot.py`
**Actions:**
2a. Replace the hard-slice inner loop at lines 168-170 with `_safe_split_block(block, max_len)` helper. Implement `_safe_split_block` as a module-level function above `_split_html_report`. The algorithm walks backward from `max_len-1` finding the last newline that is not inside `<...>`. If none, walk back further to last whitespace. If none, accept the last char before an opening `<`. Return a list recursively.
2b. At line 254-255, change `await update.message.reply_text(chunk)` to `await update.message.reply_text(html.escape(chunk))`.
**Verify:** `pytest -q tests/test_telegram_split.py` — all 11 existing tests pass. New tests (Step 6a) added first if using TDD, otherwise add after and verify.

### Step 3 — New `_send_per_ticker_reports` helper (telegram_bot.py)
**Files:** `src/utils/telegram_bot.py`
**Actions:**
3a. Add `from src.utils.telegram_alerter import TelegramBot as AlerterBot` to imports (after line 51 import block). Verify no circular import (`telegram_alerter` does not import from `telegram_bot`).
3b. Implement `async def _send_per_ticker_reports(update, wait_msg, signal_data_list, *, disable_preview=True)` at a new location between `_chunk_lines` (line 191) and `_send_or_reply_chunks` (line 194). Full behavior per section 3.
**Verify:** No existing tests broken; new tests in Step 6b will cover this.

### Step 4 — Repoint 2 of 4 reply call sites (telegram_bot.py + main.py)
**Files:** `src/utils/telegram_bot.py`, `main.py`
**Actions:**
4a. **`daily_inference` return type** (main.py:1448, 1450): change from returning `_build_combined_report(dispatched_signals)` to returning `(str, list[dict])` tuple: `(_build_combined_report(dispatched_signals), dispatched_signals)`. Add `# type: ignore[return-value]` or update the return annotation. Search `daily_inference` calls in `telegram_bot.py:472` (the `asyncio.to_thread` call) — update the unpack: `result = await asyncio.to_thread(...)`, then `report_html, signal_data_list = result if isinstance(result, tuple) else (result, [])`.
4b. **`inference_for_holdings` return type** (main.py): locate the final `return _build_sell_hold_report(...)` call; change to return a tuple of `(html_str, signal_data_for_holdings)`. Update the `suggest_sell_command` handler at `telegram_bot.py:758` to unpack similarly.
4c. **`_suggest_buy_dispatch` line 512**: after unpacking `signal_data_list` in 4a, replace `await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))` with `await _send_per_ticker_reports(update, wait_msg, signal_data_list)`. Keep the `if not report_html` guard (lines 502-508) but also check `if not signal_data_list`.
4d. **`suggest_sell_command` line 775**: after unpacking `signal_data_list` in 4b, replace `await _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))` with `await _send_per_ticker_reports(update, wait_msg, signal_data_list)`.
4e. **`verify_command` line 918**: NO change to the call site itself. The benefit for `/verify` is entirely from Step 1 (URL escaping in `_build_verify_report`). `_split_html_report` is appropriate here (single-ticker report, prose format, may use `══════` separator for sections).
4f. **`_run_audit_command` line 1098**: NO change. Post-mortem prose, not signal cards.
**Verify:** Full suite `pytest -q` — 238+ tests pass.

### Step 5 — notify_tranche_exits size guard (main.py)
**Files:** `main.py`
**Actions:**
5a. In `notify_tranche_exits` (line 1820-1832), before `TelegramBot().send_text_alert(...)`, compute `msg = "\n".join(lines)`. If `len(msg) > 3800`, truncate: keep header + as many ticker lines as fit under 3800, then append `f"\n... và {remaining} mã khác"`. Send the truncated `msg`.
**Verify:** `pytest -q` — no existing tests for this function (it's untested). No new test required per scope — the guard is a one-liner safety net. Document as known-gap.

### Step 6 — Tests
**Files:** `tests/test_telegram_split.py` (extend), new `tests/test_per_ticker_dispatch.py`, extend `tests/test_main_logic.py` or `tests/test_cards.py`
**Actions:**
6a. **HTML-validity test for splitter** in `tests/test_telegram_split.py`:
- Add `test_oversized_html_block_no_mid_tag_cut`: create a block containing `<a href="https://example.com">VnExpress</a>` text repeated to exceed 4000 chars. Assert that no chunk ends with a partial `<a` tag (regex: `r"<[^>]*$"` must not match end of any chunk). Assert all chunks are valid (each open tag has a matching close within the chunk — use a simple regex counter for `<a` vs `</a>`).
- Add `test_fallback_sends_escaped_text` (integration-level, if mocking is feasible): simulate a `BadRequest` on `reply_text`; assert the retry call uses `html.escape(chunk)` — check that `<` in the chunk becomes `&lt;` in the retry argument.
6b. **Per-ticker dispatch tests** in new `tests/test_per_ticker_dispatch.py`:
- `test_send_per_ticker_reports_calls_reply_n_times`: mock `update.message.reply_text`; pass N signal dicts; assert `reply_text` called N+1 times (header edit + N ticker cards).
- `test_send_per_ticker_reports_empty_list`: mock; assert `wait_msg.edit_text` called once with empty-signal message; assert `reply_text` NOT called.
- `test_send_per_ticker_reports_badrequest_fallback`: mock `reply_text` to raise `BadRequest` on first call; assert second call has escaped content.
6c. **URL-escaping tests** (add to `tests/test_main_logic.py` or `tests/test_cards.py`):
- `test_verify_report_url_is_clickable_link`: call `_build_verify_report` with `sentiment={"source_urls": ["https://example.com/q?a=1&b=2"], ...}`; assert `<a href="https://example.com/q?a=1&amp;b=2">` is in output (ampersand attribute-escaped) and `</a>` follows.
- `test_sell_hold_report_url_is_clickable_link`: call `_build_sell_hold_report` with a URL containing `&`; assert same pattern.
- `test_verify_report_empty_urls_graceful`: pass empty `source_urls`; assert `chưa có liên kết` in output.
**Run commands:**
```
pytest -q tests/test_telegram_split.py
pytest -q tests/test_per_ticker_dispatch.py
pytest -q tests/test_main_logic.py tests/test_cards.py
pytest -q   # full suite baseline: 238 → should be 238+new_count, all green
```

---

## 6. Data Flow (after this plan)

```
/suggest_buy20 command
  → _suggest_buy_dispatch(update, horizon=20)
  → asyncio.to_thread(daily_inference, broadcast=False, horizon=20)
      → returns (combined_html: str, signal_data_list: list[dict])
  → unpack: report_html, signal_data_list
  → if not signal_data_list: edit wait_msg "no signals"
  → _send_per_ticker_reports(update, wait_msg, signal_data_list)
      → edit wait_msg → header msg ("N tín hiệu...")
      → for sd in signal_data_list:
          card = AlerterBot._build_message(sd)  # ~600-900 chars, pure HTML
          reply_text(card, parse_mode=HTML)     # well below 4096
          except BadRequest: reply_text(html.escape(card))  # readable fallback

/suggest_sell command
  → suggest_sell_command → asyncio.to_thread(inference_for_holdings, ...)
  → returns (sell_html: str, signal_data_for_holdings: list[dict])
  → _send_per_ticker_reports(update, wait_msg, signal_data_for_holdings)
      [same pattern as above]

/verify HPG command
  → verify_command → asyncio.to_thread(verify_single_ticker, ticker)
  → returns single report_html (str)
  → _send_or_reply_chunks(update, wait_msg, _split_html_report(report_html))
      [splitter now tag-safe; fallback now html.escape(chunk)]
      [report_html uses format_source_links → clickable URLs]
```

---

## 7. Failure Modes and Mitigations

| Failure | Mitigation |
|---|---|
| `daily_inference` return type change breaks an unknown caller | Search ALL callers of `daily_inference(broadcast=False)` in the repo before implementing step 4a; the tuple return is backward-compat if unpacked with `isinstance` guard |
| `_safe_split_block` infinite loops on pathological input (all `<`) | Add a hard fallback: if backward scan exceeds 200 chars with no safe boundary, accept the raw char boundary — worst case is still the old behavior, not an infinite loop |
| Per-ticker send rate triggers Telegram flood control | PTB's `Application` handles `RetryAfter` exceptions via its built-in retry mechanism; each card is ~700 chars and a single send, well within Telegram limits |
| `AlerterBot._build_message` raises on malformed signal_data | Wrap in `try/except Exception` in `_send_per_ticker_reports`; log and skip that ticker; notify user with a single warning line |
| `inference_for_holdings` tuple return breaks existing tests | `tests/test_main_logic.py` imports `inference_for_holdings` — check and update any test assertions on its return type |
| Oversight mirror in `_send_per_ticker_reports` adds latency | Mirror is best-effort with `except Exception` swallow — same as the existing pattern in `_send_or_reply_chunks:224` |

---

## 8. Dependencies and Sequencing

- Steps 1, 2, 3 are independent and can be implemented in any order or in parallel.
- Step 4 depends on Step 3 (needs the helper to exist before repointing call sites).
- Step 4 also depends on verifying all callers of `daily_inference(broadcast=False)` and `inference_for_holdings` before changing their return types.
- Step 5 is independent of all other steps.
- Step 6 (tests) should be written for Steps 1-3 before Step 4 to catch regressions early; Step 4 tests can be added after Step 4.

**External blocker:** Confirm there are no callers of `daily_inference(broadcast=False)` outside `telegram_bot.py` before changing its return type. Run grep: `grep -n "daily_inference" main.py run_bot.py`.

---

## 9. Backwards Compatibility

- The cron path (`daily_inference(broadcast=True)`) return value is discarded at `main.py:1798` — changing the return type does NOT affect the cron path.
- `_build_combined_report` is not removed (still referenced by tests in `test_main_logic.py`). Keep it; it is still called inside `run_trade_execution` to produce the tuple's `str` component.
- `_split_html_report` is not removed — still used by `/verify` and `/audit` paths and covered by existing tests.
- `format_source_links` signature unchanged.

---

## 10. Acceptance Criteria (all must be true for DONE)

1. `pytest -q` exits 0, count >= 238 (no regressions; new tests add to count).
2. `test_oversized_html_block_no_mid_tag_cut` passes: no chunk produced by `_split_html_report` ends inside an unclosed HTML tag.
3. `test_send_per_ticker_reports_calls_reply_n_times` passes: N signal dicts → exactly N+1 Telegram sends (header + N cards).
4. `test_verify_report_url_is_clickable_link` passes: `_build_verify_report` output contains `<a href="...">` with `&amp;`-escaped query params.
5. `test_sell_hold_report_url_is_clickable_link` passes: same for `_build_sell_hold_report`.
6. Manual check (or log inspection): a simulated `/suggest_buy20` with 3 tickers produces 4 sends (1 header + 3 cards), each < 4096 chars, no `BadRequest` logged.
7. Manual check: a simulated `/verify HPG` reply with a URL containing `&` shows a clickable link, not plain text.

---

## 11. Out of Scope

- `TelegramBot._build_message` (telegram_alerter.py:116-212) — already correct; not touched.
- The cron push per-ticker loop in `_dispatch_signals` (main.py:1228-1305) — already sends one msg/ticker; not the bug path.
- Any MarkdownV2 migration — stay in HTML mode; codebase is HTML-only.
- Model, backtest, trading-decision, or sizing logic — zero changes.
- `_chunk_lines` (telegram_bot.py:178-191) — `/news` only; no bundling issue.
- Adding sentiment paper-log fields to the per-ticker cards — separate concern.
- The `builders.py:326` `SHORT_HORIZON_DAYS` fix is listed as a bonus in Step 1c; if scope needs trimming, drop it without affecting the three primary goals.

---

## 12. Verification Evidence Checklist

- [ ] `pytest -q` 0 failures, count >= 238
- [ ] `pytest -q tests/test_telegram_split.py` — all 11 + new HTML-validity test pass
- [ ] `pytest -q tests/test_per_ticker_dispatch.py` — all new per-ticker tests pass
- [ ] `pytest -q tests/test_main_logic.py tests/test_cards.py` — URL-escaping tests pass
- [ ] Grep confirms no unclosed `<a` in `_split_html_report` output for test input with embedded tags
- [ ] Grep confirms `format_source_links` is used in `_build_verify_report` output (not `html.escape(u)` for URLs)
- [ ] Grep confirms `format_source_links` is used in `_build_sell_hold_report` output
- [ ] `notify_tranche_exits` size guard present and covered by a comment explaining truncation logic
- [ ] No `isinstance(result, str)` assumption on `daily_inference(broadcast=False)` return in callers outside `telegram_bot.py`

---

## 13. Resume and Execution Handoff

**Plan file:** `process/general-plans/active/telegram-per-ticker-dispatch_PLAN_18-06-26.md`

**Execution order:**
1. Step 1 (URL escaping — lowest risk, pure builder functions, existing test coverage)
2. Step 2 (splitter hardening — existing 11 tests protect against regression)
3. Step 3 (new helper — no call sites yet, safe to add)
4. Step 6 partial (write tests for Steps 1-3 before proceeding)
5. Step 4 (repoint call sites — highest blast radius; do last when helpers and tests are stable)
6. Step 5 (notify guard — independent, low risk, any time)
7. Step 6 final (complete remaining tests)

**Pre-flight for executor:**
- Run `pytest -q` and confirm 238 green before touching any file.
- Run `grep -n "daily_inference" telegram_bot.py run_bot.py main.py` to enumerate all call sites before changing return type.
- Run `grep -n "inference_for_holdings" telegram_bot.py main.py` similarly.
- Confirm `telegram_alerter.TelegramBot` is NOT imported in `telegram_bot.py` before Step 3 (to avoid collision surprises).

**Plan validator:** `node .claude/skills/vc-generate-plan/scripts/validate-plan-artifact.mjs process/general-plans/active/telegram-per-ticker-dispatch_PLAN_18-06-26.md`
