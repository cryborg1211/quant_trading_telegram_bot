"""Background thread runner (P2).

Runs heavy inference off the Streamlit script thread so the UI does not freeze,
with a spinner overlay and a TTL result cache keyed in ``st.session_state``.

Streamlit re-runs the whole script top-to-bottom on every interaction. The
pattern here is rerun-friendly:

  1. First call for a given (fn, args): submit to the pool, store the future in
     session_state, render a spinner, then ``st.stop()`` (ends this rerun).
  2. Streamlit re-runs (auto-refreshed by the spinner); if the future is still
     running, render the spinner again and ``st.stop()``.
  3. Once the future is done: call ``future.result()`` (re-raises any exception
     so the tab error boundary catches it), cache the result + a timestamp, and
     return it.
  4. Within ``ttl`` seconds, a repeat call returns the cached result without
     re-submitting.

``streamlit`` is pinned ``>=1.35`` (``requirements_dashboard.txt``), so
``st.status`` (added in 1.28) is always available. We use ``st.spinner`` here
because the heavy work runs on another thread — ``st.spinner`` is a simple
context indicator and renders correctly across the submit→poll reruns.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import streamlit as st

# Small shared pool. max_workers=2 lets one tab's inference run while a cached
# read on another tab proceeds; DuckDBEngine hands out a process singleton conn
# so we never hold a pool-scoped connection across reruns.
_executor = ThreadPoolExecutor(max_workers=2)


def _cache_key(fn: Callable[..., Any], args: tuple[Any, ...]) -> str:
    """Stable session_state key for a (fn, args) pair."""
    digest = hashlib.md5(repr(args).encode("utf-8")).hexdigest()[:8]
    return f"_thread_{fn.__name__}_{digest}"


def _request_key(name: str) -> str:
    """Session_state flag key for a tab's load-gate."""
    return f"_load_requested_{name}"


def load_gate(
    name: str,
    *,
    prompt: str,
    button_label: str = "Tải dữ liệu",
) -> bool:
    """Defer heavy work until the user explicitly asks for it.

    Streamlit executes every tab body on every rerun, so an unconditional
    ``run_in_thread`` call fires its inference the instant the app opens — even
    when the user only wanted another tab. This gate renders a prompt + button
    and returns ``False`` until clicked; once clicked it latches ``True`` for the
    session so cached results keep rendering across reruns.
    """
    if st.session_state.get(_request_key(name)):
        return True
    st.info(prompt)
    if st.button(button_label, key=f"loadbtn_{name}", use_container_width=True):
        st.session_state[_request_key(name)] = True
        st.rerun()
    return False


def clear_cached(fn: Callable[..., Any], *args: Any) -> None:
    """Drop the cached result/timestamp/future for one (fn, args) pair.

    Lets a tab's "refresh" button force a fresh background run on the next rerun.
    """
    key = _cache_key(fn, args)
    for suffix in ("", "_ts", "_fut"):
        st.session_state.pop(key + suffix, None)


def run_in_thread(
    fn: Callable[..., Any],
    *args: Any,
    label: str = "Đang xử lý...",
    ttl: int | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``fn(*args, **kwargs)`` on a background thread; return its result.

    Reruns the Streamlit script while the work is in flight (showing ``label``
    in a spinner). Caches the result in ``st.session_state`` for ``ttl`` seconds
    when ``ttl`` is set. Any exception raised inside ``fn`` propagates out of
    this call (caught by the calling tab's error boundary).
    """
    key = _cache_key(fn, args)
    res_key = key
    ts_key = key + "_ts"
    fut_key = key + "_fut"

    # 1. Fresh cached result within TTL → return immediately, no resubmit.
    if ttl is not None and ts_key in st.session_state:
        age = time.time() - float(st.session_state[ts_key])
        if age < ttl and res_key in st.session_state:
            return st.session_state[res_key]

    fut: Future | None = st.session_state.get(fut_key)

    # 2. No in-flight future → submit one.
    if fut is None:
        fut = _executor.submit(fn, *args, **kwargs)
        st.session_state[fut_key] = fut

    # 3. Still running → spinner + stop this rerun (re-runs on next tick).
    if not fut.done():
        with st.spinner(label):
            # Brief sleep yields the GIL and paces the polling reruns so the
            # spinner is visible without a busy-loop.
            time.sleep(0.5)
        st.rerun()

    # 4. Done → collect result, clear the future, cache, and return.
    del st.session_state[fut_key]
    try:
        result = fut.result()  # re-raises any exception from the worker
    except Exception:
        # Drop any stale cached value so the next interaction retries cleanly.
        st.session_state.pop(res_key, None)
        st.session_state.pop(ts_key, None)
        raise

    st.session_state[res_key] = result
    st.session_state[ts_key] = time.time()
    return result
