"""Background thread runner (P1 stub).

In P2 this module will hold helpers that run heavy inference off the Streamlit
script thread (so the UI does not freeze), combined with ``st.cache_data`` for
result memoization and a spinner overlay.

P1: stub only. ``run_in_thread`` raises NotImplementedError so accidental
wiring fails loudly. No heavy imports.
"""

from __future__ import annotations

from typing import Any, Callable


def run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """P2: run ``fn(*args, **kwargs)`` on a background thread and return result.

    Will be paired with ``st.cache_data`` and a spinner so heavy inference does
    not block the Streamlit render thread.
    """
    raise NotImplementedError("run_in_thread is wired in P2.")
