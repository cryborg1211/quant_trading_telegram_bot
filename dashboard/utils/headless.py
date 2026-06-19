"""Headless inference wrappers (P1 stubs).

In P2 this module will hold thin wrappers that call the serve-path functions
in ``main.py`` with ``broadcast=False`` (and a preview-safe / persist=False
path per P0 GOTCHA 1) so the dashboard can render real signals WITHOUT
triggering Telegram sends or mutating the cron portfolio book.

P1: stubs only. Each function raises NotImplementedError so accidental wiring
fails loudly instead of silently returning fake data. No heavy imports here —
``main`` / ``src`` are intentionally NOT imported until P2.
"""

from __future__ import annotations


def daily_inference_headless(horizon: int) -> tuple[str, list[dict]]:
    """P2: read-only buy-signal inference for the MUA tab.

    Will wrap a persist=False variant of
    ``main.daily_inference(broadcast=False, horizon=horizon)`` and return
    ``(html, signal_data_list)``.
    """
    raise NotImplementedError("daily_inference_headless is wired in P2.")


def inference_for_holdings_headless(tickers: list[str]) -> str:
    """P2: sell/hold + rebalance inference for the BÁN tab.

    Will wrap ``main.inference_for_holdings(tickers, window_rows=120)`` and
    return the SELL/HOLD report HTML.
    """
    raise NotImplementedError("inference_for_holdings_headless is wired in P2.")


def verify_single_ticker_headless(ticker: str) -> str:
    """P2: single-ticker dual-horizon check for the Verify tab.

    Will wrap ``main.verify_single_ticker(ticker, window_rows=120)`` and return
    the result HTML.
    """
    raise NotImplementedError("verify_single_ticker_headless is wired in P2.")
