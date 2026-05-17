"""Unit tests for _split_html_report in src/utils/telegram_bot.py.

The conftest stubs out the `telegram` and `dotenv` packages so the module
imports cleanly without the full `python-telegram-bot` stack.
"""
from __future__ import annotations

import pytest

from src.utils.telegram_bot import _split_html_report

_SEP = "══════════════════════════════"  # the separator used by _build_combined_report


def _block(n: int) -> str:
    """Return a dummy block of exactly n characters (no separator)."""
    return "a" * n


def _blocks_report(*sizes: int) -> str:
    """Join dummy blocks with the visual separator."""
    return f"\n\n{_SEP}\n\n".join(_block(s) for s in sizes)


class TestSplitHtmlReport:
    def test_empty_string_returns_empty_list(self):
        assert _split_html_report("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _split_html_report("   \n  ") == []

    def test_short_report_single_chunk(self):
        report = "Hello world"
        result = _split_html_report(report, max_len=4000)
        assert len(result) == 1

    def test_short_report_content_unchanged(self):
        report = "Hello world"
        result = _split_html_report(report, max_len=4000)
        assert result[0] == report

    def test_two_small_blocks_fit_in_one_chunk(self):
        # Each block 1000 chars; total (incl separator overhead) well under 4000.
        report = _blocks_report(1000, 1000)
        result = _split_html_report(report, max_len=4000)
        assert len(result) == 1

    def test_two_large_blocks_split_into_two_chunks(self):
        # Each block 2500 chars; together they exceed 4000.
        report = _blocks_report(2500, 2500)
        result = _split_html_report(report, max_len=4000)
        assert len(result) == 2

    def test_three_blocks_split_correctly(self):
        # Each block 2000 chars; no two fit together.
        report = _blocks_report(2000, 2000, 2000)
        result = _split_html_report(report, max_len=4000)
        assert len(result) == 3

    def test_oversized_single_block_hard_sliced(self):
        # 9000 chars, max_len=4000 → chunks [4000, 4000, 1000].
        report = _block(9000)
        result = _split_html_report(report, max_len=4000)
        assert len(result) == 3
        assert len(result[0]) == 4000
        assert len(result[1]) == 4000
        assert len(result[2]) == 1000

    def test_each_chunk_within_max_len(self):
        report = _blocks_report(1500, 1500, 1500, 1500)
        result = _split_html_report(report, max_len=4000)
        assert all(len(chunk) <= 4000 for chunk in result)

    def test_separator_not_in_any_chunk(self):
        report = _blocks_report(2000, 2000, 2000)
        result = _split_html_report(report, max_len=4000)
        for chunk in result:
            assert _SEP not in chunk

    def test_custom_max_len_respected(self):
        report = _blocks_report(500, 500)
        small_result = _split_html_report(report, max_len=100)
        large_result = _split_html_report(report, max_len=4000)
        assert len(small_result) >= len(large_result)

    def test_report_exactly_at_max_len_is_one_chunk(self):
        max_len = 4000
        report = _block(max_len)
        result = _split_html_report(report, max_len=max_len)
        assert len(result) == 1
        assert len(result[0]) == max_len
