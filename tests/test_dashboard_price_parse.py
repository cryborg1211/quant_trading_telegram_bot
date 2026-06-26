"""Regression tests for dashboard price coercion (`_parse_price`).

Guards the GIỮ-tab crash where a legacy/bot-era `portfolio` row stored the entry
price as display TEXT (e.g. '47,800 VND') and `float(...)` blew up with
`could not convert string to float: '47,800 VND'`.
"""
from __future__ import annotations

import pytest

from dashboard.utils.headless import _parse_price


class TestParsePrice:
    def test_plain_float_passthrough(self) -> None:
        assert _parse_price(47800.0) == 47800.0

    def test_int_passthrough(self) -> None:
        assert _parse_price(47800) == 47800.0

    def test_formatted_vnd_string(self) -> None:
        # The exact string from the live crash.
        assert _parse_price("47,800 VND") == 47800.0

    def test_currency_symbol_and_decimal(self) -> None:
        assert _parse_price("1,234.56 ₫") == 1234.56

    def test_bare_numeric_string(self) -> None:
        assert _parse_price("50000") == 50000.0

    @pytest.mark.parametrize("bad", [None, "", "N/A", "—", "VND", "-", "."])
    def test_unparseable_returns_zero(self, bad: object) -> None:
        assert _parse_price(bad) == 0.0

    def test_negative_preserved(self) -> None:
        assert _parse_price("-1,200") == -1200.0
