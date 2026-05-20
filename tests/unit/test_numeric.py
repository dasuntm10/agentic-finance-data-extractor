"""Unit tests for the accounting-aware numeric / note-ref parsers."""
from __future__ import annotations

from decimal import Decimal

import pytest

from afde.parsing.numeric import (
    detect_currency,
    detect_units,
    detect_year_columns,
    parse_note_refs,
    parse_number,
)


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("108.8", Decimal("108.8")),
        ("1,234.5", Decimal("1234.5")),
        ("(580.8)", Decimal("-580.8")),
        ("( 4.3 )", Decimal("-4.3")),
        ("-", None),
        ("—", None),
        ("", None),
        ("  ", None),
        ("12.5*", Decimal("12.5")),
        ("12 345.6", Decimal("12345.6")),
    ],
)
def test_parse_number(cell, expected):
    assert parse_number(cell) == expected


def test_note_refs_single_letter():
    assert parse_note_refs("3(a)") == [{"note_number": 3, "sub": "a", "raw": "3(a)"}]


def test_note_refs_multi_with_letter():
    assert parse_note_refs("3(d), 26") == [
        {"note_number": 3, "sub": "d", "raw": "3(d)"},
        {"note_number": 26, "sub": None, "raw": "26"},
    ]


def test_note_refs_empty():
    assert parse_note_refs("") == []
    assert parse_note_refs("—") == []


def test_units_million():
    scale, _ = detect_units("$Million")
    assert scale == 1_000_000


def test_units_thousands():
    scale, _ = detect_units("$'000")
    assert scale == 1_000


def test_units_none():
    scale, _ = detect_units("random text with the letter m inside")
    assert scale == 1


def test_currency_default_aud():
    assert detect_currency("$Million") == "AUD"


def test_currency_usd():
    assert detect_currency("USD '000") == "USD"


def test_year_columns():
    assert detect_year_columns(["Note", "2024", "2023"]) == [(1, "2024"), (2, "2023")]
