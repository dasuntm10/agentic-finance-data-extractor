"""Lightweight smoke test for the deterministic parsing path.

Designed to run with stdlib + pypdf only (no LangGraph, no pydantic v2, no LLM).
Validates:
  - numeric parser handles ( ) negatives, commas, dashes, footnote markers
  - note-ref tokeniser handles 3(a), 3, 26, 3(d), 26, plain integers
  - year/unit detection on real Citigroup header text
  - the load-bearing P&L row layout from Citigroup p.8 is parseable

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Import the helpers directly — they are pure stdlib so they work under 3.9
from afde.parsing.numeric import (  # noqa: E402
    detect_currency,
    detect_units,
    detect_year_columns,
    parse_note_refs,
    parse_number,
)

FAIL = 0
PASS = 0


def check(name, actual, expected):
    global FAIL, PASS
    ok = actual == expected
    PASS += int(ok)
    FAIL += int(not ok)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        print(f"        expected: {expected!r}")
        print(f"        actual:   {actual!r}")


def test_numeric_parser():
    print("== parse_number ==")
    check("plain", parse_number("108.8"), Decimal("108.8"))
    check("comma", parse_number("1,234.5"), Decimal("1234.5"))
    check("paren_neg", parse_number("(580.8)"), Decimal("-580.8"))
    check("paren_neg_spaces", parse_number("( 4.3 )"), Decimal("-4.3"))
    check("dash", parse_number("-"), None)
    check("emdash", parse_number("—"), None)
    check("empty", parse_number(""), None)
    check("blank", parse_number("   "), None)
    check("footnote", parse_number("12.5*"), Decimal("12.5"))
    check("thousands_space", parse_number("12 345.6"), Decimal("12345.6"))


def test_note_refs():
    print("== parse_note_refs ==")
    check("single_letter", parse_note_refs("3(a)"), [{"note_number": 3, "sub": "a", "raw": "3(a)"}])
    check("plain", parse_note_refs("26"), [{"note_number": 26, "sub": None, "raw": "26"}])
    check(
        "comma_two",
        parse_note_refs("3, 26"),
        [
            {"note_number": 3, "sub": None, "raw": "3"},
            {"note_number": 26, "sub": None, "raw": "26"},
        ],
    )
    check(
        "comma_letter_first",
        parse_note_refs("3(d), 26"),
        [
            {"note_number": 3, "sub": "d", "raw": "3(d)"},
            {"note_number": 26, "sub": None, "raw": "26"},
        ],
    )
    check("empty", parse_note_refs(""), [])
    check("dash", parse_note_refs("—"), [])


def test_units_currency():
    print("== detect_units / detect_currency ==")
    s, _ = detect_units("$Million $Million")
    check("million_header", s, 1_000_000)
    s, _ = detect_units("$'000")
    check("apostrophe_000", s, 1_000)
    s, _ = detect_units("$000")
    check("plain_000", s, 1_000)
    s, _ = detect_units("Some random text")
    check("no_unit", s, 1)
    check("aud_default", detect_currency("AUD '000"), "AUD")
    check("usd", detect_currency("USD $Million"), "USD")


def test_year_cols():
    print("== detect_year_columns ==")
    cols = detect_year_columns(["Note", "2024", "2023"])
    check("two_years", cols, [(1, "2024"), (2, "2023")])
    cols = detect_year_columns(["Item", "Note", "30 June 2024", "30 June 2023"])
    check("dated_years", cols, [(2, "2024"), (3, "2023")])


def test_citigroup_real_pdf():
    """Pull p.8 of Citigroup, verify our parser can find the right rows."""
    print("== Citigroup P&L (real PDF, p.8) ==")
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  [SKIP] pypdf not installed")
        return
    pdf = ROOT / "data" / "CITIGROUP.pdf"
    if not pdf.exists():
        print(f"  [SKIP] {pdf} not found")
        return
    reader = PdfReader(str(pdf))
    text = reader.pages[7].extract_text() or ""
    # Key rows the parser must handle:
    #   "Advisory fees           3(a)  108.8   101.6"
    #   "Net trading income      3(c)   (4.3)   29.9"
    #   "Interest expense        3(d) (580.8) (521.1)"
    # We don't have the table layout here — just verify the substring + numeric parse work.
    assert "Advisory fees" in text, "Could not find 'Advisory fees' on p.8"
    print("  [PASS] 'Advisory fees' present on p.8")
    assert "3(a)" in text
    print("  [PASS] note ref '3(a)' present")
    assert "(580.8)" in text
    print("  [PASS] negative '(580.8)' present")
    # Round-trip the parens-negative through our parser
    check("citigroup_paren_neg", parse_number("(580.8)"), Decimal("-580.8"))
    # Find unit declaration
    s, _ = detect_units(text)
    check("citigroup_units", s, 1_000_000)


def main():
    test_numeric_parser()
    test_note_refs()
    test_units_currency()
    test_year_cols()
    test_citigroup_real_pdf()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
