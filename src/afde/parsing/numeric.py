"""Accounting-aware Decimal parser and unit detection.

- `(123.4)` → -123.4
- `1,234,567` → 1234567
- `-` / `–` / `—` / empty → None
- Detect units from header text: $Million, $'000, $AUD '000, etc.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable

_NEG_PAREN = re.compile(r"^\(\s*([0-9,.\- ]+)\s*\)$")
_DASH_ONLY = re.compile(r"^[\-–—−\s]*$")
_CLEAN = re.compile(r"[^\d\.\-]")


def parse_number(cell: str | None) -> Decimal | None:
    if cell is None:
        return None
    s = cell.strip()
    if not s or _DASH_ONLY.match(s):
        return None
    # Strip footnote markers / asterisks
    s = s.rstrip("*†‡§")
    neg = False
    m = _NEG_PAREN.match(s)
    if m:
        s = m.group(1).strip()
        neg = True
    s = s.replace(",", " ").replace(" ", "")
    # Final clean
    cleaned = _CLEAN.sub("", s)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        d = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -d if neg else d


_UNIT_PATTERNS = [
    (re.compile(r"\$?\s*million", re.I), 1_000_000),
    (re.compile(r"\$\s*m\b", re.I), 1_000_000),  # only with leading $ — avoids matching trailing 'm' in prose
    (re.compile(r"\$?\s*['’]?\s*000\s*[s]?\b", re.I), 1_000),
    (re.compile(r"\$?\s*thousand", re.I), 1_000),
]


def detect_units(text: str) -> tuple[int, str]:
    """Return (scale, label). Default to (1, '$') if no unit marker found."""
    t = text or ""
    for pat, scale in _UNIT_PATTERNS:
        m = pat.search(t)
        if m:
            return scale, m.group(0).strip()
    return 1, "$"


def detect_currency(text: str) -> str:
    t = (text or "").upper()
    for ccy in ("AUD", "USD", "EUR", "GBP", "NZD", "JPY"):
        if ccy in t:
            return ccy
    return "AUD"


# ---------------------------------------------------------------------------
# Note-reference tokeniser
# ---------------------------------------------------------------------------

_NOTE_TOKEN = re.compile(r"(\d+)\s*(?:\(([a-zA-Z]+)\))?")


def parse_note_refs(cell: str | None) -> list[dict[str, object]]:
    """Parse a Note cell into a list of {note_number, sub, raw} dicts.

    Examples:
        "3(a)"        → [{number:3, sub:'a', raw:'3(a)'}]
        "3, 26"       → [{number:3,sub:None,raw:'3'}, {number:26,sub:None,raw:'26'}]
        "3(d), 26"    → [{number:3,sub:'d',raw:'3(d)'}, {number:26,sub:None,raw:'26'}]
        ""            → []
    """
    if not cell:
        return []
    cell = cell.strip()
    if cell in ("-", "–", "—"):
        return []
    refs: list[dict[str, object]] = []
    for chunk in re.split(r"[,;]", cell):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _NOTE_TOKEN.match(chunk)
        if not m:
            continue
        num = int(m.group(1))
        sub = m.group(2).lower() if m.group(2) else None
        refs.append({"note_number": num, "sub": sub, "raw": chunk})
    return refs


# ---------------------------------------------------------------------------
# Year-column detection
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?:^|\s)(20\d{2}|19\d{2})\b")


def detect_year_columns(header: Iterable[str]) -> list[tuple[int, str]]:
    """Return [(column_index, year_label)] for header cells that look like a year."""
    out: list[tuple[int, str]] = []
    for i, cell in enumerate(header):
        if not cell:
            continue
        m = _YEAR_RE.search(cell)
        if m:
            out.append((i, m.group(1)))
    return out
