"""Agent 3 — Statement Parser.

Picks the load-bearing table on the P&L page(s), detects units / currency / FY,
and emits a structured Statement with LineItems that carry NoteRefs and Decimal
values per year. Also (optionally) extracts the Statement of Financial Position
when its page range is known, so liquidity/leverage ratios can be computed.
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal

from afde.parsing.numeric import (
    detect_currency,
    detect_units,
    detect_year_columns,
    parse_note_refs,
    parse_number,
)
from afde.schemas import (
    IngestedDocument,
    LineItem,
    NoteRef,
    SectionMap,
    Statement,
    TableBlock,
)

log = logging.getLogger(__name__)

_NOTE_HEADERS = {"note", "notes", "note(s)"}
_PERIOD_HEADER = re.compile(
    r"for the (year|period|half[\- ]?year) ended ?(.{4,40})$", re.I | re.M
)


def _score_table_for_pl(tbl: TableBlock) -> int:
    """How likely is this table the P&L? Higher is better."""
    score = 0
    has_note = False
    has_year = False
    for cell in tbl.header:
        c = cell.lower().strip()
        if c in _NOTE_HEADERS:
            has_note = True
            score += 5
        if re.search(r"20\d{2}|19\d{2}", c):
            has_year = True
            score += 3
        if "$" in c or "million" in c or "'000" in c or "000" in c:
            score += 2
    if not (has_year and len(tbl.rows) >= 3):
        return 0
    # bonus for keywords in row labels
    labels = " ".join((r[0] if r else "").lower() for r in tbl.rows[:20])
    for kw in ("revenue", "expense", "profit", "income", "tax", "interest"):
        if kw in labels:
            score += 1
    return score if has_note or has_year else 0


def _pick_pl_table(doc: IngestedDocument, pages: list[int]) -> TableBlock | None:
    candidates = [t for t in doc.tables if t.page in pages]
    if not candidates:
        # Fall back to *any* table on any of the pages (PyMuPDF may have missed Note col)
        candidates = [t for t in doc.tables if t.page in pages]
    scored = [(t, _score_table_for_pl(t)) for t in candidates]
    scored = [(t, s) for t, s in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def _find_period_end(doc: IngestedDocument, page: int) -> tuple[str, str]:
    """Return (period_end_iso_like, period_label_like) by scanning nearby pages."""
    # Search the P&L page and the page just before for "For the year ended X"
    for p in (page, max(1, page - 1), page + 1):
        pg = next((x for x in doc.pages if x.page == p), None)
        if not pg:
            continue
        m = _PERIOD_HEADER.search(pg.text)
        if m:
            raw = m.group(2).strip().rstrip(".")
            return raw, raw
    # Fall back to year token in the header row of the chosen table
    return "unknown", "unknown"


def _note_col_index(header: list[str]) -> int | None:
    for i, c in enumerate(header):
        if c.strip().lower() in _NOTE_HEADERS:
            return i
    return None


def _build_line_item(
    row: list[str],
    *,
    label_idx: int,
    note_idx: int | None,
    year_cols: list[tuple[int, str]],
    page: int,
) -> LineItem | None:
    if not row or not row[label_idx].strip():
        return None
    label = row[label_idx].strip()
    # Heuristics for header / blank rows
    if label.lower().startswith(("$", "note")) or re.fullmatch(r"20\d{2}", label):
        return None
    notes: list[NoteRef] = []
    if note_idx is not None and note_idx < len(row):
        for ref in parse_note_refs(row[note_idx]):
            notes.append(NoteRef(note_number=int(ref["note_number"]), sub=ref["sub"], raw=ref["raw"]))
    values: dict[str, Decimal] = {}
    for col_idx, year in year_cols:
        if col_idx >= len(row):
            continue
        v = parse_number(row[col_idx])
        if v is not None:
            values[year] = v
    if not values:
        return None
    is_total = bool(
        re.search(r"^(total|net (loss|profit)|profit (before|after) tax|loss before)", label, re.I)
    )
    return LineItem(
        label=label,
        note_refs=notes,
        values=values,
        is_subtotal=is_total,
        is_total=is_total,
        page=page,
    )


def _parse_statement_table(
    tbl: TableBlock,
    *,
    kind: str,
    doc: IngestedDocument,
) -> Statement | None:
    if not tbl.header or not tbl.rows:
        return None

    # Determine label column: first non-Note, non-Year text column
    note_idx = _note_col_index(tbl.header)
    year_cols = detect_year_columns(tbl.header)
    if not year_cols:
        return None
    label_idx = 0
    if note_idx == 0 and len(tbl.header) > 1:
        label_idx = 1
    # If header has only year columns, label is column 0
    skip = {note_idx} | {i for i, _ in year_cols}
    for i, _ in enumerate(tbl.header):
        if i not in skip:
            label_idx = i
            break

    # Units & currency: look in the rows just below the header (often "$Million")
    units_scale = 1
    units_label = "$"
    currency = "AUD"
    for r in tbl.rows[:3]:
        joined = " ".join(r)
        s, lab = detect_units(joined)
        if s > 1:
            units_scale, units_label = s, lab
            break
    # Also probe the page text
    page = next((p for p in doc.pages if p.page == tbl.page), None)
    if page:
        s, lab = detect_units(page.text)
        if s > units_scale:
            units_scale, units_label = s, lab
        currency = detect_currency(page.text)

    period_end, period_label_text = _find_period_end(doc, tbl.page)
    # Year labels for `values` keys (use the year extracted from the header)
    primary_year_col = year_cols[0]
    primary_period_label = primary_year_col[1]
    comp_period_label = year_cols[1][1] if len(year_cols) > 1 else None

    items: list[LineItem] = []
    for row in tbl.rows:
        li = _build_line_item(
            row, label_idx=label_idx, note_idx=note_idx, year_cols=year_cols, page=tbl.page
        )
        if li:
            items.append(li)

    if not items:
        return None

    return Statement(
        kind=kind,  # type: ignore[arg-type]
        title=period_label_text or kind,
        currency=currency,
        units_scale=units_scale,
        units_label=units_label,
        period_end=period_end,
        period_label=primary_period_label,
        comparative_period_end=None,
        comparative_period_label=comp_period_label,
        line_items=items,
        pages=[tbl.page],
    )


def run(doc: IngestedDocument, section_map: SectionMap) -> dict[str, Statement]:
    """Return {'profit_or_loss': Statement, 'balance_sheet': Statement?}."""
    out: dict[str, Statement] = {}

    if section_map.pl_pages:
        pl_tbl = _pick_pl_table(doc, section_map.pl_pages)
        if pl_tbl:
            stmt = _parse_statement_table(pl_tbl, kind="profit_or_loss", doc=doc)
            if stmt:
                out["profit_or_loss"] = stmt
                log.info(
                    "Parsed P&L: %d items, units=%s, periods=%s/%s",
                    len(stmt.line_items),
                    stmt.units_label,
                    stmt.period_label,
                    stmt.comparative_period_label,
                )

    if section_map.balance_sheet_pages:
        bs_tbl = _pick_pl_table(doc, section_map.balance_sheet_pages)
        if bs_tbl:
            stmt = _parse_statement_table(bs_tbl, kind="balance_sheet", doc=doc)
            if stmt:
                out["balance_sheet"] = stmt
                log.info("Parsed BS: %d items", len(stmt.line_items))

    return out
