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

from afde.llm.client import ToolChoice, get_llm
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
    """Return (period_end, period_label) for the financial year of the statement.

    Three-tier resolution:
      1. Regex — looks for "for the (year|period|half-year) ended <date>" on the
         P&L page and the pages immediately before and after.
      2. LLM  — if the regex finds nothing, passes the page text to Qwen2.5-7B
         with a constrained JSON schema so the response is guaranteed to be a
         date string or null. Used for unusual phrasings such as "Year ended
         30 June 2024" or "12 months to 31 December 2024".
      3. Fallback — returns "unknown" if the LLM is unavailable or returns null.
         The caller already has the year label from the table column header, so
         the pipeline continues correctly even without a full date string.
    """
    # --- Tier 1: regex ---
    candidate_pages = (page, max(1, page - 1), page + 1)
    page_texts: list[str] = []
    for p in candidate_pages:
        pg = next((x for x in doc.pages if x.page == p), None)
        if not pg:
            continue
        m = _PERIOD_HEADER.search(pg.text)
        if m:
            raw = m.group(2).strip().rstrip(".")
            log.debug("Period end found by regex: %s", raw)
            return raw, raw
        page_texts.append(pg.text[:2000])

    # --- Tier 2: LLM ---
    if page_texts:
        result = _llm_find_period_end(page_texts)
        if result:
            log.debug("Period end found by LLM: %s", result)
            return result, result

    # --- Tier 3: fallback ---
    log.warning("Could not determine period end for page %d; using 'unknown'.", page)
    return "unknown", "unknown"


def _llm_find_period_end(page_texts: list[str]) -> str | None:
    """Ask the local LLM to extract the financial year end date from page text."""
    llm = get_llm()
    tool = ToolChoice(
        name="extract_period_end",
        description=(
            "Extract the financial year-end date of the statement. "
            "Return the date exactly as it appears in the text (e.g. '31 December 2024', "
            "'30 June 2024', '31 March 2023'). Return null if no date can be found."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "period_end": {
                    "type": ["string", "null"],
                    "description": "The financial year-end date as it appears in the document, or null.",
                },
            },
            "required": ["period_end"],
        },
    )
    excerpt = "\n\n---\n\n".join(page_texts)
    try:
        result = llm.structured(
            tier="classification",
            system=(
                "You extract the financial year-end date from the text of a financial statement. "
                "Look for phrases like 'year ended', 'period ended', '12 months to', "
                "'financial year 20XX', or any date near the statement heading."
            ),
            user=f"Statement page text:\n\n{excerpt}",
            tool=tool,
            max_tokens=64,
        )
        value = result.get("period_end")
        if value and str(value).strip().lower() not in ("null", "none", "unknown", ""):
            return str(value).strip()
    except Exception as e:
        log.warning("LLM period-end extraction failed: %s", e)
    return None


def _note_col_index(header: list[str]) -> int | None:
    for i, c in enumerate(header):
        if c.strip().lower() in _NOTE_HEADERS:
            return i
    return None


def _llm_parse_row_values(
    row: list[str],
    year_cols: list[tuple[int, str]],
    label: str,
) -> dict[str, Decimal]:
    """Use the LLM to parse numeric values when standard parsing fails.

    Invoked only when parse_number() returned None for every year column on a
    row that has a valid label — e.g. OCR artifacts, unusual formatting, or
    numbers written in a way the deterministic parser doesn't recognise.
    The schema constrains the response to one string-or-null per year column,
    so the model cannot hallucinate extra fields or refuse to answer.
    """
    llm = get_llm()

    # Build per-year properties for the constrained schema
    year_labels = [year for _, year in year_cols]
    properties = {
        year: {
            "type": ["string", "null"],
            "description": (
                f"Numeric value for {year} as a plain number string "
                f"(e.g. '108.8', '-580.8'). Use null if not present or not a number."
            ),
        }
        for year in year_labels
    }

    tool = ToolChoice(
        name="parse_row_values",
        description=(
            f"Extract the numeric monetary values for each year column from the row "
            f"labelled {label!r}. Apply accounting conventions: "
            "parentheses mean negative — (580.8) = -580.8. "
            "A dash, em-dash, or empty cell means null."
        ),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": year_labels,
        },
    )

    cell_descriptions = "\n".join(
        f"  {year}: {row[col_idx] if col_idx < len(row) else '(missing)'!r}"
        for col_idx, year in year_cols
    )
    try:
        result = llm.structured(
            tier="classification",
            system=(
                "You parse numeric cell values from financial statement rows. "
                "Accounting convention: (123.4) = -123.4. Dash or blank = null. "
                "Return each value as a plain decimal string with no currency symbols or commas."
            ),
            user=f"Row label: {label!r}\nRaw cell contents:\n{cell_descriptions}",
            tool=tool,
            max_tokens=128,
        )
        values: dict[str, Decimal] = {}
        for year in year_labels:
            raw = result.get(year)
            if raw is None or str(raw).strip().lower() in ("null", "none", ""):
                continue
            # Run through parse_number first so accounting conventions are applied
            parsed = parse_number(str(raw))
            if parsed is None:
                # Last resort: direct Decimal conversion on cleaned string
                try:
                    parsed = Decimal(str(raw).replace(",", "").strip())
                except Exception:
                    continue
            values[year] = parsed
        if values:
            log.info("LLM recovered values for row %r: %s", label, values)
        return values
    except Exception as e:
        log.warning("LLM row-value parsing failed for %r: %s", label, e)
        return {}


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
    # Skip header / unit rows
    if label.lower().startswith(("$", "note")) or re.fullmatch(r"20\d{2}", label):
        return None
    notes: list[NoteRef] = []
    if note_idx is not None and note_idx < len(row):
        for ref in parse_note_refs(row[note_idx]):
            notes.append(NoteRef(note_number=int(ref["note_number"]), sub=ref["sub"], raw=ref["raw"]))

    # --- Tier 1: deterministic numeric parser ---
    values: dict[str, Decimal] = {}
    for col_idx, year in year_cols:
        if col_idx >= len(row):
            continue
        v = parse_number(row[col_idx])
        if v is not None:
            values[year] = v

    # --- Tier 2: LLM fallback when all cells failed to parse ---
    if not values:
        values = _llm_parse_row_values(row, year_cols, label)

    # If still no values the row is genuinely non-numeric (e.g. a section label)
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
