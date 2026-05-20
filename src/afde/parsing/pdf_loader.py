"""PDF loading that produces structured pages and tables.

Strategy:
  1. PyMuPDF (`fitz`) is the always-on extractor — fast, gives us text + bbox per page.
  2. Docling (when installed) is layered on top to extract logical tables. It is
     imported lazily so the pipeline still runs without it (with reduced table
     fidelity). The architecture doc names Docling as the primary table tool;
     this module is the seam where it plugs in.
  3. Pages that come back essentially empty OR full of custom-font glyph tokens
     (`/0/1/2/...`) are flagged needs_ocr; the Ingestion agent decides whether
     to invoke the OCR fallback.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from afde.schemas import BBox, Heading, IngestedDocument, PageBlock, TableBlock

log = logging.getLogger(__name__)

GLYPH_TOKEN = re.compile(r"/\d+(?:/\d+)+")  # AUSNET-style CMap garble
MIN_USEFUL_CHARS = 200


def _is_garbled(text: str) -> bool:
    if not text:
        return True
    if len(text.strip()) < MIN_USEFUL_CHARS:
        return False  # short pages aren't necessarily garbled
    tokens = GLYPH_TOKEN.findall(text)
    if not tokens:
        return False
    # If glyph tokens dominate (cover >40% of chars), the page is decoded-as-garbage
    glyph_chars = sum(len(t) for t in tokens)
    return glyph_chars / max(len(text), 1) > 0.4


def load_with_pymupdf(path: str | Path) -> tuple[list[PageBlock], list[Heading]]:
    """First-pass page extraction using PyMuPDF."""
    import fitz  # noqa: PLC0415

    pages: list[PageBlock] = []
    headings: list[Heading] = []
    doc = fitz.open(str(path))
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            stripped = text.strip()
            needs_ocr = False
            source = "native"
            if _is_garbled(text):
                needs_ocr = True
                source = "font_decoded"
            elif len(stripped) < MIN_USEFUL_CHARS:
                # likely a scanned page (only page-header chars survive)
                needs_ocr = True
                source = "native"

            pages.append(
                PageBlock(
                    page=i + 1,
                    source=source,
                    text=text,
                    char_count=len(stripped),
                    needs_ocr=needs_ocr,
                )
            )

            # cheap headings: short lines in larger fonts
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    size = max(s.get("size", 0) for s in spans)
                    txt = "".join(s.get("text", "") for s in spans).strip()
                    if size >= 11 and 3 < len(txt) < 120 and txt.isprintable():
                        if any(c.isalpha() for c in txt):
                            headings.append(Heading(page=i + 1, text=txt, level=1 if size >= 13 else 2))
    finally:
        doc.close()
    return pages, headings


def load_tables_with_docling(path: str | Path) -> list[TableBlock]:
    """Extract logical tables via Docling. Returns [] if Docling not installed."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - optional dep
        log.warning("Docling not available (%s); table extraction will fall back to PyMuPDF blocks.", e)
        return _fallback_tables_from_pymupdf(path)

    try:
        conv = DocumentConverter()
        result = conv.convert(str(path))
        out: list[TableBlock] = []
        doc = result.document
        # Docling DocumentItem table iteration
        for item, _level in doc.iterate_items():
            if getattr(item, "label", None) and "table" in str(item.label).lower():
                rows = _docling_table_to_rows(item)
                if not rows:
                    continue
                header, *data_rows = rows
                page_no = getattr(item, "prov", [None])[0]
                page_num = getattr(page_no, "page_no", None) if page_no else None
                out.append(
                    TableBlock(
                        table_id=str(uuid.uuid4())[:8],
                        page=page_num or 0,
                        header=[(c or "").strip() for c in header],
                        rows=[[(c or "").strip() for c in r] for r in data_rows],
                    )
                )
        return out
    except Exception as e:
        log.warning("Docling parse failed (%s); falling back to PyMuPDF.", e)
        return _fallback_tables_from_pymupdf(path)


def _docling_table_to_rows(item) -> list[list[str]]:
    """Best-effort conversion of a Docling TableItem to a list-of-lists."""
    rows: list[list[str]] = []
    data = getattr(item, "data", None)
    if data is None:
        return rows
    grid = getattr(data, "grid", None) or getattr(data, "table_cells", None)
    if grid is None:
        # Last-resort: use the item's text export
        text = getattr(item, "text", "") or ""
        return [[line] for line in text.splitlines() if line.strip()]
    # Newer Docling: data.grid is list[list[TableCell]]
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        for row in grid:
            rows.append([(getattr(c, "text", "") or "").strip() for c in row])
        return rows
    # Older Docling: list of cells with row_idx/col_idx
    max_r = max((getattr(c, "start_row_offset_idx", 0) for c in grid), default=-1)
    max_c = max((getattr(c, "start_col_offset_idx", 0) for c in grid), default=-1)
    if max_r < 0:
        return rows
    rows = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for c in grid:
        r = getattr(c, "start_row_offset_idx", 0)
        col = getattr(c, "start_col_offset_idx", 0)
        rows[r][col] = (getattr(c, "text", "") or "").strip()
    return rows


def _fallback_tables_from_pymupdf(path: str | Path) -> list[TableBlock]:
    """PyMuPDF's built-in table-finder. Less accurate than Docling but works offline."""
    import fitz  # noqa: PLC0415

    out: list[TableBlock] = []
    doc = fitz.open(str(path))
    try:
        for i, page in enumerate(doc):
            try:
                finder = page.find_tables()
            except Exception:
                continue
            for t in finder.tables:
                try:
                    extracted = t.extract()
                except Exception:
                    continue
                if not extracted:
                    continue
                rows = [[(c or "").strip() if c else "" for c in row] for row in extracted]
                # First non-empty row is the header
                header_idx = next((idx for idx, r in enumerate(rows) if any(r)), 0)
                header = rows[header_idx]
                data_rows = rows[header_idx + 1 :]
                bbox = BBox(page=i + 1, x0=t.bbox[0], y0=t.bbox[1], x1=t.bbox[2], y1=t.bbox[3])
                out.append(
                    TableBlock(
                        table_id=str(uuid.uuid4())[:8],
                        page=i + 1,
                        bbox=bbox,
                        header=header,
                        rows=data_rows,
                    )
                )
    finally:
        doc.close()
    return out


def load_document(path: str | Path) -> IngestedDocument:
    """Top-level entrypoint used by the Ingestion agent before OCR routing."""
    pages, headings = load_with_pymupdf(path)
    tables = load_tables_with_docling(path)
    return IngestedDocument(
        source_path=str(path),
        page_count=len(pages),
        pages=pages,
        tables=tables,
        headings=headings,
    )
