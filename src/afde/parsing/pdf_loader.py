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

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from afde.schemas import BBox, Heading, IngestedDocument, PageBlock, TableBlock

log = logging.getLogger(__name__)

GLYPH_TOKEN = re.compile(r"/\d+(?:/\d+)+")  # AUSNET-style CMap garble
MIN_USEFUL_CHARS = 200

# Docling segfaults on long PDFs (native memory growth, ≈page 32 on Windows), so we
# run it one page-chunk at a time in a fresh subprocess. Chunk size is kept well below
# the observed crash threshold; override with AFDE_DOCLING_CHUNK if a machine crashes sooner.
DOCLING_CHUNK_PAGES = int(os.environ.get("AFDE_DOCLING_CHUNK", "16"))


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


def _pdf_page_count(path: str | Path) -> int:
    import fitz  # noqa: PLC0415

    doc = fitz.open(str(path))
    try:
        return doc.page_count
    finally:
        doc.close()


def _run_docling_chunk(path: str | Path, start: int, end: int) -> list[TableBlock] | None:
    """Run Docling on pages [start, end] in an isolated subprocess.

    Returns a list of TableBlock, or ``None`` if the subprocess failed — a native
    segfault (nonzero exit), Docling not installed, or any other error — so the
    caller can fall back to PyMuPDF for just that page range.
    """
    fd, out_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "afde.parsing.docling_worker", str(path), str(start), str(end), out_json],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
            log.warning("Docling worker failed for pages %d-%d (exit %s): %s", start, end, proc.returncode, tail[0])
            return None
        with open(out_json, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        log.warning("Docling worker error for pages %d-%d: %s", start, end, e)
        return None
    finally:
        try:
            os.unlink(out_json)
        except OSError:
            pass

    blocks: list[TableBlock] = []
    for t in raw:
        rows = t.get("rows") or []
        if not rows:
            continue
        header, *data_rows = rows
        blocks.append(
            TableBlock(
                table_id=str(uuid.uuid4())[:8],
                page=t.get("page") or 0,
                header=[(c or "").strip() for c in header],
                rows=[[(c or "").strip() for c in r] for r in data_rows],
            )
        )
    return blocks


def load_tables_with_docling(path: str | Path) -> list[TableBlock]:
    """Extract logical tables via Docling, isolated per page-chunk in subprocesses.

    Docling is used purely for table *structure* (TableFormer); the text is already
    decoded by PyMuPDF. Because Docling's native pipeline accumulates memory and
    segfaults on long PDFs, each chunk runs in a fresh subprocess (see
    ``docling_worker``): the crash is contained and memory is reclaimed between
    chunks. A chunk whose subprocess dies falls back to PyMuPDF for that range.
    """
    try:
        page_count = _pdf_page_count(path)
    except Exception as e:
        log.warning("Could not read page count (%s); falling back to PyMuPDF.", e)
        return _fallback_tables_from_pymupdf(path)

    out: list[TableBlock] = []
    any_docling = False
    for start in range(1, page_count + 1, DOCLING_CHUNK_PAGES):
        end = min(start + DOCLING_CHUNK_PAGES - 1, page_count)
        blocks = _run_docling_chunk(path, start, end)
        if blocks is None:
            log.warning("Falling back to PyMuPDF tables for pages %d-%d.", start, end)
            out.extend(_fallback_tables_from_pymupdf(path, start=start, end=end))
        else:
            any_docling = True
            out.extend(blocks)
    if not any_docling:
        log.warning("Docling produced no tables on any chunk; table set is PyMuPDF-only.")
    return out


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


def _fallback_tables_from_pymupdf(
    path: str | Path, start: int | None = None, end: int | None = None
) -> list[TableBlock]:
    """PyMuPDF's built-in table-finder. Less accurate than Docling but works offline.

    ``start``/``end`` (1-based, inclusive) restrict the scan to a page range, used when
    a single Docling chunk fails and only that range needs a fallback.
    """
    import fitz  # noqa: PLC0415

    out: list[TableBlock] = []
    doc = fitz.open(str(path))
    try:
        for i, page in enumerate(doc):
            if start is not None and (i + 1) < start:
                continue
            if end is not None and (i + 1) > end:
                continue
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


def _contiguous_ranges(pages: set[int], max_len: int = DOCLING_CHUNK_PAGES) -> list[tuple[int, int]]:
    """Group page numbers into contiguous (start, end) ranges, each capped at max_len."""
    ranges: list[tuple[int, int]] = []
    for p in sorted(pages):
        if ranges and p <= ranges[-1][1] + 1 and (ranges[-1][1] - ranges[-1][0] + 1) < max_len:
            ranges[-1] = (ranges[-1][0], p)
        else:
            ranges.append((p, p))
    return ranges


def load_tables_for_pages(path: str | Path, pages: set[int]) -> list[TableBlock]:
    """Run Docling on only the given pages (grouped into contiguous ranges).

    The fast path: instead of scanning all ~40 pages, the Ingestion agent first
    finds the few pages that hold the primary statements (cheap, text-only) and
    passes them here. Each range runs in an isolated subprocess (see
    ``_run_docling_chunk``), so this is both far faster and crash-safe.
    """
    if not pages:
        return []
    out: list[TableBlock] = []
    for start, end in _contiguous_ranges(pages):
        blocks = _run_docling_chunk(path, start, end)
        if blocks is None:
            log.warning("Targeted Docling failed for pages %d-%d; PyMuPDF fallback.", start, end)
            out.extend(_fallback_tables_from_pymupdf(path, start=start, end=end))
        else:
            out.extend(blocks)
    return out


def load_document(path: str | Path, with_tables: bool = True) -> IngestedDocument:
    """Top-level entrypoint used by the Ingestion agent before OCR routing.

    ``with_tables=False`` skips Docling entirely and returns text + headings only —
    used by Ingestion to do a cheap first pass, locate the statement pages, then
    extract tables for just those pages via ``load_tables_for_pages``.
    """
    pages, headings = load_with_pymupdf(path)
    tables = load_tables_with_docling(path) if with_tables else []
    return IngestedDocument(
        source_path=str(path),
        page_count=len(pages),
        pages=pages,
        tables=tables,
        headings=headings,
    )
