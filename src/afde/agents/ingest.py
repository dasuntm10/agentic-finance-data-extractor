"""Agent 1 — Ingestion & Triage.

Loads the PDF, classifies each page (native / font_decoded / scanned), invokes
OCR on pages that need it, and infers the company name.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from afde.parsing.ocr import ocr_page
from afde.parsing.pdf_loader import load_document, load_tables_for_pages, load_tables_with_docling
from afde.schemas import DocumentQuality, IngestedDocument

log = logging.getLogger(__name__)

_COMPANY_HINT = re.compile(r"Company name\s*\n([A-Z][A-Z &\.\,\-\(\)]+)")
_ALT_NAME = re.compile(r"\bACN\s+\d{3}\s+\d{3}\s+\d{3}\b")


def _infer_company_name(pages_text: list[str], fallback: str) -> str:
    for txt in pages_text[:3]:
        m = _COMPANY_HINT.search(txt)
        if m:
            return m.group(1).strip().splitlines()[0].title()
    return fallback


def run(pdf_path: str | Path) -> IngestedDocument:
    pdf_path = Path(pdf_path)

    # Phase 1: text + headings only (fast — no Docling). Phase 2: a cheap text-only
    # pre-scan picks the few pages holding the primary statements, and Docling runs
    # on just those. This avoids reading all ~40 pages through the (slow, crash-prone)
    # Docling pipeline. Set AFDE_DOCLING_FULL=1 to force full-document extraction.
    force_full = os.environ.get("AFDE_DOCLING_FULL", "").lower() in ("1", "true", "yes")
    if force_full:
        doc = load_document(pdf_path, with_tables=True)
    else:
        from afde.agents import locate as agent_locate  # noqa: PLC0415  (avoid import cycle)

        doc = load_document(pdf_path, with_tables=False)
        wanted = agent_locate.candidate_statement_pages(doc)
        if wanted:
            log.info("Targeted table extraction on pages %s", sorted(wanted))
            doc.tables = load_tables_for_pages(pdf_path, wanted)
        else:
            log.info("Text pre-scan found no statement pages; extracting all tables.")
            doc.tables = load_tables_with_docling(pdf_path)

    # Triage: invoke OCR on flagged pages
    ocr_invoked = 0
    low_conf_pages = 0
    for page in doc.pages:
        if not page.needs_ocr:
            continue
        text, conf = ocr_page(pdf_path, page.page)
        if text:
            page.text = text
            page.char_count = len(text.strip())
            page.ocr_confidence = conf
            page.source = "ocr"
            ocr_invoked += 1
            if conf < 0.75:
                low_conf_pages += 1
        else:
            low_conf_pages += 1

    fallback_name = pdf_path.stem.replace("_", " ").title()
    doc.company_name = _infer_company_name([p.text for p in doc.pages], fallback_name)

    # Overall quality
    if low_conf_pages == 0:
        doc.quality = DocumentQuality.HIGH
    elif low_conf_pages <= 3:
        doc.quality = DocumentQuality.MEDIUM
    else:
        doc.quality = DocumentQuality.LOW

    log.info(
        "Ingest: %s pages=%d ocr=%d low_conf=%d quality=%s",
        pdf_path.name,
        doc.page_count,
        ocr_invoked,
        low_conf_pages,
        doc.quality,
    )
    return doc
