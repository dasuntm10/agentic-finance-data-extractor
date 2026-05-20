"""Agent 1 — Ingestion & Triage.

Loads the PDF, classifies each page (native / font_decoded / scanned), invokes
OCR on pages that need it, and infers the company name.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from afde.parsing.ocr import ocr_page
from afde.parsing.pdf_loader import load_document
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
    doc = load_document(pdf_path)

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
