"""Subprocess worker: convert one page-range of a PDF with Docling, emit tables as JSON.

Run as:  python -m afde.parsing.docling_worker <pdf> <start> <end> <out_json>

Run in its own process on purpose. Docling's native page pipeline (pypdfium2 + the
layout/TableFormer models) accumulates memory across pages and triggers a
``std::bad_alloc`` / segmentation fault around page ~32 on some Windows + torch
stacks. A segfault cannot be caught from Python, so the parent (``pdf_loader``)
runs Docling here, one page-chunk at a time: a crash exits nonzero and the parent
skips that chunk instead of taking the whole pipeline down with it.
"""
from __future__ import annotations

import json
import logging
import sys


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: docling_worker <pdf> <start> <end> <out_json>", file=sys.stderr)
        return 2
    pdf, start, end, out_json = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

    # Keep stdout clean for the parent; third-party chatter goes nowhere useful here.
    logging.disable(logging.WARNING)

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from afde.parsing.pdf_loader import _docling_table_to_rows

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # text PDFs; OCR adds nothing and is a crash source
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = conv.convert(pdf, page_range=(start, end))
    document = result.document

    tables: list[dict] = []
    for item, _level in document.iterate_items():
        if getattr(item, "label", None) and "table" in str(item.label).lower():
            rows = _docling_table_to_rows(item)
            if not rows:
                continue
            prov = getattr(item, "prov", [None])
            page_no = getattr(prov[0], "page_no", None) if prov else None
            tables.append({"page": page_no, "rows": rows})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(tables, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
