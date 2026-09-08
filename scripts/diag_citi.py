"""Throwaway diagnostic: why does docling segfault on Citigroup, and what does the
PyMuPDF-only path give the Locate agent? Uses only PyMuPDF (no docling) so it can't crash."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

import fitz  # PyMuPDF

PDF = "data/CITIGROUP.pdf"

doc = fitz.open(PDF)
print(f"=== {PDF}: {doc.page_count} pages ===\n")

print("--- page dimensions (pt) and largest embedded image (px) ---")
for i, page in enumerate(doc):
    r = page.rect
    imgs = page.get_images(full=True)
    max_px = 0
    max_dim = ""
    for im in imgs:
        w, h = im[2], im[3]
        if w * h > max_px:
            max_px = w * h
            max_dim = f"{w}x{h}"
    flag = "  <-- HUGE" if (r.width * r.height > 2_000_000 or max_px > 20_000_000) else ""
    print(f"p{i+1:>3}: {r.width:7.0f} x {r.height:7.0f} pt | {len(imgs)} imgs | largest={max_dim or '-':>12} ({max_px/1e6:5.1f} MP){flag}")

print("\n--- pages whose text mentions an income/P&L statement title ---")
KEYS = ["statement of comprehensive income", "income statement", "statement of income",
        "profit or loss", "profit and loss", "statement of operations"]
for i, page in enumerate(doc):
    t = (page.get_text("text") or "").lower()
    hits = [k for k in KEYS if k in t]
    if hits:
        # show the line containing the first hit
        line = next((ln.strip() for ln in (page.get_text("text") or "").splitlines()
                     if any(k in ln.lower() for k in hits)), "")
        print(f"p{i+1:>3}: {hits}  | line: {line[:80]!r}")

doc.close()

print("\n--- what load_with_pymupdf() gives Locate (headings) ---")
from afde.parsing.pdf_loader import load_with_pymupdf  # noqa: E402

pages, headings = load_with_pymupdf(PDF)
print(f"pages={len(pages)} headings={len(headings)}")
stmt_headings = [h for h in headings
                 if any(k in h.text.lower() for k in
                        ["income", "comprehensive", "profit", "loss", "operations", "financial position", "balance"])]
print(f"statement-like headings captured: {len(stmt_headings)}")
for h in stmt_headings[:25]:
    print(f"  p{h.page:>3} L{h.level}: {h.text!r}")
