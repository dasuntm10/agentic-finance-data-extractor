"""Convert ARCHITECTURE.md to a formatted Word document (ARCHITECTURE.docx).

Usage:
    python scripts/md_to_docx.py

Output: ARCHITECTURE.docx in the project root.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT     = Path(__file__).parent.parent
MD_FILE  = ROOT / "ARCHITECTURE.md"
OUT_FILE = ROOT / "ARCHITECTURE.docx"

# ── Design tokens ─────────────────────────────────────────────────────────────
C_H1       = RGBColor(0x1A, 0x3A, 0x5C)
C_H2       = RGBColor(0x2A, 0x5E, 0x8E)
C_H3       = RGBColor(0x44, 0x7A, 0xA0)
C_ICODE    = RGBColor(0xC0, 0x27, 0x47)
C_LINK     = RGBColor(0x1A, 0x6E, 0xC8)
C_TH_BG    = "DCE9F5"           # hex string for XML shading
C_ROW_ALT  = "F5F8FC"
C_CODE_BG  = "F4F4F4"
C_BORDER   = "B0C4DE"

FONT_BODY  = "Calibri"
FONT_HEAD  = "Calibri Light"
FONT_CODE  = "Consolas"

# ── XML helpers ───────────────────────────────────────────────────────────────

def _set_cell_shading(cell, fill_hex: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def _set_table_borders(tbl) -> None:
    tbl_pr = tbl._tbl.tblPr
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl._tbl.insert(0, tbl_pr)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        bd = OxmlElement(f"w:{side}")
        bd.set(qn("w:val"), "single")
        bd.set(qn("w:sz"), "4")
        bd.set(qn("w:space"), "0")
        bd.set(qn("w:color"), C_BORDER)
        borders.append(bd)
    tbl_pr.append(borders)


def _add_horizontal_rule(doc) -> None:
    para = doc.add_paragraph()
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "B0C4DE")
    pBdr.append(bottom)
    pPr.append(pBdr)
    para.paragraph_format.space_after = Pt(6)


def _add_toc(doc) -> None:
    """Insert a Word TOC field that updates on open."""
    heading = doc.add_paragraph("Table of Contents", style="Heading 1")
    heading.runs[0].font.color.rgb = C_H1

    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(6)
    run = para.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "(Right-click and select 'Update Field' to generate table of contents)"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x88)
    run.font.italic = True


def _add_hyperlink(para, text: str, url: str) -> None:
    part = para.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color_el = OxmlElement("w:color")
    color_el.set(qn("w:val"), "1A6EC8")
    u_el = OxmlElement("w:u")
    u_el.set(qn("w:val"), "single")
    rPr.append(color_el)
    rPr.append(u_el)
    t_el = OxmlElement("w:t")
    t_el.text = text
    new_run.append(rPr)
    new_run.append(t_el)
    hyperlink.append(new_run)
    para._p.append(hyperlink)


# ── Inline markdown parser ────────────────────────────────────────────────────

_INLINE_RE = re.compile(
    r"\*\*(.+?)\*\*"                    # **bold**
    r"|\*(.+?)\*"                        # *italic*
    r"|`([^`]+)`"                        # `code`
    r"|\[([^\]]+)\]\(([^\)]+)\)"         # [text](url)
    r"|__(.+?)__"                        # __bold__
    r"|_([^_]+)_"                        # _italic_
)


def add_inline(para, text: str, base_size: int = 11) -> None:
    """Parse inline markdown and add styled runs to the paragraph."""
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            para.add_run(text[pos:m.start()])
        bold1, ital1, code, link_txt, link_url, bold2, ital2 = m.groups()
        if bold1 or bold2:
            run = para.add_run(bold1 or bold2)
            run.bold = True
        elif ital1 or ital2:
            run = para.add_run(ital1 or ital2)
            run.italic = True
        elif code:
            run = para.add_run(code)
            run.font.name = FONT_CODE
            run.font.size = Pt(base_size - 1)
            run.font.color.rgb = C_ICODE
        elif link_txt and link_url:
            _add_hyperlink(para, link_txt, link_url)
        pos = m.end()
    if pos < len(text):
        para.add_run(text[pos:])


def add_inline_cell(cell, text: str) -> None:
    """Add inline-formatted text to a table cell, clearing the default empty para."""
    para = cell.paragraphs[0]
    add_inline(para, text, base_size=10)


# ── Document style setup ──────────────────────────────────────────────────────

def _setup_styles(doc: Document) -> None:
    # Normal
    normal = doc.styles["Normal"]
    normal.font.name = FONT_BODY
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = Pt(14)

    for style_name, level, color, size in [
        ("Heading 1", 1, C_H1, 16),
        ("Heading 2", 2, C_H2, 13),
        ("Heading 3", 3, C_H3, 12),
    ]:
        s = doc.styles[style_name]
        s.font.name = FONT_HEAD
        s.font.size = Pt(size)
        s.font.color.rgb = color
        s.font.bold = True
        s.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        s.paragraph_format.space_after  = Pt(6)
        s.paragraph_format.keep_with_next = True

    # List Bullet
    for sname in ("List Bullet", "List Bullet 2"):
        try:
            s = doc.styles[sname]
            s.font.name = FONT_BODY
            s.font.size = Pt(11)
            s.paragraph_format.space_after = Pt(3)
        except Exception:
            pass

    # List Number
    for sname in ("List Number", "List Number 2"):
        try:
            s = doc.styles[sname]
            s.font.name = FONT_BODY
            s.font.size = Pt(11)
            s.paragraph_format.space_after = Pt(3)
        except Exception:
            pass


# ── Cover page ────────────────────────────────────────────────────────────────

def _add_cover(doc: Document) -> None:
    # Big vertical spacer
    for _ in range(4):
        sp = doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(0)

    # Main title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Agentic Financial Data Extractor")
    r.font.name = FONT_HEAD
    r.font.size = Pt(28)
    r.font.color.rgb = C_H1
    r.font.bold = True
    title.paragraph_format.space_after = Pt(4)

    # Subtitle
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("Architecture & Implementation Plan")
    r2.font.name = FONT_HEAD
    r2.font.size = Pt(16)
    r2.font.color.rgb = C_H2
    r2.font.bold = False
    sub.paragraph_format.space_after = Pt(24)

    # Thin rule
    _add_horizontal_rule(doc)

    # Meta table
    meta = [
        ("Project",             "CreditSource — Agentic data extraction & credit scoring from Australian annual reports"),
        ("Author",              "Dasun"),
        ("Document scope",      "Optimal end-to-end design for Tasks 1, 2 (optional implementation), and 3 (optional risk model) of the CreditSource case study."),
        ("Constraints honoured","Python; reproducible via Poetry (pyproject.toml + poetry.lock); fully offline — no third-party hosted services at runtime. LLM reasoning runs on a local Qwen2.5-7B-Instruct served by Ollama (or vLLM), and embeddings come from BGE-small-en-v1.5 loaded via sentence-transformers. All other tooling (PDF parsing, OCR, orchestration, validation, scoring math) is also open-source and local."),
        ("Date",                date.today().strftime("%B %d, %Y")),
    ]

    tbl = doc.add_table(rows=len(meta), cols=2)
    tbl.style = "Table Grid"
    _set_table_borders(tbl)
    col_widths = [Inches(1.6), Inches(4.6)]

    for i, (key, val) in enumerate(meta):
        row = tbl.rows[i]
        row.cells[0].width = col_widths[0]
        row.cells[1].width = col_widths[1]
        _set_cell_shading(row.cells[0], "DCE9F5")

        kp = row.cells[0].paragraphs[0]
        kr = kp.add_run(key)
        kr.font.bold = True
        kr.font.name = FONT_BODY
        kr.font.size = Pt(10)

        vp = row.cells[1].paragraphs[0]
        vr = vp.add_run(val)
        vr.font.name = FONT_BODY
        vr.font.size = Pt(10)

    doc.add_page_break()


# ── Markdown block parser ─────────────────────────────────────────────────────

def parse_blocks(lines: list[str]) -> list[dict]:
    """Convert raw markdown lines into a list of typed block dicts."""
    blocks: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Skip the very first H1 title line (used in cover page)
        if line.startswith("# ") and i == 0:
            i += 1
            continue

        # Skip the front-matter block (Project:/Author:/etc lines at the top)
        stripped = line.strip()

        # Heading
        if line.startswith("#### "):
            blocks.append({"type": "h3", "text": line[5:].strip()})
            i += 1
            continue
        if line.startswith("### "):
            blocks.append({"type": "h2", "text": line[4:].strip()})
            i += 1
            continue
        if line.startswith("## "):
            blocks.append({"type": "h1", "text": line[3:].strip()})
            i += 1
            continue
        if line.startswith("# "):
            blocks.append({"type": "h1", "text": line[2:].strip()})
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1  # consume closing ```
            blocks.append({"type": "code", "lang": lang, "lines": code_lines})
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Table: collect all consecutive table lines
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            # Parse: first row = headers, second row = separator, rest = data
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append(cells)
            # Remove separator row (all dashes)
            data_rows = [r for r in rows if not all(re.fullmatch(r"[-: ]+", c) for c in r)]
            if data_rows:
                blocks.append({"type": "table", "rows": data_rows})
            continue

        # Blockquote
        if stripped.startswith("> "):
            bq_lines = []
            while i < n and lines[i].strip().startswith("> "):
                bq_lines.append(lines[i].strip()[2:])
                i += 1
            blocks.append({"type": "blockquote", "lines": bq_lines})
            continue

        # Bullet list item (with optional indented sub-items)
        bul_m = re.match(r"^( {0,3})([-*]) (.+)$", line)
        if bul_m:
            indent = len(bul_m.group(1))
            items = [{"level": 0 if indent < 4 else 1, "text": bul_m.group(3)}]
            i += 1
            while i < n:
                sub = lines[i]
                sub_bul = re.match(r"^( {2,})([-*]) (.+)$", sub)
                sub_num = re.match(r"^( {2,})(\d+)\. (.+)$", sub)
                if sub_bul:
                    items.append({"level": 1, "text": sub_bul.group(3), "style": "bullet"})
                    i += 1
                elif sub_num:
                    items.append({"level": 1, "text": sub_num.group(3), "style": "number"})
                    i += 1
                else:
                    break
            blocks.append({"type": "bullet_list", "items": items})
            continue

        # Numbered list item
        num_m = re.match(r"^( {0,3})(\d+)\. (.+)$", line)
        if num_m:
            indent = len(num_m.group(1))
            items = [{"level": 0 if indent < 4 else 1, "text": num_m.group(3)}]
            i += 1
            while i < n:
                sub = lines[i]
                sub_num = re.match(r"^( {2,})(\d+)\. (.+)$", sub)
                sub_bul = re.match(r"^( {2,})([-*]) (.+)$", sub)
                if sub_num:
                    items.append({"level": 1, "text": sub_num.group(3), "style": "number"})
                    i += 1
                elif sub_bul:
                    items.append({"level": 1, "text": sub_bul.group(3), "style": "bullet"})
                    i += 1
                else:
                    break
            blocks.append({"type": "num_list", "items": items})
            continue

        # Blank line
        if not stripped:
            blocks.append({"type": "blank"})
            i += 1
            continue

        # Regular paragraph
        para_lines = [line.rstrip()]
        i += 1
        while i < n and lines[i].strip() and not any([
            lines[i].startswith("#"),
            lines[i].strip().startswith("|"),
            lines[i].strip().startswith("```"),
            lines[i].strip().startswith("> "),
            re.match(r"^( {0,3})([-*]) ", lines[i]),
            re.match(r"^( {0,3})\d+\. ", lines[i]),
            re.fullmatch(r"-{3,}|\*{3,}|_{3,}", lines[i].strip()),
        ]):
            para_lines.append(lines[i].rstrip())
            i += 1
        blocks.append({"type": "para", "text": " ".join(para_lines)})

    return blocks


# ── Block renderer ────────────────────────────────────────────────────────────

def render_blocks(doc: Document, blocks: list[dict]) -> None:
    prev_blank = False

    for b in blocks:
        btype = b["type"]

        if btype == "blank":
            prev_blank = True
            continue

        if btype == "hr":
            _add_horizontal_rule(doc)
            prev_blank = False
            continue

        if btype == "h1":
            p = doc.add_paragraph(style="Heading 1")
            add_inline(p, b["text"])
            p.runs[0].font.color.rgb = C_H1 if p.runs else None
            prev_blank = False
            continue

        if btype == "h2":
            p = doc.add_paragraph(style="Heading 2")
            add_inline(p, b["text"])
            prev_blank = False
            continue

        if btype == "h3":
            p = doc.add_paragraph(style="Heading 3")
            add_inline(p, b["text"])
            prev_blank = False
            continue

        if btype == "para":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, b["text"])
            prev_blank = False
            continue

        if btype == "blockquote":
            for bline in b["lines"]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(1.0)
                p.paragraph_format.space_after = Pt(4)
                # Light grey left border
                pPr = p._p.get_or_add_pPr()
                pBdr = OxmlElement("w:pBdr")
                left = OxmlElement("w:left")
                left.set(qn("w:val"), "single")
                left.set(qn("w:sz"), "12")
                left.set(qn("w:space"), "6")
                left.set(qn("w:color"), "4A90D9")
                pBdr.append(left)
                pPr.append(pBdr)
                add_inline(p, bline)
                for run in p.runs:
                    run.font.italic = True
                    run.font.size = Pt(10.5)
            prev_blank = False
            continue

        if btype == "code":
            lines = b["lines"]
            # Strip leading/trailing blank lines
            while lines and not lines[0].strip():
                lines = lines[1:]
            while lines and not lines[-1].strip():
                lines = lines[:-1]
            for codeline in lines:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent  = Cm(0.5)
                p.paragraph_format.right_indent = Cm(0.5)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after  = Pt(0)
                # Shade the paragraph
                pPr = p._p.get_or_add_pPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), C_CODE_BG)
                pPr.append(shd)
                run = p.add_run(codeline if codeline else " ")
                run.font.name = FONT_CODE
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
            # add small space after the code block
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(4)
            prev_blank = False
            continue

        if btype == "table":
            rows = b["rows"]
            if not rows:
                continue
            ncols = max(len(r) for r in rows)
            tbl = doc.add_table(rows=len(rows), cols=ncols)
            tbl.style = "Table Grid"
            _set_table_borders(tbl)

            for ri, row in enumerate(rows):
                for ci in range(ncols):
                    cell = tbl.rows[ri].cells[ci]
                    text = row[ci] if ci < len(row) else ""
                    # Header row
                    if ri == 0:
                        _set_cell_shading(cell, C_TH_BG)
                        para = cell.paragraphs[0]
                        add_inline(para, text, base_size=10)
                        for run in para.runs:
                            run.font.bold = True
                            run.font.size = Pt(10)
                    else:
                        if ri % 2 == 0:
                            _set_cell_shading(cell, C_ROW_ALT)
                        para = cell.paragraphs[0]
                        add_inline(para, text, base_size=10)
                        for run in para.runs:
                            run.font.size = Pt(10)
            # space after table
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
            prev_blank = False
            continue

        if btype == "bullet_list":
            for item in b["items"]:
                lvl = item.get("level", 0)
                style_hint = item.get("style", "bullet")
                if style_hint == "number":
                    style = "List Number 2" if lvl > 0 else "List Number"
                else:
                    style = "List Bullet 2" if lvl > 0 else "List Bullet"
                p = doc.add_paragraph(style=style)
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, item["text"])
            prev_blank = False
            continue

        if btype == "num_list":
            for item in b["items"]:
                lvl = item.get("level", 0)
                style_hint = item.get("style", "number")
                if style_hint == "bullet":
                    style = "List Bullet 2" if lvl > 0 else "List Bullet"
                else:
                    style = "List Number 2" if lvl > 0 else "List Number"
                p = doc.add_paragraph(style=style)
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, item["text"])
            prev_blank = False
            continue

    # end of blocks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading {MD_FILE} …")
    raw = MD_FILE.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Skip the YAML-style front matter block at the top (lines 2-7 in the file)
    # These are the **Project:**, **Author:**, etc. lines — they go on the cover page.
    # We strip them from the body by removing lines that look like front-matter fields.
    body_lines: list[str] = []
    in_header = True
    for line in lines:
        if in_header:
            stripped = line.strip()
            # Front-matter lines start with **Project:**, **Author:**, etc.
            if re.match(r"^\*\*(Project|Author|Document scope|Constraints honoured)\*\*", stripped):
                continue
            if stripped in ("", "---"):
                continue
            if stripped.startswith("# "):
                in_header = False
        body_lines.append(line)

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.8)
        section.right_margin  = Cm(2.8)

    _setup_styles(doc)
    _add_cover(doc)
    _add_toc(doc)
    doc.add_page_break()

    blocks = parse_blocks(body_lines)
    render_blocks(doc, blocks)

    doc.save(OUT_FILE)
    print(f"Saved {OUT_FILE}  ({OUT_FILE.stat().st_size:,} bytes)")
    print("Open in Word and press Ctrl+A then F9 to update the Table of Contents.")


if __name__ == "__main__":
    main()
