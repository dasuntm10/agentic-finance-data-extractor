"""Generate architecture.html — interactive SVG pipeline diagram.

Standalone HTML file; no external dependencies.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "architecture.html"

# ── Layout constants (SVG units = px) ─────────────────────────────────────────
SVG_W      = 920
BOX_W      = 560
BOX_H      = 68
GAP        = 60       # vertical gap between boxes (arrows live here)
MAIN_X     = (SVG_W - BOX_W) // 2   # left edge of main chain boxes
CHAIN_CX   = MAIN_X + BOX_W // 2    # centre-x of chain

ORCH_H     = 56
ORCH_X     = 60
ORCH_W     = SVG_W - 120

RETRY_W    = 190
RETRY_H    = 72
RETRY_X    = MAIN_X - RETRY_W - 18

PADDING_TOP    = 90   # space for title
PADDING_BOTTOM = 120  # space for legend

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "orch":   {"fill": "#E8F1FC", "stroke": "#3A7BD5", "text": "#1A3A5C"},
    "ingest": {"fill": "#E8F6EE", "stroke": "#43A86A", "text": "#1A4A2A"},
    "parse":  {"fill": "#FDF3E3", "stroke": "#E0952A", "text": "#5A3A0A"},
    "valid":  {"fill": "#FDEAEA", "stroke": "#CC4444", "text": "#5A1414"},
    "score":  {"fill": "#EDEAFB", "stroke": "#6A5ACD", "text": "#2A1A6A"},
    "report": {"fill": "#E4F5F4", "stroke": "#2E9E97", "text": "#0A3A38"},
    "out":    {"fill": "#F0F0F4", "stroke": "#888899", "text": "#333344"},
    "retry":  {"fill": "#FDEAEA", "stroke": "#CC4444", "text": "#5A1414"},
}

AGENTS = [
    ("① Ingestion & Triage",
     "pypdf · PyMuPDF · Docling / TableFormer · PaddleOCR PP-OCRv4 (OCR fallback)",
     "ingest"),
    ("② Section Locator",
     "ToC title extraction · heading regex · structural signature · Qwen2.5-7B tiebreak",
     "ingest"),
    ("③ Statement Parser",
     "TableFormer table selection · accounting-aware numeric.py · 3-tier FY detection",
     "parse"),
    ("④ Note Resolver",
     "NoteIndex (number-based) · FAISS fuzzy fallback · Docling note blocks",
     "parse"),
    ("⑤ Normaliser &amp; Canonical Mapper",
     "Rule match → BGE-small-en-v1.5 embeddings (cosine) → Qwen2.5-7B enum fallback",
     "parse"),
    ("⑥ Reconciliation / Self-Check",
     "sum · sign (PBT/tax/NP) · OCI roll-up · note linkage · year completeness | no LLM",
     "valid"),
    ("⑦ Ratio Engine",
     "profitability · liquidity · leverage · coverage · YoY delta | no LLM",
     "score"),
    ("⑧ Scoring Agent",
     "piecewise-linear sub-scores · weighted composite (YAML) · 3 anomaly detectors",
     "score"),
    ("⑨ Risk Report Writer",
     "Qwen2.5-7B narrative · numeric fact-check · 1 regen attempt · template fallback · WeasyPrint PDF",
     "report"),
]


def svg_rect(x, y, w, h, rx, fill, stroke, opacity=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" opacity="{opacity}"/>')


def svg_text(x, y, text, size, weight, color, anchor="middle"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'font-family="Segoe UI, Calibri, Arial, sans-serif">{text}</text>')


def svg_sub(x, y, text, color):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="11" fill="{color}" font-style="italic" '
            f'font-family="Segoe UI, Calibri, Arial, sans-serif">{text}</text>')


def arrow_down(x, y_from, y_to, color="#222233", dashed=False):
    dash = 'stroke-dasharray="8,5"' if dashed else ""
    mid_y = (y_from + y_to) // 2
    return (
        f'<defs><marker id="ah_{y_from}" markerWidth="10" markerHeight="7" '
        f'refX="9" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}"/></marker></defs>'
        f'<line x1="{x}" y1="{y_from}" x2="{x}" y2="{y_to}" '
        f'stroke="{color}" stroke-width="2.5" {dash} '
        f'marker-end="url(#ah_{y_from})"/>'
    )


def build_svg():
    parts = []
    parts.append('<defs>')
    parts.append("""
  <marker id="arrowhead" markerWidth="10" markerHeight="7"
          refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#222233"/>
  </marker>
  <marker id="arrowhead_red" markerWidth="10" markerHeight="7"
          refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#CC4444"/>
  </marker>
  <marker id="arrowhead_red_start" markerWidth="10" markerHeight="7"
          refX="1" refY="3.5" orient="auto-start-reverse">
    <polygon points="0 0, 10 3.5, 0 7" fill="#CC4444"/>
  </marker>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#00000018"/>
  </filter>
    """)
    parts.append('</defs>')

    y = PADDING_TOP

    # ── Orchestrator ──────────────────────────────────────────────────────────
    oc = COLORS["orch"]
    parts.append(f'<g filter="url(#shadow)">')
    parts.append(svg_rect(ORCH_X, y, ORCH_W, ORCH_H, 10,
                          oc["fill"], oc["stroke"]))
    parts.append('</g>')
    parts.append(svg_text(SVG_W // 2, y + 22, "LangGraph Orchestrator",
                          14, "bold", oc["text"]))
    parts.append(svg_sub(SVG_W // 2, y + 40,
                         "StateGraph · typed PipelineState · SQLite checkpointer (resume on crash)",
                         "#555566"))

    # Arrow orch → first agent
    y_orch_bot = y + ORCH_H
    y_first_top = y_orch_bot + GAP
    parts.append(
        f'<line x1="{CHAIN_CX}" y1="{y_orch_bot}" x2="{CHAIN_CX}" y2="{y_first_top - 5}" '
        f'stroke="#222233" stroke-width="2.5" marker-end="url(#arrowhead)"/>'
    )

    y = y_first_top
    box_tops = {}
    box_bots = {}
    box_mids_svg = {}

    # ── Agent boxes ───────────────────────────────────────────────────────────
    for i, (title, sub, ckey) in enumerate(AGENTS):
        c = COLORS[ckey]
        parts.append(f'<g filter="url(#shadow)">')
        parts.append(svg_rect(MAIN_X, y, BOX_W, BOX_H, 10,
                              c["fill"], c["stroke"]))
        parts.append('</g>')
        parts.append(svg_text(CHAIN_CX, y + 26, title, 13, "bold", c["text"]))
        parts.append(svg_sub(CHAIN_CX, y + 46, sub, "#444455"))

        box_tops[i] = y
        box_bots[i] = y + BOX_H
        box_mids_svg[i] = y + BOX_H // 2

        if i < len(AGENTS) - 1:
            y_from = y + BOX_H
            y_to   = y + BOX_H + GAP
            parts.append(
                f'<line x1="{CHAIN_CX}" y1="{y_from}" x2="{CHAIN_CX}" y2="{y_to - 5}" '
                f'stroke="#222233" stroke-width="2.5" marker-end="url(#arrowhead)"/>'
            )
            # "pass" label after reconcile (index 5)
            if i == 5:
                parts.append(
                    f'<text x="{CHAIN_CX + 12}" y="{y_from + GAP // 2 + 5}" '
                    f'font-size="12" fill="#2E7D32" font-style="italic" '
                    f'font-family="Segoe UI, Calibri, Arial, sans-serif">pass</text>'
                )

        y += BOX_H + GAP

    # ── Outputs box ───────────────────────────────────────────────────────────
    OUT_BOX_W = BOX_W - 120
    OUT_BOX_X = MAIN_X + 60
    oc2 = COLORS["out"]
    parts.append(f'<g filter="url(#shadow)">')
    parts.append(svg_rect(OUT_BOX_X, y, OUT_BOX_W, 60, 10,
                          oc2["fill"], oc2["stroke"]))
    parts.append('</g>')
    parts.append(svg_text(CHAIN_CX, y + 24, "outputs/&lt;company&gt;/",
                          12, "bold", oc2["text"]))
    parts.append(svg_sub(CHAIN_CX, y + 44,
                         "extraction.json  ·  risk_report.md  ·  risk_report.pdf",
                         "#555566"))

    SVG_H = y + 60 + PADDING_BOTTOM

    # ── Retry loop ────────────────────────────────────────────────────────────
    rec_mid_y = box_mids_svg[5]
    rec_left_x = MAIN_X
    ret_cx = RETRY_X + RETRY_W // 2
    ret_mid_y = rec_mid_y

    # Draw retry box at same vertical band as reconcile
    RETRY_Y_SVG = rec_mid_y - RETRY_H // 2
    rc = COLORS["retry"]
    parts.append(f'<g filter="url(#shadow)">')
    parts.append(svg_rect(RETRY_X, RETRY_Y_SVG, RETRY_W, RETRY_H, 10,
                          rc["fill"], rc["stroke"], 0.95))
    parts.append('</g>')
    parts.append(svg_text(ret_cx, RETRY_Y_SVG + 28, "retry_parse",
                          12, "bold", rc["text"]))
    parts.append(svg_sub(ret_cx, RETRY_Y_SVG + 48, "↑ OCR DPI · stricter TableFormer",
                         "#774444"))

    # Arrow: reconcile left → retry box (curved)
    ctrl_x = rec_left_x - 80
    ctrl_y = rec_mid_y
    ret_right = RETRY_X + RETRY_W
    parts.append(
        f'<path d="M {rec_left_x} {rec_mid_y} '
        f'Q {ctrl_x} {ctrl_y} {ret_right + 5} {ret_mid_y}" '
        f'stroke="#CC4444" stroke-width="2.2" fill="none" '
        f'marker-end="url(#arrowhead_red)"/>'
    )
    parts.append(
        f'<text x="{ctrl_x - 5}" y="{rec_mid_y - 10}" text-anchor="end" '
        f'font-size="11" fill="#CC4444" font-style="italic" '
        f'font-family="Segoe UI, Calibri, Arial, sans-serif">fail (≤1 retry)</text>'
    )

    # Arrow: retry box top → parse box left (index 2) (curved dashed)
    parse_top_y = box_tops[2]
    parse_left  = MAIN_X
    # Path: go up from retry box and curve to parse
    parts.append(
        f'<path d="M {ret_cx} {RETRY_Y_SVG} '
        f'C {ret_cx} {parse_top_y - 40}, {parse_left - 40} {parse_top_y + BOX_H//2}, '
        f'{parse_left + 5} {parse_top_y + BOX_H//2}" '
        f'stroke="#CC4444" stroke-width="2.0" fill="none" '
        f'stroke-dasharray="8,5" '
        f'marker-end="url(#arrowhead_red)"/>'
    )
    parts.append(
        f'<text x="{RETRY_X - 5}" y="{(RETRY_Y_SVG + parse_top_y) // 2}" '
        f'text-anchor="end" font-size="11" fill="#CC4444" font-style="italic" '
        f'font-family="Segoe UI, Calibri, Arial, sans-serif">re-parse</text>'
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        ("Ingestion / Location",    COLORS["ingest"]["stroke"], COLORS["ingest"]["fill"]),
        ("Parsing / Normalisation", COLORS["parse"]["stroke"],  COLORS["parse"]["fill"]),
        ("Validation / Retry",      COLORS["valid"]["stroke"],  COLORS["valid"]["fill"]),
        ("Ratios / Scoring",        COLORS["score"]["stroke"],  COLORS["score"]["fill"]),
        ("Reporting / Output",      COLORS["report"]["stroke"], COLORS["report"]["fill"]),
    ]
    leg_x = MAIN_X
    leg_y = SVG_H - PADDING_BOTTOM + 20
    for li, (label, stroke, fill) in enumerate(legend_items):
        ix = leg_x + li * 168
        parts.append(f'<rect x="{ix}" y="{leg_y}" width="18" height="14" rx="3" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="{ix + 24}" y="{leg_y + 12}" font-size="11" '
                     f'fill="#333344" font-family="Segoe UI, Calibri, Arial, sans-serif">'
                     f'{label}</text>')

    svg_body = "\n".join(parts)

    return f'''<svg xmlns="http://www.w3.org/2000/svg"
     width="{SVG_W}" height="{SVG_H}"
     viewBox="0 0 {SVG_W} {SVG_H}">
  <rect width="{SVG_W}" height="{SVG_H}" fill="#F5F6FA"/>

  <!-- Title -->
  <text x="{SVG_W//2}" y="38" text-anchor="middle"
        font-size="20" font-weight="bold" fill="#111122"
        font-family="Segoe UI, Calibri, Arial, sans-serif">
    Agentic Financial Data Extractor
  </text>
  <text x="{SVG_W//2}" y="62" text-anchor="middle"
        font-size="12" fill="#666677" font-style="italic"
        font-family="Segoe UI, Calibri, Arial, sans-serif">
    Sequential 9-Agent Pipeline · LangGraph StateGraph · Fully Offline
  </text>

{svg_body}
</svg>'''


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Agentic Financial Data Extractor — Architecture</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #ECEDF2;
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px 64px;
      min-height: 100vh;
    }}
    header {{
      max-width: 960px;
      width: 100%;
      margin-bottom: 24px;
    }}
    header h1 {{
      font-size: 22px;
      color: #1A3A5C;
      margin-bottom: 4px;
    }}
    header p {{
      font-size: 13px;
      color: #666677;
    }}
    .card {{
      background: #FFFFFF;
      border-radius: 14px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.10);
      padding: 28px;
      max-width: 960px;
      width: 100%;
      overflow-x: auto;
    }}
    .card svg {{
      display: block;
      margin: 0 auto;
      max-width: 100%;
    }}
    footer {{
      margin-top: 20px;
      font-size: 11px;
      color: #999;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Architecture Diagram — Agentic Financial Data Extractor</h1>
    <p>Project: CreditSource &nbsp;·&nbsp; Author: Dasun &nbsp;·&nbsp;
       LangGraph StateGraph &nbsp;·&nbsp; 9-Agent Sequential Pipeline &nbsp;·&nbsp; Fully Offline</p>
  </header>
  <div class="card">
    {svg}
  </div>
  <footer>Generated from ARCHITECTURE.md · Fully offline pipeline · All weights local</footer>
</body>
</html>"""


def main():
    svg = build_svg()
    html = HTML_TEMPLATE.format(svg=svg)
    OUT.write_text(html, encoding="utf-8")
    print(f"Saved {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
