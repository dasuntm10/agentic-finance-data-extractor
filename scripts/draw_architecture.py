"""Generate architecture.png — high-level sequential pipeline diagram."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 28
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#F5F6FA")

# ── Palette ───────────────────────────────────────────────────────────────────
C_ORCH   = "#3A7BD5"
C_INGEST = "#43A86A"
C_PARSE  = "#E0952A"
C_VALID  = "#CC4444"
C_SCORE  = "#6A5ACD"
C_REPORT = "#2E9E97"
C_OUT    = "#777788"
C_ARROW  = "#222233"
C_RETRY  = "#CC4444"

BOX_W   = 9.0
BOX_H   = 1.10
GAP     = 0.90
CX      = (FIG_W - BOX_W) / 2        # left edge of main chain  = 4.5
CHAIN_X = CX + BOX_W / 2             # centre-x                  = 9.0

# retry box — positioned well inside the left margin
RETRY_W  = 3.2
RETRY_H  = 1.0
RETRY_CX = CX - RETRY_W - 0.8        # = 4.5 - 3.2 - 0.8 = 0.5  (safe)


def draw_box(x, y, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.12", lw=2.0,
                 edgecolor=color, facecolor=color, alpha=0.13, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.12", lw=2.0,
                 edgecolor=color, facecolor="none", zorder=3))
    ty = y + h / 2 + (0.17 if subtitle else 0)
    ax.text(x + w / 2, ty, title,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#111122", zorder=4)
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.23, subtitle,
                ha="center", va="center", fontsize=8.0,
                color="#444455", style="italic", zorder=4)


def draw_arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.4, dashed=False):
    style = "Simple,head_width=11,head_length=9" if not dashed else \
            "Simple,head_width=9,head_length=7"
    arrow = FancyArrowPatch(
        posA=(x1, y1), posB=(x2, y2),
        arrowstyle=style,
        color=color, lw=lw,
        linestyle=(0, (5, 4)) if dashed else "solid",
        connectionstyle="arc3,rad=0",
        zorder=5, shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arrow)


def edge_label(x, y, text, color="#2E7D32", ha="left", size=9.0):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=size, color=color, style="italic", zorder=6)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.55,
        "Agentic Financial Data Extractor",
        ha="center", va="center", fontsize=16,
        fontweight="bold", color="#111122")
ax.text(FIG_W / 2, FIG_H - 1.15,
        "Sequential 9-Agent Pipeline   |   LangGraph StateGraph   |   Fully Offline",
        ha="center", va="center", fontsize=10,
        color="#555566", style="italic")

# ── Orchestrator ──────────────────────────────────────────────────────────────
ORCH_Y = FIG_H - 2.6
draw_box(1.5, ORCH_Y, FIG_W - 3.0, 0.85,
         "LangGraph Orchestrator",
         "StateGraph  |  typed PipelineState  |  SQLite checkpointer (resume on crash)",
         C_ORCH)

# Arrow: orchestrator -> first agent
CHAIN_START = ORCH_Y - GAP - BOX_H
draw_arrow(CHAIN_X, ORCH_Y, CHAIN_X, CHAIN_START + BOX_H + GAP * 0.05)

# ── Agent chain ───────────────────────────────────────────────────────────────
agents = [
    ("1.  Ingestion & Triage",
     "pypdf  |  PyMuPDF  |  Docling / TableFormer  |  PaddleOCR PP-OCRv4 (OCR fallback)",
     C_INGEST),
    ("2.  Section Locator",
     "ToC title extraction  |  heading regex  |  structural signature  |  Qwen2.5-7B tiebreak",
     C_INGEST),
    ("3.  Statement Parser",
     "TableFormer table selection  |  accounting-aware numeric.py  |  3-tier FY detection",
     C_PARSE),
    ("4.  Note Resolver",
     "NoteIndex (number-based)  |  FAISS fuzzy fallback  |  Docling note blocks",
     C_PARSE),
    ("5.  Normaliser & Canonical Mapper",
     "Rule match  ->  BGE-small-en-v1.5 embeddings (cosine)  ->  Qwen2.5-7B enum fallback",
     C_PARSE),
    ("6.  Reconciliation / Self-Check",
     "sum  |  sign (PBT/tax/NP)  |  OCI roll-up  |  note linkage  |  year completeness   (no LLM)",
     C_VALID),
    ("7.  Ratio Engine",
     "profitability  |  liquidity  |  leverage  |  coverage  |  YoY delta   (no LLM)",
     C_SCORE),
    ("8.  Scoring Agent",
     "piecewise-linear sub-scores  |  weighted composite (YAML)  |  3 anomaly detectors",
     C_SCORE),
    ("9.  Risk Report Writer",
     "Qwen2.5-7B narrative  |  numeric fact-check  |  1 regen attempt  |  template fallback  |  WeasyPrint PDF",
     C_REPORT),
]

y_cur = CHAIN_START
box_bottoms = {}
box_mids    = {}

for i, (title, sub, color) in enumerate(agents):
    draw_box(CX, y_cur, BOX_W, BOX_H, title, sub, color)
    box_bottoms[i] = y_cur
    box_mids[i]    = y_cur + BOX_H / 2

    if i < len(agents) - 1:
        y_from = y_cur
        y_to   = y_cur - GAP
        draw_arrow(CHAIN_X, y_from, CHAIN_X, y_to)
        if i == 5:
            edge_label(CHAIN_X + 0.25, y_from - GAP / 2, "pass", "#2E7D32", size=9.5)

    y_cur -= BOX_H + GAP

# ── Outputs box ───────────────────────────────────────────────────────────────
OUT_Y = y_cur + GAP - 0.15
draw_box(CX + 1.2, OUT_Y, BOX_W - 2.4, 0.80,
         "outputs/<company>/",
         "extraction.json   |   risk_report.md   |   risk_report.pdf",
         C_OUT)

# ── Retry loop ────────────────────────────────────────────────────────────────
# reconcile = index 5;  parse = index 2
RETRY_Y = box_mids[5] - RETRY_H / 2

draw_box(RETRY_CX, RETRY_Y, RETRY_W, RETRY_H,
         "retry_parse",
         "raise OCR DPI\nstricter TableFormer",
         C_RETRY)

# reconcile left-side -> retry box
ax.annotate("",
            xy=(RETRY_CX + RETRY_W, RETRY_Y + RETRY_H / 2),
            xytext=(CX, box_mids[5]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_RETRY, lw=2.0,
                mutation_scale=18,
                connectionstyle="arc3,rad=0.30",
            ), zorder=5)
edge_label(CX - 0.2, box_mids[5] + 0.50,
           "fail  (<=1 retry)", C_RETRY, ha="right", size=9.0)

# retry box top -> left side of parse (index 2)
parse_left_x = CX
parse_mid_y  = box_mids[2]
ax.annotate("",
            xy=(parse_left_x, parse_mid_y),
            xytext=(RETRY_CX + RETRY_W / 2, RETRY_Y + RETRY_H),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_RETRY, lw=1.8,
                mutation_scale=16,
                linestyle=(0, (5, 4)),
                connectionstyle="arc3,rad=-0.25",
            ), zorder=5)
mid_label_y = (RETRY_Y + RETRY_H + parse_mid_y) / 2
edge_label(RETRY_CX + RETRY_W / 2 - 0.1, mid_label_y,
           "re-parse", C_RETRY, ha="center", size=9.0)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INGEST, alpha=0.25, edgecolor=C_INGEST, lw=1.5,
                   label="Ingestion / Location"),
    mpatches.Patch(facecolor=C_PARSE,  alpha=0.25, edgecolor=C_PARSE,  lw=1.5,
                   label="Parsing / Normalisation"),
    mpatches.Patch(facecolor=C_VALID,  alpha=0.25, edgecolor=C_VALID,  lw=1.5,
                   label="Validation / Retry"),
    mpatches.Patch(facecolor=C_SCORE,  alpha=0.25, edgecolor=C_SCORE,  lw=1.5,
                   label="Ratios / Scoring"),
    mpatches.Patch(facecolor=C_REPORT, alpha=0.25, edgecolor=C_REPORT, lw=1.5,
                   label="Reporting / Output"),
]
ax.legend(handles=legend_items, loc="lower right",
          bbox_to_anchor=(1.0, 0.01), fontsize=10,
          framealpha=0.93, edgecolor="#CCCCCC",
          title="Agent category", title_fontsize=9.5)

plt.tight_layout(pad=0.5)
out = "architecture.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
