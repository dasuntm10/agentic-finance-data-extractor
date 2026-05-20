"""Agent 2 — Section Locator.

Finds the page ranges of the P&L / Consolidated Statement of Comprehensive Income,
the notes block, and (optionally) the Statement of Financial Position. Uses a
deterministic ToC parser first; falls back to a structural-signature scan; falls
back to a Claude Haiku tiebreak only when multiple credible candidates exist.
"""
from __future__ import annotations

import logging
import re

from afde.llm.client import ToolChoice, get_llm
from afde.schemas import CompanyProfile, IngestedDocument, SectionMap

log = logging.getLogger(__name__)

PL_TITLE_PATTERNS = [
    re.compile(r"consolidated statement of (comprehensive income|profit (and|or) loss)", re.I),
    re.compile(r"statement of (comprehensive income|profit (and|or) loss)", re.I),
    re.compile(r"\bincome statement\b", re.I),
    re.compile(r"\bprofit (and|or) loss\b", re.I),
]
BS_TITLE_PATTERNS = [
    re.compile(r"statement of financial position", re.I),
    re.compile(r"balance sheet", re.I),
]
CF_TITLE_PATTERNS = [
    re.compile(r"statement of cash flows?", re.I),
]
NOTES_TITLE_PATTERNS = [
    re.compile(r"notes to (the )?(consolidated )?financial statements", re.I),
]

# Bank / financial-services signal words on the directors' report or front matter
FINANCIAL_SIGNALS = [
    "bank",
    "broker",
    "trading securities",
    "advisory fees",
    "net interest income",
    "repurchase agreement",
    "derivative financial instrument",
    "global markets",
]


def _classify_profile(doc: IngestedDocument) -> CompanyProfile:
    sample = " ".join(p.text.lower() for p in doc.pages[:15])
    hits = sum(1 for kw in FINANCIAL_SIGNALS if kw in sample)
    return CompanyProfile.FINANCIAL if hits >= 2 else CompanyProfile.CORPORATE


def _scan_toc(doc: IngestedDocument) -> dict[str, list[int]]:
    """Look for a 'Contents' / ToC page in the first 10 pages and parse it."""
    candidates: dict[str, list[int]] = {"pl": [], "bs": [], "cf": [], "notes": []}
    for page in doc.pages[:12]:
        text = page.text
        if not text or "page no" not in text.lower() and "contents" not in text.lower():
            # not obviously a ToC, but ToCs in our corpus don't always have "Contents"
            if not any(p.search(text) for p in PL_TITLE_PATTERNS):
                continue
        # Each non-empty line: try to extract trailing page number
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            m = re.search(r"(.+?)\s+(\d{1,3})\s*$", line_stripped)
            if not m:
                continue
            label, pno = m.group(1), int(m.group(2))
            if pno < 1 or pno > doc.page_count:
                continue
            label_l = label.lower()
            if any(p.search(label_l) for p in PL_TITLE_PATTERNS):
                candidates["pl"].append(pno)
            elif any(p.search(label_l) for p in BS_TITLE_PATTERNS):
                candidates["bs"].append(pno)
            elif any(p.search(label_l) for p in CF_TITLE_PATTERNS):
                candidates["cf"].append(pno)
            elif any(p.search(label_l) for p in NOTES_TITLE_PATTERNS):
                candidates["notes"].append(pno)
        if candidates["pl"]:  # found a ToC with at least the P&L; stop scanning
            break
    return candidates


def _scan_headings(doc: IngestedDocument) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"pl": [], "bs": [], "cf": [], "notes": []}
    for h in doc.headings:
        ht = h.text.lower()
        if any(p.search(ht) for p in PL_TITLE_PATTERNS):
            out["pl"].append(h.page)
        elif any(p.search(ht) for p in BS_TITLE_PATTERNS):
            out["bs"].append(h.page)
        elif any(p.search(ht) for p in CF_TITLE_PATTERNS):
            out["cf"].append(h.page)
        elif any(p.search(ht) for p in NOTES_TITLE_PATTERNS):
            out["notes"].append(h.page)
    return out


def _structural_pl_candidates(doc: IngestedDocument) -> list[int]:
    """Pages whose tables have a 'Note' column and a year column — likely primary statements."""
    out: list[int] = []
    for tbl in doc.tables:
        header_l = [c.lower() for c in tbl.header]
        has_note = any("note" == c.strip() or c.strip().startswith("note") for c in header_l)
        has_year = any(re.search(r"20\d{2}|19\d{2}", c) for c in tbl.header)
        if has_note and has_year:
            out.append(tbl.page)
    return out


def _pl_title_for_page(doc: IngestedDocument, page: int) -> str:
    """Find the closest preceding heading on that page or just above."""
    for h in doc.headings:
        if h.page == page:
            for p in PL_TITLE_PATTERNS:
                if p.search(h.text):
                    return h.text.strip()
    return "Statement of Profit or Loss / Comprehensive Income"


def run(doc: IngestedDocument) -> SectionMap:
    profile = _classify_profile(doc)
    toc = _scan_toc(doc)
    headings = _scan_headings(doc)
    structural = _structural_pl_candidates(doc)

    pl_pages = sorted(set(toc["pl"] + headings["pl"] + structural))
    bs_pages = sorted(set(toc["bs"] + headings["bs"]))
    cf_pages = sorted(set(toc["cf"] + headings["cf"]))
    notes_pages = sorted(set(toc["notes"] + headings["notes"]))

    # If multiple credible P&L pages exist and ToC didn't disambiguate, ask Haiku
    if len(pl_pages) > 1 and not toc["pl"]:
        pl_pages = [_llm_tiebreak(doc, pl_pages)]

    if not pl_pages:
        log.warning("No P&L page candidates found for %s", doc.source_path)
        pl_pages = []

    # Notes range: from first notes_pages entry (or P&L+2) to end of document
    if not notes_pages and pl_pages:
        notes_start = max(pl_pages) + 1
        notes_pages = list(range(notes_start, doc.page_count + 1))
    elif notes_pages:
        notes_start = min(notes_pages)
        notes_pages = list(range(notes_start, doc.page_count + 1))

    title = _pl_title_for_page(doc, pl_pages[0]) if pl_pages else "Profit or Loss"

    section_map = SectionMap(
        pl_pages=pl_pages,
        notes_pages=notes_pages,
        balance_sheet_pages=bs_pages,
        cashflow_pages=cf_pages,
        pl_title=title,
        company_profile=profile,
    )
    log.info(
        "Locate: pl=%s bs=%s cf=%s notes=%d pages profile=%s",
        pl_pages,
        bs_pages,
        cf_pages,
        len(notes_pages),
        profile,
    )
    return section_map


def _llm_tiebreak(doc: IngestedDocument, candidates: list[int]) -> int:
    llm = get_llm()
    excerpts = []
    for p in candidates[:5]:
        page = next((pg for pg in doc.pages if pg.page == p), None)
        if page:
            excerpts.append(f"--- page {p} ---\n{page.text[:1200]}")
    tool = ToolChoice(
        name="select_primary_pl_page",
        description="Choose the page number that contains the PRIMARY Statement of Profit or Loss / "
        "Consolidated Statement of Comprehensive Income (the table with line-item amounts, "
        "not a discussion or note).",
        input_schema={
            "type": "object",
            "properties": {
                "page": {"type": "integer", "enum": candidates},
                "reason": {"type": "string"},
            },
            "required": ["page"],
        },
    )
    try:
        result = llm.structured(
            tier="haiku",
            system="You disambiguate which page contains the primary financial statement.",
            user="\n\n".join(excerpts),
            tool=tool,
            max_tokens=256,
        )
        chosen = int(result.get("page") or candidates[0])
        log.info("Locate tiebreak (haiku) chose p.%d: %s", chosen, result.get("reason"))
        return chosen
    except Exception as e:
        log.warning("Locate tiebreak failed: %s — defaulting to first candidate %d", e, candidates[0])
        return candidates[0]
