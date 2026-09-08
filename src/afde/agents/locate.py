"""Agent 2 — Section Locator.

Finds the page ranges of the P&L / Consolidated Statement of Comprehensive Income,
the notes block, and (optionally) the Statement of Financial Position.

Navigation strategy (offset-safe):
  ToC page numbers are NOT used for navigation — document-printed page numbers
  differ from physical PDF page indices by an unpredictable offset (cover pages,
  un-numbered front matter, etc.). Instead:

  1. _scan_toc_titles()  — reads the ToC for *section title strings only*, not numbers.
  2. _scan_headings()    — finds physical PDF pages by matching heading text against
                           regex patterns AND the exact titles extracted from the ToC.
                           Physical page numbers come directly from PyMuPDF's page
                           objects, so there is no offset ambiguity.
  3. _structural_pl_candidates() — independently confirms candidates by table layout
                           (Note column + year column), entirely ignoring titles.
  4. Results from (2) and (3) are merged; an LLM tiebreak is invoked only when
     multiple distinct physical pages remain after merging.
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
    re.compile(r"\bstatement of income\b", re.I),
    re.compile(r"\bstatement of operations\b", re.I),
    re.compile(r"\bstatement of earnings\b", re.I),
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


def _scan_toc_titles(doc: IngestedDocument) -> dict[str, list[str]]:
    """Extract section title strings from the ToC. Returns titles only — no page numbers.

    ToC page numbers are unreliable because document-printed pagination differs
    from physical PDF page indices by an unknown offset. We take the title labels
    and use them to strengthen the heading search with exact-string matching.
    """
    titles: dict[str, list[str]] = {"pl": [], "bs": [], "cf": [], "notes": []}
    for page in doc.pages[:12]:
        text = page.text
        if not text:
            continue
        text_l = text.lower()
        if "contents" not in text_l and "page no" not in text_l:
            continue
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Strip trailing page number (e.g. "Consolidated statement ...   8")
            label = re.sub(r"\s+\d{1,3}\s*$", "", line_stripped).strip()
            if not label:
                continue
            label_l = label.lower()
            if any(p.search(label_l) for p in PL_TITLE_PATTERNS):
                titles["pl"].append(label)
            elif any(p.search(label_l) for p in BS_TITLE_PATTERNS):
                titles["bs"].append(label)
            elif any(p.search(label_l) for p in CF_TITLE_PATTERNS):
                titles["cf"].append(label)
            elif any(p.search(label_l) for p in NOTES_TITLE_PATTERNS):
                titles["notes"].append(label)
        if titles["pl"]:
            break
    if titles["pl"]:
        log.debug("ToC titles found: %s", titles)
    return titles


def _scan_headings(
    doc: IngestedDocument,
    toc_titles: dict[str, list[str]] | None = None,
) -> dict[str, list[int]]:
    """Return physical page numbers for each section type.

    Matches heading text against regex patterns. If ToC titles were found, also
    matches against those exact strings — this lets an unusual title like
    'Consolidated Statement of Income and Retained Earnings' be found even if
    it doesn't match any regex, as long as the ToC named it.

    Physical page numbers come from PyMuPDF heading objects — no offset involved.
    """
    out: dict[str, list[int]] = {"pl": [], "bs": [], "cf": [], "notes": []}

    # Build normalised exact-match sets from ToC titles
    exact: dict[str, set[str]] = {"pl": set(), "bs": set(), "cf": set(), "notes": set()}
    if toc_titles:
        for key, title_list in toc_titles.items():
            for t in title_list:
                exact[key].add(t.lower().strip())

    section_patterns = [
        ("pl",    PL_TITLE_PATTERNS),
        ("bs",    BS_TITLE_PATTERNS),
        ("cf",    CF_TITLE_PATTERNS),
        ("notes", NOTES_TITLE_PATTERNS),
    ]

    for h in doc.headings:
        ht = h.text.lower().strip()
        for key, patterns in section_patterns:
            if any(p.search(ht) for p in patterns) or ht in exact[key]:
                out[key].append(h.page)
                break  # a heading belongs to at most one section type

    return out


def _structural_pl_candidates(doc: IngestedDocument) -> list[int]:
    """Pages whose tables have a 'Note' column and a year column — likely primary statements.

    Entirely title-agnostic: works regardless of what the statement is called or
    what page number the ToC printed.
    """
    out: list[int] = []
    for tbl in doc.tables:
        header_l = [c.lower() for c in tbl.header]
        has_note = any("note" == c.strip() or c.strip().startswith("note") for c in header_l)
        has_year = any(re.search(r"20\d{2}|19\d{2}", c) for c in tbl.header)
        if has_note and has_year:
            out.append(tbl.page)
    return out


def _scan_page_text_titles(doc: IngestedDocument) -> dict[str, list[int]]:
    """Fallback page detection: scan page *text* for statement titles.

    Locate's primary path matches font-sized headings, but some reports (notably
    bank statements) render statement titles in body-sized font, so they never
    become headings and `_scan_headings` returns nothing. Here we match a title
    pattern *anchored at the start* of a short line: that catches the real title
    ('Consolidated statement of comprehensive income') while rejecting prose
    mentions ('Financial assets at fair value through profit or loss').
    """
    out: dict[str, list[int]] = {"pl": [], "bs": [], "cf": [], "notes": []}
    section_patterns = [
        ("pl", PL_TITLE_PATTERNS),
        ("bs", BS_TITLE_PATTERNS),
        ("cf", CF_TITLE_PATTERNS),
        ("notes", NOTES_TITLE_PATTERNS),
    ]
    for page in doc.pages:
        for raw_line in (page.text or "").splitlines():
            line = raw_line.strip()
            if not line or len(line) > 70:
                continue
            for key, patterns in section_patterns:
                if any(p.match(line) for p in patterns):
                    if page.page not in out[key]:
                        out[key].append(page.page)
                    break
    return out


def _notes_run_start(notes_pages: list[int]) -> int | None:
    """First page of the first *consecutive run* of notes pages.

    Notes pages are detected by their running header, which can also match a stray
    page (e.g. a contents/reference page) far from the real notes block. Requiring a
    run of three consecutive pages ignores those strays and finds where notes truly begin.
    """
    s = sorted(set(notes_pages))
    for p in s:
        if (p + 1) in s and (p + 2) in s:
            return p
    return s[0] if s else None


def candidate_statement_pages(doc: IngestedDocument, margin: int = 1) -> set[int]:
    """Cheap (no-LLM, no-table) guess at which physical pages hold the primary
    statements, used to target Docling extraction at just those pages instead of the
    whole document.

    Uses heading text + a page-text title scan, keeps only candidates *before* the
    notes block (primary statements always precede the notes), and pads each by
    ``margin`` pages. Returns an empty set when nothing matches — the caller should
    then fall back to extracting the whole document.
    """
    toc_titles = _scan_toc_titles(doc)
    headings = _scan_headings(doc, toc_titles=toc_titles)
    text = _scan_page_text_titles(doc)

    notes_start = _notes_run_start((headings["notes"] or []) + (text["notes"] or []))

    def before_notes(p: int) -> bool:
        return notes_start is None or p < notes_start

    pl = {p for p in set(headings["pl"]) | set(text["pl"]) if before_notes(p)}
    bs = {p for p in set(headings["bs"]) | set(text["bs"]) if before_notes(p)}
    seeds = pl | bs
    if not seeds:
        return set()

    out: set[int] = set()
    for p in seeds:
        for q in range(p - margin, p + margin + 1):
            if 1 <= q <= doc.page_count:
                out.add(q)
    return out


def _expand_to_continuation_pages(
    doc: IngestedDocument,
    start_page: int,
    boundary_pages: frozenset[int] = frozenset(),
    max_pages: int = 3,
) -> list[int]:
    """Walk forward from start_page, adding pages that continue the same table.

    A page is a continuation if it has no heading that starts a new named section
    and contains at least one numeric value (i.e. still has table row data).
    Stops as soon as a new section starts or a page has no numeric content.

    ``boundary_pages`` are known starts of *other* sections (balance sheet, cash
    flow, notes) detected by text-scan — essential when the PDF has no font-sized
    headings, otherwise expansion runs all the way to the end of the document.
    ``max_pages`` caps the span as a final safety net (primary statements that
    wrap do so over very few pages).
    """
    pages = [start_page]
    section_starters = BS_TITLE_PATTERNS + CF_TITLE_PATTERNS + NOTES_TITLE_PATTERNS + PL_TITLE_PATTERNS

    # Pages that open a new named section: from headings AND the text-scan boundaries.
    new_section_pages: set[int] = {p for p in boundary_pages if p > start_page}
    for h in doc.headings:
        if h.page > start_page:
            if any(p.search(h.text) for p in section_starters):
                new_section_pages.add(h.page)

    for page in doc.pages:
        if page.page <= start_page:
            continue
        if len(pages) >= max_pages:
            break
        if page.page in new_section_pages:
            break
        if not re.search(r"\d[\d,\.]+", page.text or ""):
            break
        pages.append(page.page)

    if len(pages) > 1:
        log.debug("P&L continuation detected: pages %s", pages)
    return pages


def _pl_title_for_page(doc: IngestedDocument, page: int) -> str:
    """Return the heading text on that physical page that matches a P&L pattern."""
    for h in doc.headings:
        if h.page == page:
            for p in PL_TITLE_PATTERNS:
                if p.search(h.text):
                    return h.text.strip()
    return "Statement of Profit or Loss / Comprehensive Income"


def run(doc: IngestedDocument) -> SectionMap:
    profile = _classify_profile(doc)

    # Step 1: extract titles from ToC (no page numbers used)
    toc_titles = _scan_toc_titles(doc)

    # Step 2: find physical pages via heading text + ToC title hints
    headings = _scan_headings(doc, toc_titles=toc_titles)

    # Step 3: independently find P&L candidates by table structure
    structural = _structural_pl_candidates(doc)

    # Step 3b: text-scan fallback for titles that aren't font-sized headings
    text_titles = _scan_page_text_titles(doc)

    # Merge — physical page numbers from headings and structural scan; fall back
    # to the page-text scan per section only when the primary path found nothing.
    pl_pages    = sorted(set(headings["pl"] + structural)) or sorted(set(text_titles["pl"]))
    bs_pages    = sorted(set(headings["bs"])) or sorted(set(text_titles["bs"]))
    cf_pages    = sorted(set(headings["cf"])) or sorted(set(text_titles["cf"]))
    notes_pages = sorted(set(headings["notes"])) or sorted(set(text_titles["notes"]))

    # Step 4: LLM tiebreak when multiple distinct P&L pages remain
    if len(pl_pages) > 1:
        pl_pages = [_llm_tiebreak(doc, pl_pages)]

    if not pl_pages:
        log.warning("No P&L page candidates found for %s", doc.source_path)

    # Step 5: expand to continuation pages (multi-page statements), bounded by the
    # starts of other sections so it can't run to the end of a heading-less PDF.
    if pl_pages:
        boundaries = frozenset(bs_pages + cf_pages + notes_pages)
        pl_pages = _expand_to_continuation_pages(doc, pl_pages[0], boundary_pages=boundaries)

    # Notes range: first identified notes page to end of document
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
            tier="classification",
            system="You disambiguate which page contains the primary financial statement.",
            user="\n\n".join(excerpts),
            tool=tool,
            max_tokens=256,
        )
        chosen = int(result.get("page") or candidates[0])
        log.info("Locate tiebreak chose p.%d: %s", chosen, result.get("reason"))
        return chosen
    except Exception as e:
        log.warning("Locate tiebreak failed: %s — defaulting to first candidate %d", e, candidates[0])
        return candidates[0]
