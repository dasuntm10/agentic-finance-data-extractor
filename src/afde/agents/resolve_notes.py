"""Agent 4 — Note Resolver.

Builds an index of numbered notes (1, 2, 3 with sub-letters a/b/c…) from the
notes section, then attaches them to the LineItems that referenced them.
"""
from __future__ import annotations

import logging
import re

from afde.schemas import (
    EnrichedStatement,
    IngestedDocument,
    Note,
    SectionMap,
    Statement,
    TableBlock,
)

log = logging.getLogger(__name__)

_NOTE_HEADING = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Z &/\-\(\)']{2,})\s*$", re.M)
_SUB_HEADING = re.compile(r"^\s*\(([a-z])\)\s+([A-Z].+)$", re.M)


def _build_index(doc: IngestedDocument, notes_pages: list[int]) -> dict[str, Note]:
    """Return {'3': Note, '3(a)': Note, …}."""
    idx: dict[str, Note] = {}
    current_note: int | None = None
    current_title: str | None = None
    buf_text: list[str] = []
    buf_page: int = 0

    def flush(sub: str | None = None) -> None:
        nonlocal buf_text
        if current_note is None:
            buf_text = []
            return
        key = f"{current_note}" + (f"({sub})" if sub else "")
        idx[key] = Note(
            note_number=current_note,
            sub=sub,
            title=current_title or "",
            text="\n".join(buf_text).strip(),
            tables=[],
            page=buf_page,
        )
        buf_text = []

    for page in doc.pages:
        if page.page not in notes_pages:
            continue
        text = page.text or ""
        for line in text.splitlines():
            m_note = _NOTE_HEADING.match(line)
            if m_note:
                if current_note is not None:
                    flush()
                current_note = int(m_note.group(1))
                current_title = m_note.group(2).strip()
                buf_page = page.page
                continue
            m_sub = _SUB_HEADING.match(line)
            if m_sub and current_note is not None:
                flush()
                buf_text.append(m_sub.group(2))
                # The sub-note key is created on the NEXT flush by passing the sub letter.
                # To keep it simple: emit immediately as a placeholder, then keep filling.
                sub_letter = m_sub.group(1).lower()
                key = f"{current_note}({sub_letter})"
                idx[key] = Note(
                    note_number=current_note,
                    sub=sub_letter,
                    title=m_sub.group(2).strip(),
                    text=m_sub.group(2),
                    tables=[],
                    page=page.page,
                )
                continue
            buf_text.append(line)
    if current_note is not None:
        flush()

    # Attach tables that fall on the same page as a note heading
    for tbl in doc.tables:
        if tbl.page not in notes_pages:
            continue
        # Find the closest preceding note key at or before this table's page
        candidates = [k for k, n in idx.items() if n.page <= tbl.page]
        if not candidates:
            continue
        nearest = max(candidates, key=lambda k: idx[k].page)
        idx[nearest].tables.append(tbl)

    return idx


def run(
    doc: IngestedDocument, section_map: SectionMap, statement: Statement
) -> EnrichedStatement:
    note_index = _build_index(doc, section_map.notes_pages)

    attached: dict[str, list[str]] = {}
    notes_used: dict[str, Note] = {}
    for li in statement.line_items:
        keys: list[str] = []
        for ref in li.note_refs:
            specific = ref.key()  # "3(a)"
            general = f"{ref.note_number}"  # "3"
            chosen = None
            if specific in note_index:
                chosen = specific
            elif general in note_index:
                chosen = general
            if chosen:
                keys.append(chosen)
                notes_used[chosen] = note_index[chosen]
        if keys:
            attached[li.label] = keys

    log.info(
        "Notes: indexed=%d attached_lines=%d unique_notes=%d",
        len(note_index),
        len(attached),
        len(notes_used),
    )
    return EnrichedStatement(
        statement=statement,
        notes=list(notes_used.values()),
        line_item_to_notes=attached,
    )
