"""Agent 5 — Canonical Mapper / Normaliser.

Rules → embedding similarity (BGE-small) → local-LLM (Qwen2.5-7B) tiebreak.
Output is a CanonicalStatement keyed on the canonical chart of accounts
(config/canonical_labels.yaml).
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal
from pathlib import Path

from afde.config import OUTPUT_DIR, canonical_labels
from afde.llm.client import ToolChoice, get_llm
from afde.llm.embeddings import cosine, get_embedder
from afde.schemas import (
    CanonicalLine,
    CanonicalStatement,
    CompanyProfile,
    EnrichedStatement,
    SectionMap,
    Statement,
)

log = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _rule_match(label: str, profile: CompanyProfile) -> tuple[str, float] | None:
    """Return (canonical, 1.0) on synonym match."""
    cfg = canonical_labels()
    label_n = _norm(label)
    for entry in cfg["canonical_lines"]:
        if profile.value not in entry["profiles"]:
            continue
        for syn in entry["synonyms"]:
            if _norm(syn) == label_n:
                return entry["canonical"], 1.0
        # Substring match — less confident
        for syn in entry["synonyms"]:
            if _norm(syn) in label_n or label_n in _norm(syn):
                return entry["canonical"], 0.85
    return None


def _candidates_for_profile(profile: CompanyProfile) -> list[dict]:
    cfg = canonical_labels()
    return [e for e in cfg["canonical_lines"] if profile.value in e["profiles"]]


def _embed_match(
    label: str, profile: CompanyProfile, embedder
) -> list[tuple[str, float]]:
    """Return ranked [(canonical, similarity)] using embedding cosine."""
    candidates = _candidates_for_profile(profile)
    label_vec = embedder.embed(label)
    scored: list[tuple[str, float]] = []
    for entry in candidates:
        # Embed the canonical name + its top synonyms once each; take the best.
        best = 0.0
        for syn in [entry["canonical"]] + entry["synonyms"]:
            v = embedder.embed(syn)
            sim = cosine(label_vec, v)
            if sim > best:
                best = sim
        scored.append((entry["canonical"], best))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _llm_tiebreak(label: str, top_k: list[tuple[str, float]], profile: CompanyProfile) -> str | None:
    llm = get_llm()
    enum = [c for c, _ in top_k] + ["unmatched"]
    tool = ToolChoice(
        name="map_label",
        description=(
            "Choose the canonical chart-of-accounts key that best matches the company-specific "
            "label below. Return 'unmatched' if none apply."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "canonical": {"type": "string", "enum": enum},
                "reason": {"type": "string"},
            },
            "required": ["canonical"],
        },
    )
    user = (
        f"Profile: {profile.value}\n"
        f"Original label: {label!r}\n"
        f"Top embedding candidates (canonical, cosine):\n"
        + "\n".join(f"  - {c}: {s:.3f}" for c, s in top_k)
    )
    try:
        result = llm.structured(
            tier="classification",
            system="You map noisy financial-statement labels to a fixed canonical chart of accounts.",
            user=user,
            tool=tool,
            max_tokens=256,
        )
        choice = result.get("canonical")
        if choice and choice != "unmatched":
            return str(choice)
    except Exception as e:
        log.warning("Mapper LLM tiebreak failed for %r: %s", label, e)
    return None


def _scale_values(values: dict[str, Decimal], scale: int) -> dict[str, Decimal]:
    return {k: v * Decimal(scale) for k, v in values.items()}


def run(
    company_name: str,
    enriched_pl: EnrichedStatement,
    enriched_bs: EnrichedStatement | None,
    section_map: SectionMap,
) -> CanonicalStatement:
    profile = section_map.company_profile
    stmt: Statement = enriched_pl.statement

    cache_dir = OUTPUT_DIR / company_name / "_cache"
    embedder = get_embedder(cache_dir=cache_dir)

    lines: list[CanonicalLine] = []
    unmatched: list[str] = []

    statements_to_process = [enriched_pl.statement]
    if enriched_bs is not None:
        statements_to_process.append(enriched_bs.statement)

    for s in statements_to_process:
        for li in s.line_items:
            scaled = _scale_values(li.values, s.units_scale)
            # 1. Rule match
            rule = _rule_match(li.label, profile)
            if rule:
                canonical, score = rule
                lines.append(
                    CanonicalLine(
                        canonical=canonical,
                        original_label=li.label,
                        values=scaled,
                        note_refs=[r.key() for r in li.note_refs],
                        match_method="rule",
                        match_score=score,
                    )
                )
                continue
            # 2. Embedding match
            try:
                ranked = _embed_match(li.label, profile, embedder)
            except Exception as e:
                log.debug("Embedding failed for %r: %s", li.label, e)
                ranked = []
            top = ranked[:5]
            if top and top[0][1] >= 0.78:
                lines.append(
                    CanonicalLine(
                        canonical=top[0][0],
                        original_label=li.label,
                        values=scaled,
                        note_refs=[r.key() for r in li.note_refs],
                        match_method="embedding",
                        match_score=top[0][1],
                    )
                )
                continue
            # 3. LLM tiebreak (only if there's at least one half-credible candidate)
            llm_pick = None
            if top and top[0][1] >= 0.55:
                llm_pick = _llm_tiebreak(li.label, top, profile)
            if llm_pick:
                lines.append(
                    CanonicalLine(
                        canonical=llm_pick,
                        original_label=li.label,
                        values=scaled,
                        note_refs=[r.key() for r in li.note_refs],
                        match_method="llm",
                        match_score=top[0][1] if top else None,
                    )
                )
                continue
            unmatched.append(li.label)

    log.info(
        "Canonicalise: matched=%d unmatched=%d (profile=%s)",
        len(lines),
        len(unmatched),
        profile,
    )

    return CanonicalStatement(
        company_name=company_name,
        profile=profile,
        period_end=stmt.period_end,
        period_label=stmt.period_label,
        comparative_period_label=stmt.comparative_period_label,
        currency=stmt.currency,
        base_unit=1,
        lines=lines,
        unmatched_labels=unmatched,
    )
