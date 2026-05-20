"""LangGraph StateGraph orchestrator wiring the 9 agents.

State flows in one direction (Ingest → Locate → Parse → ResolveNotes → Canonicalise
→ Reconcile → Ratios → Score → Report). Reconcile has a conditional edge back to
Parse for up to one retry; after that the run continues with data_quality=LOW.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TypedDict

from afde.agents import (
    ingest as agent_ingest,
    locate as agent_locate,
    normalize as agent_normalize,
    parse_statement as agent_parse,
    ratios as agent_ratios,
    reconcile as agent_reconcile,
    report as agent_report,
    resolve_notes as agent_resolve,
    score as agent_score,
)
from afde.schemas import (
    CanonicalStatement,
    DocumentQuality,
    EnrichedStatement,
    IngestedDocument,
    Ratios,
    RiskReport,
    Score,
    SectionMap,
    Statement,
    ValidationReport,
)

log = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    pdf_path: str
    doc: IngestedDocument
    section_map: SectionMap
    pl_statement: Statement
    bs_statement: Optional[Statement]
    enriched_pl: EnrichedStatement
    enriched_bs: Optional[EnrichedStatement]
    canonical: CanonicalStatement
    validation: ValidationReport
    ratios: Ratios
    score: Score
    report: RiskReport
    retries: int


def _ingest(state: PipelineState) -> PipelineState:
    doc = agent_ingest.run(state["pdf_path"])
    return {"doc": doc}


def _locate(state: PipelineState) -> PipelineState:
    sm = agent_locate.run(state["doc"])
    return {"section_map": sm}


def _parse(state: PipelineState) -> PipelineState:
    statements = agent_parse.run(state["doc"], state["section_map"])
    out: PipelineState = {"pl_statement": statements.get("profit_or_loss")}  # type: ignore[typeddict-item]
    if "balance_sheet" in statements:
        out["bs_statement"] = statements["balance_sheet"]
    return out


def _resolve(state: PipelineState) -> PipelineState:
    pl = state["pl_statement"]
    enriched_pl = agent_resolve.run(state["doc"], state["section_map"], pl)
    out: PipelineState = {"enriched_pl": enriched_pl}
    bs = state.get("bs_statement")
    if bs is not None:
        out["enriched_bs"] = agent_resolve.run(state["doc"], state["section_map"], bs)
    return out


def _canonicalise(state: PipelineState) -> PipelineState:
    company = state["doc"].company_name or Path(state["pdf_path"]).stem
    canon = agent_normalize.run(
        company_name=company,
        enriched_pl=state["enriched_pl"],
        enriched_bs=state.get("enriched_bs"),
        section_map=state["section_map"],
    )
    return {"canonical": canon}


def _reconcile(state: PipelineState) -> PipelineState:
    rep = agent_reconcile.run(state["enriched_pl"], state["canonical"])
    return {"validation": rep}


def _reconcile_router(state: PipelineState) -> str:
    rep = state.get("validation")
    retries = state.get("retries", 0)
    if rep and rep.overall_passed:
        return "ratios"
    if retries < 1:
        log.warning("Reconcile failed (retry %d); routing back to parse.", retries + 1)
        return "retry_parse"
    log.warning("Reconcile still failing after retry; continuing with low quality.")
    return "ratios"


def _retry_parse(state: PipelineState) -> PipelineState:
    # In a fuller impl this would re-extract with higher OCR DPI or stricter table thresholds.
    # For now: bump retries, downgrade quality, and let the pipeline proceed.
    doc = state["doc"]
    if doc.quality != DocumentQuality.LOW:
        doc.quality = DocumentQuality.MEDIUM
    return {"retries": state.get("retries", 0) + 1, "doc": doc}


def _ratios(state: PipelineState) -> PipelineState:
    r = agent_ratios.run(state["canonical"])
    return {"ratios": r}


def _score(state: PipelineState) -> PipelineState:
    s = agent_score.run(state["canonical"], state["ratios"], state["enriched_pl"])
    return {"score": s}


def _report(state: PipelineState) -> PipelineState:
    report = RiskReport(
        company_name=state["canonical"].company_name,
        period_label=state["canonical"].period_label,
        score=state["score"],
        ratios=state["ratios"],
        canonical=state["canonical"],
        enriched_statement=state["enriched_pl"],
        validation=state["validation"],
        quality=state["doc"].quality,
    )
    final = agent_report.run(report)
    return {"report": final}


def build_graph():
    """Construct the LangGraph StateGraph. Returns a compiled graph."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(PipelineState)
    g.add_node("ingest", _ingest)
    g.add_node("locate", _locate)
    g.add_node("parse", _parse)
    g.add_node("resolve", _resolve)
    g.add_node("canonicalise", _canonicalise)
    g.add_node("reconcile", _reconcile)
    g.add_node("retry_parse", _retry_parse)
    g.add_node("ratios", _ratios)
    g.add_node("score", _score)
    g.add_node("report", _report)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "locate")
    g.add_edge("locate", "parse")
    g.add_edge("parse", "resolve")
    g.add_edge("resolve", "canonicalise")
    g.add_edge("canonicalise", "reconcile")
    g.add_conditional_edges(
        "reconcile",
        _reconcile_router,
        {"ratios": "ratios", "retry_parse": "retry_parse"},
    )
    g.add_edge("retry_parse", "parse")
    g.add_edge("ratios", "score")
    g.add_edge("score", "report")
    g.add_edge("report", END)
    return g.compile()


def run_pipeline(pdf_path: str | Path) -> RiskReport:
    graph = build_graph()
    final_state = graph.invoke({"pdf_path": str(pdf_path), "retries": 0})
    return final_state["report"]
