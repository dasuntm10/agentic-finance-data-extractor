"""Agent 9 — Risk Report Writer.

Emits a per-company structured artefact (JSON) and a short analyst-facing
narrative (Markdown + PDF). The narrative is the only place a generative LLM
writes free-form text — every numeric in the output is checked against the
source JSON; unsourced figures trigger a regeneration on the same local model,
and a second failure falls back to a deterministic template-rendered narrative.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from afde.config import OUTPUT_DIR
from afde.llm.client import get_llm
from afde.schemas import RiskReport

log = logging.getLogger(__name__)

NARRATIVE_SYSTEM = (
    "You are a credit analyst. Write a concise risk summary for the company described "
    "in the structured JSON the user provides. Cite specific ratios and YoY moves. "
    "Do NOT introduce any numbers that are not present in the JSON. Output Markdown "
    "with the following sections: ## Headline, ## Key ratios, ## Material risks, "
    "## Data quality. Keep it under 400 words."
)


def _numeric_tokens(s: str) -> set[str]:
    """Extract numeric-looking tokens from text (excluding obvious section/list ordinals)."""
    out = set()
    for m in re.finditer(r"-?\d+(?:\.\d+)?%?", s):
        tok = m.group(0)
        if tok in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"):
            continue  # likely list numbering
        out.add(tok)
    return out


def _fact_check(narrative: str, source_json: str) -> list[str]:
    src_tokens = _numeric_tokens(source_json)
    nar_tokens = _numeric_tokens(narrative)
    # Allow tokens that appear in source after stripping % and trailing zeros
    src_relaxed = set()
    for t in src_tokens:
        bare = t.rstrip("%")
        src_relaxed.add(bare)
        # also add rounded forms
        try:
            f = float(bare)
            src_relaxed.add(f"{f:.1f}")
            src_relaxed.add(f"{f:.2f}")
            src_relaxed.add(f"{int(round(f))}")
        except ValueError:
            pass
    bad = []
    for t in nar_tokens:
        bare = t.rstrip("%")
        if bare not in src_relaxed:
            bad.append(t)
    return bad


def _write_narrative(report: RiskReport) -> str:
    llm = get_llm()
    payload = report.model_dump(mode="json", exclude={"narrative_markdown"})
    payload_str = json.dumps(payload, indent=2, default=str)
    truncated = payload_str[:18000]  # keep input bounded

    # First attempt — local Qwen
    try:
        narrative = llm.text(
            tier="narrative",
            system=NARRATIVE_SYSTEM,
            user=f"Structured report data (JSON):\n```json\n{truncated}\n```",
            max_tokens=1200,
        )
    except Exception as e:
        log.warning("Local LLM narrative call failed: %s", e)
        return _template_narrative(report)

    if not narrative:
        return _template_narrative(report)

    unsourced = _fact_check(narrative, payload_str)
    if not unsourced:
        return narrative

    # Regenerate with the offending span quoted back
    log.warning("Narrative had unsourced figures: %s — regenerating", unsourced[:8])
    try:
        narrative = llm.text(
            tier="narrative",
            system=NARRATIVE_SYSTEM,
            user=(
                f"Structured report data (JSON):\n```json\n{truncated}\n```\n\n"
                f"On the previous attempt these tokens appeared in your output but are NOT in the source: "
                f"{unsourced}. Rewrite without introducing numbers absent from the JSON."
            ),
            max_tokens=1200,
        )
        unsourced = _fact_check(narrative, payload_str)
        if not unsourced and narrative:
            return narrative
    except Exception as e:
        log.warning("Narrative regeneration failed: %s", e)

    # Deterministic fallback — no LLM, no hallucination surface.
    log.warning("Falling back to deterministic template narrative after fact-check failure.")
    return _template_narrative(report)


def _template_narrative(report: RiskReport) -> str:
    """Deterministic narrative built directly from the structured report.

    Used as the fallback when the LLM is unreachable or repeatedly emits unsourced
    figures. No free-form generation, so no hallucination is possible.
    """
    s = report.score
    rats = report.ratios
    composite = f"{s.composite:.1f}" if s.composite is not None else "n/a"
    parts = [
        "## Headline",
        "",
        f"{report.company_name} ({report.period_label}) — composite score **{composite} / 100**, "
        f"band **{s.band or 'n/a'}**, profile **{s.profile.value}**, "
        f"data quality **{report.quality.value}**.",
        "",
        "## Key ratios",
        "",
    ]
    for ra in rats.ratios[:8]:
        if ra.value is None:
            continue
        delta = f" (YoY {ra.yoy_delta:+.3f})" if ra.yoy_delta is not None else ""
        parts.append(f"- **{ra.name}**: {ra.value:.3f}{delta}")

    parts += ["", "## Material risks", ""]
    if s.anomalies:
        for a in s.anomalies[:6]:
            parts.append(f"- [{a.severity}] {a.description}")
    else:
        parts.append("None flagged by the anomaly detectors.")

    parts += [
        "",
        "## Data quality",
        "",
        f"Document quality: **{report.quality.value}**. "
        f"Unmatched line items: {len(report.canonical.unmatched_labels)}. "
        "Narrative generated by deterministic template (LLM unavailable or fact-check failed).",
    ]
    return "\n".join(parts)


def _render_pdf(md_text: str, out_pdf: Path) -> None:
    try:
        import markdown  # type: ignore  # noqa: PLC0415
        from weasyprint import HTML  # noqa: PLC0415

        html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        full = (
            "<html><head><meta charset='utf-8'><style>"
            "body{font-family:'Segoe UI',Arial,sans-serif;max-width:780px;margin:24px auto;color:#222}"
            "h1,h2{color:#0a3d62}table{border-collapse:collapse;width:100%;margin:8px 0}"
            "th,td{border:1px solid #ccc;padding:4px 8px;font-size:11px}"
            "code{background:#f4f4f4;padding:1px 4px}"
            "</style></head><body>" + html_body + "</body></html>"
        )
        HTML(string=full).write_pdf(str(out_pdf))
    except Exception as e:
        log.warning("PDF render failed (%s); skipping PDF.", e)


def run(report: RiskReport) -> RiskReport:
    out_dir = OUTPUT_DIR / report.company_name
    out_dir.mkdir(parents=True, exist_ok=True)

    narrative = _write_narrative(report)
    report.narrative_markdown = narrative

    # Persist JSON and Markdown + PDF
    (out_dir / "extraction.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    md = _build_markdown(report)
    (out_dir / "risk_report.md").write_text(md, encoding="utf-8")
    _render_pdf(md, out_dir / "risk_report.pdf")

    log.info("Report written: %s", out_dir)
    return report


def _build_markdown(r: RiskReport) -> str:
    s = r.score
    rats = r.ratios
    lines = [
        f"# Risk report — {r.company_name}",
        "",
        f"**Period:** {r.period_label}    **Profile:** {s.profile.value}    "
        f"**Data quality:** {r.quality.value}",
        "",
        f"**Composite score:** {('%.1f' % s.composite) if s.composite is not None else 'n/a'} / 100  "
        f"  →  band **{s.band or 'n/a'}**",
        "",
        "## Headline",
        "",
        (r.narrative_markdown or "").strip(),
        "",
        "## Ratios",
        "",
        "| Ratio | Value | YoY Δ | Sub-score | Band |",
        "|---|---|---|---|---|",
    ]
    sub_by_name = {ss.ratio: ss for ss in s.sub_scores}
    for ra in rats.ratios:
        ss = sub_by_name.get(ra.name)
        v = f"{ra.value:.3f}" if ra.value is not None else "n/a"
        d = f"{ra.yoy_delta:+.3f}" if ra.yoy_delta is not None else "—"
        sv = f"{ss.sub_score:.0f}" if ss and ss.sub_score is not None else "—"
        sb = ss.band if ss and ss.band else "—"
        lines.append(f"| {ra.name} | {v} | {d} | {sv} | {sb} |")

    lines += ["", "## Anomalies & material risks", ""]
    if s.anomalies:
        for a in s.anomalies:
            line = f"- **[{a.severity}]** ({a.kind}) {a.description}"
            if a.evidence:
                line += f"  \n  > {a.evidence}"
            lines.append(line)
    else:
        lines.append("_None detected._")

    lines += [
        "",
        "## Validation",
        "",
        f"Overall: **{'PASS' if r.validation.overall_passed else 'FAIL'}**",
        "",
    ]
    for c in r.validation.checks:
        mark = "✅" if c.passed else "❌"
        lines.append(f"- {mark} `{c.name}` — {c.detail or ''}")

    lines += [
        "",
        "## Data quality",
        "",
        f"Document quality: **{r.quality.value}**. Unmatched labels: "
        f"{len(r.canonical.unmatched_labels)}.",
    ]
    if r.canonical.unmatched_labels:
        lines.append("")
        lines.append("Unmatched line items:")
        for lab in r.canonical.unmatched_labels[:20]:
            lines.append(f"- {lab}")

    return "\n".join(lines) + "\n"
