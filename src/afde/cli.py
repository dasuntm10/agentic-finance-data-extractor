"""Typer CLI for AFDE.

Subcommands:
  afde run <pdf>           — full pipeline (ingest → report)
  afde run-all <dir>       — pipeline over every PDF in a directory
  afde extract <pdf>       — stop after canonicalise, write extraction.json
  afde score <json>        — re-score an existing extraction.json
  afde report <json>       — re-render the report from an existing extraction.json
  afde info                — print env / model / config diagnostics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from afde.config import (
    DATA_DIR,
    OUTPUT_DIR,
    anthropic_api_key,
    google_api_key,
    is_offline,
)
from afde.orchestrator import run_pipeline
from afde.schemas import RiskReport

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@app.command()
def run(pdf: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False)) -> None:
    """Run the full pipeline on a single PDF."""
    console.print(f"[bold cyan]Running pipeline on:[/] {pdf}")
    report = run_pipeline(pdf)
    _print_summary(report)


@app.command("run-all")
def run_all(directory: Path = typer.Argument(DATA_DIR, exists=True, file_okay=False)) -> None:
    """Run the pipeline on every PDF in a directory."""
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {directory}[/]")
        raise typer.Exit(code=1)
    for p in pdfs:
        console.rule(p.name)
        try:
            report = run_pipeline(p)
            _print_summary(report)
        except Exception as e:
            console.print(f"[red]Failed on {p.name}: {e}[/]")
            logging.exception("Pipeline failure")


@app.command()
def extract(pdf: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False)) -> None:
    """Run only the extraction half of the pipeline (no scoring / no narrative)."""
    from afde.agents import (
        ingest as agent_ingest,
        locate as agent_locate,
        normalize as agent_normalize,
        parse_statement as agent_parse,
        resolve_notes as agent_resolve,
    )

    doc = agent_ingest.run(pdf)
    section_map = agent_locate.run(doc)
    statements = agent_parse.run(doc, section_map)
    pl = statements.get("profit_or_loss")
    if pl is None:
        console.print("[red]No P&L statement extracted.[/]")
        raise typer.Exit(1)
    enriched_pl = agent_resolve.run(doc, section_map, pl)
    enriched_bs = None
    if "balance_sheet" in statements:
        enriched_bs = agent_resolve.run(doc, section_map, statements["balance_sheet"])
    canon = agent_normalize.run(
        company_name=doc.company_name or pdf.stem,
        enriched_pl=enriched_pl,
        enriched_bs=enriched_bs,
        section_map=section_map,
    )
    out_dir = OUTPUT_DIR / (doc.company_name or pdf.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "section_map": section_map.model_dump(mode="json"),
        "enriched_pl": enriched_pl.model_dump(mode="json"),
        "enriched_bs": enriched_bs.model_dump(mode="json") if enriched_bs else None,
        "canonical": canon.model_dump(mode="json"),
    }
    (out_dir / "extraction.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]Wrote {out_dir / 'extraction.json'}[/]")


@app.command()
def info() -> None:
    """Print environment / model / config diagnostics."""
    t = Table(show_header=False)
    t.add_row("ANTHROPIC_API_KEY", "set" if anthropic_api_key() else "[red]missing[/]")
    t.add_row("GOOGLE_API_KEY", "set" if google_api_key() else "[red]missing[/]")
    t.add_row("AFDE_OFFLINE", "yes" if is_offline() else "no")
    t.add_row("Data dir", str(DATA_DIR))
    t.add_row("Output dir", str(OUTPUT_DIR))
    console.print(t)


def _print_summary(report: RiskReport) -> None:
    s = report.score
    t = Table(title=f"{report.company_name} — {report.period_label}")
    t.add_column("Field")
    t.add_column("Value", justify="right")
    t.add_row("Composite", f"{s.composite:.1f}" if s.composite is not None else "n/a")
    t.add_row("Band", s.band or "n/a")
    t.add_row("Profile", s.profile.value)
    t.add_row("Quality", report.quality.value)
    t.add_row("Anomalies", str(len(s.anomalies)))
    t.add_row("Validation", "PASS" if report.validation.overall_passed else "FAIL")
    console.print(t)
    out = OUTPUT_DIR / report.company_name
    console.print(f"[dim]Artefacts: {out}[/]")


if __name__ == "__main__":
    app()
