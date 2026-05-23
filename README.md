# Agentic Finance Data Extractor (AFDE)

Implementation of the architecture in [`ARCHITECTURE.md`](./ARCHITECTURE.md): an agentic workflow that extracts the Profit & Loss / Consolidated Statement of Comprehensive Income (plus linked notes) from Australian annual reports, computes credit-risk ratios, and emits a per-company structured artefact and analyst-facing risk report.

## Requirements

- **Python 3.10 - 3.12** (LangGraph and pydantic v2 with PEP 604 union types).
- **[Poetry](https://python-poetry.org/docs/#installation)** as the package manager (`pipx install poetry` or `curl -sSL https://install.python-poetry.org | python3 -`).
- **[Ollama](https://ollama.com/download)** running locally to serve `qwen2.5:7b-instruct` (the LLM used for classification tiebreaks and the analyst narrative). vLLM is supported as an alternative for GPU deployments.
- *Optional:* PaddleOCR or Tesseract for the scanned-PDF branch (`B & E FOODS PTY LTD.pdf`).

## Setup

```bash
poetry install
ollama pull qwen2.5:7b-instruct        # local LLM
python scripts/fetch_models.py         # pulls BGE-small + Docling/TableFormer + PP-OCRv4 into ./models/
```

After this one-time setup the pipeline runs entirely offline — no API keys, no network calls.

Optional OCR fallback:

```bash
poetry install --extras ocr          # PaddleOCR (recommended for dense tables)
poetry install --extras tesseract    # Tesseract fallback
```

## Run

```bash
# Full pipeline on a single PDF
poetry run afde run "data/CITIGROUP.pdf"

# Every PDF in the data/ directory
poetry run afde run-all data/

# Diagnostics - verify env + config
poetry run afde info
```

Per-stage subcommands (useful when iterating on scoring thresholds without re-parsing):

```bash
poetry run afde extract "data/CITIGROUP.pdf"   # parse + canonicalise → extraction.json
```

## Smoke test (no install required)

A lightweight smoke test exercises the deterministic core (numeric parser, note-ref tokeniser, unit detection) against the real Citigroup PDF. It runs on Python 3.9+ and uses only stdlib + `pypdf`:

```bash
python scripts/smoke_test.py
```

Expected output ends with `26 passed, 0 failed`.

Full unit test suite (requires Python 3.10+ and `poetry install`):

```bash
poetry run pytest
```

## Outputs

For each input `<name>.pdf` the pipeline writes to `outputs/<name>/`:

- `extraction.json` - full structured extraction (statement + notes + canonical + ratios + score)
- `risk_report.md` - analyst-facing narrative
- `risk_report.pdf` - same, rendered via WeasyPrint
- `_cache/embeddings.jsonl` - content-hash embedding cache (re-runs are free for already-seen labels)

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md). 9 specialised agents on a LangGraph `StateGraph` with a SQLite checkpointer; Docling + PaddleOCR for parsing; a local Qwen2.5-7B-Instruct (via Ollama) for the three reasoning calls (locator tiebreak, mapper tiebreak, analyst narrative); BGE-small-en-v1.5 via `sentence-transformers` for canonical-label embedding. Fully offline at runtime.

## Layout

```
src/afde/
  agents/           # the 9 agents (ingest, locate, parse_statement, resolve_notes,
                    #               normalize, reconcile, ratios, score, report)
  llm/              # Local LLM client (Ollama / vLLM) + BGE-small embedder (cached)
  parsing/          # pdf_loader (PyMuPDF + Docling), ocr (PaddleOCR/Tesseract),
                    # numeric (accounting-aware Decimal + note-ref tokeniser)
  schemas.py        # Pydantic v2 contracts between agents
  config.py         # YAML config loading
  orchestrator.py   # LangGraph StateGraph wiring
  cli.py            # Typer entry points

config/
  canonical_labels.yaml   # canonical chart of accounts + synonyms
  scoring.yaml            # piecewise-linear band thresholds + composite weights

tests/unit/         # parser + scoring unit tests
scripts/smoke_test.py     # stdlib-only sanity check vs real Citigroup PDF
```

## Why Poetry

- PEP 517 build backend with a reproducible `poetry.lock` across platforms.
- First-class dependency groups (`dev`, optional `extras`) keep the runtime surface lean.
- `poetry run` resolves the project venv on the fly, so contributors don't need to remember activation.
- Matches the case-study brief's stated tooling preference.
