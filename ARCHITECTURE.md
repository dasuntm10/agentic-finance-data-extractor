# Agentic Financial Data Extractor - Architecture & Implementation Plan

**Project:** CreditSource - Agentic data extraction & credit scoring from Australian annual reports
**Author:** Dasun
**Document scope:** Optimal end-to-end design for Tasks 1, 2 (optional impl), and 3 (optional risk model) of the CreditSource case study.
**Constraints honoured:** Python; reproducible via **Poetry** (`pyproject.toml` + `poetry.lock`); **fully offline** — no third-party hosted services at runtime. LLM reasoning runs on a **local Qwen2.5-7B-Instruct** served by Ollama (or vLLM), and embeddings come from **BGE-small-en-v1.5** loaded via `sentence-transformers`. All other tooling (PDF parsing, OCR, orchestration, validation, scoring math) is also open-source and local.

---

## 1. Problem Restatement & Observations from the Sample Data

The four documents in `data/` are ASIC Form 388 lodgements wrapping audited annual reports of Australian public/proprietary companies. Spot-checking the corpus reveals the design-driving variability that the architecture must absorb:

| File | Pages | Text quality | Statement label | Units |
|---|---|---|---|---|
| `AUSNET PTY LTD.pdf` | 82 | Mostly extractable, **several pages encoded with a custom CMap that decodes to `/0/1/2/…`** (font-encoded text, not scanned) | TBD per doc | likely `$M` |
| `B & E FOODS PTY LTD.pdf` | 30 | **Scanned** - pages 4-28 contain ~51 chars (page header only); requires OCR | TBD | TBD |
| `CITIGROUP.pdf` | 41 | Clean text, table layout preserved | `Consolidated Statement of Comprehensive Income` | mixes `$Million` and `$000` across tables |
| `YHI PTY LTD.pdf` | 39 | Clean text | TBD | TBD |

Concrete extraction signals already observed in Citigroup p.8 (the P&L):

```
                                                  Note   2024     2023
                                                         $Million $Million
Advisory fees                                     3(a)   108.8    101.6
Brokerage and other commissions                   3(b)    66.1     52.3
Net trading income                                3(c)   (4.3)     29.9
Interest income                                   3(d)  529.6    473.2
Interest expense                                  3(d) (580.8)  (521.1)
Management fee income                              26    23.7     13.9
Income tax benefit                                  4    12.4      8.3
```

This drives several non-negotiable requirements:

1. **Note references are first-class data.** They are alphanumeric (`3(a)`, `3(d)`), can be plain numbers (`26`), and the brief explicitly states a single line item may reference **multiple notes separated by commas** (e.g. `3, 26`). The extractor must capture them as a list and resolve each to the actual note text/table.
2. **Statement title is not fixed.** "Profit and Loss", "Consolidated Statement of Comprehensive Income", "Statement of Profit or Loss and Other Comprehensive Income" are all valid. Section discovery cannot rely on a literal title match.
3. **Currency & rounding are per-statement, not per-document.** Citigroup's P&L is `$Million` but its KMP compensation note is `$000`. Both must be captured against the table that owns them.
4. **PDF substrate varies.** A single pipeline must handle text-native PDFs, custom-font-encoded PDFs, and fully scanned PDFs without operator intervention.
5. **Negative values are parenthesised.** `(580.8)` means `-580.8`. Numeric parsing must be aware of accounting conventions.
6. **Year columns vary in count and label.** Two-year comparatives are typical; some statements show a third "Note" column or a `Restated` column.

The brief explicitly calls out an agentic workflow (LangGraph / LlamaIndex Workflows or similar) and a strict offline constraint. A pure-prompt LLM extractor is **not** the right answer for this corpus - the documents are layout-heavy, the reasoning is verifiable (numbers must reconcile), and the entire stack must run locally. The optimal design splits responsibilities so that each agent's job is small enough to be solved with **specialised tooling + a tightly scoped local-LLM call**, with deterministic validators between agents.

---

## 2. Design Principles

1. **Deterministic where possible, LLM where necessary.** Numbers from tables are extracted by table-structure models, not by free-form generation. LLMs are used only for (a) classification/ambiguity resolution and (b) writing the final analyst-facing prose in the risk report.
2. **Structured outputs end-to-end.** Every agent emits Pydantic-validated JSON. Inter-agent messages are typed; nothing is passed as free text between stages.
3. **Layout-aware ingestion.** Page-level layout (columns, tables, headings) is preserved through to extraction. We do not flatten the PDF to a single text stream.
4. **Verifiable extraction.** Every extracted value carries provenance: `{page, bbox, source_table_id}` so a human (or a self-check agent) can re-open the page and audit it.
5. **Reconciliation as a first-class check.** Subtotals and totals must equal the sum of their children within a tolerance. A failed reconciliation triggers a **re-extract** loop rather than silently propagating bad numbers downstream.
6. **Local, reproducible, offline.** All weights — Docling layout/TableFormer, PaddleOCR PP-OCRv4, Qwen2.5-7B-Instruct, BGE-small-en-v1.5 — are pre-fetched under `./models/`. After a one-time setup, `poetry run afde run …` executes with no network access at all.

---

## 3. High-Level Architecture

```
                ┌────────────────────────────────────────────────────────────┐
                │                 LangGraph Orchestrator                     │
                │       (StateGraph; typed shared state; checkpointer)       │
                └────────────────────────────────────────────────────────────┘
                                          │
   ┌──────────────────┬────────────────────┼────────────────────┬─────────────────────┐
   ▼                  ▼                    ▼                    ▼                     ▼
[1 Ingestion ]   [2 Section      ]   [3 Statement     ]   [4 Note          ]   [5 Normaliser  ]
[  Triage    ]   [  Locator      ]   [  Parser        ]   [  Resolver      ]   [  + Canonical ]
                                                                                  [  Mapper    ]
                                          │
                                          ▼
                                  [6 Reconciliation /
                                    Self-Check Loop]  ──── reject / retry ───┐
                                          │                                  │
                                          ▼                                  │
                                  [7 Ratio Engine] ◄──────────────────────── │
                                          │                                  │
                                          ▼                                  │
                                  [8 Scoring Agent]                          │
                                          │                                  │
                                          ▼                                  │
                                  [9 Risk Report Writer]  → final JSON + PDF │
                                          │                                  │
                                          └─────────► outputs/<company>/     │
                                                                             │
                                          (re-extract on validation fail)────┘
```

The orchestrator is a **LangGraph `StateGraph`** with a checkpointer (SQLite) so a long run can be resumed and individual nodes can be replayed. The shared state object carries the working artefacts (`raw_pages`, `layout_blocks`, `statement`, `notes`, `canonical`, `ratios`, `score`, `report`) so each agent reads what it needs and writes only its slice.

---

## 4. Agent Specifications

Each agent is defined by **(role, inputs, tools, outputs, failure mode)**. All outputs are Pydantic models in `src/schemas.py`.

### Agent 1 - Ingestion & Triage Agent

- **Role:** Classify the PDF substrate per page and produce a unified, structured page-level representation.
- **Tools:** `pypdf` (text-extraction probe), `pdfplumber` / `PyMuPDF` (vector text & layout), `pdf2image` + `tesseract` or `paddleocr` (OCR), `docling` (layout-aware parsing for high-quality pages).
- **Logic:**
  1. For each page, attempt text extraction. If `chars/page < 200` **or** the text is dominated by `/<digits>/` glyph tokens (custom-font CMap problem visible in AUSNET pp.4-5), mark the page `needs_ocr`.
  2. For `needs_ocr` pages, rasterise at 300 DPI and run PaddleOCR (PP-OCRv4 English). PaddleOCR is preferred over Tesseract here because the corpus contains many tables and PP-StructureV2 / PP-OCR handles dense numeric layouts better than vanilla Tesseract.
  3. Pass all pages through **Docling** (IBM open-source, Apache-2.0) which returns a `DoclingDocument` with `Table`, `Section`, `Text` blocks plus bounding boxes. Docling's TableFormer model handles complex financial tables and outputs a logical row/column structure - this is the single most important upstream choice for table-heavy reports.
- **Output:** `IngestedDocument { pages: List[PageBlock], tables: List[TableBlock], headings: List[Heading] }` with provenance on every block.
- **Failure mode:** If OCR confidence on a critical page is below threshold, the page is flagged; downstream agents see the flag and the orchestrator routes to a degraded-mode path that emits the company report with a `data_quality: low` marker rather than silently producing wrong numbers.

### Agent 2 - Section Locator Agent

- **Role:** Find the page range of the Statement of Profit or Loss / Consolidated Statement of Comprehensive Income, and the page range of the Notes block.
- **Tools:** Table-of-contents heuristics + the **local Qwen2.5-7B-Instruct** LLM for ambiguity resolution.
- **Logic:**
  1. First, parse the ToC if present (Citigroup p.5 has one). Map `Consolidated statement of comprehensive income → 5` etc.
  2. If no ToC, scan headings for a regex set: `(consolidated\s+)?statement of (profit (and|or) loss|comprehensive income)`, `profit (and|or) loss`, `income statement`. Also detect by **structural signature**: a page containing a table with a `Note` column header and at least one numeric `$Million|$'000|$000` row.
  3. The local LLM is only invoked when ≥2 candidate sections exist and a deterministic tie-break is impossible - e.g. when both "Statement of Profit or Loss" and "Statement of Comprehensive Income" are present as separate pages (some entities split them). The call uses **JSON-schema-constrained decoding** (Ollama `format: "json"` with a schema whose `enum` lists the candidate page-range IDs, or `outlines` / `lm-format-enforcer` under vLLM) so the response is guaranteed structurally valid. A 7B instruct model is more than enough for constrained classification with a tiny answer space; a larger model adds no headroom on enum selection.
- **Output:** `SectionMap { pl_pages: [int], notes_pages: [int], pl_title: str }`.

### Agent 3 - Statement Parser Agent

- **Role:** From the located pages, extract structured line items.
- **Tools:** Docling/TableFormer for the table; a numeric-parser utility; `dateparser` for FY detection.
- **Logic:**
  1. Pick the largest table on `pl_pages` whose header contains `Note` and at least one column whose header looks like a year (`2024`, `2023`, `30 June 2024`).
  2. Detect **units & rounding** from the row immediately under the column header (`$Million`, `$'000`, `$AUD '000`). Store at the statement level, not the line-item level.
  3. Detect **financial year** from the closest enclosing heading (`FOR THE YEAR ENDED 31 DECEMBER 2024`). Store both `period_end` and `period_label`.
  4. For each data row: emit a `LineItem { label, note_refs: [str], values: {year: Decimal} }`.
     - `note_refs` parsing handles `3(a)`, `3, 26`, `3(d), 26`, plain `4`, and the empty cell.
     - Negative amounts in `( )` are converted to `Decimal('-…')`.
     - Subtotal / total rows are flagged via heuristic (bold, indentation, label contains `Total|Net (loss|profit)|Profit before tax`).
  5. The LLM is NOT used to read numbers. It is only consulted to **label-classify** rows (e.g. is "Brokerage and other clearing, settlement and exchange fees" an opex item?) - and only when the canonical mapper (Agent 5) cannot match it deterministically.
- **Output:** `Statement { kind, title, currency, units, period_end, period_label, line_items: List[LineItem], provenance }`.

### Agent 4 - Note Resolver Agent

- **Role:** For each `note_ref` referenced by the statement, fetch the actual note content (text + sub-tables) from the notes section, and attach it to the line item.
- **Tools:** Docling-parsed notes blocks; a note-index built once per document; FAISS only as a fallback for fuzzy lookup (not the primary mechanism - the note numbering is deterministic).
- **Logic:**
  1. Build a `NoteIndex` mapping `note_number → {title, page, sub_sections: {'a': …, 'b': …}, tables: [...]}` from the notes pages. Headings like `3.  REVENUE RECOGNITION`, then `(a) Advisory fees`, give the structure.
  2. For each `LineItem.note_refs`, resolve each token: `3(a) → notes[3].sub_sections['a']`; `26 → notes[26]` (whole note); a bare `3 → notes[3]` (whole note).
  3. Attach the resolved note(s) to the line item with provenance. Notes that themselves contain tables (e.g. Note 3(d) which breaks down interest income/expense) have those tables extracted as `Statement`-like sub-objects so downstream ratio calculation can use the disaggregated numbers if needed.
- **Output:** `EnrichedStatement` (Statement + `notes: List[Note]` attached at line-item level).

### Agent 5 - Canonical Mapper / Normaliser Agent

- **Role:** Map the company-specific labels to a canonical chart of accounts so downstream ratios work uniformly across all four companies despite their different statement layouts.
- **Tools:** Rules + embedding similarity (**BGE-small-en-v1.5** via `sentence-transformers`, 384-dim, fully local) + the **local Qwen2.5-7B-Instruct** LLM only as last resort.
- **Canonical schema (extract):**
  ```
  revenue.total
  revenue.interest_income            (banks/financials)
  expense.interest_expense
  expense.employee
  expense.cogs
  expense.operating_other
  ebit                                (derived if not stated)
  ebitda                              (derived if D&A available in notes)
  finance_costs.net                   (derived: interest_expense - interest_income for non-financials)
  profit_before_tax
  income_tax_expense
  net_profit
  other_comprehensive_income
  total_comprehensive_income
  ```
  Plus the balance sheet items needed for liquidity & leverage (current assets, current liabilities, total assets, total equity, total debt) which Agent 3 also extracts from the Statement of Financial Position when present - the brief allows this since notes attached to the P&L commonly cross-reference balance-sheet items.
- **Logic:** rules-based label match first; embedding similarity (BGE-small, cosine) for unmatched; a local-LLM tiebreak call when similarity < 0.7 OR there are competing canonical candidates above 0.7. The call uses JSON-schema-constrained decoding whose `enum` is the canonical chart of accounts - making invalid outputs structurally impossible. A 7B instruct model is the right size because the answer space is finite and the input is short (a label string plus its top-5 candidates with similarities); a larger model does not change the achievable accuracy on enum classification with strong embedding priors.

  Embedding outputs are cached in `outputs/<company>/_cache/embeddings.jsonl` keyed by the SHA-256 of the input string. Although BGE-small inference is essentially free (a few ms per label on CPU), the cache still makes re-runs deterministic and lets re-scoring skip the embedder entirely.
- **Output:** `CanonicalStatement` keyed on canonical labels, with original labels retained for audit.

### Agent 6 - Reconciliation / Self-Check Agent

- **Role:** Verify the extraction is arithmetically and structurally consistent before any ratio is computed.
- **Checks (deterministic, no LLM):**
  1. **Sum check** - totals = sum of their immediate children, within tolerance of `±0.5 × 10^(units)` (i.e. ±$500k for `$Million` reports - handles rounding).
  2. **Sign check** - `profit_before_tax + income_tax_expense ≈ net_profit` (where tax is signed correctly).
  3. **OCI roll-up** - `total_comprehensive_income = net_profit + OCI`.
  4. **Note linkage** - every `note_ref` resolved to an existing note; no dangling pointers.
  5. **Cross-year completeness** - every line item has at least the latest year value.
- **Failure mode:** Returns a `ValidationReport` listing the failed checks with provenance. The orchestrator routes back to **Agent 3** with the failing region pinned, optionally re-rasterising at higher DPI / re-running TableFormer with a stricter threshold. After 2 retries the run is flagged `data_quality: low` and continues - never silently produces a bad number.

### Agent 7 - Ratio Engine

- **Role:** Compute the financial ratios from the canonical statement. **Fully deterministic.** No LLM.
- **Ratios computed:**
  - **Profitability:** net profit margin = `net_profit / revenue.total`; gross margin (if COGS available); return on equity = `net_profit / avg_equity`; EBIT margin.
  - **Liquidity:** current ratio = `current_assets / current_liabilities`; quick ratio = `(current_assets - inventory) / current_liabilities`.
  - **Leverage:** debt-to-equity = `total_debt / total_equity`; debt-to-assets.
  - **Coverage:** interest coverage = `EBIT / interest_expense`; (for financials, this becomes `net_interest_margin`).
  - **Trend:** YoY delta of each of the above (using the comparative year already in the statement).
- **Bank-vs-corporate branch:** Banks (Citigroup) don't have a meaningful "current ratio"; the engine has two profile templates and the Section Locator's classification (financial vs non-financial corporate, derived from ANZSIC heuristics on the directors' report) selects the template. Ratios that don't apply emit `null` with a reason, not `0`.

### Agent 8 - Scoring Agent

- **Role:** Convert ratios into a composite credit risk score 0-100 (higher = healthier) with documented sub-scores. **Hybrid rules + statistical** - see §5.

### Agent 9 - Risk Report Writer

- **Role:** Emit the per-company structured artefact and a short analyst-facing narrative.
- **Tools:** the **local Qwen2.5-7B-Instruct** for the narrative; WeasyPrint for the PDF. A larger local model (Qwen2.5-14B-Instruct or Qwen2.5-32B-Instruct) is wired as an optional `narrative_model` override in `config/scoring.yaml` for engagements with a GPU budget; in routine use the 7B prose is fluent enough for an analyst-facing risk summary and runs comfortably on CPU.
- **Outputs:**
  - `outputs/<company>/extraction.json` - full structured extraction (statement + notes + canonical + ratios + score).
  - `outputs/<company>/risk_report.md` (and PDF via `weasyprint`) - narrative containing: headline score, key ratios with YoY delta, material risks/anomalies the validator flagged, data-quality notes.
- The narrative is the **only** place the LLM writes free-form text. It is constrained to facts from the structured object - prompted with the JSON and instructed not to introduce numbers not present in the input. A post-write check greps the narrative for numerics and verifies each appears in the source object; any unsourced figure triggers a regeneration with the offending span quoted back to the model. On a second failure the run falls back to a **deterministic template-rendered narrative** built directly from the structured JSON - no free-form text, no risk of hallucination. The structured artefact is always emitted regardless.

---

## 5. Scoring Model Design (Task 1 & Task 3)

### 5.1 Approach choice - hybrid rules + statistical, not ML-from-scratch

With **four companies** and no labelled default outcomes, a supervised ML credit model cannot be trained meaningfully. A pure rules-based scorecard, on the other hand, is opaque on edge cases (e.g. banks with negative current ratio relevance). The optimal design is:

- **Rules-based per-ratio sub-scores** with thresholds calibrated to public credit-rating bands (S&P / Moody's industrial thresholds, which are open and citable). This gives an auditable, defensible score for each ratio.
- **Statistical aggregation** of sub-scores into a composite using either equal weights or weights derived from the proportion of variance each sub-score explains across the sample (PCA on the four-company ratio matrix). The latter is documented but defaults to equal weights given N=4 - the PCA result is reported as a sensitivity check, not the canonical weighting.
- **Industry-aware templates** - financials and non-financials use different ratio sets and different thresholds (banks live with higher leverage but tighter coverage).

### 5.2 Scoring formula

For each ratio `r` with profile thresholds `(strong, adequate, weak)`:

```
sub_score(r) = piecewise_linear_map(value(r), thresholds(r)) ∈ [0, 100]
```

Example - interest coverage (corporate template):

| Coverage | Sub-score | S&P-equivalent band |
|---|---|---|
| ≥ 12× | 100 | AA+ and above |
| 6×   | 75  | A |
| 3×   | 50  | BBB |
| 1.5× | 25  | BB |
| ≤ 1× | 0   | distressed |

Linear interpolation between bands; values outside the range clamp to 0 or 100.

Composite score:

```
composite = Σ w_i · sub_score_i
```

with default weights `{profitability: 0.30, leverage: 0.25, coverage: 0.25, liquidity: 0.20}` for corporates, `{profitability: 0.30, capital: 0.30, asset_quality: 0.25, liquidity: 0.15}` for banks. Weights are configurable via `config/scoring.yaml`.

### 5.3 Anomaly / material-risk detection (Task 3)

Independent of the composite score, the Scoring Agent runs three anomaly detectors and surfaces them in the risk report:

1. **YoY deterioration flags** - any sub-score that fell ≥25 points YoY, or any ratio that crossed a band boundary in the wrong direction.
2. **Note-disclosure flags** - keyword/embedding search over the resolved notes for `going concern`, `material uncertainty`, `contingent liabilities`, `subsequent events`, `restatement`, `qualified opinion`. Hits get attached to the report with the surrounding sentence.
3. **Off-balance-sheet exposure** - presence of meaningful values in `Contingent liabilities` (Note 29 in Citigroup), guarantees, or operating-lease commitments outside lease liabilities. Reported as a flag with the dollar amount, not folded into the score.

### 5.4 Validation of scores

With N=4, hold-out validation is impossible. The validation we **can** do, and which we document in the report:

- **Reproducibility check** - same inputs deterministically produce same scores (no randomness in extraction or scoring).
- **Sensitivity check** - re-score with ±10% perturbation on each ratio; the composite should move smoothly. Discontinuities expose threshold cliffs.
- **External anchor check** - for Citigroup (which has a parent with a public credit rating), the bank-template score is sanity-checked against the parent's published rating band. A scorecard that puts CGM Australia in `D` while the parent is `A-` is a red flag for either the thresholds or the extraction.
- **Manual eyeball on the smallest report** - B&E FOODS is small enough (30 pages) to hand-check; this is the regression test for the OCR branch.

---

## 6. Technology Stack (Pinned & Justified)

| Concern | Choice | Why this one |
|---|---|---|
| Orchestration | **LangGraph** (StateGraph + SqliteSaver checkpointer) | Typed state, branches, conditional edges, replay - exactly the shape of this workflow. LlamaIndex Workflows is also viable; LangGraph wins on debuggability. |
| Document parsing | **Docling** (IBM, Apache-2.0) with TableFormer | Best open-source layout + table model for financial PDFs. Outputs structured `DoclingDocument`, not flat text. |
| OCR (fallback) | **PaddleOCR** (PP-OCRv4) | Stronger on dense tables than Tesseract; fully offline. |
| Layout cross-check | **PyMuPDF (fitz)** | Vector text + bbox, used to verify Docling table boundaries. |
| LLM (local) | **Qwen2.5-7B-Instruct** served by **Ollama** (default; one-line install on macOS/Linux/Windows) or **vLLM** (when a GPU is available for higher throughput). Used for (a) section-locator tie-breaks, (b) canonical-mapper tiebreaks, (c) the analyst narrative. | Strong instruction-following at 7B; Apache-2.0 weights; runs on CPU at acceptable latency for a batch-style pipeline. Fully offline at runtime - no API keys, no egress. |
| Structured outputs | **JSON-schema-constrained decoding** - Ollama's `format` parameter under the hood, or `outlines` / `lm-format-enforcer` when running on vLLM. Schemas are derived from the same Pydantic models the agents pass around. | Invalid outputs are structurally impossible because the decoder is forced to emit a JSON value that satisfies the schema. Equivalent guarantee to a hosted "forced tool call", entirely on-device. |
| Embeddings (local) | **BGE-small-en-v1.5** via `sentence-transformers` (384-dim, ~33M params, MIT) | Used for label-to-canonical mapping and note keyword search. Vectors are content-hash cached on disk so re-runs are deterministic; CPU inference is a few ms per label. |
| Vector index (fallback) | **FAISS** (local) | Only for the small per-document note index - Postgres+pgvector is overkill. |
| Validation | **Pydantic v2** | Inter-agent contracts. |
| Numerics | **`decimal.Decimal`** end-to-end; **Pandas** only at the reporting layer | Float arithmetic on currency is a footgun. |
| PDF output | **WeasyPrint** | HTML/CSS → PDF, fully offline. |
| Packaging | **Poetry** | PEP 517 build backend; dependency groups (`dev`, optional `extras`); cross-platform `poetry.lock` for reproducible installs. Matches the case-study brief's stated tooling preference. |
| Testing | **pytest** with golden JSON snapshots per document | Regression-protects extraction; one snapshot per company. |
| CLI | **Typer** | `extract run`, `extract score`, `extract report` subcommands. |

### Local model cache

- **No API keys.** The runtime makes no network calls.
- All weights - Docling layout/TableFormer, PaddleOCR PP-OCRv4, Qwen2.5-7B-Instruct, BGE-small-en-v1.5 - are pre-fetched by `scripts/fetch_models.py` into `./models/`. Ollama caches the Qwen GGUF under `~/.ollama/models/` on first pull; `fetch_models.py` issues the `ollama pull qwen2.5:7b-instruct` command as part of one-time setup.
- After setup, the entire pipeline runs air-gapped. `poetry install` + `python scripts/fetch_models.py` are the only steps that touch the network, and both happen once on a connected machine.

---

## 7. Repository Layout

```
agentic-finance-data-extractor/
├── pyproject.toml               # Poetry; pinned deps
├── poetry.lock                  # cross-platform reproducible lockfile
├── README.md                    # how to run
├── ARCHITECTURE.md              # this file
├── config/
│   ├── scoring.yaml             # weights & thresholds (editable)
│   └── canonical_labels.yaml    # canonical chart of accounts + synonyms
├── data/                        # input PDFs (already present)
├── models/                      # pre-fetched weights (gitignored)
├── outputs/                     # per-company artefacts
│   └── <company>/
│       ├── extraction.json
│       ├── risk_report.md
│       └── risk_report.pdf
├── scripts/
│   └── fetch_models.py
├── src/
│   ├── schemas.py               # Pydantic models (Statement, LineItem, …)
│   ├── orchestrator.py          # LangGraph StateGraph definition
│   ├── agents/
│   │   ├── ingest.py
│   │   ├── locate.py
│   │   ├── parse_statement.py
│   │   ├── resolve_notes.py
│   │   ├── normalize.py
│   │   ├── reconcile.py
│   │   ├── ratios.py
│   │   ├── score.py
│   │   └── report.py
│   ├── llm/
│   │   ├── client.py            # Local LLM wrapper (Ollama / vLLM) with JSON-schema decoding
│   │   ├── embeddings.py        # sentence-transformers BGE-small wrapper (cached)
│   │   └── prompts/             # versioned prompts
│   ├── parsing/
│   │   ├── docling_loader.py
│   │   ├── ocr_paddle.py
│   │   ├── numeric.py           # accounting-aware Decimal parser
│   │   └── note_index.py
│   └── cli.py                   # Typer entry point
└── tests/
    ├── unit/
    └── golden/                  # snapshot JSON per company
```

---

## 8. Walk-Through on the Sample Corpus

Concrete predictions of what each agent will do on the four files - this is the testbed for the regression suite.

| Stage | AUSNET | B&E FOODS | CITIGROUP | YHI |
|---|---|---|---|---|
| Ingest | Mixed: pp.4-5 routed to OCR (custom-font garble); rest via Docling | All pp.4-28 routed to OCR (scanned) | All Docling, clean | All Docling, clean |
| Locate | ToC → P&L page; fallback regex if ToC missing | OCR'd ToC + regex `statement of profit` | ToC p.5 → P&L p.8, notes pp.12-39 | ToC or regex |
| Parse | Standard corporate template | Standard small-co template, OCR confidence flagged on each value | Bank/dealer template; line items with notes `3(a)`, `3(d)`, `4`, `26` | Standard corporate template |
| Resolve notes | Index built from notes section | Same, OCR'd | `notes[3].sub_sections['a','b','c','d','e']`, `notes[4]`, `notes[26]` all resolved | Same |
| Normalize | Maps "Revenue from continuing operations" → `revenue.total` | Small entity may lack disaggregation; mapper falls back to top-line revenue | Maps "Advisory fees" + "Brokerage…" + "Net trading income" + "Net interest income (3(d))" → `revenue.total` for the bank template | Maps standard labels |
| Reconcile | Subtotals match; pass | OCR may produce a digit-confusion (`8` vs `0`); reconciliation catches the row that doesn't sum and triggers re-OCR at 400 DPI | Pass - totals match per p.8 sample | Pass |
| Ratios | Corporate template | Corporate template; some ratios `null` if balance sheet not fully extractable | Bank template - net interest margin, cost-to-income, leverage | Corporate template |
| Score | Composite + sub-scores | Composite with `data_quality: medium` flag | Composite with bank weights; sanity-check vs parent rating band | Composite |
| Report | MD + PDF | MD + PDF + low-quality banner | MD + PDF | MD + PDF |

---

## 9. Assumptions & Perceived Challenges

**Assumptions made:**

- All four documents are real ASIC Form 388 lodgements of audited annual reports; the data inside is correct to the source (we are not fact-checking against the company, only extracting what is on the page).
- Comparative-year columns are always present (AASB-required for going concerns). If a company shows only one year (e.g. first reporting period), YoY ratios are reported as `null` with reason.
- "P&L or Consolidated Statement of Financial Income" in the brief is read as covering the family: Statement of Profit or Loss, Statement of Comprehensive Income, Income Statement. The Section Locator treats them as equivalent unless an entity reports them as two separate primary statements (in which case both are extracted and linked).
- Balance-sheet items used for liquidity/leverage are sourced from the Statement of Financial Position on the same document - this is necessary to compute the ratios the brief explicitly lists.
- "No internet" applies to runtime extraction. A one-time `poetry install` and `python scripts/fetch_models.py` are allowed on a connected machine; thereafter the artefact runs offline.

**Challenges and how the design absorbs them:**

1. **Scanned and font-encoded PDFs** - solved by the Ingestion triage step plus an OCR fallback that is invoked per page, not per document. AUSNET will have a mix of native-text and OCR pages in the same run.
2. **Note references with sub-letters and multi-note pointers (`3(a), 26`)** - solved by treating `note_refs` as a list of structured tokens from the start, not a string. The parser regex is `r'(\d+)(\([a-z]+\))?'` applied to the comma-split contents of the Note cell.
3. **Unit/currency drift between tables** - units are bound to the statement/sub-table, not the document, and the canonical layer scales values to a single base unit ($AUD, 1.0) before ratios are computed.
4. **Banks vs corporates** - two ratio templates and two scoring profile YAMLs. Section Locator's classifier selects.
5. **OCR digit confusion** - the reconciliation agent catches it because rows must sum. A mis-OCR'd `108.8` → `100.0` will break the subtotal and trigger a retry.
6. **LLM hallucination in the narrative report** - mitigated by (a) JSON-schema-constrained decoding everywhere except the final narrative, (b) a post-write check that verifies every numeric in the narrative appears verbatim in the source JSON; unsourced figures trigger a regeneration, and a second failure falls back to a deterministic template-rendered narrative.
7. **N=4 is too small for ML scoring** - addressed by going hybrid rules + statistical (§5) and being explicit about it.
8. **Local LLM latency** - bounded by design: ≤3 LLM calls per document (locator tiebreak, mapper tiebreak, narrative). The narrative is the dominant cost - ~3-6k input tokens, ~1k output, ~10-25s on CPU for Qwen2.5-7B, single-digit seconds on a consumer GPU. Embedding inference is a few ms per label and content-hash cached so each unique label is embedded at most once across all runs.

---

## 10. How Task 1, 2, and 3 Map Onto This Design

- **Task 1 (design doc + agentic workflow):** §3-§9. Submitted as PDF generated from this MD via WeasyPrint plus the rendered diagram.
- **Task 2 (implementation):** The repo layout in §7 is the implementation. The MVP path to a runnable demo within seven days:
  1. Day 1 - Ingestion + Section Locator on Citigroup (cleanest doc) end-to-end.
  2. Day 2 - Statement Parser + Note Resolver with golden snapshot for Citigroup.
  3. Day 3 - Canonical Mapper + Reconciliation; YHI added.
  4. Day 4 - Ratio Engine + Scoring on the two clean docs.
  5. Day 5 - OCR branch on B&E FOODS; AUSNET custom-font handling.
  6. Day 6 - Report writer + PDF; sensitivity check.
  7. Day 7 - Buffer; polish CLI; freeze.
- **Task 3 (risk model):** §5 plus the anomaly detectors. The model integrates with Task 1 by consuming `CanonicalStatement` (the deterministic output of Agent 5), so the scoring and risk-model layers can be re-run without re-extracting - important for iterating on thresholds without re-paying the parsing cost.

---

## 11. Open Decisions to Confirm with CreditSource

These are points where I have made a defensible default but the interviewer may want a different choice:

1. **Local LLM choice.** Defaulting to Qwen2.5-7B-Instruct because it is the strongest Apache-2.0 7B instruct model at the time of build, runs on CPU, and supports JSON-schema-constrained decoding. Llama-3.1-8B-Instruct is wired as a swap (single config line) for shops with an existing Llama-family deployment. A 14B/32B Qwen is supported as a `narrative_model` override when a GPU is available.
2. **Whether to extract the Statement of Financial Position.** Defaulting to yes, because the brief explicitly requires liquidity & leverage ratios which cannot be computed from the P&L alone. If the brief intends "P&L only", liquidity/leverage become `null` for all companies.
3. **Scoring weights** - defaulting to the values in §5.2; configurable via YAML so the credit team can override.
4. **Risk-report format** - defaulting to Markdown + PDF. Switch to HTML or DOCX is one templating change.
5. **GPU vs CPU deployment.** Defaulting to CPU for the case-study submission (works on any machine). For production throughput, dropping in vLLM on a single consumer GPU brings the narrative call from ~10-25s to ~2-4s and lets the same agent code parallelise across documents.

---

*End of document.*
