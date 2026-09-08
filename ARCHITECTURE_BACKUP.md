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
3. **Currency & rounding are per-statement, not per-document.** Citigroup's Profit & Loss statemen is `$Million` but its Key Management Personnel compensation note is `$000`. Both must be captured against the table that owns them.
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
                                    ▼
                        ┌───────────────────────┐
                        │  1. Ingestion & Triage │
                        │  pypdf · Docling       │
                        │  PaddleOCR (fallback)  │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  2. Section Locator   │
                        │  ToC heuristics       │
                        │  Qwen2.5-7B tiebreak  │
                        └───────────────────────┘
                                    │
                          ┌─────────┘
                          │   (retry with higher DPI / stricter thresholds)
                          │   ◄────────────────────────────────────────────┐
                          ▼                                                 │
                        ┌───────────────────────┐                          │
                        │  3. Statement Parser  │                          │
                        │  TableFormer · numeric│                          │
                        │  dateparser           │                          │
                        └───────────────────────┘                          │
                                    │                                      │
                                    ▼                                      │
                        ┌───────────────────────┐                         │
                        │  4. Note Resolver     │                         │
                        │  NoteIndex · FAISS    │                         │
                        │  Docling blocks       │                         │
                        └───────────────────────┘                         │
                                    │                                      │
                                    ▼                                      │
                        ┌───────────────────────┐                         │
                        │  5. Normaliser &      │                         │
                        │  Canonical Mapper     │                         │
                        │  BGE-small · rules    │                         │
                        │  Qwen2.5-7B fallback  │                         │
                        └───────────────────────┘                         │
                                    │                                      │
                                    ▼                                      │
                        ┌───────────────────────┐    fail (≤1 retry)      │
                        │  6. Reconciliation /  │ ────────────────────────┘
                        │  Self-Check           │
                        │  sum · sign · OCI     │
                        └───────────────────────┘
                                    │ pass
                                    ▼
                        ┌───────────────────────┐
                        │  7. Ratio Engine      │
                        │  deterministic · no   │
                        │  LLM                  │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  8. Scoring Agent     │
                        │  piecewise bands      │
                        │  weighted composite   │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  9. Risk Report Writer│
                        │  Qwen2.5-7B narrative │
                        │  WeasyPrint PDF       │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  outputs/<company>/   │
                        │  extraction.json      │
                        │  risk_report.md       │
                        │  risk_report.pdf      │
                        └───────────────────────┘
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
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | OCR confidence below threshold on a table-bearing page | Page flagged `quality: LOW`; `IngestedDocument.pages` still carries the page so downstream agents can attempt parsing; orchestrator emits report with `data_quality: low` rather than crashing |
  | Custom CMap font encoding (AUSNET) — text returns as `/0/1/2/…` glyph tokens | Detected by `chars/page < 200` or dominant-glyph-token check; page automatically re-routed through PaddleOCR at 300 DPI |
  | Docling / TableFormer fails to parse a table (unusual merged cells) | Table is absent from `IngestedDocument.tables`; Agent 2's structural scan finds no match; LLM tiebreak widens to page-text search; if no table is ever found, ratios that require it are emitted as `null` |
  | `pypdf` raises on a corrupted page | Page silently skipped; all downstream `next((p for p in doc.pages if p.page == n), None)` lookups return `None` and are handled gracefully |

### Agent 2 - Section Locator Agent

- **Role:** Find the physical PDF page range of the Statement of Profit or Loss / Consolidated Statement of Comprehensive Income, the Notes block, and (optionally) the Statement of Financial Position and Statement of Cash Flows.
- **Tools:** ToC title extraction + heading scan + structural-signature scan + **local Qwen2.5-7B-Instruct** LLM for ambiguity resolution only.
- **Navigation principle — offset-safe by design:** Financial reports carry two page numbering systems: the document's own printed numbers (referenced by the ToC) and the physical PDF page index used by PyMuPDF. These differ by an unpredictable offset (cover pages, un-numbered front matter). The agent therefore **never uses ToC page numbers for navigation**. Physical page numbers from PyMuPDF heading objects are used throughout; these are unambiguous regardless of document pagination.
- **Logic:**
  1. **ToC title extraction** — scan the first 12 pages for a ToC (`"contents"` / `"page no"` signal). For each ToC line, strip the trailing printed number and keep only the label string (e.g. `"Consolidated Statement of Comprehensive Income"`). These title strings are used to strengthen the heading scan with exact-match lookup; the printed numbers are discarded.
  2. **Heading scan** — iterate every heading extracted by Agent 1 (physical page numbers from PyMuPDF). Match each heading against a regex set covering all common P&L variants (`consolidated statement of comprehensive income`, `statement of profit or loss`, `income statement`, `statement of operations`, `statement of earnings`, etc.) **and** against the exact ToC title strings from step 1. When either matches, record the heading's physical page number. Repeat for balance sheet, cash flow, and notes patterns.
  3. **Structural-signature scan** — independently identify P&L candidates by table layout alone: any page whose table has a `Note` column header and at least one year column (`2024`, `2023` etc.). Entirely title-agnostic; catches statements with unusual or undetected headings.
  4. **Merge** heading and structural results. No ToC page numbers are included.
  5. **LLM tiebreak** — invoked only when ≥2 distinct physical pages remain (e.g. "Statement of Profit or Loss" and "Statement of Comprehensive Income" appear on separate pages). Sends up to 5 candidate page excerpts to Qwen2.5-7B with a tool call whose `enum` is constrained to the candidate page numbers — structurally invalid responses are impossible. Falls back to the first candidate if the call fails.
  6. **Continuation expansion** — once the primary P&L page is resolved, walk forward page by page. A page is included as a continuation if it has no heading that opens a new named section and contains numeric content. Stops at the first page that starts a new section (balance sheet, cash flow, notes, or another P&L heading) or has no numeric content. This handles statements that span multiple pages without requiring any manual configuration.
- **Output:** `SectionMap { pl_pages: [int], notes_pages: [int], balance_sheet_pages: [int], cashflow_pages: [int], pl_title: str, company_profile: CompanyProfile }`. `pl_pages` contains the full physical page range of the statement including any continuation pages.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | No P&L candidates found after heading scan + structural scan | `pl_pages = []`; pipeline logs a warning and continues; Agents 3–5 emit empty output; Agent 9 writes a report with all ratios `null` and `data_quality: low` |
  | LLM tiebreak fails (Ollama unreachable / timeout) | Falls back to `candidates[0]` — the first result from the merged heading+structural list |
  | No ToC found in the first 12 pages | `_scan_toc_titles` returns an empty dict; heading scan still runs via regex alone; exact-match enrichment is simply skipped |
  | Continuation expansion overshoots into the balance sheet | Agent 6's OCI / sum checks detect the inconsistency and trigger a retry, which re-runs Agent 3 with the correct page range |
  | P&L uses an unusual title not covered by any regex (e.g. "Statement of Earnings") | The structural scan (Note column + year column) identifies the page independently; LLM tiebreak selects the correct page if multiple structural candidates exist |

### Agent 3 - Statement Parser Agent

- **Role:** From the located pages, extract structured line items with note references, monetary values, units, currency, and financial year. Also extracts the Statement of Financial Position when available, so liquidity and leverage ratios can be computed.
- **Tools:** Docling/TableFormer for table extraction; `numeric.py` accounting-aware parser; **local Qwen2.5-7B-Instruct** LLM as fallback for financial year detection and unparseable numeric cells.
- **Logic:**
  1. **Table selection** — score every table on `pl_pages` by signals in its header: `Note` column (+5), year column (+3), currency marker (+2), financial keywords in row labels (+1 each). The highest-scoring table is selected. A table with no year column or fewer than 3 rows is disqualified immediately. This handles pages with multiple tables (e.g. a summary box alongside the full statement).
  2. **Column role assignment** — identify the label column (first column that is neither Note nor year), the Note column (header exactly `"note"` / `"notes"`), and all year columns (any header containing a 4-digit year). Multiple year columns are all captured to preserve the comparative period.
  3. **Units & currency detection** — scan the first 3 rows of the table and the full page text for unit markers (`$Million` → ×1,000,000; `$'000` → ×1,000). Currency detected from page text (`AUD`, `USD` etc.; defaults to `AUD`). Both stored at the statement level, not per line item.
  4. **Financial year detection — three-tier:**
     - *Tier 1 (regex):* search the P&L page and adjacent pages for `"for the (year|period|half-year) ended <date>"`. Returns immediately on match.
     - *Tier 2 (LLM):* if the regex finds nothing, send up to 2 000 chars of page text to Qwen2.5-7B with a constrained schema `{"period_end": "string|null"}`. Handles unusual phrasings such as `"Year ended 30 June 2024"` or `"12 months to 31 December 2024"` that the regex misses.
     - *Tier 3 (fallback):* returns `"unknown"` if the LLM is unavailable or returns null. The year label from the table column header (`"2024"`) is already captured independently so the pipeline continues correctly.
  5. **Row parsing — two-tier:**
     - Rows are skipped if the label is empty, starts with `$` or `Note`, or is a bare year number (header/unit rows).
     - Note references are parsed from the Note column cell: `"3(a)"` → `[{note_number:3, sub:"a"}]`; `"3(d), 26"` → two refs. Handles alphanumeric sub-references and comma-separated multi-note pointers.
     - *Tier 1 (deterministic):* each year column cell is passed through the accounting-aware `parse_number()`: `(580.8)` → `-580.8`; `1,234.5` → `1234.5`; dash/empty → `None`. Values stored as `Decimal` throughout — never `float`.
     - *Tier 2 (LLM fallback):* if every year column on a row returns `None` from `parse_number()` — indicating an OCR artefact or unusual format — the raw cell strings are sent to Qwen2.5-7B with a schema constrained to one `string|null` per year column. The LLM response is run back through `parse_number()` before acceptance. Only fires on genuine parse failures; a row that is genuinely non-numeric (a section heading) is silently dropped after both tiers return nothing.
     - Subtotal / total rows are flagged (`is_total=True`) when the label matches `^(total|net (loss|profit)|profit (before|after) tax|loss before)`. Agent 6 uses this flag to verify arithmetic.
  6. **Balance sheet extraction** — if `section_map.balance_sheet_pages` is populated, the same pipeline (steps 1–5) runs on those pages with `kind="balance_sheet"`. Required for liquidity and leverage ratios.
- **Output:** `{"profit_or_loss": Statement, "balance_sheet": Statement?}` where each `Statement` carries `{ kind, currency, units_label, units_scale, period_end, period_label, comparative_period_label, line_items: List[LineItem], pages: [int] }`.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | No table on `pl_pages` scores above 0 (e.g. statement is all text, no structured table) | `_pick_pl_table` returns `None`; `run()` emits no `profit_or_loss` key; Agent 6 receives no statement → reconciliation fails → retry loop triggers |
  | Financial year undetectable (regex miss + LLM returns null) | `period_end = "unknown"` (Tier 3 fallback); year column labels (`"2024"`, `"2023"`) are still captured from the table header so all `LineItem.values` remain correctly keyed; only the date string in the output is `"unknown"` |
  | LLM row-value fallback unavailable (Ollama down) | `_llm_parse_row_values` returns `{}`; `_build_line_item` returns `None`; row silently dropped; if many rows drop, Agent 6's `latest_year_completeness` check fires |
  | Units denomination not recognised (e.g. `$Billions`) | `units_scale = 1`, `units_label = "$"`; all values stored unscaled; Agent 6's relative arithmetic checks may still pass; absolute magnitudes are wrong but flagged as `data_quality: low` in the report |
  | Balance sheet pages empty or not identified | `out["balance_sheet"]` is not emitted; Agent 7 emits `null` for all balance-sheet-dependent ratios (current ratio, debt-to-equity, etc.) |

### Agent 4 - Note Resolver Agent

- **Role:** For each `note_ref` referenced by the statement, fetch the actual note content (text + sub-tables) from the notes section, and attach it to the line item.
- **Tools:** Docling-parsed notes blocks; a note-index built once per document; FAISS only as a fallback for fuzzy lookup (not the primary mechanism - the note numbering is deterministic).
- **Logic:**
  1. Build a `NoteIndex` mapping `note_number → {title, page, sub_sections: {'a': …, 'b': …}, tables: [...]}` from the notes pages. Headings like `3.  REVENUE RECOGNITION`, then `(a) Advisory fees`, give the structure.
  2. For each `LineItem.note_refs`, resolve each token: `3(a) → notes[3].sub_sections['a']`; `26 → notes[26]` (whole note); a bare `3 → notes[3]` (whole note).
  3. Attach the resolved note(s) to the line item with provenance. Notes that themselves contain tables (e.g. Note 3(d) which breaks down interest income/expense) have those tables extracted as `Statement`-like sub-objects so downstream ratio calculation can use the disaggregated numbers if needed.
- **Output:** `EnrichedStatement` (Statement + `notes: List[Note]` attached at line-item level).
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | Note number not found in `NoteIndex` (notes pages misidentified, or OCR garbled the note heading) | Reference recorded as a dangling ref; Agent 6's `note_linkage` check fails → retry triggered, which re-runs Agents 3–4 with the corrected page range |
  | Notes section has no structured headings (flat text, no numbered notes) | `NoteIndex` is empty; all note refs become dangling; Agent 6 flags this; pipeline continues; note text is absent from `extraction.json` but score and ratios are unaffected since they use canonical values, not note text |
  | FAISS fallback import error or index build failure | Deterministic note-number lookup is always tried first; FAISS is only invoked for fuzzy fallback; its failure means fuzzy-matched notes are not resolved but exact-number matches still are |

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
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | No rule match AND embedding cosine < 0.7 AND LLM returns null / invalid key | Label recorded in `CanonicalStatement.unmatched_labels`; Agent 7 emits `null` for any ratio whose inputs came from unmatched lines; unmatched count is surfaced in the report's Data Quality section |
  | `sentence-transformers` not installed | `FallbackEmbedder` is used (hash-based pseudo-random unit vectors, not semantically meaningful); rule-based matching still works for all exact synonyms in `canonical_labels.yaml`; only novel unseen labels that would have been rescued by embedding similarity fall through to the LLM or become unmatched |
  | LLM tiebreak unreachable (Ollama down) | Labels below the 0.7 threshold go directly to `unmatched_labels`; the rule layer handles the majority of standard Australian P&L labels without the LLM |
  | Structurally invalid LLM response (key not in canonical enum) | Prevented by JSON-schema-constrained decoding (`enum` is the canonical chart of accounts); if the constraint is somehow bypassed, the response is discarded and the label goes to `unmatched_labels` |

### Agent 6 - Reconciliation / Self-Check Agent

- **Role:** Verify the extraction is arithmetically and structurally consistent before any ratio is computed.
- **Checks (deterministic, no LLM):**
  1. **Sum check** - totals = sum of their immediate children, within tolerance of `±0.5 × 10^(units)` (i.e. ±$500k for `$Million` reports - handles rounding).
  2. **Sign check** - `profit_before_tax + income_tax_expense ≈ net_profit` (where tax is signed correctly).
  3. **OCI roll-up** - `total_comprehensive_income = net_profit + OCI`.
  4. **Note linkage** - every `note_ref` resolved to an existing note; no dangling pointers.
  5. **Cross-year completeness** - every line item has at least the latest year value.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | PBT / tax / net_profit arithmetic inconsistency | `overall_passed = False`; orchestrator's conditional edge routes to `retry_parse` (if `retries < 1`); `_retry_parse` bumps the retry counter, sets `doc.quality = LOW`, and loops back to Agent 3 which re-runs with higher OCR DPI / stricter TableFormer thresholds |
  | OCI roll-up mismatch (`net_profit + OCI ≠ total_comprehensive_income`) | Same retry path as above; typically caused by Agent 3 placing an OCI row in the wrong column |
  | Dangling note references (note_linkage check) | `overall_passed = False`; retry path re-runs Agents 3–4 with corrected page range; dangling refs listed in `ValidationReport.retry_hint` and in the report's Validation section |
  | Latest-year values missing on one or more line items | Same retry path; typically caused by a missed column or OCR-dropped cell |
  | Reconciliation still fails after 1 retry | Orchestrator falls through to `ratios` anyway (no second retry); `ValidationReport.overall_passed = False` and `retry_hint` carry the failure reason into `extraction.json`; final score is labelled `data_quality: low` |

### Agent 7 - Ratio Engine

- **Role:** Compute the financial ratios from the canonical statement. **Fully deterministic.** No LLM.
- **Ratios computed:**
  - **Profitability:** net profit margin = `net_profit / revenue.total`; gross margin (if COGS available); return on equity = `net_profit / avg_equity`; EBIT margin.
  - **Liquidity:** current ratio = `current_assets / current_liabilities`; quick ratio = `(current_assets - inventory) / current_liabilities`.
  - **Leverage:** debt-to-equity = `total_debt / total_equity`; debt-to-assets.
  - **Coverage:** interest coverage = `EBIT / interest_expense`; (for financials, this becomes `net_interest_margin`).
  - **Trend:** YoY delta of each of the above (using the comparative year already in the statement).
- **Bank-vs-corporate branch:** Banks (Citigroup) don't have a meaningful "current ratio"; the engine has two profile templates and the Section Locator's classification (financial vs non-financial corporate, derived from ANZSIC heuristics on the directors' report) selects the template. Ratios that don't apply emit `null` with a reason, not `0`.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | Input value is `None` (line item unmatched in Agent 5 or balance sheet not extracted) | `_div` returns `None`; ratio emitted with `value: null`; Agent 8 skips it for composite scoring but includes it in output with null value |
  | Division by zero (e.g. `revenue.total = 0`) | `_div` explicitly checks `b == 0` and returns `None`; treated the same as a missing input |
  | Comparative period missing (first-year filer or single-year statement) | `canon.comparative_period_label = None`; YoY delta computation is skipped entirely; all `ratio.yoy_delta = None`; anomaly detector in Agent 8 has nothing to compare — no false deterioration flags are emitted |
  | All ratios null (complete upstream extraction failure) | `category_aggregates` is empty; `composite = None`; Agent 8 emits `band = None`; Agent 9 falls back to deterministic template narrative |

### Agent 8 - Scoring Agent

- **Role:** Convert ratios into a composite credit risk score 0-100 (higher = healthier) with documented sub-scores. **Hybrid rules + statistical** - see §5.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | Ratio name has no entry in `scoring.yaml` `ratio_cfg` (e.g. a new ratio added to Agent 7 without a YAML entry) | `rconf = None`; ratio skipped for scoring; it still appears in `ratios[]` in the output but with no `sub_score` |
  | All sub-scores null (all ratio inputs were `None`) | `composite = None`, `band = None`; report written with "n/a" for score; deterministic template narrative is used in Agent 9 |
  | `scoring.yaml` malformed or missing | `scoring_config()` raises at load time; pipeline fails before Agent 8 runs; fix by correcting the YAML — no code change needed |
  | Anomaly keyword list empty (removed from YAML) | Note-disclosure detector produces no flags; pipeline continues normally — silent, not a crash |
  | YoY delta unavailable (first-year filer) | YoY deterioration detector is skipped; no false anomaly flags are emitted |

### Agent 9 - Risk Report Writer

- **Role:** Emit the per-company structured artefact and a short analyst-facing narrative.
- **Tools:** the **local Qwen2.5-7B-Instruct** for the narrative; WeasyPrint for the PDF. A larger local model (Qwen2.5-14B-Instruct or Qwen2.5-32B-Instruct) is wired as an optional `narrative_model` override in `config/scoring.yaml` for engagements with a GPU budget; in routine use the 7B prose is fluent enough for an analyst-facing risk summary and runs comfortably on CPU.
- **Outputs:**
  - `outputs/<company>/extraction.json` - full structured extraction (statement + notes + canonical + ratios + score).
  - `outputs/<company>/risk_report.md` (and PDF via `weasyprint`) - narrative containing: headline score, key ratios with YoY delta, material risks/anomalies the validator flagged, data-quality notes.
- The narrative is the **only** place the LLM writes free-form text. It is constrained to facts from the structured object - prompted with the JSON and instructed not to introduce numbers not present in the input. A post-write check greps the narrative for numerics and verifies each appears in the source object; any unsourced figure triggers a regeneration with the offending span quoted back to the model. On a second failure the run falls back to a **deterministic template-rendered narrative** built directly from the structured JSON - no free-form text, no risk of hallucination. The structured artefact is always emitted regardless.
- **Failure modes & workarounds:**

  | Failure | How it is handled |
  |---------|-------------------|
  | Ollama unreachable when generating the narrative | `llm.text()` raises; immediately falls back to `_template_narrative()` — a deterministic Markdown render of the structured data; `extraction.json` and `risk_report.md` are always written |
  | Narrative contains unsourced figures (hallucination on first attempt) | Offending numeric tokens are quoted back to the model with an explicit rewrite instruction; one regeneration attempt is made |
  | Narrative still contains unsourced figures after regeneration | `_template_narrative()` fallback is used; the report's Data Quality section notes that the LLM narrative failed fact-check |
  | WeasyPrint not installed or PDF rendering crashes | Warning logged; `risk_report.pdf` is skipped; `extraction.json` and `risk_report.md` are always written regardless |
  | `outputs/` directory not writable | `write_text` raises; pipeline crashes at the final step; all upstream work is checkpointed in the LangGraph SQLite saver — rerunning the pipeline resumes from Agent 9 without re-parsing |

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

> **Important — thresholds and weights must be recalibrated for each deployment context.**
> The values shipped in `config/scoring.yaml` are reasonable starting points derived from publicly cited S&P/Moody's industrial band ranges
>
> All threshold and weight changes are made solely in `config/scoring.yaml`. No code change is required, and the file is reloaded on every run so changes take effect immediately.

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

## 10. Output Formats & Examples

The pipeline emits three files per company into `outputs/<company>/`. The examples below are drawn from the Citigroup document and illustrate every required data point.

---

### 10.1 `extraction.json` — full structured artefact

This is the primary machine-readable output. It carries every extracted number with its provenance, note links, currency context, computed ratios, and the final score. All monetary values are stored in **base currency units (AUD, cents = 0)** after scaling — the `units_label` and `units_scale` fields on the statement record what the source document used so the conversion is auditable.

```jsonc
{
  "company_name": "CITIGROUP PTY LTD",
  "period_label": "FY2024",
  "quality": "high",

  // ── (3) Currency & rounding ──────────────────────────────────────────────
  // Declared once per statement; all line item values are scaled to base AUD.
  "enriched_statement": {
    "statement": {
      "kind": "profit_or_loss",
      "title": "Consolidated Statement of Comprehensive Income",
      "currency": "AUD",
      "units_label": "$Million",          // source document denomination
      "units_scale": 1000000,             // multiply raw value × 1 000 000 → base AUD
      "period_end":  "2024-12-31",
      "period_label": "FY2024",           // (2) financial year this column refers to
      "comparative_period_end": "2023-12-31",
      "comparative_period_label": "FY2023",

      // ── (1) Note links + (2) financial year per line item ─────────────────
      "line_items": [
        {
          "label": "Advisory fees",
          "page": 8,
          "note_refs": [
            { "note_number": 3, "sub": "a", "raw": "3(a)" }   // link to Note 3(a)
          ],
          "values": {
            "FY2024": "108800000",   // 108.8 × 1 000 000 — base AUD
            "FY2023": "101600000"
          },
          "is_subtotal": false,
          "is_total": false
        },
        {
          "label": "Interest income",
          "page": 8,
          "note_refs": [
            { "note_number": 3, "sub": "d", "raw": "3(d)" }
          ],
          "values": { "FY2024": "529600000", "FY2023": "473200000" }
        },
        {
          "label": "Interest expense",
          "page": 8,
          "note_refs": [
            { "note_number": 3, "sub": "d", "raw": "3(d)" }   // same note, expense side
          ],
          "values": { "FY2024": "-580800000", "FY2023": "-521100000" }
          // negative = parenthesised in source: (580.8) → -580.8
        },
        {
          "label": "Management fee income",
          "page": 8,
          "note_refs": [
            { "note_number": 26, "sub": null, "raw": "26" }   // plain note reference
          ],
          "values": { "FY2024": "23700000", "FY2023": "13900000" }
        },
        {
          "label": "Income tax benefit",
          "page": 8,
          "note_refs": [
            { "note_number": 4, "sub": null, "raw": "4" }
          ],
          "values": { "FY2024": "12400000", "FY2023": "8300000" }
        }
        // … remaining line items …
      ]
    },

    // Resolved note text attached to each note_ref above
    "notes": [
      {
        "note_number": 3, "sub": "a",
        "title": "Advisory fees",
        "text": "Advisory fees are recognised when the related service has been provided …",
        "page": 12
      },
      {
        "note_number": 3, "sub": "d",
        "title": "Interest income and interest expense",
        "text": "Interest income and expense are recognised using the effective interest method …",
        "page": 14,
        "tables": [
          {
            "header": ["", "FY2024 $Million", "FY2023 $Million"],
            "rows": [
              ["Interest income on loans",  "312.4", "280.1"],
              ["Interest income on deposits", "217.2", "193.1"]
            ]
          }
        ]
      }
    ]
  },

  // ── Canonical mapping (normalised labels, same values) ───────────────────
  "canonical": {
    "company_name": "CITIGROUP PTY LTD",
    "profile": "financial",
    "period_label": "FY2024",
    "comparative_period_label": "FY2023",
    "currency": "AUD",
    "base_unit": 1,
    "lines": [
      {
        "canonical": "revenue.fee_income",
        "original_label": "Advisory fees",
        "values": { "FY2024": "108800000", "FY2023": "101600000" },
        "note_refs": ["3(a)"],
        "match_method": "rule",
        "match_score": 1.0
      },
      {
        "canonical": "revenue.interest_income",
        "original_label": "Interest income",
        "values": { "FY2024": "529600000", "FY2023": "473200000" },
        "note_refs": ["3(d)"],
        "match_method": "rule",
        "match_score": 1.0
      },
      {
        "canonical": "expense.interest",
        "original_label": "Interest expense",
        "values": { "FY2024": "-580800000", "FY2023": "-521100000" },
        "note_refs": ["3(d)"],
        "match_method": "rule",
        "match_score": 1.0
      },
      {
        "canonical": "income_tax_expense",
        "original_label": "Income tax benefit",
        "values": { "FY2024": "12400000", "FY2023": "8300000" },
        "note_refs": ["4"],
        "match_method": "rule",
        "match_score": 1.0
      }
    ],
    "unmatched_labels": []   // any labels the mapper could not classify
  },

  // ── (4) Financial ratios ─────────────────────────────────────────────────
  "ratios": {
    "profile": "financial",
    "period_label": "FY2024",
    "comparative_period_label": "FY2023",
    "ratios": [
      {
        "name": "net_profit_margin",
        "category": "profitability",
        "formula": "net_profit / revenue.total",
        "value": 0.021,            // 2.1 % — net profit ÷ total revenue
        "yoy_delta": 0.004,        // improved 0.4 pp vs FY2023
        "inputs_used": {
          "net_profit":    12400000,
          "revenue.total": 143700000
        }
      },
      {
        "name": "cost_to_income",   // bank-specific; replaces EBIT margin
        "category": "profitability",
        "formula": "total_operating_expenses / total_operating_income",
        "value": 0.812,
        "yoy_delta": -0.031,
        "inputs_used": {
          "total_operating_expenses": 116700000,
          "total_operating_income":   143700000
        }
      },
      {
        "name": "equity_to_assets", // bank capital adequacy proxy
        "category": "capital",
        "formula": "total_equity / total_assets",
        "value": 0.064,
        "yoy_delta": 0.002,
        "inputs_used": {
          "total_equity":  94200000,
          "total_assets": 1471900000
        }
      },
      {
        "name": "return_on_equity",
        "category": "asset_quality",
        "formula": "net_profit / total_equity",
        "value": 0.132,
        "yoy_delta": 0.011,
        "inputs_used": {
          "net_profit":   12400000,
          "total_equity": 94200000
        }
      }
      // corporate template also includes: current_ratio, quick_ratio,
      // debt_to_equity, interest_coverage, ebit_margin, gross_margin
    ]
  },

  // ── (5) Composite credit risk score ─────────────────────────────────────
  "score": {
    "company_name": "CITIGROUP PTY LTD",
    "profile": "financial",

    // Sub-scores: each ratio mapped to 0-100 via piecewise-linear band thresholds
    // (thresholds defined in config/scoring.yaml, calibrated to S&P/Moody's bands)
    "sub_scores": [
      { "ratio": "net_profit_margin", "value": 0.021, "sub_score": 60.5, "band": "BBB", "category": "profitability" },
      { "ratio": "cost_to_income",    "value": 0.812, "sub_score": 44.0, "band": "BB",  "category": "profitability" },
      { "ratio": "equity_to_assets",  "value": 0.064, "sub_score": 78.0, "band": "A",   "category": "capital"       },
      { "ratio": "return_on_equity",  "value": 0.132, "sub_score": 82.5, "band": "A",   "category": "asset_quality" }
    ],

    // Weighted composite:
    //   profitability  avg(60.5, 44.0) = 52.3  × 0.30 = 15.7
    //   capital        avg(78.0)       = 78.0  × 0.30 = 23.4
    //   asset_quality  avg(82.5)       = 82.5  × 0.25 = 20.6
    //   liquidity      n/a (no data)   → weight redistributed
    //   composite = (15.7 + 23.4 + 20.6) / (0.30+0.30+0.25) = 59.7 / 0.85 = 70.2
    "composite": 70.2,
    "band": "A",          // composite ≥ 70 → band A  (see composite_bands in scoring.yaml)

    "weights_used": {
      "profitability": 0.30,
      "capital":       0.30,
      "asset_quality": 0.25,
      "liquidity":     0.15
    },

    // Anomaly flags — independent of composite, surfaced in the report
    "anomalies": [
      {
        "kind": "yoy_deterioration",
        "severity": "warning",
        "description": "cost_to_income sub-score fell 28 pts YoY (72 → 44); raw value 0.781 → 0.812"
      }
    ]
  },

  // Arithmetic self-check results (Agent 6)
  "validation": {
    "overall_passed": true,
    "checks": [
      { "name": "sum_check",          "passed": true,  "detail": "all subtotals match within ±$500k tolerance" },
      { "name": "sign_check",         "passed": true,  "detail": "profit_before_tax + tax_benefit ≈ net_profit" },
      { "name": "oci_rollup",         "passed": true,  "detail": "total_comprehensive_income = net_profit + OCI" },
      { "name": "note_linkage",       "passed": true,  "detail": "all 7 note_refs resolved" },
      { "name": "year_completeness",  "passed": true,  "detail": "FY2024 value present on all line items" }
    ]
  }
}
```

---

### 10.2 `risk_report.md` — analyst-facing narrative

```markdown
# Risk report — CITIGROUP PTY LTD

**Period:** FY2024    **Profile:** financial    **Data quality:** high

**Composite score:** 70.2 / 100  →  band **A**

## Headline

Citigroup Pty Ltd posted a composite credit score of 70.2 (band A) for FY2024,
reflecting adequate capitalisation (equity-to-assets 6.4%) and improving
profitability (net profit margin 2.1%, up 0.4 pp YoY). The primary concern is a
rising cost-to-income ratio of 81.2%, which deteriorated 3.1 pp versus FY2023 and
contributed a warning flag on YoY sub-score deterioration. Net interest margin
remains compressed (interest income $529.6M vs expense $580.8M).

## Key ratios

| Ratio              | Value  | YoY Δ    | Sub-score | Band |
|--------------------|--------|----------|-----------|------|
| net_profit_margin  | 0.021  | +0.004   | 61        | BBB  |
| cost_to_income     | 0.812  | +0.031   | 44        | BB   |
| equity_to_assets   | 0.064  | +0.002   | 78        | A    |
| return_on_equity   | 0.132  | +0.011   | 83        | A    |

## Anomalies & material risks

- **[warning]** (yoy_deterioration) cost_to_income sub-score fell 28 pts YoY
  (72 → 44); raw value 0.781 → 0.812

## Validation

Overall: **PASS**

- ✅ `sum_check` — all subtotals match within ±$500k tolerance
- ✅ `sign_check` — profit_before_tax + tax_benefit ≈ net_profit
- ✅ `oci_rollup` — total_comprehensive_income = net_profit + OCI
- ✅ `note_linkage` — all 7 note_refs resolved
- ✅ `year_completeness` — FY2024 value present on all line items

## Data quality

Document quality: **high**. Unmatched labels: 0.
Statement currency: **AUD**, denomination: **$Million** (values scaled ×1 000 000
to base AUD in extraction.json).
```

---

### 10.3 `risk_report.pdf`

The Markdown above rendered to PDF via WeasyPrint with a clean sans-serif stylesheet and a bordered ratio table. Identical content to the `.md`; intended for distribution to stakeholders who do not work with structured data files.

---

### 10.4 Coverage of required data points

| Requirement | Where it appears in the output |
|---|---|
| **(1) Note links per line item** | `enriched_statement.statement.line_items[*].note_refs` — each ref carries `note_number`, `sub` (e.g. `"a"`), and `raw` (e.g. `"3(a)"`); full note text in `enriched_statement.notes[]` |
| **(2) Financial year per item** | `statement.period_label` / `comparative_period_label`; `line_items[*].values` keyed by year label (`"FY2024"`, `"FY2023"`) |
| **(3) Currency & rounding** | `statement.currency` (`"AUD"`), `statement.units_label` (`"$Million"`), `statement.units_scale` (`1000000`); all values in `extraction.json` are in base AUD |
| **(4) Financial ratios** | `ratios.ratios[]` — name, value, YoY delta, formula, raw inputs used; covers profitability, liquidity, leverage, coverage, and bank-specific ratios |
| **(5) Composite credit score** | `score.composite` (0–100), `score.band` (AA→D), `score.sub_scores[]` per ratio, `score.weights_used`, `score.anomalies[]`; scoring methodology detailed in §5 |

---

## 11. How Task 1, 2, and 3 Map Onto This Design

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

## 12. Open Decisions to Confirm 

These are points where I have made a defensible default but the company / lead architect may want a different choice:

1. **Local LLM choice.** Defaulting to Qwen2.5-7B-Instruct because it is the strongest Apache-2.0 7B instruct model at the time of build, runs on CPU, and supports JSON-schema-constrained decoding. Llama-3.1-8B-Instruct is wired as a swap (single config line) for shops with an existing Llama-family deployment. A 14B/32B Qwen is supported as a `narrative_model` override when a GPU is available.
2. **Whether to extract the Statement of Financial Position.** Defaulting to yes, because the brief explicitly requires liquidity & leverage ratios which cannot be computed from the P&L alone. If the brief intends "P&L only", liquidity/leverage become `null` for all companies.
3. **Scoring weights** - defaulting to the values in §5.2; configurable via YAML so the credit team can override.
4. **Risk-report format** - defaulting to Markdown + PDF. Switch to HTML or DOCX is one templating change.
5. **GPU vs CPU deployment.** Defaulting to CPU for the case-study submission (works on any machine). For production throughput, dropping in vLLM on a single consumer GPU brings the narrative call from ~10-25s to ~2-4s and lets the same agent code parallelise across documents.

---

*End of document.*
