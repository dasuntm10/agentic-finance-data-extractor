# CreditSource — Lead AI/ML Engineer Interview Q&A

**Setting:** Technical interview for the AI Engineer role. The interviewer (Lead AI/ML Engineer at CreditSource) has read `ARCHITECTURE.md`, skimmed the repo, and is now probing the design and implementation. The candidate is expected to defend trade-offs, surface hidden risks, and demonstrate that the choices are deliberate rather than incidental.

Questions are organised by theme. Answers are written as the candidate would deliver them: direct, specific, and grounded in the design.

---

## A. Framing & top-level design

### Q1. Why agentic at all? A single big-model call with the PDF attached would arguably do most of this. Defend the orchestration overhead.

A single-LLM extractor fails on this corpus for three concrete reasons. First, the documents are layout-heavy — Citigroup's P&L on p.8 has a `Note` column with alphanumeric refs (`3(a)`, `3(d), 26`) that depend on table structure; an LLM reading flattened text reorders these and silently misaligns rows. Second, the numbers must reconcile — totals equal the sum of children within a rounding tolerance — and a generative model has no obligation to make that hold. Third, AUSNET has custom-font CMap pages and B&E FOODS is fully scanned; a single call can't decide *per page* whether to OCR. The agentic split lets me use TableFormer for tables, deterministic Decimal parsing for numbers, an arithmetic reconciliation check, and reserve local-LLM calls for the three places where reasoning actually helps: classification tiebreaks, label canonicalisation, and the analyst narrative. The "overhead" is one StateGraph definition and typed Pydantic contracts — a one-time cost that pays back the first time reconciliation catches an OCR digit confusion.

### Q2. Why LangGraph over LlamaIndex Workflows, or a plain Python DAG like Prefect/Dagster?

LangGraph wins on three axes for this workload. (1) Typed shared state with conditional edges — the reconcile → re-extract loop is a first-class construct, not a hand-rolled retry. (2) The SqliteSaver checkpointer lets me resume a long run and replay individual nodes, which matters for iterating on scoring without re-paying parsing cost. (3) Debuggability — every node transition is inspectable. LlamaIndex Workflows is a close second and I'd accept it; Prefect/Dagster are heavier and built for cross-process scheduling, not in-process agent state.

### Q3. Nine agents feels like over-decomposition. Could you collapse any?

Yes — Ratio Engine and Scoring Agent could merge, and Reconciliation could be a method on the Statement Parser. I kept them separate because each has a different *failure semantics*: Reconciliation can reject and retry; Ratio Engine emits `null` with a reason; Scoring is config-driven. Collapsing them hides the boundary between deterministic checks and policy. The Section Locator, Statement Parser, Note Resolver, and Canonical Mapper, on the other hand, are genuinely different jobs — I would not merge those.

### Q4. The brief says no hosted third-party services. Walk me through how this design respects that.

Strictly. Every model in the pipeline runs locally: Docling layout + TableFormer for table structure, PP-OCRv4 for scanned pages, BGE-small-en-v1.5 for embeddings, and Qwen2.5-7B-Instruct served by Ollama for the three LLM-touched steps. The Pydantic contracts and JSON-schema-constrained decoding give the same structural guarantees a hosted "forced tool call" would — invalid LLM outputs are impossible by construction, on-device. The only network activity in the whole project is `poetry install` and a one-time `python scripts/fetch_models.py` to pull weights into `./models/`; the runtime itself is air-gapped. If CreditSource's data-handling policy forbids egress of audited financials — which it likely does — this design is already aligned with that policy out of the box.

---

## B. Document ingestion

### Q5. Walk me through what happens when AUSNET hits pp.4-5 with the custom-font CMap.

The Ingestion agent probes each page with `pypdf`'s text extraction. On those pages the extracted text is dominated by `/0/1/2/...` glyph tokens — recognisable by either very low char count or a high proportion of slash-prefixed digit tokens. The page is flagged `needs_ocr` and routed to PaddleOCR at 300 DPI. Critically the *rest* of AUSNET stays on the native-text path through Docling; OCR is per-page, not per-document. The page emits a `PageBlock` with `source="ocr"` and an `ocr_confidence` score so downstream agents know the provenance.

### Q6. Why Docling over Unstructured, LlamaParse, or AWS Textract?

Docling (IBM, Apache-2.0) ships TableFormer, which is the strongest open-source model on financial table structure I've benchmarked — it preserves logical row/column structure rather than emitting flat text. Unstructured is more general but weaker on dense numeric tables. LlamaParse and Textract are hosted services and so violate the offline constraint of the brief. Docling runs locally, weights cache to `./models/`, and the licence is permissive.

### Q7. Why PaddleOCR over Tesseract for the scanned branch?

Tesseract is fine on prose but degrades on dense numeric tables with column alignment — it misreads `108.8` as `1088` or `108 8` more often than PP-OCRv4. B&E FOODS is exactly this kind of document. PaddleOCR also exposes PP-StructureV2 which gives us table structure for free; with Tesseract I'd need a separate table-detection pass. The trade is install footprint (PaddleOCR pulls torch); for our case it's worth it.

### Q8. How do you catch OCR digit confusion? `8` vs `0` in `108.8` would still produce a valid-looking number.

Two layers. Layer one is the Reconciliation agent: rows must sum to subtotals within ±0.5 × 10^(units). A `108.8 → 100.0` mis-OCR breaks the row-sum check on the P&L's subtotal, which routes the page back to Ingestion for re-OCR at 400 DPI. Layer two is OCR confidence — PaddleOCR exposes per-token confidence; values below threshold are flagged on the line item and the document quality drops to `medium`. The combination catches both arithmetic-detectable errors and statistically-flagged ones; what neither catches is a confusion that *also* happens to balance, which I'd address with a third-pass cross-check against the year-over-year value (a ±50% YoY swing on a stable line is suspicious).

---

## C. Section location & statement parsing

### Q9. When exactly does the LLM get called in the Section Locator? Show me the failure mode if Ollama is down.

The LLM is called only when both deterministic paths fail: the ToC parse and the regex/structural-signature scan return ≥2 plausible candidates and no tiebreaker. On Citigroup that doesn't happen — the ToC names `Consolidated statement of comprehensive income → 5` unambiguously. It triggers on entities that report P&L and Comprehensive Income as two separate primary statements. If Ollama isn't running, the locator falls back to "pick the candidate page with the largest table containing a `Note` column header" — a deterministic rule. The narrative agent flags `section_location: heuristic_fallback` so the analyst knows.

### Q10. The note-ref parser regex is `(\d+)(\([a-z]+\))?` applied to comma-split cells. What breaks it?

A few patterns. (1) Range notation like `3(a)-(c)` — I'd extend the parser to enumerate. (2) Footnote markers like `3*` or daggered superscripts — these aren't note refs, they're auditor cross-references; the regex correctly ignores them. (3) Whitespace-separated rather than comma-separated, e.g. `3 26`; I normalise whitespace inside the cell before splitting on `[,\s]+`. (4) Notes embedded in the label text rather than the Note column — rare, but I'd handle via a second pass. The honest answer is the regex is good for the four sample docs; a fifth document could surface a new pattern and I'd add a golden test case.

### Q11. Citigroup mixes `$Million` in the P&L and `$'000` in the KMP note. How does that propagate?

Units are bound to the `Statement` and `TableBlock`, not to the document or to individual line items. The Statement Parser detects the unit row immediately under the column header and stores `currency='AUD', units='millions'` at the statement level. When the Note Resolver attaches Note 26 (KMP comp at `$'000`), it stores `units='thousands'` on that sub-table. The Canonical Mapper rescales everything to a single base unit (AUD, 1.0) before the Ratio Engine runs. The rescaling happens once, in the normaliser, and the original `units` field is retained for audit.

### Q12. What if the unit row is ambiguous — e.g. the header just says `$` with no scale?

Three signals, in priority order. (1) An explicit "All amounts in $X" line in the surrounding text. (2) The order of magnitude — a global bank's "Net interest income" of `1.2` is implausibly in dollars and is almost certainly millions; this is a sanity check, not a primary signal. (3) Cross-reference to the prior year on the same row and the matching value in the company's announced press-release summary if present in the document. If all three fail, the statement is flagged `units: unknown` and the document quality drops to `low` — better to surface the ambiguity than guess.

---

## D. Canonical mapping & embeddings

### Q13. Why rules → embeddings → LLM tiebreak rather than just an LLM call per label?

Determinism and latency. The canonical chart of accounts has ~30 entries; most labels match by exact synonym or trivial regex (`Revenue from continuing operations → revenue.total`). Embeddings handle the 70% that don't match exactly but are semantically obvious (cosine > 0.85). The LLM tiebreak is reserved for the ~5% where similarity is ambiguous (top-2 both > 0.7 with small gap). This costs ~30 BGE-small embedding calls per document — a few hundred ms total on CPU, all cached after first run — and 1-2 Qwen calls on average. A pure-LLM mapper would be 30 LLM calls per doc, non-deterministic across runs unless seeded, and harder to audit.

### Q14. How do you guarantee the LLM returns a valid canonical label?

JSON-schema-constrained decoding. The prompt is paired with a schema whose `canonical_label` field is `enum: [<canonical labels>]`. Under Ollama I pass `format: <schema>`; under vLLM the equivalent is `outlines` or `lm-format-enforcer`. Either way the decoder is mathematically prevented from emitting a token that would lead to an invalid JSON value — invalid labels are structurally impossible. The fallback if the LLM is unreachable (e.g. Ollama not running) is to take the top embedding candidate and flag the line item `mapping: low_confidence`.

### Q15. BGE-small is local — what's the cost and resource story?

The cache design: each unique label string is embedded at most once across all runs, keyed by SHA-256 of the input. The cache lives at `outputs/<company>/_cache/embeddings.jsonl`. On a re-run of the same corpus, embedding work is zero. First-run cost is ~30 labels × 4 docs × a few ms each on CPU — sub-second total. The model itself is ~33M params (~130MB on disk) and loads in under a second. No egress, no API key, no rate limit — that's the point.

---

## E. Reconciliation & validation

### Q16. Walk me through a real reconciliation failure on Citigroup.

Suppose Brokerage = 66.1, Net trading income = (4.3), and the Statement Parser misreads "Brokerage" as 661.0 (decimal lost). Reconciliation runs: `Operating income = sum of components`. The stated `Total operating income` on the page is, say, 250.4; the recomputed sum is 845.3. The diff exceeds tolerance (±0.5 for `$Million`). The agent emits a `ValidationReport` pinning the failing region to the Brokerage row's bbox. The orchestrator routes back to Agent 3 with that region pinned and re-rasterises at 400 DPI, re-runs TableFormer. If the re-extract produces `66.1`, the sum balances and the run proceeds. If two retries fail, the document continues with `data_quality: low` and the offending row is called out in the narrative.

### Q17. Why ±0.5 × 10^(units) tolerance? Why not zero?

Rounding. Annual reports round to the displayed unit — a P&L in `$Million` displays `108.8` for an underlying `108,750,000` or `108,849,999`. The sum of children rounded to one decimal place can legitimately differ from the rounded total by up to half a unit. ±0.5 is the mathematically correct tolerance for half-up rounding to one decimal; for `$'000` it becomes ±$500. Zero tolerance would generate false-positive retries on every document.

### Q18. What's the contract for "two retries then continue with low quality"? Why not fail hard?

Operational pragmatism. A credit analyst's workflow tolerates a flagged report — they re-check the source. A hard failure means zero output for that document, which is worse than a flagged partial output. The `data_quality: low` marker is surfaced in the narrative's first paragraph, the structured JSON has a `quality_flags: [...]` array listing the specific failures, and the analyst can choose to discard. For batch ingestion, the orchestrator's exit code distinguishes "completed with flags" (1) from "fatal" (2) so a CI pipeline can decide.

---

## F. Scoring model

### Q19. N=4. You can't train anything. Why do you have a scoring section at all rather than just emitting ratios?

The brief explicitly asks for a composite credit score (Task 1) and a risk model (Task 3). The honest answer is the score is a *rules-based scorecard with documented thresholds*, not a learned model. The thresholds are anchored to public S&P/Moody's industrial bands so they're auditable and defensible. The "statistical" part is the optional PCA-derived weighting reported as a sensitivity check — explicitly not the canonical answer at N=4. The doc is upfront about this in §5.1; pretending I'd trained an XGBoost would be the wrong answer.

### Q20. Where do the S&P thresholds come from? Are they publicly citable?

S&P's "Corporate Methodology: Ratios And Adjustments" (most recent revision publicly available) defines indicative ranges per ratio per rating band. Moody's publishes similar guidance via its rating methodologies for non-financial corporates. The thresholds in `config/scoring.yaml` are extracted from these published documents, cited inline as YAML comments with the source paragraph. They're not proprietary. A credit team at CreditSource would likely overwrite them with their own internal calibration, which is why they live in YAML and not in code.

### Q21. The bank template uses different ratios. How is "this is a bank" decided?

The Section Locator runs an industry classifier on the directors' report — ANZSIC code if disclosed, otherwise keyword signals (`authorised deposit-taking`, `licensed financial services`, presence of `Net interest income` as a primary revenue line, regulatory references to APRA). Citigroup hits all four. If signals conflict, the run goes corporate-template by default and flags the ambiguity. Banks and corporates differ on which ratios are meaningful (current ratio is largely meaningless for a bank) and on threshold bands (banks operate at materially higher leverage), so the template choice matters more than threshold tuning within a template.

### Q22. Citigroup's parent has a public rating. How do you sanity-check?

For CGM Australia specifically, I'd compare the bank-template composite to the parent's published rating band (`A-` or thereabouts as of recent reports). If my scorecard returns `B` or `D` for the subsidiary, that's a red flag for either the thresholds or the extraction — the subsidiary won't perfectly match the parent because it's a different legal entity with different risk, but a multi-notch gap warrants investigation. This is a one-document anchor, not a validation methodology — I'd be explicit in the report that it's a sanity check, not a calibration.

### Q23. What's your sensitivity-analysis methodology?

Re-score each company with ±10% perturbation applied independently to each input ratio. If the composite moves smoothly and proportionally, the scorecard is well-conditioned. Discontinuities expose threshold cliffs — a 9% input change that swings the composite 30 points means the line item is sitting right at a band boundary, which the report should call out. This is reported as a `sensitivity` block in the structured output alongside the headline score.

### Q24. Anomaly detection in Task 3 — going-concern keyword search seems brittle.

It is, and I treat it as a recall mechanism not precision. The detector flags candidate sentences containing `going concern`, `material uncertainty`, `qualified opinion`, etc., and attaches them to the report with surrounding context. The analyst reads them. False positives ("the directors are not aware of any matters that would cast doubt on the going concern basis" — a routine boilerplate statement) are common and acceptable. False negatives are the actual risk; for those I'd add a semantic similarity pass against a curated set of historically problematic disclosures, which BGE-small makes essentially free (a few ms per sentence on CPU). I haven't done that in v1 because N=4 doesn't have a positive example to anchor against.

---

## G. LLM usage & cost

### Q25. Total LLM calls per document on the happy path?

Three Qwen calls maximum: zero or one for the locator tiebreak (usually zero), zero to two for mapper tiebreaks on ambiguous labels, and one for the narrative. The narrative is the dominant call — ~3-6k input tokens, ~1k output. Embeddings are ~30 BGE-small inferences per first run of a document (sub-second on CPU), zero on re-run. If the narrative hallucination check fails twice, the run falls back to a deterministic template-rendered narrative — no third LLM call.

### Q26. Defend running the narrative on a 7B local model. Wouldn't a larger model write better?

A larger model would write more fluently, yes — but the marginal quality difference on a constrained credit-summary task is small once you pin the structure (mandatory sections, sourced numbers only, ~400 words). The 7B is fluent enough that a credit analyst is reading findings, not prose. The hard constraint is offline operation, and Qwen2.5-7B-Instruct is the strongest Apache-2.0 7B I have weights for that runs on CPU. For shops with a GPU budget, the same `LLMClient` wiring accepts `qwen2.5:14b-instruct` or `qwen2.5:32b-instruct` via a single config change (`narrative_model` in `config/scoring.yaml`) — useful for production but not required for the case study.

### Q27. The narrative is the only free-form-text touchpoint. How do you stop it from inventing numbers?

Post-write check: every numeric token in the narrative must appear verbatim in the source structured JSON. The check is a regex pass over the narrative, comparing each match against the values in `extraction.json`. Unsourced figures trigger a regeneration with the offending span quoted back to the model as a correction. A second failure falls back to a deterministic template-rendered narrative built directly from the structured JSON — no LLM, no hallucination surface. The check has caught fabricated comparatives in testing — usually the model interpolating a "growth of approximately 12%" when no growth rate was supplied.

### Q28. What's stopping the model from rephrasing numbers — "approximately 100 million" when the source says 108.8?

The regex catches `\d+(\.\d+)?` and rejects unsourced numerics. "Approximately 100" would fail because `100` is not in the source. I instruct the model explicitly to use exact figures or to omit them; round-number paraphrasing is a regeneration trigger. Words like "approximately" without a number attached are allowed.

---

## H. Operational concerns

### Q29. How long does one document take end-to-end?

On a laptop CPU (no GPU), wall-clock estimates: Citigroup (clean, 41 pages) ~45-60s, dominated by Docling layout and the Qwen narrative call (~15-25s on CPU). B&E FOODS (scanned, 30 pages) ~90-120s, dominated by PaddleOCR. AUSNET (mixed, 82 pages) ~75s. On a single consumer GPU the narrative drops to ~2-4s and OCR to ~10-15s, bringing Citigroup under 20s. The narrative runs in parallel with nothing — there's headroom to overlap it with the report PDF render, but I haven't.

### Q30. What's the resource footprint per document?

Disk: ~6GB for Qwen2.5-7B quantised (Q4_K_M in Ollama's default), ~130MB for BGE-small, ~500MB for Docling layout + TableFormer, ~200MB for PP-OCRv4. Total under 8GB of weights once and for all. RAM: ~6-8GB working set during a run (the LLM dominates). Outputs per document: tens of KB JSON + a few hundred KB PDF. No per-document monetary cost — it's all local compute.

### Q31. What breaks if Ollama isn't running?

Three failure modes by agent. (1) Section Locator's LLM tiebreak — fallback to deterministic largest-table heuristic. (2) Canonical Mapper's LLM tiebreak — fallback to top embedding candidate with `mapping: low_confidence`. (3) Narrative writer — falls back to the deterministic template-rendered narrative built directly from the structured JSON. The structured JSON still renders in all three cases; the run exits with code 1 (completed with flags) and the report banner says "narrative: deterministic fallback". The agents check `ollama` reachability at startup and warn early so the failure is loud, not silent.

### Q32. How would you scale this to 10,000 documents?

Three changes. (1) Move embeddings and the LLM tiebreak cache to a shared store (Postgres or Redis) so the dedup hit rate goes up across documents — many financial labels repeat across filings. (2) Parallelise at the document level with a queue (RQ / Celery / Temporal) — each document is independent. (3) Switch the local LLM from Ollama (per-process) to a vLLM server with continuous batching, fronted by a GPU pool — this is the single biggest throughput unlock. The agents themselves don't need to change — the LangGraph runs are stateless aside from the SQLite checkpointer, which swaps for Postgres in one config line. The bottleneck at 10k scale is OCR + LLM compute, sized by the GPU pool, not by external rate limits.

---

## I. Production readiness & extension

### Q33. CreditSource wants to add the Balance Sheet as a first-class statement. How much work?

Small. The Statement Parser is already structured around "find the table with these column headers"; I'd add a second Section Locator path for `Statement of Financial Position` and reuse the table-extraction code unchanged. The canonical mapper has the BS items (current assets, current liabilities, total debt, total equity) already declared so liquidity/leverage ratios work — I extract them today opportunistically. Making it first-class means adding a `BalanceSheet` Pydantic model parallel to `Statement`, a reconciliation check (`assets = liabilities + equity`), and one additional golden snapshot per company. Maybe a day's work.

### Q34. The credit team wants to override scoring thresholds. How?

`config/scoring.yaml` — the YAML drives everything. A credit analyst edits the thresholds and weights, re-runs `poetry run afde report <company>` (no re-parse), and the new score is emitted in 5 seconds. The sub-score breakdowns make it obvious which input drove the change. The structured JSON retains both the threshold version (`scoring_config_hash`) and the per-ratio sub-score, so historical scores remain reproducible.

### Q35. Monitoring and observability in production?

LangGraph's checkpointer is the audit log — every state transition is persisted. I'd add structured logging (per-agent latency, success/failure, retries, LLM token counts) emitting to whatever stack CreditSource uses. The metrics I'd watch: reconciliation-retry rate per document type (rising means parsing quality is drifting), narrative-regeneration rate (rising means the model is degrading), embedding cache hit rate (falling means new label vocabulary is appearing — review the canonical schema), and per-stage p99 latency.

### Q36. How do you A/B test a scoring threshold change without breaking production?

Two-track. (1) Shadow run: new thresholds compute a `score_candidate` alongside the production `score`, both stored in the structured output. The credit team compares populations before promoting. (2) Versioned scoring config — every report carries `scoring_config_hash` so historical scores remain reproducible at their config version. Promotion is a YAML PR with a sign-off; the change is data, not code.

---

## J. Implementation specifics

### Q37. Why Pydantic v2 over plain dataclasses or attrs?

Inter-agent contracts need validation, not just typing. A `LineItem` arriving at the Canonical Mapper with a malformed `note_refs` should fail fast, not silently propagate. Pydantic v2 gives me JSON serialisation, validation, and discriminated unions in one decorator. The v2 perf rewrite means it's not a hot-path concern. Dataclasses would mean reimplementing validators by hand.

### Q38. Why `Decimal` end-to-end? Where does float sneak in?

Floats lose precision on currency — `0.1 + 0.2 != 0.3` in IEEE 754. A bank's balance sheet has values that diverge from their stated totals by a cent if you parse them as floats, which the reconciliation step then catches as a false positive. `Decimal` throughout the extraction and canonical layers prevents this. Floats appear at exactly one place: the ratio layer, where division produces a `float` and that's fine because ratios are displayed at 2-3 sig figs anyway. The boundary is enforced in the type system — `LineItem.values: dict[int, Decimal]`, `Ratio.value: float`.

### Q39. What's your testing strategy?

Three layers. (1) Unit tests on the deterministic core — numeric parser, note-ref tokeniser, unit detector, reconciliation arithmetic, scoring band interpolation. These run on every commit and are fast. (2) Golden JSON snapshots per company — the full pipeline runs on each PDF and the `extraction.json` is diffed against a committed reference. Snapshot drift is a deliberate review step, not auto-accepted. (3) A stdlib-only smoke test (`scripts/smoke_test.py`) that runs without Poetry, Ollama, or weights, exercising the deterministic core against the real Citigroup PDF — useful for contributors and for CI bootstrap.

### Q40. The four PDFs are in the repo. Aren't golden snapshots fragile to model updates?

For the LLM-touched parts, yes. The canonical mapping and narrative are not snapshot-asserted — they're behaviour-asserted (`output contains a "revenue.total" key`, `narrative contains the headline score`). The snapshot-asserted parts are the deterministic extraction: page-level OCR routing, statement parsing, note resolution, reconciliation, ratios, scoring. Those are stable across model versions because they don't involve the LLM. The Qwen output is non-deterministic at temperature > 0; trying to snapshot it is the wrong test. (For mapping I pin temperature=0 and the JSON-schema decoder makes the enum-output reproducible across runs, but I still don't snapshot it because a model upgrade can flip a tiebreak legitimately.)

### Q41. CI/CD?

GitHub Actions with two jobs. (1) Lint + unit tests + the stdlib-only smoke test on every PR, Python 3.10/3.11/3.12 matrix. (2) A golden-snapshot job that runs the full pipeline on the committed PDFs inside a container that has Ollama + the Qwen weights pre-baked — this gives a deterministic regression suite with no external dependencies. The container image is rebuilt on a schedule when a new model revision is pinned; the rebuild step is the only place a network call happens in CI. Snapshot deltas surface as a PR comment for human review.

---

## K. Closing / judgement

### Q42. What's the weakest part of this design?

The reconciliation tolerance is one-size-fits-all (±0.5 × 10^units). For a small entity reporting in dollars rather than thousands, that's a 50-cent tolerance which catches almost nothing. For a bank reporting in millions, it's $500k which may miss material errors on small line items. A better design would scale tolerance to the row's value (`max(±0.5 × 10^units, 0.5% × |value|)`). I haven't implemented that. The second weakest part is the Section Locator's industry classifier — it's keyword-based on the directors' report, which works on Citigroup but is fragile to terminology drift; a small fine-tuned classifier or a Qwen-as-classifier call would be more robust.

### Q43. What would you build first if you had two more weeks?

In priority order. (1) Balance Sheet as a first-class statement (unlocks proper liquidity and leverage). (2) A shared embedding cache across documents (cuts re-run latency to zero across the corpus). (3) Replace the narrative regex check with an LLM-as-judge pass on a held-out validation set, so the hallucination detector itself is measurable. (4) A small fine-tune of TableFormer on Australian-formatted statements specifically — the public model is trained on more generic financials. (5) A web UI for the credit team that loads `extraction.json` and lets an analyst click a number to jump to its source page.

### Q44. Sell me the design in three sentences.

It uses agents for what agents are good at — typed state, conditional retries, replayable nodes — and reserves LLMs for the three places where reasoning helps: classification tiebreaks, label canonicalisation, and the analyst narrative. Numbers come from layout-aware table models, not generation; reconciliation enforces arithmetic; every value carries provenance back to a bbox on a page. Everything runs locally — Qwen2.5-7B via Ollama for reasoning, BGE-small for embeddings, Docling and PP-OCRv4 for parsing — so the entire pipeline is offline, auditable, and free to run per document.

---

*End of document.*
