"""Pydantic models that form the typed contracts between agents.

Every value extracted from a PDF carries provenance (`page`, `bbox`, `source`) so a
human auditor can re-open the source page and verify the number. Decimal is used
throughout for money; floats only appear at the ratio layer.
"""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CompanyProfile(str, Enum):
    CORPORATE = "corporate"
    FINANCIAL = "financial"


# ---------------------------------------------------------------------------
# Ingestion layer
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class PageBlock(BaseModel):
    """A single page after triage. Either native-text or OCR'd."""

    page: int
    source: Literal["native", "ocr", "font_decoded"] = "native"
    text: str
    char_count: int
    ocr_confidence: float | None = None
    needs_ocr: bool = False


class TableBlock(BaseModel):
    """A logical table parsed from a page."""

    table_id: str
    page: int
    bbox: BBox | None = None
    header: list[str]
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None


class Heading(BaseModel):
    page: int
    text: str
    level: int = 1


class IngestedDocument(BaseModel):
    source_path: str
    company_name: str | None = None
    page_count: int
    pages: list[PageBlock]
    tables: list[TableBlock] = Field(default_factory=list)
    headings: list[Heading] = Field(default_factory=list)
    quality: DocumentQuality = DocumentQuality.HIGH


# ---------------------------------------------------------------------------
# Section location
# ---------------------------------------------------------------------------


class SectionMap(BaseModel):
    pl_pages: list[int]
    notes_pages: list[int]
    balance_sheet_pages: list[int] = Field(default_factory=list)
    cashflow_pages: list[int] = Field(default_factory=list)
    pl_title: str
    company_profile: CompanyProfile = CompanyProfile.CORPORATE


# ---------------------------------------------------------------------------
# Statement extraction
# ---------------------------------------------------------------------------


class NoteRef(BaseModel):
    """A note reference token e.g. `3(a)`, `26`, or `3, 26` (multiple)."""

    note_number: int
    sub: str | None = None
    raw: str

    def key(self) -> str:
        return f"{self.note_number}{f'({self.sub})' if self.sub else ''}"


DecimalStr = Annotated[Decimal, Field(coerce_numbers_to_str=False)]


class LineItem(BaseModel):
    """A single row of a statement."""

    label: str
    note_refs: list[NoteRef] = Field(default_factory=list)
    values: dict[str, Decimal] = Field(default_factory=dict)  # year_label -> value
    is_subtotal: bool = False
    is_total: bool = False
    indent: int = 0
    page: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Statement(BaseModel):
    """A primary financial statement (P&L, BS, or CF)."""

    kind: Literal["profit_or_loss", "balance_sheet", "cash_flow"]
    title: str
    currency: str = "AUD"
    units_scale: int = 1  # 1, 1_000, 1_000_000 — multiply Decimal by this to get base units
    units_label: str = "$"  # "$Million", "$'000", etc.
    period_end: str  # e.g. "2024-12-31"
    period_label: str  # e.g. "FY2024"
    comparative_period_end: str | None = None
    comparative_period_label: str | None = None
    line_items: list[LineItem]
    pages: list[int]


# ---------------------------------------------------------------------------
# Note resolution
# ---------------------------------------------------------------------------


class Note(BaseModel):
    note_number: int
    sub: str | None = None  # 'a', 'b', …
    title: str
    text: str
    tables: list[TableBlock] = Field(default_factory=list)
    page: int


class EnrichedStatement(BaseModel):
    statement: Statement
    notes: list[Note] = Field(default_factory=list)
    line_item_to_notes: dict[str, list[str]] = Field(default_factory=dict)
    # line item label -> list of note keys ('3(a)', '26')


# ---------------------------------------------------------------------------
# Canonical mapping
# ---------------------------------------------------------------------------


class CanonicalLine(BaseModel):
    canonical: str  # e.g. "revenue.total"
    original_label: str
    values: dict[str, Decimal]
    note_refs: list[str] = Field(default_factory=list)
    match_method: Literal["rule", "embedding", "llm", "unmatched"] = "rule"
    match_score: float | None = None


class CanonicalStatement(BaseModel):
    company_name: str
    profile: CompanyProfile
    period_end: str
    period_label: str
    comparative_period_label: str | None = None
    currency: str = "AUD"
    base_unit: int = 1  # all values normalised to base currency units
    lines: list[CanonicalLine]
    unmatched_labels: list[str] = Field(default_factory=list)

    def get(self, canonical: str, period: str | None = None) -> Decimal | None:
        period = period or self.period_label
        for line in self.lines:
            if line.canonical == canonical:
                return line.values.get(period)
        return None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    expected: Decimal | None = None
    actual: Decimal | None = None
    tolerance: Decimal | None = None
    detail: str | None = None
    line_label: str | None = None


class ValidationReport(BaseModel):
    checks: list[ValidationCheck]
    overall_passed: bool
    retry_hint: str | None = None


# ---------------------------------------------------------------------------
# Ratios + scoring
# ---------------------------------------------------------------------------


class Ratio(BaseModel):
    name: str
    value: float | None  # null when inputs unavailable
    yoy_delta: float | None = None
    category: Literal["profitability", "liquidity", "leverage", "coverage", "capital", "asset_quality"]
    formula: str
    inputs_used: dict[str, float | None] = Field(default_factory=dict)


class Ratios(BaseModel):
    profile: CompanyProfile
    period_label: str
    comparative_period_label: str | None = None
    ratios: list[Ratio]


class SubScore(BaseModel):
    ratio: str
    value: float | None
    sub_score: float | None  # 0-100
    band: str | None  # AA, A, BBB, …
    category: str


class AnomalyFlag(BaseModel):
    kind: Literal["yoy_deterioration", "note_disclosure", "off_balance_sheet"]
    severity: Literal["info", "warning", "critical"]
    description: str
    evidence: str | None = None


class Score(BaseModel):
    company_name: str
    profile: CompanyProfile
    composite: float | None  # 0-100, higher = healthier
    band: str | None
    sub_scores: list[SubScore]
    weights_used: dict[str, float]
    anomalies: list[AnomalyFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final artefact
# ---------------------------------------------------------------------------


class RiskReport(BaseModel):
    company_name: str
    period_label: str
    score: Score
    ratios: Ratios
    canonical: CanonicalStatement
    enriched_statement: EnrichedStatement
    validation: ValidationReport
    quality: DocumentQuality
    narrative_markdown: str | None = None
