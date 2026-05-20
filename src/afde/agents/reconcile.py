"""Agent 6 — Reconciliation / Self-check.

Deterministic checks that verify the extraction is arithmetically consistent
before any ratio is computed. Failure produces a ValidationReport that the
orchestrator uses to decide whether to retry parsing or continue with a
data-quality flag.
"""
from __future__ import annotations

import logging
from decimal import Decimal

from afde.schemas import (
    CanonicalStatement,
    EnrichedStatement,
    ValidationCheck,
    ValidationReport,
)

log = logging.getLogger(__name__)


def _tolerance(stmt_units_scale: int) -> Decimal:
    # ±0.5 of the original unit — so ±$500k for a $Million statement
    return Decimal(stmt_units_scale) * Decimal("0.5")


def _get(canon: CanonicalStatement, key: str) -> Decimal | None:
    return canon.get(key)


def run(
    enriched_pl: EnrichedStatement,
    canon: CanonicalStatement,
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    stmt = enriched_pl.statement
    tol = _tolerance(stmt.units_scale)

    pbt = _get(canon, "profit_before_tax")
    tax = _get(canon, "income_tax_expense")
    np_ = _get(canon, "net_profit")
    if pbt is not None and tax is not None and np_ is not None:
        # In Australian statements tax is typically a *deduction* so pbt - tax == net_profit
        # but signs vary across filers. Try both interpretations within tolerance.
        diff1 = abs((pbt - tax) - np_)
        diff2 = abs((pbt + tax) - np_)
        passed = min(diff1, diff2) <= tol
        checks.append(
            ValidationCheck(
                name="pbt_tax_np_consistency",
                passed=passed,
                expected=np_,
                actual=pbt - tax if diff1 <= diff2 else pbt + tax,
                tolerance=tol,
                detail="profit_before_tax ± income_tax_expense ≈ net_profit",
            )
        )

    np_again = _get(canon, "net_profit")
    oci = _get(canon, "other_comprehensive_income")
    tci = _get(canon, "total_comprehensive_income")
    if np_again is not None and tci is not None:
        oci_v = oci if oci is not None else Decimal(0)
        diff = abs((np_again + oci_v) - tci)
        checks.append(
            ValidationCheck(
                name="oci_rollup",
                passed=diff <= tol,
                expected=tci,
                actual=np_again + oci_v,
                tolerance=tol,
                detail="net_profit + OCI ≈ total_comprehensive_income",
            )
        )

    # Note-linkage: every note ref in the enriched statement resolved to something
    note_keys_resolved = {n.note_number for n in enriched_pl.notes}
    dangling = []
    for li in stmt.line_items:
        for ref in li.note_refs:
            if ref.note_number not in note_keys_resolved:
                dangling.append(f"{li.label} -> {ref.raw}")
    checks.append(
        ValidationCheck(
            name="note_linkage",
            passed=len(dangling) == 0,
            detail=("All note refs resolved." if not dangling else "Dangling: " + "; ".join(dangling[:5])),
        )
    )

    # Cross-year completeness
    missing_latest = [
        li.label
        for li in stmt.line_items
        if stmt.period_label not in li.values
    ]
    checks.append(
        ValidationCheck(
            name="latest_year_completeness",
            passed=len(missing_latest) == 0,
            detail=("All lines have latest year." if not missing_latest else f"Missing on {len(missing_latest)} lines"),
        )
    )

    overall = all(c.passed for c in checks)
    retry_hint = None
    if not overall:
        failed_names = [c.name for c in checks if not c.passed]
        retry_hint = "Failed checks: " + ", ".join(failed_names)
        log.warning("Reconcile: %s", retry_hint)
    else:
        log.info("Reconcile: all %d checks passed", len(checks))

    return ValidationReport(checks=checks, overall_passed=overall, retry_hint=retry_hint)
