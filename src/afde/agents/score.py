"""Agent 8 — Scoring Agent.

Hybrid rules + statistical: per-ratio sub-scores from piecewise-linear band maps,
composite from configured weights. Anomaly detectors are independent of the
composite and emit flags (yoy_deterioration, note_disclosure, off_balance_sheet).
"""
from __future__ import annotations

import logging
import re

from afde.config import scoring_config
from afde.schemas import (
    AnomalyFlag,
    CanonicalStatement,
    EnrichedStatement,
    Ratios,
    Score,
    SubScore,
)

log = logging.getLogger(__name__)


def _piecewise_score(value: float | None, bands: list[dict], invert: bool) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    # bands are listed in descending sub-score order (highest score first)
    # For invert=True, lower value is better, so we compare differently.
    sorted_bands = sorted(bands, key=lambda b: b["score"], reverse=True)
    if invert:
        # value <= bands[0].value -> bands[0].score (best)
        for i, b in enumerate(sorted_bands):
            if value <= b["value"]:
                if i == 0:
                    return float(b["score"]), b["band"]
                prev = sorted_bands[i - 1]
                # interpolate between prev (worse) and b (better) using value
                if prev["value"] == b["value"]:
                    return float(b["score"]), b["band"]
                frac = (prev["value"] - value) / (prev["value"] - b["value"])
                interp = prev["score"] + frac * (b["score"] - prev["score"])
                return float(max(0, min(100, interp))), b["band"]
        # worse than worst threshold
        return float(sorted_bands[-1]["score"]), sorted_bands[-1]["band"]
    else:
        for i, b in enumerate(sorted_bands):
            if value >= b["value"]:
                if i == 0:
                    return float(b["score"]), b["band"]
                prev = sorted_bands[i - 1]
                if prev["value"] == b["value"]:
                    return float(b["score"]), b["band"]
                frac = (value - b["value"]) / (prev["value"] - b["value"])
                interp = b["score"] + frac * (prev["score"] - b["score"])
                return float(max(0, min(100, interp))), b["band"]
        return float(sorted_bands[-1]["score"]), sorted_bands[-1]["band"]


def _composite_band(score: float) -> str:
    cfg = scoring_config()
    for band in sorted(cfg["composite_bands"], key=lambda b: b["min"], reverse=True):
        if score >= band["min"]:
            return band["band"]
    return "D"


def _detect_anomalies(
    ratios: Ratios,
    sub_scores: list[SubScore],
    enriched_pl: EnrichedStatement,
    canon: CanonicalStatement,
) -> list[AnomalyFlag]:
    flags: list[AnomalyFlag] = []
    cfg = scoring_config()
    drop_threshold = float(cfg.get("yoy_deterioration_sub_score_drop", 25))

    # 1) YoY deterioration: any ratio whose value crossed a band boundary the wrong way
    by_name = {s.ratio: s for s in sub_scores}
    for r in ratios.ratios:
        if r.yoy_delta is None or r.value is None:
            continue
        prior_value = r.value - r.yoy_delta
        prior_score, _ = _piecewise_score(
            prior_value,
            cfg["profiles"][canon.profile.value]["ratios"].get(r.name, {}).get("bands", []),
            cfg["profiles"][canon.profile.value]["ratios"].get(r.name, {}).get("invert", False),
        )
        cur_score = by_name[r.name].sub_score if r.name in by_name else None
        if prior_score is not None and cur_score is not None:
            drop = prior_score - cur_score
            if drop >= drop_threshold:
                flags.append(
                    AnomalyFlag(
                        kind="yoy_deterioration",
                        severity="warning" if drop < 40 else "critical",
                        description=(
                            f"{r.name} sub-score fell {drop:.0f} pts YoY "
                            f"({prior_score:.0f} → {cur_score:.0f}); raw value {prior_value:.3f} → {r.value:.3f}"
                        ),
                    )
                )

    # 2) Note-disclosure keywords on the resolved notes attached to the P&L
    keywords = [k.lower() for k in cfg.get("anomaly_keywords", [])]
    for note in enriched_pl.notes:
        text_l = (note.text or "").lower()
        for kw in keywords:
            if kw in text_l:
                # Capture surrounding sentence
                idx = text_l.find(kw)
                start = max(0, idx - 80)
                end = min(len(text_l), idx + 160)
                snippet = (note.text or "")[start:end].strip().replace("\n", " ")
                flags.append(
                    AnomalyFlag(
                        kind="note_disclosure",
                        severity="warning",
                        description=f"Note {note.note_number}{f'({note.sub})' if note.sub else ''}: keyword '{kw}'",
                        evidence=snippet[:300],
                    )
                )

    # 3) Off-balance-sheet exposure (heuristic): unmatched canonical lines whose
    #    label mentions guarantees / contingent liabilities / commitments.
    for li in enriched_pl.statement.line_items:
        if re.search(r"contingent|guarantee|commitment|off.balance", li.label, re.I):
            v = max(li.values.values(), default=None) if li.values else None
            flags.append(
                AnomalyFlag(
                    kind="off_balance_sheet",
                    severity="info",
                    description=f"{li.label}: {v}",
                )
            )

    return flags


def run(
    canon: CanonicalStatement,
    ratios: Ratios,
    enriched_pl: EnrichedStatement,
) -> Score:
    cfg = scoring_config()
    profile_cfg = cfg["profiles"][canon.profile.value]
    weights = profile_cfg["weights"]
    ratio_cfg = profile_cfg["ratios"]

    sub_scores: list[SubScore] = []
    category_aggregates: dict[str, list[float]] = {}

    for r in ratios.ratios:
        rconf = ratio_cfg.get(r.name)
        if not rconf:
            continue
        ss, band = _piecewise_score(r.value, rconf["bands"], rconf.get("invert", False))
        sub_scores.append(
            SubScore(
                ratio=r.name,
                value=r.value,
                sub_score=ss,
                band=band,
                category=rconf["category"],
            )
        )
        if ss is not None:
            category_aggregates.setdefault(rconf["category"], []).append(ss)

    composite: float | None = None
    if category_aggregates:
        weighted_sum = 0.0
        used_weight = 0.0
        for cat, vals in category_aggregates.items():
            w = float(weights.get(cat, 0))
            if w == 0:
                continue
            cat_avg = sum(vals) / len(vals)
            weighted_sum += w * cat_avg
            used_weight += w
        composite = weighted_sum / used_weight if used_weight > 0 else None

    band = _composite_band(composite) if composite is not None else None
    anomalies = _detect_anomalies(ratios, sub_scores, enriched_pl, canon)

    log.info(
        "Score: composite=%s band=%s anomalies=%d",
        f"{composite:.1f}" if composite is not None else "n/a",
        band,
        len(anomalies),
    )

    return Score(
        company_name=canon.company_name,
        profile=canon.profile,
        composite=composite,
        band=band,
        sub_scores=sub_scores,
        weights_used=weights,
        anomalies=anomalies,
    )
