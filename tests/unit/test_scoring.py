"""Unit tests for the piecewise-linear sub-score and composite band logic."""
from __future__ import annotations

from afde.agents.score import _composite_band, _piecewise_score
from afde.config import scoring_config


def _corp_bands(ratio: str):
    cfg = scoring_config()
    r = cfg["profiles"]["corporate"]["ratios"][ratio]
    return r["bands"], r.get("invert", False)


def test_net_profit_margin_strong():
    bands, invert = _corp_bands("net_profit_margin")
    score, band = _piecewise_score(0.25, bands, invert)
    assert score == 100.0
    assert band == "AA"


def test_net_profit_margin_weak():
    bands, invert = _corp_bands("net_profit_margin")
    score, _ = _piecewise_score(-0.10, bands, invert)
    assert score == 0.0


def test_debt_to_equity_invert_strong():
    bands, invert = _corp_bands("debt_to_equity")
    # Lower D/E is better; 0.2 should yield top score
    score, band = _piecewise_score(0.2, bands, invert)
    assert score == 100.0
    assert band == "AA"


def test_debt_to_equity_invert_weak():
    bands, invert = _corp_bands("debt_to_equity")
    score, _ = _piecewise_score(3.0, bands, invert)
    assert score == 0.0


def test_interest_coverage_interpolation():
    bands, invert = _corp_bands("interest_coverage")
    # Between 6 (score 75) and 12 (score 100) — value 9 should interpolate to ~87.5
    score, _ = _piecewise_score(9.0, bands, invert)
    assert 85 <= score <= 90


def test_composite_band_aa():
    assert _composite_band(88) == "AA"


def test_composite_band_d():
    assert _composite_band(10) == "D"


def test_composite_band_bbb():
    assert _composite_band(60) == "BBB"
