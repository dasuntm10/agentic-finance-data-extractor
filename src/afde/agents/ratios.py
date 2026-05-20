"""Agent 7 — Ratio Engine. Pure deterministic math over the canonical statement."""
from __future__ import annotations

import logging
from decimal import Decimal

from afde.schemas import CanonicalStatement, CompanyProfile, Ratio, Ratios

log = logging.getLogger(__name__)


def _div(a: Decimal | None, b: Decimal | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _abs_div(a: Decimal | None, b: Decimal | None) -> float | None:
    """For coverage ratios where the denominator is conventionally |interest_expense|."""
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(abs(b))


def _ratios_corporate(canon: CanonicalStatement, period: str) -> list[Ratio]:
    g = lambda k: canon.get(k, period)  # noqa: E731
    revenue = g("revenue.total")
    net_profit = g("net_profit")
    pbt = g("profit_before_tax")
    tax = g("income_tax_expense")
    interest_exp = g("expense.interest")
    cur_assets = g("balance.current_assets")
    cur_liab = g("balance.current_liabilities")
    inventory = g("balance.inventory")
    total_debt = g("balance.total_debt")
    total_equity = g("balance.total_equity")
    total_assets = g("balance.total_assets")

    # EBIT approximation: PBT + |interest_expense| if interest reported, else PBT
    ebit = None
    if pbt is not None:
        ebit = pbt + (abs(interest_exp) if interest_exp is not None else Decimal(0))

    rats = [
        Ratio(
            name="net_profit_margin",
            value=_div(net_profit, revenue),
            category="profitability",
            formula="net_profit / revenue.total",
            inputs_used={"net_profit": _f(net_profit), "revenue.total": _f(revenue)},
        ),
        Ratio(
            name="ebit_margin",
            value=_div(ebit, revenue),
            category="profitability",
            formula="(profit_before_tax + |interest_expense|) / revenue.total",
            inputs_used={"ebit": _f(ebit), "revenue.total": _f(revenue)},
        ),
        Ratio(
            name="current_ratio",
            value=_div(cur_assets, cur_liab),
            category="liquidity",
            formula="current_assets / current_liabilities",
            inputs_used={"current_assets": _f(cur_assets), "current_liabilities": _f(cur_liab)},
        ),
        Ratio(
            name="quick_ratio",
            value=_div(
                (cur_assets - inventory) if (cur_assets is not None and inventory is not None) else cur_assets,
                cur_liab,
            ),
            category="liquidity",
            formula="(current_assets - inventory) / current_liabilities",
            inputs_used={
                "current_assets": _f(cur_assets),
                "inventory": _f(inventory),
                "current_liabilities": _f(cur_liab),
            },
        ),
        Ratio(
            name="debt_to_equity",
            value=_div(total_debt, total_equity),
            category="leverage",
            formula="total_debt / total_equity",
            inputs_used={"total_debt": _f(total_debt), "total_equity": _f(total_equity)},
        ),
        Ratio(
            name="interest_coverage",
            value=_abs_div(ebit, interest_exp),
            category="coverage",
            formula="EBIT / |interest_expense|",
            inputs_used={"ebit": _f(ebit), "interest_expense": _f(interest_exp)},
        ),
    ]
    return rats


def _ratios_financial(canon: CanonicalStatement, period: str) -> list[Ratio]:
    g = lambda k: canon.get(k, period)  # noqa: E731
    revenue = g("revenue.total")
    net_profit = g("net_profit")
    interest_income = g("revenue.interest_income")
    interest_expense = g("expense.interest")
    employee = g("expense.employee")
    other_opex = g("expense.operating_other")
    mgmt_fee = g("expense.management_fees")
    equity = g("balance.total_equity")
    assets = g("balance.total_assets")

    # Cost-to-income = total operating expenses / total revenue
    opex_parts = [x for x in (employee, other_opex, mgmt_fee) if x is not None]
    opex = sum(opex_parts, Decimal(0)) if opex_parts else None
    return [
        Ratio(
            name="net_profit_margin",
            value=_div(net_profit, revenue),
            category="profitability",
            formula="net_profit / revenue.total",
            inputs_used={"net_profit": _f(net_profit), "revenue.total": _f(revenue)},
        ),
        Ratio(
            name="cost_to_income",
            value=_div(opex, revenue),
            category="profitability",
            formula="(employee + other_opex + management_fees) / revenue.total",
            inputs_used={"opex": _f(opex), "revenue.total": _f(revenue)},
        ),
        Ratio(
            name="net_interest_margin",
            value=_div(
                (interest_income + interest_expense)
                if (interest_income is not None and interest_expense is not None)
                else None,
                assets,
            ),
            category="profitability",
            formula="(interest_income - interest_expense) / total_assets",
            inputs_used={
                "interest_income": _f(interest_income),
                "interest_expense": _f(interest_expense),
                "total_assets": _f(assets),
            },
        ),
        Ratio(
            name="equity_to_assets",
            value=_div(equity, assets),
            category="capital",
            formula="total_equity / total_assets",
            inputs_used={"total_equity": _f(equity), "total_assets": _f(assets)},
        ),
        Ratio(
            name="return_on_equity",
            value=_div(net_profit, equity),
            category="asset_quality",
            formula="net_profit / total_equity",
            inputs_used={"net_profit": _f(net_profit), "total_equity": _f(equity)},
        ),
    ]


def _f(d: Decimal | None) -> float | None:
    return None if d is None else float(d)


def run(canon: CanonicalStatement) -> Ratios:
    period = canon.period_label
    if canon.profile == CompanyProfile.FINANCIAL:
        current = _ratios_financial(canon, period)
    else:
        current = _ratios_corporate(canon, period)

    # YoY delta if comparative present
    comp = canon.comparative_period_label
    if comp:
        prior = (
            _ratios_financial(canon, comp)
            if canon.profile == CompanyProfile.FINANCIAL
            else _ratios_corporate(canon, comp)
        )
        prior_by_name = {r.name: r for r in prior}
        for r in current:
            p = prior_by_name.get(r.name)
            if r.value is not None and p and p.value is not None:
                r.yoy_delta = r.value - p.value

    return Ratios(
        profile=canon.profile,
        period_label=period,
        comparative_period_label=comp,
        ratios=current,
    )
