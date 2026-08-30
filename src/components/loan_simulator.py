"""Transparent loan affordability, resilience, and repayment simulation.

Income and cash flow determine repayment capacity. Assets describe resilience
and down-payment strength, but never increase the affordable EMI. Historical ML
is deliberately implemented elsewhere as a separate comparison signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.components.feature_contract import generate_application_id


ASSET_LIQUIDITY_WEIGHTS: dict[str, float] = {
    "cash": 1.00,
    "fixed_deposit": 0.95,
    "debt_fund": 0.85,
    "equity": 0.60,
    "gold": 0.70,
    "property_equity": 0.25,
    "vehicle_equity": 0.10,
    "epf_ppf": 0.15,
    "nps": 0.00,
}


@dataclass(frozen=True)
class LoanProductPolicy:
    code: str
    label: str
    annual_interest_rate: float
    permitted_terms_months: tuple[int, ...]
    maximum_debt_service_ratio: float
    minimum_residual_income_ratio: float
    minimum_income_stability_years: float
    maximum_age_at_maturity: int
    maximum_principal: float
    stress_rate_increase: float
    recommended_ltv: float | None = None


PRODUCT_POLICIES: dict[str, LoanProductPolicy] = {
    "personal_loan": LoanProductPolicy(
        code="personal_loan",
        label="Personal Loan",
        annual_interest_rate=0.1275,
        permitted_terms_months=(12, 24, 36, 48, 60),
        maximum_debt_service_ratio=0.40,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=1.0,
        maximum_age_at_maturity=65,
        maximum_principal=2_000_000.0,
        stress_rate_increase=0.03,
    ),
    "vehicle_loan": LoanProductPolicy(
        code="vehicle_loan",
        label="Vehicle Loan",
        annual_interest_rate=0.085,
        permitted_terms_months=(12, 24, 36, 48, 60, 72, 84),
        maximum_debt_service_ratio=0.45,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=1.0,
        maximum_age_at_maturity=65,
        maximum_principal=5_000_000.0,
        stress_rate_increase=0.02,
        recommended_ltv=0.85,
    ),
    "home_loan": LoanProductPolicy(
        code="home_loan",
        label="Home Loan",
        annual_interest_rate=0.0775,
        permitted_terms_months=(60, 120, 180, 240, 300, 360),
        maximum_debt_service_ratio=0.50,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=2.0,
        maximum_age_at_maturity=70,
        maximum_principal=20_000_000.0,
        stress_rate_increase=0.02,
        recommended_ltv=0.80,
    ),
}

EMPLOYMENT_TYPES = (
    "Salaried",
    "Self-employed",
    "Business owner",
    "Retired / pension income",
    "Other regular income",
)


@dataclass(frozen=True)
class AssetProfile:
    cash: float = 0.0
    fixed_deposit: float = 0.0
    debt_fund: float = 0.0
    equity: float = 0.0
    gold: float = 0.0
    property_value: float = 0.0
    property_liability: float = 0.0
    vehicle_value: float = 0.0
    vehicle_liability: float = 0.0
    epf_ppf: float = 0.0
    nps: float = 0.0


@dataclass(frozen=True)
class LoanApplication:
    product_type: str
    age: int
    employment_type: str
    income_stability_years: float
    monthly_net_income: float
    monthly_essential_expenses: float
    existing_monthly_debt_payments: float
    requested_loan_amount: float
    preferred_term_months: int
    assets: AssetProfile = field(default_factory=AssetProfile)
    purchase_property_value: float = 0.0
    vehicle_purchase_price: float = 0.0
    available_down_payment: float = 0.0


def calculate_emi(principal: float, annual_interest_rate: float, term_months: int) -> float:
    """Return the level monthly instalment for a standard amortizing loan."""

    if principal < 0:
        raise ValueError("principal must be non-negative")
    if annual_interest_rate < 0:
        raise ValueError("annual interest rate must be non-negative")
    if term_months <= 0:
        raise ValueError("term months must be positive")
    if principal == 0:
        return 0.0
    monthly_rate = annual_interest_rate / 12
    if monthly_rate == 0:
        return principal / term_months
    growth = (1 + monthly_rate) ** term_months
    return principal * monthly_rate * growth / (growth - 1)


def principal_from_emi(emi: float, annual_interest_rate: float, term_months: int) -> float:
    """Return the principal supported by a monthly instalment capacity."""

    if emi < 0:
        raise ValueError("EMI capacity must be non-negative")
    if annual_interest_rate < 0:
        raise ValueError("annual interest rate must be non-negative")
    if term_months <= 0:
        raise ValueError("term months must be positive")
    if emi == 0:
        return 0.0
    monthly_rate = annual_interest_rate / 12
    if monthly_rate == 0:
        return emi * term_months
    growth = (1 + monthly_rate) ** term_months
    return emi * (growth - 1) / (monthly_rate * growth)


def calculate_total_interest(
    principal: float,
    annual_interest_rate: float,
    term_months: int,
) -> float:
    return max(calculate_emi(principal, annual_interest_rate, term_months) * term_months - principal, 0.0)


def calculate_loan_plan_fit(max_affordable_emi: float, requested_emi: float) -> float:
    """Return the share of requested EMI supported by current cash flow."""

    if max_affordable_emi < 0:
        raise ValueError("maximum affordable EMI cannot be negative")
    if requested_emi < 0:
        raise ValueError("requested EMI cannot be negative")
    if requested_emi == 0:
        return 100.0
    return min(max_affordable_emi / requested_emi, 1.0) * 100


def calculate_asset_equity(market_value: float, liability: float) -> float:
    if market_value < 0 or liability < 0:
        raise ValueError("asset values and liabilities cannot be negative")
    return max(market_value - liability, 0.0)


def calculate_adjusted_liquidity(
    assets: AssetProfile,
    *,
    weights: dict[str, float] = ASSET_LIQUIDITY_WEIGHTS,
) -> float:
    property_equity = calculate_asset_equity(
        assets.property_value,
        assets.property_liability,
    )
    vehicle_equity = calculate_asset_equity(
        assets.vehicle_value,
        assets.vehicle_liability,
    )
    return (
        assets.cash * weights["cash"]
        + assets.fixed_deposit * weights["fixed_deposit"]
        + assets.debt_fund * weights["debt_fund"]
        + assets.equity * weights["equity"]
        + assets.gold * weights["gold"]
        + property_equity * weights["property_equity"]
        + vehicle_equity * weights["vehicle_equity"]
        + assets.epf_ppf * weights["epf_ppf"]
        + assets.nps * weights["nps"]
    )


def calculate_emergency_coverage(
    adjusted_liquidity: float,
    monthly_commitments: float,
) -> float | None:
    if adjusted_liquidity < 0:
        raise ValueError("adjusted liquidity cannot be negative")
    if monthly_commitments < 0:
        raise ValueError("monthly commitments cannot be negative")
    if monthly_commitments == 0:
        return None
    return adjusted_liquidity / monthly_commitments


def _coverage_band(months: float | None) -> str:
    if months is None:
        return "NOT_APPLICABLE"
    if months < 1:
        return "CRITICAL"
    if months < 3:
        return "WEAK"
    if months < 6:
        return "MODERATE"
    if months < 12:
        return "STRONG"
    return "VERY_STRONG"


def calculate_asset_resilience(
    assets: AssetProfile,
    monthly_post_loan_commitments: float,
) -> dict[str, Any]:
    values = asdict(assets)
    if any(value < 0 for value in values.values()):
        raise ValueError("asset values and liabilities cannot be negative")
    property_equity = calculate_asset_equity(
        assets.property_value,
        assets.property_liability,
    )
    vehicle_equity = calculate_asset_equity(
        assets.vehicle_value,
        assets.vehicle_liability,
    )
    total_assets = (
        assets.cash
        + assets.fixed_deposit
        + assets.debt_fund
        + assets.equity
        + assets.gold
        + assets.property_value
        + assets.vehicle_value
        + assets.epf_ppf
        + assets.nps
    )
    total_liabilities = assets.property_liability + assets.vehicle_liability
    net_equity = max(total_assets - total_liabilities, 0.0)
    adjusted_liquidity = calculate_adjusted_liquidity(assets)
    coverage = calculate_emergency_coverage(
        adjusted_liquidity,
        monthly_post_loan_commitments,
    )
    return {
        "total_assets": round(total_assets, 2),
        "total_asset_liabilities": round(total_liabilities, 2),
        "net_asset_equity": round(net_equity, 2),
        "property_equity": round(property_equity, 2),
        "vehicle_equity": round(vehicle_equity, 2),
        "adjusted_emergency_liquidity": round(adjusted_liquidity, 2),
        "monthly_post_loan_commitments": round(monthly_post_loan_commitments, 2),
        "emergency_coverage_months": round(coverage, 2) if coverage is not None else None,
        "emergency_coverage_band": _coverage_band(coverage),
        "liquidity_weights": dict(ASSET_LIQUIDITY_WEIGHTS),
    }

def validate_loan_application(application: LoanApplication) -> LoanProductPolicy:
    if application.product_type not in PRODUCT_POLICIES:
        raise ValueError(f"Unsupported product type: {application.product_type}")
    policy = PRODUCT_POLICIES[application.product_type]
    if not 18 <= application.age <= 80:
        raise ValueError("Age must be between 18 and 80")
    if application.employment_type not in EMPLOYMENT_TYPES:
        raise ValueError(f"Unsupported employment type: {application.employment_type}")
    if application.income_stability_years < 0:
        raise ValueError("Income stability years cannot be negative")
    if application.monthly_net_income <= 0:
        raise ValueError("Monthly net income must be positive")
    if application.monthly_essential_expenses < 0:
        raise ValueError("Monthly essential expenses cannot be negative")
    if application.existing_monthly_debt_payments < 0:
        raise ValueError("Existing monthly debt payments cannot be negative")
    if application.requested_loan_amount <= 0:
        raise ValueError("Requested loan amount must be positive")
    if application.preferred_term_months not in policy.permitted_terms_months:
        raise ValueError(
            f"Preferred term must be one of {list(policy.permitted_terms_months)} months"
        )
    if any(value < 0 for value in asdict(application.assets).values()):
        raise ValueError("Asset values and liabilities cannot be negative")
    if application.purchase_property_value < 0:
        raise ValueError("Purchase property value cannot be negative")
    if application.vehicle_purchase_price < 0:
        raise ValueError("Vehicle purchase price cannot be negative")
    if application.available_down_payment < 0:
        raise ValueError("Available down payment cannot be negative")
    return policy


def _eligible_terms(application: LoanApplication, policy: LoanProductPolicy) -> tuple[int, ...]:
    return tuple(
        term
        for term in policy.permitted_terms_months
        if application.age + term / 12 <= policy.maximum_age_at_maturity
    )


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def calculate_affordable_emi(
    application: LoanApplication,
    policy: LoanProductPolicy,
) -> dict[str, float]:
    income = application.monthly_net_income
    existing_debt = application.existing_monthly_debt_payments
    required_buffer = income * policy.minimum_residual_income_ratio
    debt_service_capacity = max(
        income * policy.maximum_debt_service_ratio - existing_debt,
        0.0,
    )
    cash_flow_capacity = max(
        income
        - application.monthly_essential_expenses
        - existing_debt
        - required_buffer,
        0.0,
    )
    return {
        "debt_service_capacity": debt_service_capacity,
        "cash_flow_capacity": cash_flow_capacity,
        "required_residual_buffer": required_buffer,
        "max_affordable_emi": max(min(debt_service_capacity, cash_flow_capacity), 0.0),
    }


def calculate_financial_foundation(
    application: LoanApplication,
    policy: LoanProductPolicy,
    adjusted_liquidity: float,
) -> dict[str, Any]:
    income = application.monthly_net_income
    expenses = application.monthly_essential_expenses
    debt = application.existing_monthly_debt_payments
    pre_loan_surplus = income - expenses - debt
    pre_loan_surplus_ratio = max(pre_loan_surplus, 0.0) / income
    existing_debt_ratio = debt / income
    expense_ratio = expenses / income
    pre_loan_commitments = expenses + debt
    pre_loan_coverage = calculate_emergency_coverage(
        adjusted_liquidity,
        pre_loan_commitments,
    )

    cash_flow = 30 * _bounded_score(pre_loan_surplus_ratio / 0.40)
    existing_debt = 20 * _bounded_score(
        (policy.maximum_debt_service_ratio - existing_debt_ratio)
        / policy.maximum_debt_service_ratio
    )
    expense_burden = 15 * _bounded_score((0.80 - expense_ratio) / 0.50)
    liquidity = 25 * _bounded_score((pre_loan_coverage or 0.0) / 6.0)
    stability = 10 * _bounded_score(
        application.income_stability_years
        / max(policy.minimum_income_stability_years * 2, 1)
    )
    breakdown = {
        "Pre-loan cash-flow strength": round(cash_flow, 2),
        "Existing debt burden": round(existing_debt, 2),
        "Essential-expense burden": round(expense_burden, 2),
        "Emergency liquidity": round(liquidity, 2),
        "Income stability": round(stability, 2),
    }
    score = round(sum(breakdown.values()))
    if score >= 80:
        band = "STRONG"
    elif score >= 65:
        band = "STABLE"
    elif score >= 45:
        band = "LIMITED"
    else:
        band = "WEAK"
    return {
        "score": score,
        "band": band,
        "breakdown": breakdown,
        "pre_loan_monthly_surplus": round(pre_loan_surplus, 2),
        "pre_loan_emergency_coverage_months": (
            round(pre_loan_coverage, 2) if pre_loan_coverage is not None else None
        ),
        "methodology": (
            "30% pre-loan cash flow, 20% existing debt, 15% expenses, "
            "25% emergency liquidity, and 10% income stability."
        ),
    }


def calculate_stress_scenario(
    application: LoanApplication,
    policy: LoanProductPolicy,
    max_affordable_emi: float,
    required_residual_buffer: float,
) -> dict[str, Any]:
    stressed_rate = policy.annual_interest_rate + policy.stress_rate_increase
    stressed_emi = calculate_emi(
        application.requested_loan_amount,
        stressed_rate,
        application.preferred_term_months,
    )
    stressed_debt_ratio = (
        application.existing_monthly_debt_payments + stressed_emi
    ) / application.monthly_net_income
    stressed_cash_remaining = (
        application.monthly_net_income
        - application.monthly_essential_expenses
        - application.existing_monthly_debt_payments
        - stressed_emi
    )
    stressed_after_buffer = stressed_cash_remaining - required_residual_buffer
    base_emi = calculate_emi(
        application.requested_loan_amount,
        policy.annual_interest_rate,
        application.preferred_term_months,
    )
    if stressed_emi <= max_affordable_emi:
        resilience = "RESILIENT"
    elif base_emi <= max_affordable_emi:
        resilience = "MODERATE"
    else:
        resilience = "VULNERABLE"
    return {
        "base_annual_interest_rate_percent": round(policy.annual_interest_rate * 100, 3),
        "stressed_annual_interest_rate_percent": round(stressed_rate * 100, 3),
        "rate_increase_percentage_points": round(policy.stress_rate_increase * 100, 2),
        "stressed_emi": round(stressed_emi, 2),
        "stressed_total_debt_service_ratio": round(stressed_debt_ratio, 4),
        "stressed_cash_remaining": round(stressed_cash_remaining, 2),
        "stressed_cash_remaining_after_buffer": round(stressed_after_buffer, 2),
        "resilience": resilience,
    }


def _salary_multiple_band(multiple: float) -> str:
    if multiple <= 6:
        return "RELATIVELY_CONSERVATIVE"
    if multiple <= 8:
        return "MODERATE"
    if multiple <= 10:
        return "STRETCHED"
    return "VERY_HIGH"


def _product_guidance(
    application: LoanApplication,
    policy: LoanProductPolicy,
    max_principal_selected_term: float,
) -> dict[str, Any]:
    guidance: dict[str, Any] = {
        "guidance_type": "NONE",
        "requested_ltv": None,
        "recommended_ltv": policy.recommended_ltv,
        "ltv_guided_borrowing_amount": None,
        "recommended_minimum_down_payment": None,
        "available_down_payment": round(application.available_down_payment, 2),
        "additional_down_payment_needed": None,
        "product_guided_max_principal": round(max_principal_selected_term, 2),
        "loan_to_monthly_income_multiple": None,
        "salary_multiple_band": None,
    }
    purchase_value = 0.0
    if application.product_type == "home_loan":
        purchase_value = application.purchase_property_value
        guidance["guidance_type"] = "HOME_LTV"
    elif application.product_type == "vehicle_loan":
        purchase_value = application.vehicle_purchase_price
        guidance["guidance_type"] = "VEHICLE_LTV"
    elif application.product_type == "personal_loan":
        multiple = application.requested_loan_amount / application.monthly_net_income
        guidance.update(
            {
                "guidance_type": "PERSONAL_LOAN_SALARY_MULTIPLE",
                "loan_to_monthly_income_multiple": round(multiple, 2),
                "salary_multiple_band": _salary_multiple_band(multiple),
            }
        )
        return guidance

    if purchase_value > 0 and policy.recommended_ltv is not None:
        requested_ltv = application.requested_loan_amount / purchase_value
        ltv_limit = purchase_value * policy.recommended_ltv
        minimum_down_payment = purchase_value * (1 - policy.recommended_ltv)
        guidance.update(
            {
                "requested_ltv": round(requested_ltv, 4),
                "ltv_guided_borrowing_amount": round(ltv_limit, 2),
                "recommended_minimum_down_payment": round(minimum_down_payment, 2),
                "additional_down_payment_needed": round(
                    max(minimum_down_payment - application.available_down_payment, 0.0),
                    2,
                ),
                "product_guided_max_principal": round(
                    min(max_principal_selected_term, ltv_limit),
                    2,
                ),
            }
        )
    return guidance


def find_shortest_supporting_term(repayment_plans: list[dict[str, Any]]) -> int | None:
    supporting = [plan["term_months"] for plan in repayment_plans if plan["affordable"]]
    return min(supporting) if supporting else None


def _plan_status(
    *,
    basic_requirements_met: bool,
    selected_plan_fits: bool,
    selected_term: int,
    shortest_supporting_term: int | None,
    eligible_terms: tuple[int, ...],
    max_principal_selected_term: float,
    max_principal_longest_term: float,
) -> str:
    if not basic_requirements_met:
        return "BASIC_PRODUCT_REQUIREMENTS_NOT_MET"
    if selected_plan_fits:
        return "SELECTED_PLAN_FITS"
    if shortest_supporting_term is not None and shortest_supporting_term > selected_term:
        return "LONGER_TERM_REQUIRED"
    longer_terms = [term for term in eligible_terms if term > selected_term]
    if not longer_terms and max_principal_selected_term > 0:
        return "LOWER_AMOUNT_REQUIRED"
    if longer_terms and max_principal_longest_term > max_principal_selected_term:
        return "LOWER_AMOUNT_AND_TERM_ADJUSTMENT"
    return "REQUEST_DOES_NOT_FIT_AVAILABLE_TERMS"


PLAN_STATUS_LABELS = {
    "SELECTED_PLAN_FITS": "Selected plan fits the displayed affordability limits",
    "LOWER_AMOUNT_REQUIRED": "A lower amount is required at the selected term",
    "LONGER_TERM_REQUIRED": "The requested amount requires a longer repayment term",
    "LOWER_AMOUNT_AND_TERM_ADJUSTMENT": "Both the amount and repayment term should be adjusted",
    "REQUEST_DOES_NOT_FIT_AVAILABLE_TERMS": "The request does not fit the available repayment terms",
    "BASIC_PRODUCT_REQUIREMENTS_NOT_MET": "Basic product requirements are not met",
}


def simulate_loan(application: LoanApplication) -> dict[str, Any]:
    """Evaluate cash-flow affordability, resilience, and calculated alternatives."""

    policy = validate_loan_application(application)
    eligible_terms = _eligible_terms(application, policy)
    income = application.monthly_net_income
    expenses = application.monthly_essential_expenses
    existing_debt = application.existing_monthly_debt_payments
    capacity = calculate_affordable_emi(application, policy)
    max_affordable_emi = capacity["max_affordable_emi"]
    required_buffer = capacity["required_residual_buffer"]

    requested_emi = calculate_emi(
        application.requested_loan_amount,
        policy.annual_interest_rate,
        application.preferred_term_months,
    )
    preferred_age_allowed = application.preferred_term_months in eligible_terms
    max_principal_selected_term = (
        min(
            principal_from_emi(
                max_affordable_emi,
                policy.annual_interest_rate,
                application.preferred_term_months,
            ),
            policy.maximum_principal,
        )
        if preferred_age_allowed
        else 0.0
    )
    longest_term = max(eligible_terms) if eligible_terms else None
    max_principal_longest_term = (
        min(
            principal_from_emi(
                max_affordable_emi,
                policy.annual_interest_rate,
                longest_term,
            ),
            policy.maximum_principal,
        )
        if longest_term is not None
        else 0.0
    )

    cash_remaining = income - expenses - existing_debt - requested_emi
    cash_remaining_after_buffer = cash_remaining - required_buffer
    existing_debt_ratio = existing_debt / income
    expense_ratio = expenses / income
    requested_total_debt_ratio = (existing_debt + requested_emi) / income
    loan_plan_fit = calculate_loan_plan_fit(max_affordable_emi, requested_emi)
    emi_shortfall = max(requested_emi - max_affordable_emi, 0.0)
    amount_reduction = max(
        application.requested_loan_amount - max_principal_selected_term,
        0.0,
    )

    repayment_plans: list[dict[str, Any]] = []
    stability_pass = (
        application.income_stability_years >= policy.minimum_income_stability_years
    )
    product_limit_pass = application.requested_loan_amount <= policy.maximum_principal
    for term in eligible_terms:
        emi = calculate_emi(
            application.requested_loan_amount,
            policy.annual_interest_rate,
            term,
        )
        total_repayment = emi * term
        total_ratio = (existing_debt + emi) / income
        residual = income - expenses - existing_debt - emi
        affordable = (
            stability_pass
            and product_limit_pass
            and emi <= max_affordable_emi + 0.01
            and residual >= required_buffer - 0.01
        )
        repayment_plans.append(
            {
                "term_months": term,
                "monthly_emi": round(emi, 2),
                "total_interest": round(total_repayment - application.requested_loan_amount, 2),
                "total_repayment": round(total_repayment, 2),
                "cash_remaining": round(residual, 2),
                "cash_remaining_after_buffer": round(residual - required_buffer, 2),
                "total_debt_service_ratio": round(total_ratio, 4),
                "affordable": affordable,
                "preferred": term == application.preferred_term_months,
            }
        )

    shortest_supporting_term = find_shortest_supporting_term(repayment_plans)
    supporting_plan = next(
        (
            plan
            for plan in repayment_plans
            if plan["term_months"] == shortest_supporting_term
        ),
        None,
    )
    selected_total_interest = calculate_total_interest(
        application.requested_loan_amount,
        policy.annual_interest_rate,
        application.preferred_term_months,
    )
    alternative_total_interest = (
        float(supporting_plan["total_interest"])
        if supporting_plan is not None
        else None
    )
    incremental_interest = (
        max(alternative_total_interest - selected_total_interest, 0.0)
        if alternative_total_interest is not None
        else None
    )

    basic_requirements_met = (
        preferred_age_allowed
        and stability_pass
        and product_limit_pass
        and max_affordable_emi > 0
    )
    selected_plan_fits = basic_requirements_met and requested_emi <= max_affordable_emi + 0.01
    plan_status = _plan_status(
        basic_requirements_met=basic_requirements_met,
        selected_plan_fits=selected_plan_fits,
        selected_term=application.preferred_term_months,
        shortest_supporting_term=shortest_supporting_term,
        eligible_terms=eligible_terms,
        max_principal_selected_term=max_principal_selected_term,
        max_principal_longest_term=max_principal_longest_term,
    )

    monthly_post_loan_commitments = expenses + existing_debt + requested_emi
    asset_resilience = calculate_asset_resilience(
        application.assets,
        monthly_post_loan_commitments,
    )
    foundation = calculate_financial_foundation(
        application,
        policy,
        float(asset_resilience["adjusted_emergency_liquidity"]),
    )
    stress_test = calculate_stress_scenario(
        application,
        policy,
        max_affordable_emi,
        required_buffer,
    )
    product_guidance = _product_guidance(
        application,
        policy,
        max_principal_selected_term,
    )

    coverage = asset_resilience["emergency_coverage_months"]
    thin_resilience = (
        coverage is not None and float(coverage) < 3
    ) or stress_test["resilience"] != "RESILIENT"
    ltv_stretched = (
        product_guidance["requested_ltv"] is not None
        and product_guidance["recommended_ltv"] is not None
        and float(product_guidance["requested_ltv"])
        > float(product_guidance["recommended_ltv"])
    )
    if selected_plan_fits:
        decision_summary = (
            "SUITABLE_BUT_FINANCIALLY_THIN"
            if thin_resilience or ltv_stretched
            else "SUITABLE_PLAN"
        )
    elif plan_status == "BASIC_PRODUCT_REQUIREMENTS_NOT_MET":
        decision_summary = "NO_SUITABLE_CURRENT_STRUCTURE"
    else:
        decision_summary = "RESTRUCTURE_RECOMMENDED"

    policy_checks = [
        {
            "code": "AGE_AT_MATURITY",
            "label": "Age at selected-term maturity",
            "passed": preferred_age_allowed,
            "observed": round(application.age + application.preferred_term_months / 12, 1),
            "limit": policy.maximum_age_at_maturity,
        },
        {
            "code": "INCOME_STABILITY",
            "label": "Income-source stability",
            "passed": stability_pass,
            "observed": application.income_stability_years,
            "limit": policy.minimum_income_stability_years,
        },
        {
            "code": "DEBT_SERVICE",
            "label": "Total debt-service ratio",
            "passed": requested_total_debt_ratio <= policy.maximum_debt_service_ratio,
            "observed": round(requested_total_debt_ratio, 4),
            "limit": policy.maximum_debt_service_ratio,
        },
        {
            "code": "RESIDUAL_INCOME",
            "label": "Cash remaining after safety buffer",
            "passed": cash_remaining_after_buffer >= 0,
            "observed": round(cash_remaining_after_buffer, 2),
            "limit": 0.0,
        },
        {
            "code": "PRODUCT_LIMIT",
            "label": "Illustrative product maximum",
            "passed": product_limit_pass,
            "observed": application.requested_loan_amount,
            "limit": policy.maximum_principal,
        },
        {
            "code": "SELECTED_TERM_CAPACITY",
            "label": "Requested amount within selected-term capacity",
            "passed": application.requested_loan_amount <= max_principal_selected_term + 0.01,
            "observed": application.requested_loan_amount,
            "limit": round(max_principal_selected_term, 2),
        },
    ]

    recommendations: list[dict[str, Any]] = []
    if selected_plan_fits:
        recommendations.append(
            {
                "action": "Keep the selected payment below the calculated ceiling",
                "detail": (
                    f"The requested EMI is {requested_emi:,.2f} against a calculated monthly "
                    f"capacity of {max_affordable_emi:,.2f}."
                ),
            }
        )
    else:
        if amount_reduction > 0:
            recommendations.append(
                {
                    "action": "Reduce the amount at the selected term",
                    "detail": (
                        f"The selected {application.preferred_term_months}-month term supports "
                        f"approximately {max_principal_selected_term:,.2f}; reduce the request "
                        f"by about {amount_reduction:,.2f}."
                    ),
                }
            )
        if supporting_plan is not None and shortest_supporting_term != application.preferred_term_months:
            recommendations.append(
                {
                    "action": "Consider the shortest supporting term",
                    "detail": (
                        f"The requested amount first fits at {shortest_supporting_term} months "
                        f"with an EMI of {float(supporting_plan['monthly_emi']):,.2f}. This adds "
                        f"approximately {float(incremental_interest or 0):,.2f} of total interest "
                        "versus the selected term."
                    ),
                }
            )
        elif max_principal_longest_term > max_principal_selected_term:
            recommendations.append(
                {
                    "action": "Adjust both amount and term",
                    "detail": (
                        f"No available term supports the full request. The longest eligible "
                        f"{longest_term}-month term supports approximately "
                        f"{max_principal_longest_term:,.2f}."
                    ),
                }
            )

    if emi_shortfall > 0:
        recommendations.append(
            {
                "action": "Close the monthly affordability gap",
                "detail": (
                    f"The preferred EMI is about {emi_shortfall:,.2f} above the calculated "
                    "monthly capacity. A lower EMI, lower existing debt, lower recurring "
                    "expenses, or higher verified income would be required."
                ),
            }
        )
    if coverage is not None and float(coverage) < 3:
        three_month_target = monthly_post_loan_commitments * 3
        liquidity_gap = max(
            three_month_target
            - float(asset_resilience["adjusted_emergency_liquidity"]),
            0.0,
        )
        recommendations.append(
            {
                "action": "Build emergency liquidity separately from EMI capacity",
                "detail": (
                    f"Adjusted liquid resources cover {float(coverage):.1f} month(s) of the "
                    f"submitted post-loan commitments. About {liquidity_gap:,.2f} more would "
                    "reach a three-month planning buffer; this does not increase affordable EMI."
                ),
            }
        )
    if ltv_stretched:
        recommendations.append(
            {
                "action": "Increase the down payment or reduce the financed amount",
                "detail": (
                    f"The requested LTV is {float(product_guidance['requested_ltv']):.1%} "
                    f"against the illustrative {float(product_guidance['recommended_ltv']):.0%} "
                    "guidance threshold."
                ),
            }
        )
    if application.product_type == "personal_loan" and product_guidance["salary_multiple_band"] in {"STRETCHED", "VERY_HIGH"}:
        recommendations.append(
            {
                "action": "Review the requested personal-loan size",
                "detail": (
                    f"The request equals {float(product_guidance['loan_to_monthly_income_multiple']):.1f} "
                    "months of submitted income, which is a stretched planning signal."
                ),
            }
        )

    return {
        "application_id": generate_application_id(),
        "plan_status": plan_status,
        "plan_status_label": PLAN_STATUS_LABELS[plan_status],
        "combined_decision_summary": decision_summary,
        "eligible_for_requested_loan": selected_plan_fits,
        "loan_plan_fit_pct": round(loan_plan_fit, 2),
        "requested_principal": round(application.requested_loan_amount, 2),
        "requested_emi": round(requested_emi, 2),
        "max_affordable_emi": round(max_affordable_emi, 2),
        "emi_shortfall": round(emi_shortfall, 2),
        "max_principal_selected_term": round(max_principal_selected_term, 2),
        "max_principal_longest_term": round(max_principal_longest_term, 2),
        "amount_reduction_required": round(amount_reduction, 2),
        "selected_term_months": application.preferred_term_months,
        "shortest_supporting_term_months": shortest_supporting_term,
        "longest_age_eligible_term_months": longest_term,
        "alternative_term_emi": (
            float(supporting_plan["monthly_emi"]) if supporting_plan is not None else None
        ),
        "selected_term_total_interest": round(selected_total_interest, 2),
        "alternative_term_total_interest": alternative_total_interest,
        "incremental_interest_for_alternative": (
            round(incremental_interest, 2) if incremental_interest is not None else None
        ),
        "cash_remaining": round(cash_remaining, 2),
        "required_residual_buffer": round(required_buffer, 2),
        "cash_remaining_after_buffer": round(cash_remaining_after_buffer, 2),
        "debt_service_capacity": round(capacity["debt_service_capacity"], 2),
        "cash_flow_capacity": round(capacity["cash_flow_capacity"], 2),
        "current_debt_service_ratio": round(existing_debt_ratio, 4),
        "requested_total_debt_service_ratio": round(requested_total_debt_ratio, 4),
        "essential_expense_ratio": round(expense_ratio, 4),
        "request_fits_any_available_term": shortest_supporting_term is not None,
        "financial_foundation": foundation,
        "asset_resilience": asset_resilience,
        "stress_test": stress_test,
        "product_guidance": product_guidance,
        "policy_checks": policy_checks,
        "repayment_plans": repayment_plans,
        "recommendations": recommendations,
        "policy_assumptions": {
            **asdict(policy),
            "annual_interest_rate_percent": round(policy.annual_interest_rate * 100, 3),
            "maximum_debt_service_ratio_percent": round(
                policy.maximum_debt_service_ratio * 100,
                2,
            ),
            "minimum_residual_income_ratio_percent": round(
                policy.minimum_residual_income_ratio * 100,
                2,
            ),
            "asset_liquidity_weights": dict(ASSET_LIQUIDITY_WEIGHTS),
        },
        "preferred_plan_monthly_emi": round(requested_emi, 2),
        "maximum_affordable_monthly_emi": round(max_affordable_emi, 2),
        "maximum_eligible_loan_amount": round(max_principal_selected_term, 2),
        "disclaimer": (
            "Illustrative decision-support simulation based on self-declared and unverified "
            "values. Loan Plan Fit is not an approval probability, assets never increase "
            "affordable EMI, and no bureau credit score is used."
        ),
    }
