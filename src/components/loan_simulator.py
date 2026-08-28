"""Transparent, input-only loan affordability and repayment simulation.

This module intentionally does not use the historical ML model or claim to produce
a bureau credit score. Every result is reproducible from submitted application
values and the selected illustrative product policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.components.feature_contract import generate_application_id


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


PRODUCT_POLICIES: dict[str, LoanProductPolicy] = {
    "personal_loan": LoanProductPolicy(
        code="personal_loan",
        label="Personal Loan",
        annual_interest_rate=0.14,
        permitted_terms_months=(12, 24, 36, 48, 60),
        maximum_debt_service_ratio=0.40,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=1.0,
        maximum_age_at_maturity=65,
        maximum_principal=2_000_000.0,
    ),
    "vehicle_loan": LoanProductPolicy(
        code="vehicle_loan",
        label="Vehicle Loan",
        annual_interest_rate=0.10,
        permitted_terms_months=(12, 24, 36, 48, 60, 72, 84),
        maximum_debt_service_ratio=0.45,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=1.0,
        maximum_age_at_maturity=65,
        maximum_principal=5_000_000.0,
    ),
    "home_loan": LoanProductPolicy(
        code="home_loan",
        label="Home Loan",
        annual_interest_rate=0.085,
        permitted_terms_months=(60, 120, 180, 240),
        maximum_debt_service_ratio=0.50,
        minimum_residual_income_ratio=0.20,
        minimum_income_stability_years=2.0,
        maximum_age_at_maturity=70,
        maximum_principal=20_000_000.0,
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
    return policy


def _eligible_terms(application: LoanApplication, policy: LoanProductPolicy) -> tuple[int, ...]:
    return tuple(
        term
        for term in policy.permitted_terms_months
        if application.age + term / 12 <= policy.maximum_age_at_maturity
    )


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _readiness_score(
    application: LoanApplication,
    policy: LoanProductPolicy,
    requested_emi: float,
) -> tuple[int, dict[str, float]]:
    income = application.monthly_net_income
    existing_debt_ratio = application.existing_monthly_debt_payments / income
    expense_ratio = application.monthly_essential_expenses / income
    total_debt_ratio = (
        application.existing_monthly_debt_payments + requested_emi
    ) / income
    pre_loan_surplus_ratio = max(
        income
        - application.monthly_essential_expenses
        - application.existing_monthly_debt_payments,
        0.0,
    ) / income

    cash_flow = 30 * _bounded_score(pre_loan_surplus_ratio / 0.40)
    existing_debt = 20 * _bounded_score(
        (policy.maximum_debt_service_ratio - existing_debt_ratio)
        / policy.maximum_debt_service_ratio
    )
    requested_burden = 25 * _bounded_score(
        (policy.maximum_debt_service_ratio - total_debt_ratio)
        / policy.maximum_debt_service_ratio
    )
    stability = 15 * _bounded_score(
        application.income_stability_years
        / max(policy.minimum_income_stability_years * 2, 1)
    )
    expense_control = 10 * _bounded_score((0.80 - expense_ratio) / 0.50)
    breakdown = {
        "Cash-flow capacity": round(cash_flow, 2),
        "Existing debt burden": round(existing_debt, 2),
        "Requested-loan burden": round(requested_burden, 2),
        "Income stability": round(stability, 2),
        "Expense balance": round(expense_control, 2),
    }
    return round(sum(breakdown.values())), breakdown


def _readiness_band(score: int) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 65:
        return "GOOD"
    if score >= 50:
        return "DEVELOPING"
    return "NEEDS_ATTENTION"


def simulate_loan(application: LoanApplication) -> dict[str, Any]:
    """Evaluate affordability and generate actionable repayment alternatives."""

    policy = validate_loan_application(application)
    eligible_terms = _eligible_terms(application, policy)
    income = application.monthly_net_income
    expenses = application.monthly_essential_expenses
    existing_debt = application.existing_monthly_debt_payments

    debt_service_capacity = max(
        income * policy.maximum_debt_service_ratio - existing_debt,
        0.0,
    )
    required_residual_income = income * policy.minimum_residual_income_ratio
    cash_flow_capacity = max(
        income - expenses - existing_debt - required_residual_income,
        0.0,
    )
    maximum_affordable_emi = min(debt_service_capacity, cash_flow_capacity)

    longest_term = max(eligible_terms) if eligible_terms else None
    maximum_eligible_principal = (
        min(
            principal_from_emi(
                maximum_affordable_emi,
                policy.annual_interest_rate,
                longest_term,
            ),
            policy.maximum_principal,
        )
        if longest_term is not None
        else 0.0
    )

    preferred_age_allowed = application.preferred_term_months in eligible_terms
    requested_emi = calculate_emi(
        application.requested_loan_amount,
        policy.annual_interest_rate,
        application.preferred_term_months,
    )
    requested_total_debt_ratio = (existing_debt + requested_emi) / income
    existing_debt_ratio = existing_debt / income
    expense_ratio = expenses / income
    post_loan_surplus = income - expenses - existing_debt - requested_emi

    policy_checks = [
        {
            "code": "AGE_AT_MATURITY",
            "label": "Age at loan maturity",
            "passed": preferred_age_allowed,
            "observed": round(application.age + application.preferred_term_months / 12, 1),
            "limit": policy.maximum_age_at_maturity,
        },
        {
            "code": "INCOME_STABILITY",
            "label": "Income-source stability",
            "passed": application.income_stability_years
            >= policy.minimum_income_stability_years,
            "observed": application.income_stability_years,
            "limit": policy.minimum_income_stability_years,
        },
        {
            "code": "DEBT_SERVICE",
            "label": "Total debt-service ratio",
            "passed": requested_total_debt_ratio
            <= policy.maximum_debt_service_ratio,
            "observed": round(requested_total_debt_ratio, 4),
            "limit": policy.maximum_debt_service_ratio,
        },
        {
            "code": "RESIDUAL_INCOME",
            "label": "Post-loan residual income",
            "passed": post_loan_surplus >= required_residual_income,
            "observed": round(post_loan_surplus, 2),
            "limit": round(required_residual_income, 2),
        },
        {
            "code": "PRODUCT_LIMIT",
            "label": "Illustrative product maximum",
            "passed": application.requested_loan_amount <= policy.maximum_principal,
            "observed": application.requested_loan_amount,
            "limit": policy.maximum_principal,
        },
        {
            "code": "AFFORDABLE_PRINCIPAL",
            "label": "Requested amount within affordability capacity",
            "passed": application.requested_loan_amount <= maximum_eligible_principal,
            "observed": application.requested_loan_amount,
            "limit": round(maximum_eligible_principal, 2),
        },
    ]

    hard_checks_pass = all(
        check["passed"]
        for check in policy_checks
        if check["code"] in {"AGE_AT_MATURITY", "INCOME_STABILITY", "PRODUCT_LIMIT"}
    )
    fully_eligible = hard_checks_pass and all(check["passed"] for check in policy_checks)
    if fully_eligible:
        eligibility_status = "ELIGIBLE"
    elif hard_checks_pass and maximum_eligible_principal > 0:
        eligibility_status = "ELIGIBLE_FOR_LOWER_AMOUNT_OR_DIFFERENT_TERM"
    else:
        eligibility_status = "NOT_CURRENTLY_ELIGIBLE"

    readiness_score, readiness_breakdown = _readiness_score(
        application,
        policy,
        requested_emi,
    )

    repayment_plans: list[dict[str, Any]] = []
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
            emi <= maximum_affordable_emi
            and total_ratio <= policy.maximum_debt_service_ratio
            and residual >= required_residual_income
            and application.requested_loan_amount <= policy.maximum_principal
        )
        repayment_plans.append(
            {
                "term_months": term,
                "monthly_emi": round(emi, 2),
                "total_interest": round(total_repayment - application.requested_loan_amount, 2),
                "total_repayment": round(total_repayment, 2),
                "post_emi_monthly_surplus": round(residual, 2),
                "total_debt_service_ratio": round(total_ratio, 4),
                "affordable": affordable,
                "preferred": term == application.preferred_term_months,
            }
        )

    recommendations: list[dict[str, Any]] = []
    if fully_eligible:
        affordable_plans = [plan for plan in repayment_plans if plan["affordable"]]
        if affordable_plans:
            lowest_interest = min(affordable_plans, key=lambda plan: plan["total_interest"])
            lowest_emi = min(affordable_plans, key=lambda plan: plan["monthly_emi"])
            recommendations.extend(
                [
                    {
                        "action": "Compare affordable terms",
                        "detail": (
                            f"The {lowest_interest['term_months']}-month plan minimizes total "
                            f"interest, while the {lowest_emi['term_months']}-month plan minimizes "
                            "the monthly instalment."
                        ),
                    },
                    {
                        "action": "Keep a cash buffer",
                        "detail": (
                            "Treat the maximum eligible amount as a ceiling, not a target, and "
                            "retain funds for emergencies and irregular expenses."
                        ),
                    },
                ]
            )
    else:
        if application.requested_loan_amount > maximum_eligible_principal:
            reduction = application.requested_loan_amount - maximum_eligible_principal
            recommendations.append(
                {
                    "action": "Reduce the requested amount",
                    "detail": (
                        f"The current inputs support approximately {maximum_eligible_principal:,.2f}. "
                        f"Reducing the request by about {reduction:,.2f} would bring it closer to "
                        "the simulated affordability ceiling."
                    ),
                }
            )
        required_income = (existing_debt + requested_emi) / policy.maximum_debt_service_ratio
        income_gap = max(required_income - income, 0.0)
        if income_gap > 0:
            recommendations.append(
                {
                    "action": "Increase verifiable monthly income",
                    "detail": (
                        f"An additional approximately {income_gap:,.2f} of stable monthly income "
                        "would address the debt-service-ratio shortfall for the preferred plan."
                    ),
                }
            )
        debt_reduction = max(
            existing_debt + requested_emi
            - income * policy.maximum_debt_service_ratio,
            0.0,
        )
        if debt_reduction > 0 and existing_debt > 0:
            recommendations.append(
                {
                    "action": "Reduce existing monthly debt payments",
                    "detail": (
                        f"Lowering existing scheduled debt payments by approximately "
                        f"{min(debt_reduction, existing_debt):,.2f} per month would improve capacity."
                    ),
                }
            )
        expense_reduction = max(required_residual_income - post_loan_surplus, 0.0)
        if expense_reduction > 0:
            recommendations.append(
                {
                    "action": "Improve monthly cash-flow headroom",
                    "detail": (
                        f"A monthly expense reduction or recurring-income improvement of about "
                        f"{expense_reduction:,.2f} would restore the configured residual-income buffer."
                    ),
                }
            )
        stability_gap = max(
            policy.minimum_income_stability_years - application.income_stability_years,
            0.0,
        )
        if stability_gap > 0:
            recommendations.append(
                {
                    "action": "Build a longer stable-income record",
                    "detail": (
                        f"The illustrative policy requires another {stability_gap:.1f} year(s) "
                        "of continuous income history for this product."
                    ),
                }
            )
        affordable_plans = [plan for plan in repayment_plans if plan["affordable"]]
        if affordable_plans:
            best = min(affordable_plans, key=lambda plan: plan["monthly_emi"])
            recommendations.append(
                {
                    "action": "Consider a different repayment term",
                    "detail": (
                        f"The {best['term_months']}-month illustration lowers the EMI to "
                        f"{best['monthly_emi']:,.2f}, although a longer term increases total interest."
                    ),
                }
            )

    return {
        "application_id": generate_application_id(),
        "eligibility_status": eligibility_status,
        "eligible_for_requested_loan": fully_eligible,
        "financial_readiness_score": readiness_score,
        "financial_readiness_band": _readiness_band(readiness_score),
        "maximum_eligible_loan_amount": round(maximum_eligible_principal, 2),
        "maximum_affordable_monthly_emi": round(maximum_affordable_emi, 2),
        "requested_loan_amount": round(application.requested_loan_amount, 2),
        "preferred_term_months": application.preferred_term_months,
        "preferred_plan_monthly_emi": round(requested_emi, 2),
        "preferred_plan_total_interest": round(
            requested_emi * application.preferred_term_months
            - application.requested_loan_amount,
            2,
        ),
        "post_preferred_emi_monthly_surplus": round(post_loan_surplus, 2),
        "current_debt_service_ratio": round(existing_debt_ratio, 4),
        "requested_total_debt_service_ratio": round(requested_total_debt_ratio, 4),
        "essential_expense_ratio": round(expense_ratio, 4),
        "readiness_score_breakdown": readiness_breakdown,
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
        },
        "disclaimer": (
            "Illustrative affordability simulation based only on submitted values. The Financial "
            "Readiness Score is not a bureau credit score, credit report, or lending decision."
        ),
    }
