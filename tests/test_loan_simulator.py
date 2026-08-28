import math

import pytest

from src.components.feature_contract import APPLICATION_ID_PATTERN
from src.components.loan_simulator import (
    LoanApplication,
    calculate_emi,
    principal_from_emi,
    simulate_loan,
)


def application(**overrides):
    values = {
        "product_type": "personal_loan",
        "age": 34,
        "employment_type": "Salaried",
        "income_stability_years": 5,
        "monthly_net_income": 100_000,
        "monthly_essential_expenses": 30_000,
        "existing_monthly_debt_payments": 5_000,
        "requested_loan_amount": 300_000,
        "preferred_term_months": 36,
    }
    values.update(overrides)
    return LoanApplication(**values)


def test_emi_and_principal_are_inverse_calculations():
    emi = calculate_emi(500_000, 0.12, 60)
    recovered = principal_from_emi(emi, 0.12, 60)
    assert math.isclose(recovered, 500_000, rel_tol=1e-10)
    assert calculate_emi(0, 0.12, 60) == 0
    with pytest.raises(ValueError):
        calculate_emi(-1, 0.12, 60)


def test_eligible_application_returns_plans_and_transparent_score():
    result = simulate_loan(application())
    assert APPLICATION_ID_PATTERN.fullmatch(result["application_id"])
    assert result["eligibility_status"] == "ELIGIBLE"
    assert result["eligible_for_requested_loan"] is True
    assert result["maximum_eligible_loan_amount"] >= 300_000
    assert 0 <= result["financial_readiness_score"] <= 100
    assert sum(result["readiness_score_breakdown"].values()) == pytest.approx(
        result["financial_readiness_score"], abs=1
    )
    assert any(plan["affordable"] for plan in result["repayment_plans"])
    assert all("total_interest" in plan for plan in result["repayment_plans"])
    assert result["recommendations"]


def test_over_requested_application_gets_exact_improvement_actions():
    result = simulate_loan(
        application(
            monthly_net_income=100_000,
            monthly_essential_expenses=40_000,
            existing_monthly_debt_payments=10_000,
            requested_loan_amount=2_000_000,
            preferred_term_months=60,
        )
    )
    assert result["eligibility_status"] == "ELIGIBLE_FOR_LOWER_AMOUNT_OR_DIFFERENT_TERM"
    assert result["eligible_for_requested_loan"] is False
    assert 0 < result["maximum_eligible_loan_amount"] < 2_000_000
    actions = {item["action"] for item in result["recommendations"]}
    assert "Reduce the requested amount" in actions
    assert "Increase verifiable monthly income" in actions


def test_no_cashflow_capacity_is_not_currently_eligible():
    result = simulate_loan(
        application(
            monthly_net_income=50_000,
            monthly_essential_expenses=35_000,
            existing_monthly_debt_payments=15_000,
            requested_loan_amount=500_000,
            preferred_term_months=24,
        )
    )
    assert result["eligibility_status"] == "NOT_CURRENTLY_ELIGIBLE"
    assert result["maximum_affordable_monthly_emi"] == 0
    assert result["maximum_eligible_loan_amount"] == 0
    assert not any(plan["affordable"] for plan in result["repayment_plans"])


def test_age_and_income_stability_are_explicit_policy_checks():
    result = simulate_loan(
        application(
            age=64,
            income_stability_years=0.25,
            preferred_term_months=24,
        )
    )
    checks = {check["code"]: check for check in result["policy_checks"]}
    assert checks["AGE_AT_MATURITY"]["passed"] is False
    assert checks["INCOME_STABILITY"]["passed"] is False
    assert result["eligibility_status"] == "NOT_CURRENTLY_ELIGIBLE"


def test_simulation_is_deterministic_except_for_trace_id():
    first = simulate_loan(application())
    second = simulate_loan(application())
    assert first["application_id"] != second["application_id"]
    comparable_first = {key: value for key, value in first.items() if key != "application_id"}
    comparable_second = {key: value for key, value in second.items() if key != "application_id"}
    assert comparable_first == comparable_second
