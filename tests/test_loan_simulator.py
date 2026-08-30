import math

import pytest

from src.components.feature_contract import APPLICATION_ID_PATTERN
from src.components.loan_simulator import (
    ASSET_LIQUIDITY_WEIGHTS,
    AssetProfile,
    LoanApplication,
    calculate_adjusted_liquidity,
    calculate_asset_equity,
    calculate_emi,
    calculate_emergency_coverage,
    calculate_loan_plan_fit,
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


@pytest.mark.parametrize(
    ("capacity", "requested", "expected"),
    [(100, 80, 100), (100, 100, 100), (50, 100, 50), (0, 100, 0), (100, 0, 100)],
)
def test_loan_plan_fit_boundaries(capacity, requested, expected):
    assert calculate_loan_plan_fit(capacity, requested) == expected


@pytest.mark.parametrize(("capacity", "requested"), [(-1, 100), (100, -1)])
def test_loan_plan_fit_rejects_negative_values(capacity, requested):
    with pytest.raises(ValueError):
        calculate_loan_plan_fit(capacity, requested)


def test_selected_plan_fit_and_selected_term_principal_are_headline_values():
    result = simulate_loan(application())
    assert APPLICATION_ID_PATTERN.fullmatch(result["application_id"])
    assert result["plan_status"] == "SELECTED_PLAN_FITS"
    assert result["eligible_for_requested_loan"] is True
    assert result["loan_plan_fit_pct"] == 100
    expected = principal_from_emi(result["max_affordable_emi"], 0.1275, 36)
    assert result["max_principal_selected_term"] == pytest.approx(expected, abs=0.01)
    assert result["maximum_eligible_loan_amount"] == result["max_principal_selected_term"]
    assert result["max_principal_longest_term"] > result["max_principal_selected_term"]


def test_exact_emi_gap_and_amount_reduction_are_reported():
    result = simulate_loan(application(requested_loan_amount=1_200_000))
    assert result["plan_status"] == "LONGER_TERM_REQUIRED"
    assert result["loan_plan_fit_pct"] < 100
    assert result["emi_shortfall"] == pytest.approx(
        result["requested_emi"] - result["max_affordable_emi"], abs=0.01
    )
    assert result["amount_reduction_required"] == pytest.approx(
        result["requested_principal"] - result["max_principal_selected_term"], abs=0.01
    )
    assert result["shortest_supporting_term_months"] == 48
    assert result["alternative_term_emi"] <= result["max_affordable_emi"] + 0.01
    assert result["incremental_interest_for_alternative"] > 0


def test_request_beyond_every_term_requires_amount_and_term_adjustment():
    result = simulate_loan(application(requested_loan_amount=1_900_000))
    assert result["plan_status"] == "LOWER_AMOUNT_AND_TERM_ADJUSTMENT"
    assert result["request_fits_any_available_term"] is False
    assert result["max_principal_longest_term"] > result["max_principal_selected_term"]


def test_longest_selected_term_requires_lower_amount_only():
    result = simulate_loan(
        application(requested_loan_amount=1_900_000, preferred_term_months=60)
    )
    assert result["plan_status"] == "LOWER_AMOUNT_REQUIRED"


def test_age_and_stability_are_basic_product_requirements():
    result = simulate_loan(
        application(age=64, income_stability_years=0.25, preferred_term_months=24)
    )
    checks = {check["code"]: check for check in result["policy_checks"]}
    assert checks["AGE_AT_MATURITY"]["passed"] is False
    assert checks["INCOME_STABILITY"]["passed"] is False
    assert result["plan_status"] == "BASIC_PRODUCT_REQUIREMENTS_NOT_MET"


def test_no_cashflow_capacity_has_zero_fit_and_zero_selected_cap():
    result = simulate_loan(
        application(
            monthly_net_income=50_000,
            monthly_essential_expenses=35_000,
            existing_monthly_debt_payments=15_000,
            requested_loan_amount=500_000,
            preferred_term_months=24,
        )
    )
    assert result["loan_plan_fit_pct"] == 0
    assert result["max_affordable_emi"] == 0
    assert result["max_principal_selected_term"] == 0
    assert not any(plan["affordable"] for plan in result["repayment_plans"])


def test_asset_liquidity_weights_and_negative_equity_floor():
    assets = AssetProfile(
        cash=100_000,
        fixed_deposit=100_000,
        debt_fund=100_000,
        equity=100_000,
        gold=100_000,
        property_value=500_000,
        property_liability=600_000,
        vehicle_value=200_000,
        vehicle_liability=100_000,
        epf_ppf=100_000,
        nps=1_000_000,
    )
    expected = 100_000 * (1 + 0.95 + 0.85 + 0.60 + 0.70 + 0.10 + 0.15)
    assert calculate_asset_equity(500_000, 600_000) == 0
    assert calculate_adjusted_liquidity(assets) == pytest.approx(expected)
    assert ASSET_LIQUIDITY_WEIGHTS["nps"] == 0


def test_emergency_coverage_handles_no_commitments():
    assert calculate_emergency_coverage(120_000, 40_000) == 3
    assert calculate_emergency_coverage(0, 40_000) == 0
    assert calculate_emergency_coverage(120_000, 0) is None


def test_assets_change_resilience_but_never_affordable_emi_or_plan_fit():
    without_assets = simulate_loan(application())
    with_assets = simulate_loan(
        application(
            assets=AssetProfile(cash=500_000, fixed_deposit=200_000, equity=300_000, nps=2_000_000)
        )
    )
    for key in ("requested_emi", "max_affordable_emi", "loan_plan_fit_pct", "max_principal_selected_term"):
        assert with_assets[key] == without_assets[key]
    assert with_assets["asset_resilience"]["adjusted_emergency_liquidity"] > 0
    assert with_assets["financial_foundation"]["score"] > without_assets["financial_foundation"]["score"]


def test_home_loan_ltv_and_down_payment_guidance():
    result = simulate_loan(
        application(
            product_type="home_loan",
            age=32,
            monthly_net_income=250_000,
            monthly_essential_expenses=80_000,
            requested_loan_amount=8_500_000,
            preferred_term_months=240,
            purchase_property_value=10_000_000,
            available_down_payment=1_000_000,
        )
    )
    guidance = result["product_guidance"]
    assert guidance["requested_ltv"] == pytest.approx(0.85)
    assert guidance["recommended_ltv"] == pytest.approx(0.80)
    assert guidance["recommended_minimum_down_payment"] == 2_000_000
    assert guidance["additional_down_payment_needed"] == 1_000_000


@pytest.mark.parametrize(
    ("amount", "band"),
    [(600_000, "RELATIVELY_CONSERVATIVE"), (700_000, "MODERATE"), (900_000, "STRETCHED"), (1_100_000, "VERY_HIGH")],
)
def test_personal_loan_salary_multiple_bands(amount, band):
    result = simulate_loan(application(requested_loan_amount=amount))
    assert result["product_guidance"]["salary_multiple_band"] == band


def test_stress_rate_always_produces_higher_emi():
    result = simulate_loan(application())
    assert result["stress_test"]["stressed_emi"] > result["requested_emi"]


def test_simulation_is_deterministic_except_for_trace_id():
    first = simulate_loan(application())
    second = simulate_loan(application())
    assert first["application_id"] != second["application_id"]
    comparable_first = {key: value for key, value in first.items() if key != "application_id"}
    comparable_second = {key: value for key, value in second.items() if key != "application_id"}
    assert comparable_first == comparable_second
