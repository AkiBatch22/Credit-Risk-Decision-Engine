from fastapi.testclient import TestClient

from api import create_app


APPLICATION = {
    "product_type": "personal_loan",
    "age": 34,
    "employment_type": "Salaried",
    "income_stability_years": 5,
    "monthly_net_income": 100000,
    "monthly_essential_expenses": 30000,
    "existing_monthly_debt_payments": 5000,
    "requested_loan_amount": 300000,
    "preferred_term_months": 36,
    "assets": {"cash": 250000, "fixed_deposit": 100000, "nps": 500000},
}


def test_api_health_products_and_simulation():
    client = TestClient(create_app())
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["service"] == "loan-simulator"
    assert isinstance(health["repayment_stress_model_available"], bool)
    assert "personal_loan" in client.get("/products").json()["product_types"]

    response = client.post("/simulate", json=APPLICATION)
    assert response.status_code == 200
    payload = response.json()
    assert payload["application_id"].startswith("APP-")
    assert payload["eligible_for_requested_loan"] is True
    assert payload["plan_status"] == "SELECTED_PLAN_FITS"
    assert payload["loan_plan_fit_pct"] == 100
    assert payload["max_principal_selected_term"] >= APPLICATION["requested_loan_amount"]
    assert payload["maximum_eligible_loan_amount"] == payload["max_principal_selected_term"]
    assert payload["asset_resilience"]["adjusted_emergency_liquidity"] == 345000
    assert 0 <= payload["financial_foundation"]["score"] <= 100
    assert payload["repayment_plans"]
    assert "available" in payload["repayment_stress"]
    assert "bureau credit score" in payload["disclaimer"]


def test_api_and_domain_request_mapping_are_consistent():
    from api import LoanSimulationRequest
    from src.components.loan_simulator import simulate_loan

    request = LoanSimulationRequest(**APPLICATION)
    result = simulate_loan(request.to_domain())
    assert result["requested_principal"] == APPLICATION["requested_loan_amount"]
    assert result["asset_resilience"]["adjusted_emergency_liquidity"] == 345000


def test_api_rejects_negative_assets():
    client = TestClient(create_app())
    response = client.post(
        "/simulate",
        json={**APPLICATION, "assets": {"cash": -1}},
    )
    assert response.status_code == 422


def test_api_rejects_historical_model_fields_and_invalid_term():
    client = TestClient(create_app())
    for forbidden_field in ("SK_ID_CURR", "EXT_SOURCE_1", "annual_income"):
        response = client.post(
            "/simulate",
            json={**APPLICATION, forbidden_field: 123},
        )
        assert response.status_code == 422

    response = client.post(
        "/simulate",
        json={**APPLICATION, "preferred_term_months": 37},
    )
    assert response.status_code == 422


def test_api_returns_debt_repayment_plan():
    client = TestClient(create_app())
    response = client.post(
        "/debt-repayment/simulate",
        json={
            "debts": [
                {
                    "name": "Credit card",
                    "balance": 120000,
                    "annual_interest_rate_percent": 24,
                    "minimum_payment": 6000,
                },
                {
                    "name": "Personal loan",
                    "balance": 240000,
                    "annual_interest_rate_percent": 12,
                    "minimum_payment": 8000,
                },
            ],
            "extra_monthly_payment": 5000,
            "strategy": "avalanche",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ON_TRACK"
    assert payload["first_priority_debt"] == "Credit card"
    assert payload["monthly_plan_payment"] == 19000
    assert payload["estimated_interest_saved"] > 0
    assert payload["schedule"][-1]["remaining_balance"] == 0


def test_api_validates_debt_repayment_inputs():
    client = TestClient(create_app())
    response = client.post(
        "/debt-repayment/simulate",
        json={
            "debts": [
                {
                    "name": "Card",
                    "balance": 10000,
                    "annual_interest_rate_percent": 120,
                    "minimum_payment": 500,
                }
            ],
            "extra_monthly_payment": 0,
            "strategy": "avalanche",
        },
    )

    assert response.status_code == 422
