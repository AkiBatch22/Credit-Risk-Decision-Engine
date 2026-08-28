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
    assert payload["maximum_eligible_loan_amount"] >= APPLICATION["requested_loan_amount"]
    assert 0 <= payload["financial_readiness_score"] <= 100
    assert payload["repayment_plans"]
    assert "available" in payload["repayment_stress"]
    assert "bureau credit score" in payload["disclaimer"]


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
