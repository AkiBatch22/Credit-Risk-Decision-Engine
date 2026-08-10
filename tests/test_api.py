from fastapi.testclient import TestClient
import pandas as pd

from api import create_app


APPLICATION = {
    "age": 34,
    "years_employed": 5,
    "family_members": 2,
    "number_of_children": 0,
    "annual_income": 202500,
    "requested_loan_amount": 406597.5,
    "loan_annuity": 24700.5,
    "goods_purchase_price": 351000,
    "credit_product_type": "Cash loans",
    "income_type": "Working",
    "housing_situation": "House / apartment",
    "owns_car": False,
    "owns_property": True,
}


class FakePredictionPipeline:
    def predict(self, applicants, **kwargs):
        assert "SK_ID_CURR" not in applicants
        assert not any(name.startswith("EXT_SOURCE") for name in applicants)
        return pd.DataFrame(
            [
                {
                    "application_id": "APP-7C91E28A4F",
                    "probability": 0.12,
                    "probability_percent": 12.0,
                    "risk_band": "HIGH",
                    "recommendation": "MANUAL_REVIEW",
                    "expected_loss": 7200.0,
                    "top_risk_reasons": [],
                }
            ]
        )


def test_api_health_and_new_applicant_prediction():
    client = TestClient(create_app(FakePredictionPipeline))
    assert client.get("/health").json() == {"status": "healthy"}
    response = client.post("/predict", json=APPLICATION)
    assert response.status_code == 200
    payload = response.json()
    assert payload["application_id"] == "APP-7C91E28A4F"
    assert 0 <= payload["calibrated_probability"] <= 1
    assert payload["recommendation"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}
    assert payload["expected_loss"] == 7200.0


def test_api_rejects_historical_id_and_external_scores():
    client = TestClient(create_app(FakePredictionPipeline))
    for forbidden_field in ("SK_ID_CURR", "EXT_SOURCE_1"):
        response = client.post("/predict", json={**APPLICATION, forbidden_field: 123})
        assert response.status_code == 422
