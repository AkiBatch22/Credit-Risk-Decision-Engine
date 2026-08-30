import numpy as np
import pandas as pd

from src.components.loan_simulator import LoanApplication, simulate_loan
from src.components.repayment_stress import (
    application_to_stress_features,
    estimate_repayment_stress,
    historical_to_stress_features,
    stress_band,
)
from src.utils.artifacts import save_joblib, save_json


class IdentityPreprocessor:
    def transform(self, frame):
        return frame[["AGE_YEARS", "PAYMENT_TO_MONTHLY_INCOME_RATIO"]].to_numpy()


class LinearBaseModel:
    coef_ = np.array([[-0.02, 2.0]])


class FixedCalibratedModel:
    def predict_proba(self, matrix):
        return np.tile(np.array([[0.88, 0.12]]), (len(matrix), 1))


def application():
    return LoanApplication(
        product_type="personal_loan",
        age=34,
        employment_type="Salaried",
        income_stability_years=5,
        monthly_net_income=100_000,
        monthly_essential_expenses=30_000,
        existing_monthly_debt_payments=5_000,
        requested_loan_amount=300_000,
        preferred_term_months=36,
    )


def test_application_and_historical_contracts_align():
    current = application_to_stress_features(application(), 10_000)
    historical = historical_to_stress_features(
        pd.DataFrame(
            {
                "DAYS_BIRTH": [-34 * 365.25],
                "DAYS_EMPLOYED": [-5 * 365.25],
                "AMT_INCOME_TOTAL": [1_200_000],
                "AMT_CREDIT": [300_000],
                "AMT_ANNUITY": [120_000],
                "NAME_INCOME_TYPE": ["Working"],
            }
        )
    )
    pd.testing.assert_frame_equal(current, historical, check_dtype=False)


def test_stress_bands_are_descriptive_and_ordered():
    assert stress_band(0.02) == "LOWER_HISTORICAL_STRESS"
    assert stress_band(0.08) == "MODERATE_HISTORICAL_STRESS"
    assert stress_band(0.15) == "ELEVATED_HISTORICAL_STRESS"
    assert stress_band(0.25) == "HIGHER_HISTORICAL_STRESS"


def test_estimator_reports_unavailable_without_artifacts(tmp_path):
    result = estimate_repayment_stress(
        application(),
        10_000,
        model_path=tmp_path / "missing.joblib",
        metadata_path=tmp_path / "missing.json",
    )
    assert result["available"] is False
    assert "training pipeline" in result["unavailable_reason"]


def test_ml_estimate_never_changes_deterministic_affordability(tmp_path):
    applicant = application()
    before = simulate_loan(applicant)
    estimate_repayment_stress(
        applicant,
        float(before["requested_emi"]),
        model_path=tmp_path / "missing.joblib",
        metadata_path=tmp_path / "missing.json",
    )
    after = simulate_loan(applicant)
    for key in (
        "plan_status",
        "loan_plan_fit_pct",
        "max_affordable_emi",
        "max_principal_selected_term",
    ):
        assert before[key] == after[key]


def test_estimator_returns_probability_and_reason_codes(tmp_path):
    model_path = tmp_path / "stress.joblib"
    metadata_path = tmp_path / "stress.json"
    save_joblib(
        {
            "preprocessor": IdentityPreprocessor(),
            "base_model": LinearBaseModel(),
            "calibrated_model": FixedCalibratedModel(),
            "transformed_feature_names": [
                "numeric__AGE_YEARS",
                "numeric__PAYMENT_TO_MONTHLY_INCOME_RATIO",
            ],
        },
        model_path,
    )
    save_json(
        {
            "model_name": "logistic_regression",
            "calibration_method": "sigmoid",
            "artifact_version": "test",
            "training_dataset": "fixture",
            "final_test_metrics": {
                "roc_auc": 0.65,
                "pr_auc": 0.15,
                "brier_score": 0.07,
            },
        },
        metadata_path,
    )

    result = estimate_repayment_stress(
        application(),
        10_000,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert result["available"] is True
    assert result["calibrated_payment_difficulty_probability"] == 0.12
    assert result["historical_stress_band"] == "ELEVATED_HISTORICAL_STRESS"
    assert len(result["reason_codes"]) == 2
    assert result["role_in_decision"].startswith("Secondary historical comparison")
