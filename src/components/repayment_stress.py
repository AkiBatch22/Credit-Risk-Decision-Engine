"""Secondary ML estimate of repayment stress from simulator-entered fields.

The affordability engine remains authoritative for maximum-loan and EMI results.
This module only provides a calibrated historical comparison signal. It never
turns the probability into an approve/reject recommendation or a credit score.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.components.loan_simulator import LoanApplication
from src.config import (
    REPAYMENT_STRESS_METADATA_FILE,
    REPAYMENT_STRESS_MODEL_FILE,
)
from src.utils.artifacts import load_joblib, load_json


STRESS_NUMERIC_FEATURES = (
    "AGE_YEARS",
    "INCOME_STABILITY_YEARS",
    "MONTHLY_INCOME_PROXY",
    "REQUESTED_LOAN_AMOUNT",
    "PROPOSED_MONTHLY_PAYMENT",
    "LOAN_TO_ANNUAL_INCOME_RATIO",
    "PAYMENT_TO_MONTHLY_INCOME_RATIO",
)
STRESS_CATEGORICAL_FEATURES = ("INCOME_SOURCE",)
STRESS_MODEL_FEATURES = (*STRESS_NUMERIC_FEATURES, *STRESS_CATEGORICAL_FEATURES)

HOME_CREDIT_TO_INCOME_SOURCE = {
    "Working": "Salaried",
    "State servant": "Salaried",
    "Commercial associate": "Self-employed",
    "Businessman": "Business owner",
    "Pensioner": "Retired / pension income",
}

FEATURE_LABELS = {
    "AGE_YEARS": "Applicant age",
    "INCOME_STABILITY_YEARS": "Income-source stability",
    "MONTHLY_INCOME_PROXY": "Submitted monthly income",
    "REQUESTED_LOAN_AMOUNT": "Requested loan amount",
    "PROPOSED_MONTHLY_PAYMENT": "Proposed monthly EMI",
    "LOAN_TO_ANNUAL_INCOME_RATIO": "Loan-to-annual-income ratio",
    "PAYMENT_TO_MONTHLY_INCOME_RATIO": "EMI-to-monthly-income ratio",
    "INCOME_SOURCE": "Income source",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator / denominator.replace(0, np.nan)
    return values.replace([np.inf, -np.inf], np.nan)


def historical_to_stress_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the exact inference feature contract from Home Credit rows."""

    required = {
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "NAME_INCOME_TYPE",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"historical data is missing repayment-stress fields: {missing}")

    annual_income = pd.to_numeric(dataframe["AMT_INCOME_TOTAL"], errors="coerce")
    requested = pd.to_numeric(dataframe["AMT_CREDIT"], errors="coerce")
    annual_payment = pd.to_numeric(dataframe["AMT_ANNUITY"], errors="coerce")
    employment_days = pd.to_numeric(dataframe["DAYS_EMPLOYED"], errors="coerce")
    employment_days = employment_days.mask(employment_days == 365243, np.nan)

    features = pd.DataFrame(index=dataframe.index)
    features["AGE_YEARS"] = (
        pd.to_numeric(dataframe["DAYS_BIRTH"], errors="coerce").abs() / 365.25
    )
    features["INCOME_STABILITY_YEARS"] = employment_days.abs() / 365.25
    features["MONTHLY_INCOME_PROXY"] = annual_income / 12.0
    features["REQUESTED_LOAN_AMOUNT"] = requested
    features["PROPOSED_MONTHLY_PAYMENT"] = annual_payment / 12.0
    features["LOAN_TO_ANNUAL_INCOME_RATIO"] = _safe_ratio(requested, annual_income)
    features["PAYMENT_TO_MONTHLY_INCOME_RATIO"] = _safe_ratio(
        annual_payment, annual_income
    )
    features["INCOME_SOURCE"] = (
        dataframe["NAME_INCOME_TYPE"]
        .map(HOME_CREDIT_TO_INCOME_SOURCE)
        .fillna("Other regular income")
    )
    return features.loc[:, STRESS_MODEL_FEATURES]


def application_to_stress_features(
    application: LoanApplication,
    proposed_monthly_payment: float,
) -> pd.DataFrame:
    """Create one inference row using only values entered in the simulator."""

    annual_income = application.monthly_net_income * 12.0
    return pd.DataFrame(
        [
            {
                "AGE_YEARS": float(application.age),
                "INCOME_STABILITY_YEARS": float(application.income_stability_years),
                "MONTHLY_INCOME_PROXY": float(application.monthly_net_income),
                "REQUESTED_LOAN_AMOUNT": float(application.requested_loan_amount),
                "PROPOSED_MONTHLY_PAYMENT": float(proposed_monthly_payment),
                "LOAN_TO_ANNUAL_INCOME_RATIO": (
                    float(application.requested_loan_amount) / annual_income
                ),
                "PAYMENT_TO_MONTHLY_INCOME_RATIO": (
                    float(proposed_monthly_payment) / application.monthly_net_income
                ),
                "INCOME_SOURCE": application.employment_type,
            }
        ],
        columns=STRESS_MODEL_FEATURES,
    )


def stress_band(probability: float) -> str:
    """Return a descriptive probability band, not a lending recommendation."""

    if probability < 0.05:
        return "LOWER_HISTORICAL_STRESS"
    if probability < 0.10:
        return "MODERATE_HISTORICAL_STRESS"
    if probability < 0.20:
        return "ELEVATED_HISTORICAL_STRESS"
    return "HIGHER_HISTORICAL_STRESS"


def _base_feature_name(transformed_name: str) -> str:
    stripped = transformed_name.split("__", 1)[-1]
    for feature in STRESS_CATEGORICAL_FEATURES:
        if stripped.startswith(f"{feature}_"):
            return feature
    return stripped


def _local_contributions(artifact: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    transformed = artifact["preprocessor"].transform(features)
    model = artifact["base_model"]
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        contributions = model.get_booster().predict(
            xgb.DMatrix(transformed), pred_contribs=True
        )[0, :-1]
    elif hasattr(model, "coef_"):
        dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        contributions = np.asarray(dense)[0] * np.asarray(model.coef_)[0]
    else:
        return np.array([], dtype=float)
    return np.asarray(contributions, dtype=float)


def _reason_codes(
    artifact: dict[str, Any], features: pd.DataFrame, *, top_n: int = 3
) -> list[dict[str, Any]]:
    values = _local_contributions(artifact, features)
    names = list(artifact.get("transformed_feature_names", []))
    if len(values) != len(names):
        return []

    grouped: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        base = _base_feature_name(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)

    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    reasons = []
    for feature, contribution in ranked[:top_n]:
        reasons.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "direction": "increases_stress" if contribution > 0 else "reduces_stress",
                "model_contribution": round(contribution, 6),
            }
        )
    return reasons


@lru_cache(maxsize=4)
def load_repayment_stress_artifacts(
    model_path: str = str(REPAYMENT_STRESS_MODEL_FILE),
    metadata_path: str = str(REPAYMENT_STRESS_METADATA_FILE),
) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_joblib(model_path), load_json(metadata_path)


def repayment_stress_available(
    model_path: str | Path = REPAYMENT_STRESS_MODEL_FILE,
    metadata_path: str | Path = REPAYMENT_STRESS_METADATA_FILE,
) -> bool:
    return Path(model_path).is_file() and Path(metadata_path).is_file()


def estimate_repayment_stress(
    application: LoanApplication,
    proposed_monthly_payment: float,
    *,
    model_path: str | Path = REPAYMENT_STRESS_MODEL_FILE,
    metadata_path: str | Path = REPAYMENT_STRESS_METADATA_FILE,
) -> dict[str, Any]:
    """Return a calibrated historical stress estimate or an explicit unavailable state."""

    if not repayment_stress_available(model_path, metadata_path):
        return {
            "available": False,
            "unavailable_reason": (
                "Repayment-stress artifacts are not installed. Run the dedicated full-data "
                "training pipeline to enable this optional historical comparison."
            ),
        }

    artifact, metadata = load_repayment_stress_artifacts(
        str(model_path), str(metadata_path)
    )
    features = application_to_stress_features(application, proposed_monthly_payment)
    transformed = artifact["preprocessor"].transform(features)
    probability = float(artifact["calibrated_model"].predict_proba(transformed)[0, 1])
    return {
        "available": True,
        "calibrated_payment_difficulty_probability": round(probability, 6),
        "historical_stress_band": stress_band(probability),
        "reason_codes": _reason_codes(artifact, features),
        "model_name": metadata["model_name"],
        "calibration_method": metadata["calibration_method"],
        "artifact_version": metadata["artifact_version"],
        "training_dataset": metadata["training_dataset"],
        "final_test_metrics": metadata["final_test_metrics"],
        "role_in_decision": (
            "Secondary historical comparison only; it does not set eligibility, maximum "
            "loan amount, price, or an approve/reject outcome."
        ),
        "data_scope_note": (
            "Inference uses only this form's entries. Training used historical Home Credit "
            "outcomes, whose population and income definitions may differ from this scenario."
        ),
    }
