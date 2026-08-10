from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from src.components.data_preprocessor import build_preprocessor_for_frame, prepare_model_features
from src.components.feature_contract import APPLICATION_ID_PATTERN, PRODUCTION_RAW_FEATURES
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.artifacts import save_joblib, save_json


@pytest.fixture
def tiny_prediction_pipeline(tmp_path: Path, synthetic_applications) -> PredictionPipeline:
    raw = synthetic_applications.loc[:, PRODUCTION_RAW_FEATURES]
    model_frame = prepare_model_features(raw)
    preprocessor = build_preprocessor_for_frame(model_frame, scale_numeric=True)
    matrix = preprocessor.fit_transform(model_frame)
    model = LogisticRegression(max_iter=500, random_state=42).fit(matrix, synthetic_applications["TARGET"])

    preprocessor_path = save_joblib(preprocessor, tmp_path / "preprocessor.joblib")
    model_path = save_joblib(model, tmp_path / "model.joblib")
    calibrated_path = save_joblib(model, tmp_path / "calibrated.joblib")
    metadata_path = save_json(
        {
            "raw_feature_columns": raw.columns.tolist(),
            "model_feature_columns": model_frame.columns.tolist(),
            "transformed_feature_names": preprocessor.get_feature_names_out().tolist(),
            "policy": {
                "approve_threshold": 0.08,
                "reject_threshold": 0.15,
                "risk_band_thresholds": [0.05, 0.10, 0.20],
                "lgd": 0.60,
                "net_margin_rate": 0.08,
                "assumptions_are_illustrative": True,
            },
        },
        tmp_path / "metadata.json",
    )
    return PredictionPipeline(
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        calibrated_model_path=calibrated_path,
        metadata_path=metadata_path,
    )


def test_prediction_pipeline_scores_without_retraining(tiny_prediction_pipeline, synthetic_applications):
    applicant = synthetic_applications.drop(columns="TARGET").iloc[[0]]
    result = tiny_prediction_pipeline.predict(applicant, include_explanations=True)
    assert len(result) == 1
    assert APPLICATION_ID_PATTERN.fullmatch(result.loc[0, "application_id"])
    assert result.loc[0, "source_record_id"] == applicant.iloc[0]["SK_ID_CURR"]
    assert 0 <= result.loc[0, "probability"] <= 1
    assert result.loc[0, "recommendation"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}
    assert result.loc[0, "expected_loss"] >= 0
    assert isinstance(result.loc[0, "top_risk_reasons"], list)


def test_new_applicant_needs_no_identifier_or_external_scores(
    tiny_prediction_pipeline,
):
    applicant = {
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
        "owns_car": "No",
        "owns_property": "Yes",
    }
    result = tiny_prediction_pipeline.predict(applicant)
    assert APPLICATION_ID_PATTERN.fullmatch(result.loc[0, "application_id"])
    assert "source_record_id" not in result


def test_identifiers_and_external_scores_do_not_affect_prediction(
    tiny_prediction_pipeline,
    synthetic_applications,
):
    first = synthetic_applications.drop(columns="TARGET").iloc[[1]].copy()
    second = first.copy()
    second["SK_ID_CURR"] = 999999
    second["EXT_SOURCE_1"] = 0.001
    second["EXT_SOURCE_2"] = 0.999
    second["EXT_SOURCE_3"] = 0.500
    first_result = tiny_prediction_pipeline.predict(first)
    second_result = tiny_prediction_pipeline.predict(second)
    assert first_result.loc[0, "probability"] == second_result.loc[0, "probability"]
    assert first_result.loc[0, "application_id"] != second_result.loc[0, "application_id"]


def test_prediction_pipeline_rejects_unrecognized_schema(tiny_prediction_pipeline):
    with pytest.raises(ValueError, match="no fields recognized"):
        tiny_prediction_pipeline.predict({"unknown": 1})
