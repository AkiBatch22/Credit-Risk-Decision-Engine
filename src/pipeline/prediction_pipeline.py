"""Artifact-only inference pipeline; no training occurs here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.components.data_preprocessor import prepare_model_features
from src.components.data_validator import validate_prediction_data
from src.components.explainability import explain_xgboost_predictions
from src.components.feature_contract import (
    PRODUCTION_RAW_FEATURES,
    generate_application_id,
    normalize_application_frame,
)
from src.components.risk_policy import RiskPolicy, apply_risk_policy
from src.config import (
    CALIBRATED_MODEL_FILE,
    EXPOSURE_COLUMN,
    ID_COLUMN,
    METADATA_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
)
from src.utils.artifacts import load_joblib, load_json


class PredictionPipeline:
    def __init__(
        self,
        *,
        preprocessor_path: str | Path = PREPROCESSOR_FILE,
        model_path: str | Path = MODEL_FILE,
        calibrated_model_path: str | Path = CALIBRATED_MODEL_FILE,
        metadata_path: str | Path = METADATA_FILE,
    ) -> None:
        self.preprocessor = load_joblib(preprocessor_path)
        self.model = load_joblib(model_path)
        self.calibrated_model = load_joblib(calibrated_model_path)
        self.metadata = load_json(metadata_path)
        self.raw_feature_columns = list(self.metadata["raw_feature_columns"])
        if set(self.raw_feature_columns) != set(PRODUCTION_RAW_FEATURES):
            raise ValueError(
                "Loaded artifacts do not use the deployable feature contract. "
                "Run the production training pipeline to regenerate artifacts."
            )
        self.transformed_feature_names = list(self.metadata["transformed_feature_names"])
        policy_data = self.metadata["policy"]
        self.policy = RiskPolicy(
            approve_threshold=float(policy_data["approve_threshold"]),
            reject_threshold=float(policy_data["reject_threshold"]),
            risk_band_thresholds=tuple(policy_data["risk_band_thresholds"]),
            lgd=float(policy_data["lgd"]),
            net_margin_rate=float(policy_data["net_margin_rate"]),
            assumptions_are_illustrative=bool(policy_data.get("assumptions_are_illustrative", True)),
        )

    @staticmethod
    def _to_frame(applicants: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(applicants, pd.DataFrame):
            return applicants.copy(deep=True)
        if isinstance(applicants, dict):
            return pd.DataFrame([applicants])
        raise TypeError("applicants must be a dictionary or pandas DataFrame")

    def predict(
        self,
        applicants: dict[str, Any] | pd.DataFrame,
        *,
        exposure: float | list[float] | np.ndarray | None = None,
        include_explanations: bool = False,
        top_n_reasons: int = 5,
    ) -> pd.DataFrame:
        original = self._to_frame(applicants)
        source_record_ids = None
        if "source_record_id" in original:
            source_record_ids = original["source_record_id"].copy()
        elif ID_COLUMN in original:
            source_record_ids = original[ID_COLUMN].copy()

        normalized = normalize_application_frame(original)
        report = validate_prediction_data(normalized, self.raw_feature_columns)
        report.raise_for_errors()

        aligned = normalized.reindex(columns=self.raw_feature_columns)
        model_frame = prepare_model_features(aligned)
        expected_model_columns = list(self.metadata["model_feature_columns"])
        model_frame = model_frame.reindex(columns=expected_model_columns)
        transformed = self.preprocessor.transform(model_frame)
        if self.metadata.get("calibrated_model_input") == "model_features":
            probabilities = self.calibrated_model.predict_proba(model_frame)[:, 1]
        else:  # pragma: no cover - legacy artifact compatibility guard
            probabilities = self.calibrated_model.predict_proba(transformed)[:, 1]

        if exposure is None and EXPOSURE_COLUMN in normalized:
            exposure_values: Any = pd.to_numeric(
                normalized[EXPOSURE_COLUMN], errors="coerce"
            ).to_numpy()
        elif exposure is None:
            exposure_values = None
        else:
            exposure_values = np.atleast_1d(exposure)
            if len(exposure_values) == 1 and len(original) > 1:
                exposure_values = np.repeat(exposure_values, len(original))

        output = apply_risk_policy(probabilities, exposure=exposure_values, policy=self.policy)
        output.insert(0, "application_id", [generate_application_id() for _ in range(len(output))])
        if source_record_ids is not None:
            output.insert(1, "source_record_id", source_record_ids.to_numpy())
        output["missing_input_feature_count"] = normalized.isna().sum(axis=1).to_numpy()

        if include_explanations:
            try:
                reasons = explain_xgboost_predictions(
                    self.model,
                    transformed,
                    self.transformed_feature_names,
                    top_n=top_n_reasons,
                )
            except (RuntimeError, TypeError, ImportError):
                reasons = [[] for _ in range(len(output))]
            output["top_risk_reasons"] = reasons
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Applicant JSON object or CSV file")
    parser.add_argument("--explain", action="store_true")
    arguments = parser.parse_args()
    pipeline = PredictionPipeline()
    if arguments.input.suffix.lower() == ".csv":
        applicants: dict[str, Any] | pd.DataFrame = pd.read_csv(arguments.input)
    else:
        with arguments.input.open("r", encoding="utf-8") as file:
            applicants = json.load(file)
    print(
        pipeline.predict(applicants, include_explanations=arguments.explain).to_json(
            orient="records", indent=2
        )
    )


if __name__ == "__main__":
    main()
