"""End-to-end, leakage-safe model training and artifact generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.components.data_loader import load_training_data
from src.components.data_preprocessor import (
    build_preprocessor_for_frame,
    prepare_model_features,
    select_production_raw_features,
)
from src.components.data_validator import validate_training_data
from src.components.feature_contract import (
    EXTERNAL_UNAVAILABLE_FEATURES,
    HISTORICAL_IDENTIFIER_FEATURES,
    PRODUCTION_RAW_FEATURES,
    classify_historical_feature,
    contract_as_dict,
)
from src.components.model_calibrator import calibrate_model
from src.components.model_evaluator import evaluate_probabilities
from src.components.model_trainer import build_model, train_model
from src.components.risk_policy import RiskPolicy, apply_risk_policy, optimize_policy
from src.config import (
    BENCHMARK_REFERENCE_FILE,
    CALIBRATED_MODEL_FILE,
    CALIBRATION_FOLDS,
    CALIBRATION_METHODS,
    CANDIDATE_APPROVE_THRESHOLDS,
    CANDIDATE_REJECT_THRESHOLDS,
    EXPOSURE_COLUMN,
    ID_COLUMN,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_PARAMETERS,
    MODEL_SELECTION_FOLDS,
    POLICY_SIZE,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_DATA_FILE,
)
from src.exception import CreditRiskError
from src.logger import get_logger
from src.utils.artifacts import load_json, save_joblib, save_json


CANDIDATE_MODELS: dict[str, tuple[str, dict[str, Any]]] = {
    "Logistic Regression": ("logistic_regression", {}),
    "XGBoost": ("xgboost", MODEL_PARAMETERS),
    "LightGBM": (
        "lightgbm",
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        },
    ),
}


def _uses_scaled_numeric(model_name: str) -> bool:
    return model_name == "logistic_regression"


def compare_candidate_models(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Development-only OOF comparison with fold-fitted preprocessing."""

    model_frame = prepare_model_features(features)
    folds = StratifiedKFold(
        n_splits=MODEL_SELECTION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    comparison: dict[str, dict[str, Any]] = {}
    specifications: dict[str, tuple[str, dict[str, Any]]] = {}
    for display_name, (model_name, parameters) in CANDIDATE_MODELS.items():
        probabilities = np.zeros(len(model_frame), dtype=float)
        started = time.perf_counter()
        try:
            for train_indices, validation_indices in folds.split(model_frame, target):
                fold_train = model_frame.iloc[train_indices]
                fold_validation = model_frame.iloc[validation_indices]
                preprocessor = build_preprocessor_for_frame(
                    fold_train,
                    scale_numeric=_uses_scaled_numeric(model_name),
                )
                train_matrix = preprocessor.fit_transform(fold_train)
                validation_matrix = preprocessor.transform(fold_validation)
                estimator = build_model(model_name, parameters)
                estimator.fit(train_matrix, target.iloc[train_indices])
                probabilities[validation_indices] = estimator.predict_proba(validation_matrix)[:, 1]
        except ImportError as error:
            comparison[display_name] = {"available": False, "reason": str(error)}
            continue
        metrics = evaluate_probabilities(target, probabilities, include_deciles=False)
        comparison[display_name] = {
            "available": True,
            "model_name": model_name,
            "parameters": parameters,
            "training_seconds": round(time.perf_counter() - started, 6),
            **metrics,
        }
        specifications[display_name] = (model_name, parameters)

    available = [name for name, result in comparison.items() if result.get("available")]
    if not available:
        raise RuntimeError("no production candidate model is available")
    winner = max(
        available,
        key=lambda name: (comparison[name]["roc_auc"], comparison[name]["pr_auc"]),
    )
    selected_name, selected_parameters = specifications[winner]
    return selected_name, selected_parameters, comparison


def _load_or_preserve_benchmark_reference() -> dict[str, Any] | None:
    """Keep the previously measured full-feature artifact metrics as research context."""

    if BENCHMARK_REFERENCE_FILE.is_file():
        return load_json(BENCHMARK_REFERENCE_FILE)
    if not METRICS_FILE.is_file() or not METADATA_FILE.is_file():
        return None
    previous_metrics = load_json(METRICS_FILE)
    previous_metadata = load_json(METADATA_FILE)
    if previous_metadata.get("feature_profile") == "deployable_application":
        return previous_metrics.get("benchmark_reference")
    reference = {
        "profile": "research_full_feature_benchmark",
        "model_name": previous_metadata.get("model_name"),
        "calibration": previous_metadata.get("calibration"),
        "policy": previous_metadata.get("policy"),
        "split": previous_metadata.get("split"),
        "final_test_metrics": previous_metrics.get("calibrated"),
        "source_artifact_version": previous_metadata.get("artifact_version"),
    }
    save_json(reference, BENCHMARK_REFERENCE_FILE)
    return reference


def _policy_test_summary(
    target: pd.Series, probabilities: np.ndarray, exposure: pd.Series, policy: RiskPolicy
) -> dict[str, Any]:
    decisions = apply_risk_policy(probabilities, exposure=exposure, policy=policy)
    decisions["target"] = np.asarray(target)
    summary: dict[str, Any] = {}
    for recommendation, group in decisions.groupby("recommendation"):
        summary[str(recommendation)] = {
            "applicants": int(len(group)),
            "rate": float(len(group) / len(decisions)),
            "observed_default_rate": float(group["target"].mean()),
            "mean_predicted_probability": float(group["probability"].mean()),
            "total_expected_loss": float(group["expected_loss"].sum()),
        }
    return summary


def run_training(
    data_path: str | Path = TRAIN_DATA_FILE,
    *,
    max_rows: int | None = None,
    model_name: str = MODEL_NAME,
    model_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, calibrate, select a policy, evaluate once on test, and persist artifacts."""

    logger = get_logger("credit_risk.training")
    try:
        logger.info("Loading training data from %s", data_path)
        dataframe = load_training_data(data_path)
        if max_rows is not None and max_rows < len(dataframe):
            dataframe = dataframe.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
            logger.warning("Using a development sample of %s rows", len(dataframe))

        report = validate_training_data(dataframe)
        report.raise_for_errors()
        logger.info("Data validation passed with %s diagnostics", len(report.diagnostics))

        benchmark_reference = _load_or_preserve_benchmark_reference()
        raw_features = select_production_raw_features(dataframe)
        target = dataframe[TARGET_COLUMN]
        development_x, test_x, development_y, test_y = train_test_split(
            raw_features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        model_x, policy_x, model_y, policy_y = train_test_split(
            development_x,
            development_y,
            test_size=POLICY_SIZE,
            random_state=RANDOM_STATE,
            stratify=development_y,
        )
        logger.info(
            "Split rows: model=%s policy=%s final_test=%s",
            len(model_x),
            len(policy_x),
            len(test_x),
        )

        selected_model_name, selected_parameters, model_comparison = compare_candidate_models(
            model_x,
            model_y,
        )
        logger.info("Selected %s from deployable-feature OOF comparison", selected_model_name)

        model_frame = prepare_model_features(model_x)
        policy_frame = prepare_model_features(policy_x)
        test_frame = prepare_model_features(test_x)
        scale_numeric = _uses_scaled_numeric(selected_model_name)
        preprocessor = build_preprocessor_for_frame(model_frame, scale_numeric=scale_numeric)
        model_matrix = preprocessor.fit_transform(model_frame)
        policy_matrix = preprocessor.transform(policy_frame)
        test_matrix = preprocessor.transform(test_frame)
        logger.info("Preprocessing fitted on model partition only")

        parameters = model_parameters or selected_parameters
        production_model_name = model_name if model_parameters is not None else selected_model_name
        base_model, training_metadata = train_model(
            build_model(production_model_name, parameters), model_matrix, model_y
        )
        uncalibrated_test_probabilities = base_model.predict_proba(test_matrix)[:, 1]

        calibration_candidates = {}
        calibration_selection = {}
        for method in CALIBRATION_METHODS:
            calibration_estimator = SklearnPipeline(
                [
                    (
                        "preprocessor",
                        build_preprocessor_for_frame(model_frame, scale_numeric=scale_numeric),
                    ),
                    ("model", build_model(production_model_name, parameters)),
                ]
            )
            candidate = calibrate_model(
                calibration_estimator,
                model_frame,
                model_y,
                method=method,
                cv=CALIBRATION_FOLDS,
            )
            candidate_policy_probabilities = candidate.predict_proba(policy_frame)[:, 1]
            calibration_candidates[method] = candidate
            calibration_selection[method] = evaluate_probabilities(
                policy_y, candidate_policy_probabilities, include_deciles=False
            )
        selected_calibration_method = min(
            CALIBRATION_METHODS,
            key=lambda method: calibration_selection[method]["brier_score"],
        )
        calibrated_model = calibration_candidates[selected_calibration_method]
        policy_probabilities = calibrated_model.predict_proba(policy_frame)[:, 1]
        test_probabilities = calibrated_model.predict_proba(test_frame)[:, 1]
        logger.info(
            "Selected %s calibration on policy-holdout Brier score",
            selected_calibration_method,
        )

        policy, policy_search = optimize_policy(
            policy_y,
            policy_probabilities,
            policy_x[EXPOSURE_COLUMN],
            CANDIDATE_APPROVE_THRESHOLDS,
            CANDIDATE_REJECT_THRESHOLDS,
        )
        logger.info(
            "Selected illustrative policy thresholds approve=%.3f reject=%.3f",
            policy.approve_threshold,
            policy.reject_threshold,
        )

        uncalibrated_metrics = evaluate_probabilities(test_y, uncalibrated_test_probabilities)
        calibrated_metrics = evaluate_probabilities(test_y, test_probabilities)
        metrics = {
            "feature_profile": "deployable_application",
            "evaluation_partition": "untouched_final_test",
            "model_selection_on_model_partition": model_comparison,
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
            "calibration_brier_delta": (
                calibrated_metrics["brier_score"] - uncalibrated_metrics["brier_score"]
            ),
            "calibration_selection_on_policy_holdout": calibration_selection,
            "policy_test_summary": _policy_test_summary(
                test_y, test_probabilities, test_x[EXPOSURE_COLUMN], policy
            ),
            "benchmark_reference": benchmark_reference,
        }

        transformed_names = preprocessor.get_feature_names_out().tolist()
        metadata = {
            "artifact_version": "2.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_profile": "deployable_application",
            "model_name": production_model_name,
            "model_parameters": parameters,
            "model_selection": {
                "source": "training-pipeline development-only out-of-fold comparison",
                "cv_folds": MODEL_SELECTION_FOLDS,
                "final_test_used_for_selection": False,
                "primary_metric": "roc_auc",
                "tie_breaker": "pr_auc",
                "candidates": model_comparison,
            },
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "exposure_column": EXPOSURE_COLUMN,
            "raw_feature_columns": list(PRODUCTION_RAW_FEATURES),
            "raw_feature_dtypes": {column: str(dtype) for column, dtype in raw_features.dtypes.items()},
            "model_feature_columns": model_frame.columns.tolist(),
            "transformed_feature_names": transformed_names,
            "calibrated_model_input": "model_features",
            "deployable_feature_contract": contract_as_dict(),
            "production_exclusions": {
                "external_unavailable": list(EXTERNAL_UNAVAILABLE_FEATURES),
                "historical_identifiers": list(HISTORICAL_IDENTIFIER_FEATURES),
            },
            "feature_availability_audit": {
                column: {
                    "classification": classify_historical_feature(column)[0],
                    "inference_source": classify_historical_feature(column)[1],
                }
                for column in dataframe.columns
            },
            "policy": policy.to_dict(),
            "calibration": {
                "candidate_methods": list(CALIBRATION_METHODS),
                "selected_method": selected_calibration_method,
                "selection_metric": "policy-holdout Brier score",
                "cv_folds": CALIBRATION_FOLDS,
            },
            "split": {
                "random_state": RANDOM_STATE,
                "model_rows": len(model_x),
                "policy_rows": len(policy_x),
                "final_test_rows": len(test_x),
            },
            "training": training_metadata,
            "validation_report": report.to_dict(),
            "policy_search_top_candidates": policy_search.head(20).to_dict(orient="records"),
            "benchmark_reference": benchmark_reference,
            "disclaimer": "Educational prototype; thresholds and economics are illustrative, not lending policy.",
        }

        save_joblib(preprocessor, PREPROCESSOR_FILE)
        save_joblib(base_model, MODEL_FILE)
        save_joblib(calibrated_model, CALIBRATED_MODEL_FILE)
        save_json(metrics, METRICS_FILE)
        save_json(metadata, METADATA_FILE)
        logger.info("Saved model artifacts and final-test metrics under artifacts/")
        return {"metrics": metrics, "metadata": metadata}
    except Exception as error:
        logger.exception("Training pipeline failed")
        raise CreditRiskError.from_exception("Training pipeline failed", error) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=TRAIN_DATA_FILE)
    parser.add_argument("--max-rows", type=int, default=None, help="Development-only row cap")
    arguments = parser.parse_args()
    result = run_training(arguments.data_path, max_rows=arguments.max_rows)
    print(
        f"Final test ROC-AUC: {result['metrics']['calibrated']['roc_auc']:.4f} | "
        f"PR-AUC: {result['metrics']['calibrated']['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
    BENCHMARK_REFERENCE_FILE,
