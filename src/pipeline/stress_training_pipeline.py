"""Train the optional, form-aligned repayment-stress model.

The split is explicit: model comparison and fitting use only the training
partition, probability calibration uses only the calibration partition, and the
final test partition is evaluated once after every modelling choice is fixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.components.data_loader import load_training_data
from src.components.data_preprocessor import build_preprocessor_for_frame
from src.components.data_validator import validate_training_data
from src.components.model_evaluator import evaluate_probabilities
from src.components.model_trainer import build_model
from src.components.repayment_stress import historical_to_stress_features
from src.config import (
    RANDOM_STATE,
    REPAYMENT_STRESS_METADATA_FILE,
    REPAYMENT_STRESS_METRICS_FILE,
    REPAYMENT_STRESS_MODEL_FILE,
    TARGET_COLUMN,
    TRAIN_DATA_FILE,
)
from src.exception import CreditRiskError
from src.logger import get_logger
from src.utils.artifacts import save_joblib, save_json


FINAL_TEST_SIZE = 0.20
CALIBRATION_SIZE_WITHIN_DEVELOPMENT = 0.20
MODEL_SELECTION_FOLDS = 3
CALIBRATION_METHOD = "sigmoid"

CANDIDATES: dict[str, tuple[str, dict[str, Any], bool]] = {
    "Logistic Regression": (
        "logistic_regression",
        {"class_weight": "balanced", "max_iter": 1500},
        True,
    ),
    "XGBoost": (
        "xgboost",
        {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        False,
    ),
}


def _compare_models(
    features: pd.DataFrame, target: pd.Series
) -> tuple[str, dict[str, Any], bool, dict[str, dict[str, Any]]]:
    folds = StratifiedKFold(
        n_splits=MODEL_SELECTION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    comparison: dict[str, dict[str, Any]] = {}
    for display_name, (model_name, parameters, scale_numeric) in CANDIDATES.items():
        oof = np.zeros(len(features), dtype=float)
        started = time.perf_counter()
        for train_indices, validation_indices in folds.split(features, target):
            fold_train = features.iloc[train_indices]
            fold_validation = features.iloc[validation_indices]
            preprocessor = build_preprocessor_for_frame(
                fold_train, scale_numeric=scale_numeric
            )
            train_matrix = preprocessor.fit_transform(fold_train)
            validation_matrix = preprocessor.transform(fold_validation)
            model = build_model(model_name, parameters)
            model.fit(train_matrix, target.iloc[train_indices])
            oof[validation_indices] = model.predict_proba(validation_matrix)[:, 1]
        comparison[display_name] = {
            "model_name": model_name,
            "parameters": parameters,
            "scale_numeric": scale_numeric,
            "training_seconds": round(time.perf_counter() - started, 6),
            **evaluate_probabilities(target, oof, include_deciles=False),
        }

    winner = max(
        comparison,
        key=lambda name: (
            comparison[name]["roc_auc"],
            comparison[name]["pr_auc"],
        ),
    )
    model_name, parameters, scale_numeric = CANDIDATES[winner]
    return model_name, parameters, scale_numeric, comparison


def run_stress_training(
    data_path: str | Path = TRAIN_DATA_FILE,
) -> dict[str, Any]:
    logger = get_logger("credit_risk.repayment_stress_training")
    try:
        dataframe = load_training_data(data_path)
        validation = validate_training_data(dataframe)
        validation.raise_for_errors()
        features = historical_to_stress_features(dataframe)
        target = dataframe[TARGET_COLUMN].astype(int)

        development_x, final_test_x, development_y, final_test_y = train_test_split(
            features,
            target,
            test_size=FINAL_TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        train_x, calibration_x, train_y, calibration_y = train_test_split(
            development_x,
            development_y,
            test_size=CALIBRATION_SIZE_WITHIN_DEVELOPMENT,
            random_state=RANDOM_STATE,
            stratify=development_y,
        )
        logger.info(
            "Repayment-stress split rows: train=%s calibration=%s final_test=%s",
            len(train_x),
            len(calibration_x),
            len(final_test_x),
        )

        model_name, parameters, scale_numeric, comparison = _compare_models(
            train_x, train_y
        )
        preprocessor = build_preprocessor_for_frame(
            train_x, scale_numeric=scale_numeric
        )
        train_matrix = preprocessor.fit_transform(train_x)
        calibration_matrix = preprocessor.transform(calibration_x)
        final_test_matrix = preprocessor.transform(final_test_x)

        started = time.perf_counter()
        base_model = build_model(model_name, parameters)
        base_model.fit(train_matrix, train_y)
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=CALIBRATION_METHOD,
            cv=None,
        )
        calibrated_model.fit(calibration_matrix, calibration_y)
        training_seconds = round(time.perf_counter() - started, 6)

        uncalibrated_probability = base_model.predict_proba(final_test_matrix)[:, 1]
        calibrated_probability = calibrated_model.predict_proba(final_test_matrix)[:, 1]
        uncalibrated_metrics = evaluate_probabilities(
            final_test_y, uncalibrated_probability
        )
        calibrated_metrics = evaluate_probabilities(
            final_test_y, calibrated_probability
        )

        artifact = {
            "preprocessor": preprocessor,
            "base_model": base_model,
            "calibrated_model": calibrated_model,
            "transformed_feature_names": preprocessor.get_feature_names_out().tolist(),
        }
        metadata = {
            "artifact_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_dataset": "Home Credit application_train.csv",
            "target_definition": "Home Credit TARGET: historical payment difficulty",
            "model_name": model_name,
            "model_parameters": parameters,
            "calibration_method": CALIBRATION_METHOD,
            "feature_profile": "simulator_form_aligned",
            "raw_inference_inputs": [
                "age",
                "employment_type",
                "income_stability_years",
                "monthly_net_income",
                "requested_loan_amount",
                "preferred_plan_monthly_emi",
            ],
            "split": {
                "strategy": "stratified train/calibration/final-test",
                "random_state": RANDOM_STATE,
                "train_rows": len(train_x),
                "calibration_rows": len(calibration_x),
                "final_test_rows": len(final_test_x),
                "final_test_used_for_selection": False,
            },
            "model_selection": {
                "strategy": "training-partition-only out-of-fold comparison",
                "folds": MODEL_SELECTION_FOLDS,
                "primary_metric": "roc_auc",
                "tie_breaker": "pr_auc",
                "candidates": comparison,
            },
            "final_test_metrics": {
                "roc_auc": calibrated_metrics["roc_auc"],
                "pr_auc": calibrated_metrics["pr_auc"],
                "brier_score": calibrated_metrics["brier_score"],
            },
            "limitations": [
                "The training population may not represent the simulator user or a bank portfolio.",
                "Submitted monthly net income is aligned to a monthlyized historical income proxy.",
                "Product type, expenses, and existing debts are not ML features because equivalent historical fields are unavailable.",
                "The probability is a secondary comparison signal, not an eligibility or pricing rule.",
            ],
        }
        metrics = {
            "evaluation_partition": "untouched_final_test",
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
            "model_selection_on_training_partition": comparison,
            "training_seconds_after_selection": training_seconds,
        }

        save_joblib(artifact, REPAYMENT_STRESS_MODEL_FILE)
        save_json(metadata, REPAYMENT_STRESS_METADATA_FILE)
        save_json(metrics, REPAYMENT_STRESS_METRICS_FILE)
        logger.info("Saved repayment-stress artifacts and final-test metrics")
        return {"metadata": metadata, "metrics": metrics}
    except Exception as error:
        logger.exception("Repayment-stress training failed")
        raise CreditRiskError.from_exception(
            "Repayment-stress training failed", error
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=TRAIN_DATA_FILE)
    arguments = parser.parse_args()
    result = run_stress_training(arguments.data_path)
    final = result["metrics"]["calibrated"]
    print(
        f"Repayment-stress final test ROC-AUC: {final['roc_auc']:.4f} | "
        f"PR-AUC: {final['pr_auc']:.4f} | Brier: {final['brier_score']:.5f}"
    )


if __name__ == "__main__":
    main()
