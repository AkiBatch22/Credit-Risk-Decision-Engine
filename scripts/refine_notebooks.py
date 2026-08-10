"""Rebuild the five contract-aware portfolio notebooks from reusable source cells."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


SETUP = """
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")
"""


NOTEBOOKS: dict[str, list[dict[str, object]]] = {
    "01_eda.ipynb": [
        markdown("""
        # Credit Risk Decision Engine — Exploratory Data Analysis

        This notebook preserves historical Home Credit research while explicitly auditing deployment availability. `TARGET = 1` is observed payment difficulty in the source dataset, not a universal definition of default.
        """),
        code(SETUP + dedent("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from src.components.data_loader import load_training_data
        from src.components.data_validator import validate_training_data
        from src.components.feature_contract import classify_historical_feature
        """)),
        markdown("## Load, validate, and audit every historical column"),
        code("""
        df = load_training_data()
        report = validate_training_data(df)
        report.raise_for_errors()
        print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
        availability = pd.DataFrame.from_dict(
            {column: dict(zip(["classification", "new_applicant_source"], classify_historical_feature(column))) for column in df.columns},
            orient="index",
        )
        display(availability.groupby("classification").size().rename("feature_count"))
        display(availability)
        """),
        markdown("""
        `SK_ID_CURR` is a historical row identifier used only for research traceability. It is never a predictor. A deployed request receives a separate UUID-derived `application_id` after model input construction.
        """),
        markdown("## Target imbalance, missingness, and employment anomaly"),
        code("""
        target_summary = df["TARGET"].value_counts().sort_index().to_frame("applicants")
        target_summary["rate"] = df["TARGET"].value_counts(normalize=True).sort_index()
        display(target_summary)
        display(df.isna().mean().sort_values(ascending=False).head(30).rename("missing_rate").to_frame())
        print("DAYS_EMPLOYED sentinel rows:", f"{df['DAYS_EMPLOYED'].eq(365243).sum():,}")
        """),
        markdown("## Financial and categorical context"),
        code("""
        financial = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
        display(df[financial].describe(percentiles=[0.50, 0.90, 0.99, 0.999]).T)
        for column in ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE"]:
            display(df.groupby(column, dropna=False)["TARGET"].agg(default_rate="mean", applicants="size"))
        """),
        markdown("## Historical external-score research"),
        code("""
        external = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        display(df.groupby("TARGET")[external].mean())
        display(df[external].isna().mean().rename("missing_rate").to_frame())
        correlations = df[[*external, "TARGET"]].corr()["TARGET"].drop("TARGET")
        display(correlations.rename("correlation_with_target").to_frame())
        """),
        markdown("""
        Predictive usefulness does not imply deployability. The external scores are anonymized and strong offline signals, but their generation mechanism is unavailable. They remain visible in research and are excluded from the deployable application model rather than being permanently imputed.
        """),
        markdown("""
        ## Sensitive and questionable fields

        Gender, family status, education, and occupation can be protected attributes or socioeconomic proxies and are excluded from the recommended production contract. Age, household composition, housing, income type, and asset ownership can also require jurisdiction-specific necessity and fairness review. This project does not claim real-world lending suitability.
        """),
    ],
    "02_feature_engineering.ipynb": [
        markdown("""
        # Credit Risk Decision Engine — Feature Engineering

        Production and benchmark-only transformations are deliberately separate. Both are target-independent and imported from reusable modules.
        """),
        code(SETUP + dedent("""
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        from src.components.data_loader import load_training_data
        from src.components.data_preprocessor import select_production_raw_features
        from src.components.feature_contract import PRODUCTION_ENGINEERED_FEATURES, PRODUCTION_RAW_FEATURES
        from src.components.feature_engineering import BENCHMARK_ONLY_ENGINEERED_FEATURES, create_benchmark_features, create_deployable_features
        """)),
        markdown("## Apply the deployable contract and transformation"),
        code("""
        df = load_training_data()
        source_snapshot = df.copy(deep=True)
        deployable_raw = select_production_raw_features(df)
        deployable = create_deployable_features(deployable_raw)
        pd.testing.assert_frame_equal(df, source_snapshot)
        assert set(PRODUCTION_RAW_FEATURES).issubset(deployable.columns)
        assert set(PRODUCTION_ENGINEERED_FEATURES).issubset(deployable.columns)
        assert not set(BENCHMARK_ONLY_ENGINEERED_FEATURES).intersection(deployable.columns)
        assert not np.isinf(deployable.select_dtypes(include=np.number).to_numpy()).any()
        display(deployable.head())
        """),
        markdown("""
        Production engineering creates age/employment stability, affordability, repayment burden, and household-capacity features. Division by zero becomes missing, `DAYS_EMPLOYED = 365243` becomes a missing cleaned value plus an anomaly flag, and no feature uses `TARGET`.
        """),
        markdown("## Benchmark-only features remain available for research"),
        code("""
        benchmark = create_benchmark_features(df)
        assert set(BENCHMARK_ONLY_ENGINEERED_FEATURES).issubset(benchmark.columns)
        display(benchmark[list(BENCHMARK_ONLY_ENGINEERED_FEATURES)].head())
        """),
        markdown("""
        External-score aggregates, document/contact counts, and social-circle ratios are not created during production inference. Keeping a separate benchmark function prevents accidental training-serving drift while preserving valid historical experiments.
        """),
        markdown("## Standalone signal is descriptive, not model selection"),
        code("""
        def standalone_auc(frame, column):
            sample = pd.DataFrame({"feature": frame[column], "target": df["TARGET"]}).dropna()
            if sample["feature"].nunique() < 2:
                return np.nan
            auc = roc_auc_score(sample["target"], sample["feature"])
            return max(auc, 1 - auc)

        signal = pd.Series({name: standalone_auc(deployable, name) for name in PRODUCTION_ENGINEERED_FEATURES})
        display(signal.sort_values(ascending=False).rename("standalone_auc").to_frame())
        """),
    ],
    "03_baseline_model.ipynb": [
        markdown("""
        # Credit Risk Decision Engine — Deployable Logistic Regression Baseline

        This baseline uses only the explicit production contract. The untouched final test is reserved before any comparison and is not inspected here.
        """),
        code(SETUP + dedent("""
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, train_test_split

        from src.components.data_loader import load_training_data
        from src.components.data_preprocessor import build_preprocessor_for_frame, prepare_model_features, select_production_raw_features
        from src.components.model_evaluator import evaluate_probabilities
        from src.config import MODEL_SELECTION_FOLDS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
        """)),
        markdown("## Reserve final test and generate fold-fitted development predictions"),
        code("""
        df = load_training_data()
        raw = select_production_raw_features(df)
        development_x, reserved_test_x, development_y, reserved_test_y = train_test_split(
            raw, df[TARGET_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
        )
        features = prepare_model_features(development_x)
        oof = np.zeros(len(features))
        cv = StratifiedKFold(MODEL_SELECTION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, validation_idx in cv.split(features, development_y):
            fold_train = features.iloc[train_idx]
            fold_validation = features.iloc[validation_idx]
            preprocessor = build_preprocessor_for_frame(fold_train, scale_numeric=True)
            train_matrix = preprocessor.fit_transform(fold_train)
            validation_matrix = preprocessor.transform(fold_validation)
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            model.fit(train_matrix, development_y.iloc[train_idx])
            oof[validation_idx] = model.predict_proba(validation_matrix)[:, 1]
        baseline_metrics = evaluate_probabilities(development_y, oof)
        display(pd.Series({key: value for key, value in baseline_metrics.items() if key != "risk_deciles"}))
        print("Reserved final-test rows not evaluated:", f"{len(reserved_test_x):,}")
        """),
        markdown("""
        A 0.50 classification cutoff is only a diagnostic and is unsuitable as an assumed credit policy threshold for this imbalanced target. Probability ranking, calibration, risk concentration, and an explicit review-capacity policy are evaluated separately.
        """),
    ],
    "04_model_comparison.ipynb": [
        markdown("""
        # Credit Risk Decision Engine — Deployable Model Comparison

        Logistic Regression, XGBoost, and LightGBM use identical deployable features, stratified folds, and fold-fitted preprocessing. Model selection never uses the policy holdout or final test.
        """),
        code(SETUP + dedent("""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        from src.components.data_loader import load_training_data
        from src.components.data_preprocessor import select_production_raw_features
        from src.config import POLICY_SIZE, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
        from src.pipeline.training_pipeline import compare_candidate_models
        """)),
        markdown("## Create model, policy, and untouched final-test partitions"),
        code("""
        df = load_training_data()
        raw = select_production_raw_features(df)
        development_x, reserved_test_x, development_y, reserved_test_y = train_test_split(
            raw, df[TARGET_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
        )
        model_x, policy_x, model_y, policy_y = train_test_split(
            development_x, development_y, test_size=POLICY_SIZE, random_state=RANDOM_STATE, stratify=development_y
        )
        selected_name, selected_parameters, comparison = compare_candidate_models(model_x, model_y)
        comparison_table = pd.DataFrame(comparison).T
        display(comparison_table[["available", "roc_auc", "pr_auc", "brier_score", "training_seconds"]].sort_values("roc_auc", ascending=False))
        print("Selected production candidate:", selected_name)
        print("Policy and final-test rows not used for model selection:", len(policy_x), len(reserved_test_x))
        """),
        markdown("""
        The winner is selected dynamically by development out-of-fold ROC-AUC with PR-AUC as tie-breaker. A previous full-feature XGBoost result is retained only as benchmark context; no candidate is assumed to remain best after removing unavailable features.
        """),
    ],
    "05_threshold_optimization.ipynb": [
        markdown("""
        # Credit Risk Decision Engine — Deployable Calibration and Policy

        This notebook invokes the authoritative training pipeline: contract selection, development-only candidate comparison, fold-fitted calibration, policy-holdout threshold optimization, and the sole final-test evaluation.
        """),
        code(SETUP + dedent("""
        import pandas as pd

        from src.pipeline.prediction_pipeline import PredictionPipeline
        from src.pipeline.training_pipeline import run_training
        """)),
        markdown("## Full deployable training and untouched final-test evaluation"),
        code("""
        result = run_training()
        metrics = result["metrics"]
        metadata = result["metadata"]
        display(pd.Series({
            "selected_model": metadata["model_name"],
            "selected_calibration": metadata["calibration"]["selected_method"],
            "approve_threshold": metadata["policy"]["approve_threshold"],
            "reject_threshold": metadata["policy"]["reject_threshold"],
            "final_test_roc_auc": metrics["calibrated"]["roc_auc"],
            "final_test_pr_auc": metrics["calibrated"]["pr_auc"],
            "final_test_brier": metrics["calibrated"]["brier_score"],
        }))
        display(pd.DataFrame(metadata["model_selection"]["candidates"]).T)
        display(pd.DataFrame(metrics["policy_test_summary"]).T)
        """),
        markdown("## Benchmark versus deployable performance"),
        code("""
        benchmark = metrics["benchmark_reference"]["final_test_metrics"]
        deployable = metrics["calibrated"]
        comparison = pd.DataFrame({
            "Research full-feature benchmark": {key: benchmark[key] for key in ["roc_auc", "pr_auc", "brier_score", "top_10_percent_default_capture"]},
            "Deployable application model": {key: deployable[key] for key in ["roc_auc", "pr_auc", "brier_score", "top_10_percent_default_capture"]},
        })
        display(comparison)
        """),
        markdown("""
        The deployable model intentionally sacrifices performance associated with unavailable features so that every production feature can be reproduced for a new applicant. Calibration and policy results must be reported honestly even when calibration does not improve the final-test Brier score.
        """),
        markdown("## Saved-artifact inference for a completely new applicant"),
        code("""
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
        prediction = PredictionPipeline().predict(applicant, include_explanations=True)
        assert prediction.loc[0, "application_id"].startswith("APP-")
        assert "source_record_id" not in prediction
        display(prediction)
        """),
        markdown("""
        ## Limitations and handoff

        Expected loss uses illustrative LGD and requested credit as EAD. Thresholds are portfolio demonstrations, not customer-treatment rules. The sample is historical, no protected-class fairness audit or temporal validation is claimed, and reason codes describe association rather than causation.
        """),
    ],
}


def build_notebooks() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        for index, cell in enumerate(cells):
            cell["id"] = f"cell-{index:02d}"
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Credit Risk Engine (Python 3.11)",
                    "language": "python",
                    "name": "credit-risk-engine",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        destination = NOTEBOOK_DIR / filename
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    build_notebooks()
