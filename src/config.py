"""Central configuration for the credit-risk decision engine."""

from __future__ import annotations

from pathlib import Path

from src.components.feature_contract import PRODUCTION_RAW_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

TRAIN_DATA_FILE = RAW_DATA_DIR / "application_train.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
MODEL_FILE = MODELS_DIR / "credit_risk_model.joblib"
CALIBRATED_MODEL_FILE = MODELS_DIR / "calibrated_credit_risk_model.joblib"
METRICS_FILE = METRICS_DIR / "training_metrics.json"
METADATA_FILE = METADATA_DIR / "model_metadata.json"
BENCHMARK_REFERENCE_FILE = METRICS_DIR / "research_benchmark_reference.json"
REPAYMENT_STRESS_MODEL_FILE = MODELS_DIR / "repayment_stress_model.joblib"
REPAYMENT_STRESS_METRICS_FILE = METRICS_DIR / "repayment_stress_metrics.json"
REPAYMENT_STRESS_METADATA_FILE = METADATA_DIR / "repayment_stress_metadata.json"
LOG_FILE = LOGS_DIR / "credit_risk.log"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

RANDOM_STATE = 42
TEST_SIZE = 0.20
POLICY_SIZE = 0.20  # fraction of the 80% development partition (16% overall)
MODEL_SELECTION_FOLDS = 3
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_METHODS = ("sigmoid", "isotonic")
CALIBRATION_FOLDS = 3

MODEL_NAME = "xgboost"
MODEL_PARAMETERS = {
    "n_estimators": 500,
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
}

# Illustrative prototype policy only. Training can replace these defaults with
# thresholds measured on the policy-selection holdout.
DEFAULT_APPROVE_THRESHOLD = 0.08
DEFAULT_REJECT_THRESHOLD = 0.15
CANDIDATE_APPROVE_THRESHOLDS = tuple(round(value / 100, 2) for value in range(4, 13))
CANDIDATE_REJECT_THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 26))
RISK_BAND_THRESHOLDS = (0.05, 0.10, 0.20)
MAX_MANUAL_REVIEW_RATE = 0.20
MIN_APPROVAL_RATE = 0.50
MAX_APPROVED_DEFAULT_RATE = 0.06

# Illustrative economics; these are not portfolio facts or lending policy.
ILLUSTRATIVE_LGD = 0.60
ILLUSTRATIVE_NET_MARGIN_RATE = 0.08
EXPOSURE_COLUMN = "AMT_CREDIT"

REQUIRED_APPLICATION_COLUMNS = (
    *PRODUCTION_RAW_FEATURES,
)


def ensure_project_directories() -> None:
    """Create runtime output directories without touching source data."""

    for directory in (MODELS_DIR, METRICS_DIR, METADATA_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
