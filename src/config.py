from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

LOGS_DIR = PROJECT_ROOT / "logs"


TRAIN_DATA_FILE = RAW_DATA_DIR / "application_train.csv"

PROCESSED_TRAIN_FILE = (
    PROCESSED_DATA_DIR / "train_processed.parquet"
)

PREPROCESSOR_FILE = (
    MODELS_DIR / "preprocessor.joblib"
)

MODEL_FILE = (
    MODELS_DIR / "credit_risk_model.joblib"
)


TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

TEST_SIZE = 0.20
RANDOM_STATE = 42