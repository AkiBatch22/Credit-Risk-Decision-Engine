"""Configurable model construction and training."""

from __future__ import annotations

import time
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_NAME, MODEL_PARAMETERS, RANDOM_STATE


def build_model(model_name: str = MODEL_NAME, parameters: dict[str, Any] | None = None) -> ClassifierMixin:
    name = model_name.lower()
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("XGBoost is required for the selected production model") from error
        return XGBClassifier(**(parameters or MODEL_PARAMETERS))

    if name == "logistic_regression":
        defaults: dict[str, Any] = {"max_iter": 1000, "random_state": RANDOM_STATE}
        defaults.update(parameters or {})
        return LogisticRegression(**defaults)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("LightGBM is required for this comparison model") from error
        defaults = {"objective": "binary", "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1}
        defaults.update(parameters or {})
        return LGBMClassifier(**defaults)
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(model: ClassifierMixin, features: Any, target: Any) -> tuple[ClassifierMixin, dict[str, Any]]:
    started = time.perf_counter()
    model.fit(features, target)
    metadata = {
        "model_class": type(model).__name__,
        "training_rows": int(len(target)),
        "training_seconds": round(time.perf_counter() - started, 6),
    }
    return model, metadata
