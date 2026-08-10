"""Leakage-safe cross-validated probability calibration."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV

from src.config import CALIBRATION_FOLDS, CALIBRATION_METHOD


def calibrate_model(
    estimator: ClassifierMixin,
    features: Any,
    target: Any,
    *,
    method: str = CALIBRATION_METHOD,
    cv: int = CALIBRATION_FOLDS,
) -> CalibratedClassifierCV:
    """Fit calibration folds only inside the supplied model-training data."""

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration method must be 'sigmoid' or 'isotonic'")
    calibrated = CalibratedClassifierCV(estimator=clone(estimator), method=method, cv=cv)
    calibrated.fit(features, target)
    return calibrated


def calibrated_probabilities(model: ClassifierMixin, features: Any) -> Any:
    if not hasattr(model, "predict_proba"):
        raise TypeError("calibrated model must expose predict_proba")
    return model.predict_proba(features)[:, 1]
