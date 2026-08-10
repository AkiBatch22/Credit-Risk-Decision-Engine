"""Optional reason codes using XGBoost's native TreeSHAP contributions."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.feature_contract import friendly_explanation_name


def _display_name(transformed_name: str) -> str:
    name = transformed_name.replace("numeric__", "").replace("categorical__", "")
    return friendly_explanation_name(name)


def explain_xgboost_predictions(
    model: Any,
    transformed_features: Any,
    feature_names: list[str],
    *,
    top_n: int = 5,
) -> list[list[dict[str, Any]]]:
    """Return local contributions without making the external shap package mandatory."""

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("XGBoost explanation support is unavailable") from error
    if not hasattr(model, "get_booster"):
        raise TypeError("native TreeSHAP reason codes require a fitted XGBoost model")

    contributions = model.get_booster().predict(
        xgb.DMatrix(transformed_features), pred_contribs=True
    )
    explanations: list[list[dict[str, Any]]] = []
    for row in contributions:
        values = np.asarray(row[:-1])  # final value is the expected-value bias
        indices = np.argsort(np.abs(values))[::-1][:top_n]
        explanations.append(
            [
                {
                    "feature": _display_name(feature_names[index]),
                    "contribution": float(values[index]),
                    "direction": "increases risk" if values[index] > 0 else "reduces risk",
                }
                for index in indices
                if values[index] != 0
            ]
        )
    return explanations
