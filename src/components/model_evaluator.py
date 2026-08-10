"""Structured probability and classification evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_risk_default_capture(
    target: Any, probabilities: Any, top_fraction: float = 0.10
) -> float:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    frame = pd.DataFrame({"target": np.asarray(target), "probability": probabilities}).sort_values(
        "probability", ascending=False
    )
    total_defaults = float(frame["target"].sum())
    if total_defaults == 0:
        return 0.0
    top_count = max(1, int(np.ceil(len(frame) * top_fraction)))
    return float(frame.head(top_count)["target"].sum() / total_defaults)


def risk_decile_analysis(target: Any, probabilities: Any, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"target": np.asarray(target), "probability": np.asarray(probabilities)})
    if frame.empty:
        return []
    effective_bins = min(bins, len(frame))
    frame["risk_decile"] = pd.qcut(
        frame["probability"], q=effective_bins, labels=False, duplicates="drop"
    )
    grouped = (
        frame.groupby("risk_decile", observed=True)
        .agg(
            applicants=("target", "size"),
            defaults=("target", "sum"),
            observed_default_rate=("target", "mean"),
            mean_predicted_probability=("probability", "mean"),
        )
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def evaluate_probabilities(
    target: Any,
    probabilities: Any,
    *,
    threshold: float = 0.50,
    include_deciles: bool = True,
) -> dict[str, Any]:
    y_true = np.asarray(target)
    y_probability = np.asarray(probabilities, dtype=float)
    if y_true.shape[0] != y_probability.shape[0]:
        raise ValueError("target and probabilities must contain the same number of rows")
    if np.any((y_probability < 0) | (y_probability > 1)):
        raise ValueError("probabilities must be between 0 and 1")

    predictions = (y_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "confusion_matrix": matrix.tolist(),
        "top_10_percent_default_capture": top_risk_default_capture(y_true, y_probability),
    }
    if include_deciles:
        result["risk_deciles"] = risk_decile_analysis(y_true, y_probability)
    return result
