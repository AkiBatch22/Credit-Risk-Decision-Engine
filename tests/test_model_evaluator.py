import numpy as np

from src.components.model_evaluator import evaluate_probabilities, risk_decile_analysis


def test_model_evaluator_returns_structured_metrics():
    target = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.70, 0.40, 0.60])
    metrics = evaluate_probabilities(target, probabilities, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]
    assert 0 <= metrics["brier_score"] <= 1
    assert risk_decile_analysis(target, probabilities, bins=3)
