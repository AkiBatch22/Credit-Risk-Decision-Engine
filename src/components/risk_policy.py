"""Transparent probability-to-decision policy and illustrative economics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_APPROVE_THRESHOLD,
    DEFAULT_REJECT_THRESHOLD,
    ILLUSTRATIVE_LGD,
    ILLUSTRATIVE_NET_MARGIN_RATE,
    MAX_APPROVED_DEFAULT_RATE,
    MAX_MANUAL_REVIEW_RATE,
    MIN_APPROVAL_RATE,
    RISK_BAND_THRESHOLDS,
)


@dataclass(frozen=True)
class RiskPolicy:
    approve_threshold: float = DEFAULT_APPROVE_THRESHOLD
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    risk_band_thresholds: tuple[float, float, float] = RISK_BAND_THRESHOLDS
    lgd: float = ILLUSTRATIVE_LGD
    net_margin_rate: float = ILLUSTRATIVE_NET_MARGIN_RATE
    assumptions_are_illustrative: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.approve_threshold < self.reject_threshold <= 1:
            raise ValueError("policy thresholds must satisfy 0 <= approve < reject <= 1")
        if tuple(sorted(self.risk_band_thresholds)) != self.risk_band_thresholds:
            raise ValueError("risk-band thresholds must be strictly ordered")
        if not 0 <= self.lgd <= 1:
            raise ValueError("LGD must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "approve_threshold": self.approve_threshold,
            "reject_threshold": self.reject_threshold,
            "risk_band_thresholds": list(self.risk_band_thresholds),
            "lgd": self.lgd,
            "net_margin_rate": self.net_margin_rate,
            "assumptions_are_illustrative": self.assumptions_are_illustrative,
        }


def assign_recommendation(probability: float, policy: RiskPolicy) -> str:
    if probability < policy.approve_threshold:
        return "APPROVE"
    if probability < policy.reject_threshold:
        return "MANUAL_REVIEW"
    return "REJECT"


def assign_risk_band(probability: float, thresholds: tuple[float, float, float] = RISK_BAND_THRESHOLDS) -> str:
    if probability < thresholds[0]:
        return "LOW"
    if probability < thresholds[1]:
        return "MODERATE"
    if probability < thresholds[2]:
        return "HIGH"
    return "VERY_HIGH"


def calculate_expected_loss(probability: Any, exposure: Any, lgd: float = ILLUSTRATIVE_LGD) -> Any:
    if not 0 <= lgd <= 1:
        raise ValueError("LGD must be between 0 and 1")
    probability_array = np.asarray(probability, dtype=float)
    exposure_array = np.asarray(exposure, dtype=float)
    if np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probability must be between 0 and 1")
    if np.any(exposure_array < 0):
        raise ValueError("exposure must be non-negative")
    return probability_array * lgd * exposure_array


def apply_risk_policy(
    probabilities: Iterable[float],
    *,
    exposure: Iterable[float] | None = None,
    policy: RiskPolicy | None = None,
) -> pd.DataFrame:
    selected = policy or RiskPolicy()
    probability_array = np.asarray(list(probabilities), dtype=float)
    if np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probabilities must be between 0 and 1")
    output = pd.DataFrame(
        {
            "probability": probability_array,
            "probability_percent": probability_array * 100,
            "risk_band": [assign_risk_band(value, selected.risk_band_thresholds) for value in probability_array],
            "recommendation": [assign_recommendation(value, selected) for value in probability_array],
        }
    )
    if exposure is not None:
        exposure_array = np.asarray(list(exposure), dtype=float)
        if len(exposure_array) != len(output):
            raise ValueError("exposure and probabilities must contain the same number of rows")
        output["expected_loss"] = calculate_expected_loss(probability_array, exposure_array, selected.lgd)
    return output


def optimize_policy(
    target: Iterable[int],
    probabilities: Iterable[float],
    exposure: Iterable[float],
    approve_thresholds: Iterable[float],
    reject_thresholds: Iterable[float],
    *,
    lgd: float = ILLUSTRATIVE_LGD,
    net_margin_rate: float = ILLUSTRATIVE_NET_MARGIN_RATE,
    max_review_rate: float = MAX_MANUAL_REVIEW_RATE,
    min_approval_rate: float = MIN_APPROVAL_RATE,
    max_approved_default_rate: float = MAX_APPROVED_DEFAULT_RATE,
) -> tuple[RiskPolicy, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "target": np.asarray(list(target), dtype=int),
            "probability": np.asarray(list(probabilities), dtype=float),
            "exposure": np.asarray(list(exposure), dtype=float),
        }
    )
    frame["expected_value"] = frame["exposure"] * (
        net_margin_rate - frame["probability"] * lgd
    )
    total_defaults = max(float(frame["target"].sum()), 1.0)
    records: list[dict[str, float]] = []
    for approve in approve_thresholds:
        for reject in reject_thresholds:
            if reject <= approve:
                continue
            approved = frame["probability"] < approve
            review = frame["probability"].between(approve, reject, inclusive="left")
            rejected = frame["probability"] >= reject
            approval_rate = float(approved.mean())
            review_rate = float(review.mean())
            approved_default_rate = float(frame.loc[approved, "target"].mean()) if approved.any() else np.nan
            records.append(
                {
                    "approve_threshold": float(approve),
                    "reject_threshold": float(reject),
                    "approval_rate": approval_rate,
                    "review_rate": review_rate,
                    "rejection_rate": float(rejected.mean()),
                    "approved_default_rate": approved_default_rate,
                    "rejected_default_capture": float(frame.loc[rejected, "target"].sum() / total_defaults),
                    "expected_portfolio_value": float(frame.loc[approved, "expected_value"].sum()),
                }
            )
    results = pd.DataFrame(records)
    feasible = results[
        (results["review_rate"] <= max_review_rate)
        & (results["approval_rate"] >= min_approval_rate)
        & (results["approved_default_rate"] <= max_approved_default_rate)
    ]
    if feasible.empty:
        raise ValueError("no candidate policy satisfies the configured operational constraints")
    winner = feasible.sort_values(
        ["expected_portfolio_value", "rejected_default_capture", "review_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    policy = RiskPolicy(
        approve_threshold=float(winner["approve_threshold"]),
        reject_threshold=float(winner["reject_threshold"]),
        lgd=lgd,
        net_margin_rate=net_margin_rate,
    )
    return policy, results.sort_values("expected_portfolio_value", ascending=False).reset_index(drop=True)
