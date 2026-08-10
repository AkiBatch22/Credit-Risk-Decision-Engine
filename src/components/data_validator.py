"""Fatal schema validation and non-fatal data-quality diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

from src.config import ID_COLUMN, REQUIRED_APPLICATION_COLUMNS, TARGET_COLUMN


@dataclass
class ValidationReport:
    fatal_errors: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.fatal_errors

    def raise_for_errors(self) -> None:
        if self.fatal_errors:
            raise ValueError("Data validation failed: " + "; ".join(self.fatal_errors))

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, **asdict(self)}


def validate_training_data(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
    major_missingness_threshold: float = 0.50,
) -> ValidationReport:
    report = ValidationReport()
    required = set(required_columns or (TARGET_COLUMN, ID_COLUMN, *REQUIRED_APPLICATION_COLUMNS))

    report.statistics.update({"rows": len(dataframe), "columns": len(dataframe.columns)})
    if dataframe.empty:
        report.fatal_errors.append("dataset contains no rows")
        return report

    missing_required = sorted(required.difference(dataframe.columns))
    if missing_required:
        report.fatal_errors.append(f"missing required columns: {missing_required}")

    if TARGET_COLUMN in dataframe:
        target_values = set(dataframe[TARGET_COLUMN].dropna().unique().tolist())
        missing_target_count = int(dataframe[TARGET_COLUMN].isna().sum())
        report.statistics["target_values"] = sorted(target_values)
        report.statistics["missing_target_count"] = missing_target_count
        if missing_target_count:
            report.fatal_errors.append(f"{TARGET_COLUMN} contains {missing_target_count} missing values")
        if not target_values.issubset({0, 1}) or not target_values:
            report.fatal_errors.append(f"{TARGET_COLUMN} must contain only binary values 0 and 1")

    if ID_COLUMN in dataframe:
        missing_ids = int(dataframe[ID_COLUMN].isna().sum())
        duplicate_ids = int(dataframe[ID_COLUMN].duplicated().sum())
        report.statistics.update(
            {"missing_applicant_ids": missing_ids, "duplicate_applicant_ids": duplicate_ids}
        )
        if missing_ids:
            report.fatal_errors.append(f"{ID_COLUMN} contains {missing_ids} missing values")
        if duplicate_ids:
            report.fatal_errors.append(f"{ID_COLUMN} contains {duplicate_ids} duplicate values")

    duplicate_rows = int(dataframe.duplicated().sum())
    report.statistics["duplicate_rows"] = duplicate_rows
    if duplicate_rows:
        report.diagnostics.append(f"found {duplicate_rows} duplicate rows")

    missing_rates = dataframe.isna().mean().sort_values(ascending=False)
    major_missing = {
        column: round(float(rate), 6)
        for column, rate in missing_rates.items()
        if rate >= major_missingness_threshold
    }
    report.statistics["major_missingness"] = major_missing
    if major_missing:
        report.diagnostics.append(
            f"{len(major_missing)} columns have at least {major_missingness_threshold:.0%} missing values"
        )

    if "DAYS_EMPLOYED" in dataframe:
        anomaly_count = int(dataframe["DAYS_EMPLOYED"].eq(365243).sum())
        report.statistics["days_employed_anomaly_count"] = anomaly_count
        if anomaly_count:
            report.diagnostics.append(
                f"DAYS_EMPLOYED contains {anomaly_count} sentinel values equal to 365243"
            )
    return report


def validate_prediction_data(
    dataframe: pd.DataFrame,
    expected_columns: Iterable[str],
    required_columns: Iterable[str] | None = None,
) -> ValidationReport:
    report = ValidationReport(statistics={"rows": len(dataframe), "columns": len(dataframe.columns)})
    if dataframe.empty:
        report.fatal_errors.append("prediction input contains no rows")
        return report
    if dataframe.columns.duplicated().any():
        report.fatal_errors.append("prediction input contains duplicate column names")
    if TARGET_COLUMN in dataframe:
        report.fatal_errors.append(f"prediction input must not contain {TARGET_COLUMN}")

    expected = list(expected_columns)
    required = list(required_columns or expected)
    present = [column for column in expected if column in dataframe]
    missing = [column for column in expected if column not in dataframe]
    unexpected = [column for column in dataframe.columns if column not in expected]
    report.statistics.update(
        {
            "expected_feature_count": len(expected),
            "present_feature_count": len(present),
            "missing_features": missing,
            "unexpected_features": unexpected,
        }
    )
    if not present:
        report.fatal_errors.append("prediction input has no fields recognized by the fitted model schema")
    missing_required = [column for column in required if column not in dataframe]
    if missing_required:
        report.fatal_errors.append(f"missing required prediction fields: {missing_required}")
    if missing and not missing_required:
        report.diagnostics.append(f"{len(missing)} expected fields are absent and will be imputed")
    return report
