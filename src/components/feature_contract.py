"""Feature contract for the preserved Home Credit ML serving experiment.

The historical Home Credit dataset contains many useful benchmark variables whose
inference-time source is unavailable here. This contract remains authoritative for
the research training/prediction pipeline; the production-facing affordability
simulator uses ``loan_simulator.py`` and does not use the historical ML model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd


UNKNOWN_CATEGORY = "Unknown / Not provided"
APPLICATION_ID_PATTERN = re.compile(r"^APP-[0-9A-F]{10}$")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    api_name: str
    label: str
    help: str
    source: str
    required: bool
    dtype: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    allowed_values: tuple[str, ...] = ()
    transformation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRODUCTION_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "DAYS_BIRTH", "age", "Age",
        "Enter the applicant's age in completed years.",
        "applicant-provided", True, "float", 35.0, 18.0, 75.0, 1.0,
        transformation="DAYS_BIRTH = int(-age * 365.25)",
    ),
    FeatureSpec(
        "DAYS_EMPLOYED", "years_employed", "Years Employed",
        "Enter how many years the applicant has been employed in the current or recorded employment.",
        "applicant-provided", True, "float", 5.0, 0.0, 60.0, 0.5,
        transformation="DAYS_EMPLOYED = int(-years_employed * 365.25)",
    ),
    FeatureSpec(
        "CNT_FAM_MEMBERS", "family_members", "Family Members",
        "Enter the total number of people recorded in the applicant's family or household.",
        "applicant-provided", True, "float", 2.0, 1.0, 20.0, 1.0,
    ),
    FeatureSpec(
        "CNT_CHILDREN", "number_of_children", "Number of Children",
        "Enter the number of children recorded for the applicant.",
        "applicant-provided", True, "int", 0, 0.0, 19.0, 1.0,
    ),
    FeatureSpec(
        "AMT_INCOME_TOTAL", "annual_income", "Annual Income",
        "Enter the applicant's total annual income using the same monetary unit as the credit application.",
        "applicant-provided", True, "float", 202_500.0, 0.0, 117_000_000.0, 1_000.0,
    ),
    FeatureSpec(
        "AMT_CREDIT", "requested_loan_amount", "Requested Loan Amount",
        "Enter the total amount of credit requested by the applicant.",
        "applicant-provided", True, "float", 406_597.5, 1.0, 4_050_000.0, 1_000.0,
    ),
    FeatureSpec(
        "AMT_ANNUITY", "loan_annuity", "Loan Annuity",
        "Enter the scheduled periodic repayment amount associated with the requested credit.",
        "applicant-provided", True, "float", 24_700.5, 0.0, 258_025.5, 500.0,
    ),
    FeatureSpec(
        "AMT_GOODS_PRICE", "goods_purchase_price", "Goods Purchase Price",
        "Enter the price of the goods or asset associated with the credit application, when applicable.",
        "applicant-provided", True, "float", 351_000.0, 0.0, 4_050_000.0, 1_000.0,
    ),
    FeatureSpec(
        "NAME_CONTRACT_TYPE", "credit_product_type", "Credit Product Type",
        "Select the type of credit product being requested.",
        "applicant-provided", True, "category", "Cash loans",
        allowed_values=("Cash loans", "Revolving loans"),
    ),
    FeatureSpec(
        "NAME_INCOME_TYPE", "income_type", "Income Type",
        "Select the applicant's primary income or employment category.",
        "applicant-provided", False, "category", "Working",
        allowed_values=(
            "Businessman", "Commercial associate", "Maternity leave", "Pensioner",
            "State servant", "Student", "Unemployed", "Working",
        ),
    ),
    FeatureSpec(
        "NAME_HOUSING_TYPE", "housing_situation", "Housing Situation",
        "Select the applicant's current housing arrangement.",
        "applicant-provided", False, "category", "House / apartment",
        allowed_values=(
            "Co-op apartment", "House / apartment", "Municipal apartment",
            "Office apartment", "Rented apartment", "With parents",
        ),
    ),
    FeatureSpec(
        "FLAG_OWN_CAR", "owns_car", "Owns Car",
        "Indicate whether the applicant reports owning a car.",
        "applicant-provided", True, "category", "No",
        allowed_values=("No", "Yes"), transformation="No -> N; Yes -> Y",
    ),
    FeatureSpec(
        "FLAG_OWN_REALTY", "owns_property", "Owns Property",
        "Indicate whether the applicant reports owning real property.",
        "applicant-provided", True, "category", "Yes",
        allowed_values=("No", "Yes"), transformation="No -> N; Yes -> Y",
    ),
)

PRODUCTION_RAW_FEATURES = tuple(spec.name for spec in PRODUCTION_FEATURE_SPECS)
APPLICANT_API_FIELDS = tuple(spec.api_name for spec in PRODUCTION_FEATURE_SPECS)
REQUIRED_API_FIELDS = tuple(spec.api_name for spec in PRODUCTION_FEATURE_SPECS if spec.required)
FEATURE_BY_NAME = {spec.name: spec for spec in PRODUCTION_FEATURE_SPECS}
FEATURE_BY_API_NAME = {spec.api_name: spec for spec in PRODUCTION_FEATURE_SPECS}

PRODUCTION_ENGINEERED_FEATURES = (
    "DAYS_EMPLOYED_ANOMALY",
    "DAYS_EMPLOYED_CLEAN",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "EMPLOYMENT_AGE_RATIO",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "CREDIT_GOODS_RATIO",
    "INCOME_PER_PERSON",
    "CREDIT_PER_PERSON",
    "ANNUITY_PER_PERSON",
    "CHILDREN_FAMILY_RATIO",
)

EXTERNAL_UNAVAILABLE_FEATURES = (
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX",
    "EXT_SOURCE_STD", "EXT_SOURCE_COUNT",
)
HISTORICAL_IDENTIFIER_FEATURES = ("SK_ID_CURR",)
SENSITIVE_OR_PROXY_FEATURES = (
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE",
)

FRIENDLY_EXPLANATION_NAMES = {
    **{spec.name: spec.label for spec in PRODUCTION_FEATURE_SPECS},
    "CREDIT_INCOME_RATIO": "Loan-to-Income Ratio",
    "ANNUITY_INCOME_RATIO": "Repayment-to-Income Ratio",
    "CREDIT_ANNUITY_RATIO": "Loan-to-Repayment Ratio",
    "CREDIT_GOODS_RATIO": "Loan-to-Goods-Price Ratio",
    "AGE_YEARS": "Applicant Age",
    "EMPLOYMENT_YEARS": "Employment History",
    "EMPLOYMENT_AGE_RATIO": "Employment-to-Age Ratio",
    "INCOME_PER_PERSON": "Income per Household Member",
    "CREDIT_PER_PERSON": "Credit per Household Member",
    "ANNUITY_PER_PERSON": "Repayment per Household Member",
    "CHILDREN_FAMILY_RATIO": "Children-to-Household Ratio",
    "DAYS_EMPLOYED_ANOMALY": "Employment Data Availability",
    "DAYS_EMPLOYED_CLEAN": "Employment History",
}


def contract_as_dict() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in PRODUCTION_FEATURE_SPECS]


def generate_application_id() -> str:
    """Generate a traceability identifier that is never a model feature."""

    return f"APP-{uuid4().hex[:10].upper()}"


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _translate_value(spec: FeatureSpec, value: Any) -> Any:
    if _is_missing(value) or value == UNKNOWN_CATEGORY:
        return np.nan
    if spec.api_name == "age":
        return int(-float(value) * 365.25)
    if spec.api_name == "years_employed":
        return int(-float(value) * 365.25)
    if spec.name in {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}:
        mapping = {"Yes": "Y", "No": "N", "Y": "Y", "N": "N"}
        if str(value) not in mapping:
            raise ValueError(f"{spec.label} must be Yes or No")
        return mapping[str(value)]
    if spec.dtype == "int":
        return int(value)
    if spec.dtype == "float":
        return float(value)
    return str(value)


def validate_application_values(values: Mapping[str, Any]) -> None:
    """Validate friendly application fields before conversion."""

    missing = [name for name in REQUIRED_API_FIELDS if name not in values or _is_missing(values[name])]
    if missing:
        labels = [FEATURE_BY_API_NAME[name].label for name in missing]
        raise ValueError(f"Missing required application fields: {labels}")

    for api_name, value in values.items():
        spec = FEATURE_BY_API_NAME.get(api_name)
        if spec is None or _is_missing(value) or value == UNKNOWN_CATEGORY:
            continue
        if spec.dtype in {"float", "int"}:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{spec.label} must be numeric") from error
            if spec.minimum is not None and numeric < spec.minimum:
                raise ValueError(f"{spec.label} must be at least {spec.minimum:g}")
            if spec.maximum is not None and numeric > spec.maximum:
                raise ValueError(f"{spec.label} must not exceed {spec.maximum:g}")
        elif spec.allowed_values and str(value) not in spec.allowed_values:
            raise ValueError(f"{spec.label} must be one of: {list(spec.allowed_values)}")

    age = values.get("age")
    employment = values.get("years_employed")
    if not _is_missing(age) and not _is_missing(employment):
        plausible_working_years = max(float(age) - 14.0, 0.0)
        if float(employment) > plausible_working_years:
            raise ValueError("Years Employed is not plausible relative to the applicant's age")
    children = values.get("number_of_children")
    family = values.get("family_members")
    if not _is_missing(children) and not _is_missing(family) and float(children) > float(family):
        raise ValueError("Number of Children cannot exceed Family Members")


def application_to_model_input(values: Mapping[str, Any]) -> dict[str, Any]:
    """Translate friendly application fields to the internal production schema."""

    unexpected = sorted(set(values).difference(APPLICANT_API_FIELDS))
    if unexpected:
        raise ValueError(f"Unknown application fields: {unexpected}")
    validate_application_values(values)
    output: dict[str, Any] = {}
    for spec in PRODUCTION_FEATURE_SPECS:
        value = values.get(spec.api_name, np.nan)
        output[spec.name] = _translate_value(spec, value)
    return output


def raw_contract_to_model_input(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and select already-internal contract values for historical batch input."""

    present = set(values).intersection(PRODUCTION_RAW_FEATURES)
    if not present:
        raise ValueError("prediction input has no fields recognized by the deployable feature contract")
    required_internal = {FEATURE_BY_API_NAME[name].name for name in REQUIRED_API_FIELDS}
    missing = sorted(required_internal.difference(present))
    if missing:
        raise ValueError(f"Missing required production fields: {missing}")
    output = {name: values.get(name, np.nan) for name in PRODUCTION_RAW_FEATURES}
    for name in (
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS", "CNT_CHILDREN", "DAYS_BIRTH", "DAYS_EMPLOYED",
    ):
        try:
            output[name] = float(output[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
    if output["DAYS_BIRTH"] >= 0:
        raise ValueError("DAYS_BIRTH must be a negative historical day offset")
    if output["DAYS_EMPLOYED"] > 0 and output["DAYS_EMPLOYED"] != 365243:
        raise ValueError("DAYS_EMPLOYED must be non-positive or the documented historical sentinel")
    if output["AMT_CREDIT"] <= 0:
        raise ValueError("AMT_CREDIT must be positive")
    if any(output[name] < 0 for name in ("AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN")):
        raise ValueError("income, annuity, goods price, and children must be non-negative")
    if output["CNT_FAM_MEMBERS"] < 1:
        raise ValueError("CNT_FAM_MEMBERS must be at least 1")
    for name in ("NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE"):
        value = output[name]
        if not _is_missing(value) and str(value) not in FEATURE_BY_NAME[name].allowed_values:
            raise ValueError(f"{name} contains an unsupported category: {value}")
    for name in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        if str(output[name]) not in {"Y", "N"}:
            raise ValueError(f"{name} must be Y or N")
    return output


def normalize_application_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize friendly API/UI or internal historical rows to contract columns."""

    if dataframe.empty:
        raise ValueError("prediction input contains no rows")
    if "TARGET" in dataframe.columns:
        raise ValueError("prediction input must not contain TARGET")
    uses_friendly = bool(set(dataframe.columns).intersection(APPLICANT_API_FIELDS))
    uses_internal = bool(set(dataframe.columns).intersection(PRODUCTION_RAW_FEATURES))
    if uses_friendly and uses_internal:
        raise ValueError("Do not mix friendly application fields with internal Home Credit field names")
    records = []
    for record in dataframe.to_dict(orient="records"):
        trace_fields = {"SK_ID_CURR", "source_record_id", "application_id"}
        model_values = {key: value for key, value in record.items() if key not in trace_fields}
        records.append(
            application_to_model_input(model_values)
            if uses_friendly
            else raw_contract_to_model_input(model_values)
        )
    return pd.DataFrame(records, columns=PRODUCTION_RAW_FEATURES, index=dataframe.index)


def friendly_explanation_name(feature_name: Any) -> str:
    original = str(feature_name).strip()
    normalized = re.sub(r"[^A-Z0-9]+", "_", original.upper()).strip("_")
    for prefix in ("NUMERIC_", "CATEGORICAL_"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
    for spec in PRODUCTION_FEATURE_SPECS:
        category_prefix = f"{spec.name}_"
        if spec.dtype == "category" and normalized.startswith(category_prefix):
            category = normalized.removeprefix(category_prefix).replace("_", " ").title()
            return f"{spec.label}: {category}"
    return FRIENDLY_EXPLANATION_NAMES.get(normalized, original.replace("_", " ").title())


def classify_historical_feature(feature_name: str) -> tuple[str, str]:
    """Classify a historical column for the persisted feature-availability audit."""

    if feature_name in PRODUCTION_RAW_FEATURES:
        return "applicant-provided", f"Collected as {FEATURE_BY_NAME[feature_name].label}."
    if feature_name in HISTORICAL_IDENTIFIER_FEATURES:
        return "historical-identifier", "Historical row traceability only; never a predictor."
    if feature_name in EXTERNAL_UNAVAILABLE_FEATURES:
        return "external-unavailable", "Anonymized source cannot be reproduced for a new applicant."
    if feature_name in SENSITIVE_OR_PROXY_FEATURES:
        return "questionable-sensitive-proxy", "Excluded from the recommended deployment contract."
    if feature_name == "TARGET":
        return "training-label", "Observed historical outcome; unavailable at inference."
    return "questionable-or-unavailable", "No reproducible source is implemented for a new application."
