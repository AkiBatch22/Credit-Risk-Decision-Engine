"""Deterministic, target-independent application feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.components.feature_contract import PRODUCTION_ENGINEERED_FEATURES

BENCHMARK_ONLY_ENGINEERED_FEATURES = (
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_STD",
    "EXT_SOURCE_COUNT",
    "DOCUMENT_COUNT",
    "CONTACT_COUNT",
    "DEF_30_SOCIAL_RATIO",
    "DEF_60_SOCIAL_RATIO",
)
ENGINEERED_FEATURES = PRODUCTION_ENGINEERED_FEATURES


def _series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column in dataframe:
        return pd.to_numeric(dataframe[column], errors="coerce")
    return pd.Series(np.nan, index=dataframe.index, dtype=float, name=column)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.where(denominator.ne(0)))
    return result.replace([np.inf, -np.inf], np.nan)


def create_deployable_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create only reproducible, target-independent production features."""

    result = dataframe.copy(deep=True)
    days_employed = _series(result, "DAYS_EMPLOYED")
    result["DAYS_EMPLOYED_ANOMALY"] = days_employed.eq(365243).astype(float).where(days_employed.notna())
    result["DAYS_EMPLOYED_CLEAN"] = days_employed.replace(365243, np.nan)

    result["AGE_YEARS"] = -_series(result, "DAYS_BIRTH") / 365.25
    result["EMPLOYMENT_YEARS"] = -result["DAYS_EMPLOYED_CLEAN"] / 365.25
    result["EMPLOYMENT_AGE_RATIO"] = _safe_divide(
        result["EMPLOYMENT_YEARS"], result["AGE_YEARS"]
    )

    income = _series(result, "AMT_INCOME_TOTAL")
    credit = _series(result, "AMT_CREDIT")
    annuity = _series(result, "AMT_ANNUITY")
    goods_price = _series(result, "AMT_GOODS_PRICE")
    family_members = _series(result, "CNT_FAM_MEMBERS")
    children = _series(result, "CNT_CHILDREN")

    result["CREDIT_INCOME_RATIO"] = _safe_divide(credit, income)
    result["ANNUITY_INCOME_RATIO"] = _safe_divide(annuity, income)
    result["CREDIT_ANNUITY_RATIO"] = _safe_divide(credit, annuity)
    result["CREDIT_GOODS_RATIO"] = _safe_divide(credit, goods_price)
    result["INCOME_PER_PERSON"] = _safe_divide(income, family_members)
    result["CREDIT_PER_PERSON"] = _safe_divide(credit, family_members)
    result["ANNUITY_PER_PERSON"] = _safe_divide(annuity, family_members)
    result["CHILDREN_FAMILY_RATIO"] = _safe_divide(children, family_members)

    return result.replace([np.inf, -np.inf], np.nan)


def create_benchmark_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add research-only features whose sources are unavailable in deployment."""

    result = create_deployable_features(dataframe)
    external = pd.DataFrame(
        {column: _series(result, column) for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")},
        index=result.index,
    )
    result["EXT_SOURCE_MEAN"] = external.mean(axis=1)
    result["EXT_SOURCE_MIN"] = external.min(axis=1)
    result["EXT_SOURCE_MAX"] = external.max(axis=1)
    result["EXT_SOURCE_STD"] = external.std(axis=1)
    result["EXT_SOURCE_COUNT"] = external.notna().sum(axis=1)

    document_columns = [column for column in result if column.startswith("FLAG_DOCUMENT_")]
    result["DOCUMENT_COUNT"] = (
        result[document_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if document_columns
        else np.nan
    )
    contact_columns = [
        column
        for column in (
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        )
        if column in result
    ]
    result["CONTACT_COUNT"] = (
        result[contact_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if contact_columns
        else np.nan
    )

    result["DEF_30_SOCIAL_RATIO"] = _safe_divide(
        _series(result, "DEF_30_CNT_SOCIAL_CIRCLE"),
        _series(result, "OBS_30_CNT_SOCIAL_CIRCLE"),
    )
    result["DEF_60_SOCIAL_RATIO"] = _safe_divide(
        _series(result, "DEF_60_CNT_SOCIAL_CIRCLE"),
        _series(result, "OBS_60_CNT_SOCIAL_CIRCLE"),
    )
    return result.replace([np.inf, -np.inf], np.nan)


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the deployable production transformation."""

    return create_deployable_features(dataframe)
