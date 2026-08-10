"""Reusable preprocessing fitted once and shared by training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, TARGET_COLUMN
from src.components.feature_contract import PRODUCTION_RAW_FEATURES
from src.components.feature_engineering import create_deployable_features


def select_production_raw_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select the explicit deployable raw contract; arbitrary columns cannot enter fitting."""

    missing = sorted(set(PRODUCTION_RAW_FEATURES).difference(dataframe.columns))
    if missing:
        raise ValueError(f"training data is missing deployable feature columns: {missing}")
    return dataframe.loc[:, PRODUCTION_RAW_FEATURES].copy(deep=True)


def prepare_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply production features and remove non-predictive/superseded columns."""

    selected = select_production_raw_features(dataframe)
    return create_deployable_features(selected).drop(
        columns=[TARGET_COLUMN, ID_COLUMN, "DAYS_EMPLOYED"], errors="ignore"
    )


def infer_feature_types(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical = dataframe.select_dtypes(exclude=np.number).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build an unfitted preprocessing graph from train-derived column lists."""

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor_for_frame(
    training_features: pd.DataFrame, *, scale_numeric: bool = False
) -> ColumnTransformer:
    numeric, categorical = infer_feature_types(training_features)
    return build_preprocessor(numeric, categorical, scale_numeric=scale_numeric)
