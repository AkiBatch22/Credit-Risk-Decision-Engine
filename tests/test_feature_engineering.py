import numpy as np
import pandas as pd

from src.components.feature_engineering import (
    BENCHMARK_ONLY_ENGINEERED_FEATURES,
    ENGINEERED_FEATURES,
    create_benchmark_features,
    create_features,
)


def test_feature_engineering_is_non_mutating_leakage_safe_and_finite(synthetic_applications):
    original = synthetic_applications.copy(deep=True)
    featured = create_features(synthetic_applications)
    pd.testing.assert_frame_equal(synthetic_applications, original)
    assert set(ENGINEERED_FEATURES).issubset(featured.columns)
    assert featured.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1
    assert np.isnan(featured.loc[0, "DAYS_EMPLOYED_CLEAN"])
    assert featured["TARGET"].equals(original["TARGET"])
    assert not np.isinf(featured.select_dtypes(include=np.number).to_numpy()).any()
    assert not set(BENCHMARK_ONLY_ENGINEERED_FEATURES).intersection(featured.columns)

    benchmark = create_benchmark_features(synthetic_applications)
    assert set(BENCHMARK_ONLY_ENGINEERED_FEATURES).issubset(benchmark.columns)


def test_division_by_zero_becomes_missing(synthetic_applications):
    sample = synthetic_applications.head(1).copy()
    sample["AMT_INCOME_TOTAL"] = 0
    sample["CNT_FAM_MEMBERS"] = 0
    featured = create_features(sample)
    assert np.isnan(featured.loc[sample.index[0], "CREDIT_INCOME_RATIO"])
    assert np.isnan(featured.loc[sample.index[0], "INCOME_PER_PERSON"])
