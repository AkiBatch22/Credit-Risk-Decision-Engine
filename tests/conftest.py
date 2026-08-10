from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_applications() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 120
    target = np.tile([0, 0, 0, 0, 1], rows // 5)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + rows),
            "TARGET": target,
            "DAYS_BIRTH": -rng.integers(8000, 25000, rows),
            "DAYS_EMPLOYED": -rng.integers(100, 8000, rows),
            "AMT_INCOME_TOTAL": rng.uniform(60000, 500000, rows),
            "AMT_CREDIT": rng.uniform(50000, 1500000, rows),
            "AMT_ANNUITY": rng.uniform(5000, 80000, rows),
            "AMT_GOODS_PRICE": rng.uniform(50000, 1400000, rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, rows).astype(float),
            "CNT_CHILDREN": rng.integers(0, 4, rows),
            "EXT_SOURCE_1": np.clip(rng.normal(0.55 - target * 0.20, 0.12), 0.01, 0.99),
            "EXT_SOURCE_2": np.clip(rng.normal(0.55 - target * 0.18, 0.12), 0.01, 0.99),
            "EXT_SOURCE_3": np.clip(rng.normal(0.55 - target * 0.16, 0.12), 0.01, 0.99),
            "NAME_CONTRACT_TYPE": np.where(rng.random(rows) > 0.2, "Cash loans", "Revolving loans"),
            "NAME_INCOME_TYPE": np.where(rng.random(rows) > 0.3, "Working", "Commercial associate"),
            "NAME_HOUSING_TYPE": np.where(rng.random(rows) > 0.2, "House / apartment", "Rented apartment"),
            "FLAG_OWN_CAR": np.where(rng.random(rows) > 0.6, "Y", "N"),
            "FLAG_OWN_REALTY": np.where(rng.random(rows) > 0.3, "Y", "N"),
            "CODE_GENDER": np.where(rng.random(rows) > 0.5, "F", "M"),
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": rng.integers(0, 2, rows),
            "FLAG_WORK_PHONE": rng.integers(0, 2, rows),
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": rng.integers(0, 2, rows),
            "FLAG_EMAIL": rng.integers(0, 2, rows),
            "FLAG_DOCUMENT_3": rng.integers(0, 2, rows),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 8, rows).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, rows).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 8, rows).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, rows).astype(float),
        }
    )
    frame.loc[0, "DAYS_EMPLOYED"] = 365243
    frame.loc[:70, "EXT_SOURCE_1"] = np.nan
    return frame
