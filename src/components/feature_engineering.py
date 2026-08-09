import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered credit-risk features from Home Credit applicant data.

    The function returns a copy of the input dataframe with additional
    features and does not mutate the original dataframe.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Employment anomaly handling
    # ---------------------------------------------------------

    df["DAYS_EMPLOYED_ANOMALY"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    df["DAYS_EMPLOYED_CLEAN"] = (
        df["DAYS_EMPLOYED"]
        .replace(365243, np.nan)
    )

    # ---------------------------------------------------------
    # Applicant age and employment history
    # ---------------------------------------------------------

    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED_CLEAN"] / 365.25
    )

    df["EMPLOYMENT_AGE_RATIO"] = (
        df["EMPLOYMENT_YEARS"]
        / df["AGE_YEARS"]
    )

    # ---------------------------------------------------------
    # Credit affordability features
    # ---------------------------------------------------------

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"]
        / df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_ANNUITY_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_ANNUITY"]
    )

    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_GOODS_PRICE"]
    )

    # ---------------------------------------------------------
    # Household affordability features
    # ---------------------------------------------------------

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"]
        / df["CNT_FAM_MEMBERS"]
    )

    df["CREDIT_PER_PERSON"] = (
        df["AMT_CREDIT"]
        / df["CNT_FAM_MEMBERS"]
    )

    df["ANNUITY_PER_PERSON"] = (
        df["AMT_ANNUITY"]
        / df["CNT_FAM_MEMBERS"]
    )

    df["CHILDREN_FAMILY_RATIO"] = (
        df["CNT_CHILDREN"]
        / df["CNT_FAM_MEMBERS"]
    )

    # ---------------------------------------------------------
    # External credit-score features
    # ---------------------------------------------------------

    external_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    df["EXT_SOURCE_MEAN"] = (
        df[external_columns]
        .mean(axis=1)
    )

    df["EXT_SOURCE_MIN"] = (
        df[external_columns]
        .min(axis=1)
    )

    df["EXT_SOURCE_MAX"] = (
        df[external_columns]
        .max(axis=1)
    )

    df["EXT_SOURCE_STD"] = (
        df[external_columns]
        .std(axis=1)
    )

    df["EXT_SOURCE_COUNT"] = (
        df[external_columns]
        .notna()
        .sum(axis=1)
    )

    # ---------------------------------------------------------
    # Document features
    # ---------------------------------------------------------

    document_columns = [
        column
        for column in df.columns
        if column.startswith("FLAG_DOCUMENT_")
    ]

    if document_columns:
        df["DOCUMENT_COUNT"] = (
            df[document_columns]
            .sum(axis=1)
        )
    else:
        df["DOCUMENT_COUNT"] = 0

    # ---------------------------------------------------------
    # Contact-information features
    # ---------------------------------------------------------

    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    available_contact_columns = [
        column
        for column in contact_columns
        if column in df.columns
    ]

    if available_contact_columns:
        df["CONTACT_COUNT"] = (
            df[available_contact_columns]
            .sum(axis=1)
        )
    else:
        df["CONTACT_COUNT"] = 0

    # ---------------------------------------------------------
    # Social-circle risk features
    # ---------------------------------------------------------

    if {
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
    }.issubset(df.columns):

        df["DEF_30_SOCIAL_RATIO"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"]
            / df["OBS_30_CNT_SOCIAL_CIRCLE"]
        )

    if {
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
    }.issubset(df.columns):

        df["DEF_60_SOCIAL_RATIO"] = (
            df["DEF_60_CNT_SOCIAL_CIRCLE"]
            / df["OBS_60_CNT_SOCIAL_CIRCLE"]
        )

    # ---------------------------------------------------------
    # Clean division edge cases
    # ---------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df