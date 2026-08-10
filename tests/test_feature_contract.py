import pandas as pd
import pytest

from src.components.data_preprocessor import prepare_model_features
from src.components.feature_contract import (
    APPLICATION_ID_PATTERN,
    EXTERNAL_UNAVAILABLE_FEATURES,
    HISTORICAL_IDENTIFIER_FEATURES,
    PRODUCTION_ENGINEERED_FEATURES,
    PRODUCTION_RAW_FEATURES,
    application_to_model_input,
    friendly_explanation_name,
    generate_application_id,
    normalize_application_frame,
)


FRIENDLY_APPLICATION = {
    "age": 34,
    "years_employed": 5,
    "family_members": 2,
    "number_of_children": 0,
    "annual_income": 202500,
    "requested_loan_amount": 406597.5,
    "loan_annuity": 24700.5,
    "goods_purchase_price": 351000,
    "credit_product_type": "Cash loans",
    "income_type": "Working",
    "housing_situation": "House / apartment",
    "owns_car": "No",
    "owns_property": "Yes",
}


def test_deployable_contract_excludes_unavailable_and_historical_features():
    assert not set(EXTERNAL_UNAVAILABLE_FEATURES).intersection(PRODUCTION_RAW_FEATURES)
    assert not set(HISTORICAL_IDENTIFIER_FEATURES).intersection(PRODUCTION_RAW_FEATURES)
    assert not any(name.startswith("EXT_SOURCE") for name in PRODUCTION_ENGINEERED_FEATURES)


def test_ui_translation_and_deterministic_engineering():
    raw = application_to_model_input(FRIENDLY_APPLICATION)
    assert raw["DAYS_BIRTH"] == int(-34 * 365.25)
    assert raw["DAYS_EMPLOYED"] == int(-5 * 365.25)
    assert raw["FLAG_OWN_CAR"] == "N"
    assert raw["FLAG_OWN_REALTY"] == "Y"
    model_frame = prepare_model_features(pd.DataFrame([raw]))
    assert "SK_ID_CURR" not in model_frame
    assert "application_id" not in model_frame
    assert not any(column.startswith("EXT_SOURCE") for column in model_frame)
    assert friendly_explanation_name("categorical__NAME_INCOME_TYPE_Working") == "Income Type: Working"


def test_application_id_format_and_input_validation():
    first = generate_application_id()
    second = generate_application_id()
    assert APPLICATION_ID_PATTERN.fullmatch(first)
    assert first != second
    with pytest.raises(ValueError, match="plausible"):
        application_to_model_input({**FRIENDLY_APPLICATION, "age": 20, "years_employed": 10})


def test_historical_source_id_is_not_part_of_normalized_contract():
    raw = application_to_model_input(FRIENDLY_APPLICATION)
    normalized = normalize_application_frame(pd.DataFrame([{**raw, "SK_ID_CURR": 100002}]))
    assert list(normalized.columns) == list(PRODUCTION_RAW_FEATURES)
    assert "SK_ID_CURR" not in normalized
