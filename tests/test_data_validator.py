from src.components.data_validator import validate_training_data


def test_validator_separates_fatal_errors_and_diagnostics(synthetic_applications):
    report = validate_training_data(synthetic_applications)
    assert report.is_valid
    assert report.statistics["days_employed_anomaly_count"] == 1
    assert "EXT_SOURCE_1" in report.statistics["major_missingness"]

    invalid = synthetic_applications.copy()
    invalid.loc[1, "SK_ID_CURR"] = invalid.loc[0, "SK_ID_CURR"]
    invalid.loc[2, "TARGET"] = 3
    invalid_report = validate_training_data(invalid)
    assert not invalid_report.is_valid
    assert any("duplicate" in error for error in invalid_report.fatal_errors)
    assert any("binary" in error for error in invalid_report.fatal_errors)
