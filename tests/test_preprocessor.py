from src.components.data_preprocessor import build_preprocessor_for_frame, prepare_model_features


def test_preprocessor_handles_missing_and_unknown_categories(synthetic_applications):
    train = prepare_model_features(synthetic_applications.iloc[:100])
    validation = prepare_model_features(synthetic_applications.iloc[100:].copy())
    validation.loc[validation.index[0], "NAME_CONTRACT_TYPE"] = "Previously unseen"
    preprocessor = build_preprocessor_for_frame(train, scale_numeric=True)
    train_matrix = preprocessor.fit_transform(train)
    validation_matrix = preprocessor.transform(validation)
    assert train_matrix.shape[0] == 100
    assert validation_matrix.shape[0] == 20
    assert validation_matrix.shape[1] == train_matrix.shape[1]
