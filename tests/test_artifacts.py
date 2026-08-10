import pytest

from src.utils.artifacts import load_joblib, load_json, save_joblib, save_json


def test_artifact_round_trip_and_missing_errors(tmp_path):
    object_path = save_joblib({"model": "tiny"}, tmp_path / "models" / "model.joblib")
    json_path = save_json({"metric": 0.75}, tmp_path / "metrics" / "metrics.json")
    assert load_joblib(object_path) == {"model": "tiny"}
    assert load_json(json_path) == {"metric": 0.75}
    with pytest.raises(FileNotFoundError, match="Run the training pipeline"):
        load_joblib(tmp_path / "missing.joblib")
