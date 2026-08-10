from __future__ import annotations

import pytest

from src.components.data_loader import load_training_data


def test_load_training_data_accepts_path_override(tmp_path, synthetic_applications):
    test_file = tmp_path / "sample.csv"
    synthetic_applications.to_csv(test_file, index=False)
    loaded = load_training_data(test_file)
    assert loaded.shape == synthetic_applications.shape
    assert {"TARGET", "SK_ID_CURR"}.issubset(loaded.columns)


def test_load_training_data_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_data(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        load_training_data(empty)
