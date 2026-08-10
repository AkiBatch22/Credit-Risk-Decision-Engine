"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TRAIN_DATA_FILE


def load_training_data(file_path: str | Path = TRAIN_DATA_FILE, **read_csv_kwargs: object) -> pd.DataFrame:
    """Load a non-empty Home Credit training CSV from an overridable path."""

    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"Training dataset not found at: {source}")
    if not source.is_file():
        raise ValueError(f"Training data path is not a file: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Training dataset is empty: {source}")

    try:
        dataframe = pd.read_csv(source, **read_csv_kwargs)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Training dataset has no readable columns: {source}") from error

    if dataframe.empty:
        raise ValueError(f"Training dataset contains no rows: {source}")
    return dataframe
