import pandas as pd

from src.config import TRAIN_DATA_FILE


def load_training_data(
    file_path=TRAIN_DATA_FILE,
) -> pd.DataFrame:
    """
    Load the Home Credit training dataset.

    Parameters
    ----------
    file_path:
        Path to the training CSV.

    Returns
    -------
    pd.DataFrame
        Loaded training data.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    return df