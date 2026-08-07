import pandas as pd

from src.components.data_loader import (
    load_training_data,
)


def test_load_training_data(tmp_path):

    test_file = tmp_path / "sample.csv"

    sample_data = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "TARGET": [0, 1],
            "AMT_INCOME_TOTAL": [
                100000,
                150000,
            ],
        }
    )

    sample_data.to_csv(
        test_file,
        index=False,
    )

    df = load_training_data(
        test_file
    )

    assert len(df) == 2
    assert "TARGET" in df.columns
    assert "SK_ID_CURR" in df.columns