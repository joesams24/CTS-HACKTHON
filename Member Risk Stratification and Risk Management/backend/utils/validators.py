import pandas as pd
from schemas.input_schema import REQUIRED_COLUMNS


def validate_csv(file) -> pd.DataFrame:
    """
    Validates uploaded CSV file:
    - Checks readable CSV
    - Checks required columns
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        raise ValueError("Uploaded file is not a valid CSV")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df
