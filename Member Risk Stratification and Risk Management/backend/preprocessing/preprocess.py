import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Features chosen for PoC (explainable + useful)
SELECTED_FEATURES = [
    "age",
    "time_in_hospital",
    "number_inpatient",
    "number_emergency",
    "num_medications",
    "number_diagnoses",
    "insulin"
]

TARGET_COLUMN = "readmitted"


def preprocess_data(df: pd.DataFrame):
    """
    Cleans and prepares data for modeling.
    Returns:
        X : feature matrix
        y : target vector
        encoders : fitted label encoders
    """

    df = df.copy()

    # 1. Replace '?' with NaN
    df.replace("?", pd.NA, inplace=True)

    # 2. Drop rows with missing critical values
    df = df.dropna(subset=SELECTED_FEATURES + [TARGET_COLUMN])

    # 3. Define target variable
    # <30 days readmission = high risk
    df["risk_target"] = df[TARGET_COLUMN].apply(
        lambda x: 1 if x == "<30" else 0
    )

    # 4. Encode categorical variables
    encoders = {}

    for col in SELECTED_FEATURES:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # 5. Feature matrix & target
    X = df[SELECTED_FEATURES]
    y = df["risk_target"]

    return X, y, encoders
