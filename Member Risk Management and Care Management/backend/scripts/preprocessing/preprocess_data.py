# backend/scripts/preprocessing/preprocess_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
from scripts.utils.helpers import log_message, save_pickle

# -----------------------------
# Configuration
# -----------------------------
RAW_DATA_PATH = "../../data/raw"
PROCESSED_DATA_PATH = "../../data/processed"
NUMERIC_COLUMNS = ["age", "num_visits", "lab_value"]
CATEGORICAL_COLUMNS = ["gender", "region", "membership_type"]
VALUE_COLUMNS = ["num_visits", "lab_value"]  # For LSTM sequences

# -----------------------------
# Helper Functions
# -----------------------------
def clean_data(df):
    """Handle missing values and duplicates"""
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna("Unknown")
    return df

def encode_features(df):
    """Encode categorical columns"""
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def normalize_features(df):
    """Normalize numeric columns"""
    scaler = MinMaxScaler()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[[col]] = scaler.fit_transform(df[[col]])
    return df

def generate_temporal_sequences(df, member_id_col, date_col, value_cols, window_days=30):
    """Create rolling sequences per member"""
    sequences = []
    for member_id, group in df.groupby(member_id_col):
        group = group.sort_values(date_col)
        values = group[value_cols].values
        
        # -----------------------------
        # Feature engineering per sequence
        # -----------------------------
        # Rolling averages
        rolling_avg = pd.DataFrame(values).rolling(window=window_days, min_periods=1).mean().values
        # Slopes / trend (difference between last and first in window)
        slope = (values[-1] - values[0]) / max(len(values)-1, 1)
        # Count of events (non-zero visits)
        event_count = np.count_nonzero(values[:, 0])  # assuming first column is num_visits

        if len(values) >= window_days:
            seq = values[-window_days:]
            # Combine engineered features (can append to sequence or save separately)
            seq_features = {
                "member_id": member_id,
                "sequence": seq,
                "rolling_avg": rolling_avg[-1].tolist(),
                "slope": slope.tolist() if isinstance(slope, np.ndarray) else [slope],
                "event_count": event_count
            }
            sequences.append(seq_features)
    return sequences

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    log_message("===== Starting Preprocessing Script =====", log_file="../../logs/preprocessing.log")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    synthea_path = os.path.join(RAW_DATA_PATH, "synthea_csv")
    real_path = os.path.join(RAW_DATA_PATH, "real_member_data")

    patients = pd.read_csv(os.path.join(synthea_path, "patients.csv"))
    visits = pd.read_csv(os.path.join(synthea_path, "encounters.csv"))
    df = pd.merge(patients, visits, left_on="id", right_on="patient", how="left")

    df = clean_data(df)
    df = encode_features(df)
    df = normalize_features(df)

    # Generate sequences with engineered features
    sequences_30 = generate_temporal_sequences(df, "id", "start", VALUE_COLUMNS, window_days=30)
    sequences_60 = generate_temporal_sequences(df, "id", "start", VALUE_COLUMNS, window_days=60)
    sequences_90 = generate_temporal_sequences(df, "id", "start", VALUE_COLUMNS, window_days=90)

    # Save processed sequences
    save_pickle(sequences_30, os.path.join(PROCESSED_DATA_PATH, "sequences_30.pkl"))
    save_pickle(sequences_60, os.path.join(PROCESSED_DATA_PATH, "sequences_60.pkl"))
    save_pickle(sequences_90, os.path.join(PROCESSED_DATA_PATH, "sequences_90.pkl"))

    # Save static features for XGBoost
    static_features = df.drop(columns=["start", "num_visits", "lab_value"])
    save_pickle(static_features, os.path.join(PROCESSED_DATA_PATH, "static_features.pkl"))

    log_message("===== Preprocessing Script Completed =====", log_file="../../logs/preprocessing.log")
