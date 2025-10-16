# backend/scripts/preprocessing/prepare_targets.py

import os
import pandas as pd
from scripts.utils.helpers import setup_logging, calculate_risk_tier

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(__file__)
PREPROCESSED_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
TARGETS_PATH = os.path.join(PREPROCESSED_PATH, "targets")
LOG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../logs"))

os.makedirs(TARGETS_PATH, exist_ok=True)
logger = setup_logging(LOG_DIR, "prepare_targets")

logger.info("===== Starting Target Preparation Script =====")

WINDOWS = [30, 60, 90]

# Load static features
static_file = os.path.join(PREPROCESSED_PATH, "static_features.csv")
static_features = pd.read_csv(static_file)
# Normalize columns
static_features.columns = static_features.columns.str.strip().str.upper()
logger.info(f"Loaded static features: {static_file} ({len(static_features)} rows)")

# Compute age if birthdate exists
if "BIRTHDATE" in static_features.columns:
    static_features["BIRTHDATE"] = pd.to_datetime(static_features["BIRTHDATE"], errors="coerce")
    current_year = pd.Timestamp.now().year
    static_features["AGE"] = current_year - static_features["BIRTHDATE"].dt.year
    static_features.drop(columns=["BIRTHDATE"], inplace=True)

# Ensure ID column is uppercase
if "ID" not in static_features.columns:
    raise ValueError("static_features.csv must have 'ID' column")

for window in WINDOWS:
    seq_file = os.path.join(PREPROCESSED_PATH, f"sequences_{window}.csv")
    sequences = pd.read_csv(seq_file)
    sequences.columns = sequences.columns.str.strip().str.upper()
    logger.info(f"Loaded {window}-day sequences: {seq_file} ({len(sequences)} rows)")

    # -----------------------------
    # Compute risk score from sequences
    # -----------------------------
    seq_cols = [f"SEQ_VAL_{i+1}" for i in range(5)]
    seq_cols = [col.upper() for col in seq_cols if col.upper() in sequences.columns]
    sequences["SEQ_SUM"] = sequences[seq_cols].sum(axis=1)
    sequences["SEQ_NORM"] = (sequences["SEQ_SUM"] - sequences["SEQ_SUM"].min()) / \
                            (sequences["SEQ_SUM"].max() - sequences["SEQ_SUM"].min() + 1e-6)

    # -----------------------------
    # Merge age contribution
    # -----------------------------
    if "AGE" in static_features.columns:
        sequences = sequences.merge(static_features[["ID", "AGE"]], on="ID", how="left")
        sequences["AGE_SCORE"] = sequences["AGE"] / (sequences["AGE"].max() + 1e-6)
    else:
        sequences["AGE_SCORE"] = 0

    # -----------------------------
    # Final risk score
    # -----------------------------
    sequences["RISK_SCORE"] = 0.7 * sequences["SEQ_NORM"] + 0.3 * sequences["AGE_SCORE"]

    # -----------------------------
    # Convert numeric score to categorical risk tier
    # -----------------------------
    sequences["RISK_TIER"] = sequences["RISK_SCORE"].apply(calculate_risk_tier)

    # Ensure all 5 classes exist: Minimal, Low, Moderate, High, Very High
    for tier in ["Minimal", "Low", "Moderate", "High", "Very High"]:
        if tier not in sequences["RISK_TIER"].unique():
            logger.warning(f"Adding dummy row to ensure '{tier}' exists in {window}-day target")
            sequences = sequences.append({"ID": -1, "RISK_SCORE": 0.0, "RISK_TIER": tier}, ignore_index=True)

    # -----------------------------
    # Save targets
    # -----------------------------
    target_file = os.path.join(TARGETS_PATH, f"target_{window}.csv")
    sequences[["ID", "RISK_SCORE", "RISK_TIER"]].to_csv(target_file, index=False)
    logger.info(f"Saved {window}-day target file: {target_file}")

logger.info("===== Target Preparation Completed Successfully =====")
