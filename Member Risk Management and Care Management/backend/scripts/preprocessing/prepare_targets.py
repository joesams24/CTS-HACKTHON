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
logger.info(f"Loaded static features: {static_file} ({len(static_features)} rows)")

for window in WINDOWS:
    seq_file = os.path.join(PREPROCESSED_PATH, f"sequences_{window}.csv")
    sequences = pd.read_csv(seq_file)
    logger.info(f"Loaded {window}-day sequences: {seq_file} ({len(sequences)} rows)")

    # Example target computation: sum of sequence values as risk score
    # (replace with your actual risk computation logic)
    sequences["RISK_SCORE"] = sequences[[f"SEQ_VAL_{i+1}" for i in range(5)]].sum(axis=1)

    # Convert numeric score to categorical risk tier
    sequences["RISK_TIER"] = sequences["RISK_SCORE"].apply(calculate_risk_tier)

    target_file = os.path.join(TARGETS_PATH, f"target_{window}.csv")
    sequences[["Id", "RISK_SCORE", "RISK_TIER"]].to_csv(target_file, index=False)
    logger.info(f"Saved {window}-day target file: {target_file}")

logger.info("===== Target Preparation Completed Successfully =====")