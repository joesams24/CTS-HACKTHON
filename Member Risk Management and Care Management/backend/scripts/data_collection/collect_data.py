# backend/scripts/data_collection/collect_data.py

import os
import pandas as pd
from scripts.utils.helpers import setup_logging

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Paths
SYNTH_DATA_PATH = os.path.join(BASE_DIR, "backend/data/raw/synthea_csv")
REAL_DATA_PATH = os.path.join(BASE_DIR, "backend/data/raw/real_member_data")
COMBINED_DATA_PATH = os.path.join(BASE_DIR, "backend/data/processed/combined")
LOG_DIR = os.path.join(BASE_DIR, "backend/logs")

# Ensure log and output directories exist
os.makedirs(COMBINED_DATA_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging(LOG_DIR, "data_collection")

# Required Synthea CSV files
REQUIRED_SYNTHEA_FILES = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "medications.csv",
    "observations.csv",
    "procedures.csv",
]

# -----------------------------
# Helper Functions
# -----------------------------
def verify_synthea_files(path: str) -> bool:
    """Check if all required Synthea CSV files exist."""
    if not os.path.exists(path):
        logger.error(f"Synthea folder not found: {path}")
        return False

    missing_files = [
        f for f in REQUIRED_SYNTHEA_FILES if not os.path.isfile(os.path.join(path, f))
    ]
    if missing_files:
        logger.error(f"Missing Synthea CSV files: {missing_files}")
        return False

    logger.info("All required Synthea CSV files are present.")
    return True


def load_csv_files(path: str) -> dict:
    """Load all CSV files from a directory into a dictionary of DataFrames."""
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(path, file)
        try:
            df = pd.read_csv(file_path)
            dataframes[file] = df
            logger.info(f"Loaded '{file}' with {len(df)} records and {len(df.columns)} columns.")
        except Exception as e:
            logger.warning(f"Error loading '{file}': {e}")
    return dataframes


def save_combined_data(data_dict: dict, output_path: str):
    """Save loaded data to the combined folder."""
    os.makedirs(output_path, exist_ok=True)
    for filename, df in data_dict.items():
        out_file = os.path.join(output_path, filename)
        df.to_csv(out_file, index=False)
        logger.info(f"Saved combined file '{out_file}'.")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    logger.info("===== Starting Data Collection Script =====")

    synthea_data = {}
    real_member_data = {}

    # Verify and load Synthea CSVs
    if verify_synthea_files(SYNTH_DATA_PATH):
        synthea_data = load_csv_files(SYNTH_DATA_PATH)
    else:
        logger.error("Cannot proceed without all required Synthea CSV files.")
        exit(1)

    # Load real member data if available
    if os.path.exists(REAL_DATA_PATH):
        real_member_data = load_csv_files(REAL_DATA_PATH)
        logger.info("Real member data found and loaded.")
    else:
        logger.warning(f"No real member data found at {REAL_DATA_PATH}. Proceeding with synthetic data only.")

    # Merge datasets if real data exists
    combined_data = synthea_data.copy()
    if real_member_data:
        combined_data.update(real_member_data)
        logger.info("Combined Synthea and real member datasets.")

    # Save combined data
    save_combined_data(combined_data, COMBINED_DATA_PATH)

    logger.info("===== Data Collection Script Completed Successfully =====")
