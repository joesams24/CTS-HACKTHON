# backend/scripts/data_collection/collect_data.py

import os
import pandas as pd
from datetime import datetime
from scripts.utils.helpers import log_message

# -----------------------------
# Configuration
# -----------------------------
SYNTH_DATA_PATH = "../../data/raw/synthea_csv"
REAL_DATA_PATH = "../../data/raw/real_member_data"
LOG_FILE = "../../logs/data_collection.log"  # Central logs folder outside backend

# Required Synthea CSV files
required_synthea_files = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "medications.csv",
    "observations.csv",
    "procedures.csv"
]

# -----------------------------
# Helper Functions
# -----------------------------
def verify_synthea_files(path):
    missing_files = []
    for file in required_synthea_files:
        if not os.path.isfile(os.path.join(path, file)):
            missing_files.append(file)
    if missing_files:
        log_message(f"Missing Synthea CSV files: {missing_files}", log_file=LOG_FILE)
        return False
    log_message("All required Synthea CSV files are present.", log_file=LOG_FILE)
    return True

def load_csv_files(path):
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(path, file))
        dataframes[file] = df
        log_message(f"Loaded '{file}' with {len(df)} records and {len(df.columns)} columns.", log_file=LOG_FILE)
    return dataframes

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    log_message("===== Starting Data Collection Script =====", log_file=LOG_FILE)

    # Verify Synthea CSVs
    if verify_synthea_files(SYNTH_DATA_PATH):
        synthea_data = load_csv_files(SYNTH_DATA_PATH)
    else:
        log_message("Error: Cannot proceed without all required Synthea CSV files.", log_file=LOG_FILE)
        exit(1)

    # Load real member data
    if os.path.exists(REAL_DATA_PATH):
        real_member_data = load_csv_files(REAL_DATA_PATH)
    else:
        log_message(f"No real member data found at {REAL_DATA_PATH}. Proceeding with synthetic data only.", log_file=LOG_FILE)

    log_message("===== Data Collection Script Completed =====", log_file=LOG_FILE)