# backend/scripts/preprocessing/preprocess_data.py

import os
import pandas as pd
import numpy as np
import gc
from scripts.utils.helpers import setup_logging

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(__file__)
RAW_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/raw/synthea_csv"))
OUTPUT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
LOG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../logs"))

# Ensure directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Logging setup
# -----------------------------
logger = setup_logging(LOG_DIR, "preprocessing")

logger.info(f"RAW_PATH = {RAW_PATH}")
if not os.path.exists(RAW_PATH):
    logger.error(f"RAW_PATH does not exist! Check folder location: {RAW_PATH}")
    exit(1)

logger.info("===== Starting Preprocessing Script =====")

# -----------------------------
# Load CSV files
# -----------------------------
logger.info("Loading CSV files...")

patients = pd.read_csv(
    os.path.join(RAW_PATH, "patients.csv"),
    usecols=["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY", "ZIP", "INCOME"],
    low_memory=False
)

encounters = pd.read_csv(
    os.path.join(RAW_PATH, "encounters.csv"),
    usecols=["Id", "START", "PATIENT"],
    low_memory=False
)

observations = pd.read_csv(
    os.path.join(RAW_PATH, "observations.csv"),
    usecols=["PATIENT", "CODE", "DESCRIPTION", "VALUE"],
    low_memory=False
)

# -----------------------------
# Cleaning & Transformation
# -----------------------------
logger.info("Cleaning and transforming data...")
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
if pd.api.types.is_datetime64tz_dtype(encounters["START"]):
    encounters["START"] = encounters["START"].dt.tz_convert(None)
observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

patients.dropna(subset=["Id"], inplace=True)
encounters.dropna(subset=["PATIENT"], inplace=True)
observations.dropna(subset=["PATIENT"], inplace=True)
gc.collect()

# -----------------------------
# Temporal Consistency
# -----------------------------
logger.info("Performing temporal consistency checks...")
temp = encounters.merge(
    patients[["Id", "BIRTHDATE"]],
    left_on="PATIENT",
    right_on="Id",
    how="left",
    suffixes=("_ENC", "_PAT")
)
temp = temp[temp["START"] >= temp["BIRTHDATE"]]
encounters = temp[["Id_ENC", "START", "PATIENT"]].rename(columns={"Id_ENC": "ENCOUNTER_ID"}).copy()
del temp
gc.collect()

# -----------------------------
# Aggregate Observations
# -----------------------------
logger.info("Aggregating observations per patient (mean)...")
observations_agg = (
    observations.groupby("PATIENT", observed=True)["VALUE"]
    .mean()
    .reset_index()
    .rename(columns={"VALUE": "LAB_VALUE"})
)
del observations
gc.collect()

# -----------------------------
# Merge Static Features
# -----------------------------
logger.info("Merging patients, encounters, and observations...")
df = pd.merge(patients, encounters, left_on="Id", right_on="PATIENT", how="left")
df = pd.merge(df, observations_agg, left_on="Id", right_on="PATIENT", how="left")
del patients, encounters, observations_agg
gc.collect()

# -----------------------------
# Feature Engineering
# -----------------------------
logger.info("Performing feature engineering...")
df["AGE"] = (pd.Timestamp.now() - df["BIRTHDATE"]).dt.days // 365
for col in ["GENDER", "RACE", "ETHNICITY"]:
    df[col] = df[col].astype("category").cat.codes
for col in ["LAB_VALUE", "AGE", "INCOME"]:
    df[col].fillna(df[col].median(), inplace=True)

# Encounter stats
logger.info("Computing enhanced features...")
encounter_count = df.groupby("Id", observed=True)["ENCOUNTER_ID"].count().reset_index()
encounter_count.rename(columns={"ENCOUNTER_ID": "NUM_ENCOUNTERS"}, inplace=True)
df = pd.merge(df, encounter_count, on="Id", how="left")

df_sorted = df.sort_values(["Id", "START"])
avg_gap = df_sorted.groupby("Id")["START"].diff().dt.days.groupby(df_sorted["Id"]).mean().reset_index()
avg_gap.rename(columns={"START": "AVG_DAYS_BETWEEN_ENC"}, inplace=True)
df = pd.merge(df, avg_gap, on="Id", how="left")

# -----------------------------
# Static Features
# -----------------------------
logger.info("Creating static features...")
drop_cols = ["START", "ENCOUNTER_ID", "PATIENT", "LAB_VALUE"]
static_features = df.drop(columns=[c for c in drop_cols if c in df.columns]).drop_duplicates("Id")
static_file = os.path.join(OUTPUT_PATH, "static_features.csv")
static_features.to_csv(static_file, index=False)
logger.info(f"Saved static features to {static_file}")

# -----------------------------
# Time Window Sequences (30/60/90 Days)
# -----------------------------
logger.info("Creating 30/60/90-day sequence features...")
WINDOWS = [30, 60, 90]

def create_sequences(group, window):
    last_date = group["START"].max()
    cutoff = last_date - pd.Timedelta(days=window)
    seq = group.loc[group["START"] >= cutoff].sort_values("START")["LAB_VALUE"].values
    seq_len = 5
    if len(seq) == 0:
        seq = np.zeros(seq_len)
    elif len(seq) < seq_len:
        seq = np.pad(seq, (0, seq_len - len(seq)), "constant", constant_values=0)
    else:
        seq = seq[-seq_len:]
    return pd.Series(seq, index=[f"SEQ_VAL_{i+1}" for i in range(seq_len)])

for window in WINDOWS:
    seq_df = df.groupby("Id", observed=True).apply(create_sequences, window=window).reset_index()
    seq_file = os.path.join(OUTPUT_PATH, f"sequences_{window}.csv")
    seq_df.to_csv(seq_file, index=False)
    logger.info(f"Saved {window}-day sequence file: {seq_file}")

# -----------------------------
# Cleanup
# -----------------------------
logger.info("Preprocessing complete. All files saved to OUTPUT_PATH")
logger.info("===== Finished Successfully =====")
