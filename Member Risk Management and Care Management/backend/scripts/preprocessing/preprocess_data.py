import pandas as pd
import numpy as np
import gc
import os
from datetime import datetime
from scripts.utils.helpers import log_message

# -----------------------------
# Paths
# -----------------------------
RAW_PATH = r"E:\Member Stratification and Care Management\backend\data\raw\synthea_csv"
OUTPUT_PATH = r"E:\Member Stratification and Care Management\backend\data\processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Ensure logs folder exists
LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOGS_PATH, exist_ok=True)
PREPROCESS_LOG_FILE = os.path.join(LOGS_PATH, "preprocessing.log")

def log(msg):
    """Helper to log to console and preprocessing log file"""
    log_message(msg, log_file=PREPROCESS_LOG_FILE)

log("===== Starting Preprocessing Script =====")

# -----------------------------
# Load CSV files
# -----------------------------
log("Loading CSV files...")

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
log("Cleaning and transforming data...")
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
log("Performing temporal consistency checks...")
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
log("Aggregating observations per patient (mean)...")
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
log("Merging patients, encounters, and observations...")
df = pd.merge(patients, encounters, left_on="Id", right_on="PATIENT", how="left")
df = pd.merge(df, observations_agg, left_on="Id", right_on="PATIENT", how="left")
del patients, encounters, observations_agg
gc.collect()

# -----------------------------
# Feature Engineering
# -----------------------------
log("Performing feature engineering...")
df["AGE"] = (pd.Timestamp.now() - df["BIRTHDATE"]).dt.days // 365
for col in ["GENDER", "RACE", "ETHNICITY"]:
    df[col] = df[col].astype("category").cat.codes
for col in ["LAB_VALUE", "AGE", "INCOME"]:
    df[col].fillna(df[col].median(), inplace=True)

# Enhanced features
log("Computing enhanced features...")
encounter_count = df.groupby("Id", observed=True)["ENCOUNTER_ID"].count().reset_index()
encounter_count.rename(columns={"ENCOUNTER_ID": "NUM_ENCOUNTERS"}, inplace=True)
df = pd.merge(df, encounter_count, on="Id", how="left")
df_sorted = df.sort_values(["Id", "START"])
avg_gap = df_sorted.groupby("Id")["START"].diff().dt.days.groupby(df_sorted["Id"]).mean().reset_index()
avg_gap.rename(columns={"START": "AVG_DAYS_BETWEEN_ENC"}, inplace=True)
df = pd.merge(df, avg_gap, on="Id", how="left")

# -----------------------------
# 5-Recent Observation Sequences
# -----------------------------
log("Creating 5-recent observation sequences...")
VALUE_COLUMNS = ["LAB_VALUE"]

def create_sequences(group):
    values = group.sort_values("START")[VALUE_COLUMNS].values.flatten()
    if len(values) == 0:
        values = np.zeros(5)
    elif len(values) < 5:
        values = np.pad(values, (0, 5 - len(values)), "constant", constant_values=0)
    else:
        values = values[-5:]
    return values

sequences = df.groupby("Id", observed=True).apply(create_sequences).apply(pd.Series)
sequences.columns = [f"SEQ_VAL_{i+1}" for i in range(sequences.shape[1])]
sequences_file = os.path.join(OUTPUT_PATH, "sequences_5.csv")
sequences.to_csv(sequences_file, index=True)
log(f"Saved 5-recent sequences to {sequences_file}")

# -----------------------------
# 30/60/90-Day Window Features
# -----------------------------
log("Creating 30/60/90-day LAB_VALUE window features...")
WINDOWS = [30, 60, 90]

def create_windowed_features(group):
    last_date = group["START"].max()
    features = []
    for days in WINDOWS:
        mask = group["START"] >= last_date - pd.Timedelta(days=days)
        values = group.loc[mask, "LAB_VALUE"].values
        features.append(values.mean() if len(values) > 0 else 0)
    return pd.Series(features, index=[f"LAB_LAST_{d}D" for d in WINDOWS])

windowed_features = df.groupby("Id").apply(create_windowed_features).reset_index()
windowed_file = os.path.join(OUTPUT_PATH, "windowed_features_30_60_90.csv")
windowed_features.to_csv(windowed_file, index=False)
log(f"Saved 30/60/90-day window features to {windowed_file}")

# -----------------------------
# Static Features
# -----------------------------
log("Creating static features...")
drop_cols = [c for c in VALUE_COLUMNS + ["START", "ENCOUNTER_ID", "PATIENT"] if c in df.columns]
static_features = df.drop(columns=drop_cols).drop_duplicates("Id")
static_file = os.path.join(OUTPUT_PATH, "static_features.csv")
static_features.to_csv(static_file, index=False)
log(f"Saved static features to {static_file}")

log("Preprocessing complete. Files saved to OUTPUT_PATH")
log("===== Finished Successfully =====")
