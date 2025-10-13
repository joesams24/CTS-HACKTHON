import pandas as pd
import numpy as np
import gc
import os
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===== Starting Preprocessing Script =====")

# ===== PATHS =====
RAW_PATH = r"E:\Member Stratification and Care Management\backend\data\raw\synthea_csv"
OUTPUT_PATH = r"E:\Member Stratification and Care Management\backend\data\processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ===== LOAD CSV FILES =====
print("Loading CSV files...")

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

# ===== CLEANING AND BASIC TRANSFORMATIONS =====
print("Cleaning and transforming data...")

# Convert to datetime, make tz-naive
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
# Ensure tz-naive
if pd.api.types.is_datetime64tz_dtype(encounters["START"]):
    encounters["START"] = encounters["START"].dt.tz_convert(None)

# Convert VALUE to numeric
observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

# Drop missing IDs
patients.dropna(subset=["Id"], inplace=True)
encounters.dropna(subset=["PATIENT"], inplace=True)
observations.dropna(subset=["PATIENT"], inplace=True)

gc.collect()

# ===== TEMPORAL CONSISTENCY CHECK =====
print("Performing temporal consistency checks...")

# Merge to get BIRTHDATE per encounter
temp = encounters.merge(
    patients[["Id", "BIRTHDATE"]],
    left_on="PATIENT",
    right_on="Id",
    how="left",
    suffixes=("_ENC", "_PAT")
)

# Keep only encounters after birthdate
temp = temp[temp["START"] >= temp["BIRTHDATE"]]

# Keep relevant columns for further processing
encounters = temp[["Id_ENC", "START", "PATIENT"]].rename(columns={"Id_ENC": "ENCOUNTER_ID"}).copy()
del temp
gc.collect()

# ===== AGGREGATE OBSERVATIONS =====
print("Aggregating observations per patient (mean)...")

observations_agg = (
    observations.groupby("PATIENT", observed=True)["VALUE"]
    .mean()
    .reset_index()
    .rename(columns={"VALUE": "LAB_VALUE"})
)

del observations
gc.collect()

# ===== MERGE DATASETS =====
print("Merging patients, encounters, and aggregated observations...")

df = pd.merge(
    patients,
    encounters,
    left_on="Id",
    right_on="PATIENT",
    how="left"
)

df = pd.merge(df, observations_agg, left_on="Id", right_on="PATIENT", how="left")

del patients, encounters, observations_agg
gc.collect()

# ===== FEATURE ENGINEERING =====
print("Performing feature engineering...")

# AGE
df["AGE"] = (pd.Timestamp.now() - df["BIRTHDATE"]).dt.days // 365

# Encode categorical features
for col in ["GENDER", "RACE", "ETHNICITY"]:
    df[col] = df[col].astype("category").cat.codes

# Fill missing numeric values with median
for col in ["LAB_VALUE", "AGE", "INCOME"]:
    df[col].fillna(df[col].median(), inplace=True)

# ===== ENHANCED FEATURES =====
print("Computing enhanced features...")

# Number of encounters per patient
encounter_count = df.groupby("Id", observed=True)["ENCOUNTER_ID"].count().reset_index()
encounter_count.rename(columns={"ENCOUNTER_ID": "NUM_ENCOUNTERS"}, inplace=True)
df = pd.merge(df, encounter_count, on="Id", how="left")

# Average days between encounters
df_sorted = df.sort_values(["Id", "START"])
avg_gap = df_sorted.groupby("Id")["START"].diff().dt.days.groupby(df_sorted["Id"]).mean().reset_index()
avg_gap.rename(columns={"START": "AVG_DAYS_BETWEEN_ENC"}, inplace=True)
df = pd.merge(df, avg_gap, on="Id", how="left")

# ===== CREATE TEMPORAL SEQUENCES =====
print("Creating temporal sequences per patient (5 recent observations)...")

VALUE_COLUMNS = ["LAB_VALUE"]

def create_sequences(group):
    values = group.sort_values("START")[VALUE_COLUMNS].values.flatten()
    if len(values) == 0:
        return np.zeros(5)
    elif len(values) < 5:
        values = np.pad(values, (0, 5 - len(values)), "constant", constant_values=0)
    else:
        values = values[-5:]
    return values

sequences = (
    df.groupby("Id", observed=True)
    .apply(create_sequences)
    .apply(pd.Series)
)
sequences.columns = [f"SEQ_VAL_{i+1}" for i in range(sequences.shape[1])]

final_df = pd.concat([df.drop_duplicates("Id").set_index("Id"), sequences], axis=1).reset_index()

# ===== SAVE PROCESSED DATA =====
output_file = os.path.join(OUTPUT_PATH, "preprocessed_members.csv")
final_df.to_csv(output_file, index=False)
print(f"✅ Preprocessing complete. File saved to: {output_file}")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===== Finished Successfully =====")
