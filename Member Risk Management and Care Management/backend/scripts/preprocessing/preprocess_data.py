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

# ===== LOAD ONLY NEEDED COLUMNS =====
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

encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

patients.dropna(subset=["Id"], inplace=True)
encounters.dropna(subset=["PATIENT"], inplace=True)
observations.dropna(subset=["PATIENT"], inplace=True)

# ===== AGGREGATE OBSERVATIONS =====
print("Aggregating observation data (mean per patient)...")

observations_agg = (
    observations.groupby("PATIENT", observed=True)["VALUE"]
    .mean()
    .reset_index()
    .rename(columns={"VALUE": "LAB_VALUE"})
)

del observations
gc.collect()

# ===== MERGE DATASETS =====
print("Merging patients, encounters, and observations...")

df = pd.merge(
    patients,
    encounters[["Id", "START", "PATIENT"]].rename(columns={"Id": "ENCOUNTER_ID"}),
    left_on="Id",
    right_on="PATIENT",
    how="left"
)

df = pd.merge(df, observations_agg, left_on="Id", right_on="PATIENT", how="left")

del patients, encounters, observations_agg
gc.collect()

# ===== FEATURE ENGINEERING =====
print("Performing feature engineering...")

df["AGE"] = (pd.Timestamp.now() - df["BIRTHDATE"]).dt.days // 365
df["GENDER"] = df["GENDER"].astype("category").cat.codes
df["RACE"] = df["RACE"].astype("category").cat.codes
df["ETHNICITY"] = df["ETHNICITY"].astype("category").cat.codes

df["LAB_VALUE"].fillna(df["LAB_VALUE"].median(), inplace=True)
df["AGE"].fillna(df["AGE"].median(), inplace=True)
df["INCOME"].fillna(df["INCOME"].median(), inplace=True)

# ===== CREATE TEMPORAL SEQUENCES =====
print("Creating temporal sequences per patient (sampling 5 recent observations)...")

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

# ===== SAVE =====
output_file = os.path.join(OUTPUT_PATH, "preprocessed_members.csv")
final_df.to_csv(output_file, index=False)
print(f"✅ Preprocessing complete. File saved to: {output_file}")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===== Finished Successfully =====")
