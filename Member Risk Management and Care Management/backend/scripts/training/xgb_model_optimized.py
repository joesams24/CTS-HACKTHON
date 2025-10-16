# backend/scripts/training/xgb_model_optimized.py

import os
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.utils.helpers import setup_logging

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "backend/data/processed/embeddings")
TARGET_DIR = os.path.join(BASE_DIR, "backend/data/processed/targets")
STATIC_FEATURES_PATH = os.path.join(BASE_DIR, "backend/data/processed/static_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend/models/xgboost")
LOG_DIR = os.path.join(BASE_DIR, "backend/logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging(LOG_DIR, "xgb_training_optimized_final")

# -----------------------------
# Risk tier mapping
# -----------------------------
risk_mapping = {
    "Minimal": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4
}

# -----------------------------
# Load static features
# -----------------------------
if os.path.exists(STATIC_FEATURES_PATH):
    static_features = pd.read_csv(STATIC_FEATURES_PATH)
    logger.info("Loaded static features.")
    static_features.columns = static_features.columns.str.strip().str.lower()
    for col in ["id", "member_id"]:
        if col in static_features.columns:
            static_features.rename(columns={col: "member_id"}, inplace=True)
            break
    # Convert birthdate to age if exists
    if "birthdate" in static_features.columns:
        static_features["birthdate"] = pd.to_datetime(static_features["birthdate"], errors="coerce")
        current_year = pd.Timestamp.now().year
        static_features["age"] = current_year - static_features["birthdate"].dt.year
        static_features.drop(columns=["birthdate"], inplace=True)
else:
    static_features = None
    logger.info("No static features file found; using embeddings only.")

# -----------------------------
# Target windows
# -----------------------------
TARGET_WINDOWS = ["30", "60", "90"]

for window in TARGET_WINDOWS:
    logger.info(f"Processing {window}-day target...")

    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"lstm_embeddings_{window}.csv")
    targets_path = os.path.join(TARGET_DIR, f"target_{window}.csv")

    if not os.path.exists(embeddings_path) or not os.path.exists(targets_path):
        logger.error(f"Missing files for {window}-day target. Skipping.")
        continue

    embeddings = pd.read_csv(embeddings_path)
    targets = pd.read_csv(targets_path)
    embeddings.columns = embeddings.columns.str.strip().str.lower()
    targets.columns = targets.columns.str.strip().str.lower()
    for df in [embeddings, targets]:
        for col in ["id", "member_id"]:
            if col in df.columns:
                df.rename(columns={col: "member_id"}, inplace=True)
                break

    # Merge embeddings and static features
    data = embeddings.merge(static_features, on="member_id", how="left") if static_features is not None else embeddings.copy()
    dataset = data.merge(targets, on="member_id", how="left")

    # Encode risk tier
    y = dataset["risk_tier"].map(risk_mapping)
    X = dataset.drop(columns=["member_id", "risk_tier"])

    # -----------------------------
    # Convert to numeric (float32) safely
    # -----------------------------
    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=['number']).astype('float32')
    logger.info(f"Number of features after conversion: {X.shape[1]}")

    # -----------------------------
    # Impute missing values and standardize
    # -----------------------------
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # -----------------------------
    # PCA for dimensionality reduction (keep max 300 features)
    # -----------------------------
    n_components = min(300, X_scaled.shape[1])
    logger.info(f"Applying PCA: reducing {X_scaled.shape[1]} features → {n_components}")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pd.DataFrame(pca.fit_transform(X_scaled))

    # -----------------------------
    # Multi-class SMOTE
    # -----------------------------
    class_counts = Counter(y)
    minority_count = min(class_counts.values())
    if minority_count >= 2:
        logger.info(f"Applying SMOTE for {window}-day target ({class_counts})")
        max_count = max(class_counts.values())
        sampling_strategy = {cls: max_count for cls in class_counts if class_counts[cls] < max_count}
        smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1), sampling_strategy=sampling_strategy)
        X_res, y_res = smote.fit_resample(X_reduced, y)
        logger.info(f"After SMOTE, class distribution: {Counter(y_res)}")
    else:
        X_res, y_res = X_reduced, y
        logger.info(f"Skipping SMOTE: minority class too small ({class_counts})")

    # -----------------------------
    # Train/validation/test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # -----------------------------
    # Encode classes
    # -----------------------------
    le = LabelEncoder()
    le.fit(y_res)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    num_classes = len(le.classes_)

    # -----------------------------
    # Train XGBoost with hist method for memory efficiency
    # -----------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_classes,
        n_jobs=2,
        tree_method='hist',  # optimized for memory
        enable_categorical=False
    )

    logger.info(f"Training XGBoost model for {window}-day window...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=True
    )

    # Save model
    model_file = os.path.join(MODEL_DIR, f"xgb_{window}d_optimized_final.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved XGBoost model at {model_file}")

logger.info("All XGBoost models trained and saved successfully!")
