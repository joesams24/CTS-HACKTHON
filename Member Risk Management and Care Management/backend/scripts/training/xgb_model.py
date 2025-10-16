# backend/scripts/training/xgb_model.py

import os
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

logger = setup_logging(LOG_DIR, "xgb_combined_training")

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
# Load static features (if any)
# -----------------------------
if os.path.exists(STATIC_FEATURES_PATH):
    static_features = pd.read_csv(STATIC_FEATURES_PATH)
    logger.info("Loaded static features.")

    if 'Id' in static_features.columns:
        static_features.rename(columns={'Id': 'member_id'}, inplace=True)
    else:
        raise ValueError("No 'Id' column found in static_features.csv")

    static_features.columns = static_features.columns.str.strip().str.lower()
else:
    static_features = None
    logger.info("No static features file found; using embeddings only.")

# -----------------------------
# Train XGBoost for each target window
# -----------------------------
TARGET_WINDOWS = ["30", "60", "90"]

for window in TARGET_WINDOWS:
    logger.info(f"Processing {window}-day target...")

    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"lstm_embeddings_{window}.csv")
    targets_path = os.path.join(TARGET_DIR, f"target_{window}.csv")

    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file not found: {embeddings_path}")
        continue
    if not os.path.exists(targets_path):
        logger.error(f"Target file not found: {targets_path}")
        continue

    embeddings = pd.read_csv(embeddings_path)
    targets = pd.read_csv(targets_path)

    embeddings.rename(columns={'Id': 'member_id'}, inplace=True)
    targets.rename(columns={'Id': 'member_id'}, inplace=True)
    embeddings.columns = embeddings.columns.str.strip().str.lower()
    targets.columns = targets.columns.str.strip().str.lower()

    # Merge embeddings with static features
    data = embeddings.merge(static_features, on="member_id") if static_features is not None else embeddings.copy()

    # Merge with target
    dataset = data.merge(targets, on="member_id")

    # Encode risk tier
    y = dataset["risk_tier"].map(risk_mapping)
    X = dataset.drop(columns=["member_id", "risk_tier"])

    # -----------------------------
    # Feature preprocessing
    # -----------------------------
    if 'birthdate' in X.columns:
        X['birthdate'] = pd.to_datetime(X['birthdate'], errors='coerce')
        current_year = pd.Timestamp.now().year
        X['age'] = current_year - X['birthdate'].dt.year
        X.drop(columns=['birthdate'], inplace=True)

    X = pd.get_dummies(X, drop_first=True)

    # -----------------------------
    # Handle extremely small classes using SMOTE
    # -----------------------------
    class_counts = Counter(y)
    if len(class_counts) < 2:
        logger.warning(f"Skipping {window}-day target: only one class present ({class_counts})")
        continue

    if min(class_counts.values()) < 2:
        logger.info(f"Applying SMOTE to increase minority class for {window}-day target ({class_counts})")

        # Impute missing values first
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        class_counts = Counter(y)
        logger.info(f"After SMOTE, class distribution: {class_counts}")

    # -----------------------------
    # Stratified train/val/test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # -----------------------------
    # Encode classes for XGBoost
    # -----------------------------
    le = LabelEncoder()
    le.fit(y)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    num_classes = len(le.classes_)

    # Initialize and train model
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
        num_class=num_classes
    )

    logger.info(f"Training XGBoost model for {window}-day window...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=True
    )

    # Save model
    model_file = os.path.join(MODEL_DIR, f"xgb_{window}d.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved XGBoost model at {model_file}")

logger.info("All XGBoost models trained and saved successfully!")
