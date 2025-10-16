import os
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scripts.utils.helpers import setup_logging
from scripts.utils import ml_data_utils as data_utils  # <-- import utils

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "backend/data/processed/embeddings")
TARGET_DIR = os.path.join(BASE_DIR, "backend/data/processed/targets")
STATIC_FEATURES_PATH = os.path.join(BASE_DIR, "backend/data/processed/static_features.csv")

MODEL_DIR = os.path.join(BASE_DIR, "backend/models/xgboost")
PCA_DIR = os.path.join(BASE_DIR, "backend/models/pca")
LOG_DIR = os.path.join(BASE_DIR, "backend/logs")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = setup_logging(LOG_DIR, "xgb_training_optimized_final")

# Risk tier mapping
risk_mapping = {"Minimal": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

# Load static features
static_features = None
if os.path.exists(STATIC_FEATURES_PATH):
    static_features = data_utils.load_static_features(STATIC_FEATURES_PATH)
    logger.info("Loaded static features.")

# Target windows
TARGET_WINDOWS = ["30", "60", "90"]

for window in TARGET_WINDOWS:
    logger.info(f"Processing {window}-day target...")

    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"lstm_embeddings_{window}.csv")
    targets_path = os.path.join(TARGET_DIR, f"target_{window}.csv")

    if not os.path.exists(embeddings_path) or not os.path.exists(targets_path):
        logger.warning(f"Missing files for {window}-day target. Skipping.")
        continue

    # Load embeddings and targets
    embeddings = pd.read_csv(embeddings_path)
    targets = pd.read_csv(targets_path)

    # Normalize ID columns
    embeddings = data_utils.merge_embeddings_static(embeddings)
    targets = data_utils.merge_embeddings_static(targets)

    # Merge embeddings + static features + targets
    data = embeddings.merge(static_features, on="member_id", how="left") if static_features is not None else embeddings.copy()
    dataset = data.merge(targets, on="member_id", how="left")

    # Encode risk tier
    y = dataset["risk_tier"].map(risk_mapping)
    X = dataset.drop(columns=["member_id", "risk_tier"])

    # --- 🔹 Define PCA save path inside /models/pca/
    pca_path = os.path.join(PCA_DIR, f"pca_{window}.pkl")

    # --- 🔹 Preprocess features + PCA (auto-saves PCA model)
    X_processed, pca_obj = data_utils.preprocess_features(
        X, apply_pca=True, n_components=300, pca_path=pca_path
    )

    # --- 🔹 Apply SMOTE
    X_res, y_res = data_utils.apply_smote(X_processed, y)

    # --- 🔹 Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # --- 🔹 Encode labels
    le = LabelEncoder()
    le.fit(y_res)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    num_classes = len(le.classes_)

    # --- 🔹 Train XGBoost
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
        tree_method='hist'
    )

    logger.info(f"Training XGBoost model for {window}-day window...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=True)

    # --- 🔹 Save model
    model_file = os.path.join(MODEL_DIR, f"xgb_{window}d.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved XGBoost model at {model_file}")
    logger.info(f"Saved PCA object at {pca_path}")

logger.info("✅ All XGBoost models and PCA transformers trained and saved successfully!")
