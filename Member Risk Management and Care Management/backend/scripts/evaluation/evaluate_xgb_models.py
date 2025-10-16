import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import train_test_split
from scripts.utils.helpers import setup_logging
from scripts.utils import ml_data_utils as data_utils

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

os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logging(LOG_DIR, "xgb_model_eval_balanced")

# -----------------------------
# Settings
# -----------------------------
risk_mapping = {"Minimal": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
TARGET_WINDOWS = ["30", "60", "90"]

# Load static features
static_features = None
if os.path.exists(STATIC_FEATURES_PATH):
    static_features = data_utils.load_static_features(STATIC_FEATURES_PATH)
    logger.info("Loaded static features for evaluation.")

results = []

# -----------------------------
# Evaluation Loop
# -----------------------------
for window in TARGET_WINDOWS:
    logger.info(f"--- Evaluating {window}-day model ---")

    # File paths
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"lstm_embeddings_{window}.csv")
    targets_path = os.path.join(TARGET_DIR, f"target_{window}.csv")
    model_path = os.path.join(MODEL_DIR, f"xgb_{window}d.pkl")
    pca_path = os.path.join(PCA_DIR, f"pca_{window}.pkl")

    # Check all required files exist
    if not all(os.path.exists(p) for p in [embeddings_path, targets_path, model_path, pca_path]):
        logger.warning(f"Missing files for {window}-day evaluation. Skipping.")
        continue

    # Load embeddings and targets
    embeddings = pd.read_csv(embeddings_path)
    targets = pd.read_csv(targets_path)

    # Rename target ID column if necessary
    if 'ID' in targets.columns:
        targets.rename(columns={'ID': 'member_id'}, inplace=True)
    targets.columns = targets.columns.str.strip().str.lower()

    # Merge embeddings with static features
    data = data_utils.merge_embeddings_static(embeddings, static_features)

    # Merge with targets
    dataset = data.merge(targets, on="member_id", how="left")

    # Prepare X and y
    y = dataset["risk_tier"].map(risk_mapping)
    X = dataset.drop(columns=["member_id", "risk_tier"])

    # Drop rows with NaN targets
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Filter out classes with fewer than 2 samples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    if len(y) == 0:
        logger.warning(f"No valid data for {window}-day evaluation. Skipping.")
        continue

    # Preprocess numeric features and apply PCA
    pca_obj = joblib.load(pca_path)
    X_scaled, _ = data_utils.preprocess_features(X, apply_pca=False)
    X_reduced = pd.DataFrame(pca_obj.transform(X_scaled))

    # Determine stratification
    stratify_train = y if y.value_counts().min() >= 2 else None

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_reduced, y, test_size=0.3, stratify=stratify_train, random_state=42
    )
    stratify_temp = y_temp if y_temp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=stratify_temp, random_state=42
    )

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_test_enc = le.transform(y_test)

    # Load trained model
    model = joblib.load(model_path)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average="weighted")
    precision = precision_score(y_test_enc, y_pred, average="weighted")
    recall = recall_score(y_test_enc, y_pred, average="weighted")

    # Log metrics
    logger.info(f"{window}-day Model Performance:")
    logger.info(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logger.info("\n" + classification_report(y_test_enc, y_pred))

    results.append({
        "window": window,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })

# -----------------------------
# Summary
# -----------------------------
logger.info("✅ Evaluation complete. Summary of all models:")
for res in results:
    logger.info(f"{res['window']}d -> Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}, "
                f"Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f}")
