# backend/scripts/training/xgb_model_optimized.py

import os
import pandas as pd
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

logger = setup_logging(LOG_DIR, "xgb_combined_training_optimized")

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
    # Normalize ID column
    for col in ["id", "member_id"]:
        if col in static_features.columns:
            static_features.rename(columns={col: "member_id"}, inplace=True)
            break
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

    # Normalize ID columns
    for df in [embeddings, targets]:
        for col in ["id", "member_id"]:
            if col in df.columns:
                df.rename(columns={col: "member_id"}, inplace=True)
                break

    # Merge embeddings with static features
    data = embeddings.merge(static_features, on="member_id", how="left") if static_features is not None else embeddings.copy()
    dataset = data.merge(targets, on="member_id", how="left")

    # Encode risk tier
    y = dataset["risk_tier"].map(risk_mapping)
    X = dataset.drop(columns=["member_id", "risk_tier"])

    # -----------------------------
    # Feature preprocessing
    # -----------------------------
    if "birthdate" in X.columns:
        X["birthdate"] = pd.to_datetime(X["birthdate"], errors="coerce")
        current_year = pd.Timestamp.now().year
        X["age"] = current_year - X["birthdate"].dt.year
        X.drop(columns=["birthdate"], inplace=True)

    X = pd.get_dummies(X, drop_first=True)

    # -----------------------------
    # Merge tiny classes into next class
    # -----------------------------
    class_counts = Counter(y)
    tiny_classes = [cls for cls, count in class_counts.items() if count < 2]
    if tiny_classes:
        logger.warning(f"Merging tiny classes into next class: {tiny_classes}")
        for cls in tiny_classes:
            target_merge = cls + 1 if cls + 1 in risk_mapping.values() else cls - 1
            y = y.replace(cls, target_merge)
        class_counts = Counter(y)
        logger.info(f"Class distribution after merging: {class_counts}")

    if len(class_counts) < 2:
        logger.warning(f"Skipping {window}-day target: less than 2 classes remain after merging.")
        continue

    # -----------------------------
    # Standardize and reduce features using PCA
    # -----------------------------
    logger.info(f"Original number of features: {X.shape[1]}")
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Reduce dimensions to something manageable for 16GB RAM
    n_components = min(300, X_scaled.shape[1])  # keep max 300 features
    logger.info(f"Applying PCA to reduce {X_scaled.shape[1]} features to {n_components}")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pd.DataFrame(pca.fit_transform(X_scaled))

    # -----------------------------
    # Apply multi-class SMOTE safely
    # -----------------------------
    minority_count = min(class_counts.values())
    if minority_count >= 2:
        logger.info(f"Applying SMOTE for {window}-day target ({class_counts})")
        # Use dict to specify number of samples per class
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
    # Train XGBoost
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
        n_jobs=2  # limit parallelism to save RAM
    )

    logger.info(f"Training XGBoost model for {window}-day window...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=True
    )

    model_file = os.path.join(MODEL_DIR, f"xgb_{window}d_optimized.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved XGBoost model at {model_file}")

logger.info("All XGBoost models trained and saved successfully!")
