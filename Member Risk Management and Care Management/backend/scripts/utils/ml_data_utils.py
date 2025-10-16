import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import os


# -------------------------------
# Utility: Normalize column naming
# -------------------------------
def rename_member_id(df):
    df.columns = df.columns.str.strip().str.lower()
    if "id" in df.columns:
        df.rename(columns={"id": "member_id"}, inplace=True)
    elif "member_id" in df.columns:
        df.rename(columns={"member_id": "member_id"}, inplace=True)  # for consistency
    return df


# -------------------------------
# Load static features
# -------------------------------
def load_static_features(static_path):
    df = pd.read_csv(static_path)
    df = rename_member_id(df)

    if "birthdate" in df.columns:
        df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
        current_year = pd.Timestamp.now().year
        df["age"] = current_year - df["birthdate"].dt.year
        df.drop(columns=["birthdate"], inplace=True)

    return df


# -------------------------------
# Merge embeddings with static features
# -------------------------------
def merge_embeddings_static(embeddings, static_features=None):
    embeddings = rename_member_id(embeddings)
    if static_features is not None:
        return embeddings.merge(static_features, on="member_id", how="left")
    return embeddings.copy()


# -------------------------------
# Preprocess numeric & categorical features
# -------------------------------
def preprocess_features(X, apply_pca=False, n_components=300, pca_path=None, fit_pca=True):
    """
    Preprocess features: encode categorical, impute missing, scale numeric, and optionally apply PCA.
    Parameters:
        X (pd.DataFrame): Input feature set
        apply_pca (bool): Whether to apply PCA
        n_components (int): Number of PCA components
        pca_path (str): Path to save/load PCA model
        fit_pca (bool): True during training, False during evaluation
    Returns:
        (pd.DataFrame, PCA or None)
    """

    # --- 1️⃣ One-hot encode categorical columns ---
    X = pd.get_dummies(X, drop_first=True)
    X = X.select_dtypes(include=["number"]).astype("float32")
    if X.empty:
        raise ValueError("No numeric columns found after one-hot encoding. Check feature merge step.")

    # --- 2️⃣ Impute missing numeric values ---
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --- 3️⃣ Standardize all features ---
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # --- 4️⃣ Apply PCA (if enabled) ---
    pca_obj = None
    if apply_pca:
        n_components = min(n_components, X_scaled.shape[1])
        if pca_path:
            os.makedirs(os.path.dirname(pca_path), exist_ok=True)

            if not fit_pca and os.path.exists(pca_path):
                # Load PCA for evaluation
                pca_obj = joblib.load(pca_path)
                print(f"[INFO] Loaded existing PCA model from: {pca_path}")
                X_reduced = pd.DataFrame(pca_obj.transform(X_scaled))
                return X_reduced, pca_obj

        # Fit PCA for training
        pca_obj = PCA(n_components=n_components, random_state=42)
        X_reduced = pd.DataFrame(pca_obj.fit_transform(X_scaled))

        # Save PCA model
        if pca_path:
            joblib.dump(pca_obj, pca_path)
            print(f"[INFO] PCA model saved to: {pca_path}")

        return X_reduced, pca_obj

    return X_scaled, None


# -------------------------------
# Apply SMOTE for class balancing
# -------------------------------
def apply_smote(X, y):
    class_counts = Counter(y)
    minority_count = min(class_counts.values())

    if minority_count >= 2:
        max_count = max(class_counts.values())
        sampling_strategy = {
            cls: max_count for cls in class_counts if class_counts[cls] < max_count
        }
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, minority_count - 1),
            sampling_strategy=sampling_strategy,
        )
        X_res, y_res = smote.fit_resample(X, y)
        print(f"[INFO] Applied SMOTE. Class distribution after resampling: {Counter(y_res)}")
        return X_res, y_res

    print("[WARN] SMOTE skipped — insufficient samples in one or more classes.")
    return X, y
