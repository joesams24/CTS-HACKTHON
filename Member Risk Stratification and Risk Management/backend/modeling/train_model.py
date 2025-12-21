import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from preprocessing.preprocess import preprocess_data


def train_xgboost(df):
    # Preprocess
    X, y, encoders = preprocess_data(df)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)

    # Save artifacts
    joblib.dump(model, "artifacts/xgb_model.pkl")
    joblib.dump(encoders, "artifacts/encoders.pkl")

    return auc
