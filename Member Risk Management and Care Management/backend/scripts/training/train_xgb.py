import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.helpers import setup_logging
from scripts.training.create_labels import create_labels

def load_and_merge_data(embeddings_path: str, static_path: str, labels_path: str, horizon: int):
    """Load and merge embeddings, static features, and labels"""
    
    # Load data
    df_embeddings = pd.read_csv(embeddings_path)
    df_static = pd.read_csv(static_path)
    
    # Check if labels file exists, create if not
    if not os.path.exists(labels_path):
        print(f"Labels file {labels_path} not found. Creating labels...")
        # Create labels using available data
        sequences_path = 'data/processed/sequences_5.csv'  # Adjust path as needed
        windowed_path = 'data/processed/windowed_features_30_60_90.csv'
        output_dir = os.path.dirname(labels_path)
        create_labels(sequences_path, static_path, windowed_path, output_dir)
    
    df_labels = pd.read_csv(labels_path)
    
    # Merge on member_id
    df_features = pd.merge(df_embeddings, df_static, on='member_id', how='inner')
    df_merged = pd.merge(df_features, df_labels, on='member_id', how='inner')
    
    # Separate features and labels
    feature_columns = [col for col in df_merged.columns 
                      if col not in ['member_id', 'label']]
    
    X = df_merged[feature_columns].values
    y = df_merged['label'].values
    
    return X, y, df_merged['member_id'].values

def train_xgb_model(args):
    """Train XGBoost model on embeddings and static features"""
    
    logger = setup_logging('logs', f'train_xgb_{args.horizon}')
    logger.info("Starting XGBoost training")
    
    # Load and prepare data
    X, y, member_ids = load_and_merge_data(
        args.embeddings, args.static, args.labels, args.horizon
    )
    
    # Check class balance
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, member_ids, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'eval_metric': ['auc', 'logloss'],
        'seed': 42
    }
    
    # Train model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.nrounds,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50
    )
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)
    
    logger.info(f"Test Metrics - AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, Brier: {brier:.4f}")
    
    # Calibration
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_test, y_test)
    y_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Save model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save_model(args.out)
    
    # Save calibrated model
    cal_path = args.out.replace('.json', '_calibrated.joblib')
    joblib.dump(calibrated_model, cal_path)
    
    # Save metadata
    metadata = {
        'horizon': args.horizon,
        'metrics': {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'brier_score': float(brier)
        },
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    meta_path = args.out.replace('.json', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {args.out}")
    logger.info(f"Calibrated model saved to {cal_path}")
    logger.info(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--embeddings', type=str, required=True, 
                       help='Path to embeddings CSV')
    parser.add_argument('--static', type=str, required=True, 
                       help='Path to static features CSV')
    parser.add_argument('--labels', type=str, required=True, 
                       help='Path to labels CSV')
    parser.add_argument('--out', type=str, required=True, 
                       help='Output path for trained model')
    parser.add_argument('--horizon', type=int, default=30, 
                       help='Prediction horizon')
    parser.add_argument('--max_depth', type=int, default=6, 
                       help='Maximum tree depth')
    parser.add_argument('--eta', type=float, default=0.05, 
                       help='Learning rate')
    parser.add_argument('--subsample', type=float, default=0.8, 
                       help='Subsample ratio')
    parser.add_argument('--colsample_bytree', type=float, default=0.7, 
                       help='Column subsample ratio')
    parser.add_argument('--nrounds', type=int, default=1000, 
                       help='Number of boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, 
                       help='Early stopping rounds')
    
    args = parser.parse_args()
    train_xgb_model(args)