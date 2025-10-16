import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.helpers import setup_logging, calculate_risk_tier

def evaluate_model(args):
    """Comprehensive model evaluation"""
    
    logger = setup_logging('logs', 'eval_metrics')
    logger.info("Starting model evaluation")
    
    # Load model and data
    model = xgb.Booster()
    model.load_model(args.model)
    
    # Load calibrated model if available
    cal_path = args.model.replace('.json', '_calibrated.joblib')
    calibrated_model = joblib.load(cal_path) if os.path.exists(cal_path) else None
    
    # Load test data
    from scripts.training.train_xgb import load_and_merge_data
    
    X_test, y_test, member_ids = load_and_merge_data(
        args.embeddings, args.static, args.labels
    )
    
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Predictions
    y_pred = model.predict(dtest)
    if calibrated_model:
        y_pred_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        y_pred_calibrated = y_pred
    
    # Calculate metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_calibrated),
        'auc_pr': average_precision_score(y_test, y_pred_calibrated),
        'brier_score': brier_score_loss(y_test, y_pred_calibrated),
        'log_loss': log_loss(y_test, y_pred_calibrated)
    }
    
    # Precision@K
    k_values = [0.1, 0.2, 0.3]
    precisions_at_k = {}
    for k in k_values:
        n_top = int(len(y_test) * k)
        top_indices = np.argsort(y_pred_calibrated)[-n_top:]
        precision = y_test[top_indices].mean()
        precisions_at_k[f'precision_at_{int(k*100)}'] = precision
    
    metrics.update(precisions_at_k)
    
    # Risk tier distribution
    risk_tiers = [calculate_risk_tier(score) for score in y_pred_calibrated]
    tier_distribution = pd.Series(risk_tiers).value_counts().to_dict()
    
    # Save results
    results = {
        'metrics': metrics,
        'risk_tier_distribution': tier_distribution,
        'calibration_used': calibrated_model is not None
    }
    
    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    if args.plots:
        create_evaluation_plots(y_test, y_pred_calibrated, args.plots_dir)
    
    logger.info("Evaluation completed")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"Results saved to {args.output}")

def create_evaluation_plots(y_true, y_pred, output_dir):
    """Create evaluation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision_score(y_true, y_pred):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()
    
    # Risk Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('Risk Score Distribution')
    plt.savefig(os.path.join(output_dir, 'risk_distribution.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained XGBoost model')
    parser.add_argument('--embeddings', type=str, required=True, 
                       help='Path to embeddings CSV')
    parser.add_argument('--static', type=str, required=True, 
                       help='Path to static features CSV')
    parser.add_argument('--labels', type=str, required=True, 
                       help='Path to labels CSV')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output path for evaluation results')
    parser.add_argument('--plots', action='store_true', 
                       help='Generate evaluation plots')
    parser.add_argument('--plots_dir', type=str, default='outputs/plots', 
                       help='Directory for evaluation plots')
    
    args = parser.parse_args()
    evaluate_model(args)