import pandas as pd
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.helpers import setup_logging

def create_labels(sequences_path, static_path, windowed_path, output_dir):
    """Create label files for 30, 60, 90-day horizons"""
    
    logger = setup_logging('logs', 'create_labels')
    logger.info("Creating label files for all horizons")
    
    # Load data
    sequences_df = pd.read_csv(sequences_path)
    static_df = pd.read_csv(static_path)
    windowed_df = pd.read_csv(windowed_path)
    
    # Get all unique member IDs
    member_ids = sequences_df['member_id'].unique()
    
    for horizon in [30, 60, 90]:
        logger.info(f"Creating labels for {horizon}-day horizon")
        
        # Create labels based on your specific criteria
        # This is a template - you need to define your actual label creation logic
        
        labels = []
        for member_id in member_ids:
            # Example label creation logic - REPLACE WITH YOUR ACTUAL LOGIC
            
            # 1. Get member's windowed features for this horizon
            member_windowed = windowed_df[
                (windowed_df['member_id'] == member_id) & 
                (windowed_df['window_days'] == horizon)
            ]
            
            # 2. Get member's sequence data
            member_sequences = sequences_df[sequences_df['member_id'] == member_id]
            
            # 3. Get member's static features
            member_static = static_df[static_df['member_id'] == member_id]
            
            # 4. Define your label creation logic here
            # Example: High risk if high utilization in the window
            if len(member_windowed) > 0:
                num_encounters = member_windowed['num_encounters_window'].iloc[0]
                # Simple threshold-based labeling (modify this)
                label = 1 if num_encounters > 3 else 0  # Example threshold
            else:
                label = 0
                
            labels.append(label)
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'member_id': member_ids,
            'label': labels
        })
        
        # Save labels
        output_path = os.path.join(output_dir, f'labels_{horizon}.csv')
        labels_df.to_csv(output_path, index=False)
        logger.info(f"Saved {horizon}-day labels to {output_path}")
        logger.info(f"Label distribution: {labels_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create label files for training')
    parser.add_argument('--sequences', type=str, required=True,
                       help='Path to sequences_5.csv')
    parser.add_argument('--static', type=str, required=True,
                       help='Path to static_features.csv')
    parser.add_argument('--windowed', type=str, required=True,
                       help='Path to windowed_features_30_60_90.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for label files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_labels(args.sequences, args.static, args.windowed, args.output_dir)