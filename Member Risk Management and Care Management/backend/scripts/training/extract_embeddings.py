import torch
import pandas as pd
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.training.train_lstm import BiLSTMAttention, load_sequences_data
from scripts.utils.helpers import setup_logging

def extract_embeddings(args):
    """Extract embeddings using trained LSTM model"""
    
    logger = setup_logging('logs', f'extract_embeddings_{args.window}')
    logger.info(f"Extracting embeddings for {args.window}-day horizon")
    
    # Load data
    sequences, _ = load_sequences_data(args.input, args.window)
    
    # Initialize model
    input_size = sequences.shape[2]
    model = BiLSTMAttention(
        input_size=input_size,
        hidden_size=128,  # Should match training config
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
        embedding_dim=256
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(args.lstm))
    model.eval()
    
    # Extract embeddings
    embeddings = []
    batch_size = 512
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_embeddings = model(batch)
            embeddings.append(batch_embeddings.numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Create DataFrame with member IDs
    df_input = pd.read_csv(args.input)
    member_ids = df_input['member_id']
    
    # Create embedding columns
    embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings, columns=embedding_cols)
    df_embeddings['member_id'] = member_ids.values
    
    # Reorder columns to have member_id first
    df_embeddings = df_embeddings[['member_id'] + embedding_cols]
    
    # Save embeddings
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_embeddings.to_csv(args.out, index=False)
    logger.info(f"Embeddings saved to {args.out}")
    logger.info(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract LSTM embeddings')
    parser.add_argument('--lstm', type=str, required=True, 
                       help='Path to trained LSTM model')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input sequences CSV')
    parser.add_argument('--out', type=str, required=True, 
                       help='Output path for embeddings CSV')
    parser.add_argument('--window', type=int, default=30, 
                       help='Prediction horizon window')
    
    args = parser.parse_args()
    extract_embeddings(args)