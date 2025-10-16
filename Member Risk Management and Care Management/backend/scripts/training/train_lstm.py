import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.helpers import setup_logging, save_checkpoint

class BiLSTMAttention(nn.Module):
    """Bi-directional LSTM with attention mechanism"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 bidirectional=True, dropout=0.2, embedding_dim=256):
        super(BiLSTMAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size * self.num_directions, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Apply attention
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Final projection
        output = self.fc(self.dropout(context_vector))
        return output

def load_sequences_data(filepath: str, window: int):
    """Load and prepare sequence data for training"""
    df = pd.read_csv(filepath)
    
    # Extract features and labels 
    feature_columns = [col for col in df.columns if col not in ['member_id', 'label']]
    sequences = df[feature_columns].values
    
    # Check if labels exist, otherwise create dummy labels for unsupervised pretraining
    if 'label' in df.columns:
        labels = df['label'].values
    else:
        print(f"Warning: No labels found in {filepath}. Using dummy labels for unsupervised pretraining.")
        labels = np.zeros(len(df))  # Dummy labels
    
    # Reshape sequences (assuming fixed sequence length)
    sequence_length = 5  # Based on sequences_5.csv
    num_features = sequences.shape[1] // sequence_length
    
    sequences = sequences.reshape(-1, sequence_length, num_features)
    
    return torch.FloatTensor(sequences), torch.FloatTensor(labels)

def create_labels_from_data(sequences_path, static_path, windowed_path, window):
    """Create labels from available data for a specific horizon"""
    print(f"Creating labels for {window}-day horizon from available data...")
    
    # Load the data
    sequences_df = pd.read_csv(sequences_path)
    static_df = pd.read_csv(static_path)
    windowed_df = pd.read_csv(windowed_path)
    
    # For now, create synthetic labels based on available features
    # This is a placeholder - you'll need to define your actual label creation logic
    member_ids = sequences_df['member_id']
    
    # Example: Create labels based on some feature combination
    # You should replace this with your actual label definition
    labels = []
    for member_id in member_ids:
        # Simple example: label=1 if member has high utilization pattern
        # This is just a placeholder - define your actual risk criteria
        member_data = sequences_df[sequences_df['member_id'] == member_id]
        if len(member_data) > 0:
            # Example risk calculation (modify based on your domain knowledge)
            risk_score = np.random.random()  # Replace with actual risk calculation
            label = 1 if risk_score > 0.5 else 0
        else:
            label = 0
        labels.append(label)
    
    labels_df = pd.DataFrame({
        'member_id': member_ids,
        'label': labels
    })
    
    return labels_df

def train_lstm_model(args):
    """Main training function for LSTM model"""
    
    # Setup logging
    logger = setup_logging('logs', f'training_{args.window}')
    
    logger.info(f"Starting LSTM training for {args.window}-day horizon")
    logger.info(f"Input data: {args.data}")
    logger.info(f"Output model: {args.out}")
    
    # Load data
    sequences, labels = load_sequences_data(args.data, args.window)
    
    # Create dataset and dataloader
    dataset = TensorDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    input_size = sequences.shape[2]  # Number of features per timestep
    model = BiLSTMAttention(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim
    )
    
    # Load pretrained model if specified
    if args.load:
        logger.info(f"Loading pretrained model from {args.load}")
        model.load_state_dict(torch.load(args.load))
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_sequences)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = args.out.replace('.pt', f'_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
    
    # Save final model
    torch.save(model.state_dict(), args.out)
    logger.info(f"Training completed. Model saved to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for risk prediction')
    parser.add_argument('--window', type=int, required=True, choices=[30, 60, 90], 
                       help='Prediction horizon window')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to sequences data CSV')
    parser.add_argument('--out', type=str, required=True, 
                       help='Output path for trained model')
    parser.add_argument('--hidden_size', type=int, default=128, 
                       help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, 
                       help='Number of LSTM layers')
    parser.add_argument('--bidirectional', action='store_true', default=True, 
                       help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=256, 
                       help='Output embedding dimension')
    parser.add_argument('--batch_size', type=int, default=512, 
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--load', type=str, default=None, 
                       help='Path to pretrained model for fine-tuning')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    train_lstm_model(args)