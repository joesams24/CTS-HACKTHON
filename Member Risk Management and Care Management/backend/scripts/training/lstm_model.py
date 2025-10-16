import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scripts.training.dataset import SequenceDataset

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
LSTM_MODEL_DIR = os.path.join(BASE_DIR, "backend/models/lstm_models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "backend/data/processed/embeddings")

# Ensure folders exist
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, embedding_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        embedding = self.fc(out)
        return embedding


def train_lstm(
    seq_file: str,
    model_name: str,  # e.g., "lstm_30.pth"
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 0.001,
    logger=None
) -> str:
    """Train LSTM on sequence data and generate embeddings."""
    
    # Path to save model
    lstm_model_file = os.path.join(LSTM_MODEL_DIR, model_name)

    # Load sequence CSV
    df_seq = pd.read_csv(seq_file)
    ids = df_seq["Id"].values
    seq_cols = [c for c in df_seq.columns if c.startswith("SEQ_VAL_")]
    sequences = df_seq[seq_cols].values[:, :, np.newaxis]  # Add feature dim

    # Dataset & DataLoader
    dataset = SequenceDataset(sequences, ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss
    model = LSTMEncoder(input_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            target = batch_x[:, -1, :]  # Predict last value
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if logger:
            logger.info(f"LSTM Epoch {epoch+1}/{epochs} Loss={total_loss/len(loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), lstm_model_file)
    if logger:
        logger.info(f"✅ Saved LSTM model to {lstm_model_file}")

    # Generate embeddings
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch_x, batch_ids in loader:
            emb = model(batch_x)
            for i, id_val in enumerate(batch_ids):
                embeddings.append([id_val] + emb[i].tolist())

    emb_df = pd.DataFrame(embeddings, columns=["Id"] + [f"EMB_{i}" for i in range(16)])

    # Save embeddings inside processed/embeddings/
    window = os.path.basename(seq_file).split("_")[1].split(".")[0]
    emb_file = os.path.join(EMBEDDINGS_DIR, f"lstm_embeddings_{window}.csv")
    emb_df.to_csv(emb_file, index=False)

    if logger:
        logger.info(f"✅ Saved embeddings to {emb_file}")

    return emb_file


if __name__ == "__main__":
    import argparse
    from scripts.utils.helpers import setup_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True, help="Sequence window size (30, 60, or 90)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    seq_file = os.path.join(BASE_DIR, "backend/data/processed", f"sequences_{args.window}.csv")
    model_name = f"lstm_{args.window}.pth"

    logger = setup_logging("logs", f"lstm_{args.window}")
    train_lstm(seq_file, model_name, epochs=args.epochs, logger=logger)