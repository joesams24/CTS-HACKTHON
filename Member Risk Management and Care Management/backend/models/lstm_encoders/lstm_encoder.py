import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Generic LSTM encoder for sequence embeddings.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2, embedding_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.num_dirs = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.num_dirs, embedding_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, num_features)
        Returns embedding tensor of shape (batch_size, embedding_dim)
        """
        _, (h_n, _) = self.rnn(x)  # h_n: (num_layers * num_dirs, B, H)
        last = h_n[-self.num_dirs:]  # take last layer's hidden states
        if last.dim() == 3:
            last = last.transpose(0, 1).contiguous().view(x.size(0), -1)  # (B, H * num_dirs)
        emb = self.fc(last)
        return emb
