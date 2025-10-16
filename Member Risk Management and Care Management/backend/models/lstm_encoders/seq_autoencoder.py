import torch
import torch.nn as nn
from .lstm_encoder import LSTMEncoder

class SeqAutoEncoder(nn.Module):
    """
    Sequence autoencoder wrapper around an LSTM encoder.
    """
    def __init__(self, encoder: LSTMEncoder, decoder_hidden=128, feature_dim=None):
        super().__init__()
        self.encoder = encoder
        emb_dim = encoder.fc.out_features
        self.dec_fc = nn.Linear(emb_dim, decoder_hidden)
        self.decoder = nn.LSTM(input_size=feature_dim, hidden_size=decoder_hidden, num_layers=1, batch_first=True)
        self.out_fc = nn.Linear(decoder_hidden, feature_dim)

    def forward(self, x):
        emb = self.encoder(x)
        B, T, F = x.size()
        dec_h0 = self.dec_fc(emb).unsqueeze(0)
        dec_c0 = torch.zeros_like(dec_h0)
        dec_in = torch.zeros(B, T, F, device=x.device)
        out, _ = self.decoder(dec_in, (dec_h0, dec_c0))
        recon = self.out_fc(out)
        return recon, emb
