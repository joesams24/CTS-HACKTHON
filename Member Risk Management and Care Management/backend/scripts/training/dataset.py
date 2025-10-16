import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, ids):
        self.X = sequences
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.ids[idx]
