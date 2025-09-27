import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class CustomEEGDataset(Dataset):
    def __init__(self, annotations_file):
        df = pd.read_csv(annotations_file)

        self.X = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)
        y = df.iloc[:, -1]
        y = y.astype(np.int64)
        self.y = y.to_numpy()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.as_tensor(self.y[idx])
        
        return x, y
