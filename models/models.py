from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

@dataclass
class TrainConfig:
    csv_path: str = "../data/data_preprocessed.csv"
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    model_out: str = "model.pt"
    threshold_out: str = "threshold.json"

class TinyEEGCNN(nn.Module):
    """
    Input: [B, 1, L]
    Output: raw logit per sample (use BCEWithLogitsLoss)
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),    nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),    nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [B, 64, 1]
        )
        self.classifier = nn.Linear(64, 1)  # -> [B, 1]

    def forward(self, x):
        x = self.features(x).squeeze(-1)  # [B, 64]
        logit = self.classifier(x)        # [B, 1]
        return logit