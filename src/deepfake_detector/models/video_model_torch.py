from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class VideoGRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(1)


def save_torch_checkpoint(model: nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)


def load_torch_checkpoint(model: nn.Module, ckpt_path: Path, device: str = "cpu") -> nn.Module:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def split_train_val(
    X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(X)
    split = int(n * (1 - val_ratio))
    return X[:split], X[split:], y[:split], y[split:]
