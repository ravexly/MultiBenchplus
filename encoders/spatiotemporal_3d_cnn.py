from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporal3DCNNEncoder(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(2, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.projection = nn.Linear(16 * 16 * 16 * 50, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv3d(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.projection(x))
