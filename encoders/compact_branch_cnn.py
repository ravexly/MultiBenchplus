from __future__ import annotations

import torch
import torch.nn as nn


class CompactBranchCNNEncoder(nn.Module):
    def __init__(self, stem_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        # padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.LazyConv2d(stem_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(stem_channels),
            nn.Conv2d(stem_channels, output_channels, kernel_size=kernel_size),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
