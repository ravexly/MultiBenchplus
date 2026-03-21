from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class StridedImageCNNEncoder(nn.Module):
    def __init__(self, channel_sizes: Sequence[int], output_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if len(channel_sizes) == 0:
            raise ValueError("channel_sizes must be non-empty")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for index, out_channels in enumerate(channel_sizes):
            if index == 0:
                layers.append(nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=2, padding=padding))
            else:
                layers.append(nn.Conv2d(channel_sizes[index - 1], out_channels, kernel_size=kernel_size, stride=2, padding=padding))
            layers.append(nn.ReLU())

        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(channel_sizes[-1], output_dim, kernel_size=1),
            ]
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)
