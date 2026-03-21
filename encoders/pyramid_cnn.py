from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PyramidCNNEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], output_dim: int) -> None:
        super().__init__()
        if len(channels) < 1:
            raise ValueError("channels must be non-empty")

        layers: list[nn.Module] = [
            nn.LazyConv2d(channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        ]
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(channels[-1], output_dim),
            ]
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
