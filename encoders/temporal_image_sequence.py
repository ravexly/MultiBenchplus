from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalImageSequenceEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128 * 50, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(batch_size * time_steps, -1)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, time_steps, -1)
        x = x.view(batch_size, -1)
        return self.fc2(x)
