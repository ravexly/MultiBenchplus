from __future__ import annotations

import torch
import torch.nn as nn


class SequenceTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.input_projection(x))
        x = x.permute(0, 2, 1)
        x = self.pool(x).permute(0, 2, 1)
        return x.squeeze(1)
