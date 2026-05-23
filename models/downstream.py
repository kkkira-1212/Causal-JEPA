# Linear forecasting head used for linear probe evaluation.

import torch
import torch.nn as nn


class DownstreamHead(nn.Module):

    def __init__(self, hidden_dim: int, pred_len: int = 96):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z).permute(0, 2, 1)   # (B, D, H) → (B, pred_len, D)
