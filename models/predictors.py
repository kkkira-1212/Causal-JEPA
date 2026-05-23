# Three independent cross-attention predictors (P_self, P_pa, P_non): shared architecture, independent weights.

import torch
import torch.nn as nn
from typing import Tuple


class PatchPredictor(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.mlp     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, s: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        # s: (B, D, P, H)   queries: (num_target, H)  →  (B, D, num_target, H)
        B, D, P, H = s.shape
        T = queries.shape[0]
        q_exp = queries.unsqueeze(0).expand(B * D, -1, -1).contiguous()  # contiguous: avoids in-place op on shared memory view
        kv    = self.norm_kv(s).reshape(B * D, P, H)
        out   = q_exp + self.attn(self.norm_q(q_exp), kv, kv)[0]
        out   = out + self.mlp(self.norm2(out))
        return out.reshape(B, D, T, H)


class PatchPredictors(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.P_self = PatchPredictor(hidden_dim, num_heads, mlp_ratio)
        self.P_pa   = PatchPredictor(hidden_dim, num_heads, mlp_ratio)
        self.P_non  = PatchPredictor(hidden_dim, num_heads, mlp_ratio)

    def forward(
        self,
        s_ctx: torch.Tensor, s_pa: torch.Tensor, s_non: torch.Tensor,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.P_self(s_ctx,  queries),
            self.P_pa  (s_pa,   queries),
            self.P_non (s_non,  queries),
        )
