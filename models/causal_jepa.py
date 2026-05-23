# CausalJEPA: TTMEncoder → CausalLayer → PatchPredictors with EMA target encoder.

import copy
import torch
import torch.nn as nn
from typing import Tuple

from models.encoder import TTMEncoder
from models.causal_layer import CausalLayer, TemperatureScheduler
from models.predictors import PatchPredictors


class CausalJEPA(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        m_cfg = config["model"]
        c_cfg = config["causal"]
        t_cfg = config["training"]

        self.context_encoder = TTMEncoder(
            model_path=m_cfg["ttm_model_path"],
            hidden_dim=m_cfg["hidden_dim"],
            freeze_backbone=m_cfg["freeze_backbone"],
            unfreeze_last_n_layers=m_cfg.get("unfreeze_last_n_layers", 0),
            revision=m_cfg.get("ttm_revision", None),
        )
        self._patch_size        = self.context_encoder.patch_size
        # num_patches from backbone config: TTM may output more patches than context_len // patch_size
        self.num_patches        = self.context_encoder.backbone.config.num_patches
        self.num_target_patches = m_cfg.get("num_target_patches", 3)

        hidden_dim      = m_cfg["hidden_dim"]
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        self.pos_embed  = nn.Embedding(self.num_patches, hidden_dim)
        self.causal_layer = CausalLayer(num_vars=m_cfg["num_vars"], tau_init=c_cfg["tau_start"])
        self.predictors   = PatchPredictors(hidden_dim=hidden_dim)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.ema_decay  = m_cfg["ema_decay"]
        self.mask_ratio = t_cfg.get("mask_ratio", 0.0)
        self.tau_scheduler = TemperatureScheduler(
            tau_start=c_cfg["tau_start"], tau_end=c_cfg["tau_end"],
            total_epochs=t_cfg["num_epochs"], warmup_epochs=t_cfg["warmup_epochs"],
        )

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        decay = self.ema_decay
        for p_q, p_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_k.data.mul_(decay).add_(p_q.data, alpha=1.0 - decay)

    def update_tau(self, epoch: int) -> float:
        tau = self.tau_scheduler.get_tau(epoch)
        self.causal_layer.set_tau(tau)
        return tau

    def _apply_context_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        ps        = self._patch_size
        n_patches = T // ps
        n_mask    = max(1, int(n_patches * self.mask_ratio))
        ctx_patch_mask = torch.rand(B, n_patches, device=x.device).argsort(dim=1) < n_mask
        ts_mask = ctx_patch_mask.unsqueeze(-1).expand(B, n_patches, ps).reshape(B, n_patches * ps)
        x_masked = x.clone()
        x_masked[:, :n_patches * ps, :].masked_fill_(ts_mask.unsqueeze(-1), 0.0)
        return x_masked, ctx_patch_mask

    def _sample_target_patches(self, device: torch.device) -> torch.Tensor:
        return torch.randperm(self.num_patches, device=device)[:self.num_target_patches]

    def _route(self, s_ctx: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s_pa[b,j,p,h] = Σ_i A[i,j] * s_ctx[b,i,p,h]  (parent-weighted aggregation)
        eps      = 1e-8
        s_pa     = torch.einsum('biph,ij->bjph', s_ctx, A)
        s_non    = torch.einsum('biph,ij->bjph', s_ctx, 1.0 - A)
        norm_pa  = A.sum(dim=0).view(1, -1, 1, 1) + eps
        norm_non = (1.0 - A).sum(dim=0).view(1, -1, 1, 1) + eps
        return s_pa / norm_pa, s_non / norm_non

    def forward(
        self,
        x_context: torch.Tensor,
        x_target:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.mask_ratio > 0.0:
            x_context, _ = self._apply_context_mask(x_context)

        s_ctx              = self.context_encoder(x_context)          # (B, D, num_patches, H)
        A                  = self.causal_layer.get_adjacency()         # (D, D)
        s_pa, s_non        = self._route(s_ctx, A)
        target_patch_idx   = self._sample_target_patches(x_context.device)
        queries            = self.mask_token + self.pos_embed(target_patch_idx)
        s_hat_self, s_hat_pa, s_hat_non = self.predictors(s_ctx, s_pa, s_non, queries)

        with torch.no_grad():
            s_target = self.target_encoder(x_target)
        s_target_patches = s_target[:, :, target_patch_idx, :]

        return s_hat_self, s_hat_pa, s_hat_non, s_target_patches, A, target_patch_idx

    def get_causal_graph(self, threshold: float = 0.5) -> torch.Tensor:
        return self.causal_layer.get_discrete_adjacency(threshold)
