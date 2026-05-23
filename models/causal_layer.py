# Learnable D×D adjacency Theta → soft graph A via Gumbel-Sigmoid + cosine temperature annealing.

import torch
import torch.nn as nn
import math


class TemperatureScheduler:

    def __init__(self, tau_start=1.0, tau_end=0.01, total_epochs=100, warmup_epochs=10):
        self.tau_start     = tau_start
        self.tau_end       = tau_end
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_tau(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.tau_start
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        return max(self.tau_end,
                   self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * progress)))


class CausalLayer(nn.Module):

    def __init__(self, num_vars: int, tau_init: float = 1.0):
        super().__init__()
        self.num_vars    = num_vars
        self.tau         = tau_init
        self.noise_scale = 1.0
        self.Theta       = nn.Parameter(torch.randn(num_vars, num_vars))
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_vars))

    def get_adjacency(self) -> torch.Tensor:
        if self.training and self.noise_scale > 0:
            u = torch.zeros_like(self.Theta).uniform_().clamp(1e-8, 1 - 1e-8)
            g = -torch.log(-torch.log(u))
            A = torch.sigmoid((self.Theta + g * self.noise_scale) / self.tau)
        else:
            A = torch.sigmoid(self.Theta / self.tau)
        return A * self.diag_mask

    def set_tau(self, tau: float) -> None:
        self.tau = max(tau, 1e-6)

    def get_discrete_adjacency(self, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            return (self.get_adjacency() > threshold).float()
