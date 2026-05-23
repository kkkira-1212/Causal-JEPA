# VAR(p) synthetic data with known ground-truth adjacency for causal graph validation.

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


def generate_var_data(
    num_vars: int,
    num_timesteps: int,
    lag: int = 1,
    sparsity: float = 0.3,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng      = np.random.default_rng(seed)
    true_adj = (rng.random((num_vars, num_vars)) < sparsity).astype(float)
    np.fill_diagonal(true_adj, 0)
    W = rng.uniform(-0.5, 0.5, (num_vars, num_vars)) * true_adj
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    if spectral_radius >= 1.0:
        W = W / (spectral_radius + 0.1)
    data       = np.zeros((num_timesteps, num_vars))
    data[:lag] = rng.standard_normal((lag, num_vars)) * noise_std
    for t in range(lag, num_timesteps):
        data[t] = data[t - 1] @ W.T + rng.standard_normal(num_vars) * noise_std
    return data.astype(np.float32), true_adj.astype(np.float32)


class SyntheticTimeSeriesDataset(Dataset):

    def __init__(self, data: np.ndarray, context_len: int, target_len: int):
        self.data        = torch.from_numpy(data)
        self.context_len = context_len
        self.target_len  = target_len

    def __len__(self) -> int:
        return len(self.data) - self.context_len - self.target_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.data[idx : idx + self.context_len],
                self.data[idx + self.context_len : idx + self.context_len + self.target_len])


def make_dataset(config: dict) -> Tuple[SyntheticTimeSeriesDataset, np.ndarray]:
    data, true_adj = generate_var_data(
        num_vars=config["model"]["num_vars"],
        num_timesteps=config["data"]["num_samples"],
        lag=config["data"]["lag"],
        seed=config["training"]["seed"],
    )
    return SyntheticTimeSeriesDataset(
        data, config["model"]["context_len"], config["model"]["target_len"]
    ), true_adj
