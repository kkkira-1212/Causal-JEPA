# CAUKER-based SCM generator. Deviations from CAUKER: fixed DAG, lag-1 equations, per-series z-score norm, PyTorch eigh.

import functools
import random as pyrandom
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'CauKer'))

import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from CauKer import (
    build_kernel_bank,
    random_binary_map,
    random_mean_combination,
    generate_random_dag,
)


def sample_gp_pytorch(
    kernel,
    X: np.ndarray,
    mean_vec: Optional[np.ndarray] = None,
    seed: int = 0,
    jitter: float = 1e-6,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T      = len(X)
    K      = torch.tensor(kernel(X.reshape(-1, 1), X.reshape(-1, 1)), dtype=torch.float64, device=device)
    K     += jitter * torch.eye(T, dtype=torch.float64, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    L      = eigenvectors @ torch.diag(torch.sqrt(eigenvalues.clamp(min=0.0)))
    torch.manual_seed(seed)
    sample = (L @ torch.randn(T, dtype=torch.float64, device=device)).cpu().numpy()
    return sample + mean_vec if mean_vec is not None else sample


def build_dag_and_mechanisms(
    num_vars: int,
    max_parents: int,
    dag_seed: int,
) -> Tuple[np.ndarray, nx.DiGraph, List[int], set, Dict]:
    np.random.seed(dag_seed)
    pyrandom.seed(dag_seed)

    dag      = generate_random_dag(num_vars, max_parents=max_parents)
    true_adj = np.zeros((num_vars, num_vars), dtype=np.float32)
    for src, dst in dag.edges():
        true_adj[src, dst] = 1.0

    topo_order = list(nx.topological_sort(dag))
    root_nodes = set(n for n in dag.nodes if dag.in_degree(n) == 0)

    ACTIVATIONS = ["linear", "relu", "sigmoid", "sin", "mod", "leakyrelu"]
    mechanisms: Dict[int, Tuple] = {}
    for j in topo_order:
        parents = list(dag.predecessors(j))
        if not parents:
            continue
        W   = np.random.randn(len(parents)) / (np.sqrt(len(parents)) + 1e-6)
        b   = float(np.random.uniform(-0.5, 0.5))
        act = str(np.random.choice(ACTIVATIONS))
        act_params: Dict = {}
        if act == "linear":
            act_params = {"a": float(np.random.uniform(0.5, 2.0)), "b": float(np.random.uniform(-1.0, 1.0))}
        elif act == "mod":
            act_params = {"c": float(np.random.uniform(1.0, 5.0))}
        elif act == "leakyrelu":
            act_params = {"alpha": float(np.random.uniform(0.01, 0.3))}
        mechanisms[j] = (parents, W, b, act, act_params)

    print(f"[DAG] dag_seed={dag_seed} | {int(true_adj.sum())} edges | "
          f"roots={sorted(root_nodes)} | density={true_adj.sum()/(num_vars*(num_vars-1)):.1%}")
    return true_adj, dag, topo_order, root_nodes, mechanisms


def _apply_fixed_activation(x: np.ndarray, act: str, params: Dict) -> np.ndarray:
    if act == "linear":   return params["a"] * x + params["b"]
    if act == "relu":     return np.maximum(0.0, x)
    if act == "sigmoid":  return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    if act == "sin":      return np.sin(x)
    if act == "mod":      return np.mod(x, params["c"])
    return np.where(x > 0, x, params["alpha"] * x)  # leakyrelu


def generate_one_series(
    num_vars: int,
    num_timesteps: int,
    noise_std: float,
    topo_order: List[int],
    root_nodes: set,
    mechanisms: Dict,
    kernel_bank: list,
    gp_seed: int,
) -> np.ndarray:
    np.random.seed(gp_seed)
    pyrandom.seed(gp_seed)

    X    = np.linspace(0.0, 1.0, num_timesteps)
    data = np.zeros((num_timesteps, num_vars), dtype=np.float64)

    for idx, r in enumerate(sorted(root_nodes)):
        selected = np.random.choice(kernel_bank, np.random.randint(1, 8), replace=True)
        kernel   = functools.reduce(random_binary_map, selected)
        signal   = sample_gp_pytorch(kernel, X, mean_vec=random_mean_combination(X),
                                      seed=gp_seed * 1000 + idx)
        data[:, r] = (signal - signal.mean()) / (signal.std() + 1e-6)

    rng = np.random.default_rng(gp_seed + 99999)
    for t in range(1, num_timesteps):
        for j in topo_order:
            if j in root_nodes:
                continue
            parents, W, b, act, act_params = mechanisms[j]
            data[t, j] = _apply_fixed_activation(data[t - 1, parents] @ W + b, act, act_params) \
                         + rng.standard_normal() * noise_std

    return data.astype(np.float32)


def generate_dataset(
    n_series: int,
    num_vars: int,
    num_timesteps: int,
    max_parents: int = 2,
    noise_std: float = 0.1,
    dag_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    kernel_bank = build_kernel_bank(num_timesteps)
    true_adj, _, topo_order, root_nodes, mechanisms = build_dag_and_mechanisms(
        num_vars, max_parents, dag_seed
    )
    all_series = []
    for i in range(n_series):
        print(f"  Generating series {i+1}/{n_series} (gp_seed={i})...", end="\r")
        all_series.append(generate_one_series(
            num_vars, num_timesteps, noise_std,
            topo_order, root_nodes, mechanisms, kernel_bank, gp_seed=i,
        ))
    print()

    data   = np.stack(all_series, axis=0)
    mean_v = data.mean(axis=1, keepdims=True)
    std_v  = data.std(axis=1, keepdims=True) + 1e-6
    return (data - mean_v) / std_v, true_adj


class SCMMultiSeriesDataset(Dataset):
    """Sliding-window dataset from multiple independent SCM series. Split by series, not by window."""

    def __init__(self, data: np.ndarray, context_len: int, target_len: int):
        self.windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        N, T, _ = data.shape
        for i in range(N):
            series   = torch.from_numpy(data[i])
            n_windows = T - context_len - target_len + 1
            for j in range(n_windows):
                self.windows.append((
                    series[j : j + context_len],
                    series[j + context_len : j + context_len + target_len],
                ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx]


def make_scm_datasets_from_file(
    path: str,
    config: dict,
) -> Tuple[SCMMultiSeriesDataset, SCMMultiSeriesDataset, SCMMultiSeriesDataset, np.ndarray]:
    npz      = np.load(path)
    data     = npz["data"]
    true_adj = npz["true_adj"]
    N        = len(data)
    n_test   = max(1, int(N * 0.10))
    n_val    = max(1, int(N * 0.10))
    n_train  = N - n_val - n_test
    ctx, tgt = config["model"]["context_len"], config["model"]["target_len"]

    train_ds = SCMMultiSeriesDataset(data[:n_train],              ctx, tgt)
    val_ds   = SCMMultiSeriesDataset(data[n_train:n_train+n_val], ctx, tgt)
    test_ds  = SCMMultiSeriesDataset(data[n_train+n_val:],        ctx, tgt)

    print(f"[Dataset] {N} series | train={n_train} val={n_val} test={n_test}")
    print(f"[Dataset] windows   | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[Dataset] true edges: {int(true_adj.sum())}/{config['model']['num_vars']*(config['model']['num_vars']-1)}")
    return train_ds, val_ds, test_ds, true_adj


# Legacy single-series interface

class SCMTimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, context_len: int, target_len: int):
        self.data        = torch.from_numpy(data)
        self.context_len = context_len
        self.target_len  = target_len

    def __len__(self) -> int:
        return len(self.data) - self.context_len - self.target_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.data[idx : idx + self.context_len],
                self.data[idx + self.context_len : idx + self.context_len + self.target_len])


def make_scm_dataset(config: dict) -> Tuple[SCMTimeSeriesDataset, np.ndarray]:
    T = config["data"]["num_samples"]
    kernel_bank = build_kernel_bank(T)
    true_adj, _, topo_order, root_nodes, mechanisms = build_dag_and_mechanisms(
        config["model"]["num_vars"], config["data"].get("max_parents", 2), config["training"]["seed"],
    )
    series = generate_one_series(
        config["model"]["num_vars"], T, config["data"].get("noise_std", 0.1),
        topo_order, root_nodes, mechanisms, kernel_bank, gp_seed=0,
    )
    mean_v = series.mean(axis=0, keepdims=True)
    std_v  = series.std(axis=0, keepdims=True) + 1e-6
    series = (series - mean_v) / std_v
    return SCMTimeSeriesDataset(series, config["model"]["context_len"], config["model"]["target_len"]), true_adj
