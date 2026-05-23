# ETTh1/ETTh2 sliding-window dataset. Local CSV takes priority over GitHub download.

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


def load_etth_array(name: str = "ETTh1", local_path: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    if local_path and Path(local_path).exists():
        df = pd.read_csv(local_path, parse_dates=["date"])
        print(f"[ETTh] loaded {local_path}  shape={df.shape}")
    else:
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
        df  = pd.read_csv(url, parse_dates=["date"])
        print(f"[ETTh] downloaded {name}  shape={df.shape}")
    feature_cols = [c for c in df.columns if c != "date"]
    data = df[feature_cols].values.astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std  = data.std(axis=0, keepdims=True) + 1e-8
    return (data - mean) / std, feature_cols


class ETThDataset(Dataset):

    def __init__(
        self,
        data: np.ndarray,
        context_len: int,
        target_len: int,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        T         = len(data)
        train_end = int(T * train_ratio)
        val_end   = int(T * (train_ratio + val_ratio))
        if split == "train":
            data = data[:train_end]
        elif split == "val":
            data = data[train_end:val_end]
        else:
            data = data[val_end:]
        self.data        = torch.from_numpy(data)
        self.context_len = context_len
        self.target_len  = target_len
        self.window      = context_len + target_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.window + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.data[idx : idx + self.context_len],
                self.data[idx + self.context_len : idx + self.window])


def make_etth_datasets(
    config: dict,
    name: str = "ETTh1",
    local_path: Optional[str] = None,
) -> Tuple[ETThDataset, ETThDataset, List[str]]:
    data, col_names = load_etth_array(name=name, local_path=local_path)
    config["model"]["num_vars"] = data.shape[1]
    ctx, tgt  = config["model"]["context_len"], config["model"]["target_len"]
    train_set = ETThDataset(data, ctx, tgt, split="train")
    val_set   = ETThDataset(data, ctx, tgt, split="val")
    print(f"[ETTh] D={data.shape[1]}  train={len(train_set)}  val={len(val_set)}")
    return train_set, val_set, col_names
