# 2×2 ablation: raw-forecast vs representation-prediction, with/without JEPA training.

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cauker_scm import make_scm_datasets_from_file
from models.causal_jepa import CausalJEPA
from models.downstream import DownstreamHead


def train_and_eval_raw_probe(enc, train_loader, val_loader, test_loader, device,
                              hidden_dim, pred_len=96, probe_epochs=20, tag=""):
    head = DownstreamHead(hidden_dim=hidden_dim, pred_len=pred_len).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3)

    for _ in range(probe_epochs):
        head.train()
        for x_ctx, x_tgt in train_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            with torch.no_grad():
                z = enc(x_ctx).mean(dim=2)
            opt.zero_grad()
            F.mse_loss(head(z), x_tgt[:, :pred_len, :]).backward()
            opt.step()

    def score(loader, name):
        head.eval()
        mse, mae = [], []
        with torch.no_grad():
            for x_ctx, x_tgt in loader:
                x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
                p = head(enc(x_ctx).mean(dim=2))
                mse.append(F.mse_loss(p, x_tgt[:, :pred_len, :]).item())
                mae.append(F.l1_loss (p, x_tgt[:, :pred_len, :]).item())
        print(f"  [{tag} raw  {name}]  MSE={sum(mse)/len(mse):.4f}  MAE={sum(mae)/len(mae):.4f}")

    score(val_loader,  "val ")
    score(test_loader, "test")


def train_and_eval_rep_probe(enc, train_loader, val_loader, test_loader, device,
                              hidden_dim, d_model=192, probe_epochs=20, tag=""):
    head = nn.Linear(hidden_dim, d_model).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3)

    for _ in range(probe_epochs):
        head.train()
        for x_ctx, x_tgt in train_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            with torch.no_grad():
                z_ctx = enc(x_ctx).mean(dim=2)
                z_tgt = enc.encode_backbone(x_tgt).mean(dim=2)
            B, D, H = z_ctx.shape
            opt.zero_grad()
            F.mse_loss(head(z_ctx.reshape(B*D, H)).reshape(B, D, d_model), z_tgt).backward()
            opt.step()

    def score(loader, name):
        head.eval()
        mse = []
        with torch.no_grad():
            for x_ctx, x_tgt in loader:
                x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
                z_ctx = enc(x_ctx).mean(dim=2)
                z_tgt = enc.encode_backbone(x_tgt).mean(dim=2)
                B, D, H = z_ctx.shape
                mse.append(F.mse_loss(head(z_ctx.reshape(B*D, H)).reshape(B, D, d_model), z_tgt).item())
        print(f"  [{tag} rep  {name}]  MSE={sum(mse)/len(mse):.4f}")

    score(val_loader,  "val ")
    score(test_loader, "test")


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = config["model"]["hidden_dim"]
    d_model    = 192
    bs         = config["training"]["batch_size"]

    train_set, val_set, test_set, _ = make_scm_datasets_from_file(config["data"]["dataset_path"], config)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}\n")

    print("=" * 60)
    print("1. LINEAR ONLY (random proj, no JEPA training)")
    print("=" * 60)
    model_raw = CausalJEPA(config).to(device)
    for p in model_raw.parameters():
        p.requires_grad_(False)
    model_raw.eval()
    train_and_eval_raw_probe(model_raw.context_encoder, train_loader, val_loader, test_loader,
                              device, hidden_dim, tag="linear")
    train_and_eval_rep_probe(model_raw.context_encoder, train_loader, val_loader, test_loader,
                              device, hidden_dim, d_model, tag="linear")

    ckpt = "causal_jepa_baseline.pt"
    if not os.path.exists(ckpt):
        print(f"\n{ckpt} not found — skipping JEPA comparison")
        return

    print(f"\n{'=' * 60}")
    print(f"2. JEPA TRAINED (loaded from {ckpt})")
    print("=" * 60)
    model_jepa = CausalJEPA(config).to(device)
    model_jepa.load_state_dict(torch.load(ckpt, map_location=device))
    for p in model_jepa.parameters():
        p.requires_grad_(False)
    model_jepa.eval()
    train_and_eval_raw_probe(model_jepa.context_encoder, train_loader, val_loader, test_loader,
                              device, hidden_dim, tag="jepa  ")
    train_and_eval_rep_probe(model_jepa.context_encoder, train_loader, val_loader, test_loader,
                              device, hidden_dim, d_model, tag="jepa  ")


if __name__ == "__main__":
    main()
