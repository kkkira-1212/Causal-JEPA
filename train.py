# Two-phase training: warmup (P_self + encoder only), then full with L_ADV + Theta.
# --baseline: plain JEPA without causal layer.  --fixed-graph: Theta locked at Granger init.

import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

# Muon requires a process group even for single-GPU use
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
transformers.logging.set_verbosity_error()
from torch.utils.data import DataLoader, random_split

from data.synthetic import make_dataset
from data.cauker_scm import make_scm_dataset, make_scm_datasets_from_file
from data.etth import make_etth_datasets
from models.causal_jepa import CausalJEPA
from models.downstream import DownstreamHead
from models.granger_init import granger_init_theta
from losses.losses import compute_total_loss
from muon import Muon


def evaluate_graph_recovery(pred_adj: torch.Tensor, true_adj: torch.Tensor) -> dict:
    pred = pred_adj.cpu().float()
    true = (true_adj if isinstance(true_adj, torch.Tensor) else torch.from_numpy(true_adj)).float()
    tp = (pred * true).sum().item()
    fp = (pred * (1 - true)).sum().item()
    fn = ((1 - pred) * true).sum().item()
    tn = ((1 - pred) * (1 - true)).sum().item()
    return {
        "tpr":       tp / (tp + fn + 1e-8),
        "fpr":       fp / (fp + tn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "f1":        2 * tp / (2 * tp + fp + fn + 1e-8),
    }


def evaluate_downstream(
    model: CausalJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    probe_epochs: int = 20,
    pred_len: int = 96,
) -> dict:
    head           = DownstreamHead(hidden_dim=config["model"]["hidden_dim"], pred_len=pred_len).to(device)
    probe_optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for _ in range(probe_epochs):
        head.train()
        for x_ctx, x_tgt in train_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            with torch.no_grad():
                z = model.context_encoder(x_ctx).mean(dim=2)
            probe_optimizer.zero_grad()
            F.mse_loss(head(z), x_tgt[:, :pred_len, :]).backward()
            probe_optimizer.step()

    head.eval()
    val_mse, val_mae = [], []
    with torch.no_grad():
        for x_ctx, x_tgt in val_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            pred = head(model.context_encoder(x_ctx).mean(dim=2))
            gt   = x_tgt[:, :pred_len, :]
            val_mse.append(F.mse_loss(pred, gt).item())
            val_mae.append(F.l1_loss (pred, gt).item())

    for p in model.parameters():
        p.requires_grad_(True)
    return {"mse": sum(val_mse) / len(val_mse), "mae": sum(val_mae) / len(val_mae)}


def train(config: dict, dataset_name: str = "synthetic", local_path: str = None,
          baseline: bool = False, fixed_graph: bool = False) -> None:
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "Plain JEPA (baseline)" if baseline else \
               ("Causal-JEPA (Granger graph fixed)" if fixed_graph else "Causal-JEPA")
    print(f"Device: {device}  |  Mode: {mode_str}")

    true_adj    = None
    raw_data    = None
    test_loader = None

    if dataset_name == "synthetic":
        dataset, true_adj = make_dataset(config)
        raw_data   = dataset.data.numpy() if hasattr(dataset, "data") else None
        train_size = int(0.8 * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    elif dataset_name == "scm_file":
        data_path = config["data"].get("dataset_path")
        if data_path is None:
            raise ValueError("config.data.dataset_path must be set for --dataset scm_file")
        import numpy as np
        train_set, val_set, test_set, true_adj = make_scm_datasets_from_file(data_path, config)
        raw_data = np.load(data_path)["data"][0]

    elif dataset_name == "scm":
        dataset, true_adj = make_scm_dataset(config)
        raw_data   = dataset.data.numpy()
        train_size = int(0.8 * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
        print(f"SCM dataset: {len(dataset)} windows, "
              f"true edges={int(true_adj.sum())}/{config['model']['num_vars']**2 - config['model']['num_vars']}")

    else:
        name_map = {"etth1": "ETTh1", "etth2": "ETTh2"}
        train_set, val_set, _ = make_etth_datasets(
            config, name=name_map.get(dataset_name, "ETTh1"), local_path=local_path
        )

    bs           = config["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False, drop_last=False)
    if dataset_name == "scm_file":
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, drop_last=False)
    print(f"Train: {len(train_set)} samples  Val: {len(val_set)} samples"
          + (f"  Test: {len(test_set)} samples" if test_loader else ""))

    model = CausalJEPA(config).to(device)

    if raw_data is not None and not baseline:
        # Cap at 2000 steps: more data inflates statistical power and causes
        # transitive effects to appear significant as direct edges.
        theta_init = granger_init_theta(
            data=raw_data[:2000],
            num_vars=config["model"]["num_vars"],
            maxlag=1, alpha=0.05, tau=config["causal"]["tau_start"],
            high_conf=0.65, low_conf=0.35,  # soft init: avoids L_DAG explosion at start
        )
        model.causal_layer.Theta.data.copy_(theta_init.to(device))
        print("[Granger init] Theta initialized.")

    warmup_epochs = config["training"]["warmup_epochs"]
    num_epochs    = config["training"]["num_epochs"]
    lr            = config["training"]["lr"]
    backbone_lr   = config["training"].get("backbone_lr", lr)

    theta_params    = [model.causal_layer.Theta]
    jepa_params     = (list(model.predictors.parameters()) +
                       list(model.context_encoder.proj.parameters()) +
                       [model.mask_token] + list(model.pos_embed.parameters()))
    backbone_params = [p for p in model.context_encoder.backbone.parameters() if p.requires_grad]

    param_groups = [{"params": jepa_params, "lr": lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        print(f"Optimizer: backbone lr={backbone_lr:.2e}  other lr={lr:.2e}")

    # Muon for Theta (spectral updates suit discrete graph topology changes); AdamW for everything else
    optimizer      = torch.optim.AdamW(param_groups)
    muon_optimizer = Muon(theta_params, lr=0.02, momentum=0.95)
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    adv_ramp_epochs = config["loss"].get("adv_ramp_epochs", 10)
    sparse_warmup   = config["loss"].get("sparse_warmup_epochs", 0)

    for epoch in range(num_epochs):
        is_warmup  = epoch < warmup_epochs
        use_sparse = epoch >= warmup_epochs + sparse_warmup
        theta_frozen = is_warmup or baseline or fixed_graph

        tau = model.update_tau(epoch)
        model.causal_layer.noise_scale = max(0.0, 1.0 - epoch / (num_epochs * 0.5))
        adv_progress       = min(1.0, max(0.0, (epoch - warmup_epochs) / adv_ramp_epochs))
        lambda_adv_current = config["loss"]["lambda_adv"] * adv_progress
        model.causal_layer.Theta.requires_grad_(not theta_frozen)
        model.train()
        train_logs = []

        for x_ctx, x_tgt in train_loader:
            x_ctx, x_tgt = x_ctx.to(device), x_tgt.to(device)
            s_hat_self, s_hat_pa, s_hat_non, s_tgt_patches, A, _ = model(x_ctx, x_tgt)

            if baseline:
                lam  = config["loss"].get("lambda_jepa", 1.0)
                loss = lam * F.mse_loss(s_hat_self, s_tgt_patches)
                log_dict = {"l_jepa": (loss / lam).item(), "total": loss.item()}
            else:
                loss, log_dict = compute_total_loss(
                    s_hat_self, s_hat_pa, s_hat_non, s_tgt_patches, A, config,
                    use_adv=not is_warmup, use_sparse=use_sparse,
                    lambda_adv_override=lambda_adv_current,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if not theta_frozen:
                muon_optimizer.step()
            optimizer.zero_grad()
            muon_optimizer.zero_grad()
            model.update_momentum_encoder()
            train_logs.append(log_dict)

        scheduler.step()
        avg   = {k: sum(d[k] for d in train_logs) / len(train_logs) for k in train_logs[0]}
        stage = "warmup" if is_warmup else ("baseline" if baseline else ("fixed-graph" if fixed_graph else "full"))
        print(f"Epoch {epoch+1:3d}/{num_epochs} [{stage}] tau={tau:.4f} | "
              + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        if not is_warmup and not baseline:
            with torch.no_grad():
                A_cur = model.causal_layer.get_adjacency()
                print(f"  [A]    mean={A_cur.mean():.3f}  max={A_cur.max():.3f}  "
                      f"sparsity={(A_cur < 0.1).float().mean():.2f}")
            if true_adj is not None:
                metrics = evaluate_graph_recovery(model.get_causal_graph(threshold=0.5), true_adj)
                print(f"  [graph] TPR={metrics['tpr']:.3f}  FPR={metrics['fpr']:.3f}  F1={metrics['f1']:.3f}")

    probe_pred_len = 96
    print(f"\nRunning linear probe evaluation (frozen encoder, pred_len={probe_pred_len})...")
    m = evaluate_downstream(model, train_loader, val_loader, config, device, pred_len=probe_pred_len)
    print(f"[downstream probe val]   MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  (horizon={probe_pred_len})")

    if test_loader is not None:
        m = evaluate_downstream(model, train_loader, test_loader, config, device, pred_len=probe_pred_len)
        print(f"[downstream probe test]  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  (horizon={probe_pred_len})")

    print("\nTraining complete.")
    suffix = "baseline" if baseline else ("fixed_graph" if fixed_graph else "causal")
    torch.save(model.state_dict(), f"causal_jepa_{suffix}.pt")
    print(f"Weights saved to causal_jepa_{suffix}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--dataset",     type=str, default="synthetic",
                        choices=["synthetic", "scm", "scm_file", "etth1", "etth2"])
    parser.add_argument("--local",       type=str, default=None)
    parser.add_argument("--baseline",    action="store_true")
    parser.add_argument("--fixed-graph", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, dataset_name=args.dataset, local_path=args.local,
          baseline=args.baseline, fixed_graph=args.fixed_graph)
