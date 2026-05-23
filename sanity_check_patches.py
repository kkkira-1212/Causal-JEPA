# Patch-level sanity checks: context cosine similarity decay and P_self prediction quality per position.

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from torch.utils.data import DataLoader

from data.cauker_scm import make_scm_datasets_from_file
from models.causal_jepa import CausalJEPA


@torch.no_grad()
def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_set, _, _ = make_scm_datasets_from_file(config["data"]["dataset_path"], config)
    x_context, x_target = next(iter(DataLoader(val_set, batch_size=64, shuffle=False)))
    x_context = x_context.to(device)
    x_target  = x_target.to(device)

    model = CausalJEPA(config).to(device)
    model.load_state_dict(torch.load("causal_jepa_fixed_graph.pt", map_location=device))
    model.eval()

    s_ctx    = model.context_encoder(x_context)
    s_target = model.target_encoder(x_target)
    B, D, P, H = s_ctx.shape
    s = s_ctx.reshape(B * D, P, H)

    print("=" * 60)
    print("1. Context patch cosine similarity (avg over B, D)")
    print("=" * 60)
    print(f"{'Distance k':>12} | {'mean cos-sim':>12} | {'std':>8}")
    print("-" * 40)
    for k in range(1, P):
        sims = torch.cat([
            (F.normalize(s[:, i, :], dim=-1) * F.normalize(s[:, i+k, :], dim=-1)).sum(dim=-1)
            for i in range(P - k)
        ])
        print(f"  k={k:>2} (gap={k*64:>4} steps) | {sims.mean().item():>12.4f} | {sims.std().item():>8.4f}")

    print()
    print("=" * 60)
    print("2. Context vs Target patch cosine similarity")
    print("=" * 60)
    sc = s_ctx.reshape(B * D, P, H)
    st = s_target.reshape(B * D, P, H)
    same = np.mean([
        (F.normalize(sc[:, i, :], dim=-1) * F.normalize(st[:, i, :], dim=-1)).sum(dim=-1).mean().item()
        for i in range(P)
    ])
    all_ = np.mean([
        (F.normalize(sc[:, i, :], dim=-1) * F.normalize(st[:, j, :], dim=-1)).sum(dim=-1).mean().item()
        for i in range(P) for j in range(P)
    ])
    print(f"  Same patch index ctx[i] vs tgt[i]:  mean={same:.4f}")
    print(f"  All pairs ctx[i] vs tgt[j]:          mean={all_:.4f}")

    print()
    print("=" * 60)
    print("3. P_self prediction quality per target patch position")
    print("=" * 60)
    model.eval()
    mse_by_pos = {i: [] for i in range(P)}
    cos_by_pos = {i: [] for i in range(P)}
    for _ in range(10):
        s_hat_self, _, _, s_tgt_patches, _, idx = model(x_context, x_target)
        for t, patch_idx in enumerate(idx.tolist()):
            pred = s_hat_self[:, :, t, :]
            tgt  = s_tgt_patches[:, :, t, :]
            mse_by_pos[patch_idx].append(F.mse_loss(pred, tgt).item())
            cos_by_pos[patch_idx].append(
                F.cosine_similarity(pred.reshape(-1, H), tgt.reshape(-1, H), dim=-1).mean().item()
            )

    print(f"{'Patch':>6} | {'time range':>14} | {'pred MSE':>10} | {'cos-sim':>8}")
    print("-" * 50)
    for i in range(P):
        if mse_by_pos[i]:
            print(f"  [{i}]  | {i*64:>5}-{i*64+64:<5}steps | "
                  f"{np.mean(mse_by_pos[i]):>10.4f} | {np.mean(cos_by_pos[i]):>8.4f}")
        else:
            print(f"  [{i}]  | — (not sampled)")

    overall_mse = np.mean([v for vals in mse_by_pos.values() for v in vals])
    overall_cos = np.mean([v for vals in cos_by_pos.values() for v in vals])
    print(f"\nOverall P_self: MSE={overall_mse:.4f}  cos-sim={overall_cos:.4f}")


if __name__ == "__main__":
    main()
