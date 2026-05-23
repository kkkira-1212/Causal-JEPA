# L_JEPA (P_self MSE), L_ADV (log-ratio competitive), L_DAG (NOTEARS acyclicity), L_sparse (L1 on A).

import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def loss_jepa(s_hat_self: torch.Tensor, s_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(s_hat_self, s_target)


def loss_adv(
    s_hat_pa:  torch.Tensor,
    s_hat_non: torch.Tensor,
    s_target:  torch.Tensor,
    margin:    float = 0.1,
) -> torch.Tensor:
    eps      = 1e-8
    # Per-(B, D, T) MSE averaged over H dimension
    loss_pa  = F.mse_loss(s_hat_pa,  s_target, reduction="none").mean(dim=-1)
    loss_non = F.mse_loss(s_hat_non, s_target, reduction="none").mean(dim=-1)
    # math.log(1+margin) is a Python constant — contributes zero gradient; acts as a fixed offset only
    return (torch.log(loss_pa + eps) - torch.log(loss_non + eps) + math.log(1 + margin)).mean()


def loss_dag(A: torch.Tensor) -> torch.Tensor:
    return torch.trace(torch.linalg.matrix_exp(A * A)) - A.shape[0]


def loss_sparse(A: torch.Tensor) -> torch.Tensor:
    return A.abs().sum()


def compute_total_loss(
    s_hat_self:          torch.Tensor,
    s_hat_pa:            torch.Tensor,
    s_hat_non:           torch.Tensor,
    s_target:            torch.Tensor,
    A:                   torch.Tensor,
    config:              dict,
    use_adv:             bool  = True,
    use_sparse:          bool  = True,
    lambda_adv_override: float = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_cfg     = config["loss"]
    l_jepa    = loss_jepa(s_hat_self, s_target)
    l_dag     = loss_dag(A)
    l_sp      = loss_sparse(A)
    sp_weight = l_cfg["lambda_sparse"] if use_sparse else 0.0
    total     = l_cfg.get("lambda_jepa", 1.0) * l_jepa + l_cfg["lambda_dag"] * l_dag + sp_weight * l_sp
    log_dict  = {"loss/jepa": l_jepa.item(), "loss/dag": l_dag.item(), "loss/sparse": l_sp.item()}

    if use_adv:
        lam_adv = lambda_adv_override if lambda_adv_override is not None else l_cfg["lambda_adv"]
        l_adv   = loss_adv(s_hat_pa, s_hat_non, s_target, margin=l_cfg["margin"])
        total   = total + lam_adv * l_adv
        log_dict["loss/adv"]   = l_adv.item()
        log_dict["lambda_adv"] = lam_adv
    else:
        log_dict["loss/adv"]   = 0.0
        log_dict["lambda_adv"] = 0.0

    log_dict["loss/total"] = total.item()
    return total, log_dict
