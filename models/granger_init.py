# GPU-batched Granger F-statistics → Theta initialisation for CausalLayer.

import math
import numpy as np
import torch
from scipy.stats import f as f_dist


def _batch_granger_f_stats(data: np.ndarray, maxlag: int, device: torch.device) -> np.ndarray:
    """Compute F_mat (D, D) where F_mat[j, i] = F-statistic for the hypothesis i→j."""
    T, D  = data.shape
    n     = T - maxlag
    k_u   = maxlag * 2 + 1
    X      = torch.tensor(data, dtype=torch.float64, device=device)
    y_all  = X[maxlag:, :]
    x_lags = X[:-maxlag, :]
    ones   = torch.ones(n, 1, dtype=torch.float64, device=device)

    X_r_all   = torch.stack([torch.cat([ones, x_lags[:, j:j+1]], dim=1) for j in range(D)], dim=0)
    y_col     = y_all.T.unsqueeze(-1)
    beta_r    = torch.linalg.solve(torch.bmm(X_r_all.transpose(1, 2), X_r_all),
                                    torch.bmm(X_r_all.transpose(1, 2), y_col))
    RSS_r     = ((y_col - torch.bmm(X_r_all, beta_r)) ** 2).sum(dim=1).squeeze(-1)

    x_lags_j = x_lags.T.unsqueeze(1).unsqueeze(-1).expand(D, D, n, 1)
    x_lags_i = x_lags.T.unsqueeze(0).unsqueeze(-1).expand(D, D, n, 1)
    ones_exp = ones.unsqueeze(0).unsqueeze(0).expand(D, D, -1, -1)
    X_u_flat = torch.cat([ones_exp, x_lags_j, x_lags_i], dim=-1).reshape(D * D, n, k_u)
    y_j_exp  = y_col.unsqueeze(1).expand(D, D, n, 1).reshape(D * D, n, 1)
    XtX_u    = torch.bmm(X_u_flat.transpose(1, 2), X_u_flat)
    # Ridge jitter: diagonal (i==j) pairs have duplicate columns → XtX is singular
    XtX_u   += 1e-8 * torch.eye(k_u, dtype=torch.float64, device=device).unsqueeze(0)
    beta_u   = torch.linalg.solve(XtX_u, torch.bmm(X_u_flat.transpose(1, 2), y_j_exp))
    RSS_u    = ((y_j_exp - torch.bmm(X_u_flat, beta_u)) ** 2).sum(dim=1).squeeze(-1).reshape(D, D)

    RSS_r_mat = RSS_r.unsqueeze(1).expand(D, D)
    F_mat     = ((RSS_r_mat - RSS_u) / maxlag) / (RSS_u / (n - k_u) + 1e-12)
    F_mat     = F_mat * (1 - torch.eye(D, dtype=torch.float64, device=device))
    return F_mat.cpu().numpy()


def granger_init_theta(
    data:      np.ndarray,
    num_vars:  int,
    maxlag:    int   = 1,
    alpha:     float = 0.05,
    tau:       float = 1.0,
    high_conf: float = 0.9,
    low_conf:  float = 0.1,
    soft:      bool  = False,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D      = num_vars
    print(f"[Granger init] {'Form 2 (soft)' if soft else 'Form 1 (hard binary)'}  "
          f"| {D*D-D} pairs | maxlag={maxlag} | device={device}")

    F_mat = _batch_granger_f_stats(data, maxlag, device)          # F_mat[j,i] for i→j
    df2   = data.shape[0] - maxlag - (maxlag * 2 + 1)
    p_mat = f_dist.sf(F_mat, maxlag, df2)                         # p_mat[j,i] for i→j

    high  = tau * math.log(high_conf / (1.0 - high_conf))
    low   = tau * math.log(low_conf  / (1.0 - low_conf))
    theta = torch.full((D, D), float(low))

    if not soft:
        sig_t = torch.tensor(p_mat < alpha, dtype=torch.bool).T  # sig_t[i,j]: True if i→j significant
        theta[sig_t] = high
        n_sig = sig_t.sum().item()
    else:
        p_t    = torch.tensor(p_mat.T)
        A_init = torch.clamp(1.0 - p_t, 0.05, 0.95)
        theta  = tau * torch.log(A_init / (1.0 - A_init))
        n_sig  = (p_t < alpha).sum().item()

    theta.fill_diagonal_(0.0)
    print(f"[Granger init] Significant edges: {n_sig}/{D*D-D}  (high={high:.2f}, low={low:.2f})")
    return theta
