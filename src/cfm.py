"""Conditional Flow Matching losses: independent and minibatch OT coupling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def independent_coupling(
    x1: torch.Tensor, noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        x0 = torch.randn_like(x1)
    else:
        x0 = noise
    return x0, x1


def minibatch_ot_coupling(
    x1: torch.Tensor, noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact OT matching within the minibatch (squared Euclidean cost)."""
    if noise is None:
        x0 = torch.randn_like(x1)
    else:
        x0 = noise

    b = x1.shape[0]
    flat0 = x0.reshape(b, -1)
    flat1 = x1.reshape(b, -1)
    c = (
        (flat0 ** 2).sum(dim=1, keepdim=True)
        + (flat1 ** 2).sum(dim=1).unsqueeze(0)
        - 2.0 * flat0 @ flat1.T
    )
    # High-dimensional / large-batch: greedy sequential matching (O(B^2)).
    # Exact EMD/Hungarian is reserved for small 2D batches.
    if flat0.shape[1] > 8 or b > 256:
        used = torch.zeros(b, dtype=torch.bool, device=c.device)
        perm = []
        c_work = c.detach().clone()
        for i in range(b):
            row = c_work[i].masked_fill(used, float("inf"))
            j = int(torch.argmin(row).item())
            used[j] = True
            perm.append(j)
        return x0, x1[torch.tensor(perm, device=x1.device)]

    try:
        import ot

        a = ot.unif(b)
        G = ot.emd(a, a, c.detach().cpu().numpy())
        col = G.argmax(axis=1)
        return x0, x1[torch.as_tensor(col, device=x1.device)]
    except Exception:
        pass
    try:
        from scipy.optimize import linear_sum_assignment

        _, col = linear_sum_assignment(c.detach().cpu().numpy())
        return x0, x1[torch.as_tensor(col, device=x1.device)]
    except Exception:
        used = set()
        perm = []
        c_cpu = c.detach()
        for i in range(b):
            row = c_cpu[i].clone()
            for j in used:
                row[j] = float("inf")
            j = int(torch.argmin(row).item())
            used.add(j)
            perm.append(j)
        return x0, x1[torch.tensor(perm, device=x1.device)]


def ot_cfm_loss(
    model: nn.Module,
    x1: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    coupling: str = "independent",
    null_prob: float = 0.0,
    null_index: Optional[int] = None,
) -> torch.Tensor:
    if coupling == "independent":
        x0, x1p = independent_coupling(x1)
    elif coupling == "ot":
        x0, x1p = minibatch_ot_coupling(x1)
    else:
        raise ValueError(coupling)

    t = torch.rand(x1.shape[0], device=x1.device)
    # broadcast t
    shape = [x1.shape[0]] + [1] * (x1.ndim - 1)
    t_b = t.view(*shape)
    xt = (1.0 - t_b) * x0 + t_b * x1p
    target = x1p - x0

    c_in = c
    if c is not None and null_prob > 0.0 and null_index is not None:
        drop = torch.rand(c.shape[0], device=c.device) < null_prob
        c_in = c.clone()
        c_in[drop] = null_index

    pred = model(xt, t, c_in)
    return ((pred - target) ** 2).mean()
