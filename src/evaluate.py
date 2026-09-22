"""Shared evaluation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.metrics import off_support_rate, wasserstein2_exact_2d, wasserstein2_sinkhorn
from src.solvers import VelocityField, sample_ode


@torch.no_grad()
def generate_2d(
    model,
    n: int,
    n_steps: int,
    device: torch.device,
    c: Optional[torch.Tensor] = None,
    conditioned: bool = False,
    **solver_kwargs,
):
    field = VelocityField(model, guidance=solver_kwargs.get("w", 1.0), conditioned=conditioned)
    x0 = torch.randn(n, 2, device=device)
    if c is not None:
        c = c.to(device)
    x, trace, traj = sample_ode(field, x0, c=c, n_steps=n_steps, **solver_kwargs)
    return x.cpu(), trace, traj


def eval_w2(samples: torch.Tensor, ref: torch.Tensor) -> float:
    xs = samples.numpy()
    ys = ref.numpy() if isinstance(ref, torch.Tensor) else ref
    if xs.shape[0] > 2000:
        xs = xs[:2000]
    if ys.shape[0] > 2000:
        ys = ys[:2000]
    try:
        return wasserstein2_exact_2d(xs, ys)
    except Exception:
        return wasserstein2_sinkhorn(torch.from_numpy(xs), torch.from_numpy(ys))


def eval_off_support(samples: torch.Tensor, ref: torch.Tensor) -> float:
    return off_support_rate(samples.numpy(), ref.numpy())
