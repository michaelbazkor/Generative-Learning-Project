"""ODE solvers with curvature-aware step size and CFG damping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS: ||x||_2 / sqrt(d)."""
    flat = x.reshape(x.shape[0], -1)
    d = flat.shape[1]
    return flat.norm(dim=1) / (d ** 0.5)


def cfg_velocity(
    v_cond: torch.Tensor, v_uncond: torch.Tensor, w: float | torch.Tensor
) -> torch.Tensor:
    if isinstance(w, (float, int)):
        return v_uncond + float(w) * (v_cond - v_uncond)
    shape = [v_cond.shape[0]] + [1] * (v_cond.ndim - 1)
    return v_uncond + w.view(*shape) * (v_cond - v_uncond)


@dataclass
class StepTrace:
    t: List[float] = field(default_factory=list)
    dt: List[float] = field(default_factory=list)
    a_rms: List[float] = field(default_factory=list)
    w_eff: List[float] = field(default_factory=list)
    nfe: int = 0


class VelocityField:
    """Wraps a model and optional CFG (conditional + unconditional forwards)."""

    def __init__(
        self,
        model: torch.nn.Module,
        guidance: float = 1.0,
        conditioned: bool = False,
    ):
        self.model = model
        self.guidance = guidance
        self.conditioned = conditioned
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    @torch.no_grad()
    def branches(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (v_cond, v_uncond)."""
        use_cfg = self.conditioned and c is not None
        if not use_cfg:
            v = self.model(x, t, c if self.conditioned else None)
            self.nfe += 1
            return v, v
        null = torch.full_like(c, self.model.null_index)
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        c2 = torch.cat([c, null], dim=0)
        v2 = self.model(x2, t2, c2)
        self.nfe += 2  # two logical forwards (batched)
        v_c, v_u = v2.chunk(2, dim=0)
        return v_c, v_u


def heun_acceleration(v1: torch.Tensor, v2: torch.Tensor, dt: float) -> torch.Tensor:
    return (v2 - v1) / max(dt, 1e-12)


def choose_w_affine_cap(
    a_u: torch.Tensor,
    a_c: torch.Tensor,
    w_max: float,
    alpha: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pick per-sample w in [1, w_max] with ||a(w)||_rms <= alpha when possible."""
    a_delta = a_c - a_u
    flat_u = a_u.reshape(a_u.shape[0], -1)
    flat_d = a_delta.reshape(a_delta.shape[0], -1)
    d = flat_u.shape[1]
    au2 = (flat_u ** 2).sum(dim=1) / d
    ad2 = (flat_d ** 2).sum(dim=1) / d
    aud = (flat_u * flat_d).sum(dim=1) / d
    A, B, C = ad2, aud, au2 - alpha ** 2
    disc = B ** 2 - A * C
    w = torch.full((a_u.shape[0],), w_max, device=a_u.device, dtype=a_u.dtype)
    a_at_max = rms_norm(a_u + w_max * a_delta)
    a_at_1 = rms_norm(a_u + 1.0 * a_delta)
    need = a_at_max > alpha
    good = need & (A > eps) & (disc >= 0)
    if good.any():
        sqrt_disc = torch.sqrt(disc.clamp_min(0))
        w1 = (-B + sqrt_disc) / (A + eps)
        w2 = (-B - sqrt_disc) / (A + eps)
        cand = torch.stack([w1, w2], dim=1)
        feasible = (cand >= 1.0) & (cand <= w_max)
        idx = torch.where(good)[0]
        for i in idx:
            opts = cand[i][feasible[i]]
            if opts.numel() > 0:
                w[i] = opts.max()
            else:
                w[i] = 1.0 if a_at_1[i] <= a_at_max[i] else w_max
    bad = need & ~good
    if bad.any():
        w[bad] = torch.where(a_at_1[bad] <= a_at_max[bad], torch.ones_like(w[bad]), w[bad])
    return w.clamp(1.0, w_max)


def damp_w_gamma(w: float, a_rms: torch.Tensor, gamma: float) -> torch.Tensor:
    """Plan formula: w / (1 + γ ||a||). No square on the norm."""
    return torch.full_like(a_rms, w) / (1.0 + gamma * a_rms)


@torch.no_grad()
def sample_ode(
    field: VelocityField,
    x0: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    n_steps: int = 16,
    method: str = "heun",
    step_mode: str = "uniform",
    step_p: float = 0.5,
    eta: float = 0.1,
    eps: float = 1e-5,
    dt_max: Optional[float] = None,
    guidance_mode: str = "fixed",
    w: float = 1.0,
    gamma: float = 0.5,
    alpha: float = 1.0,
    return_trace: bool = False,
    return_traj: bool = False,
) -> Tuple[torch.Tensor, StepTrace, Optional[torch.Tensor]]:
    """Integrate from t=0 to t=1.

    Uniform mode takes exactly n_steps. Adaptive mode uses only
    dt_prop = eta / (||a||_rms^p + eps), clipped to the time left, and stops at t=1.
    """
    del return_trace  # always return trace
    device = x0.device
    x = x0.clone()
    t = 0.0
    exact_n = step_mode == "uniform"
    if dt_max is None and exact_n:
        dt_max = 2.0 / n_steps
    dt_min = 1e-4
    trace = StepTrace()
    traj = [x.clone()] if return_traj else None
    prev_a_rms = torch.zeros(x.shape[0], device=device)
    field.reset_nfe()
    use_cfg = field.conditioned and c is not None
    limit = n_steps if exact_n else 256

    for step_i in range(limit):
        if t >= 1.0 - 1e-8:
            break
        n_left = n_steps - step_i
        remaining = 1.0 - t
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)

        # --- predictor branches at current state ---
        v_c1, v_u1 = field.branches(x, t_tensor, c)
        v1_ref = cfg_velocity(v_c1, v_u1, w) if use_cfg else v_c1

        # probe step for acceleration (reuse later if dt matches)
        if exact_n and n_left == 1:
            dt_probe = remaining
        elif exact_n:
            dt_probe = min(max(remaining / n_left, dt_min), dt_max)
        elif float(prev_a_rms.mean()) > 0:
            mean_prev = float(prev_a_rms.mean().clamp_min(0))
            dt_probe = max(dt_min, min(eta / (mean_prev ** step_p + eps), remaining))
        else:
            dt_probe = min(0.05, remaining)

        x_trial = x + dt_probe * v1_ref
        t2 = min(t + dt_probe, 1.0)
        t2_tensor = torch.full((x.shape[0],), t2, device=device, dtype=x.dtype)
        v_c2, v_u2 = field.branches(x_trial, t2_tensor, c)

        a_u = heun_acceleration(v_u1, v_u2, dt_probe)
        a_c = heun_acceleration(v_c1, v_c2, dt_probe)

        # --- choose guidance ---
        if (not use_cfg) or guidance_mode == "fixed" or w <= 1.0:
            w_eff = torch.full((x.shape[0],), float(w), device=device, dtype=x.dtype)
        elif guidance_mode == "gamma":
            base = prev_a_rms if float(prev_a_rms.mean()) > 0 else rms_norm(
                heun_acceleration(v1_ref, cfg_velocity(v_c2, v_u2, w), dt_probe)
            )
            w_eff = damp_w_gamma(w, base, gamma)
        elif guidance_mode == "affine_cap":
            w_eff = choose_w_affine_cap(a_u, a_c, w, alpha)
        else:
            raise ValueError(guidance_mode)

        a_eff = a_u + w_eff.view(-1, *([1] * (a_u.ndim - 1))) * (a_c - a_u)
        a_rms_eff = rms_norm(a_eff)
        prev_a_rms = a_rms_eff

        # --- choose step size ---
        if exact_n and n_left == 1:
            dt = remaining
        elif step_mode == "uniform":
            dt = remaining / n_left
        elif step_mode == "adaptive_p":
            mean_a = float(a_rms_eff.mean().clamp_min(0).item())
            dt = eta / (mean_a ** step_p + eps)
            if dt_max is not None:
                dt = min(dt, dt_max)
            dt = max(dt_min, min(dt, remaining))
        else:
            raise ValueError(step_mode)

        # --- accept Heun/Euler step with chosen w_eff ---
        v1 = cfg_velocity(v_c1, v_u1, w_eff) if use_cfg else v_c1
        if method == "euler":
            x = x + dt * v1
        elif method == "heun":
            # reuse probe corrector if dt ≈ dt_probe
            if abs(dt - dt_probe) < 1e-12:
                v2 = cfg_velocity(v_c2, v_u2, w_eff) if use_cfg else v_c2
            else:
                x_e = x + dt * v1
                te = torch.full((x.shape[0],), t + dt, device=device, dtype=x.dtype)
                v_ce, v_ue = field.branches(x_e, te, c)
                v2 = cfg_velocity(v_ce, v_ue, w_eff) if use_cfg else v_ce
            x = x + 0.5 * dt * (v1 + v2)
        else:
            raise ValueError(method)

        t = min(t + dt, 1.0)
        trace.t.append(t)
        trace.dt.append(dt)
        trace.a_rms.append(float(a_rms_eff.mean().item()))
        trace.w_eff.append(float(w_eff.mean().item()))
        if return_traj:
            traj.append(x.clone())

    trace.nfe = field.nfe
    traj_t = torch.stack(traj, dim=0) if return_traj else None
    return x, trace, traj_t


@torch.no_grad()
def estimate_partial_dt(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: float,
    dt: float,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    t1 = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    t2 = torch.full((x.shape[0],), t + dt, device=x.device, dtype=x.dtype)
    v1 = model(x, t1, c)
    v2 = model(x, t2, c)
    return (v2 - v1) / dt
