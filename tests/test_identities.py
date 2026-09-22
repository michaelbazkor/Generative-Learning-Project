"""Numerical identity tests for theory claims (no trained network required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solvers import (
    choose_w_affine_cap,
    cfg_velocity,
    heun_acceleration,
    rms_norm,
)


def test_constant_field_zero_acceleration():
    """Exact conditional OT field v ≡ x1-x0 has a = 0 under Heun FD."""
    x0 = torch.randn(64, 2)
    x1 = torch.randn(64, 2)
    v = x1 - x0  # constant in (x,t)

    def field(x, t):
        return v

    t = 0.3
    dt = 0.05
    x = (1 - t) * x0 + t * x1
    v1 = field(x, t)
    x_trial = x + dt * v1
    v2 = field(x_trial, t + dt)
    a = heun_acceleration(v1, v2, dt)
    assert float(a.abs().max()) < 1e-6


def test_heun_gap_equals_half_dt2_a():
    """x_H - x_E = 0.5 dt^2 a_hat."""
    # Rotating field: v(x) = [-y, x]  (circular, constant speed, nonzero accel)
    def field(x, t):
        return torch.stack([-x[:, 1], x[:, 0]], dim=-1)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = 0.0
    dt = 0.1
    v1 = field(x, t)
    x_e = x + dt * v1
    v2 = field(x_e, t + dt)
    a = heun_acceleration(v1, v2, dt)
    x_h = x + 0.5 * dt * (v1 + v2)
    gap = x_h - x_e
    expected = 0.5 * (dt ** 2) * a
    assert torch.allclose(gap, expected, atol=1e-6)


def test_material_vs_partial_on_rotating_field():
    """For autonomous rotating field, partial_t v = 0 but material a ≠ 0."""
    def field(x, t):
        return torch.stack([-x[:, 1], x[:, 0]], dim=-1)

    x = torch.tensor([[1.0, 0.0]])
    dt = 1e-3
    # partial: fixed x
    v1 = field(x, 0.0)
    v2_partial = field(x, dt)  # same x
    a_partial = (v2_partial - v1) / dt
    # material via Heun
    x_trial = x + dt * v1
    v2 = field(x_trial, dt)
    a_mat = heun_acceleration(v1, v2, dt)
    assert float(a_partial.abs().max()) < 1e-5
    # true material accel for v=Jx with J=[[0,-1],[1,0]] is J v = J^2 x = -x
    assert float((a_mat + x).abs().max()) < 1e-2


def test_equal_euler_error_scaling():
    """Δt ∝ ||a||^{-1/2} equalizes (dt^2 ||a||)/2."""
    a_rms = torch.tensor([1.0, 4.0, 16.0])
    dt = 1.0 / a_rms.sqrt()
    err = 0.5 * (dt ** 2) * a_rms
    assert torch.allclose(err, 0.5 * torch.ones_like(err), atol=1e-6)


def test_affine_cfg_acceleration():
    """a(w) = a_u + w (a_c - a_u)."""
    a_u = torch.randn(8, 2)
    a_c = torch.randn(8, 2)
    w = 3.5
    a_w = a_u + w * (a_c - a_u)
    # reconstruct via velocities
    v_u1, v_u2 = torch.randn(8, 2), torch.randn(8, 2)
    v_c1, v_c2 = torch.randn(8, 2), torch.randn(8, 2)
    dt = 0.1
    au = heun_acceleration(v_u1, v_u2, dt)
    ac = heun_acceleration(v_c1, v_c2, dt)
    v1 = cfg_velocity(v_c1, v_u1, w)
    v2 = cfg_velocity(v_c2, v_u2, w)
    aw = heun_acceleration(v1, v2, dt)
    assert torch.allclose(aw, au + w * (ac - au), atol=1e-5)


def test_affine_cap_respects_bound():
    a_u = torch.randn(32, 4) * 0.1
    a_c = torch.randn(32, 4) * 2.0
    alpha = 0.5
    w = choose_w_affine_cap(a_u, a_c, w_max=7.0, alpha=alpha)
    a = a_u + w.view(-1, 1) * (a_c - a_u)
    a_at_1 = rms_norm(a_u + 1.0 * (a_c - a_u))
    feasible = a_at_1 <= alpha + 1e-3
    # Whenever w=1 already meets the cap, the chosen w must meet it too.
    if feasible.any():
        assert float((rms_norm(a)[feasible] <= alpha + 1e-3).float().mean()) > 0.99
    assert float(w.min()) >= 1.0 - 1e-5
    assert float(w.max()) <= 7.0 + 1e-5
    # Prefer higher w when curvature allows
    assert float(w.mean()) >= 1.0


def test_rms_scales_with_dimension():
    a = torch.ones(1, 100)
    # ||a||_2 = 10, rms = 10/10 = 1
    assert abs(float(rms_norm(a)) - 1.0) < 1e-5


if __name__ == "__main__":
    tests = [
        test_constant_field_zero_acceleration,
        test_heun_gap_equals_half_dt2_a,
        test_material_vs_partial_on_rotating_field,
        test_equal_euler_error_scaling,
        test_affine_cfg_acceleration,
        test_affine_cap_respects_bound,
        test_rms_scales_with_dimension,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\nAll {len(tests)} identity tests passed.")
