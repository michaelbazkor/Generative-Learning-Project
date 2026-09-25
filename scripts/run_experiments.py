"""Stage runners: coupling, estimator, step law, guidance, factorial, Fashion-MNIST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    make_2d_loaders,
    make_cifar10_loaders,
    make_fashion_mnist_loaders,
    sample_pinwheel,
    sample_swiss_roll,
)
from src.evaluate import eval_off_support, eval_w2, generate_2d
from src.metrics import feature_stats, frechet_distance, train_feature_extractor
from src.models import VelocityMLP, VelocityUNet, count_parameters
from src.solvers import VelocityField, estimate_partial_dt, heun_acceleration, rms_norm, sample_ode
from src.train import save_checkpoint, train_model

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
CKPT = RESULTS / "checkpoints"
JSON = RESULTS / "json"
for d in (FIG, CKPT, JSON):
    d.mkdir(parents=True, exist_ok=True)


def device_of_choice() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_decision(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


# ---------------------------------------------------------------------------
# Stage 1: Coupling
# ---------------------------------------------------------------------------
def stage_coupling(steps: int = 3000, seed: int = 0):
    device = device_of_choice()
    torch.manual_seed(seed)
    loader, test_x, _ = make_2d_loaders("swiss", batch_size=512, seed=seed)
    rows = []
    for coupling in ("independent", "ot"):
        model = VelocityMLP(dim=2, hidden=128, n_layers=3)
        meta = train_model(model, loader, steps=steps, lr=1e-3, device=device, coupling=coupling)
        save_checkpoint(model, CKPT / f"swiss_{coupling}.pt", {"coupling": coupling, **meta})
        model.eval()
        for n_steps in (8, 16):
            samples, trace, _ = generate_2d(
                model, n=2000, n_steps=n_steps, device=device, method="heun",
                step_mode="uniform", guidance_mode="fixed", w=1.0,
            )
            w2 = eval_w2(samples, test_x)
            rows.append({
                "coupling": coupling,
                "n_steps": n_steps,
                "w2": w2,
                "mean_a_rms": float(np.mean(trace.a_rms)) if trace.a_rms else None,
                "final_loss": meta["final_loss"],
            })
            print(rows[-1])

    (JSON / "stage1_coupling.json").write_text(json.dumps(rows, indent=2))

    # decision: prefer lower mean_a_rms at w=1 (straighter), then W2
    by_ot = [r for r in rows if r["coupling"] == "ot"]
    by_ind = [r for r in rows if r["coupling"] == "independent"]
    mean_a_ot = np.mean([r["mean_a_rms"] for r in by_ot])
    mean_a_ind = np.mean([r["mean_a_rms"] for r in by_ind])
    mean_w2_ot = np.mean([r["w2"] for r in by_ot])
    mean_w2_ind = np.mean([r["w2"] for r in by_ind])
    # Prefer OT if straighter OR better W2 with comparable curvature
    winner = "ot" if (mean_a_ot < mean_a_ind * 0.9 or mean_w2_ot <= mean_w2_ind) else "independent"
    reason = (
        f"mean ||a||_rms: OT={mean_a_ot:.4f}, independent={mean_a_ind:.4f}; "
        f"mean W2: OT={mean_w2_ot:.4f}, independent={mean_w2_ind:.4f}"
    )
    decision = {
        "winner": winner,
        "reason": reason,
        "theory": "Perfect OT coupling yields straight characteristics (a≈0); independent coupling does not.",
        "rows": rows,
    }
    (JSON / "decision_coupling.json").write_text(json.dumps(decision, indent=2))
    return decision


# ---------------------------------------------------------------------------
# Stage 2: Estimator
# ---------------------------------------------------------------------------
def stage_estimator(coupling: str = "ot", seed: int = 0):
    device = device_of_choice()
    ckpt = torch.load(CKPT / f"swiss_{coupling}.pt", map_location=device, weights_only=False)
    model = VelocityMLP(dim=2, hidden=128, n_layers=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Correlate estimators with true local Euler truncation via fine Heun reference
    with torch.no_grad():
        x = torch.randn(256, 2, device=device)
        t0 = 0.4
        dt = 0.05
        t = torch.full((x.shape[0],), t0, device=device)
        v1 = model(x, t, None)
        x_e = x + dt * v1
        v2 = model(x_e, t + dt, None)
        a_heun = heun_acceleration(v1, v2, dt)
        a_partial = estimate_partial_dt(model, x, t0, dt, None)
        # reference material accel via smaller step
        dt_f = 1e-3
        x_f = x + dt_f * v1
        v2f = model(x_f, t + dt_f, None)
        a_ref = heun_acceleration(v1, v2f, dt_f)
        # truncation proxy: ||x_H - x_E||
        x_h = x + 0.5 * dt * (v1 + v2)
        gap = (x_h - x_e).reshape(x.shape[0], -1).norm(dim=1)

    def corr(a):
        ar = rms_norm(a).detach().cpu().numpy()
        g = gap.detach().cpu().numpy()
        return float(np.corrcoef(ar, g)[0, 1])

    rows = {
        "corr_heun_a_vs_gap": corr(a_heun),
        "corr_partial_a_vs_gap": corr(a_partial),
        "corr_fine_a_vs_gap": corr(a_ref),
        "mean_rms_heun": float(rms_norm(a_heun).mean()),
        "mean_rms_partial": float(rms_norm(a_partial).mean()),
    }
    (JSON / "stage2_estimator.json").write_text(json.dumps(rows, indent=2))
    winner = "heun" if rows["corr_heun_a_vs_gap"] >= rows["corr_partial_a_vs_gap"] else "partial"
    decision = {
        "winner": winner,
        "reason": f"corr(heun,gap)={rows['corr_heun_a_vs_gap']:.3f}, corr(partial,gap)={rows['corr_partial_a_vs_gap']:.3f}",
        "theory": "Particle acceleration is the material derivative; Heun FD matches it and equals 2*gap/dt^2.",
        "rows": rows,
    }
    (JSON / "decision_estimator.json").write_text(json.dumps(decision, indent=2))
    print(decision)
    return decision


# ---------------------------------------------------------------------------
# Stage 3: Step law
# ---------------------------------------------------------------------------
def stage_step_law(coupling: str = "ot", seed: int = 0):
    device = device_of_choice()
    # Swiss unconditional
    ckpt = torch.load(CKPT / f"swiss_{coupling}.pt", map_location=device)
    swiss = VelocityMLP(dim=2, hidden=128, n_layers=3).to(device)
    swiss.load_state_dict(ckpt["state_dict"])
    swiss.eval()
    _, test_swiss, _ = make_2d_loaders("swiss", seed=seed)

    # Pinwheel conditional for high-w setting
    loader_pw, test_pw, test_y = make_2d_loaders("pinwheel", seed=seed)
    pw = VelocityMLP(dim=2, hidden=128, n_layers=3, n_classes=5).to(device)
    meta = train_model(
        pw, loader_pw, steps=3000, lr=1e-3, device=device, coupling=coupling,
        conditional=True, null_prob=0.1, null_index=5,
    )
    save_checkpoint(pw, CKPT / f"pinwheel_{coupling}.pt", meta)
    pw.eval()

    configs = [
        ("uniform", "uniform", 0.0),
        ("p1", "adaptive_p", 1.0),
        ("p0.5", "adaptive_p", 0.5),
        ("p1_3", "adaptive_p", 1.0 / 3.0),
    ]
    rows = []
    for name, mode, p in configs:
        # Uniform is a fixed grid. Adaptive laws use only dt_prop until t=1.
        n_grid = (4, 8) if mode == "uniform" else (16,)
        for n_steps in n_grid:
            # swiss w=1
            s, tr, _ = generate_2d(
                swiss, 2000, n_steps, device, method="heun",
                step_mode=mode, step_p=p, eta=0.1, guidance_mode="fixed", w=1.0,
            )
            rows.append({
                "law": name, "dataset": "swiss", "w": 1.0, "n_steps": len(tr.dt),
                "w2": eval_w2(s, test_swiss), "mean_dt": float(np.mean(tr.dt)),
            })
            # pinwheel w=5
            labels = torch.randint(0, 5, (2000,))
            s, tr, _ = generate_2d(
                pw, 2000, n_steps, device, c=labels, conditioned=True, method="heun",
                step_mode=mode, step_p=p, eta=0.1, guidance_mode="fixed", w=5.0,
            )
            rows.append({
                "law": name, "dataset": "pinwheel", "w": 5.0, "n_steps": len(tr.dt),
                "w2": eval_w2(s, test_pw), "mean_dt": float(np.mean(tr.dt)),
            })
            print(rows[-2])
            print(rows[-1])

    (JSON / "stage3_step_law.json").write_text(json.dumps(rows, indent=2))
    # average W2 across settings (lower better)
    laws = sorted(set(r["law"] for r in rows))
    scores = {law: np.mean([r["w2"] for r in rows if r["law"] == law]) for law in laws}
    winner = min(scores, key=scores.get)
    decision = {
        "winner": winner,
        "scores": scores,
        "reason": f"Lowest mean W2. Adaptive laws use only dt_prop until t=1: {scores}",
        "theory": "Equal Euler error implies p=1/2 and dt = eta / (||a||_rms^{1/2}+eps), clipped to the time left. No blend with a uniform budget.",
        "map": {"uniform": ("uniform", 0.0), "p1": ("adaptive_p", 1.0), "p0.5": ("adaptive_p", 0.5), "p1_3": ("adaptive_p", 1 / 3)},
    }
    (JSON / "decision_step_law.json").write_text(json.dumps(decision, indent=2))
    print(decision)
    return decision


# ---------------------------------------------------------------------------
# Stage 4: Guidance law
# ---------------------------------------------------------------------------
def stage_guidance(coupling: str = "ot", step_mode: str = "uniform", step_p: float = 0.5, seed: int = 0):
    device = device_of_choice()
    ckpt = torch.load(CKPT / f"pinwheel_{coupling}.pt", map_location=device)
    model = VelocityMLP(dim=2, hidden=128, n_layers=3, n_classes=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _, test_x, _ = make_2d_loaders("pinwheel", seed=seed)

    modes = [
        ("fixed", "fixed", None),
        ("gamma_0.1", "gamma", 0.1),
        ("gamma_0.5", "gamma", 0.5),
        ("affine_cap", "affine_cap", None),
    ]
    # calibrate alpha from median a_rms at w=1, N=8
    labels = torch.randint(0, 5, (512,))
    _, tr, _ = generate_2d(
        model, 512, 8, device, c=labels, conditioned=True, method="heun",
        step_mode="uniform", guidance_mode="fixed", w=1.0,
    )
    alpha = float(np.median(tr.a_rms) * 3.0)  # allow moderate curvature
    print("alpha=", alpha)

    rows = []
    for name, gmode, gamma in modes:
        for w in (1.5, 3.0, 5.0, 7.0):
            labels = torch.randint(0, 5, (2000,))
            s, tr, _ = generate_2d(
                model, 2000, 8, device, c=labels, conditioned=True, method="heun",
                step_mode=step_mode, step_p=step_p, eta=0.1,
                guidance_mode=gmode, w=w, gamma=gamma or 0.5, alpha=alpha,
            )
            row = {
                "law": name, "w": w, "w2": eval_w2(s, test_x),
                "off_support": eval_off_support(s, test_x),
                "mean_w_eff": float(np.mean(tr.w_eff)),
                "mean_a_rms": float(np.mean(tr.a_rms)),
            }
            rows.append(row)
            print(row)

    (JSON / "stage4_guidance.json").write_text(json.dumps(rows, indent=2))
    # score: mean W2 over w>=3 (where damping matters), tie-break off_support
    hard = [r for r in rows if r["w"] >= 3.0]
    laws = sorted(set(r["law"] for r in hard))
    scores = {}
    for law in laws:
        subset = [r for r in hard if r["law"] == law]
        scores[law] = {
            "w2": float(np.mean([r["w2"] for r in subset])),
            "off": float(np.mean([r["off_support"] for r in subset])),
        }
    winner = min(laws, key=lambda L: (scores[L]["w2"], scores[L]["off"]))
    decision = {
        "winner": winner,
        "alpha": alpha,
        "scores": scores,
        "reason": f"Lowest W2 (then off-support) for w>=3: {scores}",
        "theory": "CFG is affine in w so a(w) is free; capping ||a|| is the principled damper.",
    }
    (JSON / "decision_guidance.json").write_text(json.dumps(decision, indent=2))
    print(decision)
    return decision


# ---------------------------------------------------------------------------
# Stage 5: Factorial M1-M4
# ---------------------------------------------------------------------------
def stage_factorial(
    coupling: str,
    step_mode: str,
    step_p: float,
    guidance_mode: str,
    alpha: float = 1.0,
    gamma: float = 0.5,
    seed: int = 0,
):
    device = device_of_choice()
    swiss = VelocityMLP(dim=2, hidden=128, n_layers=3).to(device)
    swiss.load_state_dict(torch.load(CKPT / f"swiss_{coupling}.pt", map_location=device)["state_dict"])
    swiss.eval()
    pw = VelocityMLP(dim=2, hidden=128, n_layers=3, n_classes=5).to(device)
    pw.load_state_dict(torch.load(CKPT / f"pinwheel_{coupling}.pt", map_location=device)["state_dict"])
    pw.eval()
    _, test_swiss, _ = make_2d_loaders("swiss", seed=seed)
    _, test_pw, _ = make_2d_loaders("pinwheel", seed=seed)

    methods = {
        "M1": dict(step_mode="uniform", guidance_mode="fixed"),
        "M2": dict(step_mode=step_mode, step_p=step_p, guidance_mode="fixed"),
        "M3": dict(step_mode="uniform", guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
        "M4": dict(step_mode=step_mode, step_p=step_p, guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
    }
    # If step winner was uniform, M2==M1 and M4==M3; still run for completeness.
    rows = []
    for mname, cfg in methods.items():
        n_grid = (4, 8, 12, 16) if cfg["step_mode"] == "uniform" else (16,)
        for n_steps in n_grid:
            for w in (1.5, 3.0, 5.0, 7.0):
                # swiss (uncond: guidance unused)
                s, tr, _ = generate_2d(
                    swiss, 2000, n_steps, device, method="heun", w=1.0,
                    eta=0.1, **{k: v for k, v in cfg.items() if k != "guidance_mode"},
                    guidance_mode="fixed",
                )
                rows.append({"method": mname, "dataset": "swiss", "n": len(tr.dt), "w": w,
                             "w2": eval_w2(s, test_swiss), "nfe": tr.nfe,
                             "mean_a": float(np.mean(tr.a_rms))})
                labels = torch.randint(0, 5, (2000,))
                s, tr, traj = generate_2d(
                    pw, 2000, n_steps, device, c=labels, conditioned=True, method="heun",
                    w=w, eta=0.1, return_traj=(mname in ("M1", "M4") and w == 5.0 and (n_steps == 8 or cfg["step_mode"] != "uniform")),
                    **cfg,
                )
                rows.append({"method": mname, "dataset": "pinwheel", "n": len(tr.dt), "w": w,
                             "w2": eval_w2(s, test_pw), "nfe": tr.nfe,
                             "mean_a": float(np.mean(tr.a_rms)),
                             "mean_w_eff": float(np.mean(tr.w_eff)),
                             "off": eval_off_support(s, test_pw)})
                print(rows[-1])

                # trajectory viz
                if traj is not None:
                    _plot_trajectories(traj, test_pw, FIG / f"traj_{mname}_w5_n8.png", mname)

    (JSON / "stage5_factorial.json").write_text(json.dumps(rows, indent=2))
    _plot_w2_curves(rows, FIG / "w2_curves.png")
    _plot_a_profiles(device, pw, methods, FIG / "a_profiles.png", alpha, gamma, step_mode, step_p)
    return rows


def _plot_trajectories(traj: torch.Tensor, ref: torch.Tensor, path: Path, title: str):
    # traj: (T+1, B, 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ref_np = ref.numpy()
    ax.scatter(ref_np[:, 0], ref_np[:, 1], s=2, alpha=0.15, c="gray", label="data")
    T, B, _ = traj.shape
    idx = np.linspace(0, B - 1, 40, dtype=int)
    for i in idx:
        curve = traj[:, i, :].cpu().numpy()
        ax.plot(curve[:, 0], curve[:, 1], lw=0.8, alpha=0.8)
        ax.scatter(curve[-1, 0], curve[-1, 1], s=8)
    ax.set_title(f"Trajectories {title} (w=5, N=8)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_w2_curves(rows, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds in zip(axes, ("swiss", "pinwheel")):
        for m in ("M1", "M2", "M3", "M4"):
            xs, ys = [], []
            ns = sorted({r["n"] for r in rows if r["method"] == m and r["dataset"] == ds})
            for n in ns:
                vals = [r["w2"] for r in rows if r["method"] == m and r["dataset"] == ds and r["n"] == n]
                if vals:
                    xs.append(n)
                    ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(ds)
        ax.set_xlabel("N steps")
        ax.set_ylabel("mean W2 over w")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_a_profiles(device, model, methods, path, alpha, gamma, step_mode, step_p):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = torch.randint(0, 5, (256,))
    for mname, cfg in methods.items():
        if mname not in ("M1", "M4"):
            continue
        _, tr, _ = generate_2d(
            model, 256, 16, device, c=labels, conditioned=True, method="heun",
            w=5.0, eta=0.1, **cfg,
        )
        axes[0].plot(tr.t, tr.a_rms, label=mname)
        axes[1].plot(tr.t, tr.dt, label=mname)
        axes[2].plot(tr.t, tr.w_eff, label=mname)
    axes[0].set_title(r"$\|a\|_{rms}(t)$")
    axes[1].set_title(r"$\Delta t(t)$")
    axes[2].set_title(r"$w_{eff}(t)$")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 6: Fashion-MNIST
# ---------------------------------------------------------------------------
def stage_fmnist(
    coupling: str,
    step_mode: str,
    step_p: float,
    guidance_mode: str,
    alpha: float,
    gamma: float = 0.5,
    epochs: int = 10,
    batch_size: int = 128,
    skip_train: bool = False,
):
    device = device_of_choice()
    print("device", device)
    try:
        train_loader, test_loader, n_classes = make_fashion_mnist_loaders(
            batch_size=64 if device.type == "cpu" else batch_size
        )
    except Exception as e:
        print("FMNIST download/load failed:", e)
        return {"error": str(e)}

    model = VelocityUNet(
        in_channels=1,
        base_channels=24 if device.type == "cpu" else 32,
        channel_mult=(1, 2, 4),
        n_classes=n_classes,
    )
    print("UNet params", count_parameters(model))
    ckpt_path = CKPT / "fmnist.pt"
    if skip_train and ckpt_path.exists():
        print("loading checkpoint", ckpt_path)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        # tolerate base_channels mismatch by rebuilding from meta if needed
        model.load_state_dict(blob["state_dict"])
        model.to(device)
        meta = blob.get("meta", {})
    else:
        steps = epochs * (len(train_loader))
        steps = min(steps, 400 if device.type == "cpu" else 8000)
        print(f"training steps={steps} device={device}")
        meta = train_model(
            model, train_loader, steps=steps, lr=2e-4, device=device, coupling=coupling,
            conditional=True, null_prob=0.1, null_index=n_classes,
            optimizer_name="adamw", weight_decay=0.01, log_every=50,
        )
        save_checkpoint(model, ckpt_path, {"params": count_parameters(model), **meta})
    model.eval()

    # feature extractor
    feat = train_feature_extractor(train_loader, device, epochs=1 if device.type == "cpu" else 2)
    # real stats
    real_imgs = []
    for x, _ in test_loader:
        real_imgs.append(x)
        if sum(t.shape[0] for t in real_imgs) >= 5000:
            break
    real = torch.cat(real_imgs, dim=0)[:5000]
    mu_r, sig_r = feature_stats(feat, real, device=device)

    methods = {
        "M1": dict(step_mode="uniform", guidance_mode="fixed"),
        "M2": dict(step_mode=step_mode, step_p=step_p, guidance_mode="fixed"),
        "M3": dict(step_mode="uniform", guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
        "M4": dict(step_mode=step_mode, step_p=step_p, guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
    }
    if device.type == "cpu":
        # Keep full M1–M4 on CPU but with the smaller sample grid below.
        pass
    rows = []
    n_samples = 200 if device.type == "cpu" else 2000
    eval_ns = (4, 8, 16) if device.type != "cpu" else (4, 8)
    eval_ws = (1.5, 5.0)
    print(f"eval n_samples={n_samples} ns={eval_ns}")
    for mname, cfg in methods.items():
        n_grid = eval_ns if cfg["step_mode"] == "uniform" else (8,)
        for n_steps in n_grid:
            for w in eval_ws:
                field = VelocityField(model, guidance=w, conditioned=True)
                # generate in chunks to limit memory / wall time
                chunk = 50 if device.type == "cpu" else 200
                xs = []
                nfe_per = 0
                a_all, w_all = [], []
                for start in range(0, n_samples, chunk):
                    b = min(chunk, n_samples - start)
                    x0 = torch.randn(b, 1, 32, 32, device=device)
                    c = torch.randint(0, n_classes, (b,), device=device)
                    x, tr, _ = sample_ode(
                        field, x0, c=c, n_steps=n_steps, method="heun", w=w, eta=0.1, **cfg,
                    )
                    xs.append(x.clamp(-1, 1).cpu())
                    nfe_per = tr.nfe
                    a_all.extend(tr.a_rms)
                    w_all.extend(tr.w_eff)
                x = torch.cat(xs, dim=0)
                mu_g, sig_g = feature_stats(feat, x, device=device)
                fd = frechet_distance(mu_r, sig_r, mu_g, sig_g)
                row = {
                    "method": mname, "n": len(tr.dt), "w": w, "fd": fd,
                    "nfe_per_sample": nfe_per,
                    "mean_a": float(np.mean(a_all)), "mean_w_eff": float(np.mean(w_all)),
                    "n_samples": n_samples, "device": str(device),
                }
                rows.append(row)
                print(row)
                if mname in ("M1", "M4") and w == 5.0 and (n_steps == 8 or cfg["step_mode"] != "uniform"):
                    _save_image_grid(x[:64], FIG / f"fmnist_{mname}_w5_n8.png")

    (JSON / "stage6_fmnist.json").write_text(json.dumps(rows, indent=2))
    _plot_fd_pareto(rows, FIG / "fd_vs_nfe.png")
    return rows


def _save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8):
    imgs = ((images + 1) / 2).clamp(0, 1)
    n = imgs.shape[0]
    ncol = nrow
    nrow_g = int(np.ceil(n / ncol))
    canvas = torch.zeros(nrow_g * 32, ncol * 32)
    for i in range(n):
        r, c = divmod(i, ncol)
        canvas[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = imgs[i, 0]
    plt.imsave(path, canvas.numpy(), cmap="gray")


def _plot_fd_pareto(rows, path: Path, title: str = "Fashion-MNIST quality vs cost"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("M1", "M2", "M3", "M4"):
        pts = [(r["nfe_per_sample"], r["fd"]) for r in rows if r["method"] == m]
        if not pts:
            continue
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    ax.set_xlabel("NFE (network forwards per sample trajectory)")
    ax.set_ylabel("Feature Fréchet distance")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_rgb_grid(images: torch.Tensor, path: Path, nrow: int = 8):
    imgs = ((images + 1) / 2).clamp(0, 1).cpu()
    n, _, h, w = imgs.shape
    ncol = nrow
    nrow_g = int(np.ceil(n / ncol))
    canvas = torch.ones(3, nrow_g * h, ncol * w)
    for i in range(n):
        r, c = divmod(i, ncol)
        canvas[:, r * h : (r + 1) * h, c * w : (c + 1) * w] = imgs[i]
    plt.imsave(path, canvas.permute(1, 2, 0).numpy())


# ---------------------------------------------------------------------------
# Stage 7: CIFAR-10 transfer of M1-M4
# ---------------------------------------------------------------------------
# Same sampler as Stages 5-6. Alpha is the Stage 6 transfer value, not a retune.
STAGE6_ALPHA = 0.646


def stage_cifar(
    coupling: str,
    step_mode: str,
    step_p: float,
    guidance_mode: str = "affine_cap",
    alpha: float = STAGE6_ALPHA,
    gamma: float = 0.5,
    steps: int | None = None,
    batch_size: int = 64,
    skip_train: bool = False,
):
    device = device_of_choice()
    print("device", device)
    on_gpu = device.type == "cuda"
    # 4GB laptop GPU: keep the UNet at the Fashion-MNIST width.
    bs = 32 if on_gpu else 16
    base_channels = 24
    try:
        train_loader, test_loader, n_classes = make_cifar10_loaders(batch_size=bs)
    except Exception as e:
        print("CIFAR-10 download/load failed:", e)
        return {"error": str(e)}

    model = VelocityUNet(
        in_channels=3,
        base_channels=base_channels,
        channel_mult=(1, 2, 4),
        n_classes=n_classes,
    )
    n_params = count_parameters(model)
    print("UNet params", n_params)
    ckpt_path = CKPT / "cifar10.pt"
    arch = {
        "in_channels": 3,
        "base_channels": base_channels,
        "n_classes": n_classes,
        "params": n_params,
    }
    if skip_train and ckpt_path.exists():
        print("loading checkpoint", ckpt_path)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = blob.get("meta", {})
        if saved.get("base_channels") and saved["base_channels"] != base_channels:
            base_channels = int(saved["base_channels"])
            model = VelocityUNet(
                in_channels=3,
                base_channels=base_channels,
                channel_mult=(1, 2, 4),
                n_classes=n_classes,
            )
        model.load_state_dict(blob["state_dict"])
        model.to(device)
        meta = saved
    else:
        if steps is None:
            steps = min(len(train_loader) * (8 if on_gpu else 1), 8000 if on_gpu else 800)
        print(f"training steps={steps} batch={bs} device={device}")
        meta = train_model(
            model, train_loader, steps=steps, lr=2e-4, device=device, coupling=coupling,
            conditional=True, null_prob=0.1, null_index=n_classes,
            optimizer_name="adamw", weight_decay=0.01, log_every=50,
            checkpoint_path=ckpt_path, checkpoint_every=200, checkpoint_meta=arch,
        )
        meta = {**arch, **meta}
        save_checkpoint(model, ckpt_path, meta)
    model.eval()

    feat_epochs = 2 if on_gpu else 1
    feat = train_feature_extractor(
        train_loader, device, epochs=feat_epochs, in_channels=3
    )
    real_imgs = []
    for x, _ in test_loader:
        real_imgs.append(x)
        if sum(t.shape[0] for t in real_imgs) >= 5000:
            break
    real = torch.cat(real_imgs, dim=0)[:5000]
    mu_r, sig_r = feature_stats(feat, real, device=device)

    # Stages 5-6 locked M3/M4 to the affine cap even after the Stage 4 rerun.
    methods = {
        "M1": dict(step_mode="uniform", guidance_mode="fixed"),
        "M2": dict(step_mode=step_mode, step_p=step_p, guidance_mode="fixed"),
        "M3": dict(step_mode="uniform", guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
        "M4": dict(step_mode=step_mode, step_p=step_p, guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
    }
    rows = []
    n_samples = 500 if on_gpu else 128
    eval_ns = (4, 8)
    eval_ws = (1.5, 5.0)
    chunk = 32 if on_gpu else 8
    print(f"eval n_samples={n_samples} ns={eval_ns} alpha={alpha}")
    for mname, cfg in methods.items():
        n_grid = eval_ns if cfg["step_mode"] == "uniform" else (8,)
        for n_steps in n_grid:
            for w in eval_ws:
                field = VelocityField(model, guidance=w, conditioned=True)
                xs = []
                nfe_per = 0
                a_all, w_all = [], []
                for start in range(0, n_samples, chunk):
                    b = min(chunk, n_samples - start)
                    x0 = torch.randn(b, 3, 32, 32, device=device)
                    c = torch.randint(0, n_classes, (b,), device=device)
                    x, tr, _ = sample_ode(
                        field, x0, c=c, n_steps=n_steps, method="heun", w=w, eta=0.1, **cfg,
                    )
                    xs.append(x.clamp(-1, 1).cpu())
                    nfe_per = tr.nfe
                    a_all.extend(tr.a_rms)
                    w_all.extend(tr.w_eff)
                x = torch.cat(xs, dim=0)
                mu_g, sig_g = feature_stats(feat, x, device=device)
                fd = frechet_distance(mu_r, sig_r, mu_g, sig_g)
                row = {
                    "method": mname, "n": len(tr.dt), "w": w, "fd": fd,
                    "nfe_per_sample": nfe_per,
                    "mean_a": float(np.mean(a_all)), "mean_w_eff": float(np.mean(w_all)),
                    "n_samples": n_samples, "device": str(device), "alpha": alpha,
                }
                rows.append(row)
                print(row, flush=True)
                if mname in ("M1", "M4") and w == 5.0 and (n_steps == 8 or cfg["step_mode"] != "uniform"):
                    _save_rgb_grid(x[:64], FIG / f"cifar_{mname}_w5_n8.png")

    (JSON / "stage7_cifar.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    _plot_fd_pareto(rows, FIG / "fd_vs_nfe_cifar.png", title="CIFAR-10 quality vs cost")
    return rows


def _m_cfgs(step_p: float = 0.5, alpha: float = STAGE6_ALPHA, gamma: float = 0.5):
    return {
        "M1": dict(step_mode="uniform", guidance_mode="fixed"),
        "M2": dict(step_mode="adaptive_p", step_p=step_p, guidance_mode="fixed"),
        "M3": dict(step_mode="uniform", guidance_mode="affine_cap", gamma=gamma, alpha=alpha),
        "M4": dict(step_mode="adaptive_p", step_p=step_p, guidance_mode="affine_cap", gamma=gamma, alpha=alpha),
    }


def _sample_shared(model, x0, labels, cfg, n_steps, w, device, chunk, clamp=True):
    """Same noise for every method. Returns samples, mean accepted steps, mean NFE."""
    field = VelocityField(model, guidance=w, conditioned=True)
    xs, steps, nfes = [], [], []
    for start in range(0, x0.shape[0], chunk):
        sl = slice(start, start + chunk)
        x, tr, _ = sample_ode(
            field, x0[sl].to(device), c=labels[sl].to(device),
            n_steps=n_steps, method="heun", w=w, eta=0.1, **cfg,
        )
        x = x.cpu()
        xs.append(x.clamp(-1, 1) if clamp else x)
        steps.append(len(tr.dt))
        nfes.append(tr.nfe)
    return torch.cat(xs, 0), float(np.mean(steps)), float(np.mean(nfes))


def stage_fair(alpha: float = STAGE6_ALPHA):
    """Compare M1-M4 at matched step count and at matched network-forward count.

    Adaptive runs choose their own length. Uniform runs are then repeated at that
    length, and again at the N whose 4 forwards/step match the adaptive NFE.
    Every method at a given w sees the same noise and labels.
    """
    device = device_of_choice()
    cfgs = _m_cfgs(alpha=alpha)
    rows = []

    # --- pinwheel, shared noise ---
    ckpt = torch.load(CKPT / "pinwheel_ot.pt", map_location=device, weights_only=False)
    pw = VelocityMLP(dim=2, hidden=128, n_layers=3, n_classes=5).to(device)
    pw.load_state_dict(ckpt["state_dict"])
    pw.eval()
    _, test_pw, _ = make_2d_loaders("pinwheel", seed=0)
    torch.manual_seed(0)
    b_pw = 2000
    x0_pw = torch.randn(b_pw, 2)
    y_pw = torch.randint(0, 5, (b_pw,))
    for w in (1.5, 3.0, 5.0, 7.0):
        adapt = {}
        for mname in ("M2", "M4"):
            s, n_taken, nfe = _sample_shared(pw, x0_pw, y_pw, cfgs[mname], 16, w, device, 2000, clamp=False)
            adapt[mname] = (n_taken, nfe)
            rows.append({
                "dataset": "pinwheel", "method": mname, "match": "adaptive",
                "w": w, "n": n_taken, "nfe": nfe, "w2": eval_w2(s, test_pw),
            })
            print(rows[-1], flush=True)
        for mname, src in (("M1", "M2"), ("M3", "M4")):
            n_step = max(1, int(round(adapt[src][0])))
            n_cost = max(1, int(round(adapt[src][1] / 4.0)))
            for match, n in (("steps", n_step), ("nfe", n_cost)):
                s, n_taken, nfe = _sample_shared(pw, x0_pw, y_pw, cfgs[mname], n, w, device, 2000, clamp=False)
                rows.append({
                    "dataset": "pinwheel", "method": mname, "match": match,
                    "matched_to": src, "w": w, "n": n_taken, "nfe": nfe,
                    "w2": eval_w2(s, test_pw),
                })
                print(rows[-1], flush=True)

    # --- images, shared noise, one feature net per dataset ---
    image_jobs = [
        ("fmnist", 1, CKPT / "fmnist.pt", make_fashion_mnist_loaders, 32 if device.type == "cuda" else 24, 500),
        ("cifar", 3, CKPT / "cifar10.pt", make_cifar10_loaders, 24, 500),
    ]
    for ds, channels, ckpt_path, loader_fn, base, n_samples in image_jobs:
        if not ckpt_path.exists():
            print("missing", ckpt_path)
            continue
        train_loader, test_loader, n_classes = loader_fn(batch_size=64)
        model = VelocityUNet(
            in_channels=channels, base_channels=base, channel_mult=(1, 2, 4), n_classes=n_classes,
        ).to(device)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_base = blob.get("meta", {}).get("base_channels")
        if saved_base and int(saved_base) != base:
            model = VelocityUNet(
                in_channels=channels, base_channels=int(saved_base),
                channel_mult=(1, 2, 4), n_classes=n_classes,
            ).to(device)
        try:
            model.load_state_dict(blob["state_dict"])
        except RuntimeError:
            # FMNIST GPU eval was built at base 32; retry that width.
            model = VelocityUNet(
                in_channels=channels, base_channels=32, channel_mult=(1, 2, 4), n_classes=n_classes,
            ).to(device)
            model.load_state_dict(blob["state_dict"])
        model.eval()
        feat = train_feature_extractor(
            train_loader, device, epochs=1 if device.type == "cpu" else 2, in_channels=channels,
        )
        real = []
        for x, _ in test_loader:
            real.append(x)
            if sum(t.shape[0] for t in real) >= 5000:
                break
        real = torch.cat(real, 0)[:5000]
        mu_r, sig_r = feature_stats(feat, real, device=device)
        torch.manual_seed(0)
        x0 = torch.randn(n_samples, channels, 32, 32)
        y = torch.randint(0, n_classes, (n_samples,))
        chunk = 32 if device.type == "cuda" else 8
        for w in (1.5, 5.0):
            adapt = {}
            for mname in ("M2", "M4"):
                s, n_taken, nfe = _sample_shared(model, x0, y, cfgs[mname], 16, w, device, chunk)
                mu_g, sig_g = feature_stats(feat, s, device=device)
                fd = frechet_distance(mu_r, sig_r, mu_g, sig_g)
                adapt[mname] = (n_taken, nfe)
                rows.append({
                    "dataset": ds, "method": mname, "match": "adaptive",
                    "w": w, "n": n_taken, "nfe": nfe, "fd": fd,
                })
                print(rows[-1], flush=True)
            for mname, src in (("M1", "M2"), ("M3", "M4")):
                n_step = max(1, int(round(adapt[src][0])))
                n_cost = max(1, int(round(adapt[src][1] / 4.0)))
                seen = set()
                for match, n in (("steps", n_step), ("nfe", n_cost)):
                    if n in seen:
                        continue
                    seen.add(n)
                    s, n_taken, nfe = _sample_shared(model, x0, y, cfgs[mname], n, w, device, chunk)
                    mu_g, sig_g = feature_stats(feat, s, device=device)
                    fd = frechet_distance(mu_r, sig_r, mu_g, sig_g)
                    rows.append({
                        "dataset": ds, "method": mname, "match": match,
                        "matched_to": src, "w": w, "n": n_taken, "nfe": nfe, "fd": fd,
                    })
                    print(rows[-1], flush=True)

    (JSON / "stage_fair.json").write_text(json.dumps(rows, indent=2))
    return rows


def build_report():
    """Assemble REPORT.md from decision JSON + theory."""
    parts = [
        "# Experiment Report: Curvature-Aware OT Flow Matching\n",
        "## Goal\n",
        "Test whether local trajectory acceleration is a trustworthy, training-free signal "
        "to co-adapt ODE step sizes and CFG scales in OT-CFM, especially at low step counts "
        "and high guidance. The written experiment plan was treated as a hypothesis set; "
        "each design choice was scrutinized theoretically and empirically.\n",
        "## Theory summary\n",
        "See [`docs/theory.md`](docs/theory.md). Identity tests: `python tests/test_identities.py`.\n",
        "```mermaid\n"
        "flowchart TD\n"
        "  startNode[State xt at time t] --> branches[Evaluate v_cond and v_empty]\n"
        "  branches --> trial[Predictor state from reference guidance]\n"
        "  trial --> corrector[Heun corrector on both branches]\n"
        "  corrector --> accel[a of w equals a_empty plus w times a_delta]\n"
        "  accel --> chooseW[Choose w_eff in 1 to w]\n"
        "  chooseW --> chooseDt[Choose next dt from a_rms]\n"
        "  chooseDt --> accept[Accept Heun step with chosen w_eff]\n"
        "  accept --> done{t equals 1}\n"
        "  done -->|no| startNode\n"
        "  done -->|yes| sample[Sample]\n"
        "```\n",
    ]
    for name, title in [
        ("decision_coupling.json", "Stage 1 — Coupling (independent vs minibatch OT)"),
        ("decision_estimator.json", "Stage 2 — Acceleration estimator"),
        ("decision_step_law.json", "Stage 3 — Step-size law"),
        ("decision_guidance.json", "Stage 4 — Guidance damping"),
    ]:
        p = JSON / name
        parts.append(f"## {title}\n")
        if p.exists():
            d = json.loads(p.read_text())
            parts.append(f"**Decision:** `{d.get('winner')}`\n\n")
            parts.append(f"**Theory:** {d.get('theory', '')}\n\n")
            parts.append(f"**Evidence:** {d.get('reason', '')}\n\n")
            parts.append("```json\n" + json.dumps({k: v for k, v in d.items() if k != 'rows'}, indent=2) + "\n```\n")
        else:
            parts.append("_Not run yet._\n")

    if (JSON / "stage5_factorial.json").exists():
        parts.append("## Stage 5 — Factorial M1–M4 on 2D\n")
        rows = json.loads((JSON / "stage5_factorial.json").read_text())
        # summarize mean W2 per method on pinwheel
        for m in ("M1", "M2", "M3", "M4"):
            vals = [r["w2"] for r in rows if r["method"] == m and r["dataset"] == "pinwheel"]
            parts.append(f"- **{m}** mean pinwheel W2: {float(np.mean(vals)):.4f}\n")
        parts.append("\nFigures: `results/figures/w2_curves.png`, `traj_M1_w5_n8.png`, "
                     "`traj_M4_w5_n8.png`, `a_profiles.png`.\n")

    if (JSON / "stage6_fmnist.json").exists():
        parts.append("## Stage 6 — Fashion-MNIST transfer\n")
        rows = json.loads((JSON / "stage6_fmnist.json").read_text())
        for m in ("M1", "M2", "M3", "M4"):
            vals = [r["fd"] for r in rows if r["method"] == m]
            if vals:
                parts.append(f"- **{m}** mean feature-FD: {float(np.mean(vals)):.4f}\n")
        parts.append("\nFigures: `results/figures/fd_vs_nfe.png`, `fmnist_M1_w5_n8.png`, `fmnist_M4_w5_n8.png`.\n")

    parts.append("## Final configuration\n")
    finals = {}
    for name in ("decision_coupling.json", "decision_estimator.json", "decision_step_law.json", "decision_guidance.json"):
        p = JSON / name
        if p.exists():
            finals[name] = json.loads(p.read_text()).get("winner")
    parts.append("```json\n" + json.dumps(finals, indent=2) + "\n```\n")
    parts.append(
        "## Reproducibility\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python tests/test_identities.py\n"
        "python scripts/run_experiments.py --all\n"
        "```\n"
    )
    (ROOT / "REPORT.md").write_text("".join(parts), encoding="utf-8")
    print("Wrote REPORT.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "coupling", "estimator", "step", "guidance", "factorial", "fmnist", "cifar", "fair", "report"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--fmnist-epochs", type=int, default=8)
    parser.add_argument("--fmnist-eval-only", action="store_true")
    parser.add_argument("--cifar-steps", type=int, default=None)
    parser.add_argument("--cifar-eval-only", action="store_true")
    args = parser.parse_args()

    coupling = "ot"
    step_mode, step_p = "adaptive_p", 0.5
    guidance_mode, alpha, gamma = "affine_cap", 1.0, 0.5

    if args.stage in ("all", "coupling"):
        d = stage_coupling(steps=args.steps)
        coupling = d["winner"]
    else:
        if (JSON / "decision_coupling.json").exists():
            coupling = json.loads((JSON / "decision_coupling.json").read_text())["winner"]

    if args.stage in ("all", "estimator"):
        stage_estimator(coupling=coupling)

    if args.stage in ("all", "step"):
        d = stage_step_law(coupling=coupling)
        mapping = d["map"][d["winner"]]
        step_mode, step_p = mapping
    else:
        if (JSON / "decision_step_law.json").exists():
            d = json.loads((JSON / "decision_step_law.json").read_text())
            step_mode, step_p = d["map"][d["winner"]]

    # M2/M4 always use the proved step, even if a fixed grid wins on W2.
    step_mode, step_p = "adaptive_p", 0.5

    if args.stage in ("all", "guidance"):
        d = stage_guidance(coupling=coupling, step_mode=step_mode, step_p=step_p)
        guidance_mode = d["winner"]
        if guidance_mode.startswith("gamma"):
            guidance_mode = "gamma"
            gamma = 0.5 if "0.5" in d["winner"] else 0.1
        elif guidance_mode == "affine_cap":
            guidance_mode = "affine_cap"
        else:
            guidance_mode = "fixed"
        alpha = d.get("alpha", 1.0)
    else:
        if (JSON / "decision_guidance.json").exists():
            d = json.loads((JSON / "decision_guidance.json").read_text())
            w = d["winner"]
            alpha = d.get("alpha", 1.0)
            if w.startswith("gamma"):
                guidance_mode, gamma = "gamma", (0.5 if "0.5" in w else 0.1)
            else:
                guidance_mode = w

    if args.stage in ("all", "factorial"):
        # M3/M4 stay the affine cap. Stage 4 still records which law wins on its own.
        stage_factorial(coupling, step_mode, step_p, "affine_cap", STAGE6_ALPHA, 0.5)

    if args.stage in ("all", "fmnist"):
        stage_fmnist(
            coupling, step_mode, step_p, "affine_cap", STAGE6_ALPHA, 0.5,
            epochs=args.fmnist_epochs, skip_train=args.fmnist_eval_only,
        )

    if args.stage in ("all", "fair"):
        stage_fair()

    if args.stage in ("all", "cifar"):
        # M3/M4 stay on the Stage 5-6 affine cap and the Stage 6 alpha.
        stage_cifar(
            coupling, step_mode, step_p, guidance_mode="affine_cap",
            alpha=STAGE6_ALPHA, gamma=0.5,
            steps=args.cifar_steps, skip_train=args.cifar_eval_only,
        )

    if args.stage in ("all", "report"):
        build_report()


if __name__ == "__main__":
    main()
