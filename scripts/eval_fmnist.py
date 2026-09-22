"""Lean FMNIST eval from checkpoint (CPU-friendly)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_fashion_mnist_loaders
from src.metrics import feature_stats, frechet_distance, train_feature_extractor
from src.models import VelocityUNet, count_parameters
from src.solvers import VelocityField, sample_ode

FIG = ROOT / "results" / "figures"
JSON = ROOT / "results" / "json"
CKPT = ROOT / "results" / "checkpoints"
FIG.mkdir(parents=True, exist_ok=True)


def save_grid(images, path, nrow=8):
    imgs = ((images + 1) / 2).clamp(0, 1)
    n = imgs.shape[0]
    ncol = nrow
    nrow_g = int(np.ceil(n / ncol))
    canvas = torch.zeros(nrow_g * 32, ncol * 32)
    for i in range(n):
        r, c = divmod(i, ncol)
        canvas[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = imgs[i, 0]
    plt.imsave(path, canvas.numpy(), cmap="gray")


def main():
    device = torch.device("cpu")
    print("loading data", flush=True)
    train_loader, test_loader, n_classes = make_fashion_mnist_loaders(batch_size=64)
    print("loading model", flush=True)
    model = VelocityUNet(in_channels=1, base_channels=24, channel_mult=(1, 2, 4), n_classes=n_classes)
    blob = torch.load(CKPT / "fmnist.pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print("params", count_parameters(model), flush=True)

    print("train feature extractor (1 epoch)", flush=True)
    feat = train_feature_extractor(train_loader, device, epochs=1)
    real = []
    for x, _ in test_loader:
        real.append(x)
        if sum(t.shape[0] for t in real) >= 1000:
            break
    real = torch.cat(real, dim=0)[:1000]
    mu_r, sig_r = feature_stats(feat, real, device=device)

    # load decisions for method configs
    step = json.loads((JSON / "decision_step_law.json").read_text())
    step_mode, step_p = step["map"][step["winner"]]
    guid = json.loads((JSON / "decision_guidance.json").read_text())
    alpha = guid["alpha"]
    gname = guid["winner"]
    if gname.startswith("gamma"):
        guidance_mode, gamma = "gamma", (0.5 if "0.5" in gname else 0.1)
    else:
        guidance_mode, gamma = gname, 0.5

    methods = {
        "M1": dict(step_mode="uniform", guidance_mode="fixed"),
        "M2": dict(step_mode=step_mode, step_p=step_p, guidance_mode="fixed"),
        "M3": dict(step_mode="uniform", guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
        "M4": dict(step_mode=step_mode, step_p=step_p, guidance_mode=guidance_mode, gamma=gamma, alpha=alpha),
    }

    n_samples = 128
    rows = []
    for mname, cfg in methods.items():
        for n_steps in (4, 8):
            for w in (1.5, 5.0):
                print(f"sampling {mname} N={n_steps} w={w}", flush=True)
                field = VelocityField(model, guidance=w, conditioned=True)
                xs, a_all, w_all = [], [], []
                nfe = 0
                for start in range(0, n_samples, 32):
                    b = min(32, n_samples - start)
                    x0 = torch.randn(b, 1, 32, 32)
                    c = torch.randint(0, n_classes, (b,))
                    x, tr, _ = sample_ode(
                        field, x0, c=c, n_steps=n_steps, method="heun", w=w, eta=0.1, **cfg
                    )
                    xs.append(x.clamp(-1, 1))
                    nfe = tr.nfe
                    a_all.extend(tr.a_rms)
                    w_all.extend(tr.w_eff)
                x = torch.cat(xs, dim=0)
                mu_g, sig_g = feature_stats(feat, x, device=device)
                fd = frechet_distance(mu_r, sig_r, mu_g, sig_g)
                row = {
                    "method": mname,
                    "n": n_steps,
                    "w": w,
                    "fd": fd,
                    "nfe_per_sample": nfe,
                    "mean_a": float(np.mean(a_all)),
                    "mean_w_eff": float(np.mean(w_all)),
                    "n_samples": n_samples,
                    "device": "cpu",
                }
                rows.append(row)
                print(row, flush=True)
                if mname in ("M1", "M4") and n_steps == 8 and w == 5.0:
                    save_grid(x[:64], FIG / f"fmnist_{mname}_w5_n8.png")

    (JSON / "stage6_fmnist.json").write_text(json.dumps(rows, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("M1", "M2", "M3", "M4"):
        pts = sorted((r["nfe_per_sample"], r["fd"]) for r in rows if r["method"] == m)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    ax.set_xlabel("NFE")
    ax.set_ylabel("Feature Fréchet distance")
    ax.legend()
    ax.set_title("Fashion-MNIST quality vs cost (CPU transfer)")
    fig.tight_layout()
    fig.savefig(FIG / "fd_vs_nfe.png", dpi=150)
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    main()
