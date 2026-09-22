"""Evaluation metrics: W2, off-support rate, feature Fréchet distance."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a ** 2).sum(dim=1, keepdim=True)
    b2 = (b ** 2).sum(dim=1).unsqueeze(0)
    return a2 + b2 - 2.0 * a @ b.T


def wasserstein2_sinkhorn(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 50,
) -> float:
    """Entropic W2 approximation via Sinkhorn (squared Euclidean cost)."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    n, m = x.shape[0], y.shape[0]
    C = pairwise_sq(x, y)
    K = torch.exp(-C / eps)
    u = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    v = torch.full((m,), 1.0 / m, device=x.device, dtype=x.dtype)
    for _ in range(n_iter):
        u = (1.0 / n) / (K @ v + 1e-12)
        v = (1.0 / m) / (K.T @ u + 1e-12)
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return float((P * C).sum().item())


def wasserstein2_exact_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Exact W2 for 1D projections is cheap; for 2D use scipy OT if available."""
    try:
        import ot

        a = np.ones(len(x)) / len(x)
        b = np.ones(len(y)) / len(y)
        M = ot.dist(x, y, metric="sqeuclidean")
        return float(ot.emd2(a, b, M))
    except Exception:
        # fallback Sinkhorn on torch
        return wasserstein2_sinkhorn(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            eps=0.1,
            n_iter=80,
        )


def off_support_rate(
    samples: np.ndarray,
    ref: np.ndarray,
    k: int = 5,
    threshold_quantile: float = 0.99,
) -> float:
    """Fraction of samples whose kNN distance to ref exceeds a high quantile of ref self-dist."""
    from scipy.spatial import cKDTree

    tree = cKDTree(ref)
    # self distances on ref (leave-one-out approx via k+1)
    d_ref, _ = tree.query(ref, k=k + 1)
    thresh = np.quantile(d_ref[:, -1], threshold_quantile)
    d_s, _ = tree.query(samples, k=k)
    return float((d_s[:, -1] > thresh).mean())


class FMNISTFeatureExtractor(nn.Module):
    """Small CNN classifier; penultimate layer used for Fréchet distance."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.fc(h)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    from scipy.linalg import sqrtm

    # Stabilize singular covariances (common with few samples / low-rank features).
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


@torch.no_grad()
def feature_stats(
    model: FMNISTFeatureExtractor,
    images: torch.Tensor,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = images[i : i + batch_size].to(device)
        feats.append(model.embed(x).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def train_feature_extractor(
    train_loader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
) -> FMNISTFeatureExtractor:
    model = FMNISTFeatureExtractor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    return model
