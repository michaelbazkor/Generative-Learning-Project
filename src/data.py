"""Synthetic 2D distributions, Fashion-MNIST, and CIFAR-10 loaders."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def sample_swiss_roll(n: int, noise: float = 0.05, seed: Optional[int] = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * math.pi, 4.5 * math.pi, size=n)
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.stack([x, y], axis=1)
    pts = pts + noise * rng.normal(size=pts.shape)
    pts = pts / (np.std(pts, axis=0, keepdims=True) + 1e-8)
    return torch.from_numpy(pts.astype(np.float32))


def sample_pinwheel(
    n: int,
    n_arms: int = 5,
    radial_std: float = 0.3,
    tangential_std: float = 0.05,
    rate: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pinwheel with arm labels in {0,...,n_arms-1}."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_arms, size=n)
    angle = (2 * math.pi / n_arms) * labels.astype(np.float64)
    radius = rng.normal(loc=1.0, scale=radial_std, size=n)
    angle = angle + rate * radius + rng.normal(scale=tangential_std, size=n)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    pts = np.stack([x, y], axis=1).astype(np.float32)
    pts = pts / (np.std(pts, axis=0, keepdims=True) + 1e-8)
    return torch.from_numpy(pts), torch.from_numpy(labels.astype(np.int64))


class SwissRollDataset(Dataset):
    def __init__(self, n: int = 50_000, seed: int = 0):
        self.x = sample_swiss_roll(n, seed=seed)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx]


class PinwheelDataset(Dataset):
    def __init__(self, n: int = 50_000, n_arms: int = 5, seed: int = 0):
        self.x, self.y = sample_pinwheel(n, n_arms=n_arms, seed=seed)
        self.n_classes = n_arms

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_2d_loaders(
    name: str,
    n_train: int = 40_000,
    n_test: int = 4_000,
    batch_size: int = 512,
    seed: int = 0,
    n_arms: int = 5,
):
    if name == "swiss":
        train = SwissRollDataset(n_train, seed=seed)
        test = SwissRollDataset(n_test, seed=seed + 1)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, test.x, None
    if name == "pinwheel":
        train = PinwheelDataset(n_train, n_arms=n_arms, seed=seed)
        test = PinwheelDataset(n_test, n_arms=n_arms, seed=seed + 1)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, test.x, test.y
    raise ValueError(f"Unknown 2D dataset: {name}")


def make_fashion_mnist_loaders(
    batch_size: int = 128,
    root: str = "data",
    num_workers: int = 0,
):
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.Pad(2),  # 28 -> 32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
        ]
    )
    train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, 10


def make_cifar10_loaders(
    batch_size: int = 128,
    root: str = "data",
    num_workers: int = 0,
):
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )
    train = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, 10
