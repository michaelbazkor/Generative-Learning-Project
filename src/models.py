"""Velocity networks: 2D MLP and lightweight image U-Net."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class VelocityMLP(nn.Module):
    """3-layer MLP with 128 hidden units, SiLU, sinusoidal time, optional class emb."""

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        n_layers: int = 3,
        time_dim: int = 64,
        n_classes: Optional[int] = None,
        class_dim: int = 64,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.class_emb = None
        in_dim = dim + hidden
        if n_classes is not None:
            self.class_emb = nn.Embedding(n_classes + 1, class_dim)  # +1 for null
            self.null_index = n_classes
            in_dim += class_dim

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.time_mlp(t)
        feats = [x, h]
        if self.class_emb is not None:
            if c is None:
                c = torch.full(
                    (x.shape[0],), self.null_index, device=x.device, dtype=torch.long
                )
            feats.append(self.class_emb(c))
        return self.net(torch.cat(feats, dim=-1))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class VelocityUNet(nn.Module):
    """Lightweight conditional U-Net for 32x32 grayscale (<~2M params)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mult=(1, 2, 4),
        n_classes: int = 10,
        time_dim: int = 128,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.null_index = n_classes
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(n_classes + 1, time_dim)

        chs = [base_channels * m for m in channel_mult]
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_dim))
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        self.mid = ResBlock(chs[-1], chs[-1], time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, out_ch in enumerate(reversed(chs)):
            self.up_blocks.append(ResBlock(in_ch + out_ch, out_ch, time_dim))
            if i < len(chs) - 1:
                self.upsamples.append(
                    nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                )
            in_ch = out_ch

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if c is None:
            c = torch.full((x.shape[0],), self.null_index, device=x.device, dtype=torch.long)
        emb = self.time_mlp(t) + self.class_emb(c)

        h = self.conv_in(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid(h, emb)

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.conv_out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
