"""Training utilities for 2D MLP and Fashion-MNIST U-Net."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.cfm import ot_cfm_loss


def train_model(
    model: torch.nn.Module,
    loader,
    steps: int,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    coupling: str = "independent",
    null_prob: float = 0.0,
    null_index: Optional[int] = None,
    conditional: bool = False,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    log_every: int = 200,
) -> dict:
    model.to(device).train()
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    it = iter(loader)
    pbar = tqdm(range(steps), desc=f"train/{coupling}")
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        if conditional:
            x1, c = batch
            x1, c = x1.to(device), c.to(device)
        else:
            x1 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x1 = x1.to(device)
            c = None

        opt.zero_grad(set_to_none=True)
        loss = ot_cfm_loss(
            model,
            x1,
            c=c,
            coupling=coupling,
            null_prob=null_prob,
            null_index=null_index,
        )
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
            pbar.set_postfix(loss=float(loss.item()))

    return {"history": history, "final_loss": history[-1]["loss"] if history else None}


def save_checkpoint(model: torch.nn.Module, path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
