"""Rectified flow: x_σ = (1 − σ)·x0 + σ·ε, velocity target v = ε − x0 (= dx/dσ)."""

from __future__ import annotations

import torch

REDUCTIONS = ("token", "sample")


def add_noise(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    return (1.0 - sigma) * x0 + sigma * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0


def flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    row_sample: torch.Tensor,
    num_samples: int,
    reduction: str = "token",
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE over target rows. ``token``: every row weighs the same (long clips count more); ``sample``: mean per
    sample, then over samples. Also returns the detached per-sample loss (NaN for samples without targets)."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    per_row = (pred.float() - target.float()).pow(2).mean(dim=-1)
    weight = mask.to(per_row.dtype)
    sums = torch.zeros(num_samples, device=per_row.device).index_add(0, row_sample, per_row * weight)
    counts = torch.zeros(num_samples, device=per_row.device).index_add(0, row_sample, weight)
    per_sample = sums / counts.clamp(min=1.0)
    if reduction == "token":
        loss = (per_row * weight).sum() / weight.sum().clamp(min=1.0)
    else:
        has = counts > 0
        loss = per_sample[has].mean()
    per_sample = torch.where(counts > 0, per_sample, torch.full_like(per_sample, float("nan")))
    return loss, per_sample.detach()
