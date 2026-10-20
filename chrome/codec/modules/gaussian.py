"""Diagonal Gaussian posterior used as the VAE bottleneck."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiagonalGaussian:
    """q(z | x) = N(mean, exp(logs)²), independent per channel and frame. ``logs`` is log σ."""

    mean: torch.Tensor
    logs: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return self.logs.exp()

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """Elementwise KL(q ‖ N(0, I)) in nats."""
        return 0.5 * (self.mean.pow(2) + (2.0 * self.logs).exp() - 1.0 - 2.0 * self.logs)
