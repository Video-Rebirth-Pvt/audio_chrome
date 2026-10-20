"""Per-channel latent statistics. DiT sees (μ − mean) / std; the decoder gets the inverse."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentStats:
    """Corpus mean and std of the posterior mean, one pair per latent channel.

    Frozen once the VAE is trained. Left and right ears share the same pair.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "mean", tuple(float(v) for v in self.mean))
        object.__setattr__(self, "std", tuple(float(v) for v in self.std))
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean has {len(self.mean)} channels, std has {len(self.std)}")
        if any(v <= 0.0 for v in self.std):
            raise ValueError("latent std must be positive")

    @classmethod
    def identity(cls, channels: int) -> "LatentStats":
        return cls((0.0,) * channels, (1.0,) * channels)

    @property
    def channels(self) -> int:
        return len(self.mean)

    def tensors(self, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """float32 [C, 1] each, broadcasting over [..., C, T]."""
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device).view(-1, 1)
        std = torch.tensor(self.std, dtype=torch.float32, device=device).view(-1, 1)
        return mean, std

    def whiten(self, mu: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(mu.device)
        return (mu.float() - mean) / std

    def unwhiten(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(z.device)
        return z.float() * std + mean
