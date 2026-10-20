"""Noise levels. σ = 1 is pure noise, σ = 0 is data; the DiT's AdaLN sees t = 1 − σ.

Training draws σ through a ``TimestepSampler``; inference walks ``inference_sigmas``. Both apply the same
shift ``σ = s·u / (1 + (s − 1)·u)`` (audio s = 3 in H3), so AdaLN never sees a table it was not trained on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch


def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * u / (1.0 + (shift - 1.0) * u)


def inference_sigmas(steps: int, shift: float, device=None) -> torch.Tensor:
    """[steps + 1] from 1 down to 0."""
    return shift_sigma(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device), shift)


_SAMPLERS: dict[str, type["TimestepSampler"]] = {}


def register_timestep_sampler(name: str):
    def wrap(cls):
        _SAMPLERS[name] = cls
        return cls

    return wrap


class TimestepSampler(ABC):
    @abstractmethod
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """σ for n samples, float32 on CPU."""


@register_timestep_sampler("shifted_uniform")
class ShiftedUniform(TimestepSampler):
    def __init__(self, shift: float = 3.0):
        self.shift = float(shift)

    def sample(self, n, generator=None):
        return shift_sigma(torch.rand(n, generator=generator), self.shift)


@register_timestep_sampler("logit_normal")
class ShiftedLogitNormal(TimestepSampler):
    def __init__(self, shift: float = 1.0, mean: float = 0.0, std: float = 1.0):
        self.shift, self.mean, self.std = float(shift), float(mean), float(std)

    def sample(self, n, generator=None):
        u = torch.sigmoid(self.mean + self.std * torch.randn(n, generator=generator))
        return shift_sigma(u, self.shift)


def build_timestep_sampler(cfg: Mapping) -> TimestepSampler:
    """From the ``flow`` section: ``timestep_sampler`` names the class, ``shift`` plus its own keys."""
    cfg = dict(cfg)
    kind = str(cfg.get("timestep_sampler", "shifted_uniform"))
    if kind not in _SAMPLERS:
        raise KeyError(f"unknown timestep sampler {kind!r}; known: {sorted(_SAMPLERS)}")
    kwargs = dict(cfg.get("sampler_args") or {})
    return _SAMPLERS[kind](shift=float(cfg["shift"]), **kwargs)
