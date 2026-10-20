"""ODE samplers along the shifted σ table, from noise (σ = 1) to data (σ = 0).

A sampler only sees a velocity function over a list of latents, so CFG, packing and modality layout stay in
the pipeline: ``velocity_fn(xs, sigma) -> vs`` with v = ε − x0. Samplers are registered by name; S4 adds
few-step ones next to Euler.

    sampler = build_sampler({"kind": "euler", "steps": 50, "shift": 3.0})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch

from chrome.flow.schedule import inference_sigmas

VelocityFn = Callable[[list[torch.Tensor], float], list[torch.Tensor]]

_SAMPLERS: dict[str, type["Sampler"]] = {}


def register_sampler(name: str):
    def wrap(cls):
        _SAMPLERS[name] = cls
        return cls

    return wrap


class Sampler(ABC):
    def __init__(self, steps: int = 50, shift: float = 3.0):
        self.steps, self.shift = int(steps), float(shift)

    def sigmas(self) -> torch.Tensor:
        return inference_sigmas(self.steps, self.shift)

    @abstractmethod
    def sample(self, velocity_fn: VelocityFn, noise: list[torch.Tensor], callback=None) -> list[torch.Tensor]: ...


@register_sampler("euler")
class EulerSampler(Sampler):
    """η = 0."""

    @torch.no_grad()
    def sample(self, velocity_fn, noise, callback=None):
        sigmas = self.sigmas().tolist()
        x = [n.float() for n in noise]
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            v = velocity_fn(x, sigma)
            x = [xi + (sigma_next - sigma) * vi.float() for xi, vi in zip(x, v)]
            if callback is not None:
                callback(i, sigma, x)
        return x


def build_sampler(cfg: Mapping) -> Sampler:
    cfg = dict(cfg)
    kind = str(cfg.pop("kind", "euler"))
    if kind not in _SAMPLERS:
        raise KeyError(f"unknown sampler {kind!r}; known: {sorted(_SAMPLERS)}")
    return _SAMPLERS[kind](**cfg)
