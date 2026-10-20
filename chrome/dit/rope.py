"""MM-RoPE over (t, h, w), as H3's ``MiniMaxH3Rope``, and where the two ears sit on the w axis.

Each axis gets ``freqs_per_axis`` frequencies ``theta^(-i/n)``; the three are concatenated and duplicated to
``6·n`` angles that rotate the first ``6·n`` dims of every Q/K head split-half style, the rest pass through.

T2A has no video, yet the ears still need distinct w (otherwise both sit at (t, 0, 0) with the text). H3 pins
them to the first and last column of the video patch grid; ``StereoW`` makes that choice a pluggable policy
because it is an open S5 decision (see skills.md, "左右耳的 w 绑在视频画幅上").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn

# H3 maps the visual grid to RoPE units by sqrt-area normalisation times this factor.
CANVAS_INTERP = 32


class MMRoPE(nn.Module):
    def __init__(self, freqs_per_axis: int = 16, theta: float = 10000.0):
        super().__init__()
        self.freqs_per_axis = int(freqs_per_axis)
        self.theta = float(theta)
        # Same key as H3 (``rope.inv_freq``); recomputed in forward, so it is never trained.
        self.register_buffer("inv_freq", self._inv_freq(), persistent=True)

    @torch.no_grad()
    def reset_buffers(self) -> None:
        self.inv_freq.copy_(self._inv_freq(self.inv_freq.device))

    def _inv_freq(self, device=None) -> torch.Tensor:
        steps = torch.arange(0, self.freqs_per_axis, dtype=torch.float32, device=device)
        return 1.0 / (self.theta ** (steps / self.freqs_per_axis))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions [S, 3] (t, h, w) → angles [S, 6·n] float32."""
        pos = positions.to(torch.float32)
        per_axis = pos.unsqueeze(-1) * self._inv_freq(pos.device).view(1, 1, -1)
        half = torch.cat(per_axis.unbind(dim=1), dim=-1)
        return torch.cat((half, half), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x [S, heads, head_dim], angles [S, R] with R ≤ head_dim."""
    rot = angles.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos = torch.cos(angles).to(x.dtype).unsqueeze(1)
    sin = torch.sin(angles).to(x.dtype).unsqueeze(1)
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)


def canvas_axis(dim: int, patch: int, sqrt_area: float, interp: float = CANVAS_INTERP) -> np.ndarray:
    """H3 ``_axis_from_sqrt_area``: patch-column coordinates of one side of the video latent grid."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * interp


# ---------------------------------------------------------------- stereo w policies

_STEREO: dict[str, type["StereoW"]] = {}


def register_stereo_w(name: str):
    def wrap(cls):
        _STEREO[name] = cls
        return cls

    return wrap


class StereoW(ABC):
    """w coordinate of each audio channel's rows."""

    @abstractmethod
    def values(self, channels: int) -> list[float]: ...


@register_stereo_w("canvas")
class CanvasStereoW(StereoW):
    """H3: left ear at the first patch column of the video canvas, right ear at the last.

    The default canvas is H3's 16:9 1344×768 at 16× spatial compression, i.e. a 48×84 latent:
    w_L ≈ −5.17, w_R ≈ 36.16.
    """

    def __init__(self, latent_h: int = 48, latent_w: int = 84, patch: int = 2, interp: float = CANVAS_INTERP):
        grid = canvas_axis(int(latent_w), int(patch), float(np.sqrt(latent_h * latent_w)), interp)
        self.left, self.right = float(grid[0]), float(grid[-1])

    def values(self, channels: int) -> list[float]:
        if channels == 1:
            return [self.left]
        if channels != 2:
            raise ValueError(f"canvas stereo w is defined for 1 or 2 channels, got {channels}")
        return [self.left, self.right]


@register_stereo_w("fixed")
class FixedStereoW(StereoW):
    """Explicit per-channel w, decoupled from any canvas."""

    def __init__(self, values: list[float]):
        self._values = [float(v) for v in values]

    def values(self, channels: int) -> list[float]:
        if channels > len(self._values):
            raise ValueError(f"{channels} channels but only {len(self._values)} fixed w values")
        return self._values[:channels]


def build_stereo_w(cfg: Mapping) -> StereoW:
    cfg = dict(cfg)
    kind = str(cfg.pop("kind"))
    if kind not in _STEREO:
        raise KeyError(f"unknown stereo w policy {kind!r}; known: {sorted(_STEREO)}")
    return _STEREO[kind](**cfg)
