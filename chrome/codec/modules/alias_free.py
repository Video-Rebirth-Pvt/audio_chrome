"""Anti-aliased activation: upsample with a Kaiser-windowed sinc, apply the nonlinearity, low-pass and decimate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Low-pass FIR of shape [1, 1, kernel_size] that sums to one."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time, dtype=window.dtype)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6, stride: int = 1, kernel_size: int = 12):
        super().__init__()
        if not 0.0 <= cutoff <= 0.5:
            raise ValueError(f"cutoff must be in [0, 0.5], got {cutoff}")
        self.kernel_size = kernel_size
        self.stride = stride
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        weight = self.filter.to(x.dtype).expand(channels, -1, -1)
        return F.conv1d(x, weight, stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter_ = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        weight = self.filter.to(x.dtype).expand(channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, weight, stride=self.stride, groups=channels)
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))
