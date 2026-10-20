"""DAC-style waveform encoder: Snake + dilated residual units + strided convolutions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from chrome.codec.modules.activations import Snake1d
from chrome.codec.modules.conv import WNConv1d


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Waveform [B, 1, N] → features [B, d_latent, N / prod(strides)]. Channels double at every stride."""

    def __init__(self, d_model: int, strides: Sequence[int], d_latent: int):
        super().__init__()
        layers: list[nn.Module] = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride=stride))
        layers += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
