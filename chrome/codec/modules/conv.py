"""Weight-normalised 1D convolutions stored as ``weight_g`` / ``weight_v``.

This is the key layout of the released H3 audio VAE checkpoint, so it loads with
``strict=True`` without load hooks and exports need no renaming.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _norm_except_dim0(weight: torch.Tensor) -> torch.Tensor:
    return weight.flatten(1).norm(dim=1).view(-1, *([1] * (weight.dim() - 1)))


class _WeightNormConv(nn.Module):
    def __init__(self, weight_shape: tuple[int, int, int], out_channels: int, bias: bool):
        super().__init__()
        self.weight_v = nn.Parameter(torch.empty(weight_shape))
        self.weight_g = nn.Parameter(torch.empty(weight_shape[0], 1, 1))
        self.register_parameter("bias", nn.Parameter(torch.empty(out_channels)) if bias else None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
        with torch.no_grad():
            self.weight_g.copy_(_norm_except_dim0(self.weight_v))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.weight_v.shape[1] * self.weight_v.shape[2])
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        return torch._weight_norm(self.weight_v, self.weight_g, 0)


class WNConv1d(_WeightNormConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__((out_channels, in_channels // groups, kernel_size), out_channels, bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WNConvTranspose1d(_WeightNormConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__((in_channels, out_channels // groups, kernel_size), out_channels, bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )


def init_weights(module: nn.Module) -> None:
    """H3 audio VAE init.

    Weight-normed Conv1d: v ~ TruncNormal(0.02), g = 1. Plain Conv1d: TruncNormal(0.02).
    Biases start at zero. Transposed convolutions keep their default init.
    """
    if isinstance(module, WNConv1d):
        nn.init.trunc_normal_(module.weight_v, std=0.02)
        nn.init.constant_(module.weight_g, 1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=0.02)
    else:
        return
    if module.bias is not None:
        nn.init.zeros_(module.bias)
