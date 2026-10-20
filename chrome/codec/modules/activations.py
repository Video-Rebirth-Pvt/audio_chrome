"""Periodic activations: Snake in the encoder, SnakeBeta in the decoder."""

from __future__ import annotations

import torch
from torch import nn


def snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    """x + sin²(αx) / α, one α per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha, self.alpha)


class SnakeBeta(nn.Module):
    """x + sin²(αx) / β with separate per-channel frequency α and magnitude β.

    With ``logscale`` the parameters hold log α and log β.
    """

    def __init__(self, channels: int, logscale: bool = True):
        super().__init__()
        self.logscale = logscale
        init = torch.zeros(channels) if logscale else torch.ones(channels)
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        if self.logscale:
            alpha = alpha.exp()
            beta = beta.exp()
        return snake(x, alpha, beta)
