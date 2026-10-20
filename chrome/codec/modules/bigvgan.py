"""BigVGAN decoder fed by the continuous latent instead of a mel spectrogram."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from chrome.codec.modules.activations import SnakeBeta
from chrome.codec.modules.alias_free import Activation1d
from chrome.codec.modules.conv import WNConv1d, WNConvTranspose1d, get_padding


class AMPBlock1(nn.Module):
    """Two convolutions per dilation, each preceded by an anti-aliased SnakeBeta."""

    def __init__(self, channels: int, kernel_size: int, dilation: Sequence[int], snake_logscale: bool = True):
        super().__init__()
        self.convs1 = nn.ModuleList(
            WNConv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            WNConv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
            for _ in dilation
        )
        self.activations = nn.ModuleList(
            Activation1d(SnakeBeta(channels, logscale=snake_logscale)) for _ in range(2 * len(dilation))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class BigVGAN(nn.Module):
    """[B, in_channels, T] → waveform [B, 1, T · prod(upsample_rates)] bounded to [-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        upsample_rates: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        snake_logscale: bool = True,
        use_tanh_at_final: bool = False,
        use_bias_at_final: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final

        self.conv_pre = WNConv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            nn.ModuleList(
                [
                    WNConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                ]
            )
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        )
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(channels, k, d, snake_logscale=snake_logscale))
        self.activation_post = Activation1d(SnakeBeta(channels, logscale=snake_logscale))
        self.conv_post = WNConv1d(channels, 1, 7, 1, padding=3, bias=use_bias_at_final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        if self.use_tanh_at_final:
            return torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)
