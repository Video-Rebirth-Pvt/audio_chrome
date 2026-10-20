"""H3-layout audio VAE: DAC encoder → causal attention bottleneck → diagonal Gaussian → BigVGAN decoder.

The network is mono. Stereo is folded into the batch, so left and right share every weight
and are never mixed inside the codec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn

from chrome.codec.modules.attn_proj import AttnProjection
from chrome.codec.modules.bigvgan import BigVGAN
from chrome.codec.modules.conv import init_weights
from chrome.codec.modules.encoder import Encoder
from chrome.codec.modules.gaussian import DiagonalGaussian
from chrome.codec.whitening import LatentStats

_TUPLE_FIELDS = ("encoder_rates", "decoder_rates", "decoder_kernel_sizes", "resblock_kernel_sizes")


@dataclass(frozen=True)
class AudioVAEConfig:
    sample_rate: int = 32000
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int | None = None
    latent_channels: int = 32
    attn_proj: bool = True
    attn_heads: int = 8
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    snake_logscale: bool = True
    use_tanh_at_final: bool = False
    use_bias_at_final: bool = False

    def __post_init__(self):
        for name in _TUPLE_FIELDS:
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        dilations = tuple(tuple(int(v) for v in group) for group in self.resblock_dilation_sizes)
        object.__setattr__(self, "resblock_dilation_sizes", dilations)
        if self.latent_dim is None:
            object.__setattr__(self, "latent_dim", self.encoder_dim * 2 ** len(self.encoder_rates))
        if math.prod(self.encoder_rates) != math.prod(self.decoder_rates):
            raise ValueError(
                f"encoder hop {math.prod(self.encoder_rates)} != decoder hop {math.prod(self.decoder_rates)}"
            )
        if len(self.decoder_rates) != len(self.decoder_kernel_sizes):
            raise ValueError("decoder_rates and decoder_kernel_sizes must have the same length")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("resblock_kernel_sizes and resblock_dilation_sizes must have the same length")
        if self.decoder_dim % 2 ** len(self.decoder_rates):
            raise ValueError(f"decoder_dim {self.decoder_dim} cannot halve {len(self.decoder_rates)} times")

    @classmethod
    def from_dict(cls, data: Mapping) -> "AudioVAEConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown audio VAE config keys: {sorted(unknown)}")
        return cls(**dict(data))

    def to_dict(self) -> dict:
        data = asdict(self)
        for name in _TUPLE_FIELDS:
            data[name] = list(data[name])
        data["resblock_dilation_sizes"] = [list(group) for group in self.resblock_dilation_sizes]
        return data

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self.hop_length


class ChromeAudioVAE(nn.Module):
    """State-dict keys match the released H3 audio VAE, so its checkpoint loads with ``strict=True``.

    Shapes: waveform [..., C, N] ⇄ whitened latent [..., C, latent_channels, N / hop].
    """

    def __init__(self, config: AudioVAEConfig, latent_stats: LatentStats | None = None):
        super().__init__()
        self.config = config
        self.latent_stats = latent_stats or LatentStats.identity(config.latent_channels)
        if self.latent_stats.channels != config.latent_channels:
            raise ValueError(
                f"latent stats have {self.latent_stats.channels} channels, VAE has {config.latent_channels}"
            )

        self.encoder = Encoder(config.encoder_dim, config.encoder_rates, config.latent_dim)
        if config.attn_proj:
            if config.latent_dim % config.latent_channels == 0:
                bottleneck = config.latent_channels
            else:
                bottleneck = 2 ** math.ceil(math.log2(config.latent_channels))
            self.pre_block = AttnProjection(config.latent_dim, bottleneck, num_heads=config.attn_heads)
        else:
            bottleneck = config.latent_dim
            self.pre_block = None
        self.mean_proj = nn.Conv1d(bottleneck, config.latent_channels, 1)
        self.logs_proj = nn.Conv1d(bottleneck, config.latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
        self.decoder = BigVGAN(
            in_channels=config.latent_dim,
            upsample_rates=config.decoder_rates,
            upsample_kernel_sizes=config.decoder_kernel_sizes,
            upsample_initial_channel=config.decoder_dim,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            snake_logscale=config.snake_logscale,
            use_tanh_at_final=config.use_tanh_at_final,
            use_bias_at_final=config.use_bias_at_final,
        )
        self.apply(init_weights)

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def dtype(self) -> torch.dtype:
        return self.dec_in_proj.weight.dtype

    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """Right-pad the time axis with zeros to a multiple of the hop."""
        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio, (0, right_pad))

    def _bottleneck(self, audio: torch.Tensor) -> torch.Tensor:
        z = self.encoder(audio)
        if self.pre_block is not None:
            z = self.pre_block(z.transpose(1, 2)).transpose(1, 2)
        return z

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Posterior mean [B, latent_channels, T] of hop-aligned mono audio [B, 1, N]."""
        return self.mean_proj(self._bottleneck(audio))

    def encode_posterior(self, audio: torch.Tensor) -> DiagonalGaussian:
        z = self._bottleneck(audio)
        return DiagonalGaussian(self.mean_proj(z), self.logs_proj(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Raw (unwhitened) latent [B, latent_channels, T] → mono audio [B, 1, T · hop]."""
        return self.decoder(self.dec_in_proj(z))

    def forward(
        self, audio: torch.Tensor, sample: bool = True, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode_posterior(audio)
        z = posterior.sample(generator) if sample else posterior.mean
        return self.decode(z), posterior

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform [..., C, N] → whitened posterior mean [..., C, latent_channels, ceil(N / hop)], float32."""
        lead = waveform.shape[:-1]
        audio = waveform.reshape(-1, 1, waveform.shape[-1]).to(self.dtype)
        mu = self.encode(self.preprocess(audio))
        return self.latent_stats.whiten(mu).reshape(*lead, *mu.shape[-2:])

    @torch.no_grad()
    def decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Whitened latent [..., C, latent_channels, T] → waveform [..., C, T · hop], float32."""
        lead = latents.shape[:-2]
        z = self.latent_stats.unwhiten(latents.reshape(-1, *latents.shape[-2:])).to(self.dtype)
        waveform = self.decode(z)
        return waveform.float().reshape(*lead, waveform.shape[-1])
