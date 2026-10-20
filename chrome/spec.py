"""Audio contract shared by the codec, the data pipeline and the DiT: seconds ↔ samples ↔ latent frames ↔ rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int = 32000
    channels: int = 2
    hop_length: int = 800
    latent_channels: int = 32
    min_seconds: float = 4.0
    max_seconds: float = 15.0

    @classmethod
    def from_config(cls, cfg: Mapping) -> "AudioSpec":
        """Build from the ``audio`` section of a loaded config."""
        audio = cfg["audio"]
        return cls(**{key: audio[key] for key in cls.__dataclass_fields__ if key in audio})

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def frames_for_seconds(self, seconds: float) -> int:
        return round(seconds * self.latent_rate)

    def frames_for_samples(self, samples: int) -> int:
        return math.ceil(samples / self.hop_length)

    def samples_for_frames(self, frames: int) -> int:
        return frames * self.hop_length

    def rows_for_frames(self, frames: int) -> int:
        """Packed DiT rows: every ear contributes one row per latent frame."""
        return frames * self.channels

    def check_codec(self, vae_config) -> None:
        """Raise if a VAE config does not match this contract."""
        mismatches = []
        if vae_config.sample_rate != self.sample_rate:
            mismatches.append(f"sample_rate {vae_config.sample_rate} != {self.sample_rate}")
        if vae_config.hop_length != self.hop_length:
            mismatches.append(f"hop_length {vae_config.hop_length} != {self.hop_length}")
        if vae_config.latent_channels != self.latent_channels:
            mismatches.append(f"latent_channels {vae_config.latent_channels} != {self.latent_channels}")
        if mismatches:
            raise ValueError("VAE does not match the audio spec: " + "; ".join(mismatches))
