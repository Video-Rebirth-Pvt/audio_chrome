"""DiT hyper-parameters. Presets live in configs/dit/{tiny,1b,10b}.yaml; this dataclass is the schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

ADALN_KINDS = ("per_layer", "shared")
TEXT_NORMS = ("none", "rmsnorm", "stats")


@dataclass(frozen=True)
class DiTConfig:
    num_layers: int = 2
    hidden_size: int = 256
    num_heads: int = 2
    head_dim: int = 128
    ffn_hidden_size: int = 768
    time_embed_dim: int = 128
    # H3 sets the time MLP hidden width to hidden_size; null follows that.
    time_embed_hidden_size: int | None = None
    timestep_input_dim: int = 256
    refiner_layers: int = 2
    text_dim: int = 5120
    audio_dim: int = 32
    # VisualVAE 24 channels × patch (1, 2, 2); unused until S5 but kept so the state dict matches H3.
    video_dim: int = 96
    modalities: int = 3
    rope_freqs_per_axis: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    # per_layer: H3, Linear(time_emb → hidden×6×modalities) in every block.
    # shared: one such Linear for the whole stack plus a learned [modalities, 6, hidden] table per block.
    adaln: str = "per_layer"
    # Normalization of the raw Qwen hidden states before condition_proj (layer-50 states are un-normed;
    # token 0 carries a ~1.6e4 massive activation in dim 731). none: H3. rmsnorm: learned, per token.
    # stats: fixed per-dim (x − mean) / std from the text cache, then clamped to ±text_clip.
    text_norm: str = "none"
    text_clip: float | None = None
    attention: str = "auto"
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.adaln not in ADALN_KINDS:
            raise ValueError(f"adaln must be one of {ADALN_KINDS}, got {self.adaln!r}")
        if self.text_norm not in TEXT_NORMS:
            raise ValueError(f"text_norm must be one of {TEXT_NORMS}, got {self.text_norm!r}")
        if self.rope_dim > self.head_dim:
            raise ValueError(f"MM-RoPE rotates {self.rope_dim} dims but head_dim is {self.head_dim}")

    @property
    def rope_dim(self) -> int:
        return 2 * 3 * self.rope_freqs_per_axis

    @property
    def time_hidden(self) -> int:
        return self.time_embed_hidden_size or self.hidden_size

    @classmethod
    def from_config(cls, cfg: Mapping) -> "DiTConfig":
        """From the ``dit`` section; unknown keys (e.g. ``preset``) are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(cfg).items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)
