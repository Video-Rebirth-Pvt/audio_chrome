"""ChromeDiT: single-stream packed DiT over [text | (video) | audio] rows, H3 layout at any scale.

Each modality is projected to ``hidden`` and scattered into one [S, hidden] sequence; text first goes through
``condition_proj`` and a 2-block token refiner. Blocks share attention / FFN across modalities and pick AdaLN
parameters per (timestep, modality). The model predicts one 32-d velocity per audio row (and a 96-d one
per video row once S5 adds video).

Parameter names match DiffSynth's ``MiniMaxH3DiT`` (``per_layer`` AdaLN, ``text_norm: none``), which the
parity test uses to check this implementation against it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from chrome.dit.adaln import AdaLNTable, SharedAdaLN
from chrome.dit.config import DiTConfig
from chrome.dit.layers import DiTBlock, FinalLayer, TimeEmbedder, TokenRefiner
from chrome.dit.packing import PackedSequence
from chrome.dit.rope import MMRoPE


class ChromeDiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = c = config
        hidden = c.hidden_size
        self.video_patch_proj = nn.Linear(c.video_dim, hidden, bias=True)
        self.audio_patch_proj = nn.Linear(c.audio_dim, hidden, bias=True)
        if c.text_norm == "rmsnorm":
            self.text_norm = nn.RMSNorm(c.text_dim, eps=1e-6)
        elif c.text_norm == "stats":
            # Saved with the weights, so inference does not need the stats file.
            self.register_buffer("text_mean", torch.zeros(c.text_dim))
            self.register_buffer("text_std", torch.ones(c.text_dim))
        self.condition_proj = nn.Linear(c.text_dim, hidden, bias=True)
        self.time_embedder = TimeEmbedder(c.timestep_input_dim, c.time_hidden, c.time_embed_dim)
        self.rope = MMRoPE(c.rope_freqs_per_axis, c.rope_theta)
        self.token_refiner = TokenRefiner(
            c.refiner_layers, hidden, c.num_heads, c.head_dim, c.ffn_hidden_size, c.norm_eps, c.attention
        )
        if c.adaln == "shared":
            self.shared_adaln = SharedAdaLN(hidden, c.time_embed_dim, 6, c.modalities)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden, c.num_heads, c.head_dim, c.ffn_hidden_size, c.time_embed_dim,
                    c.modalities, c.norm_eps, c.adaln, c.attention,
                )
                for _ in range(c.num_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden, c.time_embed_dim, c.video_dim, c.audio_dim, c.norm_eps)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Xavier linears, zero biases; AdaLN-Zero (gates start closed) and zero output heads.

        Sets every parameter and buffer, so it also initialises a model built on the meta device and
        materialised with ``to_empty`` (sharded 10B init). Text stats go back to identity; see ``set_text_stats``.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                module.reset_parameters()
            elif isinstance(module, AdaLNTable):
                module.reset_parameters()
        self.rope.reset_buffers()
        if self.config.text_norm == "stats":
            self.text_mean.zero_()
            self.text_std.fill_(1.0)
        for linear in (self.time_embedder.proj_in, self.time_embedder.proj_out):
            nn.init.normal_(linear.weight, std=0.02)
        for block in self.blocks:
            block.adaln_proj.zero_init()
        if hasattr(self, "shared_adaln"):
            self.shared_adaln.zero_init()
        self.final_layer.adaln_proj.zero_init()
        for head in (self.final_layer.audio_out, self.final_layer.video_out):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def set_text_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if self.config.text_norm != "stats":
            raise ValueError(f"text_norm is {self.config.text_norm!r}, not 'stats'")
        self.text_mean.copy_(mean.to(self.text_mean))
        self.text_std.copy_(std.to(self.text_std).clamp(min=1e-6))

    def set_video_trainable(self, trainable: bool) -> None:
        """T2A never feeds video rows; freezing the video heads keeps DDP free of unused parameters."""
        for module in (self.video_patch_proj, self.final_layer.video_out):
            module.requires_grad_(trainable)

    def shard_units(self) -> list[nn.Module]:
        """Modules FSDP shards (and gathers) one at a time; everything else lives in the root unit."""
        return [*self.token_refiner.blocks, *self.blocks]

    def forward(
        self,
        seq: PackedSequence,
        audio_rows: torch.Tensor,
        text_rows: torch.Tensor,
        t_rows: torch.Tensor,
        video_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """audio_rows [Na, 32], text_rows [Nt, text_dim], t_rows [S] (t = 1 − σ) → audio [Na, 32], video."""
        param_dtype = self.condition_proj.weight.dtype
        if self.config.text_norm == "stats":
            text = (text_rows.float() - self.text_mean) / self.text_std
            if self.config.text_clip:
                text = text.clamp(-self.config.text_clip, self.config.text_clip)
            text = text.to(param_dtype)
        else:
            text = text_rows.to(param_dtype)
        if hasattr(self, "text_norm"):
            text = self.text_norm(text)
        text = self.condition_proj(text)
        text = self.token_refiner(text, cu_seqlens=seq.text_cu_seqlens, max_seqlen=seq.text_max_seqlen)
        audio = self.audio_patch_proj(audio_rows.to(param_dtype))

        x = torch.zeros(seq.seq_len, self.config.hidden_size, device=audio.device, dtype=audio.dtype)
        x = x.index_add(0, seq.text_pos, text.to(x.dtype))
        x = x.index_add(0, seq.audio_pos, audio)
        if video_rows is not None and seq.video_pos.numel():
            x = x.index_add(0, seq.video_pos, self.video_patch_proj(video_rows.to(param_dtype)).to(x.dtype))

        unique_t, time_index = torch.unique(t_rows, sorted=True, return_inverse=True)
        t_emb = self.time_embedder(unique_t)
        shared = self.shared_adaln(t_emb) if hasattr(self, "shared_adaln") else None
        index = time_index * self.config.modalities + seq.token_tags.clamp(min=0)
        rope = self.rope(seq.positions)

        for block in self.blocks:
            args = (x, t_emb, shared, index, rope, seq.cu_seqlens, seq.max_seqlen)
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, *args, use_reentrant=False)
            else:
                x = block(*args)
        return self.final_layer(x, t_emb, time_index, seq.audio_pos, seq.video_pos)


def parameter_report(model: ChromeDiT) -> dict[str, int]:
    """Total, AdaLN (all blocks + shared + final), backbone (attention + FFN of the blocks)."""
    total = sum(p.numel() for p in model.parameters())
    adaln = sum(p.numel() for n, p in model.named_parameters() if "adaln" in n)
    backbone = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("blocks.") and (".attn." in n or ".mlp." in n)
    )
    return {"total": total, "adaln": adaln, "backbone": backbone}
