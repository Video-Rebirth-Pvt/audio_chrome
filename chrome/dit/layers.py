"""DiT building blocks. Attribute names follow H3 (DiffSynth ``MiniMaxH3DiT``) so state dicts line up."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from chrome.dit.adaln import PerLayerAdaLN, block_adaln, gated_residual, modulate
from chrome.dit.attention import varlen_attention
from chrome.dit.rope import apply_rope


class TimeEmbedder(nn.Module):
    """t ∈ [0, 1] (t = 1 − σ, 1 = clean) → sinusoid (cos | sin, max period 10⁴) → MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.frequency_embedding_size = input_dim
        self.proj_in = nn.Linear(input_dim, hidden_dim, bias=True)
        self.proj_out = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(t_freq.to(self.proj_in.weight.dtype))))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, eps: float, backend: str = "auto"):
        super().__init__()
        self.num_heads, self.head_dim, self.backend = num_heads, head_dim, backend
        self.qkv_proj = nn.Linear(hidden_size, num_heads * head_dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, *, rope: torch.Tensor | None, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        total = x.shape[0]
        # H3 layout: [S, heads, (q, k, v), head_dim].
        q, k, v = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim).unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = varlen_attention(q, k, v, cu_seqlens, max_seqlen, self.backend)
        return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class TokenRefinerBlock(nn.Module):
    """Pre-norm block over the text rows only: no AdaLN, no RoPE."""

    def __init__(self, hidden_size, num_heads, head_dim, ffn_hidden_size, eps, backend):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.attn = Attention(hidden_size, num_heads, head_dim, eps, backend)
        self.mlp = SwiGLU(hidden_size, ffn_hidden_size)

    def forward(self, x, *, cu_seqlens, max_seqlen):
        x = x + self.attn(self.norm1(x), rope=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, head_dim, ffn_hidden_size, eps, backend):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TokenRefinerBlock(hidden_size, num_heads, head_dim, ffn_hidden_size, eps, backend) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=eps)

    def forward(self, x, *, cu_seqlens, max_seqlen):
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, ffn_hidden_size, time_embed_dim, modalities, eps, adaln, backend):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.attn = Attention(hidden_size, num_heads, head_dim, eps, backend)
        self.mlp = SwiGLU(hidden_size, ffn_hidden_size)
        self.adaln_proj = block_adaln(adaln, hidden_size, time_embed_dim, modalities)

    def forward(self, x, t_emb, shared_adaln, index, rope, cu_seqlens, max_seqlen: int):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb, shared_adaln)
        h = modulate(self.norm1(x), shift_msa, scale_msa, index)
        h = self.attn(h, rope=rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = gated_residual(x, gate_msa, h, index)
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, index))
        return gated_residual(x, gate_mlp, h, index)


class FinalLayer(nn.Module):
    """Time-only shift/scale (no modality, no gate), then one output head per modality."""

    def __init__(self, hidden_size, time_embed_dim, video_dim, audio_dim, eps):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.adaln_proj = PerLayerAdaLN(hidden_size, time_embed_dim, chunks=2, modalities=1)
        self.video_out = nn.Linear(hidden_size, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_dim, bias=True)

    def forward(self, x, t_emb, time_index, audio_pos, video_pos):
        shift, scale = self.adaln_proj(t_emb)
        h = modulate(self.norm(x), shift, scale, time_index)
        audio = self.audio_out(h.index_select(0, audio_pos))
        video = self.video_out(h.index_select(0, video_pos)) if video_pos.numel() else None
        return audio, video
