"""Causal attention projection that narrows the 2048-d encoder features to the latent width."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    """Causal self-attention from in_dim to out_dim over [B, T, C].

    When narrowing, head outputs are averaged and the head width is pooled down to out_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        width = max(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(in_dim, width * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(width))
        self.v_bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("zero_k_bias", torch.zeros(width))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.in_dim > self.out_dim:
            x = x.mean(dim=1)
            if self.head_dim != self.out_dim:
                x = F.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(batch, length, -1)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        if out_dim % in_dim and in_dim % out_dim:
            raise ValueError(f"in_dim {in_dim} and out_dim {out_dim} must divide one another")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
