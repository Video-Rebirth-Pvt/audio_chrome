"""Modality-aware AdaLN. Rows pick their parameters by ``unique_t_index × modalities + tag``.

Both forms return ``chunks`` tensors of shape [U·M, hidden] (U distinct timesteps, M modalities), row
``u·M + m`` holding the modulation for timestep u and modality m:

- ``PerLayerAdaLN`` (H3): every block owns SiLU → Linear(time_emb → chunks·hidden·M). About 39 % of the
  parameters at 1B/10B scale.
- ``SharedAdaLN`` + ``AdaLNTable`` (adaLN-single style): one such Linear for the whole stack, each block adds
  a learned [M, chunks·hidden] table. The S2 ablation compares the two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLayerAdaLN(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, chunks: int, modalities: int):
        super().__init__()
        self.hidden_size, self.chunks, self.modalities = hidden_size, chunks, modalities
        self.linear = nn.Linear(time_embed_dim, chunks * hidden_size * modalities, bias=True)

    def forward(self, t_emb: torch.Tensor, shared: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        x = self.linear(F.silu(t_emb))
        return x.view(-1, self.chunks * self.hidden_size).chunk(self.chunks, dim=-1)

    def zero_init(self) -> None:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class SharedAdaLN(nn.Module):
    """The stack-wide projection; its output is passed to every block's ``AdaLNTable``."""

    def __init__(self, hidden_size: int, time_embed_dim: int, chunks: int, modalities: int):
        super().__init__()
        self.hidden_size, self.chunks = hidden_size, chunks
        self.linear = nn.Linear(time_embed_dim, chunks * hidden_size * modalities, bias=True)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.linear(F.silu(t_emb)).view(-1, self.chunks * self.hidden_size)

    def zero_init(self) -> None:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class AdaLNTable(nn.Module):
    def __init__(self, hidden_size: int, chunks: int, modalities: int):
        super().__init__()
        self.hidden_size, self.chunks = hidden_size, chunks
        self.table = nn.Parameter(torch.empty(modalities, chunks * hidden_size))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.table, std=self.hidden_size**-0.5)

    def forward(self, t_emb: torch.Tensor, shared: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        if shared is None:
            raise ValueError("AdaLNTable needs the shared AdaLN output")
        repeats = shared.shape[0] // self.table.shape[0]
        return (shared + self.table.repeat(repeats, 1).to(shared.dtype)).chunk(self.chunks, dim=-1)

    def zero_init(self) -> None:
        # The table keeps its random init (PixArt); the shared Linear starts at zero instead.
        pass


def block_adaln(kind: str, hidden_size: int, time_embed_dim: int, modalities: int, chunks: int = 6) -> nn.Module:
    if kind == "per_layer":
        return PerLayerAdaLN(hidden_size, time_embed_dim, chunks, modalities)
    if kind == "shared":
        return AdaLNTable(hidden_size, chunks, modalities)
    raise ValueError(f"unknown adaln kind {kind!r}")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return (x * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)).to(x.dtype)


def gated_residual(x: torch.Tensor, gate: torch.Tensor, update: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return (x + gate.index_select(0, index) * update).to(x.dtype)
