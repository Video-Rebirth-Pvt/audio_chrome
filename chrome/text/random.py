"""A fixed random embedding table: no checkpoint, deterministic, CPU friendly. Unit tests and dry runs only."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from chrome.text.base import TextEncoder, register_text_encoder


@register_text_encoder("random")
class RandomTextEncoder(TextEncoder):
    def __init__(
        self, dim: int = 64, vocab_size: int = 151936, seed: int = 0, device: str | torch.device = "cpu", log=print
    ):
        self.dim, self.vocab_size, self.seed = int(dim), int(vocab_size), int(seed)
        self.device = torch.device(device)
        generator = torch.Generator().manual_seed(self.seed)
        self.table = torch.randn(self.vocab_size, self.dim, generator=generator).to(self.device)

    @staticmethod
    def describe(dim: int = 64, vocab_size: int = 151936, seed: int = 0, **_) -> tuple[str, int]:
        return f"random:{int(dim)}:{int(vocab_size)}:{int(seed)}", int(dim)

    @property
    def fingerprint(self) -> str:
        return self.describe(self.dim, self.vocab_size, self.seed)[0]

    def encode(self, ids: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [self.table[i.to(self.device, torch.long)] for i in ids]

    def to(self, device) -> "RandomTextEncoder":
        self.device = torch.device(device)
        self.table = self.table.to(self.device)
        return self
