"""Collate packed T2A batches and encode their clips with the frozen VAE, grouped by length."""

from __future__ import annotations

from contextlib import contextmanager

import torch


def collate_t2a(items: list[dict]) -> dict:
    """Clips keep their own lengths (a packed batch mixes 4–15 s); tensors per field, lists for the rest."""
    return {
        "batch_id": items[0]["batch_id"],
        "waves": [item["wave"] for item in items],
        "seconds": torch.tensor([item["seconds"] for item in items], dtype=torch.long),
        "silence_seconds": torch.tensor([item.get("silence_seconds", 0) for item in items], dtype=torch.long),
        "valid_samples": torch.tensor([item["valid_samples"] for item in items], dtype=torch.long),
        "input_ids": [item["input_ids"] for item in items],
        "text_len": torch.tensor([len(item["input_ids"]) for item in items], dtype=torch.long),
        "cfg_dropped": torch.tensor([item["cfg_dropped"] for item in items], dtype=torch.bool),
        "source": torch.tensor([item["source"] for item in items], dtype=torch.long),
        "row": torch.tensor([item["row"] for item in items], dtype=torch.long),
        "captions": [item["caption"] for item in items],
        "paths": [item["path"] for item in items],
    }


@contextmanager
def _tf32(enabled: bool):
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


@torch.no_grad()
def encode_waves(vae, waves: list[torch.Tensor], device=None, allow_tf32: bool = False) -> list[torch.Tensor]:
    """Whitened latents [2, 32, T_i] in input order.

    Only equal-length clips share a forward pass: zero-padding a clip to a longer one changes its last ~10
    latent frames (the encoder's receptive field reaches into the padding), so clips are never padded here.
    """
    groups: dict[int, list[int]] = {}
    for i, wave in enumerate(waves):
        groups.setdefault(wave.shape[-1], []).append(i)
    latents: list[torch.Tensor | None] = [None] * len(waves)
    with _tf32(allow_tf32):
        for members in groups.values():
            stacked = torch.stack([waves[i] for i in members]).to(device, non_blocking=True)
            encoded = vae.encode_audio(stacked)
            for j, i in enumerate(members):
                latents[i] = encoded[j]
    return latents
