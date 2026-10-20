"""Variable-length self-attention over a packed sequence: each ``cu_seqlens`` segment attends only to itself.

``flash``: flash_attn_varlen_func (bf16/fp16 on CUDA). ``sdpa``: a loop of per-segment SDPA calls, exact
on any device. ``auto`` picks flash when it is installed and the input is on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover
    flash_attn_varlen_func = None

BACKENDS = ("auto", "flash", "sdpa")


def resolve_backend(backend: str, device: torch.device) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"attention backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        return "flash" if flash_attn_varlen_func is not None and device.type == "cuda" else "sdpa"
    if backend == "flash" and flash_attn_varlen_func is None:
        raise ImportError("attention=flash but flash_attn is not installed")
    return backend


def varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, backend: str = "auto"
) -> torch.Tensor:
    """q, k, v: [S, heads, head_dim]; cu_seqlens: int32 [n + 1] covering all S rows."""
    backend = resolve_backend(backend, q.device)
    scale = q.shape[-1] ** -0.5
    if backend == "flash":
        dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        out = flash_attn_varlen_func(
            q.to(dtype), k.to(dtype), v.to(dtype), cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale=scale
        )
        return out.to(v.dtype)
    q, k = q.to(v.dtype), k.to(v.dtype)
    out = torch.empty_like(v)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop > start:
            seg = [x[start:stop].transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
            out[start:stop] = F.scaled_dot_product_attention(*seg, scale=scale)[0].transpose(0, 1)
    return out
